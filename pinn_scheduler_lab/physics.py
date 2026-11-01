"""Analytically checkable 2-spatial-dimensional heat and incompressible flow."""
from dataclasses import dataclass,asdict
import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector


@dataclass
class Config:
    width:int=32
    depth:int=3
    lr:float=.001
    n_interior:int=256
    n_edge:int=32
    probe_interior:int=192
    probe_edge:int=64
    reynolds:float=20.
    check_every:int=50
    budget_seconds:float=30.
    max_steps:int=30000
    tau_bc:float=.03
    tau_pde:float=.03
    hysteresis:float=.2
    min_dwell_checks:int=2
    grid:int=41
    relobralo_alpha:float=.999
    relobralo_beta:float=.999
    relobralo_temperature:float=.1

    def to_dict(self):return asdict(self)


def fingerprint(tensor):
    if torch.is_tensor(tensor):tensor=tensor.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest()


class MLP(nn.Module):
    def __init__(self,outputs,cfg):
        super().__init__();dims=[2]+[cfg.width]*cfg.depth+[outputs]
        self.layers=nn.ModuleList(nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:]))
        for layer in self.layers:nn.init.xavier_normal_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,points,order=0):
        z=2*points-1
        if order:
            zx=torch.zeros_like(z);zx[:,0]=2
            zy=torch.zeros_like(z);zy[:,1]=2
            zxx=torch.zeros_like(z);zyy=torch.zeros_like(z)
        for i,layer in enumerate(self.layers):
            z=layer(z)
            if order:zx,zy,zxx,zyy=[q@layer.weight.T for q in (zx,zy,zxx,zyy)]
            if i<len(self.layers)-1:
                z=torch.tanh(z)
                if order:
                    a=1-z.square();b=-2*z*a
                    zxx,zyy=a*zxx+b*zx.square(),a*zyy+b*zy.square()
                    zx,zy=a*zx,a*zy
        return (z,zx,zy,zxx,zyy) if order else z


def boundary(n,midpoints=False):
    t=(torch.arange(n)+.5)/n if midpoints else torch.linspace(0,1,n)
    return torch.cat((torch.stack((t,torch.zeros_like(t)),1),torch.stack((t,torch.ones_like(t)),1),
                      torch.stack((torch.zeros_like(t),t),1),torch.stack((torch.ones_like(t),t),1)))


class Problem:
    def __init__(self,kind,cfg,seed):
        self.kind,self.cfg,self.seed=kind,cfg,seed
        self.outputs=1 if kind=='heat2d' else 3
        self.train_points=torch.quasirandom.SobolEngine(2,scramble=True,seed=seed+11).draw(cfg.n_interior).double()
        self.train_boundary=boundary(cfg.n_edge)
        self.probe_points=torch.quasirandom.SobolEngine(2,scramble=True,seed=seed+123).draw(cfg.probe_interior).double()
        self.probe_boundary=boundary(cfg.probe_edge,True)
        self.audit_points=torch.quasirandom.SobolEngine(2,scramble=True,seed=99117).draw(1024).double()
        self.gauge=torch.tensor([[0.,0.]])
        self.sample_hash=fingerprint(torch.cat((self.train_points.flatten(),self.train_boundary.flatten(),self.probe_points.flatten(),self.probe_boundary.flatten())))
        axis=torch.linspace(0,1,cfg.grid);yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        self.grid=torch.stack((xx.flatten(),yy.flatten()),1);self.axis=axis
        self.truth=self.exact(self.grid)

    def make_model(self):return MLP(self.outputs,self.cfg)

    def exact(self,points):
        x,y=points[:,:1],points[:,1:2]
        if self.kind=='heat2d':return torch.sin(math.pi*x)*torch.sin(math.pi*y)+.2*torch.sin(3*math.pi*x)*torch.sin(2*math.pi*y)
        lam=self.cfg.reynolds/2-math.sqrt(self.cfg.reynolds**2/4+4*math.pi**2)
        e=torch.exp(lam*x)
        return torch.cat((1-e*torch.cos(2*math.pi*y),lam/(2*math.pi)*e*torch.sin(2*math.pi*y),.5*(1-e.square())),1)

    def residuals(self,model,points):
        value,dx,dy,dxx,dyy=model(points,2)
        if self.kind=='heat2d':
            x,y=points[:,:1],points[:,1:2]
            source=2*math.pi**2*torch.sin(math.pi*x)*torch.sin(math.pi*y)+2.6*math.pi**2*torch.sin(3*math.pi*x)*torch.sin(2*math.pi*y)
            return (-dxx-dyy-source)/(2*math.pi**2)
        u,v=value[:,:1],value[:,1:2];nu=1/self.cfg.reynolds
        return torch.cat((u*dx[:,:1]+v*dy[:,:1]+dx[:,2:3]-nu*(dxx[:,:1]+dyy[:,:1]),
                          u*dx[:,1:2]+v*dy[:,1:2]+dy[:,2:3]-nu*(dxx[:,1:2]+dyy[:,1:2]),
                          dx[:,:1]+dy[:,1:2]),1)

    def pde_loss(self,model,probe=False):
        residual=self.residuals(model,self.probe_points if probe else self.train_points)
        return residual.square().mean(0).sum()

    def bc_components(self,model,probe=False):
        points=self.probe_boundary if probe else self.train_boundary
        predicted=model(points);target=self.exact(points)
        if self.kind=='heat2d':return (predicted-target).square().mean(0)
        # Velocity scale U=1, pressure scale rho U^2=1. A single p(0,0)=0 gauge fixes pressure offset.
        velocity=(predicted[:,:2]-target[:,:2]).square().mean(0)
        return torch.cat((velocity,model(self.gauge)[:,2].square().reshape(1)))

    def bc_loss(self,model,probe=False):return self.bc_components(model,probe).sum()

    def probe(self,model,previous=None,step_time=0.):
        lf,lb=self.pde_loss(model,True),self.bc_loss(model,True)
        params=list(model.parameters())
        # Pure derivative residuals do not depend on a constant output bias.
        def flatten(grads):return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,grads)])
        a=flatten(torch.autograd.grad(lf,params,retain_graph=True,allow_unused=True))
        b=flatten(torch.autograd.grad(lb,params,allow_unused=True))
        cosine=((a@b)/(a.norm()*b.norm()).clamp_min(1e-30)).item()
        with torch.no_grad():e_bc=self.bc_components(model,True).sqrt().max().item()
        state={'pde_loss':lf.item(),'bc_loss':lb.item(),'e_pde':math.sqrt(lf.item()),'e_bc':e_bc,
               'gradient_cosine':cosine,'recent_step_ms':step_time*1000,
               'relative_pde_drop':0. if previous is None else 1-lf.item()/max(previous['pde_loss'],1e-30),
               'relative_bc_drop':0. if previous is None else 1-lb.item()/max(previous['bc_loss'],1e-30)}
        return state

    @torch.no_grad()
    def evaluate(self,model):
        predicted=model(self.grid);error=predicted-self.truth
        if self.kind=='heat2d':
            metrics={'relative_l2':(error.norm()/self.truth.norm()).item()}
            fields={'temperature':predicted[:,0].reshape(self.cfg.grid,self.cfg.grid).tolist()}
        else:
            metrics={'relative_l2':(error[:,:2].norm()/self.truth[:,:2].norm()).item(),
                     'u_relative_l2':(error[:,0].norm()/self.truth[:,0].norm()).item(),
                     'v_relative_l2':(error[:,1].norm()/self.truth[:,1].norm()).item(),
                     'pressure_relative_l2':(error[:,2].norm()/self.truth[:,2].norm()).item()}
            fields={name:predicted[:,i].reshape(self.cfg.grid,self.cfg.grid).tolist() for i,name in enumerate(['u','v','pressure'])}
        metrics['independent_pde_rms']=self.residuals(model,self.audit_points).square().mean(0).sum().sqrt().item()
        return metrics,fields

    def metadata(self):
        names=['temperature'] if self.kind=='heat2d' else ['u','v','pressure']
        return {'id':self.kind,'title':'二维稳态热传导 / Poisson' if self.kind=='heat2d' else f'二维 Navier–Stokes / Kovasznay Re={self.cfg.reynolds:g}',
                'field_names':names,'axis':self.axis.tolist(),
                'truth':{name:self.truth[:,i].reshape(self.cfg.grid,self.cfg.grid).tolist() for i,name in enumerate(names)}}


def parameters_hash(model):return fingerprint(parameters_to_vector(model.parameters()).detach())
