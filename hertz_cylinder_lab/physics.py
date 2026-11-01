"""KKT-based mixed-variable PINN for an elastic cylinder on a rigid flat."""
from dataclasses import dataclass, asdict
import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.nn.utils import parameters_to_vector


@dataclass
class Config:
    seed: int = 42
    radius_mm: float = 10.
    young_mpa: float = 200.
    poisson: float = .3
    top_pressure_mpa: float = .5
    thickness_mm: float = 10.
    width: int = 24
    depth: int = 3
    n_interior: int = 96
    n_arc: int = 96
    steps: int = 1600
    record_every: int = 20
    grid: int = 35
    threads: int = 1
    equilibrium_weight: float = 1.
    constitutive_weight: float = 10.
    kkt_weight: float = 10.
    boundary_constitutive: bool = True
    formulation: str = 'displacement'
    weak_weight: float = 10.
    time_budget_seconds: float = 120.
    initial_vertical_bias: float = 0.

    @property
    def load_ratio(self):
        return self.top_pressure_mpa/self.young_mpa

    @property
    def local_scale(self):
        return math.sqrt(self.load_ratio)

    @property
    def displacement_um(self):
        return self.load_ratio*self.radius_mm*1000

    @property
    def stress_mpa(self):
        return self.young_mpa*self.local_scale

    @property
    def hertz_half_width_over_r(self):
        return math.sqrt(8*self.load_ratio*(1-self.poisson**2)/math.pi)

    @property
    def applied_force_n(self):
        return 2*self.top_pressure_mpa*self.radius_mm*self.thickness_mm

    def to_dict(self):
        return asdict(self)


def flat_parameters(model):
    return parameters_to_vector(model.parameters()).detach().clone()


def assign(model, flat):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(flat[offset:offset+p.numel()].view_as(p))
            offset += p.numel()


def fingerprint(tensor):
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest()


def loss_gradients(losses, params):
    return torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(value, params, retain_graph=True)]) for value in losses])


def fischer_burmeister(gap, pressure):
    return torch.sqrt(gap.square()+pressure.square()+1e-16)-gap-pressure


class MixedCylinderModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.scale = cfg.local_scale
        dims = [4]+[cfg.width]*cfg.depth+[5]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)

    def raw(self, points, derivatives=False):
        ell = self.scale
        x, y = points[:, :1], points[:, 1:2]
        z = torch.cat((x, y, x/(x+ell), y/(y+ell)), 1)*2-1
        if derivatives:
            zero, one = torch.zeros_like(x), torch.ones_like(x)
            zx = torch.cat((one, zero, ell/(x+ell).square(), zero), 1)*2
            zy = torch.cat((zero, one, zero, ell/(y+ell).square()), 1)*2
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if derivatives:
                zx, zy = zx@layer.weight.T, zy@layer.weight.T
            if i < len(self.layers)-1:
                z = torch.tanh(z)
                if derivatives:
                    zx, zy = (1-z.square())*zx, (1-z.square())*zy
        return (z, zx, zy) if derivatives else z

    def forward(self, points, order=0):
        result = self.raw(points, bool(order))
        raw = result[0] if order else result
        x, y = points[:, :1], points[:, 1:2]
        # Symmetry ux=sxy=0 at x=0; applied syy=-q, sxy=0 at y=1.
        value = torch.cat((x*raw[:, :1], raw[:, 1:2], raw[:, 2:3],
                           -self.scale+(1-y)*raw[:, 3:4], x*(1-y)*raw[:, 4:5]), 1)
        if not order:
            return value
        _, rx, ry = result
        dx = torch.cat((raw[:, :1]+x*rx[:, :1], rx[:, 1:3], (1-y)*rx[:, 3:4],
                        (1-y)*(raw[:, 4:5]+x*rx[:, 4:5])), 1)
        dy = torch.cat((x*ry[:, :1], ry[:, 1:3], -raw[:, 3:4]+(1-y)*ry[:, 3:4],
                        x*(-raw[:, 4:5]+(1-y)*ry[:, 4:5])), 1)
        return value, [dx, dy]


class CylinderModel(nn.Module):
    """Displacement PINN: stresses satisfy the plane-strain constitutive law exactly.

    Full analytic second-order jets provide stress derivatives for equilibrium.
    Even x features and ux=x*raw_ux enforce symmetry displacement AND shear.
    """
    def __init__(self, cfg):
        super().__init__()
        self.scale, self.nu = cfg.local_scale, cfg.poisson
        dims = [4]+[cfg.width]*cfg.depth+[2]
        self.layers = nn.ModuleList(nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:]))
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.layers[-1].bias[1] = cfg.initial_vertical_bias

    def jets(self, points):
        x,y=points[:,:1],points[:,1:2]; ell=self.scale; e2=ell**2
        den=x.square()+e2; zero=torch.zeros_like(x); one=torch.ones_like(x)
        z=2*torch.cat((x.square(),y,x.square()/den,y/(y+ell)),1)-1
        zx=2*torch.cat((2*x,zero,2*x*e2/den.square(),zero),1)
        zy=2*torch.cat((zero,one,zero,ell/(y+ell).square()),1)
        zxx=2*torch.cat((2*one,zero,2*e2*(e2-3*x.square())/den.pow(3),zero),1)
        zyy=2*torch.cat((zero,zero,zero,-2*ell/(y+ell).pow(3)),1)
        zxy=torch.zeros_like(z)
        for i,layer in enumerate(self.layers):
            z=layer(z)
            zx,zy,zxx,zxy,zyy=[d@layer.weight.T for d in (zx,zy,zxx,zxy,zyy)]
            if i<len(self.layers)-1:
                z=torch.tanh(z); a=1-z.square(); b=-2*z*a
                zxx,zxy,zyy=a*zxx+b*zx.square(),a*zxy+b*zx*zy,a*zyy+b*zy.square()
                zx,zy=a*zx,a*zy
        u=torch.cat((x*z[:,:1],z[:,1:]),1)
        dx=torch.cat((z[:,:1]+x*zx[:,:1],zx[:,1:]),1)
        dy=torch.cat((x*zy[:,:1],zy[:,1:]),1)
        dxx=torch.cat((2*zx[:,:1]+x*zxx[:,:1],zxx[:,1:]),1)
        dxy=torch.cat((zy[:,:1]+x*zxy[:,:1],zxy[:,1:]),1)
        dyy=torch.cat((x*zyy[:,:1],zyy[:,1:]),1)
        return u,dx,dy,dxx,dxy,dyy

    def forward(self, points, order=0):
        u,dx,dy,dxx,dxy,dyy=self.jets(points)
        nu,ell=self.nu,self.scale
        lam,mu=nu/((1+nu)*(1-2*nu)),1/(2*(1+nu))
        sxx=ell*((lam+2*mu)*dx[:,:1]+lam*dy[:,1:])
        syy=ell*(lam*dx[:,:1]+(lam+2*mu)*dy[:,1:])
        sxy=ell*mu*(dy[:,:1]+dx[:,1:])
        value=torch.cat((u,sxx,syy,sxy),1)
        if not order:return value
        sx=torch.cat((ell*((lam+2*mu)*dxx[:,:1]+lam*dxy[:,1:]),
                      ell*(lam*dxx[:,:1]+(lam+2*mu)*dxy[:,1:]),
                      ell*mu*(dxy[:,:1]+dxx[:,1:])),1)
        sy=torch.cat((ell*((lam+2*mu)*dxy[:,:1]+lam*dyy[:,1:]),
                      ell*(lam*dxy[:,:1]+(lam+2*mu)*dyy[:,1:]),
                      ell*mu*(dyy[:,:1]+dxy[:,1:])),1)
        return value,[torch.cat((dx,sx),1),torch.cat((dy,sy),1)]


def arc_points(theta):
    return torch.stack((torch.sin(theta), 1-torch.cos(theta)), 1)


class Problem:
    id = 'hertz_kkt'
    loss_names = ['equilibrium', 'constitutive', 'frictionless', 'kkt_fb', 'load_balance']
    weights = [1., 1., 1., 1., 1.]

    def __init__(self, cfg, reference):
        self.cfg, self.reference = cfg, reference
        self.weights = [cfg.equilibrium_weight, cfg.constitutive_weight, 1., cfg.kkt_weight, 1.]
        if cfg.formulation == 'displacement':
            self.loss_names = ['equilibrium', 'top_traction', 'frictionless', 'kkt_fb', 'load_balance']
        self.loss_names = self.loss_names+['virtual_work']
        self.weights = self.weights+[cfg.weak_weight]
        engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=cfg.seed+11)
        uv = engine.draw(cfg.n_interior).double()
        n = cfg.n_interior//2
        radius, theta = uv[:n, 0].sqrt(), uv[:n, 1]*(math.pi/2)
        global_points = torch.stack((radius*theta.sin(), 1-radius*theta.cos()), 1)
        local_x = uv[n:, :1].square()*min(.45, 6*cfg.local_scale)
        local_bottom = 1-torch.sqrt(1-local_x.square())
        local_y = local_bottom+uv[n:, 1:2].square()*min(.45, 6*cfg.local_scale)
        self.interior = torch.cat((global_points, torch.cat((local_x, local_y), 1)))
        half = cfg.n_arc//2
        # Potential contact arc, concentrated near the tip; no Hertz support mask.
        self.theta = torch.cat((torch.linspace(0, min(.45, 6*cfg.local_scale), half),
                                torch.linspace(min(.45, 6*cfg.local_scale)+.001, math.pi/2, cfg.n_arc-half)))
        self.arc = arc_points(self.theta)
        top_x = torch.linspace(0,1,32)
        self.top = torch.stack((top_x,torch.ones_like(top_x)),1)
        nodes, weights = np.polynomial.legendre.leggauss(32)
        split = min(.45, 6*cfg.local_scale)
        theta_q = np.concatenate(((nodes+1)/2*split, split+(nodes+1)/2*(math.pi/2-split)))
        weight_q = np.concatenate((weights/2*split, weights/2*(math.pi/2-split)))
        self.theta_q, self.weight_q = torch.tensor(theta_q), torch.tensor(weight_q)
        self.arc_q = arc_points(self.theta_q)
        # Global virtual-work checks prevent local collocation from hiding a
        # nearly stress-free bulk behind thin, unsampled boundary layers.
        gn, gw = np.polynomial.legendre.leggauss(20)
        rr, aa = np.meshgrid((gn+1)/2, (gn+1)*math.pi/4, indexing='ij')
        wr, wa = np.meshgrid(gw/2, gw*math.pi/4, indexing='ij')
        self.quad_points = torch.tensor(np.stack((rr*np.sin(aa),1-rr*np.cos(aa)),2).reshape(-1,2))
        self.quad_weights = torch.tensor((wr*wa*rr).flatten())
        self.virtual_tests = [('u',1,0),('u',1,1),('u',1,2),('u',3,0),('u',3,1),
                              ('v',0,1),('v',0,2),('v',0,3),('v',2,0),('v',2,1),('v',2,2),('v',4,0)]
        self.sample_hash = fingerprint(torch.cat((self.interior.flatten(), self.theta, self.theta_q, self.weight_q)))
        self.grid_points = torch.tensor(reference['valid_points'])
        self.truth_u = torch.tensor(reference['displacement_over_u0'])
        self.pressure_x = torch.tensor(reference['pressure_x'])
        self.pressure_points = torch.stack((self.pressure_x, 1-torch.sqrt((1-self.pressure_x.square()).clamp_min(0))), 1)
        self.truth_pressure_over_q = torch.tensor(reference['pressure_over_q'])
        self.hertz_pressure_over_q = torch.tensor(reference['hertz_pressure_over_q'])
        self.mask = np.array(reference['grid_mask'], dtype=bool)
        self.n_eval = 0

    def make_model(self):
        return CylinderModel(self.cfg) if self.cfg.formulation == 'displacement' else MixedCylinderModel(self.cfg)

    def stresses_and_gap(self, model, points):
        value = model(points)
        nx, ny = points[:, :1], points[:, 1:2]-1
        tx = value[:, 2:3]*nx+value[:, 4:5]*ny
        ty = value[:, 4:5]*nx+value[:, 3:4]*ny
        gap = points[:, 1:2]/self.cfg.load_ratio+value[:, 1:2]
        return tx, ty, gap

    def raw_residuals(self, model):
        constitutive_points = torch.cat((self.interior, self.arc)) if self.cfg.boundary_constitutive else self.interior
        value, (dx, dy) = model(constitutive_points, 1)
        ell, nu = self.cfg.local_scale, self.cfg.poisson
        lam, mu = nu/((1+nu)*(1-2*nu)), 1/(2*(1+nu))
        n = len(constitutive_points)
        eq = torch.cat((dx[:n, 2:3]+dy[:n, 4:5], dx[:n, 4:5]+dy[:n, 3:4]), 1)
        trace = dx[:, :1]+dy[:, 1:2]
        constitutive = torch.cat((value[:, 2:3]-ell*(lam*trace+2*mu*dx[:, :1]),
                                  value[:, 3:4]-ell*(lam*trace+2*mu*dy[:, 1:2]),
                                  value[:, 4:5]-ell*mu*(dy[:, :1]+dx[:, 1:2])), 1)
        if self.cfg.formulation == 'displacement':
            top = model(self.top)
            constitutive = torch.cat((top[:,4:5],top[:,3:4]+ell),1)/ell
        tx, p, gap = self.stresses_and_gap(model, self.arc)
        _, pressure_q, _ = self.stresses_and_gap(model, self.arc_q)
        balance = ((pressure_q.flatten()*self.weight_q).sum()/ell-1).reshape(1)
        qvalue = model(self.quad_points)
        x,y = self.quad_points[:,0],self.quad_points[:,1]
        ax,ay = self.arc_q[:,0],self.arc_q[:,1]
        virtual=[]
        for component,m,k in self.virtual_tests:
            ddx = m*x.pow(m-1)*y.pow(k) if m else torch.zeros_like(x)
            ddy = k*x.pow(m)*y.pow(k-1) if k else torch.zeros_like(x)
            if component == 'u':
                internal=qvalue[:,2]*ddx+qvalue[:,4]*ddy
                external=0.
            else:
                internal=qvalue[:,4]*ddx+qvalue[:,3]*ddy
                external=(pressure_q.flatten()*ax.pow(m)*ay.pow(k)*self.weight_q).sum()-ell/(m+1)
            virtual.append(((internal*self.quad_weights).sum()-external)/ell)
        return [eq, constitutive, tx, fischer_burmeister(gap, p), balance, torch.stack(virtual)]

    def losses(self, model):
        self.n_eval += 1
        return torch.stack([w*r.square().mean() for w, r in zip(self.weights, self.raw_residuals(model))])

    def residual_function(self, model):
        names, shapes, sizes = zip(*[(n, p.shape, p.numel()) for n, p in model.named_parameters()])
        def residual(flat):
            params = {n: v.view(shape) for n, v, shape in zip(names, flat.split(sizes), shapes)}
            def wrapper(points, order=0):
                return functional_call(model, params, (points, order))
            return torch.cat([(r*math.sqrt(w/r.numel())).flatten() for w, r in zip(self.weights, self.raw_residuals(wrapper))])
        return residual

    @torch.no_grad()
    def evaluate(self, model):
        cfg = self.cfg
        value = model(self.grid_points)
        u = value[:, :2]
        tx, p_arc, gap = self.stresses_and_gap(model, self.pressure_points)
        # Projected pressure differs from traction per original arc by dx/ds.
        cosine = torch.sqrt((1-self.pressure_x.square()).clamp_min(0))
        p_projected_over_q = p_arc.flatten()/cfg.local_scale/cosine.clamp_min(1e-3)
        # The vertical tangent endpoint x=1 has dx/ds=0: pressure comparison only on x<=0.3.
        pressure_mask = self.pressure_x <= .3
        p_projected_over_q[-1] = 0
        error_p = p_projected_over_q[pressure_mask]-self.truth_pressure_over_q[pressure_mask]
        hertz_error = p_projected_over_q[pressure_mask]-self.hertz_pressure_over_q[pressure_mask]
        theta = torch.linspace(0, math.pi/2, 801)
        arc = arc_points(theta)
        _, audit_p, audit_g = self.stresses_and_gap(model, arc)
        reaction_n = 2*torch.trapezoid(audit_p.flatten(), theta).item()*cfg.stress_mpa*cfg.radius_mm*cfg.thickness_mm
        # Width from pressure >1% Hertz peak: a threshold-dependent observable, clearly labeled.
        threshold = .01*self.hertz_pressure_over_q.max()
        active = pressure_mask & (p_projected_over_q > threshold)
        width = self.pressure_x[active].max().item() if active.any() else 0
        metrics = {'relative_l2': ((u-self.truth_u).norm()/self.truth_u.norm()).item(),
                   'pressure_relative_l2': (error_p.norm()/self.truth_pressure_over_q[pressure_mask].norm()).item(),
                   'hertz_pressure_relative_l2': (hertz_error.norm()/self.hertz_pressure_over_q[pressure_mask].norm()).item(),
                   'reaction_n': reaction_n, 'force_balance_relative': abs(reaction_n-cfg.applied_force_n)/cfg.applied_force_n,
                   'max_penetration_um': torch.relu(-audit_g).max().item()*cfg.displacement_um,
                   'max_tension_mpa': torch.relu(-audit_p).max().item()*cfg.stress_mpa,
                   'independent_fb_rms': fischer_burmeister(audit_g, audit_p).square().mean().sqrt().item(),
                   'contact_half_width_mm_threshold': width*cfg.radius_mm}
        fields = {}
        for name, component, unit in [('ux_um', 0, cfg.displacement_um), ('uy_um', 1, cfg.displacement_um),
                                       ('syy_mpa', 3, cfg.stress_mpa)]:
            flat = np.zeros(cfg.grid*cfg.grid)
            flat[self.mask] = (value[:, component]*unit).numpy()
            fields[name] = flat.reshape(cfg.grid, cfg.grid).tolist()
        fields['pressure_mpa'] = (p_projected_over_q*cfg.top_pressure_mpa).tolist()
        fields['gap_um'] = (gap.flatten()*cfg.displacement_um).tolist()
        return metrics, fields

    def metadata(self):
        cfg = self.cfg
        truth = {}
        for name, component in [('ux_um', 0), ('uy_um', 1)]:
            flat = np.zeros(cfg.grid*cfg.grid); flat[self.mask] = (self.truth_u[:, component]*cfg.displacement_um).numpy()
            truth[name] = flat.reshape(cfg.grid, cfg.grid).tolist()
        return {'id': self.id, 'title': '二维弹性半圆柱压刚性平面 · KKT 接触',
                'loss_names': self.loss_names, 'weights': self.weights,
                'axis': np.linspace(0, 1, cfg.grid).tolist(), 'mask': self.mask.reshape(cfg.grid, cfg.grid).tolist(),
                'truth': truth, 'pressure_x_mm': (self.pressure_x*cfg.radius_mm).tolist(),
                'pressure_fem_mpa': (self.truth_pressure_over_q*cfg.top_pressure_mpa).tolist(),
                'pressure_hertz_mpa': (self.hertz_pressure_over_q*cfg.top_pressure_mpa).tolist(),
                'gap_fem_um': (np.array(self.reference['gap_over_u0'])*cfg.displacement_um).tolist(),
                'hertz_half_width_mm': cfg.hertz_half_width_over_r*cfg.radius_mm,
                'applied_force_n': cfg.applied_force_n, 'reference_validation': self.reference['validation'],
                'field_names': ['uy_um', 'ux_um'],
                'units': {'radius_mm': cfg.radius_mm, 'young_mpa': cfg.young_mpa, 'poisson': cfg.poisson,
                          'top_pressure_mpa': cfg.top_pressure_mpa, 'thickness_mm': cfg.thickness_mm}}
