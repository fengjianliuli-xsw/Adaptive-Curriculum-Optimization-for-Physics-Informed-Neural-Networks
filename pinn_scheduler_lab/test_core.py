import math
import numpy as np
import torch
from physics import Config,Problem,MLP,parameters_hash
from schedulers import Controller

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def jets_by_autograd(function,points):
    x=points.detach().clone().requires_grad_(True);value=function(x)
    dx=[];dy=[];dxx=[];dyy=[]
    for k in range(value.shape[1]):
        g=torch.autograd.grad(value[:,k].sum(),x,create_graph=True,retain_graph=True)[0]
        h0=torch.autograd.grad(g[:,0].sum(),x,retain_graph=True,create_graph=True)[0]
        h1=torch.autograd.grad(g[:,1].sum(),x,retain_graph=True,create_graph=True)[0]
        dx.append(g[:,0]);dy.append(g[:,1]);dxx.append(h0[:,0]);dyy.append(h1[:,1])
    return (value,*[torch.stack(q,1) for q in (dx,dy,dxx,dyy)])


def test_network_jets_and_parameter_gradients():
    torch.manual_seed(13);model=MLP(3,Config(width=8,depth=2));points=torch.rand(7,2)
    analytic=model(points,2);automatic=jets_by_autograd(model,points)
    for a,b in zip(analytic,automatic):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    pa=torch.autograd.grad(sum(q.square().sum() for q in analytic),list(model.parameters()),retain_graph=True)
    pb=torch.autograd.grad(sum(q.square().sum() for q in automatic),list(model.parameters()))
    for a,b in zip(pa,pb):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-9)


def test_exact_pdes_and_boundary():
    for kind in ['heat2d','ns2d']:
        problem=Problem(kind,Config(),42)
        class Exact:
            def __call__(self,points,order=0):return jets_by_autograd(problem.exact,points) if order else problem.exact(points)
        model=Exact()
        assert problem.residuals(model,problem.train_points).abs().max()<1e-12
        assert problem.bc_loss(model).item()<1e-25


def test_equal_initialization_and_samples():
    hashes=[];samples=[]
    for _ in range(5):
        torch.manual_seed(42);problem=Problem('ns2d',Config(),42);hashes.append(parameters_hash(problem.make_model()));samples.append(problem.sample_hash)
    assert len(set(hashes))==len(set(samples))==1


def test_probe_handles_constant_output_nullspace():
    for kind in ['heat2d','ns2d']:
        p=Problem(kind,Config(),42);s=p.probe(p.make_model())
        assert all(math.isfinite(x) for x in s.values())
        assert -1.00001<=s['gradient_cosine']<=1.00001


def state(bc=.1,pde=.2,cos=0,drop=.1):
    return dict(e_bc=bc,e_pde=pde,gradient_cosine=cos,relative_pde_drop=drop,relative_bc_drop=drop)


def test_policy_revisits_and_hysteresis():
    c=Controller('E',Config(),1)
    assert c.decision(state(),0)[0]=='BC';c.action='BC'
    assert c.decision(state(.01),.1)[0]=='BC' # dwell
    assert c.decision(state(.01),.2)[0]=='PDE';c.action='PDE'
    assert c.decision(state(.1),.3)[0]=='PDE' # dwell
    assert c.decision(state(.1),.4)[0]=='BC'
    c=Controller('E',Config(),1)
    assert c.decision(state(.03),0)[0]=='Joint'
    assert c.decision(state(.01,.01),0)[0]=='Joint'


def test_policy_needs_no_solution_error():
    # State has only physical checks; arbitrary truth-only metrics cannot affect decisions.
    a=Controller('E',Config(),1);b=Controller('E',Config(),1)
    assert a.decision(state(),0)==b.decision(dict(state(),relative_l2=1e100),0)
    assert Controller('B',Config(),1).decision({},.4)[0]=='PDE'
    assert Controller('C',Config(),1).decision({},.4)[0]=='BC'


def test_relobralo_formula_and_positive_normalization():
    cfg=Config(relobralo_beta=0.,relobralo_alpha=.9,relobralo_temperature=1.)
    c=Controller('D',cfg,42);np.testing.assert_allclose(c.reweight([2.,4.]),[1.,1.])
    expected=2*np.exp([1.,.5])/np.exp([1.,.5]).sum()
    np.testing.assert_allclose(c.reweight([2.,2.]),expected)
    def soft(x):
        a=np.exp(np.asarray(x)-max(x));return 2*a/a.sum()
    np.testing.assert_allclose(c.reweight([1.,1.]),.9*soft([.5,.25])+.1*soft([.5,.5]))
    for loss in [[1e-20,1.],[.1,.7],[.4,.2]]:
        w=c.reweight(loss);assert np.all(w>0);assert abs(w.sum()-2)<1e-12


def test_relobralo_retains_history_when_rho_is_one():
    cfg=Config(relobralo_beta=1.,relobralo_alpha=.9,relobralo_temperature=1.)
    c=Controller('D',cfg,42);c.reweight([2.,4.])
    expected=.9*np.ones(2)+.1*(2*np.exp([1.,.5])/np.exp([1.,.5]).sum())
    np.testing.assert_allclose(c.reweight([2.,2.]),expected)
