import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from torch.func import jacrev
from physics import Config, Problem, CylinderModel, MixedCylinderModel, fischer_burmeister, flat_parameters
from reference import generate
from optimizers import LM, dcgd_direction, config_direction


@pytest.fixture(autouse=True)
def setup():
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1); torch.manual_seed(42)


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    cfg = Config(width=4, depth=2, grid=15, n_interior=8, n_arc=8)
    result = generate(cfg, tmp_path_factory.mktemp('fem')/'reference.json')
    return cfg, result


def test_fb_encodes_kkt():
    g = torch.tensor([0., 1., 0., -1., 1.])
    p = torch.tensor([1., 0., -1., 0., 1.])
    phi = fischer_burmeister(g, p)
    assert phi[:2].abs().max() < 1e-12
    assert phi[2:].abs().min() > .5


def test_dcgd_center_matches_two_group_config():
    grads=torch.tensor([[1.,0.,.1],[-.4,1.,.3]])
    d=dcgd_direction(grads)
    assert torch.all(grads@d > 0)
    torch.testing.assert_close(d,config_direction(grads),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(dcgd_direction(torch.tensor([[1.,0.],[-1.,0.]])),torch.zeros(2))


def test_hertz_units_and_force():
    cfg = Config()
    assert abs(cfg.hertz_half_width_over_r-.07611333607551958) < 1e-12
    assert cfg.applied_force_n == 100
    assert cfg.displacement_um == 25
    x = np.linspace(0, cfg.hertz_half_width_over_r, 10001)
    p_over_q = 4/(np.pi*cfg.hertz_half_width_over_r)*np.sqrt(np.maximum(0,1-(x/cfg.hertz_half_width_over_r)**2))
    # Full line load is 2qR, so the integral of p/q on the positive half is 1.
    assert abs(np.trapz(p_over_q, x)-1) < 1e-5


@pytest.mark.parametrize('kind',['displacement','mixed'])
def test_derivatives_and_hard_symmetry_top_conditions(kind):
    cfg = Config(width=5)
    model = CylinderModel(cfg) if kind == 'displacement' else MixedCylinderModel(cfg)
    points = torch.rand(7,2,requires_grad=True)
    value, first = model(points,1)
    for out in range(5):
        grad = torch.autograd.grad(value[:,out].sum(), points, retain_graph=True)[0]
        for axis in range(2):
            torch.testing.assert_close(first[axis][:,out], grad[:,axis])
    top=torch.rand(9,2); top[:,1]=1
    symmetry=torch.rand(9,2); symmetry[:,0]=0
    if kind == 'mixed':
        torch.testing.assert_close(model(top)[:,3], torch.full((9,),-cfg.local_scale))
        assert model(top)[:,4].abs().max() == 0
    assert model(symmetry)[:,[0,4]].abs().max() == 0


def test_reference_unilateral_contact_and_hertz(reference):
    cfg, ref=reference
    v=ref['validation']
    assert v['fine']['force_balance_relative'] < 1e-8
    assert v['fine']['max_penetration_over_r'] < 1e-8
    assert v['fine']['min_contact_nodal_reaction'] >= -1e-10
    assert v['coarse_fine_displacement_relative'] < .02
    assert v['fem_hertz_pressure_relative'] < .05


def test_material_plane_strain(reference):
    cfg,ref=reference; problem=Problem(cfg,ref)
    def homogeneous(points, order=0):
        x,y=points[:,:1],points[:,1:2]
        nu,ell=cfg.poisson,cfg.local_scale
        zero=torch.zeros_like(x)
        value=torch.cat((nu*(1+nu)*x,-(1-nu**2)*y,zero,zero-ell,zero),1)
        if not order: return value
        dx=torch.cat((zero+nu*(1+nu),zero,zero,zero,zero),1)
        dy=torch.cat((zero,zero-(1-nu**2),zero,zero,zero),1)
        return value,[dx,dy]
    residuals=problem.raw_residuals(homogeneous)
    assert residuals[0].abs().max() < 1e-14
    assert residuals[1].abs().max() < 1e-14


def test_residual_jacobian_and_descent(reference):
    cfg,ref=reference; problem=Problem(cfg,ref); model=problem.make_model()
    function=problem.residual_function(model); flat=flat_parameters(model)
    j=jacrev(function,chunk_size=32)(flat)
    vector=torch.randn_like(flat); eps=1e-6
    fd=(function(flat+eps*vector)-function(flat-eps*vector))/(2*eps)
    torch.testing.assert_close(j@vector,fd,atol=1e-7,rtol=1e-5)
    before=problem.losses(model).sum()
    torch.testing.assert_close(function(flat).square().sum(),before)
    optimizer=LM(model,problem); optimizer.step()
    assert optimizer.last_accepted
    assert problem.losses(model).sum() < before


def test_no_fem_or_hertz_truth_in_training(reference):
    cfg,ref=reference; problem=Problem(cfg,ref); model=problem.make_model()
    before=problem.losses(model).detach().clone()
    problem.truth_pressure_over_q *= 100
    problem.hertz_pressure_over_q *= -100
    problem.truth_u *= 123
    torch.testing.assert_close(before,problem.losses(model),rtol=0,atol=0)
