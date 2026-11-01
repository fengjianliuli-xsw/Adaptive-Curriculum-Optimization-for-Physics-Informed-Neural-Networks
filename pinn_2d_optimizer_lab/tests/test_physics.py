import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from torch.func import jacrev
from physics import Config, JetMLP, ContactModel, HeatProblem, ContactProblem, heat_exact, heat_source, flat_parameters
from reference import generate_reference
from optimizers import LM


@pytest.fixture(autouse=True)
def setup():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(42)


def test_heat_manufactured_source():
    points = torch.rand(11, 2, requires_grad=True)
    u = heat_exact(points)
    g = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    xx = torch.autograd.grad(g[:, 0].sum(), points, retain_graph=True)[0][:, :1]
    yy = torch.autograd.grad(g[:, 1].sum(), points)[0][:, 1:2]
    torch.testing.assert_close(-xx-yy, heat_source(points))


def test_jet_hessians_match_autodiff():
    model = JetMLP(2, 2, width=5)
    points = torch.rand(7, 2, requires_grad=True)
    value, first, second = model(points, 2)
    for out in range(2):
        grad = torch.autograd.grad(value[:, out].sum(), points, create_graph=True, retain_graph=True)[0]
        for axis in range(2):
            hessian = torch.autograd.grad(grad[:, axis].sum(), points, retain_graph=True)[0][:, axis]
            torch.testing.assert_close(first[axis][:, out], grad[:, axis])
            torch.testing.assert_close(second[axis][:, out], hessian)


def test_contact_transform_derivatives_and_bottom_constraint():
    model = ContactModel(width=5)
    points = torch.rand(7, 2, requires_grad=True)
    value, derivatives = model(points, 1)
    for out in range(5):
        grad = torch.autograd.grad(value[:, out].sum(), points, retain_graph=True)[0]
        for axis in range(2):
            torch.testing.assert_close(derivatives[axis][:, out], grad[:, axis])
    bottom = torch.rand(9, 2); bottom[:, 1] = 0
    assert model(bottom)[:, :2].abs().max() == 0
    x = torch.linspace(0, 1, 101).reshape(-1, 1)
    pressure = model.pressure(x)
    assert torch.all(pressure >= 0)
    assert pressure[(x[:, 0] < 0.2) | (x[:, 0] > 0.8)].abs().max() == 0


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    torch.set_default_dtype(torch.float64)
    cfg = Config(width=5, n_contact=8, n_sensors=8, grid=11)
    return cfg, generate_reference(cfg, tmp_path_factory.mktemp('reference')/'fem.json')


def test_fem_convergence_and_force_balance(reference):
    _, ref = reference
    assert ref['validation']['coarse_fine_displacement_relative'] < 0.01
    v = ref['validation']['fine']
    assert v['force_balance_relative'] < 1e-8
    assert v['assembled_force_n'] > 0


@pytest.mark.parametrize('kind', ['heat', 'contact'])
def test_residual_jacobian_and_lm_descent(kind, reference):
    cfg, ref = reference
    problem = HeatProblem(cfg) if kind == 'heat' else ContactProblem(cfg, ref)
    model = problem.make_model()
    residual = problem.residual_function(model)
    flat = flat_parameters(model)
    j = jacrev(residual, chunk_size=32)(flat)
    direction = torch.randn_like(flat)
    eps = 1e-6
    finite = (residual(flat+eps*direction)-residual(flat-eps*direction))/(2*eps)
    torch.testing.assert_close(j@direction, finite, atol=1e-7, rtol=1e-5)
    before = problem.losses(model).sum()
    torch.testing.assert_close(residual(flat).square().sum(), before)
    optimizer = LM(model, problem)
    optimizer.step()
    assert optimizer.last_accepted
    assert problem.losses(model).sum() < before


def test_contact_truth_not_used_by_training(reference):
    cfg, ref = reference
    problem = ContactProblem(cfg, ref)
    model = problem.make_model()
    before = problem.losses(model).detach().clone()
    problem.pressure_truth *= 1000
    problem.truth_displacement *= -2000
    problem.force_truth = 99999
    torch.testing.assert_close(problem.losses(model), before, rtol=0, atol=0)
