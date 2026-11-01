import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from torch.func import jacrev
from physics import Config, PINN, Problem, exact, flat_parameters, loss_gradients
from optimizers import LM, SOAP, config_direction, pcgrad_direction


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(42)


def test_analytic_solution_satisfies_pde_and_conditions():
    points = torch.rand(15, 2, requires_grad=True)
    u = exact(points)
    grad = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    uxx = torch.autograd.grad(grad[:, 0].sum(), points)[0][:, :1]
    torch.testing.assert_close(grad[:, 1:]-0.1*uxx, torch.zeros_like(uxx), atol=1e-12, rtol=0)
    boundary = torch.tensor([[0., .3], [1., .9]])
    torch.testing.assert_close(exact(boundary), torch.zeros(2, 1), atol=1e-14, rtol=0)


def test_propagated_derivatives_match_input_autodiff():
    model = PINN(8, 3)
    points = torch.rand(9, 2, requires_grad=True)
    u = model(points)
    grad = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    uxx = torch.autograd.grad(grad[:, :1].sum(), points, create_graph=True)[0][:, :1]
    predicted, ut, propagated_uxx = model(points, True)
    torch.testing.assert_close(predicted, u)
    torch.testing.assert_close(ut, grad[:, 1:])
    torch.testing.assert_close(propagated_uxx, uxx)


def test_residual_jacobian_matches_finite_difference():
    cfg = Config(width=4, depth=1, n_interior=5, n_initial=4, n_boundary=3)
    model, problem = PINN(4, 1), Problem(cfg)
    residual = problem.residual_function(model)
    flat = flat_parameters(model)
    direction = torch.randn_like(flat)
    j = jacrev(residual)(flat)
    eps = 1e-6
    finite = (residual(flat+eps*direction)-residual(flat-eps*direction))/(2*eps)
    torch.testing.assert_close(j@direction, finite, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(residual(flat).square().sum(), problem.losses(model).sum())


def test_fixed_sampling_and_perfect_metrics():
    a, b = Problem(Config()), Problem(Config())
    torch.testing.assert_close(a.points, b.points, rtol=0, atol=0)
    assert not torch.equal(a.interior, a.test_residual[:len(a.interior)])

    class ExactModel:
        def __call__(self, points, derivatives=False):
            if not derivatives:
                return exact(points)
            # The exact heat solution has ut = nu*uxx.
            return exact(points), torch.zeros(len(points), 1), torch.zeros(len(points), 1)
    metrics, _, _ = a.evaluate(ExactModel())
    assert metrics['relative_l2'] == 0
    assert metrics['within_tolerance_pct'] == 100


def test_config_is_common_descent_for_independent_gradients():
    grads = torch.tensor([[1., 0., 0.], [-0.5, 1., 0.], [0., -0.3, 1.]])
    direction = config_direction(grads)
    assert torch.all(grads@direction > 0)
    torch.testing.assert_close(config_direction(torch.zeros_like(grads)), torch.zeros(3))


def test_pcgrad_removes_pairwise_conflict():
    grads = torch.tensor([[1., 0.], [-1., 1.]])
    direction = pcgrad_direction(grads, torch.Generator().manual_seed(1))
    assert torch.all(grads@direction >= -1e-14)


def test_lm_accepted_step_reduces_same_objective():
    cfg = Config(width=6, depth=1, n_interior=8, n_initial=6, n_boundary=4)
    model, problem = PINN(6, 1), Problem(cfg)
    old = problem.losses(model).sum().item()
    optimizer = LM(model, problem)
    optimizer.step()
    assert optimizer.last_accepted
    assert problem.losses(model).sum().item() < old


def test_soap_remains_finite_across_basis_refresh():
    model, problem = PINN(6, 1), Problem(Config(n_interior=8, n_initial=6, n_boundary=4))
    optimizer = SOAP(model, frequency=2)
    for _ in range(5):
        model.zero_grad(set_to_none=True)
        problem.losses(model).sum().backward()
        optimizer.step()
    assert torch.isfinite(flat_parameters(model)).all()
