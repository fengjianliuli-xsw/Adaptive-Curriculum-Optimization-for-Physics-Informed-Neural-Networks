"""One spatial dimension + time. Analytic derivatives keep the demo CPU friendly."""
from dataclasses import asdict, dataclass
import hashlib
import math

import torch
from torch import nn
from torch.func import functional_call
from torch.nn.utils import parameters_to_vector


@dataclass
class Config:
    seed: int = 42
    width: int = 24
    depth: int = 2
    nu: float = 0.1
    n_interior: int = 128
    n_initial: int = 48
    n_boundary: int = 48  # per boundary
    steps: int = 1200
    record_every: int = 10
    threads: int = 1
    tolerance: float = 0.05  # absolute pointwise error, not a classification accuracy

    def to_dict(self):
        return asdict(self)


def exact(points, nu=0.1):
    x, t = points[:, :1], points[:, 1:]
    return (torch.sin(math.pi*x)*torch.exp(-nu*math.pi**2*t)
            + 0.3*torch.sin(3*math.pi*x)*torch.exp(-nu*9*math.pi**2*t))


class PINN(nn.Module):
    def __init__(self, width=24, depth=2):
        super().__init__()
        dims = [2] + [width]*depth + [1]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, points, derivatives=False):
        z = 2*points - 1
        if derivatives:
            zx = torch.zeros_like(z); zx[:, 0] = 2
            zt = torch.zeros_like(z); zt[:, 1] = 2
            zxx = torch.zeros_like(z)
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if derivatives:
                zx = zx @ layer.weight.T
                zt = zt @ layer.weight.T
                zxx = zxx @ layer.weight.T
            if i < len(self.layers)-1:
                z = torch.tanh(z)
                if derivatives:
                    a = 1-z.square()
                    zxx = a*zxx - 2*z*a*zx.square()
                    zx, zt = a*zx, a*zt
        return (z, zt, zxx) if derivatives else z


def fingerprint(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class Problem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.interior = torch.quasirandom.SobolEngine(2, scramble=True, seed=cfg.seed+1).draw(cfg.n_interior).double()
        x = torch.linspace(0, 1, cfg.n_initial)
        t = torch.linspace(0, 1, cfg.n_boundary)
        self.initial = torch.stack((x, torch.zeros_like(x)), 1)
        self.boundary = torch.cat((torch.stack((torch.zeros_like(t), t), 1),
                                   torch.stack((torch.ones_like(t), t), 1)))
        self.target_initial = exact(self.initial, cfg.nu)
        self.points = torch.cat((self.interior, self.initial, self.boundary))
        self.test_residual = torch.quasirandom.SobolEngine(2, scramble=True, seed=cfg.seed+999).draw(512).double()
        self.eval_x = torch.linspace(0, 1, 121)
        self.eval_t = torch.linspace(0, 1, 41)
        tt, xx = torch.meshgrid(self.eval_t, self.eval_x, indexing='ij')
        self.grid = torch.stack((xx.flatten(), tt.flatten()), 1)
        self.truth = exact(self.grid, cfg.nu).reshape(41, 121)
        self.slice_indices = [0, 10, 20, 30, 40]
        self.n_eval = 0

    def raw_residuals(self, output):
        u, ut, uxx = output
        nf, ni = self.cfg.n_interior, self.cfg.n_initial
        return (ut[:nf]-self.cfg.nu*uxx[:nf],
                u[nf:nf+ni]-self.target_initial,
                u[nf+ni:])

    def losses(self, model):
        self.n_eval += 1
        return torch.stack([r.square().mean() for r in self.raw_residuals(model(self.points, True))])

    def residual_function(self, model):
        names, shapes, sizes = zip(*[(n, p.shape, p.numel()) for n, p in model.named_parameters()])

        def residual(flat):
            params = {n: v.view(shape) for n, v, shape in zip(names, flat.split(sizes), shapes)}
            out = functional_call(model, params, (self.points, True))
            return torch.cat([r.flatten()/math.sqrt(r.numel()) for r in self.raw_residuals(out)])
        return residual

    @torch.no_grad()
    def evaluate(self, model):
        pred = model(self.grid).reshape_as(self.truth)
        error = pred-self.truth
        _, ut, uxx = model(self.test_residual, True)
        metrics = {
            'relative_l2': (error.norm()/self.truth.norm()).item(),
            'rmse': error.square().mean().sqrt().item(),
            'max_abs_error': error.abs().max().item(),
            'within_tolerance_pct': (error.abs() <= self.cfg.tolerance).double().mean().item()*100,
            'test_pde_mse': (ut-self.cfg.nu*uxx).square().mean().item(),
        }
        return metrics, pred[self.slice_indices].tolist(), pred.tolist()


def flat_parameters(model):
    return parameters_to_vector(model.parameters()).detach().clone()


def assign(model, flat):
    # Copy into existing storage: avoid changing optimizer parameter references.
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(flat[offset:offset+p.numel()].view_as(p))
            offset += p.numel()


def loss_gradients(losses, params):
    return torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(
        loss, params, retain_graph=True)]) for loss in losses])
