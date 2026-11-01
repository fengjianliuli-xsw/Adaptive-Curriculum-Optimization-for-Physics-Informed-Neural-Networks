"""Two spatial PDE benchmarks with common initialization and independent references."""
from dataclasses import asdict, dataclass
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
    width: int = 24
    depth: int = 2
    steps: int = 1200
    record_every: int = 20
    n_heat: int = 128
    n_contact: int = 64
    n_edge: int = 16
    n_sensors: int = 48
    noise: float = 0.0
    threads: int = 1
    grid: int = 31

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
    return torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(
        value, params, retain_graph=True)]) for value in losses])


class JetMLP(nn.Module):
    """Exact tanh chain-rule jets, differentiable w.r.t. every network parameter."""
    def __init__(self, inputs=2, outputs=1, width=24, depth=2):
        super().__init__()
        dims = [inputs]+[width]*depth+[outputs]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.inputs = inputs
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, points, order=0):
        z = 2*points-1
        if order:
            tangents = [torch.zeros_like(z) for _ in range(self.inputs)]
            seconds = [torch.zeros_like(z) for _ in range(self.inputs)]
            for i in range(self.inputs):
                tangents[i][:, i] = 2
        for index, layer in enumerate(self.layers):
            z = layer(z)
            if order:
                tangents = [d@layer.weight.T for d in tangents]
                if order == 2:
                    seconds = [d@layer.weight.T for d in seconds]
            if index != len(self.layers)-1:
                z = torch.tanh(z)
                if order:
                    a = 1-z.square()
                    if order == 2:
                        seconds = [a*h-2*z*a*d.square() for h, d in zip(seconds, tangents)]
                    tangents = [a*d for d in tangents]
        if order == 0:
            return z
        return (z, tangents, seconds) if order == 2 else (z, tangents)


class ContactModel(nn.Module):
    def __init__(self, width=24, depth=2):
        super().__init__()
        self.field = JetMLP(2, 5, width, depth)
        self.pressure_head = JetMLP(1, 1, 12, 2)

    def forward(self, points, order=0):
        result = self.field(points, order)
        y = points[:, 1:2]
        if not order:
            return torch.cat((y*result[:, :2], result[:, 2:]), 1)
        value, tangent = result
        transformed = torch.cat((y*value[:, :2], value[:, 2:]), 1)
        dx = torch.cat((y*tangent[0][:, :2], tangent[0][:, 2:]), 1)
        dy = torch.cat((value[:, :2]+y*tangent[1][:, :2], tangent[1][:, 2:]), 1)
        return transformed, [dx, dy]

    def pressure(self, x):
        # Known contact support; smooth nonnegative pressure, no truth coefficients.
        window = torch.relu(1-((x-0.5)/0.3).square()).square()
        return torch.nn.functional.softplus(self.pressure_head(x))*window


def heat_exact(points):
    x, y = points[:, :1], points[:, 1:2]
    return torch.sin(math.pi*x)*torch.sin(math.pi*y)+0.2*torch.sin(3*math.pi*x)*torch.sin(2*math.pi*y)


def heat_source(points):
    x, y = points[:, :1], points[:, 1:2]
    return 2*math.pi**2*torch.sin(math.pi*x)*torch.sin(math.pi*y)+2.6*math.pi**2*torch.sin(3*math.pi*x)*torch.sin(2*math.pi*y)


def true_pressure(x):
    # Used exclusively by reference.py and evaluation, never by training residuals.
    return np.maximum(0, 1-((np.asarray(x)-0.5)/0.3)**2)**2*(1+0.25*np.sin(2*np.pi*x)+0.15*np.cos(4*np.pi*x))


class BaseProblem:
    def finish_setup(self):
        axis = torch.linspace(0, 1, self.cfg.grid)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.grid = torch.stack((xx.flatten(), yy.flatten()), 1)
        self.axis = axis
        self.n_eval = 0
        self.sample_hash = fingerprint(torch.cat([p.flatten() for p in self.training_tensors]))

    def residual_function(self, model):
        names, shapes, sizes = zip(*[(n, p.shape, p.numel()) for n, p in model.named_parameters()])

        def residual(flat):
            params = {n: v.view(shape) for n, v, shape in zip(names, flat.split(sizes), shapes)}
            wrapper = FunctionalModel(model, params)
            return torch.cat([(r*math.sqrt(weight/r.numel())).flatten()
                              for r, weight in zip(self.raw_residuals(wrapper), self.weights)])
        return residual

    def losses(self, model):
        self.n_eval += 1
        return torch.stack([weight*r.square().mean() for r, weight in zip(self.raw_residuals(model), self.weights)])


class FunctionalModel:
    def __init__(self, model, params):
        self.model, self.params = model, params

    def __call__(self, points, order=0):
        return functional_call(self.model, self.params, (points, order))

    def pressure(self, x):
        params = {key[len('pressure_head.'):]: value for key, value in self.params.items() if key.startswith('pressure_head.')}
        window = torch.relu(1-((x-0.5)/0.3).square()).square()
        return torch.nn.functional.softplus(functional_call(self.model.pressure_head, params, (x,)))*window


def edges(n):
    axis = torch.linspace(0.01, 0.99, n)
    return [torch.stack((axis, torch.zeros_like(axis)), 1),
            torch.stack((axis, torch.ones_like(axis)), 1),
            torch.stack((torch.zeros_like(axis), axis), 1),
            torch.stack((torch.ones_like(axis), axis), 1)]


class HeatProblem(BaseProblem):
    id = 'heat2d'
    title = '二维稳态热传导 / Poisson'
    loss_names = ['pde', 'boundary']
    weights = [1.0, 1.0]
    field_names = ['temperature']
    def __init__(self, cfg, reference=None):
        self.cfg = cfg
        self.interior = torch.quasirandom.SobolEngine(2, scramble=True, seed=cfg.seed+11).draw(cfg.n_heat).double()
        self.boundary = torch.cat(edges(cfg.n_edge))
        self.source = heat_source(self.interior)
        self.training_tensors = [self.interior, self.boundary, self.source]
        self.finish_setup()
        self.truth = heat_exact(self.grid).reshape(cfg.grid, cfg.grid)

    def make_model(self):
        return JetMLP(2, 1, self.cfg.width, self.cfg.depth)

    def raw_residuals(self, model):
        _, _, second = model(self.interior, 2)
        return [(-second[0]-second[1]-self.source)/(2*math.pi**2), model(self.boundary)]

    @torch.no_grad()
    def evaluate(self, model):
        pred = model(self.grid).reshape_as(self.truth)
        error = pred-self.truth
        return {'relative_l2': (error.norm()/self.truth.norm()).item(),
                'within_tolerance_pct': (error.abs() <= 0.05).double().mean().item()*100,
                'max_abs_error': error.abs().max().item()}, {'temperature': pred.tolist()}

    def metadata(self):
        return {'id': self.id, 'title': self.title, 'axis': self.axis.tolist(),
                'field_names': self.field_names, 'truth': {'temperature': self.truth.tolist()},
                'cut_x': self.axis.tolist(), 'cut_truth': self.truth[self.cfg.grid//2].tolist(),
                'cut_label': 'y = 0.5 温度剖面', 'loss_names': self.loss_names, 'weights': self.weights,
                'equation': '−(u_xx + u_yy) = f(x,y), (x,y) ∈ [0,1]²; u|∂Ω = 0',
                'reference': '独立网格上的解析解；源项为已知输入，内部真解不参与训练。',
                'primary_label': '温度相对 L2 误差'}


class ContactProblem(BaseProblem):
    id = 'contact2d'
    title = '二维弹性接触载荷反演'
    loss_names = ['equilibrium', 'constitutive', 'traction', 'observation']
    weights = [1.0, 1.0, 1.0, 10.0]
    field_names = ['uy_mm', 'ux_mm']
    # Units: W=H*2=40 mm; thickness=10 mm; E=1 MPa; p0=0.1 MPa.
    aspect = 2.0
    poisson = 0.3
    displacement_mm = 4.0
    pressure_kpa = 100.0
    force_scale_n = 40.0

    def __init__(self, cfg, reference):
        self.cfg = cfg
        self.interior = torch.quasirandom.SobolEngine(2, scramble=True, seed=cfg.seed+11).draw(cfg.n_contact).double()
        self.top, self.left, self.right = edges(cfg.n_edge)[1:]
        self.edge_points = torch.cat((self.top, self.left, self.right))
        self.sensors = 0.1+0.8*torch.quasirandom.SobolEngine(2, scramble=True, seed=cfg.seed+37).draw(cfg.n_sensors).double()
        self.observed_clean = torch.tensor(reference['sensor_displacement'])
        gen = torch.Generator().manual_seed(cfg.seed+100)
        self.observed = self.observed_clean+cfg.noise*self.observed_clean.square().mean().sqrt()*torch.randn(self.observed_clean.shape, generator=gen)
        self.training_tensors = [self.interior, self.edge_points, self.sensors, self.observed]
        self.finish_setup()
        self.truth_displacement = torch.tensor(reference['grid_displacement'])
        self.pressure_x = torch.linspace(0, 1, 121).reshape(-1, 1)
        self.pressure_truth = torch.tensor(true_pressure(self.pressure_x.numpy()))
        self.force_truth = torch.trapezoid(self.pressure_truth.flatten(), self.pressure_x.flatten()).item()*self.force_scale_n
        self.reference_validation = reference['validation']

    def make_model(self):
        return ContactModel(self.cfg.width, self.cfg.depth)

    def raw_residuals(self, model):
        value, (dx, dy) = model(self.interior, 1)
        a, nu = self.aspect, self.poisson
        sxx, syy, sxy = value[:, 2:3], value[:, 3:4], value[:, 4:5]
        eq = torch.cat((dx[:, 2:3]+a*dy[:, 4:5], dx[:, 4:5]+a*dy[:, 3:4]), 1)
        cons = torch.cat((sxx-(dx[:, :1]+nu*a*dy[:, 1:2])/(1-nu**2),
                          syy-(nu*dx[:, :1]+a*dy[:, 1:2])/(1-nu**2),
                          sxy-(a*dy[:, :1]+dx[:, 1:2])/(2*(1+nu))), 1)
        edge = model(self.edge_points)
        n = self.cfg.n_edge
        top, sides = edge[:n], edge[n:]
        traction = torch.cat((top[:, 4:5].flatten(),
                              (top[:, 3:4]+model.pressure(self.top[:, :1])).flatten(),
                              sides[:, 2].flatten(), sides[:, 4].flatten()))
        obs = model(self.sensors)[:, :2]-self.observed
        return [eq, cons, traction, obs]

    @torch.no_grad()
    def evaluate(self, model):
        pred = model(self.grid)[:, :2]
        error = pred-self.truth_displacement
        pressure = model.pressure(self.pressure_x)
        force = torch.trapezoid(pressure.flatten(), self.pressure_x.flatten()).item()*self.force_scale_n
        p_error = pressure-self.pressure_truth
        metrics = {'relative_l2': (error.norm()/self.truth_displacement.norm()).item(),
                   'pressure_relative_l2': (p_error.norm()/self.pressure_truth.norm()).item(),
                   'force_n': force, 'force_true_n': self.force_truth,
                   'force_relative_error': abs(force-self.force_truth)/self.force_truth,
                   'within_tolerance_pct': (error.norm(dim=1)*self.displacement_mm <= 0.05).double().mean().item()*100,
                   'max_abs_error': error.abs().max().item()*self.displacement_mm}
        fields = {'ux_mm': (pred[:, 0].reshape(self.cfg.grid, self.cfg.grid)*self.displacement_mm).tolist(),
                  'uy_mm': (pred[:, 1].reshape(self.cfg.grid, self.cfg.grid)*self.displacement_mm).tolist(),
                  'pressure_kpa': (pressure.flatten()*self.pressure_kpa).tolist()}
        return metrics, fields

    def metadata(self):
        g = self.cfg.grid
        truth = {'ux_mm': (self.truth_displacement[:, 0].reshape(g, g)*self.displacement_mm).tolist(),
                 'uy_mm': (self.truth_displacement[:, 1].reshape(g, g)*self.displacement_mm).tolist()}
        return {'id': self.id, 'title': self.title, 'axis': self.axis.tolist(), 'field_names': self.field_names,
                'truth': truth, 'cut_x': self.pressure_x.flatten().tolist(),
                'cut_truth': (self.pressure_truth.flatten()*self.pressure_kpa).tolist(),
                'cut_label': '顶部接触压力 / kPa', 'force_true_n': self.force_truth,
                'sensors': self.sensors.tolist(), 'loss_names': self.loss_names, 'weights': self.weights,
                'equation': '∇·σ = 0; σ = C:ε(u); 顶部 σn = (0,−p); 底部固定；侧面无牵引',
                'reference': '独立 P2 有限元参考解；仅 48 个内部位移观测参与训练。已知接触区 x∈[0.2,0.8]。',
                'primary_label': '位移场相对 L2 误差', 'reference_validation': self.reference_validation}
