"""Small, explicit optimizers. Custom methods are documented educational implementations."""
import torch
from torch.func import jacrev
from physics import assign, flat_parameters


SPECS = {
    'dcgd': {'label': 'DCGD-Center + GD', 'color': '#a73a9b', 'lr': 0.001, 'family': '双锥方向协调', 'idea': '将总梯度投影到两组归一化梯度的角平分线；直接 GD，之后不再经过 Adam。'},
    'adam': {'label': 'Adam', 'color': '#2766d5', 'lr': 0.001, 'family': '一阶自适应', 'idea': '记住历史方向，并按每个参数的梯度尺度调整步长。'},
    'momentum': {'label': 'SGD + Momentum', 'color': '#9c6644', 'lr': 0.01, 'family': '一阶动量', 'idea': '利用惯性积累一致方向，减轻来回摆动。'},
    'rmsprop': {'label': 'RMSProp', 'color': '#cf6b24', 'lr': 0.001, 'family': '一阶自适应', 'idea': '梯度经常较大的参数采用较小有效步长。'},
    'lbfgs': {'label': 'L-BFGS', 'color': '#2e8b57', 'lr': 1.0, 'family': '拟牛顿', 'idea': '从历史参数与梯度变化估计曲率，配合强 Wolfe 线搜索。'},
    'adam_lbfgs': {'label': 'Adam → L-BFGS', 'color': '#087f8c', 'lr': 0.001, 'family': '分阶段策略', 'idea': '前半程 Adam，后半程切换 L-BFGS；总外层预算相同。'},
    'lm': {'label': 'Gauss–Newton / LM', 'color': '#873cc7', 'lr': None, 'family': '残差二阶结构', 'idea': '线性化残差，解阻尼 Gauss–Newton 方程；实际下降才接受。'},
    'multiadam': {'label': 'MultiAdam', 'color': '#d14c80', 'lr': 0.001, 'family': '多目标尺度平衡', 'idea': 'PDE 与初边值分两组，各自维护 Adam 统计，再平均更新。'},
    'pcgrad': {'label': 'PCGrad + GD', 'color': '#657721', 'lr': 0.02, 'family': '多目标方向协调', 'idea': '将冲突梯度的反向分量投影掉，再用普通梯度步更新。'},
    'config': {'label': 'ConFIG + GD', 'color': '#c04638', 'lr': 0.02, 'family': '多目标方向协调', 'idea': '用归一化梯度的伪逆求公共方向，并按投影确定幅值。'},
    'soap': {'label': 'SOAP', 'color': '#4b5563', 'lr': 0.003, 'family': '结构化预条件', 'idea': '按层的梯度相关统计旋转坐标，在旋转坐标中做 Adam。'},
}


def dcgd_direction(grads):
    """DCGD Center, paper eq. (4.3). Two groups, direct GD application.

    Degenerate zero group -> remaining gradient; exactly opposite -> zero step.
    """
    if grads.shape[0] != 2:
        raise ValueError('DCGD Center here requires exactly two physical groups')
    norms=grads.norm(dim=1)
    if (norms < 1e-14).any():
        return grads[norms >= 1e-14].sum(0)
    center=(grads/norms[:,None]).sum(0)
    norm2=center.square().sum()
    if norm2 < 1e-20:
        return torch.zeros_like(center)
    return ((grads.sum(0)@center)/norm2).clamp_min(0)*center


def pcgrad_direction(grads, generator):
    projected = grads.clone()
    for i in range(len(grads)):
        for j in torch.randperm(len(grads), generator=generator).tolist():
            dot = projected[i] @ grads[j]
            if dot < 0:
                projected[i] -= dot / grads[j].square().sum().clamp_min(1e-30) * grads[j]
    return projected.sum(0)


def config_direction(grads):
    norms = grads.norm(dim=1)
    active = norms > 1e-14
    if not active.any():
        return grads.sum(0)*0
    a = grads[active]/norms[active, None]
    # Solve only a K x K Gram matrix, rather than an enormous parameter-space SVD.
    direction = a.T @ (torch.linalg.pinv(a @ a.T, rtol=1e-10) @ torch.ones(len(a), dtype=a.dtype))
    unit = direction/direction.norm().clamp_min(1e-30)
    return (grads @ unit).sum().clamp_min(0)*unit


class MultiAdam:
    def __init__(self, model, lr=0.001):
        self.model, self.lr = model, lr
        self.m = self.v = None
        self.t = 0

    def step(self, grads):
        if self.m is None:
            self.m, self.v = torch.zeros_like(grads), torch.zeros_like(grads)
        self.t += 1
        beta = 0.99  # paper-recommended equal betas
        self.m = beta*self.m+(1-beta)*grads
        self.v = beta*self.v+(1-beta)*grads.square()
        m, v = self.m/(1-beta**self.t), self.v/(1-beta**self.t)
        direction = (m/(v.sqrt()+1e-8)).mean(0)
        assign(self.model, flat_parameters(self.model)-self.lr*direction)


class SOAP:
    """Dense per-layer SOAP core; float64, no weight decay/grafting, basis refresh every 10 steps.

    Momentum lives in original coordinates. On rotation changes, diagonal second
    moments are transported using squared basis-overlap coefficients.
    """
    def __init__(self, model, lr=0.003, frequency=10):
        self.model, self.lr, self.frequency = model, lr, frequency
        self.states = {}
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for name, p in self.model.named_parameters():
            g = p.grad
            s = self.states.setdefault(name, {'m': torch.zeros_like(p), 'v': torch.zeros_like(p)})
            s['m'].mul_(0.95).add_(g, alpha=0.05)
            if p.ndim != 2:
                s['v'].mul_(0.95).addcmul_(g, g, value=0.05)
                p.addcdiv_(s['m']/(1-0.95**self.t), (s['v']/(1-0.95**self.t)).sqrt()+1e-8, value=-self.lr)
                continue
            if 'left' not in s:
                s.update(left=torch.zeros(p.shape[0], p.shape[0]), right=torch.zeros(p.shape[1], p.shape[1]),
                         ql=torch.eye(p.shape[0]), qr=torch.eye(p.shape[1]))
            s['left'].mul_(0.95).add_(g@g.T, alpha=0.05)
            s['right'].mul_(0.95).add_(g.T@g, alpha=0.05)
            if self.t == 1 or self.t % self.frequency == 0:
                ql = torch.linalg.eigh(s['left'])[1]
                qr = torch.linalg.eigh(s['right'])[1]
                overlap_l, overlap_r = ql.T@s['ql'], s['qr'].T@qr
                s['v'] = (overlap_l.square() @ s['v'] @ overlap_r.square()).clamp_min(0)
                s['ql'], s['qr'] = ql, qr
            ql, qr = s['ql'], s['qr']
            rotated = ql.T@g@qr
            s['v'].mul_(0.95).addcmul_(rotated, rotated, value=0.05)
            rotated_m = ql.T@s['m']@qr/(1-0.95**self.t)
            update = rotated_m/((s['v']/(1-0.95**self.t)).sqrt()+1e-8)
            p.add_(ql@update@qr.T, alpha=-self.lr)


class LM:
    """Exact dense residual Jacobian; dual solve when residual count < parameter count."""
    def __init__(self, model, problem):
        self.model, self.problem = model, problem
        self.residual = problem.residual_function(model)
        self.mu = 1e-2
        self.last_accepted = False
        self.done = False
        self.rejections = 0

    def step(self):
        if self.done:
            return
        flat = flat_parameters(self.model)
        r = self.residual(flat).detach(); self.problem.n_eval += 1
        j = jacrev(self.residual, chunk_size=64)(flat).detach(); self.problem.n_eval += 1
        g = j.T@r
        if g.abs().max() < 1e-10:
            self.done = True
            return
        old = r.square().sum()
        # Woodbury identity: (J^T J + mu I)^-1 J^T r = J^T (JJ^T + mu I)^-1 r.
        dual = j.shape[0] < j.shape[1]
        gram = j@j.T if dual else j.T@j
        eye = torch.eye(gram.shape[0], dtype=gram.dtype)
        self.last_accepted = False
        for _ in range(10):
            rhs = r if dual else g
            solution = torch.linalg.solve(gram+self.mu*eye, rhs)
            delta = -(j.T@solution if dual else solution)
            with torch.no_grad():
                candidate = self.residual(flat+delta).square().sum()
            self.problem.n_eval += 1
            predicted = old-(r+j@delta).square().sum()
            ratio = ((old-candidate)/predicted.clamp_min(1e-30)).item()
            if torch.isfinite(candidate) and candidate < old and ratio > 1e-4:
                assign(self.model, flat+delta)
                self.mu = max(1e-12, self.mu*(0.3 if ratio > 0.75 else 1.0))
                self.last_accepted = True
                break
            self.mu = min(self.mu*10, 1e16)
            self.rejections += 1
