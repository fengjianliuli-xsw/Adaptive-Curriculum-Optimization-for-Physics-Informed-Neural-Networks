"""Run: python train.py --steps 1200 --out runs/demo"""
import argparse
import copy
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from physics import Config, PINN, Problem, assign, fingerprint, flat_parameters, loss_gradients
from optimizers import SPECS, LM, MultiAdam, SOAP, config_direction, pcgrad_direction


def make_lbfgs(model):
    return torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, max_eval=12,
                            history_size=50, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-10, tolerance_change=1e-12)


def run_one(name, cfg, initial_state, expected_hash, out):
    model = PINN(cfg.width, cfg.depth)
    model.load_state_dict(copy.deepcopy(initial_state))
    initial_hash = fingerprint(flat_parameters(model))
    assert initial_hash == expected_hash
    problem = Problem(cfg)
    params = list(model.parameters())
    spec = SPECS[name]
    lr = spec['lr']
    if name in ('adam', 'adam_lbfgs'):
        opt = torch.optim.Adam(params, lr=lr)
    elif name == 'momentum':
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif name == 'rmsprop':
        opt = torch.optim.RMSprop(params, lr=lr, alpha=0.99)
    elif name == 'lbfgs':
        opt = make_lbfgs(model)
    elif name == 'lm':
        opt = LM(model, problem)
    elif name == 'multiadam':
        opt = MultiAdam(model, lr)
    elif name == 'soap':
        opt = SOAP(model, lr)
    else:
        opt = None
    rng = torch.Generator().manual_seed(cfg.seed+333)
    rows, snapshots = [], []
    train_seconds = 0.0
    status, stop_step = 'completed', cfg.steps
    phase = 'Adam' if name == 'adam_lbfgs' else spec['label']
    gradient_evals = 0
    started = time.perf_counter()

    def closure():
        nonlocal gradient_evals
        model.zero_grad(set_to_none=True)
        loss = problem.losses(model).sum()
        loss.backward()
        gradient_evals += 1
        return loss

    def record(step):
        nonlocal final_grid
        # Diagnostics are outside training timing and never used for updates.
        previous_count = problem.n_eval
        losses = problem.losses(model)
        grads = loss_gradients(losses, params).detach()
        norms = grads.norm(dim=1)
        cosine = (grads@grads.T)/(norms[:, None]*norms[None, :]).clamp_min(1e-30)
        metrics, curves, final_grid = problem.evaluate(model)
        problem.n_eval = previous_count
        row = {'step': step, 'train_seconds': train_seconds,
               'loss_total': losses.sum().item(), 'loss_pde': losses[0].item(),
               'loss_initial': losses[1].item(), 'loss_boundary': losses[2].item(),
               **metrics, 'residual_evals': problem.n_eval, 'gradient_evals': gradient_evals,
               'grad_pde_initial_cos': cosine[0, 1].item(),
               'grad_pde_boundary_cos': cosine[0, 2].item(),
               'grad_initial_boundary_cos': cosine[1, 2].item(), 'phase': phase,
               'lm_damping': opt.mu if name == 'lm' else None}
        if not all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float))):
            raise FloatingPointError(f'{name}: non-finite metrics at step {step}')
        rows.append(row)
        snapshots.append({'step': step, 'train_seconds': train_seconds, 'curves': curves})

    final_grid = None
    record(0)
    print(f'{name:13s} start | L={rows[-1]["loss_total"]:.3e}', flush=True)
    for step in range(1, cfg.steps+1):
        before = flat_parameters(model)
        begin = time.perf_counter()
        if name == 'adam_lbfgs' and step == cfg.steps//2+1:
            opt = make_lbfgs(model)
            phase = 'L-BFGS'
        if name in ('lbfgs', 'adam_lbfgs') and isinstance(opt, torch.optim.LBFGS):
            opt.step(closure)
        elif name == 'lm':
            opt.step()
            gradient_evals += 1
        elif name in ('multiadam', 'pcgrad', 'config'):
            losses = problem.losses(model)
            groups = torch.stack((losses[0], losses[1]+losses[2])) if name == 'multiadam' else losses
            grads = loss_gradients(groups, params).detach()
            gradient_evals += len(groups)
            if name == 'multiadam':
                opt.step(grads)
            else:
                direction = pcgrad_direction(grads, rng) if name == 'pcgrad' else config_direction(grads)
                assign(model, before-lr*direction)
        else:
            closure()
            opt.step()
        train_seconds += time.perf_counter()-begin
        if not torch.isfinite(flat_parameters(model)).all():
            assign(model, before)
            status, stop_step = 'nonfinite_update_reverted', step-1
            if rows[-1]['step'] != stop_step:
                record(stop_step)
            break
        if step % cfg.record_every == 0 or step == cfg.steps or (name == 'lm' and opt.done):
            record(step)
        if step % max(1, cfg.steps//4) == 0:
            print(f'{name:13s} {step:4d}/{cfg.steps} | L={rows[-1]["loss_total"]:.3e} | relL2={rows[-1]["relative_l2"]:.3e} | train={train_seconds:.1f}s', flush=True)
        if name == 'lm' and opt.done:
            status, stop_step = 'gradient_tolerance_reached', step
            break
    wall = time.perf_counter()-started
    result = {'id': name, **spec, 'initial_hash': initial_hash,
              'sampling_hash': fingerprint(problem.points), 'status': status,
              'stop_step': stop_step, 'total_wall_seconds': wall,
              'training_seconds': train_seconds, 'history': rows,
              'snapshots': snapshots, 'final_grid': final_grid,
              'lm_rejected_trials': opt.rejections if name == 'lm' else None}
    torch.save(model.state_dict(), out/f'{name}_final.pt')
    with (out/f'{name}_history.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/f'{name}.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'{name:13s} done | relL2={rows[-1]["relative_l2"]:.3e} | train={train_seconds:.1f}s | status={status}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='Fair, reproducible 1D heat-equation PINN optimizer comparison')
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--record-every', type=int, default=10)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--n-interior', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--optimizers', nargs='+', choices=list(SPECS), default=list(SPECS))
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'runs'/'demo')
    parser.add_argument('--no-gif', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly allow replacing outputs in an existing run directory')
    args = parser.parse_args()
    if min(args.steps, args.record_every, args.width, args.depth, args.n_interior, args.threads) < 1:
        parser.error('steps, record interval, width, depth, point count and threads must be positive')
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        parser.error(f'{out} is not empty; choose another --out or explicitly pass --overwrite')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    cfg = Config(seed=args.seed, width=args.width, depth=args.depth, steps=args.steps,
                 record_every=args.record_every, n_interior=args.n_interior, threads=args.threads)
    model = PINN(cfg.width, cfg.depth)
    initial_state = copy.deepcopy(model.state_dict())
    initial_hash = fingerprint(flat_parameters(model))
    torch.save(initial_state, out/'initial_state.pt')
    problem = Problem(cfg)
    np.savez(out/'sampling_and_reference.npz', interior=problem.interior.numpy(),
             initial=problem.initial.numpy(), boundary=problem.boundary.numpy(),
             evaluation_points=problem.grid.numpy(), truth=problem.truth.numpy(),
             test_residual=problem.test_residual.numpy())
    manifest = {'config': cfg.to_dict(), 'initial_hash': initial_hash,
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                'numpy': np.__version__, 'platform': platform.platform(),
                                'device': 'cpu', 'dtype': 'float64'},
                'equation': 'u_t = 0.1 u_xx; x,t in [0,1]; u(0,t)=u(1,t)=0',
                'initial_condition': 'sin(pi*x) + 0.3*sin(3*pi*x)',
                'loss_weights': {'pde': 1.0, 'initial': 1.0, 'boundary': 1.0},
                'slice_times': [0, 0.25, 0.5, 0.75, 1.0], 'x': problem.eval_x.tolist(),
                't': problem.eval_t.tolist(), 'truth': problem.truth.tolist(),
                'truth_slices': problem.truth[problem.slice_indices].tolist(),
                'optimizers': []}
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    for name in dict.fromkeys(args.optimizers):
        manifest['optimizers'].append(run_one(name, cfg, initial_state, initial_hash, out))
        (out/'results.json').write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    assert len({r['sampling_hash'] for r in manifest['optimizers']}) == 1
    first_curves = [r['snapshots'][0]['curves'] for r in manifest['optimizers']]
    assert all(v == first_curves[0] for v in first_curves)
    from report import build_report
    build_report(manifest, out, gif=not args.no_gif)
    print(f'Open {out / "dashboard.html"}', flush=True)


if __name__ == '__main__':
    main()
