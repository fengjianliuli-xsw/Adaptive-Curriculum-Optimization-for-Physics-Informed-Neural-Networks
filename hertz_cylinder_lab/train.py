"""Common-initialization comparison for KKT/FB Hertz contact; all files under F:/RA."""
import argparse
import copy
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from physics import Config, Problem, flat_parameters, fingerprint, assign, loss_gradients
from reference import generate
from optimizers import SPECS, SOAP, LM, MultiAdam, pcgrad_direction, config_direction, dcgd_direction


def lbfgs(model):
    return torch.optim.LBFGS(model.parameters(), lr=1., max_iter=1, max_eval=12, history_size=50,
                            line_search_fn='strong_wolfe', tolerance_grad=1e-10, tolerance_change=1e-12)


def run_one(name, problem, state, output):
    cfg = problem.cfg
    model = problem.make_model(); model.load_state_dict(copy.deepcopy(state))
    params = list(model.parameters())
    rate = SPECS[name]['lr']
    if name in ('adam', 'adam_lbfgs'):
        opt = torch.optim.Adam(params, lr=rate)
    elif name == 'momentum':
        opt = torch.optim.SGD(params, lr=rate, momentum=.9)
    elif name == 'rmsprop':
        opt = torch.optim.RMSprop(params, lr=rate)
    elif name == 'lbfgs':
        opt = lbfgs(model)
    elif name == 'lm':
        opt = LM(model, problem)
    elif name == 'soap':
        opt = SOAP(model, rate)
    elif name == 'multiadam':
        opt = MultiAdam(model, rate)
    else:
        opt = None
    initial_hash = fingerprint(flat_parameters(model))
    history, snapshots = [], []
    elapsed, gradients_count = 0., 0
    problem.n_eval = 0
    phase, status = 'Adam' if name == 'adam_lbfgs' else SPECS[name]['label'], 'budget_completed'
    rng = torch.Generator().manual_seed(cfg.seed+333)
    start = time.perf_counter()

    def closure():
        nonlocal gradients_count
        model.zero_grad(set_to_none=True)
        loss = problem.losses(model).sum()
        loss.backward(); gradients_count += 1
        return loss

    def record(step):
        calls = problem.n_eval
        with torch.no_grad():
            losses = problem.losses(model)
            metrics, fields = problem.evaluate(model)
        problem.n_eval = calls
        row = {'step': step, 'train_seconds': elapsed, 'phase': phase,
               'loss_total': losses.sum().item(), **{f'loss_{k}': v.item() for k, v in zip(problem.loss_names, losses)},
               **metrics, 'residual_evals': calls, 'gradient_evals': gradients_count}
        if not all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float))):
            raise FloatingPointError(f'{name} has non-finite diagnostics')
        history.append(row); snapshots.append({'step': step, 'train_seconds': elapsed, 'fields': fields})

    record(0)
    last_finite_model = flat_parameters(model)
    print(f'{name:12} start L={history[-1]["loss_total"]:.3e}', flush=True)
    for step in range(1, cfg.steps+1):
        before = flat_parameters(model)
        tick = time.perf_counter()
        if name == 'adam_lbfgs' and step == cfg.steps//2+1:
            opt, phase = lbfgs(model), 'L-BFGS'
        if isinstance(opt, torch.optim.LBFGS):
            opt.step(closure)
        elif name == 'lm':
            opt.step(); gradients_count += 1
        elif name in ('multiadam', 'pcgrad', 'config', 'dcgd'):
            losses = problem.losses(model)
            groups = torch.stack((losses[0]+losses[-1], losses[1:-1].sum())) if name in ('multiadam','config','dcgd') else losses
            grads = loss_gradients(groups, params).detach(); gradients_count += len(groups)
            if name == 'multiadam':
                opt.step(grads)
            else:
                direction = pcgrad_direction(grads, rng) if name == 'pcgrad' else dcgd_direction(grads) if name == 'dcgd' else config_direction(grads)
                assign(model, before-rate*direction)
        else:
            closure(); opt.step()
        elapsed += time.perf_counter()-tick
        with torch.no_grad():
            update_is_finite = bool(torch.isfinite(flat_parameters(model)).all())
            if update_is_finite:
                update_is_finite = bool(torch.isfinite(problem.losses(model)).all())
        if not update_is_finite:
            assign(model, last_finite_model); status = 'nonfinite_update_reverted'
            break
        if step % cfg.record_every == 0 or step == cfg.steps or (name == 'lm' and opt.done):
            try:
                record(step)
                last_finite_model = flat_parameters(model)
            except FloatingPointError:
                assign(model,last_finite_model); status='nonfinite_diagnostics_reverted'
                break
        if step % max(1, cfg.steps//4) == 0:
            h = history[-1]
            print(f'{name:12} {step}/{cfg.steps} L={h["loss_total"]:.2e} uErr={h["relative_l2"]:.2e} pErr={h["pressure_relative_l2"]:.2e} F={h["reaction_n"]:.2f}N penetration={h["max_penetration_um"]:.3f}um train={elapsed:.1f}s', flush=True)
        if name == 'lm' and opt.done:
            status = 'gradient_tolerance_reached'; break
        if cfg.time_budget_seconds > 0 and elapsed >= cfg.time_budget_seconds:
            status = 'training_time_budget'
            if history[-1]['step'] != step:
                try: record(step)
                except FloatingPointError:
                    assign(model,last_finite_model);status='nonfinite_diagnostics_reverted'
            break
    result = {'id': name, **SPECS[name], 'initial_hash': initial_hash, 'sampling_hash': problem.sample_hash,
              'history': history, 'snapshots': snapshots, 'status': status,
              'total_wall_seconds': time.perf_counter()-start}
    torch.save(model.state_dict(), output/f'{name}_final.pt')
    (output/f'{name}.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    with (output/f'{name}_history.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    print(f'{name:12} DONE pErr={history[-1]["pressure_relative_l2"]:.3e}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'runs'/'baseline')
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--record-every', type=int, default=20)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--n-interior', type=int, default=96)
    parser.add_argument('--n-arc', type=int, default=96)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimizers', nargs='+', choices=list(SPECS), default=list(SPECS))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-gif', action='store_true')
    parser.add_argument('--time-budget', type=float, default=120.)
    parser.add_argument('--comparison-lr', type=float, default=None, help='Optional common direct-GD rate for DCGD and ConFIG')
    parser.add_argument('--initial-vertical-bias', type=float, default=0.)
    args = parser.parse_args()
    if args.comparison_lr is not None:
        if args.comparison_lr <= 0: parser.error('comparison-lr must be positive')
        for method in ('dcgd','config'):
            SPECS[method]['lr'] = args.comparison_lr
    if min(args.steps, args.record_every, args.width, args.n_interior, args.n_arc) < 2:
        parser.error('Counts must be >=2')
    cfg = Config(seed=args.seed, steps=args.steps, record_every=args.record_every, width=args.width,
                 n_interior=args.n_interior, n_arc=args.n_arc, time_budget_seconds=args.time_budget,
                 initial_vertical_bias=args.initial_vertical_bias)
    out = args.out.resolve()
    if not out.is_relative_to(Path('F:/RA').resolve()):
        parser.error('New experiments must live under F:/RA')
    if out.exists() and any(out.iterdir()):
        if not args.resume:
            parser.error('Nonempty output; select another directory or use --resume')
        if Config(**json.loads((out/'run_config.json').read_text())).to_dict() != cfg.to_dict():
            parser.error('Resume config mismatch')
    out.mkdir(parents=True, exist_ok=True)
    (out/'run_config.json').write_text(json.dumps(cfg.to_dict(), indent=2), encoding='utf-8')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    ref_path = out/'fem_reference.json'
    reference = json.loads(ref_path.read_text()) if ref_path.exists() else generate(cfg, ref_path)
    print('REFERENCE', reference['validation'], flush=True)
    problem = Problem(cfg, reference)
    torch.manual_seed(cfg.seed)
    model = problem.make_model(); state = copy.deepcopy(model.state_dict())
    initial_hash = fingerprint(flat_parameters(model))
    torch.save(state, out/'initial_state.pt')
    np.savez(out/'training_points.npz', interior=problem.interior.numpy(), arc=problem.arc.numpy(),
             theta=problem.theta.numpy(), theta_quadrature=problem.theta_q.numpy(), weights_quadrature=problem.weight_q.numpy())
    data = {'config': cfg.to_dict(), **problem.metadata(), 'initial_hash': initial_hash,
            'sampling_hash': problem.sample_hash, 'parameter_count': sum(p.numel() for p in model.parameters()),
            'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                            'numpy': np.__version__, 'device': 'CPU', 'dtype': 'float64'}, 'runs': []}
    for name in dict.fromkeys(args.optimizers):
        path = out/f'{name}.json'
        if args.resume and path.exists():
            result = json.loads(path.read_text(encoding='utf-8'))
            if result['initial_hash'] != initial_hash or result['sampling_hash'] != problem.sample_hash:
                raise ValueError('Cached run initial state or sampling mismatch')
            if result['lr'] != SPECS[name]['lr']:
                raise ValueError('Cached run learning rate mismatch')
        else:
            result = run_one(name, problem, state, out)
        data['runs'].append(result)
        (out/'results.json').write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    assert all(r['initial_hash'] == initial_hash for r in data['runs'])
    assert all(r['snapshots'][0]['fields'] == data['runs'][0]['snapshots'][0]['fields'] for r in data['runs'])
    from report import build_report
    build_report(data, out, gif=not args.no_gif)
    print('OPEN', out/'dashboard.html', flush=True)


if __name__ == '__main__':
    main()
