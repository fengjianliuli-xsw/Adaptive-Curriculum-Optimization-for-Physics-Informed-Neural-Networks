"""All new outputs default to F:/RA. Example: python train.py --out runs/baseline."""
import argparse
import copy
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from physics import Config, HeatProblem, ContactProblem, fingerprint, flat_parameters, assign, loss_gradients
from optimizers import SPECS, SOAP, MultiAdam, LM, pcgrad_direction, config_direction
from reference import generate_reference


def lbfgs(model):
    return torch.optim.LBFGS(model.parameters(), lr=1, max_iter=1, max_eval=12,
                            history_size=50, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-10, tolerance_change=1e-12)


def build_optimizer(name, model, problem):
    params, rate = model.parameters(), SPECS[name]['lr']
    if name in ('adam', 'adam_lbfgs'):
        return torch.optim.Adam(params, lr=rate)
    if name == 'momentum':
        return torch.optim.SGD(params, lr=rate, momentum=0.9)
    if name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=rate, alpha=0.99)
    if name == 'lbfgs':
        return lbfgs(model)
    if name == 'lm':
        return LM(model, problem)
    if name == 'multiadam':
        return MultiAdam(model, rate)
    if name == 'soap':
        return SOAP(model, rate)
    return None


def run_one(name, problem, initial_state, output):
    cfg = problem.cfg
    model = problem.make_model()
    model.load_state_dict(copy.deepcopy(initial_state))
    params = list(model.parameters())
    opt = build_optimizer(name, model, problem)
    problem.n_eval = 0
    rng = torch.Generator().manual_seed(cfg.seed+333)
    history, snapshots = [], []
    train_seconds, gradient_evals = 0., 0
    phase = 'Adam' if name == 'adam_lbfgs' else SPECS[name]['label']
    status = 'budget_completed'
    started = time.perf_counter()

    def closure():
        nonlocal gradient_evals
        model.zero_grad(set_to_none=True)
        loss = problem.losses(model).sum()
        loss.backward()
        gradient_evals += 1
        return loss

    def record(step):
        count = problem.n_eval
        losses = problem.losses(model)
        gradients = loss_gradients(losses, params).detach()
        norms = gradients.norm(dim=1)
        cosine = (gradients@gradients.T)/(norms[:, None]*norms[None, :]).clamp_min(1e-30)
        offdiag = cosine[torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)]
        metrics, fields = problem.evaluate(model)
        problem.n_eval = count
        row = {'step': step, 'train_seconds': train_seconds, 'loss_total': losses.sum().item(),
               **{f'loss_{key}': value.item() for key, value in zip(problem.loss_names, losses)},
               **metrics, 'residual_evals': count, 'gradient_evals': gradient_evals,
               'minimum_gradient_cosine': offdiag.min().item(), 'phase': phase,
               'lm_damping': opt.mu if name == 'lm' else None}
        if not all(np.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
            raise FloatingPointError(f'{problem.id}/{name}: non-finite metrics')
        history.append(row)
        snapshots.append({'step': step, 'train_seconds': train_seconds, 'fields': fields})

    record(0)
    initial_hash = fingerprint(flat_parameters(model))
    print(f'{problem.id}/{name:12s} start L={history[-1]["loss_total"]:.3e}', flush=True)
    for step in range(1, cfg.steps+1):
        before = flat_parameters(model)
        tick = time.perf_counter()
        if name == 'adam_lbfgs' and step == cfg.steps//2+1:
            opt, phase = lbfgs(model), 'L-BFGS'
        if isinstance(opt, torch.optim.LBFGS):
            opt.step(closure)
        elif name == 'lm':
            opt.step()
            gradient_evals += 1
        elif name in ('multiadam', 'pcgrad', 'config'):
            losses = problem.losses(model)
            if name == 'multiadam':
                groups = torch.stack((losses[:2].sum(), losses[2:].sum())) if problem.id == 'contact2d' else losses
            else:
                groups = losses
            gradients = loss_gradients(groups, params).detach()
            gradient_evals += len(groups)
            if name == 'multiadam':
                opt.step(gradients)
            else:
                direction = pcgrad_direction(gradients, rng) if name == 'pcgrad' else config_direction(gradients)
                assign(model, before-SPECS[name]['lr']*direction)
        else:
            closure()
            opt.step()
        train_seconds += time.perf_counter()-tick
        if not torch.isfinite(flat_parameters(model)).all():
            assign(model, before)
            status = 'nonfinite_update_reverted'
            if history[-1]['step'] != step-1:
                record(step-1)
            break
        if step % cfg.record_every == 0 or step == cfg.steps or (name == 'lm' and opt.done):
            record(step)
        if step % max(1, cfg.steps//4) == 0:
            h = history[-1]
            extra = f' force={h["force_n"]:.3f}N pErr={h["pressure_relative_l2"]:.2e}' if problem.id == 'contact2d' else ''
            print(f'{problem.id}/{name:12s} {step}/{cfg.steps} L={h["loss_total"]:.2e} err={h["relative_l2"]:.2e} train={train_seconds:.1f}s{extra}', flush=True)
        if name == 'lm' and opt.done:
            status = 'gradient_tolerance_reached'
            break
    result = {'id': name, **SPECS[name], 'initial_hash': initial_hash, 'sampling_hash': problem.sample_hash,
              'status': status, 'history': history, 'snapshots': snapshots,
              'total_wall_seconds': time.perf_counter()-started,
              'lm_rejected_trials': opt.rejections if name == 'lm' else None}
    torch.save(model.state_dict(), output/f'{name}_final.pt')
    with (output/f'{name}_history.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    (output/f'{name}.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'{problem.id}/{name:12s} DONE err={history[-1]["relative_l2"]:.3e} train={train_seconds:.1f}s', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'runs'/'baseline')
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--record-every', type=int, default=20)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise', type=float, default=0.)
    parser.add_argument('--n-contact', type=int, default=64)
    parser.add_argument('--n-heat', type=int, default=128)
    parser.add_argument('--optimizers', nargs='+', choices=list(SPECS), default=list(SPECS))
    parser.add_argument('--problems', nargs='+', choices=['heat2d', 'contact2d'], default=['heat2d', 'contact2d'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-gif', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.record_every, args.width, args.n_contact, args.n_heat) < 1 or args.noise < 0:
        parser.error('Counts must be positive and noise must be nonnegative')
    out = args.out.resolve()
    root = Path('F:/RA').resolve()
    if not out.is_relative_to(root):
        parser.error('User preference: all new experimental outputs must be under F:/RA')
    cfg = Config(seed=args.seed, steps=args.steps, record_every=args.record_every, width=args.width,
                 noise=args.noise, n_contact=args.n_contact, n_heat=args.n_heat)
    config_file = out/'config.json'
    if out.exists() and any(out.iterdir()):
        if not args.resume:
            parser.error('Nonempty output directory; choose another --out or pass --resume')
        if not config_file.exists() or json.loads(config_file.read_text()) != cfg.to_dict():
            parser.error('Resume configuration does not match saved configuration')
    out.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(cfg.to_dict(), indent=2), encoding='utf-8')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    reference = None
    if 'contact2d' in args.problems:
        ref_path = out/'contact_reference.json'
        reference = json.loads(ref_path.read_text()) if ref_path.exists() else generate_reference(cfg, ref_path)
        print('FEM validation:', reference['validation'], flush=True)
    experiment = {'config': cfg.to_dict(), 'environment': {'python': platform.python_version(),
                  'torch': torch.__version__, 'numpy': np.__version__, 'platform': platform.platform(),
                  'device': 'cpu', 'dtype': 'float64', 'environment_path': str(Path(__file__).parent.parent/'.venv')},
                  'focus': ['adam', 'adam_lbfgs', 'soap'], 'problems': []}
    for problem_id in dict.fromkeys(args.problems):
        problem = HeatProblem(cfg) if problem_id == 'heat2d' else ContactProblem(cfg, reference)
        directory = out/problem_id
        directory.mkdir(exist_ok=True)
        torch.manual_seed(cfg.seed)
        model = problem.make_model()
        initial_state = copy.deepcopy(model.state_dict())
        initial_hash = fingerprint(flat_parameters(model))
        initial_path = directory/'initial_state.pt'
        if initial_path.exists():
            saved = torch.load(initial_path, weights_only=True)
            model.load_state_dict(saved)
            if fingerprint(flat_parameters(model)) != initial_hash:
                raise ValueError('Saved initial state mismatches deterministic initialization')
        else:
            torch.save(initial_state, initial_path)
        np.savez(directory/'training_points.npz', **{f'array_{i}': t.numpy() for i, t in enumerate(problem.training_tensors)},
                 evaluation_grid=problem.grid.numpy())
        item = {**problem.metadata(), 'initial_hash': initial_hash, 'sampling_hash': problem.sample_hash,
                'parameter_count': sum(p.numel() for p in model.parameters()), 'runs': []}
        for name in dict.fromkeys(args.optimizers):
            path = directory/f'{name}.json'
            if args.resume and path.exists():
                result = json.loads(path.read_text(encoding='utf-8'))
                if result['initial_hash'] != initial_hash or result['sampling_hash'] != problem.sample_hash:
                    raise ValueError('Cached run does not share initialization or sampling')
            else:
                result = run_one(name, problem, initial_state, directory)
            item['runs'].append(result)
        assert all(r['initial_hash'] == initial_hash for r in item['runs'])
        assert all(r['snapshots'][0]['fields'] == item['runs'][0]['snapshots'][0]['fields'] for r in item['runs'])
        experiment['problems'].append(item)
        (out/'results.json').write_text(json.dumps(experiment, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    from report import build_report
    build_report(experiment, out, gif=not args.no_gif)
    print('OPEN', out/'dashboard.html', flush=True)


if __name__ == '__main__':
    main()
