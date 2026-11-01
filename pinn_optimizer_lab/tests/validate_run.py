"""Recompute stored losses and predictions from checkpoints, not just JSON consistency."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from physics import Config, PINN, Problem, fingerprint, flat_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    data = json.loads((args.run/'results.json').read_text(encoding='utf-8'))
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    cfg = Config(**data['config'])
    problem, model = Problem(cfg), PINN(cfg.width, cfg.depth)
    model.load_state_dict(torch.load(args.run/'initial_state.pt', weights_only=True))
    assert fingerprint(flat_parameters(model)) == data['initial_hash']
    init_losses = problem.losses(model).detach().numpy()
    metrics, init_curves, _ = problem.evaluate(model)
    for run in data['optimizers']:
        assert run['initial_hash'] == data['initial_hash']
        assert run['sampling_hash'] == fingerprint(problem.points)
        assert run['snapshots'][0]['curves'] == init_curves
        assert len(run['history']) == len(run['snapshots'])
        np.testing.assert_allclose([run['history'][0][k] for k in ['loss_pde', 'loss_initial', 'loss_boundary']], init_losses, rtol=1e-12)
        model.load_state_dict(torch.load(args.run/f'{run["id"]}_final.pt', weights_only=True))
        losses = problem.losses(model).detach().numpy()
        np.testing.assert_allclose([run['history'][-1][k] for k in ['loss_pde', 'loss_initial', 'loss_boundary']], losses, rtol=1e-12, atol=1e-15)
        final_metrics, curves, grid = problem.evaluate(model)
        assert curves == run['snapshots'][-1]['curves']
        assert grid == run['final_grid']
        for key, value in final_metrics.items():
            np.testing.assert_allclose(run['history'][-1][key], value, rtol=1e-12, atol=1e-15)
        assert all(b['step'] > a['step'] for a, b in zip(run['history'], run['history'][1:]))
        assert all(b['train_seconds'] >= a['train_seconds'] for a, b in zip(run['history'], run['history'][1:]))
    print(f'Validated {len(data["optimizers"])} checkpoints, losses, metrics, snapshots and shared initialization.')


if __name__ == '__main__':
    main()
