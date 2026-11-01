"""Recompute all stored metrics and training losses from saved model checkpoints."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from physics import Config, HeatProblem, ContactProblem, fingerprint, flat_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    data = json.loads((args.run/'results.json').read_text(encoding='utf-8'))
    ref = json.loads((args.run/'contact_reference.json').read_text(encoding='utf-8'))
    cfg = Config(**data['config'])
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    count = 0
    for entry in data['problems']:
        problem = HeatProblem(cfg) if entry['id'] == 'heat2d' else ContactProblem(cfg, ref)
        model = problem.make_model()
        directory = args.run/entry['id']
        model.load_state_dict(torch.load(directory/'initial_state.pt', weights_only=True))
        assert fingerprint(flat_parameters(model)) == entry['initial_hash']
        assert problem.sample_hash == entry['sampling_hash']
        initial_losses = problem.losses(model).detach().numpy()
        _, initial_fields = problem.evaluate(model)
        for run in entry['runs']:
            assert run['initial_hash'] == entry['initial_hash']
            assert run['sampling_hash'] == entry['sampling_hash']
            assert run['snapshots'][0]['fields'] == initial_fields
            np.testing.assert_allclose([run['history'][0]['loss_'+key] for key in problem.loss_names], initial_losses, rtol=1e-12)
            model.load_state_dict(torch.load(directory/f'{run["id"]}_final.pt', weights_only=True))
            losses = problem.losses(model).detach().numpy()
            np.testing.assert_allclose([run['history'][-1]['loss_'+key] for key in problem.loss_names], losses, rtol=1e-12, atol=1e-15)
            metrics, fields = problem.evaluate(model)
            assert fields == run['snapshots'][-1]['fields']
            for key, value in metrics.items():
                np.testing.assert_allclose(run['history'][-1][key], value, rtol=1e-12, atol=1e-15)
            assert all(b['step'] > a['step'] for a, b in zip(run['history'], run['history'][1:]))
            count += 1
    print(f'Validated {count} saved models: initialization, sampling, individual losses, fields, pressure, force, metrics.')


if __name__ == '__main__':
    main()
