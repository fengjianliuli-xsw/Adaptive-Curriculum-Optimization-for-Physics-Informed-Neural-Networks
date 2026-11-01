"""Post-training physical checks on independent points; never used to update models."""
import argparse
import hashlib
import json
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from physics import Config, HeatProblem, ContactProblem, heat_source


def audit_run(directory):
    directory = Path(directory)
    data = json.loads((directory/'results.json').read_text(encoding='utf-8'))
    reference = json.loads((directory/'contact_reference.json').read_text(encoding='utf-8'))
    cfg = Config(**data['config'])
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    independent = torch.quasirandom.SobolEngine(2, scramble=True, seed=cfg.seed+9999).draw(1024).double()
    audit = []
    with torch.no_grad():
        for entry in data['problems']:
            problem = HeatProblem(cfg) if entry['id'] == 'heat2d' else ContactProblem(cfg, reference)
            problem.interior = independent
            if entry['id'] == 'heat2d':
                problem.source = heat_source(independent)
            for run in entry['runs']:
                model = problem.make_model()
                model.load_state_dict(torch.load(directory/entry['id']/f'{run["id"]}_final.pt', weights_only=True))
                residuals = problem.raw_residuals(model)
                row = {'problem': entry['id'], 'optimizer': run['id'], 'independent_points': len(independent),
                       'independent_pde_mse': residuals[0].square().mean().item()}
                if entry['id'] == 'contact2d':
                    x = torch.linspace(0, 1, 501)
                    bottom = torch.stack((x, torch.zeros_like(x)), 1)
                    reaction = -torch.trapezoid(model(bottom)[:, 3], x).item()*problem.force_scale_n
                    pressure_force = torch.trapezoid(model.pressure(x[:, None]).flatten(), x).item()*problem.force_scale_n
                    row.update(independent_constitutive_mse=residuals[1].square().mean().item(),
                               estimated_bottom_reaction_n=reaction, estimated_pressure_force_n=pressure_force,
                               predicted_force_balance_relative=abs(reaction-pressure_force)/max(abs(pressure_force), 1e-12))
                audit.append(row)
    (directory/'independent_physics_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    lines = ['# Independent physical audit', '',
             'Post-training checks on 1024 held-out Sobol points; never used to update or choose models.', '',
             '| PDE | Optimizer | Held-out PDE MSE | Held-out constitutive MSE | Predicted pressure force (N) | Predicted bottom reaction (N) | Predicted force imbalance |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in audit:
        lines.append('| '+row['problem']+' | '+row['optimizer']+' | '+f'{row["independent_pde_mse"]:.4e}'+' | '+
                     (f'{row["independent_constitutive_mse"]:.4e}' if 'independent_constitutive_mse' in row else '—')+' | '+
                     (f'{row["estimated_pressure_force_n"]:.5f}' if 'estimated_pressure_force_n' in row else '—')+' | '+
                     (f'{row["estimated_bottom_reaction_n"]:.5f}' if 'estimated_bottom_reaction_n' in row else '—')+' | '+
                     (f'{row["predicted_force_balance_relative"]:.3%}' if 'predicted_force_balance_relative' in row else '—')+' |')
    (directory/'independent_physics_audit.md').write_text('\n'.join(lines), encoding='utf-8')
    versions = {lib: importlib.metadata.version(lib) for lib in ['torch', 'numpy', 'scipy', 'scikit-fem', 'matplotlib', 'pillow', 'pytest']}
    source = Path(__file__).parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
    provenance = {'dependency_versions': versions, 'source_sha256': hashes,
                  'config_sha256': hashlib.sha256((directory/'config.json').read_bytes()).hexdigest(),
                  'note': 'Post-training audit only; independent points did not participate in training, model selection or stopping.'}
    (directory/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'Independent physical audit completed for {len(audit)} models.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    audit_run(args.directory)
