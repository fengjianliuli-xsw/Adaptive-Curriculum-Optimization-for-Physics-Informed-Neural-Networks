"""Recompute model outputs and independently audit equilibrium on held-out points."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import torch
from physics import Config,Problem,flat_parameters,fingerprint,arc_points


def validate(directory):
    directory=Path(directory);data=json.loads((directory/'results.json').read_text(encoding='utf-8'))
    cfg=Config(**data['config']);torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    ref=json.loads((directory/'fem_reference.json').read_text())
    problem=Problem(cfg,ref);model=problem.make_model()
    model.load_state_dict(torch.load(directory/'initial_state.pt',weights_only=True))
    assert fingerprint(flat_parameters(model))==data['initial_hash']
    _,initial=problem.evaluate(model)
    uv=torch.quasirandom.SobolEngine(2,scramble=True,seed=cfg.seed+9999).draw(1024).double()
    r,theta=uv[:,0].sqrt(),uv[:,1]*np.pi/2
    points=torch.stack((r*theta.sin(),1-r*theta.cos()),1)
    audit=[]
    for run in data['runs']:
        assert run['initial_hash']==data['initial_hash']
        assert run['sampling_hash']==problem.sample_hash
        assert run['snapshots'][0]['fields']==initial
        model.load_state_dict(torch.load(directory/f'{run["id"]}_final.pt',weights_only=True))
        with torch.no_grad():
            losses=problem.losses(model);metrics,fields=problem.evaluate(model)
            np.testing.assert_allclose([run['history'][-1]['loss_'+n] for n in problem.loss_names],losses.numpy(),rtol=1e-11,atol=1e-14)
            assert fields==run['snapshots'][-1]['fields']
            for k,v in metrics.items():np.testing.assert_allclose(run['history'][-1][k],v,rtol=1e-11,atol=1e-14)
            _,(dx,dy)=model(points,1)
            residual=torch.cat((dx[:,2:3]+dy[:,4:5],dx[:,4:5]+dy[:,3:4]),1)
            _,cp,cg=problem.stresses_and_gap(model,arc_points(torch.linspace(0,np.pi/2,801)))
            audit.append({'optimizer':run['id'],'independent_equilibrium_mse':residual.square().mean().item(),
                          'maximum_complementarity_abs':(cp*cg).abs().max().item(),**metrics})
    (directory/'run_config.json').write_text(json.dumps(data['config'],indent=2),encoding='utf-8')
    for row in audit:
        invalid=[key for key,value in row.items() if isinstance(value,float) and not np.isfinite(value)]
        for key in invalid:row[key]=None
        row['nonfinite_independent_audit_fields']=invalid
    (directory/'independent_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
    fem=np.load(directory/'fem_reference.npz')
    fem_kkt={'minimum_nodal_gap_over_r':float(fem['gap_over_r'].min()),
             'minimum_nodal_pressure_over_e':float(fem['pressure_over_e'].min()),
             'maximum_nodal_complementarity':float(np.abs(fem['gap_over_r']*fem['pressure_over_e']).max())}
    assert fem_kkt['minimum_nodal_gap_over_r'] >= -1e-8
    assert fem_kkt['minimum_nodal_pressure_over_e'] >= -1e-8
    assert fem_kkt['maximum_nodal_complementarity'] < 1e-8
    (directory/'reference_kkt_audit.json').write_text(json.dumps(fem_kkt,indent=2),encoding='utf-8')
    versions={k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','scikit-fem','matplotlib']}
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    (directory/'provenance.json').write_text(json.dumps({'versions':versions,'source_sha256':hashes},indent=2),encoding='utf-8')
    print(f'Validated {len(audit)} model checkpoints, shared initialization, losses, fields and independent physics.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();validate(args.directory)
