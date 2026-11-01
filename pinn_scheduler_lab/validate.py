"""Independent checkpoint audit; does not change training results or choose models."""
import argparse
import json
import math
from pathlib import Path
import torch
from physics import Config,Problem,parameters_hash
from train import ROOT,save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',default='scheduler60');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    folder=ROOT/'runs'/args.suite;data=json.loads((folder/'data.json').read_text(encoding='utf-8'));checks=[]
    for r in data['runs']:
        location=folder/f"{r['kind']}_{r['plan']}_{r['seed']}";cfg=Config(**r['config']);p=Problem(r['kind'],cfg,r['seed']);model=p.make_model()
        initial=torch.load(location/'initial.pt',weights_only=False);model.load_state_dict(initial['model'])
        assert parameters_hash(model)==r['initial_hash'];assert p.sample_hash==r['sample_hash']
        saved=torch.load(location/'final.pt',weights_only=False);model.load_state_dict(saved['model'])
        metrics,fields=p.evaluate(model);probe=p.probe(model)
        for key,value in metrics.items():assert math.isclose(value,r['final'][key],rel_tol=1e-10,abs_tol=1e-12),(r['plan'],key)
        for key in ['pde_loss','bc_loss','e_bc','e_pde','gradient_cosine']:
            assert math.isclose(probe[key],r['final'][key],rel_tol=1e-9,abs_tol=1e-11),(r['plan'],key)
        for field,values in fields.items():assert torch.allclose(torch.tensor(values),torch.tensor(r['snapshots'][-1]['fields'][field]),rtol=1e-10,atol=1e-12)
        optimizer_steps=max(float(v['step']) for v in saved['optimizer']['state'].values())
        assert optimizer_steps==r['steps'],'Adam was reset across stages'
        assert r['status']=='complete'
        assert cfg.budget_seconds<=r['charged_seconds']<cfg.budget_seconds+.5
        for event in r['events']:assert 'relative_l2' not in event['state'] and 'pressure_relative_l2' not in event['state']
        assert math.isclose(sum(r['costs'][k] for k in ['optimization','probe','policy']),r['charged_seconds'],rel_tol=1e-12)
        checks.append(dict(kind=r['kind'],plan=r['plan'],seed=r['seed'],passed=True,optimizer_steps=optimizer_steps))
    for kind in data['problems']:
        seeds={r['seed'] for r in data['runs'] if r['kind']==kind}
        for seed in seeds:
            group=[r for r in data['runs'] if r['kind']==kind and r['seed']==seed]
            assert len(group)==5;assert len({r['initial_hash'] for r in group})==len({r['sample_hash'] for r in group})==1
    save_json(folder/'validation.json',dict(passed=True,checks=checks))
    print(f'Independent validation passed: {len(checks)} final checkpoints, initialization/sampling hashes, Adam continuity, physical probes, metrics and saved fields.')


if __name__=='__main__':main()
