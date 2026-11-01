"""Baseline gate then frozen A--E comparison. Existing baseline work is reused verbatim."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from physics import Config

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--budget',type=float,default=60.)
    parser.add_argument('--suite',default='scheduler60');parser.add_argument('--baseline',default='pilot60')
    parser.add_argument('--seeds',type=int,nargs='+',default=[42]);args=parser.parse_args()
    command=[sys.executable,str(ROOT/'train.py'),'--budget',str(args.budget),'--seeds',*[str(s) for s in args.seeds]]
    subprocess.run(command+['--suite',args.baseline,'--plans','A','--resume'],cwd=ROOT,check=True)
    gates=[]
    for kind in ['heat2d','ns2d']:
        for seed in args.seeds:
            source=ROOT/'runs'/args.baseline/f'{kind}_A_{seed}'
            r=json.loads((source/'result.json').read_text(encoding='utf-8'));f=r['final']
            passed=r['status']=='complete' and f['relative_l2']<=.05 and (kind=='heat2d' or f['pressure_relative_l2']<=.1)
            gates.append(dict(kind=kind,seed=seed,passed=passed,relative_l2=f['relative_l2'],pressure_relative_l2=f.get('pressure_relative_l2')))
    target=ROOT/'runs'/args.suite;target.mkdir(parents=True,exist_ok=True)
    (target/'baseline_gate.json').write_text(json.dumps(gates,indent=2),encoding='utf-8')
    if not all(g['passed'] for g in gates):raise SystemExit('Baseline gate failed; comparison not launched. Inspect baseline_gate.json.')
    for kind in ['heat2d','ns2d']:
        for seed in args.seeds:
            name=f'{kind}_A_{seed}'
            if not (target/name).exists():shutil.copytree(ROOT/'runs'/args.baseline/name,target/name)
    subprocess.run(command+['--suite',args.suite,'--plans','A','B','C','D','E','--resume'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'report.py'),'--suite',args.suite],cwd=ROOT,check=True)


if __name__=='__main__':main()
