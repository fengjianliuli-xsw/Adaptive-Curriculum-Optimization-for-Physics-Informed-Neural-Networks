"""Reproducible equal-accounted-time experiments. All runtime files stay beside this script."""
import argparse
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import torch
from physics import Config,Problem,parameters_hash
from schedulers import Controller,PLANS

ROOT=Path(__file__).resolve().parent


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def run(kind,plan,seed,cfg,folder):
    torch.manual_seed(seed)
    problem=Problem(kind,cfg,seed);model=problem.make_model()
    initial_hash=parameters_hash(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    policy=Controller(plan,cfg,seed)
    folder.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'config':cfg.to_dict(),'kind':kind,'seed':seed},folder/'initial.pt')
    history=[];snapshots=[];events=[]
    costs=dict(optimization=0.,probe=0.,policy=0.,reference=0.)
    action_seconds=dict(BC=0.,PDE=0.,Joint=0.);action_steps=dict(BC=0,PDE=0,Joint=0)
    previous=None;last_step_seconds=0.;step=0;last_checkpoint=0;status='complete'
    begin=time.perf_counter()
    def charged():return costs['optimization']+costs['probe']+costs['policy']
    def checkpoint(final=False):
        nonlocal previous,last_checkpoint
        tick=time.perf_counter()
        state=problem.probe(model,previous,last_step_seconds)
        with torch.no_grad():train_lf=problem.pde_loss(model).item();train_lb=problem.bc_loss(model).item()
        costs['probe']+=time.perf_counter()-tick
        if not all(math.isfinite(v) for v in state.values()):raise FloatingPointError('nonfinite probe')
        tick=time.perf_counter()
        action,reason=policy.decision(state,charged()/cfg.budget_seconds)
        policy.action=action
        costs['policy']+=time.perf_counter()-tick
        events.append(dict(step=step,charged_seconds=charged(),action=action,reason=reason,state=state.copy(),final=final))
        tick=time.perf_counter();metrics,fields=problem.evaluate(model)
        costs['reference']+=time.perf_counter()-tick
        row=dict(step=step,charged_seconds=charged(),action_next=action,weights=policy.weights.tolist(),
                 train_pde_loss=train_lf,train_bc_loss=train_lb,train_total=train_lf+train_lb,**state,**metrics)
        history.append(row)
        if step==0 or final or len(history)%5==0:snapshots.append(dict(step=step,charged_seconds=charged(),fields=fields))
        previous=state;last_checkpoint=step
        return row
    checkpoint()
    while charged()<cfg.budget_seconds and step<cfg.max_steps:
        action=policy.action
        tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        lf=problem.pde_loss(model) if action!='BC' else None
        lb=problem.bc_loss(model) if action!='PDE' else None
        if plan=='D':
            # Remove the measured weight update from optimizer time and charge it to policy time.
            t=time.perf_counter();weights=policy.reweight([lf.item(),lb.item()]);elapsed=time.perf_counter()-t
            costs['policy']+=elapsed
            objective=float(weights[0])*lf+float(weights[1])*lb
        else:
            elapsed=0.;objective=lf if action=='PDE' else lb if action=='BC' else lf+lb
        if not torch.isfinite(objective):status='nonfinite_objective';break
        objective.backward();optimizer.step()
        last_step_seconds=time.perf_counter()-tick-elapsed
        costs['optimization']+=last_step_seconds;action_seconds[action]+=last_step_seconds;action_steps[action]+=1;step+=1
        if step%cfg.check_every==0:checkpoint()
    if last_checkpoint!=step:checkpoint(final=True)
    else:
        # Save the final field without another charged diagnostic when already checked.
        tick=time.perf_counter();fields=problem.evaluate(model)[1];costs['reference']+=time.perf_counter()-tick
        snapshots.append(dict(step=step,charged_seconds=charged(),fields=fields))
    if step>=cfg.max_steps and charged()<cfg.budget_seconds:status='step_cap'
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg.to_dict(),
                'kind':kind,'seed':seed,'plan':plan,'initial_hash':initial_hash,'sample_hash':problem.sample_hash},folder/'final.pt')
    result=dict(kind=kind,plan=plan,label=PLANS[plan],seed=seed,config=cfg.to_dict(),initial_hash=initial_hash,
                sample_hash=problem.sample_hash,status=status,steps=step,costs=costs,charged_seconds=charged(),
                wall_seconds=time.perf_counter()-begin,action_seconds=action_seconds,action_steps=action_steps,
                history=history,snapshots=snapshots,events=events,final=history[-1])
    save_json(folder/'result.json',result)
    print(f'{kind} {plan} seed={seed}: L2={history[-1]["relative_l2"]:.4%}, BC={history[-1]["e_bc"]:.4g}, '
          f'PDE={history[-1]["e_pde"]:.4g}, steps={step}, budget={charged():.3f}s, actions={action_steps}',flush=True)
    return result,problem.metadata()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',default='pilot');parser.add_argument('--pdes',nargs='+',default=['heat2d','ns2d'],choices=['heat2d','ns2d'])
    parser.add_argument('--plans',nargs='+',default=list(PLANS),choices=list(PLANS));parser.add_argument('--seeds',nargs='+',type=int,default=[42])
    parser.add_argument('--budget',type=float,default=30.);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=Config(budget_seconds=args.budget)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64)
    suite=ROOT/'runs'/args.suite;suite.mkdir(parents=True,exist_ok=True)
    save_json(suite/'manifest.json',dict(config=cfg.to_dict(),pdes=args.pdes,plans=args.plans,seeds=args.seeds,
               python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
               cpu=platform.processor(),threads=1,dtype='float64',baseline_gate=dict(heat_l2=.05,ns_velocity_l2=.05,ns_pressure_l2=.1)))
    results=[];metadata={}
    # Rotate order by seed/PDE so one method is not always first on the host.
    for ki,kind in enumerate(args.pdes):
        for si,seed in enumerate(args.seeds):
            order=args.plans[(si+ki)%len(args.plans):]+args.plans[:(si+ki)%len(args.plans)]
            for plan in order:
                folder=suite/f'{kind}_{plan}_{seed}'
                if args.resume and (folder/'result.json').exists():
                    result=json.loads((folder/'result.json').read_text(encoding='utf-8'));meta=Problem(kind,cfg,seed).metadata()
                    if result['config']!=cfg.to_dict():raise ValueError('resume config mismatch')
                else:result,meta=run(kind,plan,seed,cfg,folder)
                results.append(result);metadata[kind]=meta
                save_json(suite/'data.json',dict(problems=metadata,runs=results,plans=PLANS,config=cfg.to_dict()))
    for kind in args.pdes:
        for seed in args.seeds:
            group=[r for r in results if r['kind']==kind and r['seed']==seed]
            assert len({r['initial_hash'] for r in group})==len({r['sample_hash'] for r in group})==1
    print('Saved',suite,flush=True)


if __name__=='__main__':main()
