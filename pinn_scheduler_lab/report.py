"""Offline dashboard, figures, animation and truthful experiment report."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from train import ROOT,save_json
from schedulers import PLANS

COLORS=dict(A='#367db7',B='#cb851e',C='#8b5aca',D='#cf5c8b',E='#138c79')


def rebounds(run):
    result=dict(pde_intervals=0,pde_bc_increases=0,pde_bc_max_ratio=1.,pde_bc_max_error=0.,bc_intervals=0,bc_pde_increases=0,bc_pde_max_ratio=1.,bc_pde_max_rms=0.,switches=0)
    h=run['history']
    for a,b in zip(h,h[1:]):
        if a['action_next']=='PDE':
            ratio=b['bc_loss']/max(a['bc_loss'],1e-12);result['pde_intervals']+=1
            result['pde_bc_increases']+=int(ratio>1.01);result['pde_bc_max_ratio']=max(result['pde_bc_max_ratio'],ratio)
            result['pde_bc_max_error']=max(result['pde_bc_max_error'],b['e_bc'])
        if a['action_next']=='BC':
            ratio=b['pde_loss']/max(a['pde_loss'],1e-12);result['bc_intervals']+=1
            result['bc_pde_increases']+=int(ratio>1.01);result['bc_pde_max_ratio']=max(result['bc_pde_max_ratio'],ratio)
            result['bc_pde_max_rms']=max(result['bc_pde_max_rms'],b['e_pde'])
    actions=[e['action'] for e in run['events'] if e['step']<run['steps']]
    result['switches']=sum(a!=b for a,b in zip(actions,actions[1:]))
    return result


def stat(values):return dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else None)


def summarize(data):
    summary={}
    for kind in data['problems']:
        group=[r for r in data['runs'] if r['kind']==kind];methods=[]
        for p in PLANS:
            rs=[r for r in group if r['plan']==p]
            if not rs:continue
            m=dict(plan=p)
            for key in ['relative_l2','pressure_relative_l2','e_bc','independent_pde_rms']:
                vals=[r['final'][key] for r in rs if key in r['final']]
                if vals:m[key]=stat(vals)
            for key in ['steps','charged_seconds','wall_seconds']:m[key]=stat([r[key] for r in rs])
            for key in ['probe','policy']:m[key+'_fraction']=stat([r['costs'][key]/r['charged_seconds'] for r in rs])
            methods.append(m)
        by={m['plan']:m['relative_l2']['mean'] for m in methods};best=min(by,key=by.get)
        if 'E' not in by:verdict='基线预试：尚未进行 A–E 对照。'
        else:
            competitors={p:by[p] for p in ['B','C','D'] if p in by};best_other=min(competitors,key=competitors.get)
            ratio=by['E']/by[best_other]
            verdict=f"本轮主指标最优：{best}（{by[best]*100:.3f}%）。E 为 {by['E']*100:.3f}%，是 B/C/D 中最佳 {best_other} 的 {ratio:.2f} 倍。"
            if best=='E':verdict+=' E 在该有限设置中领先；尚需多种子、调参后的阶段基线和阈值消融确认。'
            elif by['E']<by.get('A',0):verdict+=' E 虽优于 A，但未胜过最佳顺序/加权基线，不能证明自主切换有优势。'
            else:verdict+=' 当前规则未显示精度优势；应先分析切换与约束反弹，而不是增加控制器网络。'
        summary[kind]=dict(methods=methods,seeds=sorted({r['seed'] for r in group}),best=best,verdict=verdict)
    return summary


def figures(data,folder):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figures_dir=folder/'figures';figures_dir.mkdir(exist_ok=True)
    for kind,problem in data['problems'].items():
        rs=sorted([r for r in data['runs'] if r['kind']==kind and r['seed']==data['summary'][kind]['seeds'][0]],key=lambda r:r['plan'])
        budget=data['config']['budget_seconds']
        fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        specs=[('train_pde_loss','PDE training loss',True),('train_bc_loss','BC training loss',True),('relative_l2','Solution relative L2',True),('e_bc','Normalized boundary error',True),('gradient_cosine','Gradient cosine',False)]
        for ax,(key,title,log) in zip(axs.flat,specs):
            for r in rs:ax.plot([h['charged_seconds'] for h in r['history']],[h[key] for h in r['history']],label=r['plan'],color=COLORS[r['plan']],lw=1.4)
            if log:ax.set_yscale('log')
            if key=='e_bc':ax.axhline(data['config']['tau_bc'],color='gray',ls='--',label='BC tolerance')
            ax.set(title=title,xlabel='Accounted training seconds',xlim=(0,budget));ax.grid(alpha=.18)
        ax=axs.flat[5]
        for i,r in enumerate(rs):
            es=[e for e in r['events'] if e['step']<r['steps']]
            for j,e in enumerate(es):
                end=min(budget,es[j+1]['charged_seconds'] if j+1<len(es) else r['charged_seconds'])
                ax.barh(i,max(0,end-e['charged_seconds']),left=e['charged_seconds'],color={'BC':'#cb851e','PDE':'#367db7','Joint':'#138c79'}[e['action']],height=.65)
        ax.set(yticks=range(len(rs)),yticklabels=[r['plan'] for r in rs],title='Action timeline: orange BC / blue PDE / green Joint',xlabel='Accounted training seconds',xlim=(0,budget))
        axs.flat[0].legend(ncol=5);fig.suptitle(f'{kind}: fixed Adam / seed {rs[0]["seed"]} / {budget:g}s each',fontsize=15)
        fig.savefig(figures_dir/f'{kind}_training.png',dpi=160);plt.close(fig)
        for field in problem['field_names']:
            truth=np.asarray(problem['truth'][field]);axis=np.asarray(problem['axis']);n=len(axis)
            fig,axs=plt.subplots(2,3,figsize=(12,7.5),layout='constrained')
            arrays=[truth]+[np.asarray(r['snapshots'][-1]['fields'][field]) for r in rs]
            vmin=min(x.min() for x in arrays);vmax=max(x.max() for x in arrays)
            for ax,values,title in zip(axs.flat,arrays,['Analytic truth']+[r['plan'] for r in rs]):
                im=ax.imshow(values,origin='lower',extent=(0,1,0,1),vmin=vmin,vmax=vmax,cmap='viridis',aspect='equal');ax.set(title=title,xlabel='x',ylabel='y')
            fig.colorbar(im,ax=list(axs.flat),shrink=.7,label=field);fig.suptitle(f'{kind} / {field} / final predictions')
            fig.savefig(figures_dir/f'{kind}_{field}_final.png',dpi=140);plt.close(fig)
            # A standalone synchronized cross-section animation, actual snapshots only.
            fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');ax.plot(axis,truth[n//2],color='black',ls='--',lw=2,label='Analytic truth')
            lines={r['plan']:ax.plot([],[],color=COLORS[r['plan']],label=r['plan'],lw=1.6)[0] for r in rs}
            all_slices=[truth[n//2]]+[np.asarray(s['fields'][field])[n//2] for r in rs for s in r['snapshots']]
            lo=min(x.min() for x in all_slices);hi=max(x.max() for x in all_slices);margin=.08*(hi-lo or 1)
            ax.set(xlim=(0,1),ylim=(lo-margin,hi+margin),xlabel='x (y = 0.5)',ylabel=field);ax.grid(alpha=.2);ax.legend(ncol=6,loc='upper center');title=ax.set_title('')
            times=np.linspace(0,budget,65)
            def update(t):
                for r in rs:
                    available=[s for s in r['snapshots'] if s['charged_seconds']<=t+1e-6]
                    snapshot=(available or r['snapshots'][:1])[-1]
                    if t==budget:snapshot=r['snapshots'][-1] # final probe may overrun slightly
                    lines[r['plan']].set_data(axis,np.asarray(snapshot['fields'][field])[n//2])
                title.set_text(f'{kind} / {field} / {t:.1f} accounted seconds');return list(lines.values())+[title]
            animation=FuncAnimation(fig,update,frames=times,interval=130,blit=False)
            animation.save(figures_dir/f'{kind}_{field}_slice.gif',writer=PillowWriter(fps=8),dpi=90);plt.close(fig)


def markdown(data,folder):
    cfg=data['config'];lines=['# 二维 PINN 损失调度实验记录','',f"每方法 {cfg['budget_seconds']:g} 秒计费预算；固定 Adam，学习率 {cfg['lr']}；只改变训练目标。",'',
        '## 结论（最终检查点）','']
    for kind,s in data['summary'].items():
        lines += [f"### {data['problems'][kind]['title']}",'',s['verdict'],'',
                  '|方案|相对 L²|压力相对 L²|归一化 e_bc|独立 PDE RMS|步数|诊断%|策略%|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for m in s['methods']:
            def number(key,percent=False):
                if key not in m:return '—'
                st=m[key];scale=100 if percent else 1;v=f"{st['mean']*scale:.4g}"+('%' if percent else '')
                return v+(f" ± {st['std']*scale:.3g}" if st['std'] is not None else '')
            lines.append('|'+ '|'.join([m['plan']+' '+PLANS[m['plan']],number('relative_l2',True),number('pressure_relative_l2',True),number('e_bc'),number('independent_pde_rms'),number('steps'),number('probe_fraction',True),number('policy_fraction',True)])+'|')
        lines += ['',f"种子：{s['seeds']}；图：[训练曲线](figures/{kind}_training.png)。",'']
    lines += ['## 反弹与动作切换','', '只统计前一检查点决定的动作确实执行到下一检查点的区间。上涨定义为检查损失 > 前次的 1.01 倍。倍数下限设 1，分母保护 1e-12；接近零的 BC 损失会产生很大的倍数，必须结合绝对 e_bc 曲线阅读。','',
              '|PDE|方案|PDE 时 BC 上涨 / 区间|最大倍数|BC 时 PDE 上涨 / 区间|最大倍数|切换次数|','|---|---|---:|---:|---:|---:|---:|']
    for r in data['runs']:
        b=r['rebounds'];lines.append(f"|{r['kind']}|{r['plan']}|{b['pde_bc_increases']}/{b['pde_intervals']}|{b['pde_bc_max_ratio']:.3g}|{b['bc_pde_increases']}/{b['bc_intervals']}|{b['bc_pde_max_ratio']:.3g}|{b['switches']}|")
    lines += ['', '## E 的行为检查','']
    for r in data['runs']:
        if r['plan']!='E':continue
        b=r['rebounds'];shares={a:r['action_seconds'][a]/r['costs']['optimization']*100 for a in ['BC','PDE','Joint']}
        lines += [f"- {r['kind']}：动作切换 {b['switches']} 次；BC / PDE / Joint 分别占实际优化秒的 {shares['BC']:.1f}% / {shares['PDE']:.1f}% / {shares['Joint']:.1f}%。PDE 检查段中最大 e_bc={b['pde_bc_max_error']:.4g}（目标 τbc={cfg['tau_bc']}）。"]
    lines += ['', '这些是检查点观测到的峰值，不能当成逐步最大值。阈值是触发规则，不是硬约束：PDE 更新可能在两次检查之间显著破坏边界；最小驻留还可能延迟恢复。大量时间最终落在 Joint 时，不能把精度改进归因于长期的纯 PDE 学习。', '', '## 开销是否值得','']
    for r in data['runs']:
        if r['plan']!='E':continue
        baseline=next(x for x in data['runs'] if x['kind']==r['kind'] and x['seed']==r['seed'] and x['plan']=='A')
        lines += [f"- {r['kind']}：E 规则计算 {r['costs']['policy']:.5f}s，全部物理诊断 {r['costs']['probe']:.3f}s；A 相应诊断 {baseline['costs']['probe']:.3f}s。规则计算很便宜，但状态信息不是免费的；按步数检查还使不同方案诊断次数不同。应将两列一起考虑。"]
    lines += ['', '本版所有方法均带同样诊断，没有另外运行免诊断 A，因此不能声称已经准确测出部署时 E 相对最轻量 A 的全部增量开销。', '', '## 公平性与限制','',
        '- 每个 PDE/种子内，A–E 初始权重与采样 SHA256 相同；Adam 状态跨阶段连续保留。各方法最后步数可以不同，BC 单步成本较低。',
        '- 计费秒 = 优化耗时 + 固定检查耗时 + 策略耗时。所有方法每 50 步接受相同探针和梯度冲突诊断；D 额外逐步更新权重。解析网格和最终审计不进入控制状态，评估时间单列。计费时间不是训练循环墙钟时间（不含初始化、初始保存和最终 JSON/图表生成）；后者包含 Python 调度、评估及保存。',
        '- 最后一步及最终诊断允许少量预算超时。曲线采用实际计费时间；没有按真解误差早停或挑选最优检查点。',
        '- 单种子是流程验证，不支持显著性推断。B/C 只用了等时间三阶段，尚未精调分段；E 即使领先也不能证明优于最优手工阶段。',
        '- 阈值与迟滞、最小驻留是人为选择。未做阈值、K、Re、网络或高频/瞬态问题的消融。没有新增控制器网络，也没有训练学习策略。',
        '- 当前研究对象是稳态、单位方域和 Re=20 的解析解问题；不据此推断复杂真实流动表现。',
        '- 先运行 A 基线门槛（热/速度 <5%、压力 <10%），通过后冻结设置，才运行 B–E。30 秒预试压力未过门槛，因此全组统一采用 60 秒。试验选择过程完整保留在 pilot30、pilot60，不隐藏负结果。','',
        '详见项目根 README.md 的方程、公式、规则与复现命令。']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',default='scheduler60');parser.add_argument('--skip-media',action='store_true');args=parser.parse_args()
    folder=ROOT/'runs'/args.suite;data=json.loads((folder/'data.json').read_text(encoding='utf-8'))
    for r in data['runs']:r['rebounds']=rebounds(r)
    data['summary']=summarize(data)
    save_json(folder/'summary.json',data['summary'])
    # Round display data only; source result.json retains full precision.
    def display_round(value):
        if isinstance(value,float):return float(f'{value:.7g}')
        if isinstance(value,list):return [display_round(v) for v in value]
        if isinstance(value,dict):return {k:display_round(v) for k,v in value.items()}
        return value
    (folder/'data.js').write_text('window.PINN_LAB='+json.dumps(display_round(data),ensure_ascii=False,allow_nan=False)+';',encoding='utf-8')
    shutil.copyfile(ROOT/'dashboard.html',folder/'dashboard.html')
    markdown(data,folder)
    with (folder/'final_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        keys=['kind','plan','seed','status','steps','charged_seconds','relative_l2','pressure_relative_l2','e_bc','e_pde','independent_pde_rms','probe_seconds','policy_seconds','reference_seconds','wall_seconds']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in data['runs']:
            row={k:r.get(k,r['final'].get(k)) for k in keys};row.update(probe_seconds=r['costs']['probe'],policy_seconds=r['costs']['policy'],reference_seconds=r['costs']['reference']);writer.writerow(row)
    if not args.skip_media:figures(data,folder)
    print('Report:',folder/'dashboard.html',flush=True)


if __name__=='__main__':main()
