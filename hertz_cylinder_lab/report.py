"""Scientific exports and self-contained replay for the cylinder/rigid-plane contact."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def build_report(data,out,gif=True):
    out=Path(out)
    audit_path=out/'independent_audit.json'
    if audit_path.exists():data['independent_audit']=json.loads(audit_path.read_text(encoding='utf-8'))
    template=Path(__file__).with_name('dashboard.template.html')
    if template.exists():
        packed=json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</','<\\/')
        (out/'dashboard.html').write_text(template.read_text(encoding='utf-8').replace('__DATA__',packed),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    runs=data['runs']; x=np.array(data['pressure_x_mm']); near=x<=3
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    axes[0].plot(x[near],np.array(data['pressure_hertz_mpa'])[near],'k--',lw=2,label='Hertz local theory')
    axes[0].plot(x[near],np.array(data['pressure_fem_mpa'])[near],color='#169c9c',lw=2,label='Active-set FEM')
    axes[1].plot(x[near],np.array(data['gap_fem_um'])[near],'k--',lw=2,label='FEM gap')
    axes[1].axhline(0,color='#6c7d8b',lw=1)
    for run in runs:
        field=run['snapshots'][-1]['fields']; hist=run['history']
        axes[0].plot(x[near],np.array(field['pressure_mpa'])[near],color=run['color'],label=run['label'])
        axes[1].plot(x[near],np.array(field['gap_um'])[near],color=run['color'],label=run['label'])
        axes[2].semilogy([h['train_seconds'] for h in hist],[max(1e-16,h['pressure_relative_l2']) for h in hist],color=run['color'],label=run['label'])
    axes[0].set(xlabel='x (mm), right half',ylabel='Contact pressure (MPa)',title='Unknown contact region / pressure')
    axes[1].set(xlabel='x (mm)',ylabel='Gap (um)',title='KKT: gap >= 0; negative = penetration')
    axes[2].set(xlabel='Measured training time (s)',ylabel='Pressure relative L2 vs FEM',title='Accuracy versus computation')
    axes[0].set_ylim(-.1*max(data['pressure_hertz_mpa']),2*max(data['pressure_hertz_mpa']))
    axes[0].set_title('Pressure (reference scale; divergent values clipped)')
    gap_scale=max(1.,float(np.array(data['gap_fem_um'])[near].max()))
    axes[1].set_ylim(-.1*gap_scale,1.2*gap_scale)
    axes[2].set_ylim(1e-6,1e6)
    for ax in axes:ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8,frameon=False)
    fig.savefig(out/'pressure_gap_and_time.png',dpi=160);plt.close(fig)
    keys=[('loss_total','Total physical loss')]+[(f'loss_{n}',n) for n in data['loss_names']]+[
        ('pressure_relative_l2','Pressure error vs FEM'),('max_penetration_um','Max penetration (um)'),
        ('independent_fb_rms','Independent KKT/FB RMS'),('relative_l2','Displacement field error'),('force_balance_relative','Load balance relative error')]
    fig,axes=plt.subplots((len(keys)+3)//4,4,figsize=(17,10),squeeze=False,constrained_layout=True)
    for ax,(key,title) in zip(axes.flat,keys):
        for r in runs:ax.semilogy([h['step'] for h in r['history']],[max(1e-16,h[key]) for h in r['history']],color=r['color'],label=r['label'])
        ax.set(xlabel='Outer update',title=title);ax.grid(alpha=.2)
        ax.set_ylim(1e-16,1e6)
    fig.suptitle('Reference-scale plots: divergent values outside displayed limits; full values in CSV')
    for ax in list(axes.flat)[len(keys):]:ax.set_visible(False)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=8,frameon=False)
    fig.savefig(out/'training_curves.png',dpi=150);fig.savefig(out/'training_curves.svg');plt.close(fig)
    # Actual deformation, without misleading magnification across a rigid plane.
    ref=np.load(out/'fem_reference.npz'); nodes=ref['nodes'];disp=ref['displacement_over_r']
    radius=data['config']['radius_mm']; circle=np.linalg.norm(nodes-np.array([0,1]),axis=1)
    arc=np.flatnonzero(np.abs(circle-1)<1e-8);arc=arc[np.argsort(nodes[arc,0])]
    original=nodes[arc]*radius;deformed=(nodes[arc]+disp[arc])*radius
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax in axes:
        for sign in [-1,1]:
            ax.plot(sign*original[:,0],original[:,1],'--',color='#849bab',lw=1,label='Original' if sign==1 else None)
            ax.plot(sign*deformed[:,0],deformed[:,1],color='#1765c5',lw=2,label='Deformed FEM (1x)' if sign==1 else None)
        ax.axhline(0,color='#183044',lw=2,label='Rigid plane');ax.grid(alpha=.2);ax.set(xlabel='x (mm)',ylabel='y (mm)')
    axes[0].set(xlim=(-radius*1.05,radius*1.05),ylim=(-.1,radius*1.05),title='Lower half-cylinder: actual-scale deformation')
    axes[0].set_aspect('equal');axes[1].set(xlim=(-2,2),ylim=(-.02,.25),title='Contact-zone zoom (no deformation amplification)');axes[1].legend(fontsize=8)
    fig.savefig(out/'geometry_and_deformation.png',dpi=170);plt.close(fig)
    mask=np.array(data['mask'],dtype=bool)
    truth=np.array(data['truth']['uy_um'])
    lo,hi=truth[mask].min(),truth[mask].max()
    count=len(runs)+1;cols=4
    fig,axes=plt.subplots((count+cols-1)//cols,cols,figsize=(14,3*((count+cols-1)//cols)),squeeze=False,constrained_layout=True)
    for ax,field,label in zip(axes.flat,[truth]+[np.array(r['snapshots'][-1]['fields']['uy_um']) for r in runs],['FEM reference']+[r['label'] for r in runs]):
        image=ax.imshow(np.ma.array(field,mask=~mask),origin='lower',extent=[0,1,0,1],cmap='viridis',vmin=lo,vmax=hi)
        ax.set(title=label,xlabel='x/R',ylabel='y/R')
    for ax in list(axes.flat)[count:]:ax.set_visible(False)
    fig.colorbar(image,ax=list(axes.flat)[:count],label='uy (um); reference limits, out-of-range values clipped',shrink=.65)
    fig.savefig(out/'displacement_fields.png',dpi=150);plt.close(fig)
    if gif:
        focus=[r for r in runs if r['id'] in ['adam','adam_lbfgs','soap','dcgd','config']] or runs[:3]
        fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
        axes[0].plot(x[near],np.array(data['pressure_hertz_mpa'])[near],'k--',label='Hertz')
        axes[0].plot(x[near],np.array(data['pressure_fem_mpa'])[near],color='#169c9c',label='FEM')
        axes[1].axhline(0,color='black',lw=1)
        lines=[]
        for run in focus:
            line,=axes[0].plot(x[near],np.array(run['snapshots'][0]['fields']['pressure_mpa'])[near],color=run['color'],label=run['label'])
            gapline,=axes[1].plot(x[near],np.array(run['snapshots'][0]['fields']['gap_um'])[near],color=run['color'])
            lines.append((line,gapline))
        pvalues=[np.array(s['fields']['pressure_mpa'])[near] for r in focus for s in r['snapshots']]
        gvalues=[np.array(s['fields']['gap_um'])[near] for r in focus for s in r['snapshots']]
        axes[0].set(xlabel='x (mm)',ylabel='Pressure (MPa)',ylim=(min(v.min() for v in pvalues)-.1,max(max(v.max() for v in pvalues),max(data['pressure_hertz_mpa']))*1.05))
        axes[1].set(xlabel='x (mm)',ylabel='Gap (um)',ylim=(min(v.min() for v in gvalues)-1,max(v.max() for v in gvalues)+1))
        axes[0].legend(fontsize=8);title=fig.suptitle('KKT cylinder contact - training replay')
        for ax in axes:ax.grid(alpha=.2)
        fig.canvas.draw();fig.set_layout_engine(None)
        horizon=max(r['history'][-1]['step'] for r in focus)
        def update(step):
            for r,(line,gapline) in zip(focus,lines):
                index=max(0,np.searchsorted([h['step'] for h in r['history']],step,side='right')-1)
                field=r['snapshots'][index]['fields'];line.set_ydata(np.array(field['pressure_mpa'])[near]);gapline.set_ydata(np.array(field['gap_um'])[near])
            title.set_text(f'KKT cylinder / rigid plane | outer update {step}')
            return [title]
        animation=FuncAnimation(fig,update,frames=np.unique(np.linspace(0,horizon,41,dtype=int)),blit=False)
        animation.save(out/'contact_evolution.gif',writer=PillowWriter(fps=7),dpi=90);plt.close(fig)
    rows=[{'optimizer':r['id'],'status':r['status'],**r['history'][-1]} for r in runs]
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 二维圆柱–刚性平面 KKT 接触', '',
           '力控制、平面应变、小变形、无摩擦的半圆柱基准；用对称性求解四分之一圆截面。',
           '载荷是已知输入，合力平衡是约束检查，不是未知接触力反演。', '',
           f'Hertz 接触半宽：{data["hertz_half_width_mm"]:.6f} mm；总载荷：{data["applied_force_n"]:.3f} N。',
           'FEM 采用主动集解 KKT；Hertz 压力、接触半宽与 FEM 位移都未进入 PINN 训练。', '',
           '| 方法 | 更新数 | 训练秒 | 压力相对 FEM 误差 | 位移相对误差 | 最大穿透 μm | 最大拉应力 MPa | 独立 FB RMS |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(runs,key=lambda r:r['history'][-1]['pressure_relative_l2']):
        h=r['history'][-1];lines.append(f'| {r["label"]} | {h["step"]} | {h["train_seconds"]:.2f} | {h["pressure_relative_l2"]:.2%} | {h["relative_l2"]:.2%} | {h["max_penetration_um"]:.4f} | {h["max_tension_mpa"]:.4f} | {h["independent_fb_rms"]:.3e} |')
    lines+=['','相同初始化、采样和物理损失；各方法有相同最大步数及优化耗时软预算，完成整步后停止，实际步数可能不同。',
            'KKT 罚残差较低不自动证明正确；应联合查看压力/位移误差、独立平衡残差、非穿透、无拉力及互补性。',
            '压力误差只在接触附近 x/R≤0.3 比较，避免竖直切线处投影除以零；域外弧上的违规另由全弧 KKT 检查覆盖。']
    lines+=['','静态压力/间隙图固定为参考尺度，超范围失稳曲线在图中截断；原始数值和失败状态完整保存在 CSV/JSON。']
    (out/'experiment_report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('results',type=Path);parser.add_argument('--no-gif',action='store_true');args=parser.parse_args()
    build_report(json.loads(args.results.read_text(encoding='utf-8')),args.results.parent,gif=not args.no_gif)
