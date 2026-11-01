"""Standalone offline dashboard + shareable scientific plots and GIF."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def build_report(data, out, gif=True):
    out = Path(out)
    template = Path(__file__).with_name('dashboard.template.html').read_text(encoding='utf-8')
    packed = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('</', '<\\/')
    (out/'dashboard.html').write_text(template.replace('__RESULTS_JSON__', packed), encoding='utf-8')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    runs = data['optimizers']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    keys = ['loss_total', 'loss_pde', 'loss_initial', 'loss_boundary', 'relative_l2', 'within_tolerance_pct']
    titles = ['Total training loss', 'PDE residual MSE', 'Initial condition MSE',
              'Boundary condition MSE', 'Relative L2 error (evaluation grid)', 'Points within absolute error 0.05 (%)']
    for ax, key, title in zip(axes.flat, keys, titles):
        for run in runs:
            history = run['history']
            ax.plot([r['step'] for r in history], [max(1e-16, r[key]) for r in history],
                    color=run['color'], label=run['label'], lw=1.4)
        if key != 'within_tolerance_pct':
            ax.set_yscale('log')
        else:
            ax.set_ylim(0, 102)
        ax.set(xlabel='Optimizer outer step', title=title)
        ax.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5, frameon=False)
    fig.savefig(out/'training_curves.png', dpi=160)
    fig.savefig(out/'training_curves.svg')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for run in runs:
        ax.semilogy([r['train_seconds'] for r in run['history']],
                    [max(1e-16, r['relative_l2']) for r in run['history']],
                    color=run['color'], label=run['label'])
    ax.set(xlabel='Measured optimizer training time (s), excludes diagnostics', ylabel='Relative L2 error',
           title='Error versus actual training time')
    ax.grid(alpha=0.2); ax.legend(ncol=2, fontsize=8)
    fig.savefig(out/'error_vs_time.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, index in zip(axes, [0, 2, 4]):
        ax.plot(data['x'], data['truth_slices'][index], 'k--', lw=2, label='Exact')
        for run in runs:
            ax.plot(data['x'], run['snapshots'][-1]['curves'][index],
                    color=run['color'], label=run['label'], alpha=0.85)
        ax.set(xlabel='x', ylabel='u(x,t)', title=f'Final predictions, t={data["slice_times"][index]:.2f}')
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=6, fontsize=8, frameon=False)
    fig.savefig(out/'final_predictions.png', dpi=160); plt.close(fig)
    if gif:
        cols = min(3, len(runs)); rows = (len(runs)+cols-1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 2.8*rows), squeeze=False, constrained_layout=True)
        lines, texts = [], []
        lower = min(min(s['curves'][2]) for r in runs for s in r['snapshots'])-0.05
        upper = max(max(s['curves'][2]) for r in runs for s in r['snapshots'])+0.05
        for ax, run in zip(axes.flat, runs):
            ax.plot(data['x'], data['truth_slices'][2], 'k--', lw=1.5, label='Exact')
            line, = ax.plot(data['x'], run['snapshots'][0]['curves'][2], color=run['color'], lw=2, label='PINN')
            lines.append(line)
            texts.append(ax.text(0.03, 0.95, '', transform=ax.transAxes, va='top', fontsize=8))
            ax.set(title=run['label'], xlabel='x', ylim=(lower, upper))
            ax.grid(alpha=0.2)
        for ax in list(axes.flat)[len(runs):]:
            ax.set_visible(False)
        # Resolve layout once. Re-running constrained layout on 10 axes per frame
        # would dominate animation export time without changing the scientific plot.
        fig.suptitle('PINN training convergence | fixed physical time t=0.50')
        fig.canvas.draw()
        fig.set_layout_engine(None)
        horizon = max(r['history'][-1]['step'] for r in runs)
        steps = np.unique(np.linspace(0, horizon, min(81, horizon+1), dtype=int))

        def update(frame):
            for run, line, label in zip(runs, lines, texts):
                idx = max(0, np.searchsorted([s['step'] for s in run['snapshots']], frame, side='right')-1)
                snap, metric = run['snapshots'][idx], run['history'][idx]
                line.set_ydata(snap['curves'][2])
                label.set_text(f'step {snap["step"]} | rel L2 {metric["relative_l2"]:.2e}')
            fig.suptitle(f'PINN training convergence | fixed physical time t=0.50 | outer step {frame}')
            return lines+texts
        animation = FuncAnimation(fig, update, frames=steps, interval=100, blit=False)
        animation.save(out/'prediction_evolution.gif', writer=PillowWriter(fps=10), dpi=85)
        plt.close(fig)
    summary = {r['id']: {**r['history'][-1], 'status': r['status'],
                         'initial_hash': r['initial_hash'], 'sampling_hash': r['sampling_hash']}
               for r in runs}
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 一维热传导 PINN 优化器实验结果', '',
             f'随机种子 {data["config"]["seed"]}；每种方法最大外层步数 {data["config"]["steps"]}。', '',
             '所有方法的初始参数、固定全批采样点、损失权重和评估网格相同。',
             '单 PDE、单种子、固定超参数的教学实验；误差较低不等同于耗时更少。', '',
             '| 方法 | 最终 step | 相对 L2 误差 | 误差达标率 | 训练耗时 (s) | 总损失 |',
             '|---|---:|---:|---:|---:|---:|']
    for run in sorted(runs, key=lambda r: r['history'][-1]['relative_l2']):
        h = run['history'][-1]
        lines.append(f'| {run["label"]} | {h["step"]} | {h["relative_l2"]:.4%} | {h["within_tolerance_pct"]:.2f}% | {h["train_seconds"]:.3f} | {h["loss_total"]:.4e} |')
    lines.extend(['', '达标率的定义为固定网格上绝对误差不超过 0.05 的点占比。',
                  '训练耗时排除诊断、快照和导出；实际总运行耗时另存于 results.json。',
                  '残差调用次数不能当作等价 FLOPs，尤其 LM 构建了完整残差 Jacobian。', '',
                  '打开 dashboard.html 可播放训练并切换迭代数/耗时/残差调用轴；完整原始数据见 results.json。'])
    (out/'experiment_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('--no-gif', action='store_true')
    args = parser.parse_args()
    build_report(json.loads(args.results.read_text(encoding='utf-8')), args.results.parent, gif=not args.no_gif)
