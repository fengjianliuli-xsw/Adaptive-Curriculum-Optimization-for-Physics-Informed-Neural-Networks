"""Build offline replay, static scientific plots, contact force report and animations."""
import argparse
import csv
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def closest(run, step):
    return max(0, np.searchsorted([h['step'] for h in run['history']], step, side='right')-1)


def plot_curves(problem, out):
    runs = problem['runs']
    specs = [('loss_total', 'Weighted total loss')]+[(f'loss_{key}', f'{key} (weighted MSE)') for key in problem['loss_names']]
    specs += [('relative_l2', 'Relative L2 field error')]
    if problem['id'] == 'contact2d':
        specs += [('pressure_relative_l2', 'Contact pressure relative L2'), ('force_relative_error', 'Total contact force relative error')]
    specs += [('within_tolerance_pct', 'Displacement error <= 0.05 mm (%)' if problem['id'] == 'contact2d' else 'Points within absolute error 0.05 (%)')]
    cols = 3
    rows = (len(specs)+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 3.2*rows), squeeze=False, constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, specs):
        for run in runs:
            ax.plot([h['step'] for h in run['history']], [max(1e-16, h[key]) for h in run['history']],
                    label=run['label'], color=run['color'], lw=1.4)
        ax.set(title=title, xlabel='Optimizer outer step')
        if key != 'within_tolerance_pct':
            ax.set_yscale('log')
        else:
            ax.set_ylim(0, 102)
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(specs):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5, frameon=False, fontsize=9)
    fig.savefig(out/f'{problem["id"]}_training_curves.png', dpi=150)
    fig.savefig(out/f'{problem["id"]}_training_curves.svg')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2 if problem['id'] == 'contact2d' else 1, figsize=(12 if problem['id'] == 'contact2d' else 8, 4), squeeze=False, constrained_layout=True)
    for ax, key in zip(axes.flat, ['relative_l2', 'force_relative_error']):
        for run in runs:
            ax.semilogy([h['train_seconds'] for h in run['history']], [max(1e-16, h[key]) for h in run['history']],
                        label=run['label'], color=run['color'])
        ax.set(xlabel='Measured optimizer time (s), diagnostics excluded', ylabel=key)
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8, ncol=2)
    fig.savefig(out/f'{problem["id"]}_error_vs_time.png', dpi=150)
    plt.close(fig)


def field_figure(problem, out):
    key = problem['field_names'][0]
    truth = np.array(problem['truth'][key])
    runs = problem['runs']
    count = len(runs)+1
    cols, rows = 4, (count+3)//4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3*rows), constrained_layout=True, squeeze=False)
    all_fields = [truth]+[np.array(r['snapshots'][-1]['fields'][key]) for r in runs]
    lo, hi = min(a.min() for a in all_fields), max(a.max() for a in all_fields)
    for ax, field, title in zip(axes.flat, all_fields, ['Reference']+[r['label'] for r in runs]):
        image = ax.imshow(field, origin='lower', extent=[0, 1, 0, 1], cmap='viridis', vmin=lo, vmax=hi, aspect='equal')
        ax.set(title=title, xlabel='x / W', ylabel='y / H')
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    fig.colorbar(image, ax=list(axes.flat)[:count], label=key, shrink=.65)
    fig.savefig(out/f'{problem["id"]}_final_fields.png', dpi=150)
    plt.close(fig)


def contact_pressure(problem, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    x, truth = problem['cut_x'], problem['cut_truth']
    axes[0].plot(x, truth, 'k--', lw=2.4, label='Independent reference')
    for run in problem['runs']:
        axes[0].plot(x, run['snapshots'][-1]['fields']['pressure_kpa'], color=run['color'], label=run['label'], alpha=.9)
        axes[1].plot([h['step'] for h in run['history']], [h['force_n'] for h in run['history']], color=run['color'], label=run['label'])
    axes[1].axhline(problem['force_true_n'], color='black', linestyle='--', lw=2, label='True force')
    axes[0].set(xlabel='x / W', ylabel='Pressure (kPa)', title='Recovered contact pressure')
    axes[1].set(xlabel='Optimizer outer step', ylabel='Total normal force (N)', title='Contact force estimate during training')
    for ax in axes:
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False, fontsize=8)
    fig.savefig(out/'contact2d_pressure_and_force.png', dpi=170)
    plt.close(fig)


def export_animation(problem, out):
    focus = [r for r in problem['runs'] if r['id'] in ['adam', 'adam_lbfgs', 'soap']]
    if not focus:
        focus = problem['runs'][:3]
    key = problem['field_names'][0]
    truth = np.array(problem['truth'][key])
    contact = problem['id'] == 'contact2d'
    fig = plt.figure(figsize=(13, 7 if contact else 4), constrained_layout=True)
    grid = fig.add_gridspec(2 if contact else 1, len(focus)+1)
    axes = [fig.add_subplot(grid[0, i]) for i in range(len(focus)+1)]
    lo = min(truth.min(), *[np.array(s['fields'][key]).min() for r in focus for s in r['snapshots']])
    hi = max(truth.max(), *[np.array(s['fields'][key]).max() for r in focus for s in r['snapshots']])
    images, labels = [], []
    for i, ax in enumerate(axes):
        image = ax.imshow(truth if i == 0 else focus[i-1]['snapshots'][0]['fields'][key], origin='lower',
                          extent=[0, 1, 0, 1], cmap='viridis', vmin=lo, vmax=hi)
        ax.set(title='Reference' if i == 0 else focus[i-1]['label'], xlabel='x/W', ylabel='y/H')
        if i:
            images.append(image)
            labels.append(ax.text(.03, .04, '', transform=ax.transAxes, color='white', fontsize=8,
                                  bbox={'facecolor': 'black', 'alpha': .5, 'pad': 2}))
    fig.colorbar(image, ax=axes, label=key, shrink=.6)
    lines = []
    if contact:
        ax_pressure = fig.add_subplot(grid[1, :])
        ax_pressure.plot(problem['cut_x'], problem['cut_truth'], 'k--', label='True pressure', lw=2)
        max_pressure = max(max(problem['cut_truth']), *[max(s['fields']['pressure_kpa']) for r in focus for s in r['snapshots']])
        for r in focus:
            line, = ax_pressure.plot(problem['cut_x'], r['snapshots'][0]['fields']['pressure_kpa'], color=r['color'], label=r['label'])
            lines.append(line)
        ax_pressure.set(xlabel='x/W', ylabel='Pressure (kPa)', ylim=(0, max_pressure*1.05))
        ax_pressure.legend(ncol=4, fontsize=8); ax_pressure.grid(alpha=.2)
    title = fig.suptitle('2D PINN convergence: training step 0')
    fig.canvas.draw(); fig.set_layout_engine(None)
    horizon = max(r['history'][-1]['step'] for r in focus)
    frames = np.unique(np.linspace(0, horizon, 41, dtype=int))

    def update(step):
        for i, run in enumerate(focus):
            idx = closest(run, step)
            fields, metrics = run['snapshots'][idx]['fields'], run['history'][idx]
            images[i].set_data(fields[key])
            labels[i].set_text(f'step {metrics["step"]}\nrel L2 {metrics["relative_l2"]:.2e}')
            if contact:
                lines[i].set_ydata(fields['pressure_kpa'])
        title.set_text(f'2D PINN convergence: training step {step} | {problem["id"]}')
        return images+labels+lines+[title]
    animation = FuncAnimation(fig, update, frames=frames, interval=140, blit=False)
    animation.save(out/f'{problem["id"]}_evolution.gif', writer=PillowWriter(fps=7), dpi=85)
    plt.close(fig)


def build_report(data, out, gif=True):
    out = Path(out)
    audit_path = out/'independent_physics_audit.json'
    if audit_path.exists():
        data['independent_audit'] = json.loads(audit_path.read_text(encoding='utf-8'))
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    template = Path(__file__).with_name('dashboard.template.html').read_text(encoding='utf-8')
    geometry = Path(__file__).with_name('assets').joinpath('contact_geometry.svg').read_text(encoding='utf-8')
    contact_case = next((p for p in data['problems'] if p['id'] == 'contact2d'), None)
    if contact_case:
        dots = ''.join(f'<circle cx="{150+640*x:.3f}" cy="{520-320*y:.3f}" r="4" fill="white" stroke="#286eb0" stroke-width="1.5"/>' for x, y in contact_case['sensors'])
        geometry = re.sub(r'<g id="sensorPositions">.*?</g>', '<g id="sensorPositions">'+dots+'</g>', geometry, flags=re.S)
    template = template.replace('__CONTACT_GEOMETRY_SVG__', geometry)
    packed = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    (out/'dashboard.html').write_text(template.replace('__RESULTS_JSON__', packed), encoding='utf-8')
    summary = []
    lines = ['# 二维 PINN 优化器实验结果', '',
             '两个空间二维 PDE × 10 种优化方法；单种子、固定超参数的合成验证。',
             '每个 PDE 内各方法共享初始化、训练采样、观测、损失及最大迭代预算。', '',
             '重点比较 Adam、Adam → L-BFGS、SOAP，同时保留其他方法。', '']
    for problem in data['problems']:
        plot_curves(problem, out)
        field_figure(problem, out)
        if problem['id'] == 'contact2d':
            contact_pressure(problem, out)
        if gif:
            export_animation(problem, out)
        lines += [f'## {problem["title"]}', '', problem['equation'], '',
                  '| 方法 | 场相对 L2 | 压力相对 L2 | 合力相对误差 | 合力 (N) | 训练秒 |',
                  '|---|---:|---:|---:|---:|---:|']
        for run in sorted(problem['runs'], key=lambda r: r['history'][-1]['relative_l2']):
            h = run['history'][-1]
            summary.append({'problem': problem['id'], 'optimizer': run['id'], 'status': run['status'], **h})
            pressure = f'{h["pressure_relative_l2"]:.3%}' if 'pressure_relative_l2' in h else '—'
            force_error = f'{h["force_relative_error"]:.3%}' if 'force_relative_error' in h else '—'
            force = f'{h["force_n"]:.4f}' if 'force_n' in h else '—'
            lines.append(f'| {run["label"]} | {h["relative_l2"]:.3%} | {pressure} | {force_error} | {force} | {h["train_seconds"]:.3f} |')
        if problem['id'] == 'contact2d':
            validation = problem['reference_validation']
            lines += ['', f'真实合力：{problem["force_true_n"]:.6f} N。',
                      f'参考 FEM 粗细网格位移差异：{validation["coarse_fine_displacement_relative"]:.4%}。',
                      f'参考力平衡相对误差：{validation["fine"]["force_balance_relative"]:.3e}。',
                      '已知接触区域、材料和几何；无摩擦法向载荷反演，不包含未知接触分离的互补条件。']
        lines.append('')
    if data.get('independent_audit'):
        lines += ['## 独立物理检查', '',
                  '另在 1,024 个未参与训练的 Sobol 点计算 PDE 残差，并检查预测顶部压力合力与预测底部反力。',
                  '训练 loss 低不保证整个区域满足 PDE；应同时查看独立残差和力平衡。', '',
                  '| PDE | 方法 | 独立 PDE MSE | 预测力平衡相对误差 |',
                  '|---|---|---:|---:|']
        for item in data['independent_audit']:
            balance = f'{item["predicted_force_balance_relative"]:.3%}' if 'predicted_force_balance_relative' in item else '—'
            lines.append(f'| {item["problem"]} | {item["optimizer"]} | {item["independent_pde_mse"]:.4e} | {balance} |')
        lines += ['', '完整独立本构残差与反力数据见 independent_physics_audit.json。']
    (out/'experiment_report.md').write_text('\n'.join(lines), encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    keys = list(dict.fromkeys(k for row in summary for k in row))
    with (out/'summary.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('--no-gif', action='store_true')
    args = parser.parse_args()
    build_report(json.loads(args.results.read_text(encoding='utf-8')), args.results.parent, gif=not args.no_gif)
