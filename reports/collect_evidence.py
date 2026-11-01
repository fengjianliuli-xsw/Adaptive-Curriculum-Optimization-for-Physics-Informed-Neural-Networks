"""Read-only experiment inventory. Only writes beneath F:/RA/reports."""
from pathlib import Path
import ast,csv,hashlib,json,math,subprocess,sys

OUT=Path(__file__).resolve().parent
ROOTS=[Path(r'C:\Users\ASUS\Documents\ChatGPT\RA\pinn_optimizer_lab'),Path('F:/RA/pinn_2d_optimizer_lab'),Path('F:/RA/hertz_cylinder_lab'),Path('F:/RA/pinn_scheduler_lab')]
NAMES={'adam','adam_lbfgs','momentum','rmsprop','lbfgs','lm','multiadam','pcgrad','config','soap','dcgd','result'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def num(value):return '尚无记录' if value is None else f'{value:.7g}' if isinstance(value,(float,int)) else str(value)


def main():
    OUT.mkdir(exist_ok=True);inventory=[];suites=[];sources={};configs={}
    for root in ROOTS:
        for source in root.glob('*.py'):sources[str(source)]=sha(source)
        for suite in sorted((root/'runs').iterdir()):
            if not suite.is_dir():continue
            found=[]
            for path in sorted(suite.rglob('*.json')):
                if path.stem not in NAMES:continue
                r=read(path);h=r.get('history',[])
                if not h:continue # config.json may be a configuration OR the ConFIG result.
                f=h[-1];source=dict(path=str(path),sha256=sha(path));sources[str(path)]=source['sha256']
                row=dict(project=root.name,suite=suite.name,problem=r.get('kind',path.parent.name if path.parent!=suite else 'heat1d' if root==ROOTS[0] else 'hertz'),
                         method=r.get('plan',r.get('id',path.stem)),seed=r.get('seed'),status=r.get('status'),step=f['step'],
                         training_seconds=f.get('train_seconds',r.get('charged_seconds')),wall_seconds=r.get('total_wall_seconds',r.get('wall_seconds')),
                         loss_total=f.get('loss_total',f.get('train_total')),relative_l2=f.get('relative_l2'),pressure_relative_l2=f.get('pressure_relative_l2'),
                         final=f,initial_hash=r.get('initial_hash'),sampling_hash=r.get('sampling_hash',r.get('sample_hash')),
                         residual_evals=f.get('residual_evals'),gradient_evals=f.get('gradient_evals'),lm_rejected_trials=r.get('lm_rejected_trials'),
                         costs=r.get('costs'),source=source)
                inventory.append(row);found.append(row)
            # Historical pilots are not filled with current dataclass defaults.
            cp=suite/'run_config.json'
            if not cp.exists():cp=suite/'config.json'
            config=None
            if cp.exists():
                x=read(cp)
                if not x.get('history'):config=x
            if config is None and (suite/'manifest.json').exists():
                cp=suite/'manifest.json';config=read(cp).get('config')
            if config is not None:
                configs[str(suite)]={'path':str(cp),'values':config};sources[str(cp)]=sha(cp)
            suites.append(dict(project=root.name,suite=suite.name,runs=len(found),statuses=sorted({str(r['status']) for r in found}),config_path=str(cp) if config is not None else None))
    verification=[]
    for root,script,suite in [(ROOTS[0],ROOTS[0]/'tests/validate_run.py','demo'),(ROOTS[1],ROOTS[1]/'tests/validate_run.py','baseline')]:
        proc=subprocess.run([sys.executable,'-B',str(script),str(root/'runs'/suite)],capture_output=True,text=True,encoding='utf-8',errors='replace')
        verification.append(dict(project=root.name,command=proc.args,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
    # Audit the existing Hertz checkpoints without invoking validate.py, which overwrites old audit files.
    code=r'''
import json,sys,numpy as np,torch
from pathlib import Path
root=Path(sys.argv[1]);directory=root/'runs/kkt_all11';sys.path.insert(0,str(root))
from physics import Config,Problem,flat_parameters,fingerprint
d=json.loads((directory/'results.json').read_text(encoding='utf-8'));ref=json.loads((directory/'fem_reference.json').read_text())
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
p=Problem(Config(**d['config']),ref);m=p.make_model()
for r in d['runs']:
 m.load_state_dict(torch.load(directory/(r['id']+'_final.pt'),weights_only=True))
 with torch.no_grad():
  loss=p.losses(m).numpy();metrics,_=p.evaluate(m)
 np.testing.assert_allclose(loss,[r['history'][-1]['loss_'+key] for key in p.loss_names],rtol=1e-10,atol=1e-12)
 for key,value in metrics.items():np.testing.assert_allclose(value,r['history'][-1][key],rtol=1e-10,atol=1e-12)
 assert r['sampling_hash']==p.sample_hash
 assert r['initial_hash']==d['initial_hash']
print('Read-only Hertz audit: 11 checkpoint losses/metrics and shared recorded hashes verified.')
'''
    proc=subprocess.run([sys.executable,'-B','-c',code,str(ROOTS[2])],capture_output=True,text=True,encoding='utf-8',errors='replace')
    verification.append(dict(project=ROOTS[2].name,command='inline read-only checkpoint audit',returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
    # Existing scheduler validator writes validation.json; use an isolated read-only audit instead.
    code=r'''
import json,sys,torch,numpy as np
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from physics import Config,Problem,parameters_hash
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
for file in (root/'runs/scheduler60').glob('*/result.json'):
 r=json.loads(file.read_text(encoding='utf-8'));p=Problem(r['kind'],Config(**r['config']),r['seed']);m=p.make_model()
 initial=torch.load(file.parent/'initial.pt',weights_only=False);m.load_state_dict(initial['model']);assert parameters_hash(m)==r['initial_hash']
 saved=torch.load(file.parent/'final.pt',weights_only=False);m.load_state_dict(saved['model']);metrics,_=p.evaluate(m);probe=p.probe(m)
 for key,value in metrics.items():np.testing.assert_allclose(value,r['final'][key],rtol=1e-10,atol=1e-12)
 for key in ['pde_loss','bc_loss','e_bc','e_pde']:np.testing.assert_allclose(probe[key],r['final'][key],rtol=1e-10,atol=1e-12)
 assert p.sample_hash==r['sample_hash']
print('Read-only scheduler audit: 10 checkpoint metrics/probes and initialization/sampling hashes verified.')
'''
    proc=subprocess.run([sys.executable,'-B','-c',code,str(ROOTS[3])],capture_output=True,text=True,encoding='utf-8',errors='replace')
    verification.append(dict(project=ROOTS[3].name,command='inline read-only checkpoint audit',returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
    mainrows=[r for r in inventory if r['suite'] in ['demo','baseline','kkt_all11','scheduler60']]
    # Compute checkpoint-observed time-to-target, without interpolation or extrapolation.
    targets=[]
    for root,suite,problem in [(ROOTS[0],'demo','heat1d'),(ROOTS[1],'baseline','heat2d')]:
        for method in ['adam','lbfgs','adam_lbfgs','soap','lm']:
            file=root/'runs'/suite/(problem if problem=='heat2d' else '')/(method+'.json');r=read(file)
            for threshold in [.01,.001]:
                first=next((x for x in r['history'] if x['relative_l2']<=threshold),None)
                targets.append(dict(problem=problem,method=method,threshold=threshold,time_seconds=first['train_seconds'] if first else None,source=str(file),meaning='first recorded checkpoint; excludes diagnostics; not sustained attainment'))
    payload=dict(roots=list(map(str,ROOTS)),inventory=inventory,suites=suites,configs=configs,source_sha256=sources,verification=verification,time_to_target=targets)
    (OUT/'experiment_inventory.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (OUT/'verified_metrics.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        keys=['project','suite','problem','method','status','step','training_seconds','wall_seconds','loss_total','relative_l2','pressure_relative_l2','residual_evals','gradient_evals','lm_rejected_trials']
        writer=csv.DictWriter(handle,keys,extrasaction='ignore');writer.writeheader();writer.writerows(inventory)
    lines=['# 实验证据与可信度记录','', '本文件来自只读代码/日志盘点和最终权重复算；本任务没有运行新训练、没有修改实验源码或覆盖实验结果。报告只摘取主实验。辅助文件 experiment_inventory.json 保留全部已找到的逐方法终点、实际损失分项、源路径/SHA256、配置与复核输出；verified_metrics.csv 便于筛选。','',
    '## 1. 范围与实验完成状态','', '|项目|运行目录|方法记录数|记录状态|','|---|---|---:|---|']
    for x in suites:lines.append(f"|{x['project']}|{x['suite']}|{x['runs']}|{', '.join(x['statuses']) or '仅参考解/无逐方法训练记录'}|")
    lines += ['', 'smoke 为流程检查，pilot 为不同建模/采样/预算的探索，不与主实验混排。contact_seed_check 名称虽含 seed，实际配置 seed=42、initial_vertical_bias=-2，不是第二个随机种子。pilot30/pilot60 是调度实验的基线预试，A 被原样复用，不能计为独立重复。当前未找到未结束的后台训练或额外无日志算法；未实现的方法见下。','',
    '## 2. 主实验配置、PDE 与边界','',
    '所有主实验均为 CPU 单线程、float64；没有 GPU 训练/显存记录，不填 0 MB。评估误差为固定离散网格上的相对范数，不等于连续 L² 的严格积分。相同初始化/采样只在同一 PDE 的方法内成立。','',
    '|基准|PDE / 边界条件|网络与参数量|训练点/残差数|损失与评价|','|---|---|---|---|---|',
    '|1D heat / demo|u_t=0.1u_xx；x,t∈[0,1]；两端零 Dirichlet；u(x,0)=sin(πx)+0.3sin(3πx)|2-24-24-1 tanh，697 参数|内点128，IC48，BC每端48；残差272|三项 MSE 等权；41×121 时空网格；另512点 PDE审计|',
    '|2D heat / baseline|-ΔT=q；单位方域，零 Dirichlet；T*=sin(πx)sin(πy)+0.2sin(3πx)sin(2πy)|2-24-24-1 tanh，697 参数|内点128，边界每边16；残差192|Lf=mean(((-ΔT-q)/(2π²))²)，Lbc=mean(T²)，等权；31×31网格|',
    '|2D inverse elasticity / baseline|40×20 mm 矩形，平面应力，E=1 MPa、ν=0.3、厚10 mm；底部固定位移，侧面自由；顶面未知法向压力、切向零；48个无噪声位移观测|场网络2-24-24-5，压力头1-12-12-1，共990参数；底部位移硬满足；已知压力支撑[0.2,0.8]与非负窗口|内点64；顶部与两侧各16；观测48；聚合残差512|平衡、本构、牵引、观测 MSE 权重[1,1,1,10]；P2 FEM参考；不是未知接触区/KKT接触问题|',
    '|Hertz / kkt_all11|四分之一圆柱压刚性平面，平面应变、无摩擦；R=10mm，E=200MPa，ν=.3，顶载.5MPa，厚10mm；载荷已知100N，FB平滑KKT约束|4-24-24-24-2 tanh，1370参数；偶对称/局部输入特征；应力由位移导出|内点256+弧128；顶部32，接触积分64，全局积分400，虚功测试12；当前代码残差1101|平衡/顶牵引/无摩擦/FB/载荷/虚功权重[1,10,1,10,1,10]；FEM位移与压力；压力只在x/R≤.3比较|',
    '|scheduler60 heat / NS|热传导同上；NS为Re=20单位方域Kovasznay稳态不可压缩流，四边速度Dirichlet，单点p(0,0)=0|热2-32-32-32-1：2241参数；NS输出(u,v,p)，2307参数|固定内点256，BC每边32；探针内点192+BC每边64；额外独立1024点残差审计|Adam lr=.001；Lf+Lbc（D动态加权）；41×41网格；e_bc为按物理尺度归一化后各分量RMSE的最大值|','',
    'NS 解析解：λ=Re/2−sqrt(Re²/4+4π²)，u=1−exp(λx)cos(2πy)，v=λ exp(λx)sin(2πy)/(2π)，p=(1−exp(2λx))/2。Lf 为两动量与连续性 MSE 之和；Lbc 为 u/v 边界 MSE 加单点压力规范。热的 q 由解析解取 -ΔT；内部真解不参与训练。','',
    '## 3. 优化器与超参数','',
    '- Adam：lr=0.001，默认β=(0.9,0.999)、ε=1e-8，无weight decay。AdamW无实现/运行记录。SGD+Momentum：lr=.01、momentum=.9；RMSProp：lr=.001、alpha=.99。',
    '- L-BFGS：lr=1，max_iter=1，max_eval=12，history_size=50，strong_wolfe，tol_grad=1e-10，tol_change=1e-12。外层1步可能多次closure，不等于一次梯度评估。Adam→L-BFGS在1200步任务中第601步切换；Hertz2400步配置在第1201步切换。',
    '- LM是阻尼Gauss–Newton合并标签，不是分别测试GN与LM：mu初始1e-2；jacrev(chunk_size=64)显式完整Jacobian；残差数小于参数数时解(JJᵀ+mu I)z=r，更新-Jᵀz。每轮最多10次试探；实际下降且gain ratio>1e-4接受，ratio>.75时mu×.3，否则不变；拒绝时mu×10，范围[1e-12,1e16]；max|Jᵀr|<1e-10可停止。',
    '- SOAP为教学性自实现核心：lr=.003，β1=β2=.95，矩阵相关统计衰减.95，每10步eigh更新基，旋转二阶矩，ε=1e-8；无weight decay/grafting。不是官方完整实现的严格复现。Shampoo单独实现无记录。',
    '- MultiAdam自实现按任务组分别累积统计，β1=β2=.99，lr=.001；PCGrad+GD与ConFIG+GD通常lr=.02；Hertz的ConFIG/DCGD控制组都lr=.001。二者非退化两组方向在该实现下等价，不能将近乎相同轨迹视为两个独立证据。Hertz ConFIG=.02失败保存在failed_attempts.json，未隐藏。',
    '- BFGS、纯GN、Newton-CG、NNCG、单独Shampoo：当前仓库未找到独立训练记录，仅列入调研/计划。',
    '- ReLoBRaLo仅在scheduler60：α=.999、Bernoulli概率=.999、temperature=.1，初始权重各1；PDE/BC两组，权重和2；独立RNG。E使用τbc=τf=.03，迟滞±20%，最小驻留2次检查，每50步检查；BC停滞与冲突分支详见schedulers.py。E未用GN/JVP；并非已实现的曲率调度器。','',
    '## 4. 主实验终点数据','',
    '训练秒在前三项目为优化器更新计时（不含定期诊断）；scheduler60计费秒包含优化、诊断、策略。wall为训练循环墙钟，边界因项目而异，不含完整初始化/报告管线；禁止直接跨项目当作等预算排名。1D/2D优化器对照是1200外层步，不是等时间；Hertz有2400步与120秒软上限；scheduler为等60秒计费预算。所有表格是最终记录，不按真解误差挑最优。','',
    '|项目/基准|方法|状态|最终步|Loss合计|相对L2|压力相对L2|训练/计费秒|循环wall秒|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in mainrows:lines.append('|'+ '|'.join([r['project']+'/'+r['problem'],r['method'],r['status'],num(r['step']),num(r['loss_total']),num(r['relative_l2']),num(r['pressure_relative_l2']),num(r['training_seconds']),num(r['wall_seconds'])])+'|')
    lines += ['', '## 5. GN/LM耗时：可证实与不可证实','', '|问题|LM/SOAP训练时间比|LM/L-BFGS训练时间比|Jacobian维数（由代码/配置推得）|LM拒绝试探总数|','|---|---:|---:|---|---:|']
    for project,suite,problem,m,n in [(ROOTS[0].name,'demo','heat1d',272,697),(ROOTS[1].name,'baseline','heat2d',192,697),(ROOTS[1].name,'baseline','contact2d',512,990)]:
        group={r['method']:r for r in mainrows if r['project']==project and r['suite']==suite and r['problem']==problem}
        lines.append(f"|{problem}|{group['lm']['training_seconds']/group['soap']['training_seconds']:.3f}×|{group['lm']['training_seconds']/group['lbfgs']['training_seconds']:.3f}×|{m}×{n}，使用{m}×{m}对偶系统|{group['lm']['lm_rejected_trials']}|")
    lines += ['', '上述比值只反映单种子、相同外层步终点的代价；不是达到同一精度的耗时比。LMJacobian次数由执行路径推断，日志gradient_evals只把一次Jacobian记作一次，并不能与普通backward公平等价。代码可确认每轮完整Jacobian、Gram构造、稠密solve、候选残差与阻尼重试；但没有Jacobian/Gram/solve/候选/诊断分项计时和内存峰值。**因此无法量化哪一环节主导耗时；需要后续profile，不能凭复杂度替代实测。**','',
    '首次记录达到阈值（只读日志计算；不插值、不假设持续达到；不含诊断）：','', '|问题|方法|阈值relL2|首次记录训练秒|','|---|---|---:|---:|']
    for r in targets:lines.append(f"|{r['problem']}|{r['method']}|{r['threshold']}|{num(r['time_seconds']) if r['time_seconds'] is not None else '未达到（右删失）'}|")
    lines += ['', '## 6. 重要局限/负结果','',
    '- 二维反演 LM 训练平衡MSE=1.190988e-12，而1024留出点平衡MSE=1.0023e-2（精确值见原audit JSON）；与SOAP的1.3905e-4、L-BFGS的1.9057e-4相比，存在强烈泛化/配点覆盖问题。压力误差低不能证明域内平衡已学好。',
    '- Hertz所有有限主结果压力相对FEM误差约94.0%–98.4%；PCGrad另为数值失稳、非有限更新回退，不能当正常排名。部分方法loss下降不等于接触解成功；主实验已知载荷，不能称为合力反演。',
    '- scheduler热：E=0.1976%，A=.5348%，D=.06445%；NS速度：E=22.0347%，A=.6389%，D=.6819%。E在NS实际动作变化95次，末e_bc=.24>τbc=.03。最小驻留可能压过异常边界恢复，且检查有间隔；失败机制尚未经消融确定。',
    '- 同PDE方法内初始/采样哈希一致；方法学习率不同、未做等量超参数搜索，单种子不能给普遍优劣/统计显著结论。1D numpy记录2.4.6，而F盘项目1.26.4；不跨环境直接排名。',
    '- 边界反弹提供研究动机，但现有数据没有GN方向曲率预测值、JVP耗时或曲率调度实验，不能声称H1/H2/H3已获验证。','',
    '## 7. 本次只读复核','']
    for v in verification:lines += [f"- {v['project']}：returncode={v['returncode']}；{v['stdout'].strip() or v['stderr'][-1200:]}"]
    lines += ['', '已有 independent_physics_audit.json / independent_audit.json 是历史审计文件，本次没有覆盖；本次复核只读取权重、重算训练损失与现有指标。通过复算只说明记录与模型一致，并不构成算法有效性或物理正确性的额外证明。','',
    '## 8. 可追溯路径与配置','']
    for root in ROOTS:lines += [f"### {root}",'',f"源码：`{root/'physics.py'}`、`{root/'optimizers.py' if root!=ROOTS[3] else root/'schedulers.py'}`、`{root/'train.py'}`。"]
    for path,d in configs.items():
        lines += ['',f"#### {path}",f"配置源：`{d['path']}`",'```json',json.dumps(d['values'],ensure_ascii=False,indent=2),'```']
    lines += ['', '逐方法源JSON路径、SHA256、每项最终损失和完整配置均见 experiment_inventory.json；图的数据来自 baseline/heat2d/*_history.csv 与 scheduler60/ns2d_*/result.json，并由build_research_note.py读取生成。','',
    '## 9. 缺失数据及发送前复核','',
    '尚无记录：GPU显存、CPU内存峰值、GN/LM分段profile、严格等端到端wall预算的全部对照、多种子统计、GN曲率预测与JVP运行数据；完整B/C最优分段调参未做。旧pilot不完整配置不补当前默认值。源文件可能在历史运行后被更改；provenance哈希可用于核查，不能将所有pilot视为当前源码的可复现实验。',
    '建议人工复核：署名/日期与导师偏好的术语；确认只引用自己的合成实验且可分享；确认愿意报告反演平衡泛化差和Hertz失败；核对未来实验预算与主要误差阈值；文献对应版本与正式题名；全文不得把构想写作原创或已验证。']
    (OUT/'experiment_evidence.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Inventory:',len(inventory),'method runs;',len(suites),'directories. Read-only audits:',[(v['project'],v['returncode']) for v in verification])
    if any(v['returncode'] for v in verification):print('Audit warnings retained in experiment_inventory.json; do not claim all checks passed.')


if __name__=='__main__':main()
