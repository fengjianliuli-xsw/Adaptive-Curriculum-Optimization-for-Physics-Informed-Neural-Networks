# 实验证据与可信度记录

本文件来自只读代码/日志盘点和最终权重复算；本任务没有运行新训练、没有修改实验源码或覆盖实验结果。报告只摘取主实验。辅助文件 experiment_inventory.json 保留全部已找到的逐方法终点、实际损失分项、源路径/SHA256、配置与复核输出；verified_metrics.csv 便于筛选。

## 1. 范围与实验完成状态

|项目|运行目录|方法记录数|记录状态|
|---|---|---:|---|
|pinn_optimizer_lab|demo|10|completed|
|pinn_optimizer_lab|smoke|10|completed|
|pinn_2d_optimizer_lab|baseline|20|budget_completed|
|pinn_2d_optimizer_lab|smoke|20|budget_completed|
|hertz_cylinder_lab|boundary_pilot|2|budget_completed|
|hertz_cylinder_lab|contact_seed_check|1|budget_completed|
|hertz_cylinder_lab|dcgd_config_control|2|budget_completed|
|hertz_cylinder_lab|dense_pilot|2|budget_completed|
|hertz_cylinder_lab|displacement_pilot|2|budget_completed|
|hertz_cylinder_lab|kkt_all11|11|budget_completed, nonfinite_update_reverted, training_time_budget|
|hertz_cylinder_lab|physics_pilot|2|budget_completed|
|hertz_cylinder_lab|pilot|3|budget_completed|
|hertz_cylinder_lab|reference|0|仅参考解/无逐方法训练记录|
|hertz_cylinder_lab|weak_pilot|2|budget_completed|
|pinn_scheduler_lab|pilot30|2|complete|
|pinn_scheduler_lab|pilot60|2|complete|
|pinn_scheduler_lab|scheduler60|10|complete|

smoke 为流程检查，pilot 为不同建模/采样/预算的探索，不与主实验混排。contact_seed_check 名称虽含 seed，实际配置 seed=42、initial_vertical_bias=-2，不是第二个随机种子。pilot30/pilot60 是调度实验的基线预试，A 被原样复用，不能计为独立重复。当前未找到未结束的后台训练或额外无日志算法；未实现的方法见下。

## 2. 主实验配置、PDE 与边界

所有主实验均为 CPU 单线程、float64；没有 GPU 训练/显存记录，不填 0 MB。评估误差为固定离散网格上的相对范数，不等于连续 L² 的严格积分。相同初始化/采样只在同一 PDE 的方法内成立。

|基准|PDE / 边界条件|网络与参数量|训练点/残差数|损失与评价|
|---|---|---|---|---|
|1D heat / demo|u_t=0.1u_xx；x,t∈[0,1]；两端零 Dirichlet；u(x,0)=sin(πx)+0.3sin(3πx)|2-24-24-1 tanh，697 参数|内点128，IC48，BC每端48；残差272|三项 MSE 等权；41×121 时空网格；另512点 PDE审计|
|2D heat / baseline|-ΔT=q；单位方域，零 Dirichlet；T*=sin(πx)sin(πy)+0.2sin(3πx)sin(2πy)|2-24-24-1 tanh，697 参数|内点128，边界每边16；残差192|Lf=mean(((-ΔT-q)/(2π²))²)，Lbc=mean(T²)，等权；31×31网格|
|2D inverse elasticity / baseline|40×20 mm 矩形，平面应力，E=1 MPa、ν=0.3、厚10 mm；底部固定位移，侧面自由；顶面未知法向压力、切向零；48个无噪声位移观测|场网络2-24-24-5，压力头1-12-12-1，共990参数；底部位移硬满足；已知压力支撑[0.2,0.8]与非负窗口|内点64；顶部与两侧各16；观测48；聚合残差512|平衡、本构、牵引、观测 MSE 权重[1,1,1,10]；P2 FEM参考；不是未知接触区/KKT接触问题|
|Hertz / kkt_all11|四分之一圆柱压刚性平面，平面应变、无摩擦；R=10mm，E=200MPa，ν=.3，顶载.5MPa，厚10mm；载荷已知100N，FB平滑KKT约束|4-24-24-24-2 tanh，1370参数；偶对称/局部输入特征；应力由位移导出|内点256+弧128；顶部32，接触积分64，全局积分400，虚功测试12；当前代码残差1101|平衡/顶牵引/无摩擦/FB/载荷/虚功权重[1,10,1,10,1,10]；FEM位移与压力；压力只在x/R≤.3比较|
|scheduler60 heat / NS|热传导同上；NS为Re=20单位方域Kovasznay稳态不可压缩流，四边速度Dirichlet，单点p(0,0)=0|热2-32-32-32-1：2241参数；NS输出(u,v,p)，2307参数|固定内点256，BC每边32；探针内点192+BC每边64；额外独立1024点残差审计|Adam lr=.001；Lf+Lbc（D动态加权）；41×41网格；e_bc为按物理尺度归一化后各分量RMSE的最大值|

NS 解析解：λ=Re/2−sqrt(Re²/4+4π²)，u=1−exp(λx)cos(2πy)，v=λ exp(λx)sin(2πy)/(2π)，p=(1−exp(2λx))/2。Lf 为两动量与连续性 MSE 之和；Lbc 为 u/v 边界 MSE 加单点压力规范。热的 q 由解析解取 -ΔT；内部真解不参与训练。

## 3. 优化器与超参数

- Adam：lr=0.001，默认β=(0.9,0.999)、ε=1e-8，无weight decay。AdamW无实现/运行记录。SGD+Momentum：lr=.01、momentum=.9；RMSProp：lr=.001、alpha=.99。
- L-BFGS：lr=1，max_iter=1，max_eval=12，history_size=50，strong_wolfe，tol_grad=1e-10，tol_change=1e-12。外层1步可能多次closure，不等于一次梯度评估。Adam→L-BFGS在1200步任务中第601步切换；Hertz2400步配置在第1201步切换。
- LM是阻尼Gauss–Newton合并标签，不是分别测试GN与LM：mu初始1e-2；jacrev(chunk_size=64)显式完整Jacobian；残差数小于参数数时解(JJᵀ+mu I)z=r，更新-Jᵀz。每轮最多10次试探；实际下降且gain ratio>1e-4接受，ratio>.75时mu×.3，否则不变；拒绝时mu×10，范围[1e-12,1e16]；max|Jᵀr|<1e-10可停止。
- SOAP为教学性自实现核心：lr=.003，β1=β2=.95，矩阵相关统计衰减.95，每10步eigh更新基，旋转二阶矩，ε=1e-8；无weight decay/grafting。不是官方完整实现的严格复现。Shampoo单独实现无记录。
- MultiAdam自实现按任务组分别累积统计，β1=β2=.99，lr=.001；PCGrad+GD与ConFIG+GD通常lr=.02；Hertz的ConFIG/DCGD控制组都lr=.001。二者非退化两组方向在该实现下等价，不能将近乎相同轨迹视为两个独立证据。Hertz ConFIG=.02失败保存在failed_attempts.json，未隐藏。
- BFGS、纯GN、Newton-CG、NNCG、单独Shampoo：当前仓库未找到独立训练记录，仅列入调研/计划。
- ReLoBRaLo仅在scheduler60：α=.999、Bernoulli概率=.999、temperature=.1，初始权重各1；PDE/BC两组，权重和2；独立RNG。E使用τbc=τf=.03，迟滞±20%，最小驻留2次检查，每50步检查；BC停滞与冲突分支详见schedulers.py。E未用GN/JVP；并非已实现的曲率调度器。

## 4. 主实验终点数据

训练秒在前三项目为优化器更新计时（不含定期诊断）；scheduler60计费秒包含优化、诊断、策略。wall为训练循环墙钟，边界因项目而异，不含完整初始化/报告管线；禁止直接跨项目当作等预算排名。1D/2D优化器对照是1200外层步，不是等时间；Hertz有2400步与120秒软上限；scheduler为等60秒计费预算。所有表格是最终记录，不按真解误差挑最优。

|项目/基准|方法|状态|最终步|Loss合计|相对L2|压力相对L2|训练/计费秒|循环wall秒|
|---|---|---|---:|---:|---:|---:|---:|---:|
|pinn_optimizer_lab/heat1d|adam|completed|1200|0.02661341|0.08327921|尚无记录|4.667106|6.486736|
|pinn_optimizer_lab/heat1d|adam_lbfgs|completed|1200|3.436542e-05|0.003377754|尚无记录|7.952166|9.707062|
|pinn_optimizer_lab/heat1d|config|completed|1200|0.04173589|0.111605|尚无记录|9.391454|11.23696|
|pinn_optimizer_lab/heat1d|lbfgs|completed|1200|9.551834e-06|0.001223681|尚无记录|11.76165|13.53346|
|pinn_optimizer_lab/heat1d|lm|completed|1200|1.000173e-10|0.0001531534|尚无记录|182.8422|184.524|
|pinn_optimizer_lab/heat1d|momentum|completed|1200|0.02764363|0.0868763|尚无记录|4.3314|6.287955|
|pinn_optimizer_lab/heat1d|multiadam|completed|1200|0.03944715|0.105745|尚无记录|6.702019|8.397283|
|pinn_optimizer_lab/heat1d|pcgrad|completed|1200|0.03015618|0.1009373|尚无记录|9.952069|11.90322|
|pinn_optimizer_lab/heat1d|rmsprop|completed|1200|0.02954583|0.1069948|尚无记录|4.975385|7.02876|
|pinn_optimizer_lab/heat1d|soap|completed|1200|1.558764e-05|0.002641933|尚无记录|5.027335|6.817093|
|pinn_2d_optimizer_lab/contact2d|adam|budget_completed|1200|0.01643549|0.4546708|0.6928895|7.929941|9.705982|
|pinn_2d_optimizer_lab/contact2d|adam_lbfgs|budget_completed|1200|0.001215957|0.04629492|0.134484|13.60463|15.44091|
|pinn_2d_optimizer_lab/contact2d|config|budget_completed|1200|0.02317005|0.5308654|0.5837784|16.52203|17.98195|
|pinn_2d_optimizer_lab/contact2d|lbfgs|budget_completed|1200|0.0007329604|0.03579732|0.09846662|17.62467|19.28055|
|pinn_2d_optimizer_lab/contact2d|lm|budget_completed|1200|8.210498e-10|0.008362375|0.01498073|174.7405|176.407|
|pinn_2d_optimizer_lab/contact2d|momentum|budget_completed|1200|0.01401344|0.3712708|0.6716369|3.313846|4.252967|
|pinn_2d_optimizer_lab/contact2d|multiadam|budget_completed|1200|0.03315834|0.5297603|0.8490186|11.56357|13.17293|
|pinn_2d_optimizer_lab/contact2d|pcgrad|budget_completed|1200|0.01883365|0.4488072|0.7361734|17.79476|19.29397|
|pinn_2d_optimizer_lab/contact2d|rmsprop|budget_completed|1200|0.02373484|0.4607677|0.7126275|7.848624|9.605154|
|pinn_2d_optimizer_lab/contact2d|soap|budget_completed|1200|0.0004094001|0.02599341|0.07205282|8.633242|10.22712|
|pinn_2d_optimizer_lab/heat2d|adam|budget_completed|1200|0.004176491|0.02551629|尚无记录|4.353997|5.134927|
|pinn_2d_optimizer_lab/heat2d|adam_lbfgs|budget_completed|1200|2.018651e-05|0.00207307|尚无记录|8.893049|9.80361|
|pinn_2d_optimizer_lab/heat2d|config|budget_completed|1200|0.5030874|0.6772391|尚无记录|7.540791|8.361186|
|pinn_2d_optimizer_lab/heat2d|lbfgs|budget_completed|1200|1.092654e-05|0.001023778|尚无记录|12.597|13.46862|
|pinn_2d_optimizer_lab/heat2d|lm|budget_completed|1200|8.454311e-11|3.108527e-05|尚无记录|96.36843|97.21919|
|pinn_2d_optimizer_lab/heat2d|momentum|budget_completed|1200|0.03027372|0.06926394|尚无记录|4.738481|5.662957|
|pinn_2d_optimizer_lab/heat2d|multiadam|budget_completed|1200|0.158219|0.2408414|尚无记录|8.116392|9.020464|
|pinn_2d_optimizer_lab/heat2d|pcgrad|budget_completed|1200|0.09772611|0.09108227|尚无记录|7.467415|8.315639|
|pinn_2d_optimizer_lab/heat2d|rmsprop|budget_completed|1200|0.02125785|0.05683902|尚无记录|5.009973|5.897674|
|pinn_2d_optimizer_lab/heat2d|soap|budget_completed|1200|1.096121e-05|0.001955977|尚无记录|6.161968|7.128504|
|hertz_cylinder_lab/hertz|adam|budget_completed|2400|0.01357441|0.8868426|0.9548474|28.1364|29.03916|
|hertz_cylinder_lab/hertz|adam_lbfgs|budget_completed|2400|0.008028816|1.042827|0.9833447|45.68613|46.59016|
|hertz_cylinder_lab/hertz|config|budget_completed|2400|0.02279768|1.043087|0.9623083|116.3737|153.7297|
|hertz_cylinder_lab/hertz|dcgd|budget_completed|2400|0.0227975|1.043086|0.9623061|72.89652|74.23476|
|hertz_cylinder_lab/hertz|lbfgs|training_time_budget|1687|0.007880732|1.05006|0.9839713|120.1194|145.6869|
|hertz_cylinder_lab/hertz|lm|training_time_budget|27|0.008049276|1.032369|0.9789468|120.855|121.3169|
|hertz_cylinder_lab/hertz|momentum|budget_completed|2400|3.671758|5.97402e+09|0.9500292|77.08983|113.7701|
|hertz_cylinder_lab/hertz|multiadam|budget_completed|2400|0.01200731|0.9042646|0.9595881|115.3691|152.6686|
|hertz_cylinder_lab/hertz|pcgrad|nonfinite_update_reverted|120|8.597898e+285|5.604149e+141|3.122798e+139|11.30829|13.91775|
|hertz_cylinder_lab/hertz|rmsprop|budget_completed|2400|0.04560737|0.8775575|0.9401523|71.76849|105.0471|
|hertz_cylinder_lab/hertz|soap|budget_completed|2400|0.007738302|1.03174|0.9728788|31.49015|32.44223|
|pinn_scheduler_lab/heat2d|A|complete|8053|1.259305e-05|0.005347608|尚无记录|60.01146|61.12721|
|pinn_scheduler_lab/heat2d|B|complete|14995|0.0001771668|0.007895128|尚无记录|60.01228|62.02479|
|pinn_scheduler_lab/heat2d|C|complete|14968|0.0001743111|0.007942753|尚无记录|60.01312|61.97859|
|pinn_scheduler_lab/heat2d|D|complete|8520|5.481544e-06|0.0006445293|尚无记录|60.00974|61.05129|
|pinn_scheduler_lab/heat2d|E|complete|8656|1.798951e-05|0.001976041|尚无记录|60.01482|61.23233|
|pinn_scheduler_lab/ns2d|A|complete|6200|0.0002490602|0.006388687|0.08058825|60.01054|60.9821|
|pinn_scheduler_lab/ns2d|B|complete|11148|0.002201252|0.0391468|0.4890632|60.01799|61.72153|
|pinn_scheduler_lab/ns2d|C|complete|11231|0.003961692|0.03983493|0.6265795|60.01655|61.71273|
|pinn_scheduler_lab/ns2d|D|complete|6431|0.0002470169|0.006819281|0.06204912|60.01305|60.98924|
|pinn_scheduler_lab/ns2d|E|complete|12450|0.06355938|0.2203467|0.5641171|60.01875|61.79864|

## 5. GN/LM耗时：可证实与不可证实

|问题|LM/SOAP训练时间比|LM/L-BFGS训练时间比|Jacobian维数（由代码/配置推得）|LM拒绝试探总数|
|---|---:|---:|---|---:|
|heat1d|36.370×|15.546×|272×697，使用272×272对偶系统|85|
|heat2d|15.639×|7.650×|192×697，使用192×192对偶系统|98|
|contact2d|20.240×|9.915×|512×990，使用512×512对偶系统|101|

上述比值只反映单种子、相同外层步终点的代价；不是达到同一精度的耗时比。LMJacobian次数由执行路径推断，日志gradient_evals只把一次Jacobian记作一次，并不能与普通backward公平等价。代码可确认每轮完整Jacobian、Gram构造、稠密solve、候选残差与阻尼重试；但没有Jacobian/Gram/solve/候选/诊断分项计时和内存峰值。**因此无法量化哪一环节主导耗时；需要后续profile，不能凭复杂度替代实测。**

首次记录达到阈值（只读日志计算；不插值、不假设持续达到；不含诊断）：

|问题|方法|阈值relL2|首次记录训练秒|
|---|---|---:|---:|
|heat1d|adam|0.01|未达到（右删失）|
|heat1d|adam|0.001|未达到（右删失）|
|heat1d|lbfgs|0.01|2.970444|
|heat1d|lbfgs|0.001|未达到（右删失）|
|heat1d|adam_lbfgs|0.01|4.229522|
|heat1d|adam_lbfgs|0.001|未达到（右删失）|
|heat1d|soap|0.01|2.839609|
|heat1d|soap|0.001|未达到（右删失）|
|heat1d|lm|0.01|4.509067|
|heat1d|lm|0.001|9.146063|
|heat2d|adam|0.01|未达到（右删失）|
|heat2d|adam|0.001|未达到（右删失）|
|heat2d|lbfgs|0.01|2.869559|
|heat2d|lbfgs|0.001|未达到（右删失）|
|heat2d|adam_lbfgs|0.01|4.887934|
|heat2d|adam_lbfgs|0.001|未达到（右删失）|
|heat2d|soap|0.01|3.848481|
|heat2d|soap|0.001|未达到（右删失）|
|heat2d|lm|0.01|3.175586|
|heat2d|lm|0.001|4.677353|

## 6. 重要局限/负结果

- 二维反演 LM 训练平衡MSE=1.190988e-12，而1024留出点平衡MSE=1.0023e-2（精确值见原audit JSON）；与SOAP的1.3905e-4、L-BFGS的1.9057e-4相比，存在强烈泛化/配点覆盖问题。压力误差低不能证明域内平衡已学好。
- Hertz所有有限主结果压力相对FEM误差约94.0%–98.4%；PCGrad另为数值失稳、非有限更新回退，不能当正常排名。部分方法loss下降不等于接触解成功；主实验已知载荷，不能称为合力反演。
- scheduler热：E=0.1976%，A=.5348%，D=.06445%；NS速度：E=22.0347%，A=.6389%，D=.6819%。E在NS实际动作变化95次，末e_bc=.24>τbc=.03。最小驻留可能压过异常边界恢复，且检查有间隔；失败机制尚未经消融确定。
- 同PDE方法内初始/采样哈希一致；方法学习率不同、未做等量超参数搜索，单种子不能给普遍优劣/统计显著结论。1D numpy记录2.4.6，而F盘项目1.26.4；不跨环境直接排名。
- 边界反弹提供研究动机，但现有数据没有GN方向曲率预测值、JVP耗时或曲率调度实验，不能声称H1/H2/H3已获验证。

## 7. 本次只读复核

- pinn_optimizer_lab：returncode=0；Validated 10 checkpoints, losses, metrics, snapshots and shared initialization.
- pinn_2d_optimizer_lab：returncode=0；Validated 20 saved models: initialization, sampling, individual losses, fields, pressure, force, metrics.
- hertz_cylinder_lab：returncode=0；Read-only Hertz audit: 11 checkpoint losses/metrics and shared recorded hashes verified.
- pinn_scheduler_lab：returncode=0；Read-only scheduler audit: 10 checkpoint metrics/probes and initialization/sampling hashes verified.

已有 independent_physics_audit.json / independent_audit.json 是历史审计文件，本次没有覆盖；本次复核只读取权重、重算训练损失与现有指标。通过复算只说明记录与模型一致，并不构成算法有效性或物理正确性的额外证明。

## 8. 可追溯路径与配置

### C:\Users\ASUS\Documents\ChatGPT\RA\pinn_optimizer_lab

源码：`C:\Users\ASUS\Documents\ChatGPT\RA\pinn_optimizer_lab\physics.py`、`C:\Users\ASUS\Documents\ChatGPT\RA\pinn_optimizer_lab\optimizers.py`、`C:\Users\ASUS\Documents\ChatGPT\RA\pinn_optimizer_lab\train.py`。
### F:\RA\pinn_2d_optimizer_lab

源码：`F:\RA\pinn_2d_optimizer_lab\physics.py`、`F:\RA\pinn_2d_optimizer_lab\optimizers.py`、`F:\RA\pinn_2d_optimizer_lab\train.py`。
### F:\RA\hertz_cylinder_lab

源码：`F:\RA\hertz_cylinder_lab\physics.py`、`F:\RA\hertz_cylinder_lab\optimizers.py`、`F:\RA\hertz_cylinder_lab\train.py`。
### F:\RA\pinn_scheduler_lab

源码：`F:\RA\pinn_scheduler_lab\physics.py`、`F:\RA\pinn_scheduler_lab\schedulers.py`、`F:\RA\pinn_scheduler_lab\train.py`。

#### C:\Users\ASUS\Documents\ChatGPT\RA\pinn_optimizer_lab\runs\demo
配置源：`C:\Users\ASUS\Documents\ChatGPT\RA\pinn_optimizer_lab\runs\demo\manifest.json`
```json
{
  "seed": 42,
  "width": 24,
  "depth": 2,
  "nu": 0.1,
  "n_interior": 128,
  "n_initial": 48,
  "n_boundary": 48,
  "steps": 1200,
  "record_every": 10,
  "threads": 1,
  "tolerance": 0.05
}
```

#### C:\Users\ASUS\Documents\ChatGPT\RA\pinn_optimizer_lab\runs\smoke
配置源：`C:\Users\ASUS\Documents\ChatGPT\RA\pinn_optimizer_lab\runs\smoke\manifest.json`
```json
{
  "seed": 42,
  "width": 24,
  "depth": 2,
  "nu": 0.1,
  "n_interior": 128,
  "n_initial": 48,
  "n_boundary": 48,
  "steps": 20,
  "record_every": 5,
  "threads": 1,
  "tolerance": 0.05
}
```

#### F:\RA\pinn_2d_optimizer_lab\runs\baseline
配置源：`F:\RA\pinn_2d_optimizer_lab\runs\baseline\config.json`
```json
{
  "seed": 42,
  "width": 24,
  "depth": 2,
  "steps": 1200,
  "record_every": 20,
  "n_heat": 128,
  "n_contact": 64,
  "n_edge": 16,
  "n_sensors": 48,
  "noise": 0.0,
  "threads": 1,
  "grid": 31
}
```

#### F:\RA\pinn_2d_optimizer_lab\runs\smoke
配置源：`F:\RA\pinn_2d_optimizer_lab\runs\smoke\config.json`
```json
{
  "seed": 42,
  "width": 8,
  "depth": 2,
  "steps": 4,
  "record_every": 1,
  "n_heat": 16,
  "n_contact": 12,
  "n_edge": 16,
  "n_sensors": 48,
  "noise": 0.0,
  "threads": 1,
  "grid": 31
}
```

#### F:\RA\hertz_cylinder_lab\runs\boundary_pilot
配置源：`F:\RA\hertz_cylinder_lab\runs\boundary_pilot\config.json`
```json
{
  "seed": 42,
  "radius_mm": 10.0,
  "young_mpa": 200.0,
  "poisson": 0.3,
  "top_pressure_mpa": 0.5,
  "thickness_mm": 10.0,
  "width": 24,
  "depth": 3,
  "n_interior": 96,
  "n_arc": 96,
  "steps": 4000,
  "record_every": 100,
  "grid": 35,
  "threads": 1,
  "equilibrium_weight": 1.0,
  "constitutive_weight": 10.0,
  "kkt_weight": 10.0,
  "boundary_constitutive": true
}
```

#### F:\RA\hertz_cylinder_lab\runs\contact_seed_check
配置源：`F:\RA\hertz_cylinder_lab\runs\contact_seed_check\run_config.json`
```json
{
  "seed": 42,
  "radius_mm": 10.0,
  "young_mpa": 200.0,
  "poisson": 0.3,
  "top_pressure_mpa": 0.5,
  "thickness_mm": 10.0,
  "width": 24,
  "depth": 3,
  "n_interior": 256,
  "n_arc": 128,
  "steps": 2400,
  "record_every": 60,
  "grid": 35,
  "threads": 1,
  "equilibrium_weight": 1.0,
  "constitutive_weight": 10.0,
  "kkt_weight": 10.0,
  "boundary_constitutive": true,
  "formulation": "displacement",
  "weak_weight": 10.0,
  "time_budget_seconds": 120.0,
  "initial_vertical_bias": -2.0
}
```

#### F:\RA\hertz_cylinder_lab\runs\dcgd_config_control
配置源：`F:\RA\hertz_cylinder_lab\runs\dcgd_config_control\run_config.json`
```json
{
  "seed": 42,
  "radius_mm": 10.0,
  "young_mpa": 200.0,
  "poisson": 0.3,
  "top_pressure_mpa": 0.5,
  "thickness_mm": 10.0,
  "width": 24,
  "depth": 3,
  "n_interior": 256,
  "n_arc": 128,
  "steps": 2400,
  "record_every": 60,
  "grid": 35,
  "threads": 1,
  "equilibrium_weight": 1.0,
  "constitutive_weight": 10.0,
  "kkt_weight": 10.0,
  "boundary_constitutive": true,
  "formulation": "displacement",
  "weak_weight": 10.0,
  "time_budget_seconds": 120.0
}
```

#### F:\RA\hertz_cylinder_lab\runs\dense_pilot
配置源：`F:\RA\hertz_cylinder_lab\runs\dense_pilot\config.json`
```json
{
  "seed": 42,
  "radius_mm": 10.0,
  "young_mpa": 200.0,
  "poisson": 0.3,
  "top_pressure_mpa": 0.5,
  "thickness_mm": 10.0,
  "width": 24,
  "depth": 3,
  "n_interior": 512,
  "n_arc": 128,
  "steps": 3000,
  "record_every": 100,
  "grid": 35,
  "threads": 1,
  "equilibrium_weight": 1.0,
  "constitutive_weight": 10.0,
  "kkt_weight": 10.0,
  "boundary_constitutive": true,
  "formulation": "displacement"
}
```

#### F:\RA\hertz_cylinder_lab\runs\displacement_pilot
配置源：`F:\RA\hertz_cylinder_lab\runs\displacement_pilot\config.json`
```json
{
  "seed": 42,
  "radius_mm": 10.0,
  "young_mpa": 200.0,
  "poisson": 0.3,
  "top_pressure_mpa": 0.5,
  "thickness_mm": 10.0,
  "width": 24,
  "depth": 3,
  "n_interior": 96,
  "n_arc": 96,
  "steps": 1600,
  "record_every": 100,
  "grid": 35,
  "threads": 1,
  "equilibrium_weight": 1.0,
  "constitutive_weight": 10.0,
  "kkt_weight": 10.0,
  "boundary_constitutive": true,
  "formulation": "displacement"
}
```

#### F:\RA\hertz_cylinder_lab\runs\kkt_all11
配置源：`F:\RA\hertz_cylinder_lab\runs\kkt_all11\run_config.json`
```json
{
  "seed": 42,
  "radius_mm": 10.0,
  "young_mpa": 200.0,
  "poisson": 0.3,
  "top_pressure_mpa": 0.5,
  "thickness_mm": 10.0,
  "width": 24,
  "depth": 3,
  "n_interior": 256,
  "n_arc": 128,
  "steps": 2400,
  "record_every": 60,
  "grid": 35,
  "threads": 1,
  "equilibrium_weight": 1.0,
  "constitutive_weight": 10.0,
  "kkt_weight": 10.0,
  "boundary_constitutive": true,
  "formulation": "displacement",
  "weak_weight": 10.0,
  "time_budget_seconds": 120.0,
  "initial_vertical_bias": 0.0
}
```

#### F:\RA\hertz_cylinder_lab\runs\physics_pilot
配置源：`F:\RA\hertz_cylinder_lab\runs\physics_pilot\config.json`
```json
{
  "seed": 42,
  "radius_mm": 10.0,
  "young_mpa": 200.0,
  "poisson": 0.3,
  "top_pressure_mpa": 0.5,
  "thickness_mm": 10.0,
  "width": 24,
  "depth": 3,
  "n_interior": 96,
  "n_arc": 96,
  "steps": 1200,
  "record_every": 40,
  "grid": 35,
  "threads": 1,
  "equilibrium_weight": 1.0,
  "constitutive_weight": 10.0,
  "kkt_weight": 10.0
}
```

#### F:\RA\hertz_cylinder_lab\runs\pilot
配置源：`F:\RA\hertz_cylinder_lab\runs\pilot\config.json`
```json
{
  "seed": 42,
  "radius_mm": 10.0,
  "young_mpa": 200.0,
  "poisson": 0.3,
  "top_pressure_mpa": 0.5,
  "thickness_mm": 10.0,
  "width": 24,
  "depth": 3,
  "n_interior": 96,
  "n_arc": 96,
  "steps": 400,
  "record_every": 20,
  "grid": 35,
  "threads": 1
}
```

#### F:\RA\hertz_cylinder_lab\runs\weak_pilot
配置源：`F:\RA\hertz_cylinder_lab\runs\weak_pilot\config.json`
```json
{
  "seed": 42,
  "radius_mm": 10.0,
  "young_mpa": 200.0,
  "poisson": 0.3,
  "top_pressure_mpa": 0.5,
  "thickness_mm": 10.0,
  "width": 24,
  "depth": 3,
  "n_interior": 256,
  "n_arc": 128,
  "steps": 2400,
  "record_every": 100,
  "grid": 35,
  "threads": 1,
  "equilibrium_weight": 1.0,
  "constitutive_weight": 10.0,
  "kkt_weight": 10.0,
  "boundary_constitutive": true,
  "formulation": "displacement",
  "weak_weight": 10.0
}
```

#### F:\RA\pinn_scheduler_lab\runs\pilot30
配置源：`F:\RA\pinn_scheduler_lab\runs\pilot30\manifest.json`
```json
{
  "width": 32,
  "depth": 3,
  "lr": 0.001,
  "n_interior": 256,
  "n_edge": 32,
  "probe_interior": 192,
  "probe_edge": 64,
  "reynolds": 20.0,
  "check_every": 50,
  "budget_seconds": 30.0,
  "max_steps": 30000,
  "tau_bc": 0.03,
  "tau_pde": 0.03,
  "hysteresis": 0.2,
  "min_dwell_checks": 2,
  "grid": 41,
  "relobralo_alpha": 0.999,
  "relobralo_beta": 0.999,
  "relobralo_temperature": 0.1
}
```

#### F:\RA\pinn_scheduler_lab\runs\pilot60
配置源：`F:\RA\pinn_scheduler_lab\runs\pilot60\manifest.json`
```json
{
  "width": 32,
  "depth": 3,
  "lr": 0.001,
  "n_interior": 256,
  "n_edge": 32,
  "probe_interior": 192,
  "probe_edge": 64,
  "reynolds": 20.0,
  "check_every": 50,
  "budget_seconds": 60.0,
  "max_steps": 30000,
  "tau_bc": 0.03,
  "tau_pde": 0.03,
  "hysteresis": 0.2,
  "min_dwell_checks": 2,
  "grid": 41,
  "relobralo_alpha": 0.999,
  "relobralo_beta": 0.999,
  "relobralo_temperature": 0.1
}
```

#### F:\RA\pinn_scheduler_lab\runs\scheduler60
配置源：`F:\RA\pinn_scheduler_lab\runs\scheduler60\manifest.json`
```json
{
  "width": 32,
  "depth": 3,
  "lr": 0.001,
  "n_interior": 256,
  "n_edge": 32,
  "probe_interior": 192,
  "probe_edge": 64,
  "reynolds": 20.0,
  "check_every": 50,
  "budget_seconds": 60.0,
  "max_steps": 30000,
  "tau_bc": 0.03,
  "tau_pde": 0.03,
  "hysteresis": 0.2,
  "min_dwell_checks": 2,
  "grid": 41,
  "relobralo_alpha": 0.999,
  "relobralo_beta": 0.999,
  "relobralo_temperature": 0.1
}
```

逐方法源JSON路径、SHA256、每项最终损失和完整配置均见 experiment_inventory.json；主图直接读取 baseline/heat2d/*.json 与 scheduler60/ns2d_*/result.json 的history数组；对应CSV为可交叉核对的副本，由build_research_note.py读取JSON生成。

## 9. 缺失数据及发送前复核

尚无记录：GPU显存、CPU内存峰值、GN/LM分段profile、严格等端到端wall预算的全部对照、多种子统计、GN曲率预测与JVP运行数据；完整B/C最优分段调参未做。旧pilot不完整配置不补当前默认值。源文件可能在历史运行后被更改；provenance哈希可用于核查，不能将所有pilot视为当前源码的可复现实验。
建议人工复核：署名/日期与导师偏好的术语；确认只引用自己的合成实验且可分享；确认愿意报告反演平衡泛化差和Hertz失败；核对未来实验预算与主要误差阈值；文献对应版本与正式题名；全文不得把构想写作原创或已验证。


## 10. 数学约定与计划的执行口径

平方损失采用1/2范数约定。对每个权重为w、以N作平均分母的MSE分量，令规范残差 r=sqrt(2w/N)e；多个分量分别缩放后堆叠。现有LM代码的残差是sqrt(w/N)e、损失为其范数平方，二者只差整体常数，但阻尼数值要随约定一致缩放。对C3残差映射，一般GN遗漏项为O(||r|| ||d||²)；当r_bc=0时 leading term=1/2||J_bc d||²、余项O(||d||³)。若J_bc d=0，二次项不严格为正，仍可能有更高阶误差，不能宣称零风险。候选p_a为有符号方向，d_a=alpha_a p_a、alpha_a>0，改善量定义为旧损失减新损失，故公式有两个负号。

未来6组每次分别固定Adam或SOAP；两种优化器独立比较，避免同时改调度和优化器。主实验先5共同种子、每组120秒，失败种子也报告；热与NS采用同一阈值的预注册定义（解/速度相对L2≤1%且归一化边界误差≤.03）。端到端训练计时从模型/优化器初始化开始至最终约束检查，含候选生成、梯度/JVP、策略、回退和统一评估，绘图/导出邮件附件不计入。记录首次检查达标时间，不插值，同时报告保持达标/反弹次数。任何因时间耗尽未达标的运行均记为右删失，不从均值中剔除。预设联合达标未达时不能只按PDE loss宣称成功。

上述都是计划：没有运行新训练、没有实现或测试曲率调度。参考文献[4]的出版社网页本次超时，但已通过Crossref官方接口核实题名、作者、1963年、11卷2期、431-441页与DOI，原始核对元数据保存在reference_metadata_check.json；其余采用论文作者、正式会议或期刊页核实。ReLoBRaLo正式期刊版为CMAME 439 (2025) 117914，历史代码注释引用了2021预印本，不将版本差异隐去。

重建脚本会重新生成MD与PDF：如人工修改MD，先备份，或同时修改build_research_note.py中的共享内容；当前脚本不是通用Markdown转PDF工具。原实验源码、训练结果与历史审计不被脚本写入。
