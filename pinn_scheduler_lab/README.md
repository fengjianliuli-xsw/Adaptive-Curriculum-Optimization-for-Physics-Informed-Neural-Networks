# 二维 PINN：先学边界，还是先学方程？

本项目实现一次小规模受控对照：**同一个 Adam、同一个初始化，只改变损失学习顺序**。没有控制器神经网络、没有切换优化器、没有数据监督内点。全部新文件位于 `F:\RA\pinn_scheduler_lab`。

第一版采用用户开头指定的二维热传导与 Navier–Stokes；Poisson 已作为稳态热传导实现。Burgers 与 Helmholtz 暂未加入。这里“二维”指两个空间坐标，两个算例均为**稳态**。

## 快速查看与复现

打开 `runs/scheduler60/dashboard.html`，直接离线使用。选择 PDE 和输出场，拖动训练时间，或点“播放”。页面提供二维预测、截面动画、每项损失、解误差、调度轨迹、反弹统计与成本分解。独立 PNG/GIF 位于 `figures/`，原始 CSV 和逐检查点 JSON 均保留。

在 PowerShell 中：

```powershell
Set-Location F:\RA\pinn_scheduler_lab
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
F:\RA\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=F:\RA\pinn_scheduler_lab\test_tmp
F:\RA\.venv\Scripts\python.exe experiment.py --suite scheduler60 --baseline pilot60 --budget 60 --seeds 42
F:\RA\.venv\Scripts\python.exe validate.py --suite scheduler60
```

`experiment.py` 先检查联合训练 A 是否通过门槛，再复用该基线进行 B–E，已有结果会跳过；如要全新复现，请使用新的 suite 和 baseline 名。例：`--suite repeat60 --baseline repeat_baseline60`。多种子扩展使用 `--seeds 42 43 44`，每个种子都须先通过 A 的门槛，禁止只挑通过的种子报告。

门槛预先设置为热传导/合并速度相对 L² ≤ 5%，NS 压力 ≤ 10%。30 秒预试压力未通过，统一增加为 60 秒后通过；此后冻结配置，没有按 E 的结果调参。预试位于 `pilot30`、`pilot60`。

## 通俗理解

把学 PDE 看成“学内部规律”，把学 BC 看成“守住边界”。传统 PINN 每一步两门课一起学：

\[
\theta_{t+1}=\operatorname{Adam}(\theta_t,\nabla_\theta[\mathcal L_f+\mathcal L_{bc}]).
\]

现在只改每一步的“课程”：

\[
a_t=\pi(s_t),\quad
\mathcal L_{a_t}=\begin{cases}\mathcal L_{bc}&a_t=\mathrm{BC},\\\mathcal L_f&a_t=\mathrm{PDE},\\\mathcal L_f+\mathcal L_{bc}&a_t=\mathrm{Joint}.\end{cases}
\]

BC/PDE 单独训练时省去未选中任务的前后向；所有方法每 K=50 步重新检查全部约束。切换时**保留 Adam 的动量和二阶矩**。固定学习率 0.001；PyTorch Adam 默认 β=(0.9,0.999)、ε=1e-8、无 weight decay，无梯度裁剪或学习率调度。优化器本身不是本轮变量。

|组|训练方式|本轮精确定义|
|---|---|---|
|A|始终 Joint|每步 Lf + Lbc|
|B|BC → PDE → Joint|分别占约 1/3 计费时间；只在检查点切换|
|C|PDE → BC → Joint|B 的顺序反转，同样等时间三阶段|
|D|ReLoBRaLo|每步 λf Lf + λbc Lbc，自适应权重和为 2|
|E|规则切换|每 50 步根据物理约束决定 BC / PDE / Joint|

## PDE 与解析解

两例均在单位方域 Ω=[0,1]²；网络输入映射到 [-1,1]²。

### 二维稳态热传导

令无量纲温度尺度、导热系数均为 1：

\[
-\Delta T=q,\qquad T|_{\partial\Omega}=0,
\]
\[
T^*=\sin(\pi x)\sin(\pi y)+0.2\sin(3\pi x)\sin(2\pi y),
\]
\[
q=2\pi^2\sin(\pi x)\sin(\pi y)+2.6\pi^2\sin(3\pi x)\sin(2\pi y).
\]

源项使解具有两个空间频率。定义固定归一化残差 r_f=(-ΔT-q)/(2π²)，Lf=mean(r_f²)，Lbc=mean((T-0)²)。归一化因子对 A–E 完全相同。

### 二维不可压缩 Navier–Stokes：Kovasznay 流

取 Re=20、ν=1/Re、ρ=1、速度尺度 U₀=1、长度 L₀=1、压力 p₀=ρU₀²=1：

\[
u u_x+v u_y+p_x-\nu\Delta u=0,\\
u v_x+v v_y+p_y-\nu\Delta v=0,\\
u_x+v_y=0.
\]
\[
\lambda=\frac{Re}{2}-\sqrt{\frac{Re^2}{4}+4\pi^2},\quad
u^*=1-e^{\lambda x}\cos(2\pi y),\quad
v^*=\frac{\lambda}{2\pi}e^{\lambda x}\sin(2\pi y),\quad
p^*=\frac{1-e^{2\lambda x}}{2}.
\]

四边给定解析**速度**，另加单点压力规范 p(0,0)=0。没有给整个压力边界，更没有用内点解析压力监督。输出三个头 (u,v,p)，共享隐藏层；本轮没有额外改变多头结构。Lf 是两个动量残差和连续性残差的各自 MSE 之和；Lbc 为 u、v 边界 MSE 加压力规范误差平方。主精度为合并速度相对 L²，另报 u、v 和压力误差。

方程和解析流场可对照 [DeepXDE 官方 Kovasznay 示例的代码](https://deepxde.readthedocs.io/en/v1.15.0/demos/pinn_forward/Kovasznay.flow.html)。本项目使用单位方域、压力单点规范及固定 Adam，与该示例的域、边界设置和训练方案不同，不声称复现其数值结果。

## E：物理尺度归一化的规则控制器

检查点状态为

\[
s_t=(e_{bc},e_f,\Delta_f,\Delta_{bc},c_g,t_{\mathrm{step}}),\quad
\Delta_i=1-\frac{L_i(t)}{\max(L_i(t-K),10^{-30})},
\]
\[
c_g=\frac{\nabla L_f\cdot\nabla L_{bc}}{\|\nabla L_f\|\|\nabla L_{bc}\|}.
\]

梯度在固定的独立物理检查点上计算，分母保护 1e-30。热传导 e_bc 为边界温度 RMSE/T₀；NS e_bc 为 u、v 的边界 RMSE/U₀ 与单点压力误差/p₀ 三者的最大值。它是**无量纲 RMSE，不是原始 loss**；e_f=√Lf。阈值 τbc=τf=0.03，表示相对于选定物理尺度的 3%，不是解相对误差 3%。BC 采用最大分量，避免某个条件被其他项平均稀释。

本轮完整规则（`schedulers.py`）如下：

1. 距上次动作切换不足 2 个检查间隔时保留动作，防止抖动。
2. e_bc > 1.2τbc 时：若 e_bc > 2τbc 或当前就在 BC，则选 BC；否则选 Joint。
3. 边界未严重违反且 e_f ≤ τf 时选 Joint 精修。
4. e_bc ≤ 0.8τbc 且 PDE 未达标时选 PDE；但若梯度余弦 < -0.5 且 PDE 相对下降 <0.001，则选 Joint。
5. 中间迟滞带选 Joint。若 BC 连续停滞（下降 <0.001），且误差小于 2τbc，则从 BC 退出到 Joint。

最近一步耗时写入状态和日志，**本版不参与动作分支**。相对下降按检查次数、不是按秒归一化；后续若比较效率型策略，可改成每秒下降率。所有分支都不读取解析解误差、测试网格或最终审计残差。

阈值只用于离散检查后的决策，**不保证整段训练始终满足边界约束**；每次切换后至少驻留两个检查间隔，即使发生严重反弹也可能等到驻留期结束才切回。第一版保留该规则以观察失败模式，未在见到结果后增加紧急切换或重跑 E。

## D：ReLoBRaLo 的公式

让下降较慢的目标得到更多关注。对 i∈{f,bc}，定义两项和为 2 的相对进展权重：

\[
\widehat\lambda_i(t,t_0)=2\,\operatorname{softmax}_i\!\left(\frac{L_i(t)}{T\max(L_i(t_0),10^{-12})}\right),
\]
\[
\lambda_i(t)=\alpha[\rho_t\lambda_i(t-1)+(1-\rho_t)\widehat\lambda_i(t,0)]
+(1-\alpha)\widehat\lambda_i(t,t-1).
\]

ρt 为 Bernoulli 随机变量，概率 0.999；α=0.999，T=0.1，初始权重均为 1。权重使用当前更新前的损失，计算后停止梯度，再用于本步 Adam。使用独立 NumPy RNG，不改变网络/采样随机流。实现根据 [ReLoBRaLo 原论文 Eq. 11](https://arxiv.org/abs/2110.09813)，两项按 PDE/BC 聚合；未针对本项目调优 α、T、ρ。

## 公平预算与评估

- 网络统一为 3 层 × 32 tanh，Xavier normal，零偏置；CPU 单线程 float64。为了降低高阶自动求导成本，用精确 tanh 链式法则传播一、二阶空间导数，仍由 PyTorch 自动求参数梯度。测试与普通 autograd 比较了空间导数和参数梯度。
- 每组内点固定 256 个 scrambled Sobol，边界每边 32 点。物理检查内点 192 个独立 Sobol，边界每边 64 个中点。跨方法不重采样。训练和检查点坐标哈希记录在 JSON；共享种子保证相同初始参数。
- 解析误差在固定 41×41 均匀网格计算。另用固定 1024 个、与训练及控制器不同的 Sobol 点审计 PDE 残差。解析真解只在报告侧使用；已知边界值属于问题输入。
- 每组预算为优化 + 诊断 + 决策耗时。每 50 步各方法都执行相同完整诊断（含梯度余弦、训练项检查），E 不免费获得额外信息。D 逐步权重更新额外计入策略成本。该设置下的 E “策略时间”只含规则计算；实际所需的信息成本应结合诊断列阅读。**无“免诊断 A”的额外实验，不能把 E 全部控制开销只说成规则计算时间。**
- 解析解评估、独立审计、存盘与 Python 循环开销不包含在计费预算，另列 reference_seconds 和训练循环墙钟 wall_seconds（不含初始化、初始权重保存、最终 JSON 和图表生成）。因而这是“等计费训练时间”，并非严格等端到端时间；如部署成本是研究目标，应增加严格墙钟停止的对照。
- B/C 依据已消耗计费时间的三等分，只在检查点调整动作，因此存在一个检查间隔的分段偏差。最终检查允许少量超时，实际时间均保留。
- 没有将 1−相对误差称作“准确率”；科学回归问题主要用相对 L²，越小越好。所有结果取最终检查点，无真解选模。

## 如何解释

E 只是显式手工策略，本版没有“学会”任何策略。若 E 只好于 A、未好于 B/C/D，不能证明自动调度的额外价值。即使 E 在一次实验领先，也需要多种子、等端到端时间、精调 B/C 分段、阈值/K/驻留消融、更多 PDE 和频率/Re 范围验证。

若边界反复反弹而 E 频繁切回 BC，应优先检查守护阈值、Joint 的恢复能力和 Adam 动量的作用。此时直接加入强化学习或控制器网络，可能只是把尚未理解的失败藏进另一个训练问题。

## 文件

`physics.py` 方程/网络/物理诊断；`schedulers.py` A–E 策略；`train.py` 单次或批量训练；`experiment.py` 门槛后对照；`report.py` 离线看板/图表/GIF；`validate.py` 重新加载权重独立审计；`test_core.py` 数学与策略测试；`runs/*` 原始训练数据和成果。`manifest.json` 记录环境、参数与门槛。所需包：torch、numpy、matplotlib、pillow；pytest 用于验证，playwright 仅用于页面冒烟检查。
