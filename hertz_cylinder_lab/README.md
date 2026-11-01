# Hertz 圆柱–刚性平面：KKT 接触与优化器实验

所有新代码、数据和结果位于 `F:\RA\hertz_cylinder_lab`。环境为 `F:\RA\.venv`。

## 打开结果

- `runs/kkt_all11/dashboard.html`：原有十种方法加 DCGD，包含 ConFIG，所有实际轨迹完整保留。
- `runs/dcgd_config_control/dashboard.html`：DCGD-Center 与 ConFIG 的**相同两组目标、相同学习率 0.001**控制实验。
- 页面包含 loss、压力误差、压力/间隙回放、穿透/拉应力/互补性、有限元实际变形及完整最终表。
- 图与记录可查看 `training_curves.png`、`pressure_gap_and_time.png`、`contact_evolution.gif`、`summary.csv`、`independent_audit.json`。

**当前实验是探索基线。有限元参考解已经通过检查，但不能因 PINN 的总 loss 下降或反力接近输入，就宣称压力分布已正确。若压力误差较大，页面会明确显示“尚未通过压力精度验证”。**

## 基准物理

采用 Sahin 等的 Hertz 半圆柱基准比例 `R=1, E=200, ν=0.3, q=0.5`；工程单位取 R=10 mm、E=200 MPa、顶部压力 q=0.5 MPa、轴向厚度 b=10 mm。使用四分之一圆的对称模型，圆心 (0,R)，刚性平面 y=0，顶部水平切面 y=R 均布向下压力。

这是**力控制正问题**。已知全模型载荷 `F=2Rq b=100 N`，而接触半宽和压力分布未知。反力平衡是物理约束与核验，不是未知合力反演。

平面应变、线弹性、小变形、无摩擦。主实验用位移网络，应力由本构精确计算：

\[
\nabla\cdot\sigma=0,\quad\sigma=\lambda\operatorname{tr}(\varepsilon)I+2\mu\varepsilon,
\quad\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)},\quad\mu=\frac{E}{2(1+\nu)}.
\]

圆弧初始外法向为 `n=(x/R, y/R-1)`，刚性平面接触反力沿竖直方向。令 `g=Y+u_y`，`p=(σn)_y`，则

\[
g\ge0,\quad p\ge0,\quad gp=0,\quad (\sigma n)_x=0.
\]

KKT 用 Fischer–Burmeister 残差训练：
\[
\phi(g,p)=\sqrt{g^2+p^2+10^{-16}}-g-p.
\]
间隙、应力先无量纲化；没有预设接触区，也没有将 Hertz 半宽或压力当作训练标签。对称轴通过偶函数输入及 `u_x=x raw_u_x` 硬满足位移和剪切对称条件。顶部牵引作为显式损失。

## 尺度、采样与损失

`eps=q/E=0.0025`，位移尺度 `u0=eps R=25 μm`，应力尺度 `s0=sqrt(Eq)=10 MPa`。间隙除以 u0，压力除以 s0。网络输入含全局坐标及由 `sqrt(q/E)` 决定的局部坐标特征；这个尺度来自已知材料/载荷，不含真实接触宽度。精确一、二阶链式法则导数用于应力和强平衡残差。

主实验使用相同的 256 个内部点（全局 + 接触附近加密）、128 个潜在接触弧点、32 个顶部点、64 个接触积分点及 20×20 全局极坐标积分点。额外用 12 个多项式虚位移施加全局虚功平衡，抑制强形式稀疏配点的隐藏边界层。

损失为：平衡残差 + 10×顶部牵引 + 无摩擦 + 10×FB + 载荷平衡 + 10×虚功。每项均为平方均值，未使用 FEM 位移或 Hertz 压力监督。虚功为

\[
\int_\Omega\sigma:\varepsilon(w)\,d\Omega
+q\int_{\Gamma_{top}}w_y\,dX
-\int_{\Gamma_{arc}}p\,w_y\,dS=0.
\]

`runs/*pilot*` 是早期结构/采样探索记录，部分采用不同混合场网络和权重，不应用当前代码无条件复现或与主实验直接排名；正式可复算对照为 `kkt_all11` 与 `dcgd_config_control`，配置见各自 `run_config.json`。

## DCGD 与 ConFIG

两组目标明确固定为：

1. A = 平衡方程 + 全局虚功；
2. B = 顶部牵引 + 无摩擦 + KKT + 载荷平衡。

DCGD 使用论文的 Center 变体：
\[
c=\frac{g_A}{\|g_A\|}+\frac{g_B}{\|g_B\|},\qquad
d=\frac{c^T(g_A+g_B)}{c^Tc}c,\qquad\theta^+=\theta-\eta d.
\]
ConFIG 使用归一化梯度伪逆求公共方向，再按原梯度投影确定幅值。**对于两个非退化目标，它可与 DCGD-Center 等价**；单元测试对这个恒等关系进行了验证。因此同组、同学习率对照非常重要。两者均直接 GD，不在后面再接 Adam。恰好相反的梯度返回零方向；单个零梯度使用剩余梯度。

ConFIG=0.02 的早期探索出现非有限数值，失败保存在 failed_attempts.json。正式全方法对照与控制组中，DCGD 和 ConFIG 都采用 0.001；其他方法保留各自明示的学习率。ConFIG 不是 DCGD 的别名，多目标数量或方向/幅值规则改变时一般不再等价。

## 有限元与 Hertz 校核

独立参考为 P1 三角有限元的原始–对偶主动集求解，直接解离散接触不等式，没有调用 PINN 或 Hertz 来确定活动节点。粗网格 32×112（径向×角向），细网格 48×160。细网格约 15,458 个位移自由度，7 次主动集迭代；离散最大穿透为零，反力平衡误差约 7e-12，粗细网格位移差约 0.134%、压力差约 1.27%。

Hertz 局部理论：
\[
E^*=\frac{E}{1-\nu^2},\quad P'=2Rq,\quad
a=\sqrt{\frac{4P'R}{\pi E^*}},\quad
p_H(x)=\frac{2P'}{\pi a}\sqrt{1-(x/a)^2}.
\]
理论半宽 a≈0.7611 mm；FEM 细网格接触边缘括区约 [0.7541,0.8049] mm。FEM 与 Hertz 压力曲线相对差约 2.1%，其中包含有限半圆柱与局部半空间近似的区别及离散误差。

压力由原始弧长牵引换算为水平投影密度，误差只在 x/R≤0.3 比较，避开 x/R=1 的竖直切线奇异投影。全弧 801 个独立检查点用于穿透、拉力、FB 和反力检查。Hertz 阈值下的“接触半宽”仅为后处理指标，不是学习到的精确活动集边缘。

## 运行与复算

```powershell
Set-Location F:\RA\hertz_cylinder_lab
F:\RA\.venv\Scripts\python.exe train.py --steps 2400 --n-interior 256 --n-arc 128 --record-every 60 --time-budget 120 --comparison-lr .001 --out runs/new_all
F:\RA\.venv\Scripts\python.exe train.py --steps 2400 --n-interior 256 --n-arc 128 --record-every 60 --time-budget 120 --comparison-lr .001 --optimizers dcgd config --out runs/new_control
F:\RA\.venv\Scripts\python.exe -m pytest tests -q
F:\RA\.venv\Scripts\python.exe validate.py runs/kkt_all11
```

每种方法同一个参数初始化和固定采样，最多 2400 步或 120 秒优化耗时软预算，完整一步结束后再检查，可能略超预算。L-BFGS 一步有线搜索，LM 一步有 Jacobian/线性求解，所以实际步数不同。误差–耗时曲线是主要效率视角，计时包含系统负载波动。训练计时不含快照、参考求解和导出。

非空目录默认不覆盖；相同配置可用 `--resume`，会核对初始参数、采样和学习率。ConFIG 原始记录文件为 `config.json`，实验配置单独保存为 `run_config.json`，两者用途不同。

## 来源

- [Sahin et al., contact PINNs with KKT/FB](https://arxiv.org/abs/2308.12716)，本项目不是逐行或性能复现。
- [DCGD，式 4.3 / E.3](https://arxiv.org/abs/2409.18426)。
- [ConFIG](https://arxiv.org/abs/2408.11104)。
- 有限元使用 [scikit-fem](https://scikit-fem.readthedocs.io/en/stable/api.html)。

## 本轮控制实验的观测

同学习率、同两组目标下，DCGD 与 ConFIG 最终压力曲线相对差约 3.56e-5；两者压力相对 FEM 误差均约 96.23%，所以这不是成功的 Hertz PINN 解。它验证了两目标情形下方向等价的实现，却没有解决该接触配置的收敛问题。接触初始位移偏置试验也没有明显改善；保留在本机 runs/contact_seed_check，不与零偏置主实验混合排名。

参考有限元通过离散非穿透、非负接触反力、互补性、力平衡与粗细网格检查；这些检查不能替代对 PINN 自身的验证。
