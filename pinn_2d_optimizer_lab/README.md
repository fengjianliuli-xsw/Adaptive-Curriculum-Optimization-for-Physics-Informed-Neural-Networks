# 二维 PINN 优化器实验室

本项目延续一维实验，完整比较同一组 **10 种优化方法**：Adam、SGD + Momentum、RMSProp、L-BFGS、Adam → L-BFGS、Gauss–Newton/LM、MultiAdam、PCGrad + GD、ConFIG + GD、SOAP。默认突出 Adam、Adam → L-BFGS、SOAP，其他方法均真实训练并可在页面勾选查看。

新实验根目录按用户要求固定为 `F:\RA`；本项目位于 `F:\RA\pinn_2d_optimizer_lab`，新环境位于 `F:\RA\.venv`。上一轮 C 盘项目保持原位。`F:\RA\AGENTS.md` 记录此后的实验路径偏好。

## 查看结果

打开项目根目录 `index.html`，或 `runs/baseline/dashboard.html`，无需服务器和网络。首页可直接进入接触实验。`dashboard.template.html` 是生成模板，直接打开会显示明确的入口提示，不能作为实验结果页。

接触实验页已加入 `assets/contact_geometry.svg` 示意图，按实际 2:1 截面比例绘制几何、加载区、边界约束和本次记录的 48 个观测位置。外部接触体的形状与半径未定义，不把此例解释为球头/圆柱压头的赫兹接触。

- 切换两个 PDE；点击“全部方法”显示全部 10 种结果。
- 查看加权分项 loss、场相对 L2、误差达标率以及接触压力和合力误差。
- 播放、暂停和拖动真实训练快照，按迭代/实际训练耗时/残差调用次数同步。
- 查看二维温度或位移场，切换绝对误差图；接触图显示位移观测位置。
- 查看温度剖面、接触压力分布及各方法最终表格，下载 CSV 或当前图片。
- `heat2d_evolution.gif` 与 `contact2d_evolution.gif` 为重点三种方法的可分享动画；交互页面支持全部方法。

所有展示都使用实际计算记录，没有人工构造收敛曲线，也不在采样帧之间伪造插值。

## 实验 A：二维稳态热传导 / Poisson

两个独立空间坐标 `(x,y)`，不将一维空间加时间冒充二维空间：

\[
-\Delta u=f(x,y),\quad (x,y)\in[0,1]^2,\quad u|_{\partial\Omega}=0.
\]
选择可验证的解析解：
\[
u^*=\sin(\pi x)\sin(\pi y)+0.2\sin(3\pi x)\sin(2\pi y),
\]
对应已知源项：
\[
f=2\pi^2\sin(\pi x)\sin(\pi y)+2.6\pi^2\sin(3\pi x)\sin(2\pi y).
\]
训练使用已知源项和边界，不使用内部真解标签。PDE 残差除以 `2π²` 作为明示的物理尺度归一化：

\[
L_f=\operatorname{mean}\left[\frac{-u_{xx}-u_{yy}-f}{2\pi^2}\right]^2,
\quad L_{BC}=\operatorname{mean}(u_{\partial\Omega}^2),\quad L=L_f+L_{BC}.
\]
默认 128 个固定内部 Sobol 点，每边 16 个边界点，评估网格 31×31。网络 `2→24→24→1`，tanh，697 个参数；导数通过精确链式法则传播，参数反向传播仍由 PyTorch 自动微分完成。

## 实验 B：二维平面应力弹性接触载荷反演

**已知接触区域下的无摩擦法向载荷反演**，从内部稀疏位移观测恢复顶部压力与合力。它不包含未知间隙、接触分离互补条件、摩擦或大变形接触。

已知条件：矩形宽 W=40 mm、高 H=20 mm、厚度 b=10 mm，E=1 MPa、ν=0.3；底部位移固定，两侧无牵引，顶部剪切牵引为零。已知接触区 `x/W ∈ [0.2,0.8]`，区外压力为零。采用平面应力假设。

物理方程：
\[
\nabla\cdot\sigma=0,\qquad \sigma=C:\varepsilon(u),
\qquad \sigma n=(0,-p)\ \text{on top}.
\]
引入归一化坐标 `x=X/W, y=Y/H`、压力尺度 `p0=100 kPa`、位移尺度 `u0=p0 W/E=4 mm`，`a=W/H=2`。对归一化位移 `(u,v)` 和应力 `(sxx,syy,sxy)`：

\[
s_{xx,x}+a s_{xy,y}=0,\qquad s_{xy,x}+a s_{yy,y}=0,
\]
\[
s_{xx}=\frac{u_x+\nu a v_y}{1-\nu^2},\quad
s_{yy}=\frac{\nu u_x+a v_y}{1-\nu^2},\quad
s_{xy}=\frac{a u_y+v_x}{2(1+\nu)}.
\]
网络共有 990 个参数：一个 `2→24→24→5` 场网络联合输出位移和应力，一个 `1→12→12→1` 压力头。位移变换 `u=y raw_u, v=y raw_v` 硬满足底部固定。压力用

\[
p(x)=\mathrm{softplus}(h_\phi(x))\left[\max\left(0,1-\left(\frac{x-0.5}{0.3}\right)^2\right)\right]^2
\]
保证非负和已知支撑区；不向模型提供真实压力系数。

默认 64 个固定内部点，顶部及左右侧每边 16 个牵引点，48 个内部二维位移传感点。训练损失：

\[
L=L_{eq}+L_{constitutive}+L_{traction}+10 L_{obs}.
\]
三个物理项与观测项均为明确定义的均方残差。页面分项显示已乘上述权重的值，总损失是它们之和。底部约束通过结构硬满足，所以没有额外底部 loss。

合力的物理单位：

\[
F=b\int_0^W p(X)\,dX
=p_0 W b\int_0^1 \bar p(x)\,dx
=40\,\mathrm{N}\int_0^1\bar p(x)\,dx.
\]
这里报告的是指定 10 mm 厚度下的力 N，不把二维截面单位厚度的线载荷混称为力。

### 独立参考与观测生成

`reference.py` 使用 scikit-fem 的 P2 三角有限元独立求解正问题。FEM 坐标为 `(X/W,Y/W)`，高度为 0.5；网络使用 `(X/W,Y/H)`，两者通过明确坐标映射连接。

FEM 平面应力材料系数使用 `λ*=ν/(1-ν²), μ*=1/[2(1+ν)]`，不是三维 Lamé λ。粗网格 33×17、细网格 65×33（细网格 4,096 个三角形，16,770 个位移自由度）。检查粗细网格的位移差异小于 1%，以及装配载荷与底部反力的平衡误差小于 1e-8；实际测得约 0.0341% 与 1.46e-13。

合成真实压力为“已知支撑窗 × 一个平滑非对称函数”，真实系数只在 FEM 数据生成和误差评估中使用。当前真实合力约 13.3953 N。训练只接收 48 个位置的位移观测和物理方程；内部全场真值、压力真值、真实合力不会进入训练、线搜索或停止判断。

默认无噪声。`--noise 0.01` 可加入相对于二维位移观测整体 RMS 的 1% 高斯噪声，同一 PDE 内所有优化器使用相同的噪声样本。

## 公平性与解释边界

- 每个 PDE 内优化器从同一个完整 `state_dict` 副本开始；保存参数 SHA-256 与采样/观测 SHA-256，并核对第 0 步场预测严格相同。不同 PDE 的模型结构不同，不强求参数向量相同。
- 最大外层预算均为 1,200 步，float64、CPU、单线程、固定全批采样。没有隐藏重采样或梯度裁剪。若出现非有限更新会恢复之前的有限模型并明示状态；LM 梯度达到容限可提前停止。
- 学习率沿用一维项目的固定配置：Adam 0.001、动量 SGD 0.01、RMSProp 0.001、L-BFGS 1、MultiAdam 0.001、PCGrad/ConFIG 0.02、SOAP 0.003。没有用当前真解系统调参，某些方法在此预算下会欠收敛。
- Adam → L-BFGS 在总预算一半处切换。L-BFGS `max_iter=1`、history=50、强 Wolfe 线搜索；一步可多次调用 closure。
- LM 使用所有残差的精确稠密 Jacobian；使用缩小后的对偶系统，接受实际下降的阻尼 Gauss–Newton 更新。训练残差向量的平方和严格等于加权总损失。
- MultiAdam：热传导分为 PDE/边界两组；接触分为 (平衡+本构)/(牵引+观测) 两组，β₁=β₂=0.99。PCGrad、ConFIG 使用每个加权分项的梯度，之后直接 GD，避免后接 Adam 改写方向。
- SOAP 是上一轮延续的教学实现，按层统计特征基、在旋转坐标中更新，使用对角二阶矩近似转移，每 10 步刷新基；不宣称官方复现性能。
- `train_seconds` 计优化步骤，排除诊断、快照和导出；`total_wall_seconds` 另记。普通工作站上有系统负载与计时波动，不能把一次测得的毫秒差当作稳定优劣。
- 残差调用/Jacobian 调用/梯度调用的成本不同，计数不等价 FLOPs；请结合误差–耗时曲线。
- 这是单随机种子的基线验证，不足以证明普遍排名。接触合力准确也不代表压力局部形状准确；独立网格、独立 PDE 残差与力平衡检查用于揭示训练点欠采样和过拟合。

## 运行

已有环境可直接执行：

```powershell
Set-Location F:\RA\pinn_2d_optimizer_lab
F:\RA\.venv\Scripts\python.exe train.py --out runs/my_run
```

完整基线：两种 PDE × 全部 10 种方法（默认值）。输出目录非空时默认拒绝覆盖。`--resume` 仅在配置、初始化、采样匹配时复用已完成方法，完整保留其原始记录。

```powershell
# 同配置断点续跑
F:\RA\.venv\Scripts\python.exe train.py --out runs/baseline --resume

# 延长三种重点方法；必须使用新的结果目录
F:\RA\.venv\Scripts\python.exe train.py --steps 5000 --optimizers adam adam_lbfgs soap --out runs/focus_5000

# 加噪声，重新比较全部方法
F:\RA\.venv\Scripts\python.exe train.py --noise 0.01 --out runs/noise_1pct

# 小规模代码检查，不代表正式效果
F:\RA\.venv\Scripts\python.exe train.py --steps 4 --record-every 1 --width 8 --n-contact 12 --n-heat 16 --out runs/quick --no-gif
```

新装环境（所有新文件在 F 盘）：

```powershell
python -m venv F:\RA\.venv
F:\RA\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
F:\RA\.venv\Scripts\python.exe -m pip install -r F:\RA\pinn_2d_optimizer_lab\requirements.txt
```

当前环境使用已有系统 NumPy/SciPy/Matplotlib 以及 F 盘安装的 PyTorch/scikit-fem 等包；精确版本记录于 `provenance.json`。如需严格隔离，可新建一个不含 `--system-site-packages` 的虚拟环境再安装依赖。

## 验证与重建报告

```powershell
F:\RA\.venv\Scripts\python.exe -m pytest tests -q
F:\RA\.venv\Scripts\python.exe tests/validate_run.py runs/baseline
F:\RA\.venv\Scripts\python.exe audit.py runs/baseline
F:\RA\.venv\Scripts\python.exe report.py runs/baseline/results.json
```

`tests/validate_run.py` 从全部保存模型重新计算各项 loss、二维场、压力与力，核验展示记录。`audit.py` 在 1,024 个独立 Sobol 点检查 PDE 和本构残差，并比较预测底部反力与预测顶部压力合力；检查只在训练之后进行。浏览器检查可额外安装 `playwright`，使用本机 Edge：`python tests/browser_check.py runs/baseline/dashboard.html`。

## 文件

- `physics.py`：二维网络、精确导数链、PDE、混合弹性、压力头和指标。
- `reference.py`：独立 FEM、位移观测、网格与力平衡检查。
- `optimizers.py`：沿用的 10 种方法。
- `train.py`：公平初始化、固定采样、训练记录、模型与断点续跑。
- `report.py` / `dashboard.template.html`：离线交互报告、科学图和动画。
- `runs/baseline/heat2d`、`contact2d`：各方法模型、CSV、原始 JSON、初始模型和训练点。
- `runs/baseline/results.json`：两种 PDE 的完整结果及快照。
- `runs/baseline/contact_reference.json`：FEM 场、传感器读数和参考核验数据。
- `runs/baseline/summary.csv`、`experiment_report.md`：最终指标与实验说明。
- `runs/baseline/*_training_curves.png/.svg`、`*_error_vs_time.png`、`*_final_fields.png`、`contact2d_pressure_and_force.png`、`*_evolution.gif`：可分享图与动画。
- `runs/baseline/independent_physics_audit.json`、`provenance.json`：独立物理核验、依赖版本与源码哈希。

## 文献和 API

- [scikit-fem 官方 API](https://scikit-fem.readthedocs.io/en/stable/api.html)
- [scikit-fem 官方线弹性示例](https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex11.py)
- [Solving Forward and Inverse Problems of Contact Mechanics using PINNs](https://arxiv.org/abs/2308.12716)：混合变量与接触 PINN 的相关研究；本项目只取已知支撑区的载荷反演子问题，没有复现论文中的完整接触互补条件。
- [SOAP](https://arxiv.org/abs/2409.11321)、[MultiAdam](https://proceedings.mlr.press/v202/yao23c.html)、[ConFIG](https://arxiv.org/abs/2408.11104)、[PCGrad](https://arxiv.org/abs/2001.06782)。
