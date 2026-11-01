# PINN 优化器实验室

用真实训练记录对比 10 种优化方法，包含可离线打开的中文交互仪表盘、损失/精度曲线和预测演化 GIF。所有优化器共享网络初始化和固定训练采样点。

## 快速打开已生成结果

双击 `runs/demo/dashboard.html`。不需要服务器或联网。点击“播放训练”，或拖动进度条；切换物理时间查看不同温度剖面。支持按外层迭代、实际训练耗时、残差计算次数同步。每张预测卡片显示实际采用的记录 step；记录之间不伪造插值，提前停止的方法保持最后结果。

## 安装和运行

Python 3.10+。从本目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --steps 1200 --record-every 10 --out runs/my_run
```

本工作区已安装的环境位于上一级 `.venv`，可直接用：

```powershell
..\.venv\Scripts\python.exe train.py --out runs/my_run
..\.venv\Scripts\python.exe -m pytest tests -q
```

随附 `requirements-demo-lock.txt` 记录生成演示时的直接依赖版本，`manifest.json` 记录实际运行环境。
可用 `python tests/validate_run.py runs/demo` 从保存的模型重算损失与预测，核验记录。可选浏览器检查需额外安装 `playwright` 并有本机 Edge：`python tests/browser_check.py runs/demo/dashboard.html`。

只比较部分方法：

```powershell
..\.venv\Scripts\python.exe train.py --optimizers adam lbfgs adam_lbfgs lm multiadam config soap --steps 2000 --out runs/longer
```

快速检查：`--steps 20 --record-every 5 --no-gif --out runs/quick`。
已有非空结果目录默认不覆盖；需要明确传入 `--overwrite`。重建报告：

```powershell
..\.venv\Scripts\python.exe report.py runs/demo/results.json
```

## PDE、解析解和损失

这是一个空间维度加时间的一维热传导问题，不是一维 ODE：

\[
u_t=\nu u_{xx},\quad \nu=0.1,\quad (x,t)\in[0,1]^2,
\]
\[
u(0,t)=u(1,t)=0,\qquad u(x,0)=\sin(\pi x)+0.3\sin(3\pi x).
\]
解析解：
\[
u^*(x,t)=e^{-0.1\pi^2t}\sin(\pi x)+0.3e^{-0.9\pi^2t}\sin(3\pi x).
\]
训练目标使用等权的三个均方损失：
\[
L=L_f+L_{IC}+L_{BC},\quad L_f=\mathrm{mean}(u_t-0.1u_{xx})^2.
\]
默认 128 个固定 Sobol 内部点、48 个初值点、每端 48 个边界点；网络为 `2 → 24 → 24 → 1`，tanh，697 个参数，输入映射到 [-1,1]，CPU float64 单线程。初始参数使用同一个 state_dict 的完整副本，优化器状态各自独立。

内部解析解从未进入训练 loss、步长接受或停止判断。初值标签是给定的物理条件。评估网格是固定 121×41 网格，不用于优化；另用独立 Sobol 点评估 PDE 残差。网格边界与给定初边值会有自然重合，因此相对 L2 是“整个固定网格误差”，不称为严格全体样本均未见过的测试集。

空间/时间导数通过 tanh 链式法则精确传播，参数梯度仍由 PyTorch 自动微分获得。例如 `h=tanh(z)` 时：`h_x=(1-h²)z_x`，`h_xx=(1-h²)z_xx-2h(1-h²)z_x²`。测试将此结果与输入自动微分逐项比对，避免以有限差分代替 PDE 导数。

## 精度指标

- 相对 L2：`||prediction - truth||₂ / ||truth||₂`，越低越好，是主要精度指标。
- RMSE 和最大绝对误差，保存在 CSV/JSON。
- 误差达标率：评估网格上满足 `|prediction-truth| <= 0.05` 的点占比。不是分类准确率，也不是 `1-relative_error`。真实值接近零时不使用不稳定的逐点相对误差。
- `test_pde_mse`：独立内部采样点上的 PDE 残差。

## 方法及公式

实现入口在 `optimizers.py` 和 `train.py`。学习率等配置固定写在 `SPECS` 中，结果也完整记录。

| 方法 | 核心更新或实现 |
|---|---|
| Adam | PyTorch Adam，lr=0.001，β=(0.9,0.999) |
| SGD + Momentum | PyTorch SGD，lr=0.01，momentum=0.9 |
| RMSProp | PyTorch RMSprop，lr=0.001，alpha=0.99 |
| L-BFGS | PyTorch LBFGS，lr=1，history=50，`max_iter=1`，强 Wolfe 线搜索；一次外层调用可做多次 closure |
| Adam → L-BFGS | 前一半外层步使用 Adam，剩余步使用 L-BFGS，切换时新建拟牛顿状态 |
| Gauss–Newton / LM | 精确残差 Jacobian，求 `(JᵀJ+μI)δ=-Jᵀr`，仅实际下降且预测/实际下降比满足阈值时接受，失败增大阻尼重试 |
| MultiAdam | PDE 和 (IC+BC) 两组，分别维护一、二阶矩，β₁=β₂=0.99，平均两个 Adam 更新；lr=0.001 |
| PCGrad + GD | 三个损失分别求梯度，随机顺序移除冲突分量，求和后直接 GD，lr=0.02；不再后接 Adam |
| ConFIG + GD | 三个损失梯度行归一化为 A；`v=Aᵀ(AAᵀ)^†1`，`u=v/||v||`，`d=(Σgᵢᵀu)u`，直接 GD，lr=0.02 |
| SOAP | 按层梯度相关矩阵构建左右特征基，在旋转坐标中做 Adam，lr=0.003，β=(0.95,0.95)，每 10 步更新基；偏置使用 Adam |

LM 的残差各组除以各自点数平方根，因此 `||r||²` 与等权训练损失完全一致。当残差数小于参数数时，使用等价的对偶公式 `δ=-Jᵀ(JJᵀ+μI)⁻¹r`。初始 μ=0.01，最多重试 10 次；梯度最大绝对值小于 1e-10 时提前停止，图中不将它伪装成完成全部步数。这里没有显式矩阵求逆。

SOAP 是透明的教学实现，包含动量、旋转基变换和对角二阶矩基变换，不含 weight decay、grafting 和大型矩阵分块。二阶矩基变换采用对角协方差近似，更新特征基的时序与一些官方实现有所不同，不应把它当作官方基准复现。ConFIG 采用伪逆截断 rtol=1e-10；零梯度会被排除，秩退化时不保证严格共同下降。PCGrad 多目标投影本身也不保证所有实际有限步均下降。

这批方法覆盖一阶、拟牛顿、残差二阶、结构化预条件、多目标平衡五类。没有实现 NNCG、SSBroyden、SQP 等上一轮讨论的所有方法；这些方法不应以其他近似算法冒名展示。

## 公平比较与限制

1. 所有方法使用相同网络参数、采样点、损失权重和最大外层步预算；第 0 步预测严格相同。每个结果保存初始化/采样 SHA-256。共同起点并不意味着各优化器应使用相同学习率。
2. 外层 step 不等价于计算预算：L-BFGS 线搜索、LM Jacobian、多目标多次反向都更昂贵，因此同时导出误差–实际训练耗时图。
3. `train_seconds` 只测优化步骤；记录指标、计算诊断、保存模型和生成图不计入。另保存整个单方法运行的 `total_wall_seconds`。CPU 同机顺序运行，计时存在噪声。
4. `residual_evals` 只统计优化期间目标/残差函数调用，LM 的 Jacobian 调用计为一次，不代表与普通前向同成本；`gradient_evals` 记录反向/求 Jacobian 的调用次数，也不等价 FLOPs。诊断不进入这些计数。
5. 全部使用固定全批数据，无隐藏重采样、学习率调度、梯度裁剪或解析解监督。不同超参数和单随机种子的影响仍然存在。本实验用于理解训练流程，不是证明某个优化器普遍最好。正式比较需多种子、多 PDE、超参数预算控制。
6. 相邻记录点之间不插值。播放中的误差/达标率是全评估网格指标，预测卡片显示的是选定物理时间的剖面。

## 结果文件

| 文件 | 内容 |
|---|---|
| `dashboard.html` | 内嵌数据的离线交互演示，支持训练播放、拖动、时间切片、方法开关、图片/CSV 下载 |
| `prediction_evolution.gif` | 固定物理时间 t=0.5、多个优化器同时演化的动画 |
| `training_curves.png/.svg` | 总损失、三项损失、相对 L2、误差达标率 |
| `error_vs_time.png` | 实际训练耗时对比 |
| `final_predictions.png` | t=0、0.5、1 的最终预测 |
| `results.json` | 完整实验、训练轨迹和预测快照 |
| `*_history.csv` | 每项损失、误差、耗时、计数、梯度余弦、优化阶段 |
| `initial_state.pt` / `*_final.pt` | 原始初始化与各方法最终模型 |
| `sampling_and_reference.npz` | 全部训练点、评估点、参考解和独立残差检查点 |
| `manifest.json` / `summary.json` | 配置、环境、哈希与结果摘要 |

## 文献与 API

- [PyTorch LBFGS](https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html)
- [PyTorch jacrev](https://docs.pytorch.org/docs/stable/generated/torch.func.jacrev.html)
- [MultiAdam, ICML 2023](https://proceedings.mlr.press/v202/yao23c.html)
- [PCGrad, NeurIPS 2020](https://arxiv.org/abs/2001.06782)
- [ConFIG, ICLR 2025](https://arxiv.org/abs/2408.11104)
- [SOAP](https://arxiv.org/abs/2409.11321)
- [PINN 梯度对齐与 SOAP](https://arxiv.org/abs/2502.00604)
- [Gauss–Newton 自然梯度与 PINN](https://arxiv.org/abs/2402.10680)
