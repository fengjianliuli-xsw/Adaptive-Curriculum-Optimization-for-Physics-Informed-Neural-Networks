"""Generate a two-page research note from existing logs only; no training is imported."""
from pathlib import Path
import hashlib,html,io,json,sys
ROOT=Path(__file__).resolve().parent
# Reuse the bundled PDF libraries without installing into or changing experiment environments.
sys.path.append(r'C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from pypdf import PdfReader

BASE=Path('F:/RA/pinn_2d_optimizer_lab/runs/baseline')
SCHEDULE=Path('F:/RA/pinn_scheduler_lab/runs/scheduler60')
ASSETS=ROOT/'assets';TMP=ROOT/'tmp/pdfs'
METHODS=['adam','lbfgs','adam_lbfgs','soap','lm']
LABELS=dict(adam='Adam',lbfgs='L-BFGS',adam_lbfgs='Adam → L-BFGS',soap='SOAP*',lm='GN/LM*')
COLOR=dict(adam='#3779b5',lbfgs='#50874e',adam_lbfgs='#bd852c',soap='#9470b2',lm='#1b8f88')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


REFS=[
 ('Kingma & Ba. Adam: A Method for Stochastic Optimization. ICLR, 2015.','https://arxiv.org/abs/1412.6980'),
 ('Loshchilov & Hutter. Decoupled Weight Decay Regularization. ICLR, 2019.','https://arxiv.org/abs/1711.05101'),
 ('Liu & Nocedal. On the limited memory BFGS method for large scale optimization. Math. Program. 45:503–528, 1989.','https://doi.org/10.1007/BF01589116'),
 ('Marquardt. An Algorithm for Least-Squares Estimation of Nonlinear Parameters. J. SIAM 11:431–441, 1963.','https://doi.org/10.1137/0111030'),
 ('Rathore et al. Challenges in Training PINNs: A Loss Landscape Perspective. ICML, 2024.','https://proceedings.mlr.press/v235/rathore24a.html'),
 ('Gupta et al. Shampoo: Preconditioned Stochastic Tensor Optimization. ICML, 2018.','https://proceedings.mlr.press/v80/gupta18a.html'),
 ('Vyas et al. SOAP: Improving and Stabilizing Shampoo using Adam for Language Modeling. ICLR, 2025.','https://openreview.net/attachment?id=IDxZhXrpNf&name=pdf'),
 ('Liu, Chu & Thuerey. ConFIG: Towards Conflict-Free Training of Physics Informed Neural Networks. ICLR, 2025.','https://arxiv.org/abs/2408.11104'),
 ('Hwang & Lim. Dual Cone Gradient Descent for Training Physics-Informed Neural Networks. NeurIPS, 2024.','https://papers.nips.cc/paper_files/paper/2024/hash/b2b781badeeb49896c4b324c466ec442-Abstract-Conference.html'),
 ('Bischof & Kraus. Multi-Objective Loss Balancing for Physics-Informed Deep Learning. CMAME 439:117914, 2025.','https://doi.org/10.1016/j.cma.2025.117914'),
 ('Krishnapriyan et al. Characterizing possible failure modes in physics-informed neural networks. NeurIPS, 2021.','https://papers.nips.cc/paper_files/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html'),
]

CATEGORIES=[
 ['类别 / 代表方法','主要作用与代价','仓库实际记录'],
 ['一阶自适应：Adam / AdamW [1,2]','逐参数矩统计，更新成本低','Adam 已测；AdamW 未测'],
 ['拟牛顿：BFGS / L-BFGS [3]','割线曲率与线搜索；closure 可重复','L-BFGS、Adam→L-BFGS 已测'],
 ['二阶/残差：GN、LM、Newton-CG、NNCG [4,5]','GN用 JᵀJ；Newton-CG/NNCG用HVP/CG','仅阻尼 GN/LM* 已测，其余未测'],
 ['结构化预条件：Shampoo / SOAP [6,7]','分层矩阵统计与基变换；非完整Hessian','SOAP* 已测；Shampoo 未单测'],
 ['多目标方向：ConFIG / DCGD [8,9]','协调冲突梯度；需多任务梯度','ConFIG+GD、DCGD-Center+GD 已测'],
 ['加权 / 课程：ReLoBRaLo、课程学习 [10,11]','调权或改变训练任务/难度','ReLoBRaLo、固定/规则顺序已测'],
]

MOTIVATION='PDE 微分算子可能导致残差优化病态 [5]；PDE 与初边值约束的 Jacobian/曲率结构不同，梯度方向也可能冲突。问题不只在于如何下降总损失，还在于如何保留已经满足的物理约束。本文汇总已有预实验，并提出一个尚未实现的方向曲率诊断构想。'
CONFIG='主表为二维稳态热传导 −ΔT=q：单位方域、零 Dirichlet；T*=sin(πx)sin(πy)+0.2sin(3πx)sin(2πy)。2-24-24-1 tanh（697 参数），内点128、边界64；Lf为除以2π²后的PDE残差MSE，Lbc为边界MSE，等权。CPU单线程float64、seed=42、同初始化/采样，均1200外层步。[E2]'
QUESTION='能否仅用少量方向曲率，判断下一阶段应学 PDE、BC 还是 Joint，并降低已满足约束的退化风险，而不执行完整 GN/LM？目标选择在形式上可视为离散权重/切换控制；不能仅以“动态选任务”主张新颖性，相关加权与课程研究已存在 [10,11]。'
MATH1='令下标 i 表示物理约束块，n 为参数数、m_i 为残差数。残差按固定物理尺度归一化，并吸收 MSE 的样本数与固定权重，使以下平方范数与代码损失一致。'
MATH2='GN 忽略残差加权的二阶导数项，一般只是局部近似：残差或步长大时可能失准。边界残差恰为零且映射足够光滑时，下式成立；仅当更新方向不在边界 Jacobian 的零空间中，二次项才严格为正。这是已知最小二乘结构，不是新定理。'
ALGORITHM='固定基础优化器；从同一 Adam/SOAP 状态模拟 PDE、BC、Joint 三种有符号候选方向，未接受前不提交动量。以正步长缩放得到候选更新，对归一化物理探针上的各约束，预测改善量为：'
ALGOSTEPS=[
 '触发诊断：每 K 步或约束下降停滞/接近阈值时，估计状态；第一版不用控制器网络。',
 '风险守护：用线性化残差按分量检查已满足约束的容许阈值；在可行候选中按预测收益/预计总成本排序。无可行候选则缩短步长并检查 Joint/BC；小梯度不能替代可行性判断。',
 '检验并更新：试探一步，用实际/预测下降比检验局部模型，借鉴 LM/信赖域缩放 α；接受后提交优化器状态。一个方向的预测不保证随后 K 步，须限制执行窗口并重新检查。',
]
JVP='只计算方向JVP，不形成完整Jacobian或Hessian；三候选×两类约束最多需6个分块JVP/次诊断。PINN残差含空间导数，JVP可能涉及较高阶混合求导；触发、候选生成、回退、探针与动量状态成本都须计入，效率收益尚未验证。'
PLANROWS=[
 ['对照（全部为未来计划）','固定设置与主要辨别目标'],
 ['① 固定权重 Adam / SOAP；② BC→PDE→Joint','基础解法与精调阶段长度基线；两种优化器分开比较'],
 ['③ ReLoBRaLo；④ 仅一阶信息的目标调度','一阶调度使用损失进展、梯度及相同候选/守护检查'],
 ['⑤ GN-JVP曲率目标调度；⑥ GN仅调步长','⑤与④检验曲率增量；⑤与⑥区分选任务和调步长'],
]
HYPOTHESES='H1：在BC近满足处，对比一阶/GN对边界反弹的预测误差、校准与风险排序；H2：固定候选、步长控制及调参预算，用④⑤⑥区分选目标与调步长的收益；H3：包含全部诊断与回退，比较达到预注册精度的墙钟耗时。'
PLANSET='先用二维热传导与Re=20 Kovasznay NS，固定网络/采样/验证点，5个共同种子、每组120秒端到端训练预算。到达目标须解/速度相对L²≤1%且e_bc≤.03；未达记删失。报告PDE/BC loss、Time-to-target、wall、反弹峰值/超阈次数、调度成本；H3另用免额外诊断基线，分项profile Jacobian/JVP、Gram、solve与回退，并做K/最小驻留消融。'


def make_plot():
    ASSETS.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':6.4})
    fig,axs=plt.subplots(1,2,figsize=(7.25,1.85),layout='constrained')
    for name in METHODS:
        r=read(BASE/'heat2d'/f'{name}.json');h=r['history']
        axs[0].semilogy([x['train_seconds'] for x in h],[x['loss_total'] for x in h],label=LABELS[name].replace('→','to'),color=COLOR[name],lw=1.25)
    axs[0].set(title='(a) 2D heat: optimizer updates only',xlabel='Optimization time (s)',ylabel='Training loss: Lf + Lbc',xlim=(0,100))
    axs[0].legend(ncol=2,loc='upper right',framealpha=.9,handlelength=1.4,columnspacing=.7)
    for name,color in [('A','#3779b5'),('D','#bd852c'),('E','#1b8f88')]:
        r=read(SCHEDULE/f'ns2d_{name}_42/result.json');h=r['history']
        axs[1].semilogy([x['charged_seconds'] for x in h],[x['e_bc'] for x in h],label={'A':'A Joint','D':'D ReLoBRaLo','E':'E Rule switching'}[name],color=color,lw=1.05)
    axs[1].axhline(.03,color='#707780',ls='--',lw=.8,label='BC tolerance = 0.03')
    axs[1].set(title='(b) NS: boundary constraint rebound',xlabel='Accounted training time (s)',ylabel='Normalized BC error (RMSE)',xlim=(0,60))
    axs[1].legend(ncol=2,loc='upper right',handlelength=1.3,framealpha=.92,columnspacing=.5)
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(ASSETS/'loss_time_and_boundary.png',dpi=300);fig.savefig(ASSETS/'loss_time_and_boundary.svg');plt.close(fig)


def formula(tex,name,fontsize=12):
    fig=plt.figure(figsize=(8,1));fig.patch.set_alpha(0)
    text=fig.text(0,0,'$'+tex+'$',fontsize=fontsize,color='#123c46')
    path=TMP/(name+'.png');fig.savefig(path,dpi=400,bbox_inches='tight',pad_inches=.035,transparent=True);plt.close(fig)
    return path


EQ_GN=r'\Delta\mathcal{L}_i\ \approx\ g_i^\top d+\frac{1}{2}\|J_i d\|_2^2'
EQ_BC=r'r_{bc}=0\ \Longrightarrow\ g_{bc}=0,\qquad\Delta\mathcal{L}_{bc}=\frac{1}{2}\|J_{bc}d\|_2^2+O(\|d\|_2^3)'
EQ_DEF=r'\theta\in\mathbb{R}^n,\quad r_i\in\mathbb{R}^{m_i},\quad J_i=\frac{\partial r_i}{\partial\theta}\in\mathbb{R}^{m_i\times n},\quad\mathcal{L}_i=\frac{1}{2}\|r_i\|_2^2,\quad g_i=J_i^\top r_i'
EQ_PRED=r'd_a=\alpha_a p_a,\quad\widehat I_i(a)=-\alpha_a g_i^\top p_a-\frac{\alpha_a^2}{2}\|J_i p_a\|_2^2,\quad a\in\{\mathrm{PDE},\mathrm{BC},\mathrm{Joint}\}'


def observations():
    runs={name:read(BASE/'heat2d'/f'{name}.json') for name in METHODS}
    lm=runs['lm']['history'][-1];soap=runs['soap']['history'][-1]
    lm1=read(Path(r'C:\Users\ASUS\Documents\ChatGPT\RA\pinn_optimizer_lab\runs\demo\lm.json'))['history'][-1]
    soap1=read(Path(r'C:\Users\ASUS\Documents\ChatGPT\RA\pinn_optimizer_lab\runs\demo\soap.json'))['history'][-1]
    return [
      f'精度/成本：二维热传导LM终点耗时为SOAP的{lm["train_seconds"]/soap["train_seconds"]:.1f}倍；一维对应{lm1["train_seconds"]/soap1["train_seconds"]:.1f}倍 [E1,E2]。但首次记录达到1%误差，LM/L-BFGS/SOAP分别为3.18/2.87/3.85秒；仅LM达到0.1%（4.68秒）。等步终点比不能当等精度效率比；GN/LM是同一阻尼实现。',
      '耗时归因尚缺证据：代码每轮用jacrev显式构造J，再构造Gram、稠密求解并试探阻尼；热2D J为192×697、1D为272×697。没有分段profile，不能断言Jacobian或solve主导；没有GPU显存记录。',
      '损失下降不等于物理成功：二维载荷反演LM的训练平衡MSE约1.19e-12，但留出点为1.00e-2；圆柱接触的有限主结果压力误差约94%–98%，尚未成功 [E2,E3]。',
      '调度负结果：等60秒计费预算下，热传导E/A/D误差为0.198%/0.535%/0.064%；NS速度为22.03%/0.639%/0.682%。E切换95次，末e_bc=0.24；当前规则未胜过强基线，失败原因需消融 [E4]。',
    ]


def make_md(runs,obs):
    lines=['# PINN Optimization: Preliminary Experiments and Research Ideas','', '*Curvature-Guided Adaptive Curriculum Optimization*','',
           'Research Note | 2026-11-01 | 单种子预实验与待验证构想，不主张新颖性或已验证收益。','',
           '## 第1页：调研与初步实验','', '### Research Motivation','',MOTIVATION,'','### Existing Optimization Methods','']
    lines += ['|'+'|'.join(CATEGORIES[0])+'|','|---|---|---|']+['|'+'|'.join(row)+'|' for row in CATEGORIES[1:]]
    lines += ['','另测SGD+Momentum、RMSProp、MultiAdam、PCGrad。*为教学性自实现核心，不是官方完整实现复现。','', '### Preliminary Experiments','',CONFIG,'',
        '|方法|最终 Lf+Lbc|相对L² (%)|更新计时(s)|循环wall(s)|外层步|','|---|---:|---:|---:|---:|---:|']
    for name in METHODS:
        r=runs[name];f=r['history'][-1]
        lines.append(f"|{LABELS[name]}|{f['loss_total']:.3e}|{f['relative_l2']*100:.5f}|{f['train_seconds']:.2f}|{r['total_wall_seconds']:.2f}|{f['step']}|")
    lines += ['', '![真实日志：Loss-Time与边界反弹](assets/loss_time_and_boundary.png)','',
        '图1：(a) 上表热传导原始检查点，优化计时不含诊断。(b) Kovasznay Re=20、2-32-32-32-3、内点256/BC128，Adam lr=.001、K=50；计费秒包含诊断与策略，循环wall约61–62秒。两图计时口径不同，不跨图比较速度。[E2,E4]','', '### Key Observations','']
    lines += [f'{i+1}. {text}' for i,text in enumerate(obs)]
    lines += ['', '以上为seed=42的终点，不是统计推断；方法学习率和线搜索代价不同，未做同等调参搜索。所有主实验已有模型/日志复核；完整配置、其他方法和负结果见experiment_evidence.md。','',
        '## 第2页：研究构想（尚未实现与验证）','', '### Research Question','',QUESTION,'','### Mathematical Motivation','',MATH1,'', '\[',EQ_DEF,'\]','', '\[', EQ_GN,'\]','',MATH2,'','\[',EQ_BC,'\]','',
        '### Preliminary Algorithm Idea','',ALGORITHM,'','\[',EQ_PRED,'\]','']
    lines += [f'{i+1}. {text}' for i,text in enumerate(ALGOSTEPS)]
    lines += ['',JVP,'', '### Planned Experiments','', '以下是下一轮计划，不将已有A–E结果误称为这些消融已完成。','',
              '|'+ '|'.join(PLANROWS[0])+'|','|---|---|']+['|'+'|'.join(row)+'|' for row in PLANROWS[1:]]
    lines += ['',HYPOTHESES,'',PLANSET,'', '### References','']
    for i,(text,url) in enumerate(REFS,1):lines.append(f'[{i}] [{text}]({url})')
    lines += ['', '### 实验来源标识','',
        r'[E1] `C:\Users\ASUS\Documents\ChatGPT\RA\pinn_optimizer_lab\runs\demo`；',
        r'[E2] `F:\RA\pinn_2d_optimizer_lab\runs\baseline`；',
        r'[E3] `F:\RA\hertz_cylinder_lab\runs\kkt_all11`；',
        r'[E4] `F:\RA\pinn_scheduler_lab\runs\scheduler60`。',
        '主文数值来自逐方法JSON/CSV；独立物理审计来自相应audit文件；source SHA256与本次只读复核输出见experiment_inventory.json。']
    (ROOT/'PINN_Optimization_Research_Note.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def make_pdf(runs,obs):
    pdfmetrics.registerFont(TTFont('CJK',r'C:\Windows\Fonts\msyh.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CJKB',r'C:\Windows\Fonts\msyhbd.ttc',subfontIndex=0))
    pdfmetrics.registerFontFamily('CJK',normal='CJK',bold='CJKB',italic='CJK',boldItalic='CJKB')
    W,H=595.276,841.89;M=35;cw=W-2*M
    style=ParagraphStyle('body',fontName='CJK',fontSize=8.8,leading=13.0,textColor=colors.HexColor('#253440'),spaceAfter=4,wordWrap='CJK')
    small=ParagraphStyle('small',parent=style,fontSize=7.4,leading=10.5,spaceAfter=3,textColor=colors.HexColor('#566774'))
    tablebody=ParagraphStyle('table',parent=style,fontSize=7.7,leading=11.1,spaceAfter=0)
    refstyle=ParagraphStyle('ref',fontName='Helvetica',fontSize=6.7,leading=8.6,textColor=colors.HexColor('#495a64'),spaceAfter=1.2)
    heading=ParagraphStyle('heading',parent=style,fontName='CJKB',fontSize=10.6,leading=14.5,spaceBefore=6,spaceAfter=5,textColor=colors.HexColor('#136f72'))
    title=ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=17.5,leading=20,textColor=colors.HexColor('#153e4b'),spaceAfter=4)
    subtitle=ParagraphStyle('subtitle',fontName='Helvetica-Oblique',fontSize=10.4,leading=13,textColor=colors.HexColor('#21837f'),spaceAfter=7)
    def P(t,s=style):return Paragraph(html.escape(t),s)
    def head(t):return P(t,heading)
    def table(rows,widths):
        data=[[P(str(c),tablebody) for c in row] for row in rows]
        t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf3f3')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#47918d')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d9e1e4')),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('VALIGN',(0,0),(-1,-1),'TOP')]))
        return t
    def eq(tex,name,box=False):
        path=formula(tex,name);im=PILImage.open(path);width=min(cw-24,im.width/400*72);height=width*im.height/im.width
        pic=Image(str(path),width=width,height=height);pic.hAlign='CENTER'
        if not box:return [Spacer(1,2),pic,Spacer(1,4)]
        t=Table([[pic]],colWidths=[cw]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#edf6f4')),('BOX',(0,0),(-1,-1),.65,colors.HexColor('#8dbbb4')),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));return [t,Spacer(1,5)]
    story=[P('PINN Optimization:',title),P('Preliminary Experiments and Research Ideas',title),P('Curvature-Guided Adaptive Curriculum Optimization',subtitle),P('2026-11-01  |  单种子预实验与待验证构想；不主张新颖性或已验证收益。',small),head('Research Motivation | 研究动机'),P(MOTIVATION),head('Existing Optimization Methods | 方法分类'),table(CATEGORIES,[186,185,cw-371]),P('另测SGD+Momentum、RMSProp、MultiAdam、PCGrad。*为教学性自实现核心，非官方完整实现复现。',small),head('Preliminary Experiments | 已完成的对照'),P(CONFIG,small)]
    rows=[['方法','最终 Lf+Lbc','相对 L² (%)','更新秒','循环wall秒','外层步']]
    for name in METHODS:
        r=runs[name];f=r['history'][-1];rows.append([LABELS[name],f"{f['loss_total']:.3e}",f"{f['relative_l2']*100:.5f}",f"{f['train_seconds']:.2f}",f"{r['total_wall_seconds']:.2f}",str(f['step'])])
    story += [table(rows,[107,99,92,64,87,cw-449]),Spacer(1,5)]
    img=PILImage.open(ASSETS/'loss_time_and_boundary.png');story.append(Image(str(ASSETS/'loss_time_and_boundary.png'),width=cw,height=cw*img.height/img.width))
    story += [P('图1 真实检查点。(a) 上表数据，更新秒不含诊断。(b) NS Re=20、2307参数、内点256/BC128，Adam lr=.001，K=50；计费秒含诊断/策略，循环wall约61–62秒。两图计时口径不同。[E2,E4]',small),head('Key Observations | 结论与证据边界')]
    for i,text in enumerate(obs,1):story.append(P(f'{i}. {text}',small))
    story += [P('单种子、方法学习率不同且未等量调参；不支持普遍排名或显著性推断。完整配置、全部方法与来源见 experiment_evidence.md。',small),PageBreak(),head('Research Question | 待检验的问题'),P(QUESTION),head('Mathematical Motivation | 核心数学依据'),P(MATH1)]
    story += eq(EQ_DEF,'definitions')+eq(EQ_GN,'gn',box=True)+[P(MATH2)]+eq(EQ_BC,'bc',box=True)
    story += [head('Preliminary Algorithm Idea | 曲率辅助选目标（构想）'),P(ALGORITHM)]+eq(EQ_PRED,'pred')
    for i,text in enumerate(ALGOSTEPS,1):story.append(P(f'{i}. {text}'))
    story += [P(JVP,small),head('Planned Experiments | 下一轮计划，不是已完成结果'),P('已有单种子A–E提供动机；以下按新协议重跑，并新增曲率与步长消融。',small),table(PLANROWS,[cw*.51,cw*.49]),Spacer(1,4),P(HYPOTHESES),P(PLANSET,small),head('References | 已核实的正式文献')]
    for i,(text,url) in enumerate(REFS,1):
        # Short, clickable bibliographic entries; full URLs are preserved in the editable note.
        story.append(Paragraph(f'[{i}] <a href="{html.escape(url,quote=True)}" color="#316c86">{html.escape(text.replace("–","-"))}</a>',refstyle))
    story.append(P('证据标识：[E1] 1D demo；[E2] 2D baseline；[E3] Hertz kkt_all11；[E4] scheduler60。绝对路径、原始数值、缺失项及复核日志见随附 experiment_evidence.md。',small))
    def page(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#c1d4d7'));canvas.setLineWidth(.5);canvas.line(M,29,W-M,29)
        canvas.setFont('Helvetica',7);canvas.setFillColor(colors.HexColor('#687d87'));canvas.drawString(M,18,'PINN Optimization | Preliminary experiments and research ideas');canvas.drawRightString(W-M,18,f'{doc.page} / 2');canvas.restoreState()
    path=ROOT/'PINN_Optimization_Research_Note.pdf'
    doc=SimpleDocTemplate(str(path),pagesize=(W,H),leftMargin=M,rightMargin=M,topMargin=25,bottomMargin=37,title='PINN Optimization: Preliminary Experiments and Research Ideas',author='',subject='Curvature-Guided Adaptive Curriculum Optimization; preliminary evidence and proposed experiments')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    pdf=PdfReader(path);print('PDF pages:',len(pdf.pages),'bytes:',path.stat().st_size)
    for i,p in enumerate(pdf.pages,1):print('page',i,'text characters',len(p.extract_text() or ''))
    if len(pdf.pages)!=2:raise RuntimeError('Layout must be exactly 2 pages; inspect and adjust flow before delivery')


def main():
    audit=read(ROOT/'experiment_inventory.json')
    assert all(x['returncode']==0 for x in audit['verification']),'Resolve audit discrepancies before authoring conclusions'
    # Ensure the inventory's source records have not changed between checking and authoring.
    for path,expected in audit['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    runs={name:read(BASE/'heat2d'/f'{name}.json') for name in METHODS};obs=observations()
    make_plot();make_md(runs,obs);make_pdf(runs,obs)


if __name__=='__main__':main()
