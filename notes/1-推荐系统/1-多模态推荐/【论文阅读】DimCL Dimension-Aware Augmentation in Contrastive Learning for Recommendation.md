# 【论文阅读】DimCL: Dimension-Aware Augmentation in Contrastive Learning for Recommendation

标题：Dimcl：推荐对比学习中的维度感知增强。

发表：KDD 2025

关键词：

Recommender system, collaborative filtering, dimension-aware augmentation, contrastive learning, self-supervised learning



## 摘要

对比学习( CL )在解决协同过滤( CF )推荐系统( RSs )中的数据稀疏性问题方面取得了显著的成功。其关键原理是在给定用户-项目交互图的情况下，生成不同的增强视图。

**问题：**然而，先前的工作主要集中在通过随机函数进行增广，例如通过将扰动均匀地注入到不同的隐藏维度中。如果没有精细的控制，增强的隐藏表示可能包含对CL有害而与RS无关的噪声维度。由于以下两个主要瓶颈，去除特定维度的噪声是一项具有挑战性的任务。它很难( i )区分不同维度对CL的功效，( ii )弥合CL和RS之间的语义鸿沟。忽视这些限制可能会导致增强视图的隐藏维度中存在冗余、误报和不相关的噪声

**方法：**在本文中，我们从鲁棒学习和课程学习的角度解决了上述挑战，并提出了一种新颖的推荐对比学习维度感知增强（DimCL）。在DimCL中，我们首先从理论上分析了CL和RS不同维度的难易程度。  经过深入分析，我们提出了两个命题，揭示了增强视图不同维度的梯度可能与优化 CL 和 RS 的学习难度有关。梯度的比较可以提供可检测的信号来反映 CL 不同维度的功效以及 CL 和 RS 之间的语义差距。根据分析结果，我们设计了三个去噪因子，这可以帮助 DimCL 将难以学习的维度识别为冗余或误报噪声，并在不同的增强视图中精确定位维度，将 RS 的不一致困难视为不相关噪声，而不需要额外的监督标签。去噪后，DimCL 可以去除维度级噪声，以减少不必要的难度，使 CL 更容易，并在 RS 中保持更一致的难度。



## 方法

![11-10-3](E:\2-科研\论文阅读笔记\fig\11-10-3.png)

### Training Data

输入是用户–物品交互图/矩阵（左下角的 0/1）。这只是原始监督信号，用来算推荐损失 $\mathcal L_{\text{Rec}}$（通常是 BPR）。

### Representation Learning Module

- 任何基础编码器 $F_{\text{enc}}$（论文里常用 LightGCN 或含 CL 的编码器）。

- 得到**基础嵌入**：用户 $\mathbf h_u$、物品 $\mathbf h_v$。此时还没加扰动。

###  Contrastive View Generation Module

分别对用户侧、物品侧用一个小 MLP 生成**可学习的逐维扰动**：$\boldsymbol\varepsilon'_u,\boldsymbol\varepsilon''_u$ 与 $\boldsymbol\varepsilon'_v,\boldsymbol\varepsilon''_v$。

先形成两套“对比视图”：
$$
\tilde{\mathbf h}'_u=\mathbf h_u+\boldsymbol\varepsilon'_u,\quad
\tilde{\mathbf h}''_u=\mathbf h_u+\boldsymbol\varepsilon''_u
$$
物品侧同理。这里的 $\tilde{\cdot}$ 还是“**未去噪**”的视图（图中用虚线块表示将被筛掉的维度）。

### **Dimension-aware Augmentation Module**（核心）

这块用两条**梯度信号**来判别每一维扰动是否“该保留”：

- 从推荐损失 $\mathcal L_{\text{Rec}}$ 回传来的梯度——反映**排序难度**；
- 从对比损失 $\mathcal L_{\text{CL}}$ 回传来的梯度——反映两视图的**匹配难度/偏离度**。

依据梯度构成三类“**去噪因子**”：

1. **Irrelevant（无关）**：对推荐损失影响弱或两视图对排序影响不一致的维度；
2. **Redundant（冗余）**：梯度幅度长期很小，信息量低的维度；
3. **False-positive（伪正）**：让正对比样本更“远”（使 $\mathcal L_{\text{CL}}$ 更大）的维度。

把这三类指标送入一个逐维的门控（实现上用带 **Gumbel-Softmax** 的二分类/采样），得到权重向量 $\mathbf w\in\{0,1\}^d$ 或其平滑近似，然后对扰动做**逐维开关**：
$$
\mathbf h'_u=\mathbf h_u+\mathbf w_u\!\odot\!\boldsymbol\varepsilon'_u,\quad
\mathbf h''_u=\mathbf h_u+\mathbf w_u\!\odot\!\boldsymbol\varepsilon''_u
$$
物品侧同理。这样就“把坏维度的噪声剪掉”。

### Joint Optimizing

用“去噪后的视图”做 **对比学习损失** $\mathcal L_{\text{CL}}$，用**基础嵌入**（或视图）做 **推荐损失** $\mathcal L_{\text{Rec}}$。

总目标：$\mathcal L=\mathcal L_{\text{Rec}}+\lambda\,\mathcal L_{\text{CL}}+\text{reg}$。训练时门控由两路梯度驱动，推理阶段只用基础嵌入/主干编码器，**无额外推理开销**。