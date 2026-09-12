你的 MAIL 相比原始 IDFREE，主要创新可以概括为：用模态语义调节身份编码，再通过热门度惩罚补充语义邻居，并用结构对比学习兼顾协同信息与个体差异。 对应方法图，就是左侧的 MAIC 身份构建和中间的 CSL 结构学习两个核心模块。

先说明比较依据：你上传的 idfree.py 已经包含模态门控、反事实邻居和解耦损失，并非原始 IDFREE 的纯基线版本。因此，下面的创新比较以 IDFREE 原论文为准，具体实现以 mail.py 为准。

一、MAIL 相比 IDFREE 的主要创新

比较维度	原始 IDFREE	你的 MAIL
身份构建	将固定正弦位置编码直接加入模态表示	根据文本、图像内容生成门控，逐维调节位置编码
图结构	构建语义 KNN 图，并自适应调整边权	在基础图上补充经过热门度惩罚筛选的语义邻居
对比约束	主要进行文本与图像之间的对齐	保留跨模态对齐，新增跨层对齐与语义—结构解耦约束
核心关注	如何不用可学习 ID embedding 实现推荐	如何让 ID-free 身份更贴合内容，并减轻结构传播偏置

这里的 ID-free 指不使用每个用户、物品独立的可学习 ID 向量表，仍然允许使用节点编号生成固定位置编码。

你的创新逻辑是连贯的：

MAIC：解决“身份怎么构造”。 固定位置编码提供区分性，但不随内容变化，因此用模态语义控制它的注入方式。
反事实邻居增强：解决“从谁那里聚合”。 在语义相关的前提下，降低热门候选的选择优势，为低交互物品补充传播机会。
结构对比增强：解决“聚合之后保留什么”。 让表示吸收有效协同关系，同时保留节点自身的语义特点。

其中，用户历史内容平均、模态投影、基础相似图、LightGCN 和跨模态 InfoNCE 属于继承的技术基础，不宜单独列为新增贡献。

二、方法图左侧：MAIC 模态感知身份构建

1. 输入物品内容，并从历史交互得到用户内容。

图中四类输入分别是：

符号	含义
\(\mathbf{x}_i^v\)	物品 i 的图像特征
\(\mathbf{x}_i^t\)	物品 i 的文本特征
\(\mathbf{x}_u^v\)	用户交互过的物品图像特征的平均值
\(\mathbf{x}_u^t\)	用户交互过的物品文本特征的平均值

用户的模态特征计算为：

$$ \mathbf{x}_u^m = \frac{1}{|\mathcal{N}_u|} \sum_{i\in\mathcal{N}_u}\mathbf{x}_i^m, \qquad m\in\{t,v\}. $$

其中，\(\mathcal{N}_u\) 是用户的历史交互物品集合。

例如，一个用户经常购买运动鞋和运动服，其用户内容特征便会包含较强的运动相关信息。

2. 将不同模态投影到统一维度。

文本与图像分别经过：

全连接层 → Tanh → LayerNorm。

对应公式为：

$$ \mathbf{z}_x^m = \operatorname{LN} \left( \tanh\left( \mathbf{W}_m\mathbf{x}_x^m+\mathbf{b}_m \right) \right), \qquad x\in\mathcal{U}\cup\mathcal{I}. $$

得到统一维度的文本表示与图像表示：

$$ \mathbf{z}_x^t,\mathbf{z}_x^v\in\mathbb{R}^{d}. $$

同一模态下，用户和物品共享投影网络，便于在共同空间中建模。

3. 根据内容生成位置编码门控。

文本和图像各自经过线性层与 Sigmoid，得到逐维门控向量，控制每个维度注入多少身份信号：

$$ \boldsymbol{\gamma}_x^t = \sigma\left( \mathbf{W}_g^t\mathbf{z}_x^t+\mathbf{b}_g^t \right), $$ $$ \boldsymbol{\gamma}_x^v = \sigma\left( \mathbf{W}_g^v\mathbf{z}_x^v+\mathbf{b}_g^v \right). $$

然后融合两种模态的门控：

$$ \mathbf{g}_x = \alpha_p\boldsymbol{\gamma}_x^t + (1-\alpha_p)\boldsymbol{\gamma}_x^v. $$

用融合门控逐维调制固定位置编码：

$$ \mathbf{p}'_x = \mathbf{g}_x\odot\mathbf{p}_x. $$

其中，\(\odot\) 表示逐元素相乘。

固定的位置编码提供节点区分信息，随内容变化的门控决定如何使用这些信息。 这就是 MAIL 相比 IDFREE 直接加入固定 PE 的关键变化。

需要区分：\(\alpha_p\) 是配置的融合权重；真正随节点内容变化的是两个模态门控及其融合结果。

4. 注入身份信息，融合得到初始表示。

同一个调制位置编码加入两个模态：

$$ \widetilde{\mathbf{z}}_x^t = \mathbf{z}_x^t+\mathbf{p}'_x, \qquad \widetilde{\mathbf{z}}_x^v = \mathbf{z}_x^v+\mathbf{p}'_x. $$

再进行加权融合：

$$ \begin{aligned} \mathbf{e}_x^{(0)} &= \alpha_m\widetilde{\mathbf{z}}_x^t + (1-\alpha_m)\widetilde{\mathbf{z}}_x^v\\ &= \alpha_m\mathbf{z}_x^t + (1-\alpha_m)\mathbf{z}_x^v + \mathbf{p}'_x. \end{aligned} $$

因此，\(\mathbf{e}_x^{(0)}\) 是尚未经过图传播的、包含内容和身份信息的初始节点表示。

两个融合系数承担不同作用：

\(\alpha_m\)：控制文本与图像的内容融合比例。
\(\alpha_p\)：控制文本与图像的门控融合比例。

三、方法图中间上方：反事实邻居增强

这一部分决定物品能够从哪些其他物品获取信息。

1. 用不含位置编码的内容表示计算语义相似度。

先融合投影后的文本和图像：

$$ \mathbf{h}_i = \alpha_m\mathbf{z}_i^t + (1-\alpha_m)\mathbf{z}_i^v. $$

然后计算物品之间的余弦相似度：

$$ s_{ij} = \frac{ \mathbf{h}_i^{\top}\mathbf{h}_j }{ \|\mathbf{h}_i\|_2\|\mathbf{h}_j\|_2 }. $$

这里采用不含 PE 的表示，使邻居选择依据主要来自内容语义。

2. 对候选邻居的热门度施加惩罚。

首先，根据交互次数衡量热门度：

$$ \operatorname{pop}(j) = \log(1+n_j), $$

其中，\(n_j\) 是物品 j 的交互次数。

论文中的反事实选择分数为：

$$ s_{ij}^{cf} = \frac{ s_{ij} }{ \left(\operatorname{pop}(j)+\epsilon\right)^{\lambda_{cf}} }. $$

其中：

\(\epsilon\)：数值稳定项。
\(\lambda_{cf}\)：热门度惩罚强度。

对于语义相似度为正、相关程度接近的候选物品，热门度更高的候选受到更强惩罚，低交互候选更有机会进入邻居集合。

例如，某款运动鞋原先主要连接热门运动鞋；调整选择分数后，一些内容相似但交互较少的运动鞋也可能成为邻居。

这里“低曝光”是通过交互次数间接刻画的，并没有直接使用曝光日志。“反事实”更准确地表达为：减弱热门度影响后，哪些语义邻居可能被选中。 当前机制是受反事实思想启发的图增强，并没有显式估计因果干预效应。

3. 选择邻居，再融合到基础图。

按照调整后的分数选择邻居：

$$ \mathcal{N}_i^{cf} = \operatorname{TopK}_{j\in\mathcal{I},\,j\ne i} \left(s_{ij}^{cf},K_{cf}\right). $$

但反事实图的边权仍采用原始余弦相似度：

$$ R_I^{cf}(i,j) = \begin{cases} s_{ij}, & j\in\mathcal{N}_i^{cf},\\ 0, & \text{otherwise}. \end{cases} $$

也就是：

用惩罚后分数决定“选谁”，用原始语义相似度决定“边有多强”。

然后将反事实图融合到基础物品图：

$$ \mathbf{R}_I^{aug} = \mathbf{R}_I^{base} + \eta\mathbf{R}_I^{cf}. $$

这样，热门度影响邻居的选择，原始语义相似度控制信息传播强度。

随后将三类关系组合起来：

用户—用户语义关系。
用户—物品真实交互关系。
增强后的物品—物品关系。

整体邻接矩阵可以表示为：

$$ \mathbf{A}^{aug} = \begin{bmatrix} \mathbf{R}_U & \mathbf{R}\\ \mathbf{R}^{\top} & \mathbf{R}_I^{aug} \end{bmatrix}. $$

其中，\(\mathbf{R}\) 是用户—物品交互矩阵，\(\mathbf{R}_U\) 是用户—用户关系矩阵。

反事实新增边主要发生在物品侧；实际图传播同时包含用户和物品。

四、方法图中间下方：结构对比增强

增强图构建完成后，将左侧得到的初始表示输入 LightGCN：

$$ \mathbf{E}^{(l+1)} = \widehat{\mathbf{A}}^{aug}\mathbf{E}^{(l)}, $$ $$ \widehat{\mathbf{A}}^{aug} = \mathbf{D}^{-\frac12} \mathbf{A}^{aug} \mathbf{D}^{-\frac12}. $$

最终对初始层与各传播层取平均：

$$ \overline{\mathbf{E}} = \frac{1}{L+1} \sum_{l=0}^{L}\mathbf{E}^{(l)}. $$

因此：

表示	含义
\(\mathbf{e}_x^{(0)}\)	传播前的内容与身份表示
\(\mathbf{e}_x^{(1)}\)	聚合一跳邻居信息后的表示
\(\overline{\mathbf{e}}_x\)	初始层与各传播层融合后的最终表示

这一部分设计了两个互补约束。

1. 跨层对齐：让身份语义与相关交互的结构信息对应。

具体的正样本关系是：

$$ \mathbf{e}_u^{(0)} \longleftrightarrow \mathbf{e}_i^{(1)}, \qquad i\in\mathcal{P}(u), $$ $$ \mathbf{e}_i^{(0)} \longleftrightarrow \mathbf{e}_u^{(1)}, \qquad u\in\mathcal{P}(i). $$

也就是：

用户的初始表示，对齐其正样本物品的一层表示。
物品的初始表示，对齐相关用户的一层表示。

以用户侧为例，损失为：

$$ \mathcal{L}_A^u = -\log \frac{ \displaystyle\sum_{i\in\mathcal{P}(u)} \exp\left( \operatorname{sim} \left(\mathbf{e}_u^{(0)},\mathbf{e}_i^{(1)}\right)/\tau_A \right) }{ \displaystyle\sum_{j\in\mathcal{B}_I} \exp\left( \operatorname{sim} \left(\mathbf{e}_u^{(0)},\mathbf{e}_j^{(1)}\right)/\tau_A \right) }. $$

其中，\(\mathcal{P}(u)\) 是当前批次中用户的正样本物品集合，\(\mathcal{B}_I\) 是批次中的候选物品集合。

加上物品侧的对称约束：

$$ \mathcal{L}_A = \mathcal{L}_A^u+\mathcal{L}_A^i. $$

因此，这里是基于交互关系的用户—物品跨层对齐，不能简单理解为“同一节点第 0 层与第 1 层对齐”。

其作用可以理解为：用户自身表达的兴趣，应与其交互物品聚合到的结构信息保持匹配，物品侧同理。

代码中的正样本由当前 batch 的用户—正物品对构建，并没有直接把新增反事实物品边作为对比正样本。

2. 解耦判别：保留自身语义，抑制结构同质化。

按照论文设计，图中的三个视图分别是：

视图	含义
\(\mathbf{e}_i^{(0)}\)	含身份信息的初始表示
\(\mathbf{m}_i\)	原始多模态融合语义
\(\overline{\mathbf{e}}_i\)	经过图传播和层融合后的表示

物品侧解耦判别损失为：

$$ \mathcal{L}_D^i = -\log \frac{ \exp\left( \operatorname{sim} \left(\mathbf{e}_i^{(0)},\mathbf{m}_i\right)/\tau_D \right) }{ \displaystyle\sum_{j\in\mathcal{B}_I} \exp\left( \operatorname{sim} \left(\mathbf{e}_i^{(0)},\overline{\mathbf{e}}_j\right)/\tau_D \right) }. $$

用户侧采用类似的约束：

$$ \mathcal{L}_D = \mathcal{L}_D^u+\mathcal{L}_D^i. $$

目标是让初始身份表示靠近自身内容语义，同时与过度融合的结构表示保持区别。

直观来说：一双小众跑鞋可以学习其他运动商品提供的协同信息，但仍应保留自身的内容特点。

两项约束的配合关系是：

跨层对齐帮助吸收相关结构信息，解耦判别尝试保留个体语义差异。

不过，解耦判别本身没有显式分离“热门成分”，因此更严谨的作用描述是抑制表示同质化；热门度偏置的针对性处理主要来自上方的邻居选择机制。

五、方法图右侧：推荐与联合训练

得到最终用户表示和物品表示后，通过内积计算分数：

$$ \widehat{r}_{ui} = \overline{\mathbf{e}}_u^{\top} \overline{\mathbf{e}}_i. $$

按分数排序，得到 Top-K 推荐列表。

训练时包含三类目标：

损失	作用
\(\mathcal{L}_{rec}\)	提高正样本相对于负样本的匹配程度
\(\mathcal{L}_m\)	对齐同一用户或物品的文本与图像表示，继承自 IDFREE
\(\mathcal{L}_{SCE}\)	通过跨层对齐和解耦判别约束结构学习

推荐损失为：

$$ \mathcal{L}_{rec} = \frac{1}{|\mathcal{B}|} \sum_{(u,i^+)\in\mathcal{B}} \log \sum_{i^-\in\mathcal{I}_u^-} \exp \left( \frac{ \operatorname{sim} \left(\overline{\mathbf{e}}_u,\overline{\mathbf{e}}_{i^-}\right) - \operatorname{sim} \left(\overline{\mathbf{e}}_u,\overline{\mathbf{e}}_{i^+}\right) }{\tau} \right). $$

它鼓励用户与正样本的相似度高于负样本。

论文中的结构对比目标为：

$$ \mathcal{L}_{SCE} = \mathcal{L}_A+\mathcal{L}_D. $$

总目标为：

$$ \mathcal{L} = \mathcal{L}_{rec} + \lambda_m\mathcal{L}_m + \lambda_S\mathcal{L}_{SCE}. $$

你的代码实际还对解耦判别设置了额外权重：

$$ \boxed{ \mathcal{L} = \mathcal{L}_{rec} + \lambda_m\mathcal{L}_m + \lambda_S \left( \mathcal{L}_A+\beta\mathcal{L}_D \right) } $$

其中，\(\beta\) 对应 dcl_weight。推荐损失采用余弦相似度的排序目标，预测阶段使用内积，不能将这里的训练损失直接称为 BPR。

六、当前方法图与代码需要注意的三处差异

第一，图中的 MLP 雪花标记与代码不一致。

论文将投影参数定义为可学习参数，当前代码也没有冻结投影 MLP；物品特征还设置了 freeze=False。如果雪花表示冻结，需要调整标记。

第二，图底部公式混用了两个融合系数。

按论文式（4）—（5）和代码，正确展开应为：

$$ \boxed{ \mathbf{e}_x^{(0)} = \alpha_m\mathbf{z}_x^t + (1-\alpha_m)\mathbf{z}_x^v + \left[ \alpha_p\boldsymbol{\gamma}_x^t + (1-\alpha_p)\boldsymbol{\gamma}_x^v \right]\odot\mathbf{p}_x } $$

其中，内容融合使用 \(\alpha_m\)，门控融合使用 \(\alpha_p\)。

另外，图中的反事实图融合省略了权重，论文和代码对应的是：

$$ \mathbf{R}_I^{aug} = \mathbf{R}_I^{base} + \eta\mathbf{R}_I^{cf}. $$

第三，当前解耦判别的“原始语义视图”存在实质性的实现差异。

代码传入 i_modal 和 u_modal 的是已经加入调制 PE 后的融合表示。在 use_id=False 时，它们与对应的 layer 0 表示相同：

$$ \mathbf{m}_x^{\text{code}} = \alpha_m\widetilde{\mathbf{z}}_x^t + (1-\alpha_m)\widetilde{\mathbf{z}}_x^v = \mathbf{e}_x^{(0)}. $$

因此，解耦判别的正样本相似度实际上变成：

$$ \operatorname{sim} \left( \mathbf{e}_x^{(0)},\mathbf{m}_x^{\text{code}} \right) = \operatorname{sim} \left( \mathbf{e}_x^{(0)},\mathbf{e}_x^{(0)} \right) \approx 1. $$

对于正常的非零向量，这一项接近常数，无法承担论文所描述的“拉近初始身份与原始语义”的学习作用。其主要优化压力来自分母中的结构表示项。

若要与论文设计一致，原始语义视图应使用加入 PE 之前的文本、图像融合表示：

$$ \mathbf{m}_x = \alpha_m\mathbf{z}_x^t + (1-\alpha_m)\mathbf{z}_x^v. $$

此时，初始身份视图与原始语义视图的关系才是：

$$ \mathbf{e}_x^{(0)} = \mathbf{m}_x+\mathbf{p}'_x. $$

此外，图中的原始语义视图与最终结构视图之间也画了 push，但当前损失没有单独针对这两个视图的排斥项。

讲解 MAIL 时，最核心的贡献表述可以落在：通过内容感知的身份构建改善初始表示，再通过热门度惩罚的邻居增强与结构对比约束，改善 ID-free 推荐中的信息传播与表示区分性。 其中解耦判别的完整作用，需要先统一论文、方法图与代码的视图定义。