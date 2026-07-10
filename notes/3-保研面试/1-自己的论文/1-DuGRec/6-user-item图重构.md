# user-item图重构
User-item 图重构的核心不是凭空预测新边，而是：

> 用户已经喜欢物品 A，而物品图认为 A 和 B、C 很相似，那么给用户和 B、C 补充较弱的潜在连接。

整体分为四步：
$$
\boxed{
\text{筛选物品邻居}
\rightarrow
\text{用户兴趣沿物品图扩散一步}
\rightarrow
\text{融合原始与扩展交互}
\rightarrow
\text{构建用户—物品二部图}
}
$$

## 1. 先筛选可靠的物品关系

前面已经得到最终 item 图：
$$
G_{ii}\in\mathbb R^{I\times I}
$$
其中 $G_{ii}[i,j]$ 表示物品 $i$ 和物品 $j$ 的关联强度。

为了避免扩展出太多噪声边，模型先做两次筛选：

1. 只保留权重大于阈值 $\theta$ 的边。
2. 每个物品最多保留 Top-$K$ 个邻居。

然后进行行归一化：
$$
S_{\theta,k}
=
\operatorname{RowNorm}
\left(
\operatorname{TopK}
\{G_{ii}(i,j)\geq\theta\}
\right)
$$
$S_{\theta,k}$ 可以理解为一张“物品兴趣转移表”。

例如：
$$
S_{\theta,k}=
\begin{bmatrix}
0 & 0.7 & 0.3\\
0.6 & 0 & 0.4\\
0.2 & 0.8 & 0
\end{bmatrix}
$$
第一行表示：
$$
A\rightarrow B:0.7,\qquad A\rightarrow C:0.3
$$
即与物品 A 相关的兴趣，可以按 70% 和 30% 向 B、C 扩散。

代码对应：

```python
S = self.mm_adj

S_f = self._filter_sparse_topk_threshold(
    S,
    topk=self.cui_topk,
    theta=self.cui_theta
)
```

`_filter_sparse_topk_threshold()` 会先按阈值过滤，再逐行保留 Top-$K$，最后行归一化。

------

## 2. 让用户兴趣沿 item 图扩散一步

原始用户—物品交互矩阵记为：
$$
R\in\mathbb R^{U\times I}
$$
其中：
$$
R_{ui}=
\begin{cases}
1,&用户u交互过物品i\\
0,&没有交互
\end{cases}
$$
模型计算：
$$
\widetilde R=RS_{\theta,k}
$$
通俗理解就是：

> 每个用户从自己交互过的物品出发，沿 item 图走一步，找到这些物品的相似邻居。

假设某用户只交互过物品 A：
$$
R_u=[1,0,0]
$$
物品 A 与 B、C 的转移关系为：
$$
S_A=[0,0.7,0.3]
$$
那么：
$$
\widetilde R_u
=
R_uS
=
[1,0,0]
\begin{bmatrix}
0&0.7&0.3\\
0.6&0&0.4\\
0.2&0.8&0
\end{bmatrix}
=
[0,0.7,0.3]
$$
这表示模型给用户补充了两个潜在兴趣：

- 用户可能喜欢 B，强度为 0.7。
- 用户可能喜欢 C，强度为 0.3。

代码就是：

```python
R = self.interaction_matrix.tocsr()
S_csr = self.torch_sparse_to_scipy_csr(S_f)

R_tilde = (R @ S_csr).tocsr()
```

其中：

- `R` 的形状是 `[用户数, 物品数]`。
- `S_csr` 的形状是 `[物品数, 物品数]`。
- `R_tilde` 的形状仍是 `[用户数, 物品数]`。

------

## 3. 原始交互和扩展交互融合

不能完全相信推测出来的新边，所以模型保留原始交互，并与扩展结果加权融合：
$$
R_{\mathrm{mix}}
=
\gamma_{\mathrm{cui}}R
+
(1-\gamma_{\mathrm{cui}})\widetilde R
$$
其中：

- $\gamma_{\mathrm{cui}}$ 越大，越相信真实历史交互。
- $\gamma_{\mathrm{cui}}$ 越小，越相信 item 图扩展的潜在兴趣。

假设：
$$
\gamma_{\mathrm{cui}}=0.8
$$
则：
$$
R_u=[1,0,0]
$$
融合后：
$$
R_{\mathrm{mix},u}
=
0.8[1,0,0]+0.2[0,0.7,0.3]
$$
含义是：

- 用户和 A：真实交互，连接最强。
- 用户和 B：补充的潜在连接。
- 用户和 C：更弱的潜在连接。

代码对应：

```python
R_mix = (
    self.cui_gamma * R
    + (1.0 - self.cui_gamma) * R_tilde
)
```

论文中把 $\widetilde R$ 称为用户通过相似物品得到的一步可达结果，再与原始交互进行凸组合。

------

## 4. 构造用户—物品二部图

得到 $R_{\mathrm{mix}}$ 后，构造完整邻接矩阵：
$$
A=
\begin{bmatrix}
0&R_{\mathrm{mix}}\\
R_{\mathrm{mix}}^\top&0
\end{bmatrix}
$$
左上角和右下角为 0，是因为这张图只连接：
$$
\text{用户}\leftrightarrow\text{物品}
$$
而不直接连接用户—用户或物品—物品。

最后做对称归一化：
$$
\widetilde A=D^{-1/2}AD^{-1/2}
$$
归一化的作用是避免热门物品因为邻居太多，在传播时占据过大权重。这个归一化图随后用于 LightGCN 消息传播。

代码入口是：

```python
return self.build_norm_bipartite_from_ui(
    R_mix.tocoo()
)
```

## 一个生活化类比

假设用户买过“跑步鞋”。

item 图发现：

- 跑步鞋和运动袜高度相关。
- 跑步鞋和运动手环中度相关。
- 跑步鞋和篮球低度相关，但低于阈值被删除。

于是原图只有：
$$
用户\rightarrow跑步鞋
$$
重构后变成：
$$
用户\rightarrow跑步鞋
$$
实线是真实行为，虚线是根据相似物品补充的潜在兴趣。

所以它本质上是：
$$
\boxed{
\text{把用户对已交互物品的兴趣，向可靠的相似物品扩散一步}
}
$$

## 当前代码的一个实现细节

论文公式中的 $R_{\mathrm{mix}}$ 是带权交互矩阵，但当前代码在 `build_norm_bipartite_from_ui()` 中使用了：

```python
data_dict = dict(
    zip(
        zip(ui.row, ui.col + n_u),
        [1] * ui.nnz
    )
)
```

这会把 $R_{\mathrm{mix}}$ 中所有非零值转换为权重 1。也就是说，当前代码最终主要保留的是：
$$
\boxed{\text{原始边和扩展边的并集}}
$$
而不是完整保留 $0.8、0.14、0.06$ 这些连续边权。按论文公式讲解时，应表述为带权融合；按当前代码实际执行，则是先用 $R_{\mathrm{mix}}$ 决定哪些边存在，再将这些边二值化后做度归一化。