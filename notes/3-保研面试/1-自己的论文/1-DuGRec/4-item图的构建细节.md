# item图构图细节

> 每个物品是一个节点，如果两个物品在内容、频域或用户行为上相关，就在它们之间连一条边。

最终流程是：
$$
\text{空间语义图}
+
\text{频域语义图}
+
\text{行为共现图}
\longrightarrow
G_{ii}
$$

## 1. 构建空间语义图

先分别用图像特征和文本特征计算物品之间的余弦相似度：
$$
s_{ij}^{m}
=
\frac{(e_i^{m})^\top e_j^{m}}
{\|e_i^{m}\|_2\|e_j^{m}\|_2},
\qquad m\in\{v,t\}
$$
其中 $v$ 表示图像，$t$ 表示文本。

对每个物品，只保留最相似的 Top-$K$ 个物品，分别得到：
$$
G_{\mathrm{img}},\qquad G_{\mathrm{text}}
$$
然后融合：
$$
G_{\mathrm{spatial}}
=
\alpha_{\mathrm{img}}G_{\mathrm{img}}
+
(1-\alpha_{\mathrm{img}})G_{\mathrm{text}}
$$
通俗来说，就是：

> 给每个商品寻找图片最像和文本最像的若干邻居，然后把两种关系合起来。

论文中也是先计算图像和文本余弦相似度，再进行 Top-$K$ 筛选和行归一化。

代码核心是：

```python
emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-12)
sim = torch.mm(emb, emb.t())
sim.fill_diagonal_(-1.0)

_, knn_ind = torch.topk(sim, k, dim=-1)
```

这里先归一化，所以：

```python
torch.mm(emb, emb.t())
```

计算的就是物品两两余弦相似度。

然后：

```python
G_sem = (
    alpha_image * G_img
    + (1.0 - alpha_image) * G_txt
)
```

对应空间图融合公式。

需要注意：代码在确定 Top-$K$ 邻居后，把边权设成了 1，再做行归一化：

```python
values = torch.ones(rows.numel())
G_img = self._row_normalize_sparse(...)
```

所以余弦相似度主要用于**选邻居**，而不是直接作为最终边权。

------

## 2. 构建频域语义图

先对图像和文本特征做 DCT：
$$
W_v=\operatorname{DCT}(V),\qquad
W_t=\operatorname{DCT}(T)
$$
再把图像和文本频域特征拼接：
$$
W_s=[W_v\mid W_t]
$$
代码对应：

```python
w_v = dct.dct(self.v_feat, norm='ortho')
w_t = dct.dct(self.t_feat, norm='ortho')
w_vt = self.interleaved_feat
```

然后把频域特征切成低频和高频：
$$
(W_{\mathrm{lf}},W_{\mathrm{hf}})
=
\operatorname{Split}(W,r)
$$
分别构建低频图、高频图和图文混合频域图：
$$
G_{\mathrm{lf}},\qquad
G_{\mathrm{hf}},\qquad
G_{\mathrm{mix}}
$$
最后融合：
$$
G_{\mathrm{spec}}
=
\beta_{\mathrm{lf}}G_{\mathrm{lf}}
+
\beta_{\mathrm{hf}}G_{\mathrm{hf}}
+
\beta_{\mathrm{mix}}G_{\mathrm{mix}}
$$
通俗来说：

> 空间图看商品原始特征像不像，频域图看商品的整体变化模式和细节变化模式像不像。

代码对应：

```python
v_lf, v_hf = self._split_dct_bands(
    w_v, self.dct_keep_ratio
)

t_lf, t_hf = self._split_dct_bands(
    w_t, self.dct_keep_ratio
)

G_v_lf = self._knn_from_feat(v_lf, k, shape)
G_v_hf = self._knn_from_feat(v_hf, k, shape)
G_t_lf = self._knn_from_feat(t_lf, k, shape)
G_t_hf = self._knn_from_feat(t_hf, k, shape)

G_mix = self._knn_from_feat(w_vt, k, shape)
```

然后加权：

```python
G_spec = (
    self.beta_lf * G_lf
    + self.beta_hf * G_hf
    + self.beta_mix * G_mix
)
```



------

## 3. 融合空间图与频域图

空间图和频域图共同组成内容语义图：
$$
G_{\mathrm{sem}}
=
(1-\lambda_{\mathrm{spec}})G_{\mathrm{spatial}}
+
\lambda_{\mathrm{spec}}G_{\mathrm{spec}}
$$
其中 $\lambda_{\mathrm{spec}}$ 控制频域信息占多少。

代码对应：

```python
G_spatial = self.build_item_semantic_graph(...)
G_spec = self.build_item_semantic_graph_spectral(...)

G_sem_joint = (
    (1.0 - self.lambda_spec) * G_spatial
    + self.lambda_spec * G_spec
)
```



------

## 4. 构建物品共现图

除了看商品内容，还要看用户行为。

设用户—物品交互矩阵为 $R$，计算：
$$
C=R^\top R
$$
其中：
$$
C_{ij}
$$
表示物品 $i$ 和物品 $j$ 被同一批用户共同交互了多少次。

例如很多用户同时购买手机和手机壳，那么这两个物品的共现值就较高。

然后保留每个物品共现次数最高的 Top-$K$ 邻居，并做 RowSoftmax：
$$
G_{\mathrm{co}}
=
\operatorname{RowSoftmax}
\left(
\operatorname{TopK}(R^\top R)
\right)
$$
代码对应：

```python
R = self.interaction_matrix.tocsr()
C = (R.T @ R).tocsr().astype(np.float32)

C.setdiag(0.0)
C.eliminate_zeros()
```

再逐行保留 Top-$K$：

```python
loc = np.argpartition(row_val, -topk)[-topk:]
```

最后做行 Softmax：

```python
return self._row_softmax_sparse(
    indices, values,
    (self.n_items, self.n_items)
)
```



------

## 5. 三类关系融合成最终 item 图

最终 item 图为：
$$
G_{ii}
=
\alpha_{\mathrm{co}}G_{\mathrm{co}}
+
(1-\alpha_{\mathrm{co}})G_{\mathrm{sem}}
$$
代码对应：

```python
self.mm_adj = (
    alpha_item_co * G_co
    + (1.0 - alpha_item_co) * G_sem
)
```

其中：

- $G_{\mathrm{sem}}$：商品内容上是否相似。
- $G_{\mathrm{co}}$：用户行为上是否经常一起出现。
- $\alpha_{\mathrm{co}}$：控制协同共现信息占多大比例。

代码还增加了一个频域门控：如果两个物品的低频能量比例差异很大，就降低它们在共现图中的边权：

```python
gate = torch.exp(
    -(rho[ii[0]] - rho[ii[1]]).abs()
    / self.spec_gate_tau
)

G_co = torch.sparse_coo_tensor(
    ii, vv * gate, ...
)
```

最终图会保存到缓存，避免每次训练都重新构建。

## 一句话记忆

$$
\boxed{
\text{原始图文找相似邻居}
+
\text{DCT频域找稳定邻居}
+
\text{用户行为找共现邻居}
=
\text{最终物品图 }G_{ii}
}
$$

其中 $G_{ii}[i,j]>0$ 表示物品 $j$ 是物品 $i$ 的邻居，值越大表示后续图传播时，物品 $j$ 对物品 $i$ 的影响越强。