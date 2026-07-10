# 频域构图KNN

## 1. 通用 KNN 构图过程

假设某个频带的物品特征矩阵为：
$$
F\in\mathbb R^{N\times d}
$$
其中 $N$ 是物品数，每一行 $f_i$ 是物品 $i$ 的特征。

### 第一步：归一化物品特征

$$
\bar f_i=\frac{f_i}{\|f_i\|_2}
$$

代码：

```
F = F / (F.norm(p=2, dim=1, keepdim=True) + 1e-12)
```

### 第二步：计算所有物品两两相似度

$$
S=\bar F\bar F^\top
$$

其中：
$$
S_{ij}=\bar f_i^\top \bar f_j
      =\cos(f_i,f_j)
$$
代码：

```
sim = torch.mm(F, F.t())
```

因为前面已经归一化，所以这里的点积就是余弦相似度。

### 第三步：排除物品自己

物品和自己相似度一定最高，所以把对角线设成 $-1$：

```
sim.fill_diagonal_(-1.0)
```

这样 Top-$K$ 时不会把自己选进去。

### 第四步：为每个物品选出最相似的 $K$ 个邻居

$$
\mathcal N_K(i)
=
\operatorname{TopK}_{j\neq i}S_{ij}
$$

代码：

```
_, nn_idx = torch.topk(sim, k, dim=-1)
```

例如 $K=2$，物品 $i_1$ 与其他物品的相似度为：
$$
[0.8,\ 0.3,\ 0.9]
$$
那么选择相似度最高的两个物品：
$$
\mathcal N_2(i_1)=\{i_4,i_2\}
$$
于是构造两条边：
$$
i_1\rightarrow i_4,\qquad i_1\rightarrow i_2
$$

### 第五步：构造邻接矩阵

$$
A_{ij}=
\begin{cases}
1,&j\in\mathcal N_K(i)\\
0,&\text{其他}
\end{cases}
$$

代码：

```python
rows = torch.arange(nn_idx.size(0)).unsqueeze(1)
rows = rows.expand(-1, k).reshape(-1)

cols = nn_idx.reshape(-1)

idx = torch.stack([rows, cols], 0)
val = torch.ones(rows.numel())
```

这里需要注意：**相似度只用于选择邻居，真正写入图中的原始边权是 1，而不是余弦相似度值。**

### 第六步：行归一化

$$
G_{ij}
=
\frac{A_{ij}}
{\sum_j A_{ij}}
$$

代码：

```python
return self._row_normalize_sparse(
    idx, val, shape
).coalesce()
```

如果每个物品正好有 $K$ 个邻居，那么通常每条边的权重约为：
$$
G_{ij}=\frac{1}{K}
$$
整个 KNN 构图函数就是：

```python
def _knn_from_feat(self, F, k, shape):
    # 1. 特征归一化
    F = F / (F.norm(p=2, dim=1, keepdim=True) + 1e-12)

    # 2. 计算余弦相似度
    sim = torch.mm(F, F.t())

    # 3. 排除自身
    sim.fill_diagonal_(-1.0)

    # 4. 每个物品选择K个最近邻
    _, nn_idx = torch.topk(sim, k, dim=-1)

    # 5. 构造边
    rows = torch.arange(nn_idx.size(0), device=F.device) \
        .unsqueeze(1).expand(-1, k).reshape(-1)
    cols = nn_idx.reshape(-1)

    idx = torch.stack([rows, cols], 0)
    val = torch.ones(rows.numel(), device=F.device)

    # 6. 行归一化
    return self._row_normalize_sparse(
        idx, val, shape
    ).coalesce()
```



------

## 2. 低频图怎么构建

首先对视觉和文本特征分别做 DCT：
$$
W_v=\operatorname{DCT}(V)
$$
然后按照比例 $r$ 切分：
$$
W_v=[W_v^{lf}\mid W_v^{hf}]
$$
代码：

```python
v_lf, v_hf = self._split_dct_bands(
    w_v, self.dct_keep_ratio
)

t_lf, t_hf = self._split_dct_bands(
    w_t, self.dct_keep_ratio
)
```

假设视觉特征 DCT 后有 1000 维，`dct_keep_ratio=0.2`，那么：
$$
W_v^{lf}=W_v[:,0:200]
$$
接下来分别对视觉低频和文本低频做 KNN：
$$
G_v^{lf}
=
\operatorname{KNN}(W_v^{lf})
$$
代码：

```python
G_v_lf = self._knn_from_feat(v_lf, k, shape)
G_t_lf = self._knn_from_feat(t_lf, k, shape)
```

再融合成最终低频图：
$$
G_{lf}
=
\alpha_{\mathrm{img}}G_v^{lf}
+
(1-\alpha_{\mathrm{img}})G_t^{lf}
$$
代码：

```
G_lf = _sum(
    G_v_lf,
    G_t_lf,
    alpha_image
)
```

通俗来说：

> 分别根据图像低频语义和文本低频语义寻找邻居，再把两个邻居图融合起来。

------

## 3. 高频图怎么构建

高频图与低频图完全相同，只是输入换成剩余的高频系数：
$$
G_v^{hf}
=
\operatorname{KNN}(W_v^{hf})
$$
再融合：
$$
G_{hf}
=
\alpha_{\mathrm{img}}G_v^{hf}
+
(1-\alpha_{\mathrm{img}})G_t^{hf}
$$
代码：

```python
G_v_hf = self._knn_from_feat(v_hf, k, shape)
G_t_hf = self._knn_from_feat(t_hf, k, shape)

G_hf = _sum(
    G_v_hf,
    G_t_hf,
    alpha_image
)
```

通俗来说：

> 根据图像和文本中的细节变化，为每个物品寻找高频模式最相似的 $K$ 个邻居。

------

## 4. 图文混合频域图怎么构建

代码先把完整的视觉频域特征和文本频域特征拼接：
$$
W_s=[W_v\mid W_t]
$$
代码：

```
self.interleaved_feat = torch.cat((w_v, w_t), 1)
```

然后直接对拼接后的频域向量做 KNN：
$$
G_{\mathrm{mix}}
=
\operatorname{KNN}(W_s)
$$
代码：

```
w_vt = self.interleaved_feat
G_mix = self._knn_from_feat(w_vt, k, shape)
```

通俗来说：

> 低频图和高频图分别观察图像、文本，而混合图把完整的图像频域和文本频域拼在一起，再综合寻找 $K$ 个最近邻。

需要注意，按照你当前代码，`G_mix` 使用的是**完整图像 DCT 特征与完整文本 DCT 特征的拼接**，不是只拼接低频，也不是只拼接高频。

------

## 5. 三个图如何融合

最终频域图为：
$$
G_{\mathrm{spec}}
=
\beta_{lf}G_{lf}
+
\beta_{hf}G_{hf}
+
\beta_{mix}G_{mix}
$$
代码：

```
G_spec = (
    self.beta_lf * G_lf
    + self.beta_hf * G_hf
    + self.beta_mix * G_mix
)
```

代码默认参数为：
$$
\beta_{lf}=0.5,\qquad
\beta_{hf}=0.1,\qquad
\beta_{mix}=0.4
$$
也就是：
$$
G_{\mathrm{spec}}
=
0.5G_{lf}
+
0.1G_{hf}
+
0.4G_{mix}
$$
低频图权重最高，因为低频通常承载较稳定的整体信息；高频图权重较低，用于补充细节；混合图综合图像和文本频域关系。论文同样采用低频、高频和混合频域 KNN 图加权融合的方式。

## 最简记忆

$$
\boxed{
\text{每一种频域特征}
\rightarrow
\text{算物品余弦相似度}
\rightarrow
\text{每个物品选Top-}K
\rightarrow
\text{连边并归一化}
}
$$

具体得到：
$$
\boxed{
\begin{aligned}
G_{lf}
&=\text{图像低频KNN图}+\text{文本低频KNN图}\\
G_{hf}
&=\text{图像高频KNN图}+\text{文本高频KNN图}\\
G_{mix}
&=\text{图像与文本完整频域拼接后的KNN图}
\end{aligned}
}
$$
