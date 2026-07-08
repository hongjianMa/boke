# DCT， Discrete Cosine Transform，离散余弦变换。

一幅图像总是由一些主要的色彩和一些<u>不重要的细节</u>构成，丢掉这部分细节其实对整幅图的影响并不大。 

它的作用是把一个实值向量从原始特征空间变换到频域空间，用一组不同频率的余弦基去表示原始信号。  

换句话说 DCT 就是把一个特征向量拆成“低频主干信息”和“高频细节信息”。 

> 在推荐系统里，低频部分更像稳定的整体语义，高频部分更像局部细节或噪声。

通俗讲，DCT 就是：**把一个原始特征向量，拆成一组不同频率的余弦波成分**。

## 1. DCT 公式是什么

最常用的是 **DCT-II**，代码里 `torch_dct.dct(..., norm='ortho')` 对应的就是正交归一化版本。

给定一个长度为 $N$ 的特征向量：
$$
x = [x_0, x_1, \dots, x_{N-1}]
$$
DCT 后得到频域系数：
$$
X_k = \alpha_k \sum_{n=0}^{N-1} x_n \cos\left[\frac{\pi}{N}\left(n+\frac{1}{2}\right)k\right]
$$
其中：
$$
\alpha_0 = \sqrt{\frac{1}{N}}
$$
这里：

- $x_n$：原始特征向量第 $n$ 个维度的值。
- $X_k$：第 $k$ 个频率成分的强度。
- $k=0$：最低频，表示整体平均趋势。
- $k$ 越大：频率越高，表示越细碎、变化越快的部分。
- $\alpha_k$：归一化系数，让变换前后尺度更稳定。

------

## 2. 通俗怎么记忆这个公式

你只需要记一句话：

**DCT 的每个输出 $X_k$，就是原始特征 $x$ 和第 $k$ 条余弦波模板做一次相似度匹配。**

公式可以拆成三块记：
$$
X_k = \alpha_k \sum x_n \cos(\text{位置} \times \text{频率})
$$
更直观地说：

- $\sum$：把所有维度的信息加起来。
- $x_n \cos(\cdots)$：看原始特征在第 $n$ 个位置和某条余弦波有多匹配。
- $k$：控制余弦波的频率，$k$ 越大，波动越快。
- $n+\frac{1}{2}$：表示在每个位置的小格子中间采样。
- $\alpha_k$：只是缩放，不改变核心含义。



**DCT 可以理解为用一组余弦基函数去扫描原始特征，计算原始特征在<u>不同频率</u>模式上的投影强度。低频系数保留整体语义，高频系数保留细节变化。**

## 3.论文中的DCT

你的论文不是为了做信号压缩，而是为了构造一个额外的 **频域视图 spectral view**。

原始图像/文本特征直接算相似度时，可能会因为表面模态相似引入噪声边。比如两个商品图片颜色相似，但用户偏好并不一定相似。

所以 DuGRec 做了这件事：
$$
\mathbf{W}_v = DCT(\mathbf{V})\\
\mathbf{W}_t = DCT(\mathbf{T})\\
\mathbf{W}_s = [\mathbf{W}_v \mid \mathbf{W}_t]
$$
其中 $\mathbf{V}$ 是图像特征，$\mathbf{T}$ 是文本特征，$\mathbf{W}_v$、$\mathbf{W}_t$ 是 DCT 后的频域表示，$\mathbf{W}_s$ 是把图像频域特征和文本频域特征拼接起来。论文随后把频谱划分为低频和高频子空间，用于构建频域物品图。

## 4. DCT 后的特征怎么继续使用

代码里还有一个函数：

```python
def _split_dct_bands(self, W, keep_ratio):
    D = W.size(1)
    k = max(1, int(D * keep_ratio))
    return W[:, :k], W[:, k:]
```

它的作用是把 DCT 后的频域特征分成两部分：

```python
W[:, :k]   # 低频部分
W[:, k:]   # 高频部分
```

对应论文公式：
$$
(\mathbf{W}_{lf}, \mathbf{W}_{hf}) = Split(\mathbf{W}, r)\\
k_r = \lfloor r \cdot D \rfloor
$$
其中 $r$ 就是保留低频比例，代码里对应：

```py
self.dct_keep_ratio
```

你的默认配置里是：

```python
self.dct_keep_ratio = float(getattr(config, 'dct_keep_ratio', 0.2))
```

**也就是默认取前 20% 作为低频部分**。

## 5. 频域图是怎么构建的

代码中 `build_item_semantic_graph_spectral()` 负责构建频域物品图。

核心流程是：

```python
w_v = dct.dct(self.v_feat, norm='ortho')
w_t = dct.dct(self.t_feat, norm='ortho')
w_vt = self.interleaved_feat
```

然后分别切分低频和高频：

```python
v_lf, v_hf = self._split_dct_bands(w_v, self.dct_keep_ratio)
t_lf, t_hf = self._split_dct_bands(w_t, self.dct_keep_ratio)
```

再根据低频、高频、混合频域特征分别构图：

```python
G_v_lf = self._knn_from_feat(v_lf, k, shape)
G_v_hf = self._knn_from_feat(v_hf, k, shape)
G_t_lf = self._knn_from_feat(t_lf, k, shape)
G_t_hf = self._knn_from_feat(t_hf, k, shape)
G_mix = self._knn_from_feat(w_vt, k, shape)
```

最后融合成频域图：

```python
G_spec = self.beta_lf * G_lf + self.beta_hf * G_hf + self.beta_mix * G_mix
```

这正好对应论文公式：
$$
\mathbf{G}_{spec}
=
\beta_{lf}\mathbf{G}_{lf}
+
\beta_{hf}\mathbf{G}_{hf}
+
\beta_{mix}\mathbf{G}_{mix}
$$
论文里也说明，先把 $\mathbf{W}_v$、$\mathbf{W}_t$、$\mathbf{W}_s$ 分成低频和高频频带，然后分别构建 KNN 图，再融合成频域物品图。

> DCT 是离散余弦变换，可以把原始图像和文本特征从空间域映射到频域。它的核心思想是用一组不同频率的余弦基函数去表示原始特征，其中低频部分通常保留主要语义信息，高频部分包含细节变化。我的方法中使用 DCT 不是为了压缩，而是为了构建一个频域语义视图。具体来说，我对图像特征和文本特征分别做 DCT，得到频域表示，再划分低频和高频频带，并分别构建 KNN 物品图，最后和空间语义图、共现图融合，从而减少单纯空间相似度构图带来的模态噪声。

