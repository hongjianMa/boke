# EMA
EMA 是 **Exponential Moving Average，指数移动平均**。它的作用是：

> 不直接使用当前这一轮计算出的注意力权重，而是把“历史权重”和“当前权重”做平滑融合，避免权重突然跳动。

## 1. 公式

在你的模型中：
$$
\boldsymbol{\alpha}_{ema}
\leftarrow
(1-\beta_e)\boldsymbol{\alpha}_{ema}
+
\beta_e\boldsymbol{\alpha}_{new}
$$
其中：

- $\boldsymbol{\alpha}_{ema}$：历史平滑后的注意力权重。
- $\boldsymbol{\alpha}_{new}$：当前轮新计算出的注意力权重。
- $\beta_e$：更新率，控制模型相信新权重的程度。

论文使用 EMA 平滑视觉、文本和融合语义三个模态的个性化注意力，再用平滑后的权重融合用户表示。

## 2. 通俗例子

假设某个用户上一轮的模态权重为：
$$
\boldsymbol{\alpha}_{ema}
=
[0.5,0.3,0.2]
$$
分别表示：
$$
[\text{视觉},\text{文本},\text{融合语义}]
$$
这一轮新计算的权重为：
$$
\boldsymbol{\alpha}_{new}
=
[0.2,0.6,0.2]
$$
设更新率：
$$
\beta_e=0.1
$$
那么：
$$
\boldsymbol{\alpha}_{ema}
=
0.9[0.5,0.3,0.2]
+
0.1[0.2,0.6,0.2]
$$
得到：
$$
\boldsymbol{\alpha}_{ema}
=
[0.47,0.33,0.20]
$$
虽然当前结果突然认为文本更重要，但 EMA 不会立即从：
$$
[0.5,0.3,0.2]
$$
跳到：
$$
[0.2,0.6,0.2]
$$
而是缓慢调整到：
$$
[0.47,0.33,0.20]
$$
可以把它理解为：

> 历史意见占 90%，当前意见占 10%。

## 3. 为什么你的模型需要 EMA

用户的模态注意力是根据当前表示之间的余弦相似度计算的：
$$
s_v
=
\cos(z_v,z_t)+\cos(z_v,z_s)
$$
再经过 Softmax 得到：
$$
\boldsymbol{\alpha}_{new}
=
\operatorname{Softmax}
\left(
\frac{[s_v,s_t,s_s]}{\tau}
\right)
$$
但是训练过程中，用户表示会受到随机采样、DropEdge 和参数更新的影响，导致注意力权重在不同轮次之间波动。EMA 相当于一个平滑器，使用户的模态偏好变化更加稳定。

## 4. 代码实现

首先计算当前注意力：

```python
sc_v = (z_v * z_t).sum(1, keepdim=True) \
     + (z_v * z_s).sum(1, keepdim=True)

sc_t = (z_t * z_v).sum(1, keepdim=True) \
     + (z_t * z_s).sum(1, keepdim=True)

sc_s = (z_s * z_v).sum(1, keepdim=True) \
     + (z_s * z_t).sum(1, keepdim=True)

att_new = F.softmax(
    torch.cat([sc_v, sc_t, sc_s], dim=1)
    / self.loss.clu_tau,
    dim=1
)
```

`att_new` 的形状是：

```
[用户数量, 3]
```

每一行是一个用户的视觉、文本和融合语义权重。

然后计算当前轮次的 EMA 更新率：

```python
beta_e = self.decay_weight * (
    1.0 - self.decay_base ** max(1, self.cur_epoch)
)

beta_e = float(max(0.0, min(1.0, beta_e)))
```

再进行 EMA 更新：

```python
self._att_ema = (
    (1.0 - beta_e) * self._att_ema
    + beta_e * att_new.detach()
)
```

最后使用平滑后的权重：

```python
self._att_used = self._att_ema
```

并融合用户的三个模态表示：

```python
fused = torch.cat([
    self._att_used[:, [0]] * v_rep,
    self._att_used[:, [1]] * t_rep,
    self._att_used[:, [2]] * s_rep
], dim=1)
```

这段代码完整实现了“计算当前权重—EMA 平滑—加权融合”的过程。

## 5. `detach()` 的作用

代码中：

```python
att_new.detach()
```

表示 EMA 更新只是保存一个平滑的历史状态，不让梯度通过 EMA 更新过程反向传播。

也就是说：

- `att_new` 用来更新历史注意力。
- `_att_ema` 被当作稳定的状态量。
- 避免跨训练轮次形成复杂的梯度依赖。

## 6. $\beta_e$ 大小的含义

当：
$$
\beta_e\approx 0
$$
模型主要相信历史权重，更新慢但稳定。

当：
$$
\beta_e\approx 1
$$
模型主要相信当前权重，更新快但更容易波动。

你的代码中 $\beta_e$ 会随训练轮数变化：
$$
\beta_e
=
w\left(1-b^{epoch}\right)
$$
训练初期 $\beta_e$ 较小，更多保留初始模态先验；随着训练进行，$\beta_e$ 增大，逐渐相信模型新学到的注意力。

一句话记忆：
$$
\boxed{\text{EMA}=\text{历史结果为主}+\text{少量吸收当前结果}}
$$
在 DuGRec 中，它负责让用户的视觉、文本和融合语义偏好权重更加平稳。