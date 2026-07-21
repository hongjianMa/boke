# KL散度

## KL 散度是啥

一句话：

> **KL 散度用来衡量两个概率分布 $P$ 和 $Q$ 的差异。**

假设：

- $P$：真实分布
- $Q$：模型预测分布

那么 KL 散度：
$$
D_{KL}(P\|Q)
=
\sum_i P(i)\log\frac{P(i)}{Q(i)}
$$
你可以先简单记成：
$$
\boxed{\text{KL散度 = P 和 Q 有多不一样}}
$$
例如真实分布：

```
P = [0.8, 0.2]
```

模型预测：

```
Q = [0.7, 0.3]
```

两者比较接近，所以 KL 较小。

但如果：

```
P = [0.8, 0.2]
Q = [0.1, 0.9]
```

两者差别很大，所以 KL 较大。

------

## 公式

原始公式：
$$
D_{KL}(P\|Q)
=
\sum_i P_i\log\frac{P_i}{Q_i}
$$
把 log 拆开：
$$
D_{KL}(P\|Q)
=
\sum_i P_i(\log P_i-\log Q_i)
$$
所以代码其实非常简单：

```
P * (log(P) - log(Q))
```

然后：

```
sum()
```

就完了。

因此手撕代码的核心直接记：

```
kl = torch.sum(p * (torch.log(p) - torch.log(q)))
```

------

## 代码实现

面试让你手写，我建议先写这个版本：

```python
import torch

def kl_divergence(p, q):
    return torch.sum(p * (torch.log(p) - torch.log(q)))

p = torch.tensor([0.8, 0.2])
q = torch.tensor([0.7, 0.3])

kl = kl_divergence(p, q)

print(kl)
```

对应公式：
$$
D_{KL}(P\|Q)
=
\sum_i
P_i
(
\log P_i-\log Q_i
)
$$


