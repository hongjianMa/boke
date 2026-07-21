# Safe Softmax

> 普通 Softmax 先对 logits 减去最大值，再做 exp，避免指数溢出。

## 普通 Softmax

公式：
$$
\mathrm{Softmax}(x_i)
=
\frac{e^{x_i}}
{\sum_j e^{x_j}}
$$
最直接代码：

```python
import torch

def softmax(x):
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

但这个写法有数值稳定性问题。

比如：

```python
x = torch.tensor([1000.0, 1001.0, 1002.0])
```

计算：

```
torch.exp(1000)
```

数值太大，可能直接变成：

```
inf
```

然后就可能出现：

```
inf / inf → nan
```

------

## Safe Softmax

先减最大值：
$$
x_i = x_i-\max(x)
$$
然后：
$$
\mathrm{Softmax}(x_i)
=
\frac{
e^{x_i-\max(x)}
}{
\sum_j e^{x_j-\max(x)}
}
$$
代码：

```python
import torch

def safe_softmax(x):
    x_max = x.max(dim=-1, keepdim=True).values
    x = x - x_max

    exp_x = torch.exp(x)

    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```



------

## 为什么减最大值不改变 Softmax？

这是面试很容易追问的一点。

原公式：
$$
\frac{e^{x_i}}
{\sum_j e^{x_j}}
$$
所有元素同时减去一个常数 $c$：
$$
\frac{e^{x_i-c}}
{\sum_j e^{x_j-c}}
$$
因为：
$$
e^{x_i-c}
=
\frac{e^{x_i}}{e^c}
$$
所以：
$$
\frac{
e^{x_i}/e^c
}{
\sum_j e^{x_j}/e^c
}
=
\frac{e^{x_i}}
{\sum_j e^{x_j}}
$$
因此：
$$
\boxed{
Softmax(x)=Softmax(x-c)
}
$$
所以我们直接取：
$$
c=\max(x)
$$
