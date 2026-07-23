# BPR Loss

它的核心思想就一句话：

> **让用户喜欢的正样本分数，比不喜欢的负样本分数更高。**

比如用户 $u$：

```
正样本 i：用户买过的商品      score = 5
负样本 j：用户没买过的商品    score = 2
```

我们希望：
$$
score_{pos} > score_{neg}
$$
而且两者差距最好越大越好。

------

## 公式

$$
\boxed{
L_{BPR}=-\log \sigma(s_{pos}-s_{neg})
}
$$

### code

```python
import torch

def bpr_loss(pos_score, neg_score):
    diff = pos_score - neg_score
    loss = -torch.log(torch.sigmoid(diff))
    return loss.mean()
```


