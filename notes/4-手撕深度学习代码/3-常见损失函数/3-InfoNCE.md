# InfoNCE

### 公式
$$
\boxed{
L_i
=
-\log
\frac{
\exp(\operatorname{sim}(z_i,z_i^+)/\tau)
}{
\sum_j \exp(\operatorname{sim}(z_i,z_j)/\tau)
}
}
$$





比如
$$
\boxed{
-\log
\frac{\text{正样本相似度}}
{\text{正样本相似度 + 所有负样本相似度}}
}
$$

### code

```python
import torch
import torch.nn.functional as F

def info_nce_loss(query, key, temperature=0.1):
    # 归一化，点积就等价于余弦相似度
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)

    # 计算所有样本之间的相似度
    logits = query @ key.T

    # 除以温度系数
    logits = logits / temperature

    # 第 i 个 query 的正样本就是第 i 个 key
    labels = torch.arange(query.size(0))

    return F.cross_entropy(logits, labels)
```

