# Gram-Augmented Linear Attention (GALA)

维护 key 的 Gram 矩阵 $G = \sum k k^\top$ 作为额外状态，用它来**精化 query**再做读取——让读取变尖锐，而不是让存储变精确。

## 1. 为什么这么设计

**问题**：Linear attention 的读取 $o = S^\top q$，权重 $w_i = q \cdot k_i$ 是线性的，不够尖锐（Property 2 不满足）。

**现有思路都在改存储端**：DeltaNet 用 delta rule 让存储更精确，代价是定向擦除导致远距离信息丢失（行为类似 SWA）。

**我们的思路——改读取端**：存储保持简单累积（不丢信息），用 key 的统计信息 $G = \sum k_i k_i^\top$ 来精化 query，使读取更尖锐。

**理论动机**：定义能量 $E(\tilde{q}) = -\frac{\eta}{2} \tilde{q}^\top G \tilde{q} + \frac{1}{2} \|\tilde{q} - q\|^2$，梯度下降一步得到 $\tilde{q} = q + \eta G q$，正好是 query 精化公式。这是 **Hopfield 迭代的低秩压缩近似**——用压缩统计 $G$（$O(d_r^2)$）替代完整 pattern 矩阵 $X$（$O(n)$），复杂度从 $O(n^2)$ 降到 $O(d_r^2)$。

**定位**：

| | 改什么 | 长距离 | 复杂度 |
|---|---|---|---|
| DeltaNet | 存储（delta rule） | 受限（SWA 行为） | $O(n)$ |
| Hopfield | 读取（softmax 迭代） | 保留 | $O(n^2)$ |
| **GALA** | **读取（Gram 精化）** | **保留** | **$O(n)$** |

## 2. 结构

### 2.1 状态更新

$$S_t = \alpha_t \, S_{t-1} + \gamma_t \, k_{r,t} \otimes v_t \quad \in \mathbb{R}^{d_r \times d_v}$$

$$G_t = \alpha_t \, G_{t-1} + \gamma_t \, k_{r,t} \otimes k_{r,t} \quad \in \mathbb{R}^{d_r \times d_r}$$

门控：$s_t = q_{r,t} \cdot k_{r,t}$，$\alpha_t = \sigma(w_\alpha s_t + b_\alpha)$，$\gamma_t = \sigma(w_\gamma s_t + b_\gamma)$。S 和 G 共享门控。

无 delta rule，无定向擦除。

### 2.2 Query 精化 + 读取

$$\tilde{q}_t = \text{L2norm}(q_{r,t} + \lambda \cdot G_t \, q_{r,t})$$

$$o_t = S_t^\top \, \tilde{q}_t$$

$\lambda$ 是 per-head 可学习标量，初始化为 0（退化为 vanilla linear attention）。$G_t$ 在精化时 detach（不传梯度到历史 key）。

### 2.3 完整前向流程

```
Input: x ∈ R^{B×T×D}

1. 投影 + ShortConv + SiLU → q, k, v
2. Reshape + RoPE + L2norm
3. 低秩投影: q_r, k_r = L2norm(W_proj @ q), L2norm(W_proj @ k)   # d_k → d_r
4. 门控: α, γ from sigmoid(w·(q_r·k_r) + b)
5. 状态更新: S = α*S + γ*k_r⊗v,  G = α*G + γ*k_r⊗k_r
6. 精化: q̃ = L2norm(q_r + λ*G@q_r)
7. 读取: o = S^T @ q̃
8. RMSNorm → o_proj
```

### 2.4 状态大小（SmolLM-135M, $d_r$=16, $d_v$=64, H=9）

| 组件 | 大小/head |
|---|---|
| S | $d_r \times d_v$ = 1024 |
| G | $d_r \times d_r$ = 256 |
| **总计/head** | **1280 floats** |

对比 SOAM：16384/head（大 12 倍）。

## 3. 可选扩展

- **多步精化**：$\tilde{q}^{(m+1)} = \text{L2norm}(q_r + \lambda G \tilde{q}^{(m)})$，M=2~3 步
- **门控混合**：$o = g \odot S^\top q_r + (1-g) \odot S^\top \tilde{q}$，让模型选择尖锐/模糊读取
- **归一化读取**：$o = S^\top \tilde{q} \,/\, (n^\top \tilde{q} + \epsilon)$，其中 $n_t = \alpha_t n_{t-1} + \gamma_t k_{r,t}$
