# Softmax-in-State Linear Attention (SiSA)

在压缩状态的 $d_r$ 个维度上做 softmax（而不是 $n$ 个 token 上），获得维度级指数竞争。

## 1. 三类干扰与三个性质

Linear attention 的三类干扰：

- **尺度干扰**：无归一化时输出随 token 数增长 → 需要 **P1（归一化）**
- **加性噪声干扰**：非目标 token 的 value 混入输出 → 需要 **P2（选择度）**
- **破坏性干扰**：$q \cdot k_i < 0$ 时 token 反向抵消正确答案 → 需要 **P3（非负权重）**

## 2. 为什么多项式不够

所有基于多项式权重 $w_i = P_d(q \cdot k_i)$ 的方法（包括二阶 Taylor 展开、SOAM 的平方读取），选择度上界是：

$$\text{SEL} = \left(\frac{a_j}{a_i}\right)^d$$

$d$ 是多项式阶数。而 softmax 的选择度是 $\exp(\beta(a_j - a_i))$，随 $\beta$ 指数增长，无上界。

| $a_j=0.9, a_i=0.7$ | SEL |
|---|---|
| 线性 ($d$=1) | 1.29 |
| 平方 ($d$=2) | 1.65 |
| 四次 ($d$=4) | 2.73 |
| $\exp(\beta=10)$ | **7.39** |
| $\exp(\beta=20)$ | **54.6** |

多项式的根本问题：有限阶 vs 指数增长，不是展开到几阶能解决的。

Performers 等方法用高维随机特征近似 $\exp(q \cdot k)$，但状态仍是一阶（$S = \sum \phi(k)v^\top$），读取权重仍是线性内积——**升维 ≠ 升阶，升阶 ≠ 超越多项式**。

## 3. 核心 insight：在压缩空间内做 softmax

Softmax attention 之所以是 $O(n^2)$，是因为 softmax 作用在 $n$ 个 token 上。

但如果 softmax 作用在压缩状态的 $d_r$ 个维度上，复杂度只有 $O(d_r)$——和一次向量加法一样便宜。

关键观察：$Gq = \sum_i (q \cdot k_i) k_i \in \mathbb{R}^{d_r}$，每个分量 $(Gq)_l$ 编码了"query 在 key 空间第 $l$ 个方向上的累积响应"。对 $Gq$ 做 softmax：

$$w = \text{softmax}(\beta \cdot G_t \, q_r) \in \mathbb{R}^{d_r}$$

这是 $d_r$ 个方向之间的**指数级竞争**——响应最强的方向指数级碾压其他方向。

然后用 $w$ 门控 query：$\tilde{q} = \text{L2norm}(q_r \odot w)$。

效果：$\tilde{q}$ 集中在 key 空间中与 query 最匹配的方向上，集中程度是指数级的（由 $\beta$ 控制）。

## 4. 结构

### 4.1 状态

只需一阶状态 + Gram 矩阵：

$$S_t = \alpha_t S_{t-1} + \gamma_t \, k_r \otimes v \quad \in \mathbb{R}^{d_r \times d_v}$$
$$G_t = \alpha_t G_{t-1} + \gamma_t \, k_r k_r^\top \quad \in \mathbb{R}^{d_r \times d_r}$$
$$z_t = \alpha_t z_{t-1} + \gamma_t \, k_r \quad \in \mathbb{R}^{d_r}$$

门控：$\alpha_t = \sigma(w_\alpha s_t + b_\alpha)$，$\gamma_t = \sigma(w_\gamma s_t + b_\gamma)$，其中 $s_t = q_r \cdot k_r$。

无 delta rule，无三阶张量。

### 4.2 读取

$$a_t = G_t \, q_r \quad \in \mathbb{R}^{d_r} \qquad \text{(Gram-query 响应)}$$
$$w_t = \text{softmax}(\beta \cdot a_t) \quad \in \mathbb{R}^{d_r} \qquad \text{(维度级指数竞争)}$$
$$\tilde{q}_t = \text{L2norm}(q_r \odot w_t) \qquad \text{(门控精化 query)}$$
$$o_t = S_t^\top \tilde{q}_t \;/\; (z_t^\top \tilde{q}_t + \epsilon) \qquad \text{(归一化读取)}$$

$\beta$ 是 per-head 可学习标量。初始化小值（接近 vanilla linear attention），训练中自动学到合适的锐度。

### 4.3 三个性质

**P1（归一化）**：$z_t^\top \tilde{q}_t$ 提供分母归一化。✓

**P2（选择度）**：softmax 给出指数级竞争。两个维度 $l_1, l_2$ 的权重比：
$$w_{l_1}/w_{l_2} = \exp(\beta \cdot ((Gq)_{l_1} - (Gq)_{l_2}))$$
$\beta$ 越大，选择越尖锐。**没有多项式阶数的天花板。** ✓

**P3（非负性）**：softmax 输出 $w_l > 0$，所以 $\tilde{q}$ 的每个分量与 $q_r$ 同号。当 key 在投影空间中使用非负激活（如 ReLU）时，$\tilde{q} \cdot k_r \geq 0$。部分满足。

### 4.4 完整前向流程

```
Input: x ∈ R^{B×T×D}

1. 投影 + ShortConv + SiLU → q, k, v
2. Reshape + RoPE + L2norm
3. 低秩投影: q_r, k_r = L2norm(W_proj @ q), L2norm(W_proj @ k)
4. 门控: α, γ from sigmoid(w·(q_r·k_r) + b)

5. 状态更新:
   S = α*S + γ*k_r⊗v       # [d_r, d_v]
   G = α*G + γ*k_r⊗k_r     # [d_r, d_r]
   z = α*z + γ*k_r           # [d_r]

6. 读取:
   a = G @ q_r               # [d_r]
   w = softmax(β * a)        # [d_r]  ← 指数竞争
   q̃ = L2norm(q_r * w)       # [d_r]
   o = S^T @ q̃ / (z^T @ q̃ + ε)

7. RMSNorm → o_proj
```

### 4.5 状态大小（SmolLM-135M, $d_r$=16, $d_v$=64, H=9）

| 状态 | 大小/head |
|---|---|
| $S$ | $16 \times 64 = 1024$ |
| $G$ | $16 \times 16 = 256$ |
| $z$ | $16$ |
| **总计/head** | **1296** |

对比：SOAM 16384/head（大 12.6 倍），Taylor-2 17745/head（大 13.7 倍）。

## 5. 与所有方案的对比

| | P1 | P2 | P3 | 状态/head | 非线性 |
|---|---|---|---|---|---|
| Vanilla linear | ✗ | 线性 | ✗ | 1024 | 无 |
| Performers | ✓ | 线性 | ✓ | ~1024 | kernel trick |
| DeltaNet | ✗ | 靠存储 | ✗ | 1024 | 无 |
| GALA | ✗ | $(1+\lambda\sigma)^M$ | ✗ | 1280 | 线性精化 |
| SOAM | ✗ | 平方 | ✓ | 16384 | 二次型 |
| Taylor-2 | ✓ | 平方 | ✓ | 17745 | 多项式 |
| **SiSA** | **✓** | **指数级** | **部分** | **1296** | **softmax** |

SiSA 的定位：用最小的状态（只比 vanilla 多一个 Gram 矩阵），通过 readout 阶段的 softmax 非线性获得最强的选择度。

## 6. 关键问题：维度集中 ≠ token 集中

### 6.1 问题

P2 的本意是 **token 级选择**："从所有历史 token 中精准取出第 $j$ 个"。

SiSA 的 softmax 作用在 $d_r$ 个维度上，给的是**特征方向选择**："在 key 空间中集中到一个方向"。这两者不同。

维度集中后，$\tilde{q}$ 集中在方向 $l^*$ 上，readout 权重 $\tilde{q} \cdot k_i \approx k_{i,l^*}$——**所有在该方向上投影大的 token 都被选中**，不是只选一个。

### 6.2 实际效果：指数级粗筛 + 线性细选

- **跨类区分**（"人名" vs "动词"）：不同语义类落在不同维度 → 维度 softmax 指数级过滤无关类 → **有效**
- **类内区分**（"巴黎" vs "伦敦"，都是城市名）：两者在相同维度上投影相似 → 维度 softmax 无法区分 → **无效**

token 级选择度 = 维度间选择度（指数级） × 维度内选择度（线性）。总体效果取决于 key 的投影结构。

### 6.3 相关工作

**XCiT**（NeurIPS 2021）是最接近的先例：把 softmax 从 token 维度（$N \times N$）转到特征维度（$d \times d$），在 vision transformer 上有效。但 XCiT 每次仍从所有 token 计算 $K^\top Q$（$O(n)$），不是递推压缩状态上的操作。

在 linear attention 递推状态上对 Gram-query 乘积做 softmax，未找到直接先例。

### 6.4 诚实评估

维度 softmax 对 P2 的贡献是**间接的、有条件的**。它提供了一种新的非线性通道，但不能保证 token 级指数选择度。是否真正有效取决于：

1. $W_{\text{proj}}$ 能否学到把不同 token 分到不同维度
2. $d_r = 16$ 是否提供足够的维度来分离关键 token
3. 对实际 LM 任务，"粗筛 + 细选"的组合是否够用

这些只有实验能回答。
