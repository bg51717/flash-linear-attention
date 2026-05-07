# SiSA: Softmax-in-State Attention — 实现方案

以三性质（P1, P2, P3）为设计约束，用最小状态实现指数级选择度的 linear attention。

## 1. 设计目标

满足三性质，同时最小化状态开销：

| 性质 | 含义 | 要求 |
|------|------|------|
| P1 归一化 | 权重和为 1，输出有界 | 分母归一化 |
| P2 选择度 | 能尖锐选出目标 token | 超越线性/多项式的选择机制 |
| P3 非负性 | 不出现破坏性干扰 | 权重 ≥ 0 |

**约束**：状态大小 ≤ 1300/head（不接受 Taylor-2 的 17745 或 SOAM 的 16384）。

## 2. 核心思路：问题在读取，不在存储

简单累积状态 $S_t = \sum \alpha^{t-i} \gamma_i \, k_r \otimes v$ 已保留所有历史信息。DeltaNet 的 delta rule 是在修补存储端（代价：SWA 行为，远距离丢失）。

我们的策略：**存储不动，读取变聪明**。用 Gram 矩阵 $G = \sum k_r k_r^\top$（只多 $d_r^2 = 256$ 个数）编码 key 的分布结构，在 readout 时引入非线性。

关键观察：$Gq_r \in \mathbb{R}^{d_r}$ 的第 $l$ 个分量 $(Gq_r)_l = \sum_i (k_{r,i})_l \, (q_r \cdot k_{r,i})$，编码了"query 在 key 空间第 $l$ 个方向上的累积响应强度"。对这 $d_r$ 个响应做 softmax → 指数级竞争 → 自动集中到最相关的方向。

## 3. 结构

### 3.1 状态递推

$$S_t = \alpha_t \, S_{t-1} + \gamma_t \, k_r \otimes v \quad \in \mathbb{R}^{d_r \times d_v}$$
$$G_t = \alpha_t \, G_{t-1} + \gamma_t \, k_r k_r^\top \quad \in \mathbb{R}^{d_r \times d_r}$$
$$z_t = \alpha_t \, z_{t-1} + \gamma_t \, k_r \quad \in \mathbb{R}^{d_r}$$

门控：$s_t = q_r \cdot k_r$，$\alpha_t = \sigma(w_\alpha s_t + b_\alpha)$，$\gamma_t = \sigma(w_\gamma s_t + b_\gamma)$。

无 delta rule，纯累积 + 全局衰减。

### 3.2 读取

$$a_t = G_t \, q_r \quad \in \mathbb{R}^{d_r} \qquad \text{(Gram-query 响应)}$$
$$w_t = \mathrm{softmax}(\beta \cdot a_t) \quad \in \mathbb{R}^{d_r} \qquad \text{(维度级指数竞争)}$$
$$\tilde{q}_t = \mathrm{L2norm}(q_r \odot w_t) \qquad \text{(门控精化 query)}$$
$$o_t = S_t^\top \tilde{q}_t \qquad \text{(读取)}$$

$\beta$ 是 per-head 可学习标量。$G_t$ 在 softmax 计算时 detach（不传梯度到历史 key）。

### 3.3 关于分母（P1）

WLA 的教训：分母 $z^\top q_w + \epsilon$ 在训练初期可能接近 0，导致梯度爆炸（loss 到 4400 万）。

SiSA 的情况更好：$\tilde{q}$ 被推**向** key 密集方向（GALA 方向），$z^\top \tilde{q}$ 天然较大。但仍有风险。

**方案 A（稳妥）**：不用分母，依赖 output RMSNorm 处理 P1。WLA 去掉分母后 loss 收敛到 ~1.12，证明可行。先跑这个。

**方案 B（完整 P1）**：用分母 $o = S^\top \tilde{q} \;/\; (z^\top \tilde{q} + \epsilon)$，但需要：
- $\beta$ 初始化较小（~0.1），使训练初期 $w \approx \mathrm{uniform}$，$\tilde{q} \approx q_r$，分母稳定
- $\epsilon$ 可以设大一些（1.0 而非 1e-6）作为安全网
- 作为 ablation 实验验证 P1 的实际影响

**建议**：先跑方案 A，确认收敛后再做方案 B 的 ablation。

### 3.4 三性质满足情况

**P1（归一化）**：方案 A 通过 output norm 间接满足；方案 B 通过 z 分母直接满足。

**P2（选择度）**：$w_{l_1}/w_{l_2} = \exp(\beta \cdot ((Gq)_{l_1} - (Gq)_{l_2}))$，$\beta$ 可学习，**指数级**选择度，无多项式阶数天花板。

**P3（非负性）**：部分满足。softmax 输出 $w_l > 0$，$\tilde{q}$ 各分量与 $q_r$ 同号。但 $\tilde{q} \cdot k_i$ 仍可能为负。实践中，SiSA 的门控将 $\tilde{q}$ 集中到与 key 分布最匹配的方向，使得大部分相关 token 的权重为正。

P3 不完全满足是 **honest trade-off**：要在小状态下完全保证 P3，需要非负核（如 ReLU/exp），但会损失表达能力。论文中诚实讨论，实验中做 ablation 验证影响。

## 4. 完整前向流程

```
Input: x ∈ R^{B×T×D}

1. 投影 + ShortConv + SiLU → q, k, v
2. Reshape + RoPE + L2norm
3. 低秩投影: q_r, k_r = L2norm(W_proj @ q), L2norm(W_proj @ k)   # d_k → d_r
4. 门控: α, γ from sigmoid(w·(q_r·k_r) + b)

5. 状态更新:
   S = α*S + γ*k_r⊗v       # [d_r, d_v]
   G = α*G + γ*k_r⊗k_r     # [d_r, d_r]  (readout 时 detach)
   z = α*z + γ*k_r           # [d_r]       (仅方案 B)

6. 读取:
   a = G_detach @ q_r        # [d_r]
   w = softmax(β * a)        # [d_r]  ← 指数竞争
   q̃ = L2norm(q_r * w)       # [d_r]
   o = S^T @ q̃               # [d_v]  (方案 A)
   # o = S^T @ q̃ / (z^T @ q̃ + ε)  (方案 B)

7. output_norm → o_proj
```

## 5. 超参数与初始化

| 参数 | 初始化 | 说明 |
|------|--------|------|
| $d_r$ | 16 | 投影维度，与 WLA/SOAM 一致 |
| $\beta$ | 1.0（`log_beta` 初始化为 0） | per-head 可学习，$\beta = \exp(\text{log\_beta})$ 保证正值。初始化 1.0 对应中等竞争强度，既不太软（$\beta \to 0$，退化 vanilla）也不太硬 |
| $w_\alpha, b_\alpha$ | 0.0, 2.0 | 衰减门初始偏高（α ≈ 0.88），保留大部分历史 |
| $w_\gamma, b_\gamma$ | 1.0, 0.0 | 写入门适中 |
| output_norm | rmsnorm 或 identity | 先用 identity 看基线 |
| qk L2 norm | True, eps=1e-6 | q_r, k_r 投影后做 L2 归一化 |

## 6. 状态大小

| 状态 | 大小/head | 占比 |
|------|-----------|------|
| $S$ | $d_r \times d_v = 1024$ | 79% |
| $G$ | $d_r \times d_r = 256$ | 20% |
| $z$ | $d_r = 16$ | 1% |
| **总计** | **1296** | |

对比：

| 方法 | 状态/head | P2 选择度 | P3 |
|------|-----------|-----------|-----|
| Vanilla | 1024 | 线性 | ✗ |
| DeltaNet | 1024 | 靠存储 | ✗ |
| **SiSA** | **1296** | **指数级** | **部分** |
| SOAM | 16384 | 平方 | ✓ |
| Taylor-2 | 17745 | 平方 | ✓ |

用 **+272 个数**（一个 $16 \times 16$ 矩阵）换来**指数级 → 平方级**的选择度提升。

## 7. 与现有方法的关系

- **vs Vanilla**：加了 G 状态和 softmax 非线性读取，其余完全一致。$\beta \to 0$ 时退化为 vanilla。
- **vs GALA**：GALA 做 $q + \lambda Gq$（线性精化），SiSA 做 $q \odot \mathrm{softmax}(\beta Gq)$（指数级门控）。方向相同（用 G 精化 query），非线性更强。
- **vs WLA**：WLA 用 $(G + \sigma^2 I)^{-1}q$ 白化（抑制噪声方向），SiSA 用 $\mathrm{softmax}(\beta Gq)$ 门控（放大响应方向）。方向相反但互补。
- **vs DeltaNet**：DeltaNet 改存储端（delta rule），SiSA 改读取端（softmax 门控）。SiSA 保留长距离信息。

## 8. 论文定位

**框架贡献**：三性质（P1 归一化、P2 选择度、P3 非负性）作为 linear attention 的诊断和设计工具。分析现有方法（DeltaNet/GLA/Mamba/Performer）各自满足/不满足哪些性质。

**方法贡献**：SiSA 以最小状态代价（+$d_r^2$）获得指数级选择度。核心 insight：在压缩状态的 $d_r$ 维度上做 softmax（$O(d_r)$），替代在 $n$ 个 token 上做 softmax（$O(n^2)$）。

**实验设计**：
1. SmolLM-135M 蒸馏：SiSA vs SOAM vs DeltaNet vs WLA
2. Ablation：去掉 softmax 门控（$\beta = 0$）→ 验证 P2 的贡献
3. Ablation：去掉分母（方案 A vs B）→ 验证 P1 的贡献
4. $\beta$ 学到的值分析：不同层/不同 head 学到什么 $\beta$？
5. 长距离 benchmark：SiSA vs DeltaNet，验证"存储 vs 读取"的论点

## 9. 需要诚实讨论的问题

**维度选择 ≠ token 选择**：softmax 作用在 $d_r$ 个维度上，给的是方向选择，不是 token 选择。同方向上多个 token 无法区分。论文需要：
1. 承认这个局限
2. 论证：实际 LM 中 $d_r = 16$ 的投影空间足以分离主要语义类
3. 实验：在需要精确 token 检索的任务上测试（如 key-value recall）

**P3 部分满足**：$\tilde{q} \cdot k_i$ 仍可负。论文需要：
1. 统计实际训练中负权重的比例
2. 对比加 ReLU 核的变体（完全 P3 但损失表达力）
3. 论证：SiSA 的门控自然减少负权重（集中到匹配方向）
