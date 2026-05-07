# Taylor-2 Linear Attention

在 $O(1)$ 状态约束下，用二阶多项式权重最优近似 softmax 的三个关键性质。

## 1. Softmax 的三个性质与三类干扰

Softmax attention 的有效权重 $p_i = \exp(q \cdot k_i) / \sum_j \exp(q \cdot k_j)$ 同时满足三个性质：

**P1（归一化）**：$p_i \in (0,1)$，$\sum p_i = 1$。

解决**尺度干扰**：没有归一化时，$\sum p_i v_i$ 的绝对值随存储 token 数增长。不同序列长度下输出尺度不同，下游层无法稳定处理。归一化让输出始终是 value 的加权平均，尺度有界。

**P2（选择度）**：对任意目标 key $k_j$，总能找到 $q$ 使 $p_j \approx 1$。

解决**加性噪声干扰**：当读取权重不够尖锐时，非目标 token 的 value 混入输出，稀释目标信号。选择度 $\text{SEL} = p_j / \max_{i \neq j} p_i$ 衡量信噪比。SEL 越高，噪声越小。

**P3（非负性）**：$p_i > 0$ 对所有 $i$。

解决**破坏性干扰**：这是比噪声更严重的问题。当 $q \cdot k_i < 0$ 时，token $i$ 对输出的贡献是负的——它不是在稀释正确答案，而是在**主动抵消**正确答案。加性噪声让信号模糊，破坏性干扰让信号反转。

**现有 linear attention 对三个性质的满足情况：**

| | P1 归一化 | P2 选择度 | P3 非负性 |
|---|---|---|---|
| Vanilla linear ($S^\top q$) | ✗ | 差（线性） | ✗（$q \cdot k_i$ 可负） |
| + normalizer ($S^\top q / z^\top q$) | ✓ | 差 | ✗ |
| + ReLU kernel ($\phi(q)^\top \phi(k)$) | ✓ | 差 | ✓ |
| DeltaNet | ✗ | 靠存储精度 | ✗ |
| SOAM ($q^\top T q$) | ✗ | 中（平方） | ✓（$a_i^2 \geq 0$） |

没有一个同时满足三者。

## 2. 为什么需要二阶状态

### 2.1 多项式阶数 ↔ 状态阶数

读取函数能多尖锐，取决于它是什么形式的函数：

| 读取权重形式 | 需要的状态 | 选择度 | 非负性 |
|---|---|---|---|
| $(q \cdot k_i)^1$（线性） | $S = \sum k \otimes v$（一阶） | SEL = $a_j/a_i$ | ✗ |
| $(q \cdot k_i)^2$（平方） | $T = \sum (k \otimes k) \otimes v$（二阶） | SEL = $a_j^2/a_i^2$ | ✓ |
| $(q \cdot k_i)^p$ | $p$ 阶张量 | SEL = $a_j^p/a_i^p$ | $p$ 偶数时 ✓ |

**这是数学上的必然**：想从压缩状态中计算 $(q \cdot k_i)^2 v_i$ 的求和，必须存储 $(k_i \otimes k_i) \otimes v_i$ 的累积——这就是二阶张量。一阶状态无论维度多高（Performers 用 $D \gg d_k$ 的随机特征），权重始终是线性的，P2 上不去。

**二阶是最优权衡**：
- 一阶：选择度不够（P2 差），权重可负（P3 差）
- 二阶：选择度平方提升（P2 中），权重恒正（P3 ✓），状态 $O(d_r^2 d_v)$ 可行
- 三阶：选择度立方，但状态 $O(d_r^3 d_v)$ 不现实

### 2.2 和 Performers 的本质区别

Performers 用随机特征 $\phi(q)^\top \phi(k) \approx \exp(q \cdot k)$ 近似 kernel，但状态仍然是一阶：$S = \sum \phi(k) v^\top$。再高维的 $\phi$，读取权重 $\phi(q) \cdot \phi(k_i)$ 仍是线性内积——**升维不等于升阶**。

我们直接升到二阶状态。这不是"对 softmax 做 Taylor 展开"这么简单——而是认识到：**要近似 softmax 的选择度（P2），状态的阶数必须升高，这是一阶框架的根本限制，不是特征映射的问题。**

### 2.3 Taylor 展开的角色

Taylor 展开回答的不是"要不要升阶"（上面已经回答了），而是**升阶之后各阶怎么组合**。

$\exp(x) \approx 1 + x + \frac{1}{2}x^2$ 给出最优的二阶多项式近似。代入 softmax 的分子分母：

$$o \approx \frac{\sum_i [1 + a_i + \frac{1}{2}a_i^2] \, v_i}{\sum_i [1 + a_i + \frac{1}{2}a_i^2]} = \frac{\bar{v} + S^\top q + \frac{1}{2}\,q^\top T\,q}{n + z^\top q + \frac{1}{2}\,q^\top G\,q}$$

Taylor 展开唯一确定了：需要哪些状态（$\bar{v}, S, T, z, G, n$），以及它们以什么系数组合（$1 : 1 : \frac{1}{2}$）。

## 3. 结构

### 3.1 状态递推

| 状态 | 递推 | 大小 |
|---|---|---|
| $\bar{v}_t$ | $\alpha_t \bar{v}_{t-1} + \gamma_t v_t$ | $d_v$ |
| $S_t$ | $\alpha_t S_{t-1} + \gamma_t k_r \otimes v$ | $d_r \times d_v$ |
| $T_t$ | $\alpha_t T_{t-1} + \gamma_t (k_r \otimes k_r) \otimes v$ | $d_r^2 \times d_v$ |
| $z_t$ | $\alpha_t z_{t-1} + \gamma_t k_r$ | $d_r$ |
| $G_t$ | $\alpha_t G_{t-1} + \gamma_t k_r k_r^\top$ | $d_r \times d_r$ |
| $n_t$ | $\alpha_t n_{t-1} + \gamma_t$ | 1 |

门控：$s_t = q_r \cdot k_r$，$\alpha_t = \sigma(w_\alpha s_t + b_\alpha)$，$\gamma_t = \sigma(w_\gamma s_t + b_\gamma)$。

无 delta rule。所有状态纯累积 + 全局衰减。

### 3.2 读取

$$o_t = \frac{\mu_0 \bar{v}_t + \mu_1 S_t^\top q_r + \mu_2 \, q_r^\top T_t \, q_r}{\mu_0 n_t + \mu_1 z_t^\top q_r + \mu_2 \, q_r^\top G_t \, q_r + \epsilon}$$

$\mu_0, \mu_1, \mu_2$ 是 per-head 可学习标量，控制各阶的比例。模型可以自己学：
- $\mu_2 \gg \mu_0, \mu_1$：尖锐检索模式（SEL → $a_j^2/a_i^2$）
- $\mu_1$ 主导：平滑融合模式（上下文理解）
- $\mu_0$ 主导：全局默认值（fallback）

### 3.3 三个性质的满足

**P1（归一化）**：分母 $= \sum_i \mu_0 + \mu_1 a_i + \mu_2 a_i^2$。当 $\mu$ 为正且接近 Taylor 系数时，每个 token 的贡献 $\mu_0 + \mu_1 a_i + \mu_2 a_i^2 > 0$（见下），总和归一化。✓

**P2（选择度）**：当 $\mu_2$ 主导时，有效权重 $\propto a_i^2$，SEL = $a_j^2/a_i^2$。比线性好一个量级。✓

**P3（非负性）**：Taylor 系数下，$1 + a + \frac{1}{2}a^2 = \frac{1}{2}(1+a)^2 + \frac{1}{2} \geq \frac{1}{2}$，对 $a \in [-1, 1]$ 恒正。当 $\mu$ 接近 Taylor 系数时，每个 token 的权重恒正，无破坏性干扰。✓

**首次同时满足三个性质。**

### 3.4 完整前向流程

```
Input: x ∈ R^{B×T×D}

1. 投影 + ShortConv + SiLU → q, k, v
2. Reshape + RoPE + L2norm
3. 低秩投影: q_r, k_r = L2norm(W_proj @ q), L2norm(W_proj @ k)
4. 门控: α, γ from sigmoid(w·(q_r·k_r) + b)

5. 状态更新:
   v̄ = α*v̄ + γ*v                      # [d_v]
   S = α*S + γ*k_r⊗v                   # [d_r, d_v]
   T = α*T + γ*(k_r⊗k_r)⊗v            # [d_r, d_r, d_v]
   z = α*z + γ*k_r                      # [d_r]
   G = α*G + γ*k_r⊗k_r                 # [d_r, d_r]
   n = α*n + γ                          # scalar

6. 读取:
   num = μ₀*v̄ + μ₁*S^T@q_r + μ₂*q_r^T@T@q_r
   den = μ₀*n + μ₁*z^T@q_r + μ₂*q_r^T@G@q_r + ε
   o = num / den

7. RMSNorm → o_proj
```

### 3.5 状态大小（SmolLM-135M, $d_r$=16, $d_v$=64, H=9）

$T$ 占 93%（16384/head），其余 ~1361。总计 17745/head，比 SOAM 多 ~8%。

## 4. 与现有方法对比

| | P1 | P2 | P3 | 长距离 | 状态主体 |
|---|---|---|---|---|---|
| Vanilla linear | ✗ | 差 | ✗ | ✓ | 一阶 |
| Performers | ✓ | 差 | ✓ | ✓ | 一阶（高维） |
| DeltaNet | ✗ | 中 | ✗ | ✗（SWA） | 一阶 + delta |
| SOAM | ✗ | 中 | ✓ | ✗（delta） | 二阶 + delta |
| **Taylor-2** | **✓** | **中~高** | **✓** | **✓** | **二阶，无 delta** |
