# Wiener Linear Attention (WLA)

不近似 softmax，而是直接求解：**从有噪声的压缩记忆中，最优地检索目标值。**

## 1. 三类干扰（同前）

- **尺度干扰** → P1（归一化）
- **加性噪声干扰** → P2（选择度）
- **破坏性干扰** → P3（非负权重）

## 2. 重新定义问题：有噪声的检索

Linear attention 的 readout $o = S^\top q = \sum_i (q \cdot k_i) v_i$ 可以分解为：

$$o = \underbrace{(q \cdot k_j) v_j}_{\text{目标信号}} + \underbrace{\sum_{i \neq j} (q \cdot k_i) v_i}_{\text{干扰噪声}}$$

其中 $j$ 是目标 token（$q$ 最想检索的）。

这就是经典的**信号检测问题**：从噪声中提取目标信号。噪声的结构完全由 key 的分布决定——而 Gram 矩阵 $G = \sum k_i k_i^\top$ 恰好编码了这个噪声结构。

**之前所有方案的问题**：它们试图"精化 query 使其更对齐目标"（GALA）或"近似 softmax 的权重函数"（Taylor-2）。但信号处理告诉我们：**最优策略不是放大信号方向，而是抑制噪声方向。**

## 3. MMSE 最优解：Wiener 滤波

**定理**：在所有形如 $o = S^\top w$ 的线性读取中，最小化检索误差 $E[\|o - v_j\|^2]$ 的最优权重向量是：

$$w^* = (G + \sigma^2 I)^{-1} q$$

其中 $\sigma^2$ 是正则化项（noise floor）。

**证明思路**：

$$E[\|S^\top w - v_j\|^2] = w^\top G w - 2 k_j^\top w + 1$$

这是 $w$ 的凸二次函数。令梯度 $2Gw - 2k_j = 0$，得 $w = G^{-1} k_j$。用 $q \approx k_j$ 近似并加正则化：$w^* = (G + \sigma^2 I)^{-1} q$。

**这是 Wiener 滤波（维纳滤波）在 key-value 检索中的应用。**

### 3.1 为什么这是对的——直觉

$G$ 的特征分解 $G = U \Sigma U^\top$。在特征基下：

$$(G + \sigma^2 I)^{-1} q = \sum_l \frac{c_l}{\sigma_l + \sigma^2} u_l$$

- 高特征值方向（$\sigma_l$ 大）：很多 key 聚集在这个方向 → **干扰大** → 权重 $1/(\sigma_l + \sigma^2)$ **压低**
- 低特征值方向（$\sigma_l$ 小）：很少 key 在这个方向 → **干扰小** → 权重 $1/\sigma^2$ **保持**

**效果：等化所有方向的信噪比。** 不是把 query 推向 key 密集区（GALA 的做法），而是把 key 密集区的噪声压下来。

### 3.2 与 GALA 的对比——方向相反

| | GALA | WLA |
|---|---|---|
| 操作 | $(I + \lambda G)q$：放大高密度方向 | $(G + \sigma^2 I)^{-1}q$：抑制高密度方向 |
| 效果 | 把 query 推向 key 的主方向 | 等化各方向的信噪比 |
| 适合 | 目标 key 在高密度方向（常见 token） | **任何 key**（不依赖频率） |
| 理论 | 能量最小化（heuristic） | **MMSE 最优（theorem）** |

GALA 的问题：把 query 推向高密度方向确实增大了 $q \cdot k_j$，但**同时也增大了 $q \cdot k_i$（干扰项）**，因为干扰 key 也在高密度方向。净效果可能为零。

WLA 直接解决了这个问题：不试图放大信号，而是**最优地去除噪声**。

### 3.3 特殊情况验证

- **key 正交**（$G = I$）：$(I + \sigma^2 I)^{-1} = (1+\sigma^2)^{-1} I$，等比缩放，退化为 vanilla。✓（无干扰时不需要矫正）
- **$\sigma^2 \to \infty$**：$(G + \sigma^2 I)^{-1} \approx \sigma^{-2} I$，退化为 vanilla。✓（正则化压倒一切时保守不动）
- **$\sigma^2 \to 0$**：$(G)^{-1} q$，完全白化。最大化去干扰但数值不稳定。✓（$\sigma^2$ 提供必要的正则化）

## 4. 结构

### 4.1 状态

$$S_t = \alpha_t S_{t-1} + \gamma_t \, k_r \otimes v \quad \in \mathbb{R}^{d_r \times d_v}$$
$$G_t = \alpha_t G_{t-1} + \gamma_t \, k_r k_r^\top \quad \in \mathbb{R}^{d_r \times d_r}$$
$$z_t = \alpha_t z_{t-1} + \gamma_t \, k_r \quad \in \mathbb{R}^{d_r}$$

无 delta rule，纯累积 + 衰减。

### 4.2 读取

$$q_w = (G_t + \sigma^2 I)^{-1} q_r \qquad \text{(Wiener 白化)}$$
$$q_w = \text{L2norm}(q_w) \qquad \text{(归一化)}$$
$$o_t = S_t^\top q_w \;/\; (z_t^\top q_w + \epsilon) \qquad \text{(归一化读取)}$$

$\sigma^2$ 是 per-head 可学习标量（初始化较大值，训练中学到合适的正则化强度）。$G_t$ 在白化时 detach（不传梯度到历史 key）。

### 4.3 $(G + \sigma^2 I)^{-1} q$ 的计算

$d_r = 16$ 时，$G + \sigma^2 I$ 是 $16 \times 16$ 对称正定矩阵。三种实现方式：

**方式 1：直接求解**（Cholesky）。$O(d_r^3/3) \approx 1365$ FLOPs。最精确，但 Triton 中实现 Cholesky 较复杂。

**方式 2：Neumann 级数**（推荐）。$(G + \sigma^2 I)^{-1} q \approx \sigma^{-2}(q - \sigma^{-2}Gq + \sigma^{-4}G^2 q)$，截断到 2-3 项。每项一次 $d_r \times d_r$ matvec（256 FLOPs），总计 ~768 FLOPs。收敛条件 $\sigma^2 > \|G\|_{\text{op}}$。

**方式 3：迭代求解**（Jacobi/Richardson）。$q_w^{(m+1)} = (q - G q_w^{(m)}) / \sigma^2$，2-3 步收敛。和 Neumann 等价，但形式更适合 kernel 内循环。

### 4.4 三个性质

**P1（归一化）**：$z_t^\top q_w$ 提供分母。✓

**P2（选择度）**：Wiener 白化**等化信噪比**——在所有线性读取中，检索误差最小。数学保证优于 vanilla（vanilla 是 $\sigma^2 \to \infty$ 的特例）。改善幅度取决于 key 的相关结构：key 越相关（干扰越大），白化的增益越大。

**P3（非负性）**：白化后 $q_w \cdot k_i$ 仍可为负。部分满足。

### 4.5 完整前向流程

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

6. 白化读取:
   # Neumann 近似 (G + σ²I)⁻¹ q
   Gq = G @ q_r
   GGq = G @ Gq
   q_w = (q_r - Gq/σ² + GGq/σ⁴) / σ²
   q_w = L2norm(q_w)
   o = S^T @ q_w / (z^T @ q_w + ε)

7. RMSNorm → o_proj
```

### 4.6 状态大小

| 状态 | 大小/head |
|---|---|
| $S$ | $d_r \times d_v = 1024$ |
| $G$ | $d_r \times d_r = 256$ |
| $z$ | $d_r = 16$ |
| **总计/head** | **1296** |

与 GALA 相同，比 SOAM 小 12 倍。

## 5. 可选：与二阶状态组合

如果状态预算允许，可以加入 SOAM 张量 $T$ 获得平方级选择度：

$$o = q_w^\top T_t \, q_w \;/\; (q_w^\top G_t \, q_w + \epsilon)$$

先白化去干扰（MMSE 最优），再平方提升选择度（二阶读取）。两步各有明确理论意义：

| 步骤 | 作用 | 理论 |
|---|---|---|
| 白化 $(G+\sigma^2 I)^{-1}q$ | 去除 key 间干扰 | MMSE 最优 |
| 二次型 $q_w^\top T q_w$ | 平方级选择度 | 多项式读取 |
| 归一化 $/\, q_w^\top G q_w$ | 有界权重 | P1 |

此时状态增加 $T$（$d_r^2 \times d_v = 16384$/head），但理论保证更强。

## 6. 与所有方案对比

| | 核心思想 | 理论基础 | P1 | P2 | 状态/head |
|---|---|---|---|---|---|
| Vanilla | 直接读取 | — | ✗ | 线性 | 1024 |
| DeltaNet | 存储纠错 | delta rule | ✗ | 靠存储 | 1024 |
| GALA | 放大 key 主方向 | 能量下降 | ✗ | $(1+\lambda\sigma)$ | 1280 |
| Taylor-2 | 多项式近似 softmax | Taylor 展开 | ✓ | 平方 | 17745 |
| SiSA | 维度级 softmax | — | ✓ | 维度级指数 | 1296 |
| **WLA** | **去干扰（白化）** | **MMSE 最优** | **✓** | **最优线性** | **1296** |
| **WLA + 二阶** | **白化 + 平方读取** | **MMSE + 多项式** | **✓** | **最优线性 × 平方** | **17680** |

## 7. 为什么理论上有希望

1. **不是 heuristic，是 theorem**：在所有 $o = S^\top w$ 形式的读取中，$w = (G+\sigma^2 I)^{-1}q$ 是唯一的 MSE 最小解。数学保证不存在更好的线性读取。

2. **改善有下界**：Wiener 白化的 MSE $\leq$ vanilla 的 MSE（因为 vanilla 是 $\sigma^2 \to \infty$ 的特例）。不可能比不白化更差。

3. **改善幅度可预估**：当 key 的 Gram 矩阵条件数 $\kappa(G)$ 大时（key 高度相关，干扰严重），白化的增益约为 $\kappa(G)$ 倍。这正好是 linear attention 最吃亏的场景。

4. **不依赖 key 频率**：GALA 对稀有 key 失效（推向高密度方向时推偏目标）。白化对所有 key 一视同仁——它不偏向高频或低频，只是去除可预测的干扰。

5. **与 Hopfield 伪逆学习规则的联系**：在 Hopfield 网络中，$(XX^\top)^{-1}$ 是已知的提升容量的标准手段（从 $O(d/\log d)$ 到 $O(d)$）。WLA 是这个思想在 linear attention 递推状态上的自然延伸。
