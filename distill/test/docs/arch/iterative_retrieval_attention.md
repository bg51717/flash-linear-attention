# Iterative Retrieval Linear Attention (IRLA)

## 1. 出发点：Softmax Attention 的三个关键性质

**性质 1**：归一化权重 $p_i \in (0, 1)$。

**性质 2（选择性）**：无论其他 token 的 key 如何，总能找到一个 query 使 $p_i \approx 1$。这是因为 $\exp(q \cdot k_i)$ 可以相对其他项指数级地大。

**性质 3（因果性）**：可以按时间步逐步计算。

Linear attention 的根本困难在于性质 2。核函数近似 $\phi(q)^\top \phi(k_i) \approx \exp(q \cdot k_i)$ 后，归一化权重变成：
$$p_i = \frac{\phi(q)^\top \phi(k_i)}{\phi(q)^\top \sum_j \phi(k_j)}$$

分子分母都是线性函数，无法产生指数级的分离度——除非 $\phi(k_j)$ 恰好接近零（低概率事件）。所以**线性读取 + 非负核 = 无法尖锐选择**。

## 2. Delta Rule 为什么像滑动窗口

DeltaNet 的更新：
$$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$$

$(I - \beta_t k_t k_t^\top)$ 在 $k_t$ 方向上做投影擦除。如果后续 token 的 key 与某个早期 token 的 key 相近，早期信息就被覆盖。

假设 key 空间有效维度约为 $d_k$，那么大约 $d_k$ 个不同方向的 key 之后，状态里的早期信息基本被覆盖殆尽。这意味着 DeltaNet 的**有效记忆长度约为 $O(d_k)$**，在行为上类似于一个窗口大小为 $d_k$ 的滑动窗口注意力。

后续改进（GatedDeltaNet、decay gate 等）缓解但不改变这个本质：delta rule 是**定向遗忘**，长期信息不可避免地被覆盖。

## 3. 核心观察：问题在读取，不在存储

简单累积的状态：
$$S_t = \alpha \, S_{t-1} + k_t \otimes v_t = \sum_{i=1}^{t} \alpha^{t-i} \, k_i \otimes v_i$$

当 $\alpha$ 接近 1 时，所有历史 token 的信息都保留在 $S_t$ 中（只有缓慢的指数衰减）。**信息没有丢失。**

问题出在读取：
$$o = S_t^\top q = \sum_i \alpha^{t-i} (q \cdot k_i) \, v_i$$

权重 $(q \cdot k_i)$ 是线性的，无法尖锐。Delta rule 选择"让存储变精确"来绕过这个问题。但我们也可以反过来：**保留模糊存储，让读取变尖锐**。

## 4. 提案：迭代读取（Iterative Retrieval）

### 4.1 状态更新（纯累积，无定向擦除）

$$S_t = \alpha_t \, S_{t-1} + \gamma_t \, k_t \otimes v_t$$

- $\alpha_t = \sigma(w_\alpha \cdot s_t + b_\alpha)$：soft global decay（全局缓慢衰减，非定向擦除）
- $\gamma_t = \sigma(w_\gamma \cdot s_t + b_\gamma)$：write gate（决定新信息是否写入）
- $s_t = q_t \cdot k_t$：内容相关的门控信号

关键区别：$\alpha_t$ 是**全局衰减**（所有方向统一缩放），不是 delta rule 的定向擦除。旧信息只是缓慢变淡，不会被覆盖。

### 4.2 迭代读取

$$q^{(0)} = q_t$$

$$\hat{v}^{(m)} = S_t^\top \, q^{(m-1)} \qquad \text{(用当前 query 从状态中读取)}$$

$$q^{(m)} = \text{normalize}\!\left(q^{(m-1)} + W_m \, \hat{v}^{(m)}\right) \qquad \text{(用读出结果修正 query)}$$

$$o_t = \hat{v}^{(M)}$$

其中 $W_m \in \mathbb{R}^{d_k \times d_v}$ 是可学习的投影矩阵（每步迭代一个，或者共享），normalize 是 L2 归一化。

### 4.3 为什么迭代能变尖锐

**第 1 步**：$\hat{v}^{(1)} = \sum_i w_i^{(0)} v_i$，权重 $w_i^{(0)} = q \cdot k_i$，线性的，模糊。

$W_1$ 把 $\hat{v}^{(1)}$ 近似映射回 key 空间，所以：
$$q^{(1)} \approx \text{norm}\left(q + \sum_i (q \cdot k_i) \, k_i\right)$$

如果 $q$ 与目标 key $k_j$ 最对齐，$q^{(1)}$ 会更偏向 $k_j$。

**第 2 步**：权重变成 $w_i^{(1)} = q^{(1)} \cdot k_i \propto (q \cdot k_i)^2$ 量级（近似），更尖锐。

**一般地**：每迭代一步，有效权重的"对比度"大致翻倍（类似幂法 / power iteration）。$M$ 步后，有效权重近似 $(q \cdot k_i)^{2^M}$。

| 迭代步数 M | 有效权重量级 | 等价尖锐度 |
|---|---|---|
| 0 | $(q \cdot k_i)^1$ | 线性（vanilla linear attn） |
| 1 | $(q \cdot k_i)^2$ | 二次 |
| 2 | $(q \cdot k_i)^4$ | 四次 |
| 3 | $(q \cdot k_i)^8$ | 八次（接近 softmax） |

M = 2~3 步即可获得接近 softmax 的选择性，而状态始终保留所有历史信息。

### 4.4 与现有方法的本质区别

| | Kernel Approx | Delta Rule | **Iterative Retrieval** |
|---|---|---|---|
| 存储策略 | 累积 | 定向覆盖 | 累积（soft decay） |
| 读取策略 | 单步线性 | 单步线性 | **多步迭代** |
| 尖锐性 | 差（线性） | N/A（靠精确存储） | **好（指数级收敛）** |
| 远距离信息 | 保留 | 丢失（类 SWA） | **保留** |
| 计算量 per token | $O(d_k d_v)$ | $O(d_k d_v)$ | $O(M \cdot d_k d_v)$ |

## 5. 可叠加的扩展方向

### 5.1 多尺度状态

多个状态并行，衰减率不同：
$$S_t^{(r)} = \alpha_r \, S_{t-1}^{(r)} + k_t \otimes v_t, \quad r = 1, \ldots, R$$

$\alpha_1 \approx 1$（长记忆），$\alpha_R \ll 1$（短记忆）。迭代读取在所有状态上进行：
$$\hat{v}^{(m)} = \sum_r \lambda_r \, (S_t^{(r)})^\top q^{(m-1)}$$

混合系数 $\lambda_r$ 可学习或通过门控动态调整。

### 5.2 高维随机投影增强单步尖锐度

将 key/query 投影到高维空间：$\phi(k) = \sigma(Rk) \in \mathbb{R}^D$，$D \gg d_k$。

状态变为 $S_t \in \mathbb{R}^{D \times d_v}$，单步读取的尖锐度就提高了（高维向量更倾向正交，交叉干扰更小）。

迭代读取仍然可以叠加在上面，进一步提升尖锐度。代价是状态大小从 $d_k d_v$ 增长到 $D d_v$。

### 5.3 双组件记忆

分离长期和短期记忆（类似工作记忆 vs 长期记忆）：

- **累积组件** $A_t = A_{t-1} + \gamma_t^A \, k_t \otimes v_t$（无衰减，只做 write gating）
- **易失组件** $V_t = \alpha_t \, V_{t-1} + \gamma_t^V \, k_t \otimes v_t$（快速衰减）

读取时混合：$o = f(A_t^\top q, \; V_t^\top q)$，其中 $f$ 可以是门控混合或拼接后投影。

## 6. 待验证的关键问题

1. **迭代读取的训练稳定性**：多步迭代的反向传播是否会导致梯度爆炸/消失？可能需要 stop-gradient 或 detach 某些中间步。

2. **$W_m$ 的学习难度**：$W_m$ 需要近似学到 value→key 的映射，这个映射是否容易学？是否需要结构化约束（比如让 $W_m = K_{\text{proj}}^\top V_{\text{proj}}^{-\top}$）？

3. **M 的选择**：M 越大越尖锐但越慢。实际需要多大的 M？是否可以让 M 自适应（简单情况 M=1，困难情况 M=3）？

4. **与 softmax 的差距**：即使有效权重达到 $(q \cdot k_i)^8$，这和 $\exp(q \cdot k_i)$ 的行为仍然不同（多项式 vs 指数）。在长序列上是否足够？

5. **Triton 实现可行性**：迭代读取需要在每个时间步做 M 次 $S^\top q$。状态 $S$ 在寄存器中，$q$ 也在寄存器中，矩阵向量乘可以 inline 在循环里。初步看可行，但需要验证寄存器压力。

## 7. 最小实验方案

如果要验证这个想法，最小实验：

1. 在现有 SOAM/DeltaNet 代码基础上，只改 readout 部分为 2 步迭代
2. 状态更新保持简单累积 + soft decay
3. 先用 PyTorch naive 实现跑 SmolLM-135M 蒸馏
4. 对比：vanilla linear attn（M=0） vs 迭代（M=1,2,3） vs DeltaNet
5. 重点看：长距离任务（TriviaQA 等生成式评测）的差异
