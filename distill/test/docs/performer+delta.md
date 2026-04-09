设
$$
S_t = W_t,
$$

$$
\phi_t = \phi_{\mathrm{FAVOR+}}(k_t),\qquad
\psi_t = \phi_{\mathrm{FAVOR+}}(q_t).
$$

其中 FAVOR+ 特征映射为
$$
h(x)=\frac{1}{\sqrt{2}}\exp\left(-\frac{\|x\|^2}{2}\right),
$$

$$
\phi_{\mathrm{FAVOR+}}(x)
=
\frac{h(x)}{\sqrt{m}}
\begin{bmatrix}
\exp(Rx)\\
\exp(-Rx)
\end{bmatrix}.
$$

对 key/query 做 sum normalization：
$$
\bar{\phi}_t
=
\frac{\phi_t}{\mathbf{1}^\top \phi_t + \varepsilon},
\qquad
\bar{\psi}_t
=
\frac{\psi_t}{\mathbf{1}^\top \psi_t + \varepsilon}.
$$

先读取当前 key 上的旧值：
$$
r_t = W_{t-1}\bar{\phi}_t.
$$

Update:
$$
\mathrm{Update}_{\mathrm{SN\text{-}Perf\text{-}\Delta}}(W_{t-1},k_t,v_t):
\qquad
W_t
=
W_{t-1}
+
\left(v_t-r_t\right)\bar{\phi}_t^\top.
$$

把 $r_t$ 展开后，也可以直接写成
$$
W_t
=
W_{t-1}
+
\left(v_t-W_{t-1}\bar{\phi}_t\right)\bar{\phi}_t^\top.
$$

Output:
$$
\mathrm{Output}_{\mathrm{SN\text{-}Perf\text{-}\Delta}}(W_t,q_t)
=
W_t\bar{\psi}_t.
$$