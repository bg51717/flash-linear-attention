from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def _torch_forward_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_alpha: torch.Tensor,
    gate_bias: torch.Tensor,
    z_score_eps: float,
) -> torch.Tensor:
    qf = q.float()
    kf = k.float()
    vf = v.float()
    ga = gate_alpha.float()
    gb = gate_bias.float()
    n, t, k_dim = qf.shape
    v_dim = vf.shape[-1]

    d_state = torch.zeros((n, v_dim, k_dim), device=q.device, dtype=torch.float32)
    nu = torch.zeros((n, v_dim), device=q.device, dtype=torch.float32)
    kap = torch.zeros((n, k_dim), device=q.device, dtype=torch.float32)
    o_prev = torch.zeros((n, v_dim), device=q.device, dtype=torch.float32)
    q_prev = torch.zeros((n, k_dim), device=q.device, dtype=torch.float32)
    count = torch.zeros((n,), device=q.device, dtype=torch.float32)
    mu = torch.zeros((n,), device=q.device, dtype=torch.float32)
    m2 = torch.zeros((n,), device=q.device, dtype=torch.float32)

    out = torch.empty((n, t, v_dim), device=q.device, dtype=torch.float32)
    for i in range(t):
        q_i = qf[:, i]
        k_i = kf[:, i]
        v_i = vf[:, i]

        c_t = count + 1.0
        alpha_w = c_t.reciprocal()
        lam = count * alpha_w

        x_i = v_i - nu
        y_i = k_i - kap
        d_state = d_state + lam[:, None, None] * torch.einsum("nv,nk->nvk", x_i, y_i)

        dq_i = q_i - q_prev
        inv_count = torch.where(count > 0, count.reciprocal(), torch.zeros_like(count))
        o_hist = o_prev + inv_count[:, None] * torch.einsum("nvk,nk->nv", d_state, dq_i)

        score = torch.sum(k_i * q_i, dim=-1)
        delta1 = score - mu.detach()
        var_prev = m2.detach() / torch.clamp(count.detach(), min=1.0)
        z = delta1 / torch.sqrt(var_prev + z_score_eps)
        gate_val = ga * z + gb
        beta = torch.sigmoid(gate_val)

        o_i = o_hist + beta[:, None] * (v_i - o_hist)
        out[:, i] = o_i

        with torch.no_grad():
            mu_new = mu + (score - mu) / c_t
            delta2 = score - mu_new
            m2 = m2 + (score - mu) * delta2
            mu = mu_new

        nu = nu + alpha_w[:, None] * (v_i - nu)
        kap = kap + alpha_w[:, None] * (k_i - kap)
        o_prev = o_i
        q_prev = q_i
        count = c_t

    return out.to(q.dtype)


def _bwd_torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_alpha: torch.Tensor,
    gate_bias: torch.Tensor,
    do: torch.Tensor,
    z_score_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        a_ = gate_alpha.detach().requires_grad_(True)
        b_ = gate_bias.detach().requires_grad_(True)
        o_ = _torch_forward_dense(q_, k_, v_, a_, b_, z_score_eps)
        dq, dk, dv, da, db = torch.autograd.grad(o_, (q_, k_, v_, a_, b_), do)
    return dq, dk, dv, da, db


def _external_state_to_internal(
    state: tuple[torch.Tensor, ...] | None,
    batch: int,
    n_heads: int,
    k_dim: int,
    v_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...] | None:
    if state is None:
        return None
    if len(state) != 8:
        raise ValueError("initial_state must be an 8-tuple: (D, nu, kappa, o_prev, q_prev, count, mu, m2).")
    d_state, nu, kap, o_prev, q_prev, count, mu, m2 = state
    n = batch * n_heads
    return (
        d_state.reshape(n, k_dim, v_dim).transpose(-2, -1).contiguous().to(device=device, dtype=torch.float32),
        nu.reshape(n, v_dim).contiguous().to(device=device, dtype=torch.float32),
        kap.reshape(n, k_dim).contiguous().to(device=device, dtype=torch.float32),
        o_prev.reshape(n, v_dim).contiguous().to(device=device, dtype=torch.float32),
        q_prev.reshape(n, k_dim).contiguous().to(device=device, dtype=torch.float32),
        count.reshape(n).contiguous().to(device=device, dtype=torch.float32),
        mu.reshape(n).contiguous().to(device=device, dtype=torch.float32),
        m2.reshape(n).contiguous().to(device=device, dtype=torch.float32),
    )


def _internal_state_to_external(
    state: tuple[torch.Tensor, ...],
    batch: int,
    n_heads: int,
    k_dim: int,
    v_dim: int,
) -> tuple[torch.Tensor, ...]:
    d_state, nu, kap, o_prev, q_prev, count, mu, m2 = state
    return (
        d_state.reshape(batch, n_heads, v_dim, k_dim).transpose(-2, -1).contiguous(),
        nu.reshape(batch, n_heads, v_dim).contiguous(),
        kap.reshape(batch, n_heads, k_dim).contiguous(),
        o_prev.reshape(batch, n_heads, v_dim).contiguous(),
        q_prev.reshape(batch, n_heads, k_dim).contiguous(),
        count.reshape(batch, n_heads).contiguous(),
        mu.reshape(batch, n_heads).contiguous(),
        m2.reshape(batch, n_heads).contiguous(),
    )


if _TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=["T"])
    def _approxnet_v4_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        alpha_ptr,
        bias_ptr,
        o_ptr,
        T,
        Z_SCORE_EPS,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        n = tl.program_id(0).to(tl.int64)

        offs_k = tl.arange(0, BK)
        offs_v = tl.arange(0, BV)
        mask_k = offs_k < K
        mask_v = offs_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        b_d = tl.zeros([BV, BK], dtype=tl.float32)
        b_nu = tl.zeros([BV], dtype=tl.float32)
        b_kap = tl.zeros([BK], dtype=tl.float32)
        b_o_prev = tl.zeros([BV], dtype=tl.float32)
        b_q_prev = tl.zeros([BK], dtype=tl.float32)
        b_count = tl.zeros([], dtype=tl.float32)
        b_mu = tl.zeros([], dtype=tl.float32)
        b_m2 = tl.zeros([], dtype=tl.float32)

        b_alpha_val = tl.load(alpha_ptr + n).to(tl.float32)
        b_bias_val = tl.load(bias_ptr + n).to(tl.float32)

        p_q = q_ptr + n * T * K + offs_k
        p_k = k_ptr + n * T * K + offs_k
        p_v = v_ptr + n * T * V + offs_v
        p_o = o_ptr + n * T * V + offs_v

        for _ in range(0, T):
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

            b_c_t = b_count + 1.0
            b_alpha_w = 1.0 / b_c_t
            b_lam = b_count * b_alpha_w

            b_x = b_v - b_nu
            b_y = b_k - b_kap
            b_d += b_lam * (b_x[:, None] * b_y[None, :])

            b_dq = b_q - b_q_prev
            b_inv_count = tl.where(b_count > 0.0, 1.0 / b_count, 0.0)
            b_hist = b_o_prev + b_inv_count * tl.sum(b_d * b_dq[None, :], axis=1)

            b_score = tl.sum(b_k * b_q, axis=0)
            b_delta1 = b_score - b_mu
            b_var_prev = b_m2 / tl.maximum(b_count, 1.0)
            b_z = b_delta1 / tl.sqrt(b_var_prev + Z_SCORE_EPS)
            b_gate = b_alpha_val * b_z + b_bias_val
            b_beta = tl.sigmoid(b_gate)

            b_o = b_hist + b_beta * (b_v - b_hist)
            tl.store(p_o, b_o.to(o_ptr.dtype.element_ty), mask=mask_v)

            b_mu_new = b_mu + b_delta1 / b_c_t
            b_delta2 = b_score - b_mu_new
            b_m2 = b_m2 + b_delta1 * b_delta2
            b_mu = b_mu_new

            b_nu += b_alpha_w * (b_v - b_nu)
            b_kap += b_alpha_w * (b_k - b_kap)
            b_o_prev = b_o
            b_q_prev = b_q
            b_count = b_c_t

            p_q += K
            p_k += K
            p_v += V
            p_o += V

    @triton.jit(do_not_specialize=["T"])
    def _approxnet_v4_fwd_state_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        alpha_ptr,
        bias_ptr,
        o_ptr,
        d_in_ptr,
        nu_in_ptr,
        kap_in_ptr,
        o_in_ptr,
        qp_in_ptr,
        c_in_ptr,
        mu_in_ptr,
        m2_in_ptr,
        d_out_ptr,
        nu_out_ptr,
        kap_out_ptr,
        o_out_ptr,
        qp_out_ptr,
        c_out_ptr,
        mu_out_ptr,
        m2_out_ptr,
        T,
        Z_SCORE_EPS,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        n = tl.program_id(0).to(tl.int64)

        offs_k = tl.arange(0, BK)
        offs_v = tl.arange(0, BV)
        mask_k = offs_k < K
        mask_v = offs_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        b_d = tl.load(
            d_in_ptr + n * V * K + offs_v[:, None] * K + offs_k[None, :],
            mask=mask_h, other=0,
        ).to(tl.float32)
        b_nu = tl.load(nu_in_ptr + n * V + offs_v, mask=mask_v, other=0).to(tl.float32)
        b_kap = tl.load(kap_in_ptr + n * K + offs_k, mask=mask_k, other=0).to(tl.float32)
        b_o_prev = tl.load(o_in_ptr + n * V + offs_v, mask=mask_v, other=0).to(tl.float32)
        b_q_prev = tl.load(qp_in_ptr + n * K + offs_k, mask=mask_k, other=0).to(tl.float32)
        b_count = tl.load(c_in_ptr + n).to(tl.float32)
        b_mu = tl.load(mu_in_ptr + n).to(tl.float32)
        b_m2 = tl.load(m2_in_ptr + n).to(tl.float32)

        b_alpha_val = tl.load(alpha_ptr + n).to(tl.float32)
        b_bias_val = tl.load(bias_ptr + n).to(tl.float32)

        p_q = q_ptr + n * T * K + offs_k
        p_k = k_ptr + n * T * K + offs_k
        p_v = v_ptr + n * T * V + offs_v
        p_o = o_ptr + n * T * V + offs_v

        for _ in range(0, T):
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

            b_c_t = b_count + 1.0
            b_alpha_w = 1.0 / b_c_t
            b_lam = b_count * b_alpha_w

            b_x = b_v - b_nu
            b_y = b_k - b_kap
            b_d += b_lam * (b_x[:, None] * b_y[None, :])

            b_dq = b_q - b_q_prev
            b_inv_count = tl.where(b_count > 0.0, 1.0 / b_count, 0.0)
            b_hist = b_o_prev + b_inv_count * tl.sum(b_d * b_dq[None, :], axis=1)

            b_score = tl.sum(b_k * b_q, axis=0)
            b_delta1 = b_score - b_mu
            b_var_prev = b_m2 / tl.maximum(b_count, 1.0)
            b_z = b_delta1 / tl.sqrt(b_var_prev + Z_SCORE_EPS)
            b_gate = b_alpha_val * b_z + b_bias_val
            b_beta = tl.sigmoid(b_gate)

            b_o = b_hist + b_beta * (b_v - b_hist)
            tl.store(p_o, b_o.to(o_ptr.dtype.element_ty), mask=mask_v)

            b_mu_new = b_mu + b_delta1 / b_c_t
            b_delta2 = b_score - b_mu_new
            b_m2 = b_m2 + b_delta1 * b_delta2
            b_mu = b_mu_new

            b_nu += b_alpha_w * (b_v - b_nu)
            b_kap += b_alpha_w * (b_k - b_kap)
            b_o_prev = b_o
            b_q_prev = b_q
            b_count = b_c_t

            p_q += K
            p_k += K
            p_v += V
            p_o += V

        tl.store(
            d_out_ptr + n * V * K + offs_v[:, None] * K + offs_k[None, :],
            b_d.to(d_out_ptr.dtype.element_ty), mask=mask_h,
        )
        tl.store(nu_out_ptr + n * V + offs_v, b_nu.to(nu_out_ptr.dtype.element_ty), mask=mask_v)
        tl.store(kap_out_ptr + n * K + offs_k, b_kap.to(kap_out_ptr.dtype.element_ty), mask=mask_k)
        tl.store(o_out_ptr + n * V + offs_v, b_o_prev.to(o_out_ptr.dtype.element_ty), mask=mask_v)
        tl.store(qp_out_ptr + n * K + offs_k, b_q_prev.to(qp_out_ptr.dtype.element_ty), mask=mask_k)
        tl.store(c_out_ptr + n, b_count.to(c_out_ptr.dtype.element_ty))
        tl.store(mu_out_ptr + n, b_mu.to(mu_out_ptr.dtype.element_ty))
        tl.store(m2_out_ptr + n, b_m2.to(m2_out_ptr.dtype.element_ty))

    @triton.jit(do_not_specialize=["T", "t_start", "L"])
    def _approxnet_v4_chunk_state_kernel(
        q_ptr, k_ptr, v_ptr, alpha_ptr, bias_ptr,
        d_in_ptr, nu_in_ptr, kap_in_ptr, o_in_ptr, qp_in_ptr, c_in_ptr, mu_in_ptr, m2_in_ptr,
        d_out_ptr, nu_out_ptr, kap_out_ptr, o_out_ptr, qp_out_ptr, c_out_ptr, mu_out_ptr, m2_out_ptr,
        t_start, L, T, Z_SCORE_EPS,
        K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    ):
        i = tl.program_id(0)
        offs_k = tl.arange(0, BK)
        offs_v = tl.arange(0, BV)
        mask_k = offs_k < K
        mask_v = offs_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        b_d = tl.load(d_in_ptr + i * V * K + offs_v[:, None] * K + offs_k[None, :], mask=mask_h, other=0).to(tl.float32)
        b_nu = tl.load(nu_in_ptr + i * V + offs_v, mask=mask_v, other=0).to(tl.float32)
        b_kap = tl.load(kap_in_ptr + i * K + offs_k, mask=mask_k, other=0).to(tl.float32)
        b_o_prev = tl.load(o_in_ptr + i * V + offs_v, mask=mask_v, other=0).to(tl.float32)
        b_q_prev = tl.load(qp_in_ptr + i * K + offs_k, mask=mask_k, other=0).to(tl.float32)
        b_count = tl.load(c_in_ptr + i).to(tl.float32)
        b_mu = tl.load(mu_in_ptr + i).to(tl.float32)
        b_m2 = tl.load(m2_in_ptr + i).to(tl.float32)

        b_alpha_val = tl.load(alpha_ptr + i).to(tl.float32)
        b_bias_val = tl.load(bias_ptr + i).to(tl.float32)

        p_q = q_ptr + i * T * K + t_start * K + offs_k
        p_k = k_ptr + i * T * K + t_start * K + offs_k
        p_v = v_ptr + i * T * V + t_start * V + offs_v

        for _ in range(0, L):
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

            b_c_t = b_count + 1.0
            b_alpha_w = 1.0 / b_c_t
            b_lam = b_count * b_alpha_w

            b_x = b_v - b_nu
            b_y = b_k - b_kap
            b_d += b_lam * (b_x[:, None] * b_y[None, :])

            b_dq = b_q - b_q_prev
            b_inv_count = tl.where(b_count > 0.0, 1.0 / b_count, 0.0)
            b_hist = b_o_prev + b_inv_count * tl.sum(b_d * b_dq[None, :], axis=1)

            b_score = tl.sum(b_k * b_q, axis=0)
            b_delta1 = b_score - b_mu
            b_var_prev = b_m2 / tl.maximum(b_count, 1.0)
            b_z = b_delta1 / tl.sqrt(b_var_prev + Z_SCORE_EPS)
            b_gate = b_alpha_val * b_z + b_bias_val
            b_beta = tl.sigmoid(b_gate)
            b_o = b_hist + b_beta * (b_v - b_hist)

            b_mu_new = b_mu + b_delta1 / b_c_t
            b_delta2 = b_score - b_mu_new
            b_m2 = b_m2 + b_delta1 * b_delta2
            b_mu = b_mu_new

            b_nu += b_alpha_w * (b_v - b_nu)
            b_kap += b_alpha_w * (b_k - b_kap)
            b_o_prev = b_o
            b_q_prev = b_q
            b_count = b_c_t

            p_q += K
            p_k += K
            p_v += V

        tl.store(d_out_ptr + i * V * K + offs_v[:, None] * K + offs_k[None, :], b_d.to(d_out_ptr.dtype.element_ty), mask=mask_h)
        tl.store(nu_out_ptr + i * V + offs_v, b_nu.to(nu_out_ptr.dtype.element_ty), mask=mask_v)
        tl.store(kap_out_ptr + i * K + offs_k, b_kap.to(kap_out_ptr.dtype.element_ty), mask=mask_k)
        tl.store(o_out_ptr + i * V + offs_v, b_o_prev.to(o_out_ptr.dtype.element_ty), mask=mask_v)
        tl.store(qp_out_ptr + i * K + offs_k, b_q_prev.to(qp_out_ptr.dtype.element_ty), mask=mask_k)
        tl.store(c_out_ptr + i, b_count.to(c_out_ptr.dtype.element_ty))
        tl.store(mu_out_ptr + i, b_mu.to(mu_out_ptr.dtype.element_ty))
        tl.store(m2_out_ptr + i, b_m2.to(m2_out_ptr.dtype.element_ty))

    @triton.jit(do_not_specialize=["T", "t_start", "L"])
    def _approxnet_v4_chunk_fused_bwd_kernel(
        q_ptr, k_ptr, v_ptr, do_ptr,
        alpha_ptr, bias_ptr,
        cp_d_ptr, cp_nu_ptr, cp_kap_ptr, cp_o_ptr, cp_qp_ptr, cp_c_ptr, cp_mu_ptr, cp_m2_ptr,
        dD_next_ptr, dnu_next_ptr, dkap_next_ptr, do_next_ptr, dqp_next_ptr,
        dq_ptr, dk_ptr, dv_ptr,
        dD_prev_ptr, dnu_prev_ptr, dkap_prev_ptr, do_prev_ptr, dqp_prev_ptr,
        d_alpha_ptr, d_bias_ptr,
        local_nu_ptr, local_kap_ptr, local_o_ptr, local_qp_ptr, local_c_ptr,
        local_mu_ptr, local_m2_ptr,
        t_start, L, T, Z_SCORE_EPS,
        K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
        CHUNK_MAX: tl.constexpr,
    ):
        i = tl.program_id(0)
        offs_k = tl.arange(0, BK)
        offs_v = tl.arange(0, BV)
        mask_k = offs_k < K
        mask_v = offs_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        b_alpha_val = tl.load(alpha_ptr + i).to(tl.float32)
        b_bias_val = tl.load(bias_ptr + i).to(tl.float32)

        # ---------- forward recompute ----------
        b_d = tl.load(cp_d_ptr + i * V * K + offs_v[:, None] * K + offs_k[None, :], mask=mask_h, other=0).to(tl.float32)
        b_nu = tl.load(cp_nu_ptr + i * V + offs_v, mask=mask_v, other=0).to(tl.float32)
        b_kap = tl.load(cp_kap_ptr + i * K + offs_k, mask=mask_k, other=0).to(tl.float32)
        b_o_prev = tl.load(cp_o_ptr + i * V + offs_v, mask=mask_v, other=0).to(tl.float32)
        b_q_prev = tl.load(cp_qp_ptr + i * K + offs_k, mask=mask_k, other=0).to(tl.float32)
        b_count = tl.load(cp_c_ptr + i).to(tl.float32)
        b_mu = tl.load(cp_mu_ptr + i).to(tl.float32)
        b_m2 = tl.load(cp_m2_ptr + i).to(tl.float32)

        p_lnu_base = local_nu_ptr + i * (CHUNK_MAX + 1) * V + offs_v
        p_lkap_base = local_kap_ptr + i * (CHUNK_MAX + 1) * K + offs_k
        p_lo_base = local_o_ptr + i * (CHUNK_MAX + 1) * V + offs_v
        p_lqp_base = local_qp_ptr + i * (CHUNK_MAX + 1) * K + offs_k
        p_lc_base = local_c_ptr + i * (CHUNK_MAX + 1)
        p_lmu_base = local_mu_ptr + i * (CHUNK_MAX + 1)
        p_lm2_base = local_m2_ptr + i * (CHUNK_MAX + 1)

        tl.store(p_lnu_base, b_nu.to(local_nu_ptr.dtype.element_ty), mask=mask_v)
        tl.store(p_lkap_base, b_kap.to(local_kap_ptr.dtype.element_ty), mask=mask_k)
        tl.store(p_lo_base, b_o_prev.to(local_o_ptr.dtype.element_ty), mask=mask_v)
        tl.store(p_lqp_base, b_q_prev.to(local_qp_ptr.dtype.element_ty), mask=mask_k)
        tl.store(p_lc_base, b_count.to(local_c_ptr.dtype.element_ty))
        tl.store(p_lmu_base, b_mu.to(local_mu_ptr.dtype.element_ty))
        tl.store(p_lm2_base, b_m2.to(local_m2_ptr.dtype.element_ty))

        p_lnu = p_lnu_base + V
        p_lkap = p_lkap_base + K
        p_lo = p_lo_base + V
        p_lqp = p_lqp_base + K
        p_lc = p_lc_base + 1
        p_lmu = p_lmu_base + 1
        p_lm2 = p_lm2_base + 1

        p_q = q_ptr + i * T * K + t_start * K + offs_k
        p_k = k_ptr + i * T * K + t_start * K + offs_k
        p_v = v_ptr + i * T * V + t_start * V + offs_v

        for _ in range(0, L):
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

            b_c_t = b_count + 1.0
            b_alpha_w = 1.0 / b_c_t
            b_lam = b_count * b_alpha_w

            b_x = b_v - b_nu
            b_y = b_k - b_kap
            b_d += b_lam * (b_x[:, None] * b_y[None, :])

            b_dq_step = b_q - b_q_prev
            b_inv_count = tl.where(b_count > 0.0, 1.0 / b_count, 0.0)
            b_hist = b_o_prev + b_inv_count * tl.sum(b_d * b_dq_step[None, :], axis=1)

            b_score = tl.sum(b_k * b_q, axis=0)
            b_delta1 = b_score - b_mu
            b_var_prev = b_m2 / tl.maximum(b_count, 1.0)
            b_z = b_delta1 / tl.sqrt(b_var_prev + Z_SCORE_EPS)
            b_gate = b_alpha_val * b_z + b_bias_val
            b_beta = tl.sigmoid(b_gate)
            b_o = b_hist + b_beta * (b_v - b_hist)

            b_mu_new = b_mu + b_delta1 / b_c_t
            b_delta2 = b_score - b_mu_new
            b_m2 = b_m2 + b_delta1 * b_delta2
            b_mu = b_mu_new

            b_nu += b_alpha_w * (b_v - b_nu)
            b_kap += b_alpha_w * (b_k - b_kap)
            b_o_prev = b_o
            b_q_prev = b_q
            b_count = b_c_t

            tl.store(p_lnu, b_nu.to(local_nu_ptr.dtype.element_ty), mask=mask_v)
            tl.store(p_lkap, b_kap.to(local_kap_ptr.dtype.element_ty), mask=mask_k)
            tl.store(p_lo, b_o_prev.to(local_o_ptr.dtype.element_ty), mask=mask_v)
            tl.store(p_lqp, b_q_prev.to(local_qp_ptr.dtype.element_ty), mask=mask_k)
            tl.store(p_lc, b_count.to(local_c_ptr.dtype.element_ty))
            tl.store(p_lmu, b_mu.to(local_mu_ptr.dtype.element_ty))
            tl.store(p_lm2, b_m2.to(local_m2_ptr.dtype.element_ty))

            p_q += K
            p_k += K
            p_v += V
            p_lnu += V
            p_lkap += K
            p_lo += V
            p_lqp += K
            p_lc += 1
            p_lmu += 1
            p_lm2 += 1

        # ---------- backward ----------
        b_dD = tl.load(dD_next_ptr + i * V * K + offs_v[:, None] * K + offs_k[None, :], mask=mask_h, other=0).to(tl.float32)
        b_dnu = tl.load(dnu_next_ptr + i * V + offs_v, mask=mask_v, other=0).to(tl.float32)
        b_dkap = tl.load(dkap_next_ptr + i * K + offs_k, mask=mask_k, other=0).to(tl.float32)
        b_do_carry = tl.load(do_next_ptr + i * V + offs_v, mask=mask_v, other=0).to(tl.float32)
        b_dqp_carry = tl.load(dqp_next_ptr + i * K + offs_k, mask=mask_k, other=0).to(tl.float32)

        b_da_accum = tl.zeros([], dtype=tl.float32)
        b_db_accum = tl.zeros([], dtype=tl.float32)

        p_q = q_ptr + i * T * K + (t_start + L - 1) * K + offs_k
        p_k = k_ptr + i * T * K + (t_start + L - 1) * K + offs_k
        p_v = v_ptr + i * T * V + (t_start + L - 1) * V + offs_v
        p_do = do_ptr + i * T * V + (t_start + L - 1) * V + offs_v
        p_dq = dq_ptr + i * T * K + (t_start + L - 1) * K + offs_k
        p_dk = dk_ptr + i * T * K + (t_start + L - 1) * K + offs_k
        p_dv = dv_ptr + i * T * V + (t_start + L - 1) * V + offs_v

        p_nu_prev = p_lnu_base + (L - 1) * V
        p_kap_prev = p_lkap_base + (L - 1) * K
        p_o_prev_rd = p_lo_base + (L - 1) * V
        p_q_prev_rd = p_lqp_base + (L - 1) * K
        p_c_prev = p_lc_base + (L - 1)
        p_mu_prev = p_lmu_base + (L - 1)
        p_m2_prev = p_lm2_base + (L - 1)

        for _ in range(0, L):
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_do_val = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)

            b_nu_pv = tl.load(p_nu_prev, mask=mask_v, other=0).to(tl.float32)
            b_kap_pv = tl.load(p_kap_prev, mask=mask_k, other=0).to(tl.float32)
            b_o_pv = tl.load(p_o_prev_rd, mask=mask_v, other=0).to(tl.float32)
            b_q_pv = tl.load(p_q_prev_rd, mask=mask_k, other=0).to(tl.float32)
            b_c_pv = tl.load(p_c_prev).to(tl.float32)
            b_mu_pv = tl.load(p_mu_prev).to(tl.float32)
            b_m2_pv = tl.load(p_m2_prev).to(tl.float32)

            b_d_t = b_d

            b_c_t = b_c_pv + 1.0
            b_alpha_w = 1.0 / b_c_t
            b_lam = b_c_pv * b_alpha_w
            b_inv_count = tl.where(b_c_pv > 0.0, 1.0 / b_c_pv, 0.0)

            b_dq_step = b_q - b_q_pv
            b_h = b_o_pv + b_inv_count * tl.sum(b_d_t * b_dq_step[None, :], axis=1)

            b_score = tl.sum(b_k * b_q, axis=0)
            b_delta1 = b_score - b_mu_pv
            b_var_prev = b_m2_pv / tl.maximum(b_c_pv, 1.0)
            b_rsqrt_var = 1.0 / tl.sqrt(b_var_prev + Z_SCORE_EPS)
            b_z = b_delta1 * b_rsqrt_var
            b_gate = b_alpha_val * b_z + b_bias_val
            b_beta = tl.sigmoid(b_gate)

            b_do_tot = b_do_val + b_do_carry
            b_d_beta = tl.sum(b_do_tot * (b_v - b_h), axis=0)
            b_dh = b_do_tot * (1.0 - b_beta)

            b_dq_out = b_dqp_carry
            b_dk_out = tl.zeros([BK], dtype=tl.float32)
            b_dv_out = b_do_tot * b_beta

            # gate backward: beta = sigmoid(gate), gate = alpha*z + bias
            b_d_gate = b_d_beta * b_beta * (1.0 - b_beta)
            b_da_accum += b_d_gate * b_z
            b_db_accum += b_d_gate
            b_dz = b_d_gate * b_alpha_val
            b_d_delta1 = b_dz * b_rsqrt_var
            b_d_score = b_d_delta1
            b_dk_out += b_d_score * b_q
            b_dq_out += b_d_score * b_k

            # hist backward
            b_dD = b_dD + b_inv_count * b_dh[:, None] * b_dq_step[None, :]
            b_ddq = b_inv_count * tl.sum(b_d_t * b_dh[:, None], axis=0)
            b_dq_out += b_ddq
            b_dqp_pv = -b_ddq
            b_do_pv = b_dh

            # D update backward
            b_x = b_v - b_nu_pv
            b_y = b_k - b_kap_pv
            b_dX = b_lam * tl.sum(b_dD * b_y[None, :], axis=1)
            b_dY = b_lam * tl.sum(b_dD * b_x[:, None], axis=0)
            b_dv_out += b_dX
            b_dnu_pv = -b_dX
            b_dk_out += b_dY
            b_dkap_pv = -b_dY
            b_dD_prev = b_dD

            # mean update backward
            b_dnu_pv += b_dnu * (1.0 - b_alpha_w)
            b_dv_out += b_dnu * b_alpha_w
            b_dkap_pv += b_dkap * (1.0 - b_alpha_w)
            b_dk_out += b_dkap * b_alpha_w

            tl.store(p_dq, b_dq_out.to(dq_ptr.dtype.element_ty), mask=mask_k)
            tl.store(p_dk, b_dk_out.to(dk_ptr.dtype.element_ty), mask=mask_k)
            tl.store(p_dv, b_dv_out.to(dv_ptr.dtype.element_ty), mask=mask_v)

            b_dD = b_dD_prev
            b_dnu = b_dnu_pv
            b_dkap = b_dkap_pv
            b_do_carry = b_do_pv
            b_dqp_carry = b_dqp_pv

            # undo D update
            b_d = b_d - b_lam * (b_x[:, None] * b_y[None, :])

            p_q -= K
            p_k -= K
            p_v -= V
            p_do -= V
            p_dq -= K
            p_dk -= K
            p_dv -= V
            p_nu_prev -= V
            p_kap_prev -= K
            p_o_prev_rd -= V
            p_q_prev_rd -= K
            p_c_prev -= 1
            p_mu_prev -= 1
            p_m2_prev -= 1

        tl.store(dD_prev_ptr + i * V * K + offs_v[:, None] * K + offs_k[None, :], b_dD.to(dD_prev_ptr.dtype.element_ty), mask=mask_h)
        tl.store(dnu_prev_ptr + i * V + offs_v, b_dnu.to(dnu_prev_ptr.dtype.element_ty), mask=mask_v)
        tl.store(dkap_prev_ptr + i * K + offs_k, b_dkap.to(dkap_prev_ptr.dtype.element_ty), mask=mask_k)
        tl.store(do_prev_ptr + i * V + offs_v, b_do_carry.to(do_prev_ptr.dtype.element_ty), mask=mask_v)
        tl.store(dqp_prev_ptr + i * K + offs_k, b_dqp_carry.to(dqp_prev_ptr.dtype.element_ty), mask=mask_k)

        # accumulate d_alpha, d_bias across chunks
        old_da = tl.load(d_alpha_ptr + i).to(tl.float32)
        tl.store(d_alpha_ptr + i, (old_da + b_da_accum).to(d_alpha_ptr.dtype.element_ty))
        old_db = tl.load(d_bias_ptr + i).to(tl.float32)
        tl.store(d_bias_ptr + i, (old_db + b_db_accum).to(d_bias_ptr.dtype.element_ty))


def _fwd_triton_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_alpha: torch.Tensor,
    gate_bias: torch.Tensor,
    z_score_eps: float,
) -> torch.Tensor:
    n, t, k_dim = q.shape
    v_dim = v.shape[-1]
    bk = triton.next_power_of_2(k_dim)
    bv = triton.next_power_of_2(v_dim)
    if bk > 128 or bv > 128:
        raise ValueError(f"Unsupported head dims K={k_dim}, V={v_dim}; Triton path supports <=128.")

    out = torch.empty((n, t, v_dim), device=q.device, dtype=q.dtype)
    num_warps = 4 if max(bk, bv) >= 64 else 2
    _approxnet_v4_fwd_kernel[(n,)](
        q_ptr=q, k_ptr=k, v_ptr=v,
        alpha_ptr=gate_alpha, bias_ptr=gate_bias,
        o_ptr=out, T=t,
        Z_SCORE_EPS=float(z_score_eps),
        K=k_dim, V=v_dim, BK=bk, BV=bv,
        num_warps=num_warps, num_stages=1,
    )
    return out


def _fwd_triton_with_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_alpha: torch.Tensor,
    gate_bias: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None,
    output_final_state: bool,
    z_score_eps: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    n, t, k_dim = q.shape
    v_dim = v.shape[-1]
    bk = triton.next_power_of_2(k_dim)
    bv = triton.next_power_of_2(v_dim)
    if bk > 128 or bv > 128:
        raise ValueError(f"Unsupported head dims K={k_dim}, V={v_dim}; Triton path supports <=128.")
    num_warps = 4 if max(bk, bv) >= 64 else 2

    out = torch.empty((n, t, v_dim), device=q.device, dtype=q.dtype)

    if initial_state is None:
        d_in = torch.zeros((n, v_dim, k_dim), device=q.device, dtype=torch.float32)
        nu_in = torch.zeros((n, v_dim), device=q.device, dtype=torch.float32)
        kap_in = torch.zeros((n, k_dim), device=q.device, dtype=torch.float32)
        o_in = torch.zeros((n, v_dim), device=q.device, dtype=torch.float32)
        qp_in = torch.zeros((n, k_dim), device=q.device, dtype=torch.float32)
        c_in = torch.zeros((n,), device=q.device, dtype=torch.float32)
        mu_in = torch.zeros((n,), device=q.device, dtype=torch.float32)
        m2_in = torch.zeros((n,), device=q.device, dtype=torch.float32)
    else:
        d_in, nu_in, kap_in, o_in, qp_in, c_in, mu_in, m2_in = (
            s.to(device=q.device, dtype=torch.float32).contiguous() for s in initial_state
        )

    d_out = torch.empty_like(d_in)
    nu_out = torch.empty_like(nu_in)
    kap_out = torch.empty_like(kap_in)
    o_out = torch.empty_like(o_in)
    qp_out = torch.empty_like(qp_in)
    c_out = torch.empty_like(c_in)
    mu_out = torch.empty_like(mu_in)
    m2_out = torch.empty_like(m2_in)

    _approxnet_v4_fwd_state_kernel[(n,)](
        q_ptr=q, k_ptr=k, v_ptr=v,
        alpha_ptr=gate_alpha, bias_ptr=gate_bias,
        o_ptr=out,
        d_in_ptr=d_in, nu_in_ptr=nu_in, kap_in_ptr=kap_in,
        o_in_ptr=o_in, qp_in_ptr=qp_in, c_in_ptr=c_in,
        mu_in_ptr=mu_in, m2_in_ptr=m2_in,
        d_out_ptr=d_out, nu_out_ptr=nu_out, kap_out_ptr=kap_out,
        o_out_ptr=o_out, qp_out_ptr=qp_out, c_out_ptr=c_out,
        mu_out_ptr=mu_out, m2_out_ptr=m2_out,
        T=t, Z_SCORE_EPS=float(z_score_eps),
        K=k_dim, V=v_dim, BK=bk, BV=bv,
        num_warps=num_warps, num_stages=1,
    )
    final_state = (d_out, nu_out, kap_out, o_out, qp_out, c_out, mu_out, m2_out) if output_final_state else None
    return out, final_state


def _bwd_triton_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_alpha: torch.Tensor,
    gate_bias: torch.Tensor,
    do: torch.Tensor,
    z_score_eps: float,
    recompute_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, t, k_dim = q.shape
    v_dim = v.shape[-1]
    chunk = max(1, int(recompute_chunk_size))
    n_chunks = math.ceil(t / chunk)

    qf = q.float().contiguous()
    kf = k.float().contiguous()
    vf = v.float().contiguous()
    gaf = gate_alpha.float().contiguous()
    gbf = gate_bias.float().contiguous()
    dof = do.float().contiguous()

    dqf = torch.empty_like(qf)
    dkf = torch.empty_like(kf)
    dvf = torch.empty_like(vf)

    bk = triton.next_power_of_2(k_dim)
    bv = triton.next_power_of_2(v_dim)
    if bk > 128 or bv > 128:
        raise ValueError(f"Unsupported head dims K={k_dim}, V={v_dim}; Triton path supports <=128.")
    num_warps = 4 if max(bk, bv) >= 64 else 2

    # Checkpoint arrays
    cp_d = torch.zeros((n_chunks + 1, n, v_dim, k_dim), device=q.device, dtype=torch.float32)
    cp_nu = torch.zeros((n_chunks + 1, n, v_dim), device=q.device, dtype=torch.float32)
    cp_kap = torch.zeros((n_chunks + 1, n, k_dim), device=q.device, dtype=torch.float32)
    cp_o = torch.zeros((n_chunks + 1, n, v_dim), device=q.device, dtype=torch.float32)
    cp_qp = torch.zeros((n_chunks + 1, n, k_dim), device=q.device, dtype=torch.float32)
    cp_c = torch.zeros((n_chunks + 1, n), device=q.device, dtype=torch.float32)
    cp_mu = torch.zeros((n_chunks + 1, n), device=q.device, dtype=torch.float32)
    cp_m2 = torch.zeros((n_chunks + 1, n), device=q.device, dtype=torch.float32)

    for c in range(n_chunks):
        s = c * chunk
        l = min(chunk, t - s)
        _approxnet_v4_chunk_state_kernel[(n,)](
            q_ptr=qf, k_ptr=kf, v_ptr=vf, alpha_ptr=gaf, bias_ptr=gbf,
            d_in_ptr=cp_d[c], nu_in_ptr=cp_nu[c], kap_in_ptr=cp_kap[c],
            o_in_ptr=cp_o[c], qp_in_ptr=cp_qp[c], c_in_ptr=cp_c[c],
            mu_in_ptr=cp_mu[c], m2_in_ptr=cp_m2[c],
            d_out_ptr=cp_d[c + 1], nu_out_ptr=cp_nu[c + 1], kap_out_ptr=cp_kap[c + 1],
            o_out_ptr=cp_o[c + 1], qp_out_ptr=cp_qp[c + 1], c_out_ptr=cp_c[c + 1],
            mu_out_ptr=cp_mu[c + 1], m2_out_ptr=cp_m2[c + 1],
            t_start=s, L=l, T=t, Z_SCORE_EPS=float(z_score_eps),
            K=k_dim, V=v_dim, BK=bk, BV=bv,
            num_warps=num_warps, num_stages=1,
        )

    # Local state for fused backward
    local_nu = torch.empty((n, chunk + 1, v_dim), device=q.device, dtype=torch.float32)
    local_kap = torch.empty((n, chunk + 1, k_dim), device=q.device, dtype=torch.float32)
    local_o = torch.empty((n, chunk + 1, v_dim), device=q.device, dtype=torch.float32)
    local_qp = torch.empty((n, chunk + 1, k_dim), device=q.device, dtype=torch.float32)
    local_c = torch.empty((n, chunk + 1), device=q.device, dtype=torch.float32)
    local_mu = torch.empty((n, chunk + 1), device=q.device, dtype=torch.float32)
    local_m2 = torch.empty((n, chunk + 1), device=q.device, dtype=torch.float32)

    # Gradient carries (5 carries: dD, dnu, dkap, do, dqp; no dr)
    dD_next = torch.zeros((n, v_dim, k_dim), device=q.device, dtype=torch.float32)
    dnu_next = torch.zeros((n, v_dim), device=q.device, dtype=torch.float32)
    dkap_next = torch.zeros((n, k_dim), device=q.device, dtype=torch.float32)
    do_next = torch.zeros((n, v_dim), device=q.device, dtype=torch.float32)
    dqp_next = torch.zeros((n, k_dim), device=q.device, dtype=torch.float32)

    dD_prev = torch.empty_like(dD_next)
    dnu_prev = torch.empty_like(dnu_next)
    dkap_prev = torch.empty_like(dkap_next)
    do_prev = torch.empty_like(do_next)
    dqp_prev = torch.empty_like(dqp_next)

    # Accumulated gate param gradients
    d_alpha_out = torch.zeros((n,), device=q.device, dtype=torch.float32)
    d_bias_out = torch.zeros((n,), device=q.device, dtype=torch.float32)

    for c in range(n_chunks - 1, -1, -1):
        s = c * chunk
        l = min(chunk, t - s)
        _approxnet_v4_chunk_fused_bwd_kernel[(n,)](
            q_ptr=qf, k_ptr=kf, v_ptr=vf, do_ptr=dof,
            alpha_ptr=gaf, bias_ptr=gbf,
            cp_d_ptr=cp_d[c], cp_nu_ptr=cp_nu[c], cp_kap_ptr=cp_kap[c],
            cp_o_ptr=cp_o[c], cp_qp_ptr=cp_qp[c], cp_c_ptr=cp_c[c],
            cp_mu_ptr=cp_mu[c], cp_m2_ptr=cp_m2[c],
            dD_next_ptr=dD_next, dnu_next_ptr=dnu_next, dkap_next_ptr=dkap_next,
            do_next_ptr=do_next, dqp_next_ptr=dqp_next,
            dq_ptr=dqf, dk_ptr=dkf, dv_ptr=dvf,
            dD_prev_ptr=dD_prev, dnu_prev_ptr=dnu_prev, dkap_prev_ptr=dkap_prev,
            do_prev_ptr=do_prev, dqp_prev_ptr=dqp_prev,
            d_alpha_ptr=d_alpha_out, d_bias_ptr=d_bias_out,
            local_nu_ptr=local_nu, local_kap_ptr=local_kap,
            local_o_ptr=local_o, local_qp_ptr=local_qp, local_c_ptr=local_c,
            local_mu_ptr=local_mu, local_m2_ptr=local_m2,
            t_start=s, L=l, T=t, Z_SCORE_EPS=float(z_score_eps),
            K=k_dim, V=v_dim, BK=bk, BV=bv, CHUNK_MAX=chunk,
            num_warps=num_warps, num_stages=1,
        )
        dD_next, dD_prev = dD_prev, dD_next
        dnu_next, dnu_prev = dnu_prev, dnu_next
        dkap_next, dkap_prev = dkap_prev, dkap_next
        do_next, do_prev = do_prev, do_next
        dqp_next, dqp_prev = dqp_prev, dqp_next

    return dqf.to(q.dtype), dkf.to(k.dtype), dvf.to(v.dtype), d_alpha_out, d_bias_out


class _ApproxNetV4DenseFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate_alpha: torch.Tensor,
        gate_bias: torch.Tensor,
        z_score_eps: float,
        recompute_chunk_size: int,
    ) -> torch.Tensor:
        out = _fwd_triton_dense(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            gate_alpha=gate_alpha.contiguous(),
            gate_bias=gate_bias.contiguous(),
            z_score_eps=z_score_eps,
        )
        ctx.save_for_backward(q, k, v, gate_alpha, gate_bias)
        ctx.z_score_eps = float(z_score_eps)
        ctx.recompute_chunk_size = int(recompute_chunk_size)
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, gate_alpha, gate_bias = ctx.saved_tensors
        dq, dk, dv, d_alpha, d_bias = _bwd_triton_dense(
            q=q,
            k=k,
            v=v,
            gate_alpha=gate_alpha,
            gate_bias=gate_bias,
            do=do.contiguous(),
            z_score_eps=ctx.z_score_eps,
            recompute_chunk_size=ctx.recompute_chunk_size,
        )
        return dq, dk, dv, d_alpha, d_bias, None, None


def approxnet_v4_linear_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_alpha: torch.Tensor,
    gate_bias: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    z_score_eps: float = 1.0,
    recompute_chunk_size: int = 128,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, dict[str, float]]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape [B, T, H, D].")
    if not _TRITON_AVAILABLE:
        raise RuntimeError("ApproxNet-v4 Triton kernels are unavailable in current environment.")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("ApproxNet-v4 Triton kernels require CUDA tensors.")
    needs_backward = q.requires_grad or k.requires_grad or v.requires_grad or gate_alpha.requires_grad or gate_bias.requires_grad
    if torch.is_grad_enabled() and needs_backward and (initial_state is not None or output_final_state):
        raise RuntimeError("ApproxNet-v4 Triton state path currently supports inference forward only (no backward).")

    b, t, h, k_dim = q.shape
    v_dim = v.shape[-1]
    qf = q.permute(0, 2, 1, 3).contiguous().reshape(b * h, t, k_dim)
    kf = k.permute(0, 2, 1, 3).contiguous().reshape(b * h, t, k_dim)
    vf = v.permute(0, 2, 1, 3).contiguous().reshape(b * h, t, v_dim)

    alpha_expanded = gate_alpha.repeat(b).to(device=q.device)
    bias_expanded = gate_bias.repeat(b).to(device=q.device)

    if cu_seqlens is None:
        if initial_state is None and not output_final_state and torch.is_grad_enabled():
            out_flat = _ApproxNetV4DenseFunction.apply(
                qf, kf, vf,
                alpha_expanded, bias_expanded,
                float(z_score_eps),
                int(recompute_chunk_size),
            )
            final_state = None
        else:
            init_internal = _external_state_to_internal(
                state=initial_state, batch=b, n_heads=h, k_dim=k_dim, v_dim=v_dim, device=q.device,
            )
            out_flat, final_internal = _fwd_triton_with_state(
                q=qf, k=kf, v=vf,
                gate_alpha=alpha_expanded, gate_bias=bias_expanded,
                initial_state=init_internal, output_final_state=output_final_state,
                z_score_eps=z_score_eps,
            )
            final_state = (
                _internal_state_to_external(final_internal, b, h, k_dim, v_dim)
                if output_final_state and final_internal is not None
                else None
            )
    else:
        if b != 1:
            raise ValueError("When cu_seqlens is provided, expected flattened varlen input with batch=1.")
        cu = cu_seqlens.tolist()
        n_seq = len(cu) - 1
        if initial_state is not None and initial_state[0].shape[0] != n_seq:
            raise ValueError(
                f"initial_state first dim ({initial_state[0].shape[0]}) must match num sequences ({n_seq}) for varlen mode.",
            )
        parts = []
        final_chunks = [[] for _ in range(8)] if output_final_state else None
        for i_seq in range(len(cu) - 1):
            bos, eos = int(cu[i_seq]), int(cu[i_seq + 1])
            if eos > bos:
                use_autograd_dense = (
                    initial_state is None
                    and not output_final_state
                    and torch.is_grad_enabled()
                )
                init_internal_i = None
                if initial_state is not None:
                    init_internal_i = _external_state_to_internal(
                        state=tuple(x[i_seq : i_seq + 1] for x in initial_state),
                        batch=1, n_heads=h, k_dim=k_dim, v_dim=v_dim, device=q.device,
                    )
                if use_autograd_dense:
                    parts.append(
                        _ApproxNetV4DenseFunction.apply(
                            qf[:, bos:eos, :], kf[:, bos:eos, :], vf[:, bos:eos, :],
                            alpha_expanded, bias_expanded,
                            float(z_score_eps), int(recompute_chunk_size),
                        ),
                    )
                else:
                    out_i, final_i_internal = _fwd_triton_with_state(
                        q=qf[:, bos:eos, :], k=kf[:, bos:eos, :], v=vf[:, bos:eos, :],
                        gate_alpha=alpha_expanded, gate_bias=bias_expanded,
                        initial_state=init_internal_i, output_final_state=output_final_state,
                        z_score_eps=z_score_eps,
                    )
                    parts.append(out_i)
                    if output_final_state:
                        if final_i_internal is None:
                            raise RuntimeError("Expected non-empty final state when output_final_state=True.")
                        final_i = _internal_state_to_external(final_i_internal, 1, h, k_dim, v_dim)
                        for idx in range(8):
                            final_chunks[idx].append(final_i[idx][0])
            elif output_final_state:
                if initial_state is not None:
                    for idx in range(8):
                        final_chunks[idx].append(initial_state[idx][i_seq])
                else:
                    final_chunks[0].append(torch.zeros((h, k_dim, v_dim), device=q.device, dtype=torch.float32))
                    final_chunks[1].append(torch.zeros((h, v_dim), device=q.device, dtype=torch.float32))
                    final_chunks[2].append(torch.zeros((h, k_dim), device=q.device, dtype=torch.float32))
                    final_chunks[3].append(torch.zeros((h, v_dim), device=q.device, dtype=torch.float32))
                    final_chunks[4].append(torch.zeros((h, k_dim), device=q.device, dtype=torch.float32))
                    final_chunks[5].append(torch.zeros((h,), device=q.device, dtype=torch.float32))
                    final_chunks[6].append(torch.zeros((h,), device=q.device, dtype=torch.float32))
                    final_chunks[7].append(torch.zeros((h,), device=q.device, dtype=torch.float32))
        out_flat = torch.cat(parts, dim=1) if parts else torch.empty_like(vf)
        final_state = tuple(torch.stack(chunks, dim=0) for chunks in final_chunks) if output_final_state else None

    out = out_flat.reshape(b, h, t, v_dim).permute(0, 2, 1, 3).contiguous()
    return out, final_state, {}


__all__ = [
    "approxnet_v4_linear_attention_triton",
    "_TRITON_AVAILABLE",
]
