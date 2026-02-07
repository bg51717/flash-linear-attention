from __future__ import annotations

import os
import torch
import triton
import triton.language as tl


_EPS = 1e-6
_USE_TRITON_BWD = os.getenv('LA_PDF_USE_TRITON_BWD', '1') == '1'


def _torch_recompute_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a0: torch.Tensor,
    b0: torch.Tensor,
    s0: torch.Tensor,
    z0: torch.Tensor,
    m0: torch.Tensor,
    q0: torch.Tensor,
) -> torch.Tensor:
    qf, kf, vf = q.float(), k.float(), v.float()
    a = a0.float()
    b = b0.float()
    s = s0.float()
    z = z0.float()
    m = m0.float()
    q_prev = q0.float()

    n, t, _ = q.shape
    o = torch.empty((n, t, v.shape[-1]), device=q.device, dtype=torch.float32)

    for i in range(t):
        q_i = qf[:, i]
        k_i = kf[:, i]
        v_i = vf[:, i]

        d_i = torch.sum(q_i * k_i, dim=-1)
        m_new = torch.maximum(m, d_i)
        alpha = torch.exp(m - m_new)
        w = torch.exp(d_i - m_new)
        dq = q_i - q_prev

        adq = torch.einsum('nvk,nk->nv', a, dq)
        bdq = torch.sum(b * dq, dim=-1)

        s = alpha[:, None] * (s + adq) + w[:, None] * v_i
        z = alpha * (z + bdq) + w
        a = alpha[:, None, None] * a + w[:, None, None] * torch.einsum('nv,nk->nvk', v_i, k_i)
        b = alpha[:, None] * b + w[:, None] * k_i

        m = m_new
        q_prev = q_i
        o[:, i] = s / (z[:, None] + _EPS)

    return o.to(q.dtype)


def _bwd_torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    a0: torch.Tensor,
    b0: torch.Tensor,
    s0: torch.Tensor,
    z0: torch.Tensor,
    m0: torch.Tensor,
    q0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        o_ = _torch_recompute_forward(
            q=q_,
            k=k_,
            v=v_,
            a0=a0,
            b0=b0,
            s0=s0,
            z0=z0,
            m0=m0,
            q0=q0,
        )
        dq, dk, dv = torch.autograd.grad(o_, (q_, k_, v_), do)
    return dq, dk, dv


@triton.jit(do_not_specialize=['T'])
def _first_order_linear_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    m_hist_ptr,
    a_ptr,
    b_ptr,
    s_ptr,
    z_ptr,
    m_ptr,
    qprev_ptr,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    EPS: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)

    offs_k = tl.arange(0, BK)
    m_k = offs_k < K

    p_a_n = a_ptr + n * V * K
    p_b_n = b_ptr + n * K
    p_s_n = s_ptr + n * V
    p_qprev_n = qprev_ptr + n * K

    b_q_prev = tl.load(p_qprev_n + offs_k, mask=m_k, other=0).to(tl.float32)

    for t in range(0, T):
        q_base = (n * T + t) * K
        v_base = (n * T + t) * V

        b_q = tl.load(q_ptr + q_base + offs_k, mask=m_k, other=0).to(tl.float32)
        b_k = tl.load(k_ptr + q_base + offs_k, mask=m_k, other=0).to(tl.float32)
        b_dq = b_q - b_q_prev

        d_t = tl.sum(b_q * b_k, axis=0)
        m_prev = tl.load(m_ptr + n).to(tl.float32)
        m_new = tl.maximum(m_prev, d_t)
        alpha = tl.exp(m_prev - m_new)
        w = tl.exp(d_t - m_new)

        b_state = tl.load(p_b_n + offs_k, mask=m_k, other=0).to(tl.float32)
        bdq = tl.sum(b_state * b_dq, axis=0)

        z_prev = tl.load(z_ptr + n).to(tl.float32)
        z_new = alpha * (z_prev + bdq) + w
        tl.store(z_ptr + n, z_new)

        for v0 in range(0, V, BV):
            offs_v = v0 + tl.arange(0, BV)
            m_v = offs_v < V
            m_vk = m_v[:, None] & m_k[None, :]

            p_a = p_a_n + offs_v[:, None] * K + offs_k[None, :]
            a_prev = tl.load(p_a, mask=m_vk, other=0).to(tl.float32)

            adq = tl.sum(a_prev * b_dq[None, :], axis=1)
            s_prev = tl.load(p_s_n + offs_v, mask=m_v, other=0).to(tl.float32)
            v_t = tl.load(v_ptr + v_base + offs_v, mask=m_v, other=0).to(tl.float32)

            s_new = alpha * (s_prev + adq) + w * v_t
            tl.store(p_s_n + offs_v, s_new, mask=m_v)

            o_t = s_new / (z_new + EPS)
            tl.store(o_ptr + v_base + offs_v, o_t.to(o_ptr.dtype.element_ty), mask=m_v)

            a_new = alpha * a_prev + w * (v_t[:, None] * b_k[None, :])
            tl.store(p_a, a_new.to(a_ptr.dtype.element_ty), mask=m_vk)

        b_new = alpha * b_state + w * b_k
        tl.store(p_b_n + offs_k, b_new.to(b_ptr.dtype.element_ty), mask=m_k)
        tl.store(m_ptr + n, m_new)
        tl.store(m_hist_ptr + n * T + t, m_new)
        tl.store(p_qprev_n + offs_k, b_q.to(qprev_ptr.dtype.element_ty), mask=m_k)
        b_q_prev = b_q


@triton.jit(do_not_specialize=['T'])
def _first_order_linear_attn_recompute_hist_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    a0_ptr,
    b0_ptr,
    s0_ptr,
    z0_ptr,
    m0_ptr,
    q0_ptr,
    a_hist_ptr,
    b_hist_ptr,
    s_hist_ptr,
    z_hist_ptr,
    m_hist_ptr,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)

    offs_k = tl.arange(0, BK)
    m_k = offs_k < K

    p_a0_n = a0_ptr + n * V * K
    p_b0_n = b0_ptr + n * K
    p_s0_n = s0_ptr + n * V
    p_q0_n = q0_ptr + n * K
    p_m0_n = m0_ptr + n
    p_z0_n = z0_ptr + n

    for t in range(0, T):
        q_base = (n * T + t) * K
        v_base = (n * T + t) * V

        b_q = tl.load(q_ptr + q_base + offs_k, mask=m_k, other=0).to(tl.float32)
        b_k = tl.load(k_ptr + q_base + offs_k, mask=m_k, other=0).to(tl.float32)

        if t > 0:
            b_q_prev = tl.load(q_ptr + q_base - K + offs_k, mask=m_k, other=0).to(tl.float32)
            m_prev = tl.load(m_hist_ptr + n * T + t - 1).to(tl.float32)
            z_prev = tl.load(z_hist_ptr + n * T + t - 1).to(tl.float32)
            b_prev = tl.load(b_hist_ptr + (n * T + t - 1) * K + offs_k, mask=m_k, other=0).to(tl.float32)
        else:
            b_q_prev = tl.load(p_q0_n + offs_k, mask=m_k, other=0).to(tl.float32)
            m_prev = tl.load(p_m0_n).to(tl.float32)
            z_prev = tl.load(p_z0_n).to(tl.float32)
            b_prev = tl.load(p_b0_n + offs_k, mask=m_k, other=0).to(tl.float32)
        b_dq_step = b_q - b_q_prev

        d_t = tl.sum(b_q * b_k, axis=0)
        m_cur = tl.maximum(m_prev, d_t)
        alpha = tl.exp(m_prev - m_cur)
        w = tl.exp(d_t - m_cur)

        bdq_prev = tl.sum(b_prev * b_dq_step, axis=0)
        z_cur = alpha * (z_prev + bdq_prev) + w

        for v0 in range(0, V, BV):
            offs_v = v0 + tl.arange(0, BV)
            m_v = offs_v < V
            m_vk = m_v[:, None] & m_k[None, :]

            if t > 0:
                p_a_prev = a_hist_ptr + ((n * T + t - 1) * V + offs_v[:, None]) * K + offs_k[None, :]
                p_s_prev = s_hist_ptr + (n * T + t - 1) * V + offs_v
            else:
                p_a_prev = p_a0_n + offs_v[:, None] * K + offs_k[None, :]
                p_s_prev = p_s0_n + offs_v

            b_v = tl.load(v_ptr + v_base + offs_v, mask=m_v, other=0).to(tl.float32)
            b_s_prev = tl.load(p_s_prev, mask=m_v, other=0).to(tl.float32)
            b_a_prev = tl.load(p_a_prev, mask=m_vk, other=0).to(tl.float32)
            b_adq_prev = tl.sum(b_a_prev * b_dq_step[None, :], axis=1)

            b_s_cur = alpha * (b_s_prev + b_adq_prev) + w * b_v
            b_a_cur = alpha * b_a_prev + w * (b_v[:, None] * b_k[None, :])

            tl.store(s_hist_ptr + (n * T + t) * V + offs_v, b_s_cur.to(s_hist_ptr.dtype.element_ty), mask=m_v)
            tl.store(
                a_hist_ptr + ((n * T + t) * V + offs_v[:, None]) * K + offs_k[None, :],
                b_a_cur.to(a_hist_ptr.dtype.element_ty),
                mask=m_vk,
            )

        b_cur = alpha * b_prev + w * b_k
        tl.store(b_hist_ptr + (n * T + t) * K + offs_k, b_cur.to(b_hist_ptr.dtype.element_ty), mask=m_k)
        tl.store(z_hist_ptr + n * T + t, z_cur.to(z_hist_ptr.dtype.element_ty))
        tl.store(m_hist_ptr + n * T + t, m_cur.to(m_hist_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=['T'])
def _first_order_linear_attn_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    a_hist_ptr,
    b_hist_ptr,
    s_hist_ptr,
    z_hist_ptr,
    m_hist_ptr,
    a0_ptr,
    b0_ptr,
    s0_ptr,
    z0_ptr,
    m0_ptr,
    q0_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    da_ptr,
    db_ptr,
    ds_ptr,
    dz_ptr,
    dm_ptr,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    EPS: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)

    offs_k = tl.arange(0, BK)
    m_k = offs_k < K

    p_a0_n = a0_ptr + n * V * K
    p_b0_n = b0_ptr + n * K
    p_s0_n = s0_ptr + n * V
    p_q0_n = q0_ptr + n * K
    p_m0_n = m0_ptr + n
    p_z0_n = z0_ptr + n

    p_da_n = da_ptr + n * V * K
    p_db_n = db_ptr + n * K
    p_ds_n = ds_ptr + n * V

    for i in range(0, T):
        t = T - 1 - i
        q_base = (n * T + t) * K
        v_base = (n * T + t) * V

        b_q = tl.load(q_ptr + q_base + offs_k, mask=m_k, other=0).to(tl.float32)
        b_k = tl.load(k_ptr + q_base + offs_k, mask=m_k, other=0).to(tl.float32)
        if t > 0:
            b_q_prev = tl.load(q_ptr + q_base - K + offs_k, mask=m_k, other=0).to(tl.float32)
            m_prev = tl.load(m_hist_ptr + n * T + t - 1).to(tl.float32)
            z_prev = tl.load(z_hist_ptr + n * T + t - 1).to(tl.float32)
            b_prev = tl.load(b_hist_ptr + (n * T + t - 1) * K + offs_k, mask=m_k, other=0).to(tl.float32)
        else:
            b_q_prev = tl.load(p_q0_n + offs_k, mask=m_k, other=0).to(tl.float32)
            m_prev = tl.load(p_m0_n).to(tl.float32)
            z_prev = tl.load(p_z0_n).to(tl.float32)
            b_prev = tl.load(p_b0_n + offs_k, mask=m_k, other=0).to(tl.float32)
        b_dq_step = b_q - b_q_prev

        d_t = tl.sum(b_q * b_k, axis=0)
        m_cur = tl.load(m_hist_ptr + n * T + t).to(tl.float32)
        z_cur = tl.load(z_hist_ptr + n * T + t).to(tl.float32)
        inv_z = 1.0 / (z_cur + EPS)

        dz_cur = tl.load(dz_ptr + n).to(tl.float32)
        dm_cur = tl.load(dm_ptr + n).to(tl.float32)
        b_db_cur = tl.load(p_db_n + offs_k, mask=m_k, other=0).to(tl.float32)

        alpha = tl.exp(m_prev - m_cur)
        w = tl.exp(d_t - m_cur)

        g_alpha = tl.zeros([], dtype=tl.float32)
        g_w = tl.zeros([], dtype=tl.float32)
        g_dq_total = tl.zeros([BK], dtype=tl.float32)
        b_dk_t = tl.zeros([BK], dtype=tl.float32)

        for v0 in range(0, V, BV):
            offs_v = v0 + tl.arange(0, BV)
            m_v = offs_v < V
            m_vk = m_v[:, None] & m_k[None, :]

            if t > 0:
                p_a_prev = a_hist_ptr + ((n * T + t - 1) * V + offs_v[:, None]) * K + offs_k[None, :]
                p_s_prev = s_hist_ptr + (n * T + t - 1) * V + offs_v
            else:
                p_a_prev = p_a0_n + offs_v[:, None] * K + offs_k[None, :]
                p_s_prev = p_s0_n + offs_v

            p_da = p_da_n + offs_v[:, None] * K + offs_k[None, :]

            b_v = tl.load(v_ptr + v_base + offs_v, mask=m_v, other=0).to(tl.float32)
            b_do = tl.load(do_ptr + v_base + offs_v, mask=m_v, other=0).to(tl.float32)

            b_s_cur = tl.load(s_hist_ptr + (n * T + t) * V + offs_v, mask=m_v, other=0).to(tl.float32)
            b_s_prev = tl.load(p_s_prev, mask=m_v, other=0).to(tl.float32)

            b_a_prev = tl.load(p_a_prev, mask=m_vk, other=0).to(tl.float32)
            b_da_cur = tl.load(p_da, mask=m_vk, other=0).to(tl.float32)
            b_ds_cur = tl.load(p_ds_n + offs_v, mask=m_v, other=0).to(tl.float32)

            b_ds_cur += b_do * inv_z
            dz_cur += -tl.sum(b_do * b_s_cur, axis=0) * inv_z * inv_z

            b_outer_vk = b_v[:, None] * b_k[None, :]
            b_s_pre = b_s_prev + tl.sum(b_a_prev * b_dq_step[None, :], axis=1)
            b_g_u_s = b_ds_cur * alpha
            b_ds_prev = b_g_u_s

            g_alpha += tl.sum(b_ds_cur * b_s_pre, axis=0)
            g_alpha += tl.sum(tl.sum(b_da_cur * b_a_prev, axis=1), axis=0)

            g_w += tl.sum(b_ds_cur * b_v, axis=0)
            g_w += tl.sum(tl.sum(b_da_cur * b_outer_vk, axis=1), axis=0)

            b_dv_t = w * b_ds_cur
            b_dv_t += w * tl.sum(b_da_cur * b_k[None, :], axis=1)

            g_dq_total += tl.sum(b_a_prev * b_g_u_s[:, None], axis=0)
            b_dk_t += w * tl.sum(b_da_cur * b_v[:, None], axis=0)

            b_da_prev = alpha * b_da_cur + b_g_u_s[:, None] * b_dq_step[None, :]

            tl.store(p_da, b_da_prev.to(da_ptr.dtype.element_ty), mask=m_vk)
            tl.store(dv_ptr + v_base + offs_v, b_dv_t.to(dv_ptr.dtype.element_ty), mask=m_v)
            tl.store(p_ds_n + offs_v, b_ds_prev.to(ds_ptr.dtype.element_ty), mask=m_v)

        z_pre = z_prev + tl.sum(b_prev * b_dq_step, axis=0)
        dot_db_k = tl.sum(b_db_cur * b_k, axis=0)
        g_u_z = dz_cur * alpha
        b_db_prev = alpha * b_db_cur + g_u_z * b_dq_step
        g_dq_total += b_prev * g_u_z
        g_alpha += dz_cur * z_pre + tl.sum(b_db_cur * b_prev, axis=0)
        g_w += dz_cur + dot_db_k
        b_dk_t += w * b_db_cur

        tl.store(dz_ptr + n, g_u_z.to(dz_ptr.dtype.element_ty))

        g_d = g_w * w
        g_m = dm_cur - g_alpha * alpha - g_w * w
        is_new_max = d_t >= m_prev
        g_d = g_d + tl.where(is_new_max, g_m, 0.0)
        g_m_prev = g_alpha * alpha + tl.where(is_new_max, 0.0, g_m)
        tl.store(dm_ptr + n, g_m_prev.to(dm_ptr.dtype.element_ty))

        b_dq_t = g_dq_total + g_d * b_k
        b_dk_t += g_d * b_q

        p_dq_t = dq_ptr + q_base + offs_k
        dq_old = tl.load(p_dq_t, mask=m_k, other=0).to(tl.float64)
        tl.store(p_dq_t, (dq_old + b_dq_t.to(tl.float64)).to(dq_ptr.dtype.element_ty), mask=m_k)

        if t > 0:
            p_dq_prev = dq_ptr + q_base - K + offs_k
            dq_prev_old = tl.load(p_dq_prev, mask=m_k, other=0).to(tl.float64)
            tl.store(p_dq_prev, (dq_prev_old - g_dq_total.to(tl.float64)).to(dq_ptr.dtype.element_ty), mask=m_k)

        tl.store(dk_ptr + q_base + offs_k, b_dk_t.to(dk_ptr.dtype.element_ty), mask=m_k)
        tl.store(p_db_n + offs_k, b_db_prev.to(db_ptr.dtype.element_ty), mask=m_k)


def _fwd_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a0: torch.Tensor,
    b0: torch.Tensor,
    s0: torch.Tensor,
    z0: torch.Tensor,
    m0: torch.Tensor,
    q0: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    n, t, k_dim = q.shape
    v_dim = v.shape[-1]

    a = a0.contiguous().clone()
    b = b0.contiguous().clone()
    s = s0.contiguous().clone()
    z = z0.contiguous().clone()
    m = m0.contiguous().clone()
    q_prev = q0.contiguous().clone()
    m0_saved = m.clone()
    q0_saved = q_prev.clone()

    o = torch.empty((n, t, v_dim), device=q.device, dtype=q.dtype)
    m_hist = torch.empty((n, t), device=q.device, dtype=torch.float32)

    bk = triton.next_power_of_2(k_dim)
    if bk > 512:
        raise ValueError(f"Unsupported K={k_dim}. This kernel supports K <= 512.")
    bv = min(64, triton.next_power_of_2(v_dim))

    _first_order_linear_attn_fwd_kernel[(n,)](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        o_ptr=o,
        m_hist_ptr=m_hist,
        a_ptr=a,
        b_ptr=b,
        s_ptr=s,
        z_ptr=z,
        m_ptr=m,
        qprev_ptr=q_prev,
        T=t,
        K=k_dim,
        V=v_dim,
        BK=bk,
        BV=bv,
        EPS=_EPS,
        num_warps=4,
        num_stages=2,
    )
    return o, a, b, s, z, m, q_prev, m_hist, m0_saved, q0_saved


def _bwd_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    a0: torch.Tensor,
    b0: torch.Tensor,
    s0: torch.Tensor,
    z0: torch.Tensor,
    m0: torch.Tensor,
    q0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    do = do.contiguous()
    a0 = a0.contiguous()
    b0 = b0.contiguous()
    s0 = s0.contiguous()
    z0 = z0.contiguous()
    m0 = m0.contiguous()
    q0 = q0.contiguous()

    n, t, k_dim = q.shape
    v_dim = v.shape[-1]
    a_hist = torch.empty((n, t, v_dim, k_dim), device=q.device, dtype=torch.float32)
    b_hist = torch.empty((n, t, k_dim), device=q.device, dtype=torch.float32)
    s_hist = torch.empty((n, t, v_dim), device=q.device, dtype=torch.float32)
    z_hist = torch.empty((n, t), device=q.device, dtype=torch.float32)
    m_hist = torch.empty((n, t), device=q.device, dtype=torch.float32)

    dq = torch.zeros_like(q, dtype=torch.float64)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv = torch.empty_like(v, dtype=torch.float32)

    da = torch.zeros((n, v_dim, k_dim), device=q.device, dtype=torch.float32)
    db = torch.zeros((n, k_dim), device=q.device, dtype=torch.float32)
    ds = torch.zeros((n, v_dim), device=q.device, dtype=torch.float32)
    dz = torch.zeros((n,), device=q.device, dtype=torch.float32)
    dm = torch.zeros((n,), device=q.device, dtype=torch.float32)

    bk = triton.next_power_of_2(k_dim)
    if bk > 512:
        raise ValueError(f"Unsupported K={k_dim}. This kernel supports K <= 512.")
    bv = min(64, triton.next_power_of_2(v_dim))

    _first_order_linear_attn_recompute_hist_kernel[(n,)](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        a0_ptr=a0,
        b0_ptr=b0,
        s0_ptr=s0,
        z0_ptr=z0,
        m0_ptr=m0,
        q0_ptr=q0,
        a_hist_ptr=a_hist,
        b_hist_ptr=b_hist,
        s_hist_ptr=s_hist,
        z_hist_ptr=z_hist,
        m_hist_ptr=m_hist,
        T=t,
        K=k_dim,
        V=v_dim,
        BK=bk,
        BV=bv,
        num_warps=4,
        num_stages=2,
    )

    _first_order_linear_attn_bwd_kernel[(n,)](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        do_ptr=do,
        a_hist_ptr=a_hist,
        b_hist_ptr=b_hist,
        s_hist_ptr=s_hist,
        z_hist_ptr=z_hist,
        m_hist_ptr=m_hist,
        a0_ptr=a0,
        b0_ptr=b0,
        s0_ptr=s0,
        z0_ptr=z0,
        m0_ptr=m0,
        q0_ptr=q0,
        dq_ptr=dq,
        dk_ptr=dk,
        dv_ptr=dv,
        da_ptr=da,
        db_ptr=db,
        ds_ptr=ds,
        dz_ptr=dz,
        dm_ptr=dm,
        T=t,
        K=k_dim,
        V=v_dim,
        BK=bk,
        BV=bv,
        EPS=_EPS,
        num_warps=4,
        num_stages=2,
    )
    return dq, dk, dv


class _FirstOrderLinearAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a0: torch.Tensor,
        b0: torch.Tensor,
        s0: torch.Tensor,
        z0: torch.Tensor,
        m0: torch.Tensor,
        q0: torch.Tensor,
    ):
        o, a, b, s, z, m, q_prev, m_hist, m0_saved, q0_saved = _fwd_triton(
            q=q,
            k=k,
            v=v,
            a0=a0,
            b0=b0,
            s0=s0,
            z0=z0,
            m0=m0,
            q0=q0,
        )
        ctx.save_for_backward(q, k, v, a0, b0, s0, z0, m0_saved, q0_saved)
        ctx.mark_non_differentiable(a, b, s, z, m, q_prev)
        return o, a, b, s, z, m, q_prev

    @staticmethod
    def backward(ctx, do, da, db, ds, dz, dm, dq_prev):
        q, k, v, a0, b0, s0, z0, m0, q0 = ctx.saved_tensors
        if _USE_TRITON_BWD:
            dq, dk, dv = _bwd_triton(
                q=q,
                k=k,
                v=v,
                do=do,
                a0=a0,
                b0=b0,
                s0=s0,
                z0=z0,
                m0=m0,
                q0=q0,
            )
        else:
            dq, dk, dv = _bwd_torch_reference(
                q=q,
                k=k,
                v=v,
                do=do,
                a0=a0,
                b0=b0,
                s0=s0,
                z0=z0,
                m0=m0,
                q0=q0,
            )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None, None


def triton_first_order_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a0: torch.Tensor,
    b0: torch.Tensor,
    s0: torch.Tensor,
    z0: torch.Tensor,
    m0: torch.Tensor,
    q0: torch.Tensor,
    output_final_state: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    o, a, b, s, z, m, q_prev = _FirstOrderLinearAttentionFunction.apply(
        q,
        k,
        v,
        a0,
        b0,
        s0,
        z0,
        m0,
        q0,
    )
    if output_final_state:
        return o, (a, b, s, z, m, q_prev)
    return o, None


__all__ = ['triton_first_order_linear_attention']
