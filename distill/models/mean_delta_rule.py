# Copyright (c) 2026

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.utils import input_guard

from .mean_delta_rule_naive import mean_delta_rule_recurrence


@triton.heuristics({
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_mean_delta_rule_fwd_kernel(
    q,
    k,
    v,
    o,
    h0,
    mean_v0,
    mean_k0,
    count0,
    ht,
    mean_vt,
    mean_kt,
    countt,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        Tcur = eos - bos
    else:
        bos = i_n * T
        Tcur = T

    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_q = q + (bos * H + i_h) * K + offs_k
    p_k = k + (bos * H + i_h) * K + offs_k
    p_v = v + (bos * H + i_h) * V + offs_v
    p_o = o + (bos * H + i_h) * V + offs_v

    # Internal matrix state layout: [V, K], while external cache is [K, V].
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_mean_v = tl.zeros([BV], dtype=tl.float32)
    b_mean_k = tl.zeros([BK], dtype=tl.float32)
    b_count = tl.zeros([], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + offs_k[None, :] * V + offs_v[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        p_mean_v0 = mean_v0 + i_nh * V + offs_v
        b_mean_v += tl.load(p_mean_v0, mask=mask_v, other=0).to(tl.float32)

        p_mean_k0 = mean_k0 + i_nh * K + offs_k
        b_mean_k += tl.load(p_mean_k0, mask=mask_k, other=0).to(tl.float32)

        p_count0 = count0 + i_nh
        b_count += tl.load(p_count0).to(tl.float32)

    for _ in range(0, Tcur):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale

        # S_t = S_{t-1} + (v_t - mean_v_{t-1}) k_t^T + v_t (k_t^T - mean_k_{t-1}^T)
        b_x = b_v - b_mean_v
        b_y = b_k - b_mean_k
        b_h += b_x[:, None] * b_k[None, :] + b_v[:, None] * b_y[None, :]

        b_o = tl.sum(b_h * b_q[None, :], axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Running means update for next step.
        b_count_new = b_count + 1.0
        b_inv = 1.0 / b_count_new
        b_mean_v += (b_v - b_mean_v) * b_inv
        b_mean_k += (b_k - b_mean_k) * b_inv
        b_count = b_count_new

        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_o += H * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + offs_k[None, :] * V + offs_v[:, None]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_mean_vt = mean_vt + i_nh * V + offs_v
        tl.store(p_mean_vt, b_mean_v.to(p_mean_vt.dtype.element_ty), mask=mask_v)

        p_mean_kt = mean_kt + i_nh * K + offs_k
        tl.store(p_mean_kt, b_mean_k.to(p_mean_kt.dtype.element_ty), mask=mask_k)

        p_countt = countt + i_nh
        tl.store(p_countt, b_count.to(p_countt.dtype.element_ty))


@triton.jit(do_not_specialize=["T", "t_start", "L"])
def fused_recurrent_mean_delta_rule_chunk_state_kernel(
    k,
    v,
    state_in,
    mean_v_in,
    mean_k_in,
    count_in,
    state_out,
    mean_v_out,
    mean_k_out,
    count_out,
    t_start,
    L,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_bh = tl.program_id(0)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_state_in = state_in + i_bh * V * K + offs_v[:, None] * K + offs_k[None, :]
    b_h = tl.load(p_state_in, mask=mask_h, other=0).to(tl.float32)
    p_mv_in = mean_v_in + i_bh * V + offs_v
    b_mv = tl.load(p_mv_in, mask=mask_v, other=0).to(tl.float32)
    p_mk_in = mean_k_in + i_bh * K + offs_k
    b_mk = tl.load(p_mk_in, mask=mask_k, other=0).to(tl.float32)
    b_c = tl.load(count_in + i_bh).to(tl.float32)

    p_k = k + i_bh * T * K + t_start * K + offs_k
    p_v = v + i_bh * T * V + t_start * V + offs_v
    for _ in range(0, L):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        b_x = b_v - b_mv
        b_y = b_k - b_mk
        b_h += b_x[:, None] * b_k[None, :] + b_v[:, None] * b_y[None, :]

        b_c_new = b_c + 1.0
        b_inv = 1.0 / b_c_new
        b_mv += (b_v - b_mv) * b_inv
        b_mk += (b_k - b_mk) * b_inv
        b_c = b_c_new

        p_k += K
        p_v += V

    p_state_out = state_out + i_bh * V * K + offs_v[:, None] * K + offs_k[None, :]
    tl.store(p_state_out, b_h.to(p_state_out.dtype.element_ty), mask=mask_h)
    p_mv_out = mean_v_out + i_bh * V + offs_v
    tl.store(p_mv_out, b_mv.to(mean_v_out.dtype.element_ty), mask=mask_v)
    p_mk_out = mean_k_out + i_bh * K + offs_k
    tl.store(p_mk_out, b_mk.to(mean_k_out.dtype.element_ty), mask=mask_k)
    tl.store(count_out + i_bh, b_c.to(count_out.dtype.element_ty))


@triton.jit(do_not_specialize=["T", "t_start", "L"])
def fused_recurrent_mean_delta_rule_chunk_states_kernel(
    k,
    v,
    state_in,
    mean_v_in,
    mean_k_in,
    count_in,
    local_states,
    local_mean_v,
    local_mean_k,
    local_count,
    t_start,
    L,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNK_MAX: tl.constexpr,
):
    i_bh = tl.program_id(0)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_state_in = state_in + i_bh * V * K + offs_v[:, None] * K + offs_k[None, :]
    b_h = tl.load(p_state_in, mask=mask_h, other=0).to(tl.float32)
    p_mv_in = mean_v_in + i_bh * V + offs_v
    b_mv = tl.load(p_mv_in, mask=mask_v, other=0).to(tl.float32)
    p_mk_in = mean_k_in + i_bh * K + offs_k
    b_mk = tl.load(p_mk_in, mask=mask_k, other=0).to(tl.float32)
    b_c = tl.load(count_in + i_bh).to(tl.float32)

    p_ls_base = local_states + i_bh * (CHUNK_MAX + 1) * V * K + offs_v[:, None] * K + offs_k[None, :]
    p_lmv_base = local_mean_v + i_bh * (CHUNK_MAX + 1) * V + offs_v
    p_lmk_base = local_mean_k + i_bh * (CHUNK_MAX + 1) * K + offs_k
    p_lc_base = local_count + i_bh * (CHUNK_MAX + 1)

    tl.store(p_ls_base, b_h.to(local_states.dtype.element_ty), mask=mask_h)
    tl.store(p_lmv_base, b_mv.to(local_mean_v.dtype.element_ty), mask=mask_v)
    tl.store(p_lmk_base, b_mk.to(local_mean_k.dtype.element_ty), mask=mask_k)
    tl.store(p_lc_base, b_c.to(local_count.dtype.element_ty))

    p_k = k + i_bh * T * K + t_start * K + offs_k
    p_v = v + i_bh * T * V + t_start * V + offs_v
    p_ls = p_ls_base + V * K
    p_lmv = p_lmv_base + V
    p_lmk = p_lmk_base + K
    p_lc = p_lc_base + 1

    for _ in range(0, L):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        b_x = b_v - b_mv
        b_y = b_k - b_mk
        b_h += b_x[:, None] * b_k[None, :] + b_v[:, None] * b_y[None, :]

        b_c_new = b_c + 1.0
        b_inv = 1.0 / b_c_new
        b_mv += (b_v - b_mv) * b_inv
        b_mk += (b_k - b_mk) * b_inv
        b_c = b_c_new

        tl.store(p_ls, b_h.to(local_states.dtype.element_ty), mask=mask_h)
        tl.store(p_lmv, b_mv.to(local_mean_v.dtype.element_ty), mask=mask_v)
        tl.store(p_lmk, b_mk.to(local_mean_k.dtype.element_ty), mask=mask_k)
        tl.store(p_lc, b_c.to(local_count.dtype.element_ty))

        p_ls += V * K
        p_lmv += V
        p_lmk += K
        p_lc += 1
        p_k += K
        p_v += V


@triton.jit(do_not_specialize=["T", "t_start", "L"])
def fused_recurrent_mean_delta_rule_chunk_bwd_kernel(
    q,
    k,
    v,
    do,
    local_states,
    local_mean_v,
    local_mean_k,
    local_count,
    dS_next,
    dmv_next,
    dmk_next,
    dq,
    dk,
    dv,
    dS_prev,
    dmv_prev,
    dmk_prev,
    scale,
    t_start,
    L,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNK_MAX: tl.constexpr,
):
    i_bh = tl.program_id(0)
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_dS = dS_next + i_bh * V * K + offs_v[:, None] * K + offs_k[None, :]
    b_dS = tl.load(p_dS, mask=mask_h, other=0).to(tl.float32)
    b_dmv = tl.load(dmv_next + i_bh * V + offs_v, mask=mask_v, other=0).to(tl.float32)
    b_dmk = tl.load(dmk_next + i_bh * K + offs_k, mask=mask_k, other=0).to(tl.float32)

    p_q = q + i_bh * T * K + (t_start + L - 1) * K + offs_k
    p_k = k + i_bh * T * K + (t_start + L - 1) * K + offs_k
    p_v = v + i_bh * T * V + (t_start + L - 1) * V + offs_v
    p_do = do + i_bh * T * V + (t_start + L - 1) * V + offs_v
    p_dq = dq + i_bh * T * K + (t_start + L - 1) * K + offs_k
    p_dk = dk + i_bh * T * K + (t_start + L - 1) * K + offs_k
    p_dv = dv + i_bh * T * V + (t_start + L - 1) * V + offs_v

    p_ls_base = local_states + i_bh * (CHUNK_MAX + 1) * V * K + offs_v[:, None] * K + offs_k[None, :]
    p_lmv_base = local_mean_v + i_bh * (CHUNK_MAX + 1) * V + offs_v
    p_lmk_base = local_mean_k + i_bh * (CHUNK_MAX + 1) * K + offs_k
    p_lc_base = local_count + i_bh * (CHUNK_MAX + 1)

    p_S_prev = p_ls_base + (L - 1) * V * K
    p_S_t = p_ls_base + L * V * K
    p_mv_prev = p_lmv_base + (L - 1) * V
    p_mk_prev = p_lmk_base + (L - 1) * K
    p_c_prev = p_lc_base + (L - 1)

    for _ in range(0, L):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)

        b_S_t = tl.load(p_S_t, mask=mask_h, other=0).to(tl.float32)
        b_mv_prev = tl.load(p_mv_prev, mask=mask_v, other=0).to(tl.float32)
        b_mk_prev = tl.load(p_mk_prev, mask=mask_k, other=0).to(tl.float32)
        b_c_prev = tl.load(p_c_prev).to(tl.float32)

        b_dS += b_do[:, None] * (b_q * scale)[None, :]
        b_dq = tl.sum(b_S_t * b_do[:, None], axis=0) * scale

        b_c_t = b_c_prev + 1.0
        b_inv = 1.0 / b_c_t
        b_dv_mean = b_dmv * b_inv
        b_dmv_prev = b_dmv * (1.0 - b_inv)
        b_dk_mean = b_dmk * b_inv
        b_dmk_prev = b_dmk * (1.0 - b_inv)

        b_x = b_v - b_mv_prev
        b_y = b_k - b_mk_prev

        b_dX = tl.sum(b_dS * b_k[None, :], axis=1)
        b_dY = tl.sum(b_dS * b_v[:, None], axis=0)

        b_dk_s = tl.sum(b_dS * b_x[:, None], axis=0) + b_dY
        b_dv_s = tl.sum(b_dS * b_y[None, :], axis=1) + b_dX

        b_dk = b_dk_s + b_dk_mean
        b_dv = b_dv_s + b_dv_mean

        b_dmv = b_dmv_prev - b_dX
        b_dmk = b_dmk_prev - b_dY

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)

        p_q -= K
        p_k -= K
        p_v -= V
        p_do -= V
        p_dq -= K
        p_dk -= K
        p_dv -= V
        p_S_prev -= V * K
        p_S_t -= V * K
        p_mv_prev -= V
        p_mk_prev -= K
        p_c_prev -= 1

    p_dS_prev = dS_prev + i_bh * V * K + offs_v[:, None] * K + offs_k[None, :]
    tl.store(p_dS_prev, b_dS.to(p_dS_prev.dtype.element_ty), mask=mask_h)
    tl.store(dmv_prev + i_bh * V + offs_v, b_dmv.to(dmv_prev.dtype.element_ty), mask=mask_v)
    tl.store(dmk_prev + i_bh * K + offs_k, b_dmk.to(dmk_prev.dtype.element_ty), mask=mask_k)


def _mean_forward_step(
    state: torch.Tensor,
    mean_v: torch.Tensor,
    mean_k: torch.Tensor,
    count: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_t = v_t - mean_v
    y_t = k_t - mean_k
    state = state + x_t.unsqueeze(-1) * k_t.unsqueeze(1) + v_t.unsqueeze(-1) * y_t.unsqueeze(1)
    count_new = count + 1.0
    inv = count_new.reciprocal().unsqueeze(-1)
    mean_v = mean_v + (v_t - mean_v) * inv
    mean_k = mean_k + (k_t - mean_k) * inv
    return state, mean_v, mean_k, count_new


def _mean_backward_segment_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    initial_state_vk: torch.Tensor,
    initial_mean_v: torch.Tensor,
    initial_mean_k: torch.Tensor,
    initial_count: torch.Tensor,
    dht_vk: torch.Tensor,
    dmean_v_t: torch.Tensor,
    dmean_k_t: torch.Tensor,
    recompute_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # q/k/v/do: [B, T, H, ...]
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH = B * H

    qf = q.float().permute(0, 2, 1, 3).reshape(BH, T, K).contiguous()
    kf = k.float().permute(0, 2, 1, 3).reshape(BH, T, K).contiguous()
    vf = v.float().permute(0, 2, 1, 3).reshape(BH, T, V).contiguous()
    dof = do.float().permute(0, 2, 1, 3).reshape(BH, T, V).contiguous()

    h0 = initial_state_vk.float().reshape(BH, V, K).contiguous()
    mv0 = initial_mean_v.float().reshape(BH, V).contiguous()
    mk0 = initial_mean_k.float().reshape(BH, K).contiguous()
    c0 = initial_count.float().reshape(BH).contiguous()

    dht = dht_vk.float().reshape(BH, V, K).contiguous()
    dmv_t = dmean_v_t.float().reshape(BH, V).contiguous()
    dmk_t = dmean_k_t.float().reshape(BH, K).contiguous()

    dq = torch.empty(BH, T, K, device=q.device, dtype=torch.float32)
    dk = torch.empty(BH, T, K, device=q.device, dtype=torch.float32)
    dv = torch.empty(BH, T, V, device=q.device, dtype=torch.float32)

    chunk = max(1, int(recompute_chunk_size))
    n_chunks = math.ceil(T / chunk)

    checkpoints_h = torch.empty(n_chunks + 1, BH, V, K, device=q.device, dtype=torch.float32)
    checkpoints_mv = torch.empty(n_chunks + 1, BH, V, device=q.device, dtype=torch.float32)
    checkpoints_mk = torch.empty(n_chunks + 1, BH, K, device=q.device, dtype=torch.float32)
    checkpoints_c = torch.empty(n_chunks + 1, BH, device=q.device, dtype=torch.float32)

    checkpoints_h[0].copy_(h0)
    checkpoints_mv[0].copy_(mv0)
    checkpoints_mk[0].copy_(mk0)
    checkpoints_c[0].copy_(c0)

    state = h0
    mean_v = mv0
    mean_k = mk0
    count = c0

    for c in range(n_chunks):
        s = c * chunk
        e = min(T, s + chunk)
        for t in range(s, e):
            state, mean_v, mean_k, count = _mean_forward_step(
                state=state,
                mean_v=mean_v,
                mean_k=mean_k,
                count=count,
                k_t=kf[:, t, :],
                v_t=vf[:, t, :],
            )
        checkpoints_h[c + 1].copy_(state)
        checkpoints_mv[c + 1].copy_(mean_v)
        checkpoints_mk[c + 1].copy_(mean_k)
        checkpoints_c[c + 1].copy_(count)

    dS = dht
    dmv = dmv_t
    dmk = dmk_t

    for c in range(n_chunks - 1, -1, -1):
        s = c * chunk
        e = min(T, s + chunk)
        l = e - s

        local_h = torch.empty(l + 1, BH, V, K, device=q.device, dtype=torch.float32)
        local_mv = torch.empty(l + 1, BH, V, device=q.device, dtype=torch.float32)
        local_mk = torch.empty(l + 1, BH, K, device=q.device, dtype=torch.float32)
        local_c = torch.empty(l + 1, BH, device=q.device, dtype=torch.float32)

        local_h[0].copy_(checkpoints_h[c])
        local_mv[0].copy_(checkpoints_mv[c])
        local_mk[0].copy_(checkpoints_mk[c])
        local_c[0].copy_(checkpoints_c[c])

        state = local_h[0]
        mean_v = local_mv[0]
        mean_k = local_mk[0]
        count = local_c[0]
        for i in range(l):
            t = s + i
            state, mean_v, mean_k, count = _mean_forward_step(
                state=state,
                mean_v=mean_v,
                mean_k=mean_k,
                count=count,
                k_t=kf[:, t, :],
                v_t=vf[:, t, :],
            )
            local_h[i + 1].copy_(state)
            local_mv[i + 1].copy_(mean_v)
            local_mk[i + 1].copy_(mean_k)
            local_c[i + 1].copy_(count)

        for i in range(l - 1, -1, -1):
            t = s + i
            S_prev = local_h[i]
            S_t = local_h[i + 1]
            mv_prev = local_mv[i]
            mk_prev = local_mk[i]
            c_prev = local_c[i]

            q_t = qf[:, t, :]
            k_t = kf[:, t, :]
            v_t = vf[:, t, :]
            do_t = dof[:, t, :]

            dS = dS + do_t.unsqueeze(-1) * (q_t * scale).unsqueeze(1)
            dq[:, t, :] = torch.bmm(S_t.transpose(1, 2), do_t.unsqueeze(-1)).squeeze(-1) * scale

            c_t = c_prev + 1.0
            beta = c_t.reciprocal().unsqueeze(-1)
            dv_mean = dmv * beta
            dmv_prev = dmv * (1.0 - beta)
            dk_mean = dmk * beta
            dmk_prev = dmk * (1.0 - beta)

            x_t = v_t - mv_prev
            y_t = k_t - mk_prev

            dX = torch.bmm(dS, k_t.unsqueeze(-1)).squeeze(-1)
            dY = torch.bmm(dS.transpose(1, 2), v_t.unsqueeze(-1)).squeeze(-1)

            dk_s = torch.bmm(dS.transpose(1, 2), x_t.unsqueeze(-1)).squeeze(-1) + dY
            dv_s = torch.bmm(dS, y_t.unsqueeze(-1)).squeeze(-1) + dX

            dk[:, t, :] = dk_s + dk_mean
            dv[:, t, :] = dv_s + dv_mean

            dmv = dmv_prev - dX
            dmk = dmk_prev - dY
            # dS already equals dS_prev due identity path through S_t = S_{t-1} + ...
            _ = S_prev

    dh0 = dS.reshape(B, H, V, K).contiguous()
    dmean_v0 = dmv.reshape(B, H, V).contiguous()
    dmean_k0 = dmk.reshape(B, H, K).contiguous()
    dq = dq.reshape(B, H, T, K).permute(0, 2, 1, 3).contiguous()
    dk = dk.reshape(B, H, T, K).permute(0, 2, 1, 3).contiguous()
    dv = dv.reshape(B, H, T, V).permute(0, 2, 1, 3).contiguous()
    return dq, dk, dv, dh0, dmean_v0, dmean_k0


def _mean_backward_recompute_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    dmean_v_t: torch.Tensor | None,
    dmean_k_t: torch.Tensor | None,
    scale: float,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    cu_seqlens: torch.LongTensor | None,
    recompute_chunk_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if cu_seqlens is None:
        B, _, H, K = q.shape
        V = v.shape[-1]
        if initial_state is None:
            h0 = q.new_zeros(B, H, K, V, dtype=torch.float32)
            mv0 = q.new_zeros(B, H, V, dtype=torch.float32)
            mk0 = q.new_zeros(B, H, K, dtype=torch.float32)
            c0 = q.new_zeros(B, H, dtype=torch.float32)
        else:
            h0, mv0, mk0, c0 = initial_state
            h0 = h0.float()
            mv0 = mv0.float()
            mk0 = mk0.float()
            c0 = c0.float()

        h0_vk = h0.transpose(-2, -1).contiguous()

        if dht is None:
            dht_vk = q.new_zeros(B, H, V, K, dtype=torch.float32)
        else:
            dht_vk = dht.float().transpose(-2, -1).contiguous()
        if dmean_v_t is None:
            dmv_t = q.new_zeros(B, H, V, dtype=torch.float32)
        else:
            dmv_t = dmean_v_t.float()
        if dmean_k_t is None:
            dmk_t = q.new_zeros(B, H, K, dtype=torch.float32)
        else:
            dmk_t = dmean_k_t.float()

        dq, dk, dv, dh0_vk, dmv0, dmk0 = _mean_backward_segment_recompute(
            q=q,
            k=k,
            v=v,
            do=do,
            scale=scale,
            initial_state_vk=h0_vk,
            initial_mean_v=mv0,
            initial_mean_k=mk0,
            initial_count=c0,
            dht_vk=dht_vk,
            dmean_v_t=dmv_t,
            dmean_k_t=dmk_t,
            recompute_chunk_size=recompute_chunk_size,
        )
        if initial_state is None:
            dh0 = None
            dmv0_out = None
            dmk0_out = None
        else:
            dh0 = dh0_vk.transpose(-2, -1).contiguous()
            dmv0_out = dmv0
            dmk0_out = dmk0
        return dq, dk, dv, dh0, dmv0_out, dmk0_out

    if q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
        )

    cu = cu_seqlens.to(torch.long)
    n_seq = cu.numel() - 1
    H = q.shape[2]
    K = q.shape[3]
    V = v.shape[3]

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    dh0 = None
    dmv0 = None
    dmk0 = None
    if initial_state is not None:
        h0, mv0, mk0, _ = initial_state
        dh0 = torch.zeros_like(h0, dtype=torch.float32)
        dmv0 = torch.zeros_like(mv0, dtype=torch.float32)
        dmk0 = torch.zeros_like(mk0, dtype=torch.float32)

    for i in range(n_seq):
        bos = int(cu[i].item())
        eos = int(cu[i + 1].item())
        h0_i = None if initial_state is None else tuple(x[i:i + 1] for x in initial_state)
        dht_i = None if dht is None else dht[i:i + 1]
        dmv_i = None if dmean_v_t is None else dmean_v_t[i:i + 1]
        dmk_i = None if dmean_k_t is None else dmean_k_t[i:i + 1]

        dq_i, dk_i, dv_i, dh0_i, dmv0_i, dmk0_i = _mean_backward_recompute_torch(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            do=do[:, bos:eos],
            dht=dht_i,
            dmean_v_t=dmv_i,
            dmean_k_t=dmk_i,
            scale=scale,
            initial_state=h0_i,
            cu_seqlens=None,
            recompute_chunk_size=recompute_chunk_size,
        )

        dq[:, bos:eos] = dq_i
        dk[:, bos:eos] = dk_i
        dv[:, bos:eos] = dv_i
        if dh0 is not None and dh0_i is not None and dmv0 is not None and dmv0_i is not None and dmk0 is not None and dmk0_i is not None:
            dh0[i] = dh0_i[0]
            dmv0[i] = dmv0_i[0]
            dmk0[i] = dmk0_i[0]

    return dq, dk, dv, dh0, dmv0, dmk0


def _mean_backward_recompute_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    dmean_v_t: torch.Tensor | None,
    dmean_k_t: torch.Tensor | None,
    scale: float,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    recompute_chunk_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH = B * H
    chunk = max(1, int(recompute_chunk_size))
    n_chunks = math.ceil(T / chunk)

    qf = q.float().permute(0, 2, 1, 3).reshape(BH, T, K).contiguous()
    kf = k.float().permute(0, 2, 1, 3).reshape(BH, T, K).contiguous()
    vf = v.float().permute(0, 2, 1, 3).reshape(BH, T, V).contiguous()
    dof = do.float().permute(0, 2, 1, 3).reshape(BH, T, V).contiguous()

    if initial_state is None:
        h0 = q.new_zeros(B, H, K, V, dtype=torch.float32)
        mv0 = q.new_zeros(B, H, V, dtype=torch.float32)
        mk0 = q.new_zeros(B, H, K, dtype=torch.float32)
        c0 = q.new_zeros(B, H, dtype=torch.float32)
    else:
        h0, mv0, mk0, c0 = initial_state
        h0 = h0.float()
        mv0 = mv0.float()
        mk0 = mk0.float()
        c0 = c0.float()

    h0_vk = h0.transpose(-2, -1).reshape(BH, V, K).contiguous()
    mv0f = mv0.reshape(BH, V).contiguous()
    mk0f = mk0.reshape(BH, K).contiguous()
    c0f = c0.reshape(BH).contiguous()

    if dht is None:
        dS_next = q.new_zeros(BH, V, K, dtype=torch.float32)
    else:
        dS_next = dht.float().transpose(-2, -1).reshape(BH, V, K).contiguous()
    if dmean_v_t is None:
        dmv_next = q.new_zeros(BH, V, dtype=torch.float32)
    else:
        dmv_next = dmean_v_t.float().reshape(BH, V).contiguous()
    if dmean_k_t is None:
        dmk_next = q.new_zeros(BH, K, dtype=torch.float32)
    else:
        dmk_next = dmean_k_t.float().reshape(BH, K).contiguous()

    dS_prev = torch.empty_like(dS_next)
    dmv_prev = torch.empty_like(dmv_next)
    dmk_prev = torch.empty_like(dmk_next)

    dqf = torch.empty_like(qf)
    dkf = torch.empty_like(kf)
    dvf = torch.empty_like(vf)

    checkpoints_h = torch.empty(n_chunks + 1, BH, V, K, device=q.device, dtype=torch.float32)
    checkpoints_mv = torch.empty(n_chunks + 1, BH, V, device=q.device, dtype=torch.float32)
    checkpoints_mk = torch.empty(n_chunks + 1, BH, K, device=q.device, dtype=torch.float32)
    checkpoints_c = torch.empty(n_chunks + 1, BH, device=q.device, dtype=torch.float32)
    checkpoints_h[0].copy_(h0_vk)
    checkpoints_mv[0].copy_(mv0f)
    checkpoints_mk[0].copy_(mk0f)
    checkpoints_c[0].copy_(c0f)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    num_warps = 4 if max(BK, BV) > 64 else 2

    for c in range(n_chunks):
        s = c * chunk
        l = min(chunk, T - s)
        fused_recurrent_mean_delta_rule_chunk_state_kernel[(BH,)](
            kf,
            vf,
            checkpoints_h[c],
            checkpoints_mv[c],
            checkpoints_mk[c],
            checkpoints_c[c],
            checkpoints_h[c + 1],
            checkpoints_mv[c + 1],
            checkpoints_mk[c + 1],
            checkpoints_c[c + 1],
            s,
            l,
            T=T,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            num_warps=num_warps,
            num_stages=1,
        )

    local_states = torch.empty(BH, chunk + 1, V, K, device=q.device, dtype=torch.float32)
    local_mv = torch.empty(BH, chunk + 1, V, device=q.device, dtype=torch.float32)
    local_mk = torch.empty(BH, chunk + 1, K, device=q.device, dtype=torch.float32)
    local_c = torch.empty(BH, chunk + 1, device=q.device, dtype=torch.float32)

    for c in range(n_chunks - 1, -1, -1):
        s = c * chunk
        l = min(chunk, T - s)

        fused_recurrent_mean_delta_rule_chunk_states_kernel[(BH,)](
            kf,
            vf,
            checkpoints_h[c],
            checkpoints_mv[c],
            checkpoints_mk[c],
            checkpoints_c[c],
            local_states,
            local_mv,
            local_mk,
            local_c,
            s,
            l,
            T=T,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            CHUNK_MAX=chunk,
            num_warps=num_warps,
            num_stages=1,
        )

        fused_recurrent_mean_delta_rule_chunk_bwd_kernel[(BH,)](
            qf,
            kf,
            vf,
            dof,
            local_states,
            local_mv,
            local_mk,
            local_c,
            dS_next,
            dmv_next,
            dmk_next,
            dqf,
            dkf,
            dvf,
            dS_prev,
            dmv_prev,
            dmk_prev,
            scale,
            s,
            l,
            T=T,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            CHUNK_MAX=chunk,
            num_warps=num_warps,
            num_stages=1,
        )
        dS_next, dS_prev = dS_prev, dS_next
        dmv_next, dmv_prev = dmv_prev, dmv_next
        dmk_next, dmk_prev = dmk_prev, dmk_next

    dh0 = dS_next.reshape(B, H, V, K).transpose(-2, -1).contiguous()
    dmv0 = dmv_next.reshape(B, H, V).contiguous()
    dmk0 = dmk_next.reshape(B, H, K).contiguous()
    dq = dqf.reshape(B, H, T, K).permute(0, 2, 1, 3).contiguous()
    dk = dkf.reshape(B, H, T, K).permute(0, 2, 1, 3).contiguous()
    dv = dvf.reshape(B, H, T, V).permute(0, 2, 1, 3).contiguous()
    return dq, dk, dv, dh0, dmv0, dmk0


def _mean_backward_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    dmean_v_t: torch.Tensor | None,
    dmean_k_t: torch.Tensor | None,
    scale: float,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    cu_seqlens: torch.LongTensor | None,
    recompute_chunk_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    use_triton = (
        cu_seqlens is None
        and q.is_cuda
        and max(q.shape[-1], v.shape[-1]) <= 128
    )
    if use_triton:
        dq, dk, dv, dh0, dmv0, dmk0 = _mean_backward_recompute_triton(
            q=q,
            k=k,
            v=v,
            do=do,
            dht=dht,
            dmean_v_t=dmean_v_t,
            dmean_k_t=dmean_k_t,
            scale=scale,
            initial_state=initial_state,
            recompute_chunk_size=recompute_chunk_size,
        )
        if initial_state is None:
            return dq, dk, dv, None, None, None
        return dq, dk, dv, dh0, dmv0, dmk0

    return _mean_backward_recompute_torch(
        q=q,
        k=k,
        v=v,
        do=do,
        dht=dht,
        dmean_v_t=dmean_v_t,
        dmean_k_t=dmean_k_t,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        recompute_chunk_size=recompute_chunk_size,
    )


def fused_recurrent_mean_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    if BK > 128 or BV > 128:
        raise RuntimeError(
            f"Mean Delta Triton fused recurrent currently supports head dims <= 128, got K={K}, V={V}."
        )

    o = q.new_empty(B, T, H, V)

    final_kv = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    final_mean_v = q.new_empty(N, H, V, dtype=torch.float32) if output_final_state else None
    final_mean_k = q.new_empty(N, H, K, dtype=torch.float32) if output_final_state else None
    final_count = q.new_empty(N, H, dtype=torch.float32) if output_final_state else None

    h0 = mean_v0 = mean_k0 = count0 = None
    if initial_state is not None:
        h0, mean_v0, mean_k0, count0 = initial_state

    num_warps = 4 if max(BK, BV) > 64 else 2
    fused_recurrent_mean_delta_rule_fwd_kernel[(N * H,)](
        q,
        k,
        v,
        o,
        h0,
        mean_v0,
        mean_k0,
        count0,
        final_kv,
        final_mean_v,
        final_mean_k,
        final_count,
        cu_seqlens,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, final_kv, final_mean_v, final_mean_k, final_count


class FusedRecurrentMeanDeltaFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        initial_h0: torch.Tensor | None,
        initial_mean_v0: torch.Tensor | None,
        initial_mean_k0: torch.Tensor | None,
        initial_count0: torch.Tensor | None,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        recompute_chunk_size: int = 128,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        init_tuple = None
        if initial_h0 is not None:
            init_tuple = (initial_h0, initial_mean_v0, initial_mean_k0, initial_count0)

        o, final_kv, final_mean_v, final_mean_k, final_count = fused_recurrent_mean_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=init_tuple,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            initial_h0 if initial_h0 is not None else torch.tensor([], device=q.device),
            initial_mean_v0 if initial_mean_v0 is not None else torch.tensor([], device=q.device),
            initial_mean_k0 if initial_mean_k0 is not None else torch.tensor([], device=q.device),
            initial_count0 if initial_count0 is not None else torch.tensor([], device=q.device),
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cu_seqlens = cu_seqlens
        ctx.initial_state_was_none = initial_h0 is None
        ctx.recompute_chunk_size = int(recompute_chunk_size)
        return o, final_kv, final_mean_v, final_mean_k, final_count

    @staticmethod
    @input_guard
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor | None,
        dmean_v_t: torch.Tensor | None,
        dmean_k_t: torch.Tensor | None,
        dcount_t: torch.Tensor | None,
    ):
        del dcount_t
        (
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            initial_h0_saved,
            initial_mean_v_saved,
            initial_mean_k_saved,
            initial_count_saved,
        ) = ctx.saved_tensors

        initial_state = None
        if not ctx.initial_state_was_none:
            initial_state = (
                initial_h0_saved,
                initial_mean_v_saved,
                initial_mean_k_saved,
                initial_count_saved,
            )

        dq, dk, dv, dh0, dmv0, dmk0 = _mean_backward_recompute(
            q=q,
            k=k,
            v=v,
            do=do,
            dht=dht,
            dmean_v_t=dmean_v_t,
            dmean_k_t=dmean_k_t,
            scale=ctx.scale,
            initial_state=initial_state,
            cu_seqlens=ctx.cu_seqlens,
            recompute_chunk_size=ctx.recompute_chunk_size,
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        dcount0 = None
        if not ctx.initial_state_was_none:
            # Running token count is treated as a constant statistics carrier.
            dcount0 = torch.zeros_like(initial_count_saved, dtype=initial_count_saved.dtype)

        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            None,
            dh0 if not ctx.initial_state_was_none else None,
            dmv0 if not ctx.initial_state_was_none else None,
            dmk0 if not ctx.initial_state_was_none else None,
            dcount0,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def fused_recurrent_mean_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    recompute_chunk_size: int = 128,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """
    Fused recurrent mean-delta rule operator.

    Update:
      S_t = S_{t-1} + (v_t - mean_{v,t-1}) k_t^T + v_t (k_t^T - mean_{k,t-1}^T)

    External state layout:
      kv_state: [N, H, K, V]
      mean_v: [N, H, V]
      mean_k: [N, H, K]
      count: [N, H]
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        if scale <= 0:
            raise ValueError("scale must be positive.")

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            )
        if initial_state is not None and initial_state[0].shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                "The number of initial states is expected to equal the number of input sequences."
            )

    if max(q.shape[-1], v.shape[-1]) > 128:
        if use_qk_l2norm_in_kernel:
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            k = torch.nn.functional.normalize(k, p=2, dim=-1)
        return mean_delta_rule_recurrence(
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

    h0 = mean_v0 = mean_k0 = count0 = None
    if initial_state is not None:
        h0, mean_v0, mean_k0, count0 = initial_state

    o, final_kv, final_mean_v, final_mean_k, final_count = FusedRecurrentMeanDeltaFunction.apply(
        q,
        k,
        v,
        scale,
        h0,
        mean_v0,
        mean_k0,
        count0,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        recompute_chunk_size,
    )

    final_state = None
    if output_final_state:
        final_state = (
            final_kv,
            final_mean_v,
            final_mean_k,
            final_count,
        )
    return o, final_state
