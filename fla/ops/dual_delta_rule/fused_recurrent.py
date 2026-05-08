# Copyright (c) 2026

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.utils import input_guard

from .naive import dual_delta_rule_recurrence


@triton.heuristics({
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_dual_delta_rule_fwd_kernel(
    q,
    k,
    v,
    o,
    h0,
    ht,
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

    # Internal state layout: [V, K], while external cache layout is [K, V].
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + offs_k[None, :] * V + offs_v[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, Tcur):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale

        # a = S k, b = S^T v
        b_a = tl.sum(b_h * b_k[None, :], axis=1)      # [V]
        b_b = tl.sum(b_h * b_v[:, None], axis=0)      # [K]

        b_x = b_v - b_a
        b_y = b_k - b_b

        # S <- S + x k^T + v y^T
        b_h += b_x[:, None] * b_k[None, :] + b_v[:, None] * b_y[None, :]

        b_o = tl.sum(b_h * b_q[None, :], axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_o += H * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + offs_k[None, :] * V + offs_v[:, None]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.jit(do_not_specialize=["T", "t_start", "L"])
def fused_recurrent_dual_delta_rule_chunk_state_kernel(
    k,
    v,
    state_in,
    state_out,
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

    p_k = k + i_bh * T * K + t_start * K + offs_k
    p_v = v + i_bh * T * V + t_start * V + offs_v
    for _ in range(0, L):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        b_a = tl.sum(b_h * b_k[None, :], axis=1)
        b_b = tl.sum(b_h * b_v[:, None], axis=0)
        b_x = b_v - b_a
        b_y = b_k - b_b
        b_h += b_x[:, None] * b_k[None, :] + b_v[:, None] * b_y[None, :]

        p_k += K
        p_v += V

    p_state_out = state_out + i_bh * V * K + offs_v[:, None] * K + offs_k[None, :]
    tl.store(p_state_out, b_h.to(p_state_out.dtype.element_ty), mask=mask_h)


@triton.jit(do_not_specialize=["T", "t_start", "L"])
def fused_recurrent_dual_delta_rule_chunk_states_kernel(
    k,
    v,
    state_in,
    local_states,
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

    p_ls_base = local_states + i_bh * (CHUNK_MAX + 1) * V * K + offs_v[:, None] * K + offs_k[None, :]
    tl.store(p_ls_base, b_h.to(local_states.dtype.element_ty), mask=mask_h)

    p_k = k + i_bh * T * K + t_start * K + offs_k
    p_v = v + i_bh * T * V + t_start * V + offs_v
    p_ls = p_ls_base + V * K

    for _ in range(0, L):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        b_a = tl.sum(b_h * b_k[None, :], axis=1)
        b_b = tl.sum(b_h * b_v[:, None], axis=0)
        b_x = b_v - b_a
        b_y = b_k - b_b
        b_h += b_x[:, None] * b_k[None, :] + b_v[:, None] * b_y[None, :]

        tl.store(p_ls, b_h.to(local_states.dtype.element_ty), mask=mask_h)
        p_ls += V * K
        p_k += K
        p_v += V


@triton.jit(do_not_specialize=["T", "t_start", "L"])
def fused_recurrent_dual_delta_rule_chunk_bwd_kernel(
    q,
    k,
    v,
    do,
    local_states,
    dS_next,
    dq,
    dk,
    dv,
    dS_prev,
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

    p_q = q + i_bh * T * K + (t_start + L - 1) * K + offs_k
    p_k = k + i_bh * T * K + (t_start + L - 1) * K + offs_k
    p_v = v + i_bh * T * V + (t_start + L - 1) * V + offs_v
    p_do = do + i_bh * T * V + (t_start + L - 1) * V + offs_v
    p_dq = dq + i_bh * T * K + (t_start + L - 1) * K + offs_k
    p_dk = dk + i_bh * T * K + (t_start + L - 1) * K + offs_k
    p_dv = dv + i_bh * T * V + (t_start + L - 1) * V + offs_v

    p_ls_base = local_states + i_bh * (CHUNK_MAX + 1) * V * K + offs_v[:, None] * K + offs_k[None, :]
    p_S_prev = p_ls_base + (L - 1) * V * K
    p_S_t = p_ls_base + L * V * K

    for _ in range(0, L):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)

        b_S_prev = tl.load(p_S_prev, mask=mask_h, other=0).to(tl.float32)
        b_S_t = tl.load(p_S_t, mask=mask_h, other=0).to(tl.float32)

        b_dS += b_do[:, None] * (b_q * scale)[None, :]

        b_a = tl.sum(b_S_prev * b_k[None, :], axis=1)
        b_b = tl.sum(b_S_prev * b_v[:, None], axis=0)
        b_x = b_v - b_a
        b_y = b_k - b_b

        b_dX = tl.sum(b_dS * b_k[None, :], axis=1)
        b_dY = tl.sum(b_dS * b_v[:, None], axis=0)
        b_da = -b_dX
        b_db = -b_dY

        b_dk = (
            tl.sum(b_dS * b_x[:, None], axis=0)
            + b_dY
            + tl.sum(b_S_prev * b_da[:, None], axis=0)
        )
        b_dv = (
            tl.sum(b_dS * b_y[None, :], axis=1)
            + b_dX
            + tl.sum(b_S_prev * b_db[None, :], axis=1)
        )
        b_dq = tl.sum(b_S_t * b_do[:, None], axis=0) * scale

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)

        b_dS += b_da[:, None] * b_k[None, :] + b_v[:, None] * b_db[None, :]

        p_q -= K
        p_k -= K
        p_v -= V
        p_do -= V
        p_dq -= K
        p_dk -= K
        p_dv -= V
        p_S_prev -= V * K
        p_S_t -= V * K

    p_dS_prev = dS_prev + i_bh * V * K + offs_v[:, None] * K + offs_k[None, :]
    tl.store(p_dS_prev, b_dS.to(p_dS_prev.dtype.element_ty), mask=mask_h)


def _dual_step(state_vk: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
    a_t = torch.einsum("bhvk,bhk->bhv", state_vk, k_t)
    b_t = torch.einsum("bhvk,bhv->bhk", state_vk, v_t)
    x_t = v_t - a_t
    y_t = k_t - b_t
    return (
        state_vk
        + torch.einsum("bhv,bhk->bhvk", x_t, k_t)
        + torch.einsum("bhv,bhk->bhvk", v_t, y_t)
    )


def _dual_backward_segment_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    initial_state_vk: torch.Tensor,
    dht_vk: torch.Tensor,
    recompute_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # q/k/v/do: [B, T, H, ...]
    B, T, H, K = q.shape
    V = v.shape[-1]
    BH = B * H

    # flatten BH to improve GEMM efficiency in the recurrent backward loop
    qf = q.float().permute(0, 2, 1, 3).reshape(BH, T, K).contiguous()
    kf = k.float().permute(0, 2, 1, 3).reshape(BH, T, K).contiguous()
    vf = v.float().permute(0, 2, 1, 3).reshape(BH, T, V).contiguous()
    dof = do.float().permute(0, 2, 1, 3).reshape(BH, T, V).contiguous()
    h0 = initial_state_vk.float().reshape(BH, V, K).contiguous()
    dht = dht_vk.float().reshape(BH, V, K).contiguous()

    dq = torch.empty(BH, T, K, device=q.device, dtype=torch.float32)
    dk = torch.empty(BH, T, K, device=q.device, dtype=torch.float32)
    dv = torch.empty(BH, T, V, device=q.device, dtype=torch.float32)

    chunk = max(1, int(recompute_chunk_size))
    n_chunks = math.ceil(T / chunk)

    # store only chunk boundaries to trade memory for compute
    checkpoints = torch.empty(n_chunks + 1, BH, V, K, device=q.device, dtype=torch.float32)
    checkpoints[0].copy_(h0)
    state = h0

    for c in range(n_chunks):
        s = c * chunk
        e = min(T, s + chunk)
        for t in range(s, e):
            k_t = kf[:, t, :]                                  # [BH, K]
            v_t = vf[:, t, :]                                  # [BH, V]
            a_t = torch.bmm(state, k_t.unsqueeze(-1)).squeeze(-1)                  # [BH, V]
            b_t = torch.bmm(state.transpose(1, 2), v_t.unsqueeze(-1)).squeeze(-1)  # [BH, K]
            x_t = v_t - a_t
            y_t = k_t - b_t
            state = state + x_t.unsqueeze(-1) * k_t.unsqueeze(1) + v_t.unsqueeze(-1) * y_t.unsqueeze(1)
        checkpoints[c + 1].copy_(state)

    dS_next = dht
    for c in range(n_chunks - 1, -1, -1):
        s = c * chunk
        e = min(T, s + chunk)
        l = e - s

        # recompute local states in this chunk once
        local_states = torch.empty(l + 1, BH, V, K, device=q.device, dtype=torch.float32)
        local_states[0].copy_(checkpoints[c])
        state = local_states[0]
        for i in range(l):
            t = s + i
            k_t = kf[:, t, :]
            v_t = vf[:, t, :]
            a_t = torch.bmm(state, k_t.unsqueeze(-1)).squeeze(-1)
            b_t = torch.bmm(state.transpose(1, 2), v_t.unsqueeze(-1)).squeeze(-1)
            x_t = v_t - a_t
            y_t = k_t - b_t
            state = state + x_t.unsqueeze(-1) * k_t.unsqueeze(1) + v_t.unsqueeze(-1) * y_t.unsqueeze(1)
            local_states[i + 1].copy_(state)

        dS = dS_next
        for i in range(l - 1, -1, -1):
            t = s + i
            S_prev = local_states[i]
            S_t = local_states[i + 1]
            q_t = qf[:, t, :]   # [BH, K]
            k_t = kf[:, t, :]   # [BH, K]
            v_t = vf[:, t, :]   # [BH, V]
            do_t = dof[:, t, :] # [BH, V]

            dS = dS + do_t.unsqueeze(-1) * (q_t * scale).unsqueeze(1)  # [BH, V, K]

            a_t = torch.bmm(S_prev, k_t.unsqueeze(-1)).squeeze(-1)                  # [BH, V]
            b_t = torch.bmm(S_prev.transpose(1, 2), v_t.unsqueeze(-1)).squeeze(-1)  # [BH, K]
            x_t = v_t - a_t
            y_t = k_t - b_t

            dX = torch.bmm(dS, k_t.unsqueeze(-1)).squeeze(-1)                        # [BH, V]
            dY = torch.bmm(dS.transpose(1, 2), v_t.unsqueeze(-1)).squeeze(-1)        # [BH, K]
            da = -dX
            db = -dY

            dk[:, t, :] = (
                torch.bmm(dS.transpose(1, 2), x_t.unsqueeze(-1)).squeeze(-1)
                + dY
                + torch.bmm(S_prev.transpose(1, 2), da.unsqueeze(-1)).squeeze(-1)
            )
            dv[:, t, :] = (
                torch.bmm(dS, y_t.unsqueeze(-1)).squeeze(-1)
                + dX
                + torch.bmm(S_prev, db.unsqueeze(-1)).squeeze(-1)
            )
            dq[:, t, :] = torch.bmm(S_t.transpose(1, 2), do_t.unsqueeze(-1)).squeeze(-1) * scale

            dS = dS + da.unsqueeze(-1) * k_t.unsqueeze(1) + v_t.unsqueeze(-1) * db.unsqueeze(1)

        dS_next = dS

    dh0 = dS_next.reshape(B, H, V, K).contiguous()  # [B,H,V,K]
    dq = dq.reshape(B, H, T, K).permute(0, 2, 1, 3).contiguous()
    dk = dk.reshape(B, H, T, K).permute(0, 2, 1, 3).contiguous()
    dv = dv.reshape(B, H, T, V).permute(0, 2, 1, 3).contiguous()
    return dq, dk, dv, dh0


def _dual_backward_recompute_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    recompute_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if cu_seqlens is None:
        B, _, H, K = q.shape
        V = v.shape[-1]
        if initial_state is None:
            h0 = q.new_zeros(B, H, K, V, dtype=torch.float32)
        else:
            h0 = initial_state.float()
        h0_vk = h0.transpose(-2, -1).contiguous()

        if dht is None:
            dht_vk = q.new_zeros(B, H, V, K, dtype=torch.float32)
        else:
            dht_vk = dht.float().transpose(-2, -1).contiguous()

        dq, dk, dv, dh0_vk = _dual_backward_segment_recompute(
            q=q,
            k=k,
            v=v,
            do=do,
            scale=scale,
            initial_state_vk=h0_vk,
            dht_vk=dht_vk,
            recompute_chunk_size=recompute_chunk_size,
        )
        dh0 = dh0_vk.transpose(-2, -1).contiguous()
        return dq, dk, dv, dh0

    # varlen fallback: process sequence by sequence (B must be 1)
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
    if initial_state is not None:
        dh0 = torch.zeros_like(initial_state, dtype=torch.float32)

    for i in range(n_seq):
        bos = int(cu[i].item())
        eos = int(cu[i + 1].item())
        h0_i = None if initial_state is None else initial_state[i:i + 1]
        dht_i = None if dht is None else dht[i:i + 1]

        dq_i, dk_i, dv_i, dh0_i = _dual_backward_recompute_torch(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            do=do[:, bos:eos],
            dht=dht_i,
            scale=scale,
            initial_state=h0_i,
            cu_seqlens=None,
            recompute_chunk_size=recompute_chunk_size,
        )
        dq[:, bos:eos] = dq_i
        dk[:, bos:eos] = dk_i
        dv[:, bos:eos] = dv_i
        if dh0 is not None:
            dh0[i] = dh0_i[0]

    return dq, dk, dv, dh0


def _dual_backward_recompute_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    recompute_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
    else:
        h0 = initial_state.float()
    h0_vk = h0.transpose(-2, -1).reshape(BH, V, K).contiguous()

    if dht is None:
        dS_next = q.new_zeros(BH, V, K, dtype=torch.float32)
    else:
        dS_next = dht.float().transpose(-2, -1).reshape(BH, V, K).contiguous()
    dS_prev = torch.empty_like(dS_next)

    dqf = torch.empty_like(qf)
    dkf = torch.empty_like(kf)
    dvf = torch.empty_like(vf)

    checkpoints = torch.empty(n_chunks + 1, BH, V, K, device=q.device, dtype=torch.float32)
    checkpoints[0].copy_(h0_vk)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    num_warps = 4 if max(BK, BV) > 64 else 2

    # Build chunk boundary checkpoints in Triton.
    for c in range(n_chunks):
        s = c * chunk
        l = min(chunk, T - s)
        fused_recurrent_dual_delta_rule_chunk_state_kernel[(BH,)](
            kf,
            vf,
            checkpoints[c],
            checkpoints[c + 1],
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

    # Reverse over chunks; per-chunk backward fully in Triton.
    for c in range(n_chunks - 1, -1, -1):
        s = c * chunk
        l = min(chunk, T - s)

        fused_recurrent_dual_delta_rule_chunk_states_kernel[(BH,)](
            kf,
            vf,
            checkpoints[c],
            local_states,
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
        fused_recurrent_dual_delta_rule_chunk_bwd_kernel[(BH,)](
            qf,
            kf,
            vf,
            dof,
            local_states,
            dS_next,
            dqf,
            dkf,
            dvf,
            dS_prev,
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

    dh0 = dS_next.reshape(B, H, V, K).transpose(-2, -1).contiguous()
    dq = dqf.reshape(B, H, T, K).permute(0, 2, 1, 3).contiguous()
    dk = dkf.reshape(B, H, T, K).permute(0, 2, 1, 3).contiguous()
    dv = dvf.reshape(B, H, T, V).permute(0, 2, 1, 3).contiguous()
    return dq, dk, dv, dh0


def _dual_backward_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    recompute_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if (
        cu_seqlens is None
        and q.is_cuda
        and max(q.shape[-1], v.shape[-1]) <= 128
    ):
        return _dual_backward_recompute_triton(
            q=q,
            k=k,
            v=v,
            do=do,
            dht=dht,
            scale=scale,
            initial_state=initial_state,
            recompute_chunk_size=recompute_chunk_size,
        )

    return _dual_backward_recompute_torch(
        q=q,
        k=k,
        v=v,
        do=do,
        dht=dht,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        recompute_chunk_size=recompute_chunk_size,
    )


def fused_recurrent_dual_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    if BK > 128 or BV > 128:
        raise RuntimeError(
            f"Dual Delta Triton fused recurrent currently supports head dims <= 128, got K={K}, V={V}."
        )

    o = q.new_empty(B, T, H, V)
    final_state = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    # One kernel instance handles one (sequence, head) and full K/V state.
    num_warps = 4 if max(BK, BV) > 64 else 2
    fused_recurrent_dual_delta_rule_fwd_kernel[(N * H,)](
        q,
        k,
        v,
        o,
        initial_state,
        final_state,
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
    return o, final_state


class FusedRecurrentDualDeltaFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
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

        o, final_state = fused_recurrent_dual_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, initial_state if initial_state is not None else torch.tensor([], device=q.device))
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cu_seqlens = cu_seqlens
        ctx.initial_state_was_none = initial_state is None
        ctx.recompute_chunk_size = int(recompute_chunk_size)
        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        q, q_rstd, k, k_rstd, v, initial_state_saved = ctx.saved_tensors
        initial_state = None if ctx.initial_state_was_none else initial_state_saved

        dq, dk, dv, dh0 = _dual_backward_recompute(
            q=q,
            k=k,
            v=v,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            cu_seqlens=ctx.cu_seqlens,
            recompute_chunk_size=ctx.recompute_chunk_size,
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        dh0_out = None if ctx.initial_state_was_none else dh0
        return dq.to(q), dk.to(k), dv.to(v), None, dh0_out, None, None, None, None


@torch.compiler.disable
def fused_recurrent_dual_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    recompute_chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Fused recurrent dual-symmetric Delta rule operator.
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
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                "The number of initial states is expected to equal the number of input sequences."
            )

    # Fallback path: use pure torch recurrence for uncommon shapes where current Triton
    # kernel is not specialized enough.
    if max(q.shape[-1], v.shape[-1]) > 128:
        if use_qk_l2norm_in_kernel:
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            k = torch.nn.functional.normalize(k, p=2, dim=-1)
        return dual_delta_rule_recurrence(
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

    o, final_state = FusedRecurrentDualDeltaFunction.apply(
        q,
        k,
        v,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        recompute_chunk_size,
    )
    return o, final_state
