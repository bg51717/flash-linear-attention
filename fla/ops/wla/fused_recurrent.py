"""Triton fused recurrent kernels for WLA (Wiener Linear Attention)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0_S'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht_S'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def _fused_recurrent_wla_fwd_kernel(
    q_r, k_r, v, alpha, beta, sigma2,
    o, q_n_out,
    h0_S, h0_G,
    ht_S, ht_G,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    DR: tl.constexpr,
    DV: tl.constexpr,
    BDV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_dv, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T

    offs_dr = tl.arange(0, DR)
    offs_dv = i_dv * BDV + tl.arange(0, BDV)
    mask_dv = offs_dv < DV

    p_qr = q_r + (bos * H + i_h) * DR + offs_dr
    p_kr = k_r + (bos * H + i_h) * DR + offs_dr
    p_v = v + (bos * H + i_h) * DV + offs_dv
    p_o = o + (bos * H + i_h) * DV + offs_dv
    p_qn = q_n_out + (bos * H + i_h) * DR + offs_dr
    p_alpha = alpha + bos * H + i_h
    p_beta = beta + bos * H + i_h

    b_s2 = tl.load(sigma2 + i_h).to(tl.float32)
    b_s2_sq = b_s2 * b_s2
    b_s2_cu = b_s2_sq * b_s2

    b_S = tl.zeros([DR, BDV], dtype=tl.float32)
    b_G = tl.zeros([DR, DR], dtype=tl.float32)

    if USE_INITIAL_STATE:
        offs_S = tl.arange(0, DR)[:, None] * DV + offs_dv[None, :]
        b_S = tl.load(h0_S + i_nh * DR * DV + offs_S,
                       mask=mask_dv[None, :], other=0.0).to(tl.float32)
        offs_G = tl.arange(0, DR)[:, None] * DR + tl.arange(0, DR)[None, :]
        b_G = tl.load(h0_G + i_nh * DR * DR + tl.arange(0, DR * DR)).to(tl.float32)
        b_G = tl.reshape(b_G, [DR, DR])

    for _ in range(0, T):
        b_q = tl.load(p_qr).to(tl.float32)
        b_k = tl.load(p_kr).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_dv, other=0.0).to(tl.float32)
        b_a = tl.load(p_alpha).to(tl.float32)
        b_b = tl.load(p_beta).to(tl.float32)

        b_S = b_a * b_S + b_b * b_k[:, None] * b_v[None, :]
        b_G = b_a * b_G + b_b * b_k[:, None] * b_k[None, :]

        b_Gq = tl.sum(b_G * b_q[None, :], axis=1)
        b_GGq = tl.sum(b_G * b_Gq[None, :], axis=1)
        b_qw = b_q / b_s2 - b_Gq / b_s2_sq + b_GGq / b_s2_cu

        b_qw_norm = tl.sqrt(tl.sum(b_qw * b_qw) + 1e-12)
        b_qn = b_qw / b_qw_norm

        if i_dv == 0:
            tl.store(p_qn, b_qn.to(p_qn.dtype.element_ty))

        b_out = tl.sum(b_S * b_qn[:, None], axis=0)
        tl.store(p_o, b_out.to(p_o.dtype.element_ty), mask=mask_dv)

        p_qr += H * DR
        p_kr += H * DR
        p_v += H * DV
        p_o += H * DV
        p_qn += H * DR
        p_alpha += H
        p_beta += H

    if STORE_FINAL_STATE:
        offs_S = tl.arange(0, DR)[:, None] * DV + offs_dv[None, :]
        tl.store(ht_S + i_nh * DR * DV + offs_S,
                 b_S.to(ht_S.dtype.element_ty), mask=mask_dv[None, :])
        if i_dv == 0:
            b_G_flat = tl.reshape(b_G, [DR * DR])
            tl.store(ht_G + i_nh * DR * DR + tl.arange(0, DR * DR),
                     b_G_flat.to(ht_G.dtype.element_ty))


# ---------------------------------------------------------------------------
# Backward: two-phase, register-only
# Phase 1 (reverse): propagate dS backward, compute dk, dv, dbeta
# Phase 2 (forward): recompute state, compute dq via Neumann backward
# dalpha always zeros (decay gate frozen/detached)
# dG always zeros (Gram detached during whitening)
# ---------------------------------------------------------------------------

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0_S'] is not None,
    'USE_DH0': lambda args: args['dh0_S'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht_S'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def _fused_recurrent_wla_bwd_kernel(
    q_r, k_r, v,
    alpha, beta, sigma2,
    q_n_saved,
    do_grad,
    h0_S, h0_G,
    dh0_S,
    dht_S,
    dq, dk, dv_out, dbeta_out, dsigma2_out,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    DR: tl.constexpr,
    DV: tl.constexpr,
    BDV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_DH0: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_dv, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all_T = T
        T = eos - bos
    else:
        bos = i_n * T
        all_T = B * T

    offs_dr = tl.arange(0, DR)
    offs_dv = i_dv * BDV + tl.arange(0, BDV)
    mask_dv = offs_dv < DV

    b_s2 = tl.load(sigma2 + i_h).to(tl.float32)
    b_s2_sq = b_s2 * b_s2
    b_s2_cu = b_s2_sq * b_s2

    last = T - 1

    # ===== Phase 1: reverse pass =====
    p_qn = q_n_saved + (bos * H + i_h) * DR + offs_dr + last * H * DR
    p_kr = k_r + (bos * H + i_h) * DR + offs_dr + last * H * DR
    p_v = v + (bos * H + i_h) * DV + offs_dv + last * H * DV
    p_do = do_grad + (bos * H + i_h) * DV + offs_dv + last * H * DV
    p_alpha = alpha + bos * H + i_h + last * H
    p_beta = beta + bos * H + i_h + last * H
    p_dk = dk + (i_dv * all_T + bos + last) * H * DR + i_h * DR + offs_dr
    p_dv = dv_out + (bos * H + i_h) * DV + offs_dv + last * H * DV
    p_dbeta = dbeta_out + (i_dv * all_T + bos + last) * H + i_h

    b_dS = tl.zeros([DR, BDV], dtype=tl.float32)

    if USE_FINAL_STATE_GRADIENT:
        offs_S = tl.arange(0, DR)[:, None] * DV + offs_dv[None, :]
        b_dS = tl.load(dht_S + i_nh * DR * DV + offs_S,
                        mask=mask_dv[None, :], other=0.0).to(tl.float32)

    for _ in range(T):
        b_qn = tl.load(p_qn).to(tl.float32)
        b_k = tl.load(p_kr).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_dv, other=0.0).to(tl.float32)
        b_do_val = tl.load(p_do, mask=mask_dv, other=0.0).to(tl.float32)
        b_a = tl.load(p_alpha).to(tl.float32)
        b_b = tl.load(p_beta).to(tl.float32)

        b_dS = b_dS + b_qn[:, None] * b_do_val[None, :]

        b_dk_val = b_b * tl.sum(b_dS * b_v[None, :], axis=1)
        b_dv_val = b_b * tl.sum(b_dS * b_k[:, None], axis=0)
        b_dbeta_val = tl.sum(b_k[:, None] * b_v[None, :] * b_dS)

        tl.store(p_dk, b_dk_val.to(p_dk.dtype.element_ty))
        tl.store(p_dv, b_dv_val.to(p_dv.dtype.element_ty), mask=mask_dv)
        tl.store(p_dbeta, b_dbeta_val.to(p_dbeta.dtype.element_ty))

        b_dS = b_a * b_dS

        p_qn -= H * DR
        p_kr -= H * DR
        p_v -= H * DV
        p_do -= H * DV
        p_alpha -= H
        p_beta -= H
        p_dk -= H * DR
        p_dv -= H * DV
        p_dbeta -= H

    if USE_DH0:
        offs_S = tl.arange(0, DR)[:, None] * DV + offs_dv[None, :]
        tl.store(dh0_S + i_nh * DR * DV + offs_S,
                 b_dS.to(dh0_S.dtype.element_ty), mask=mask_dv[None, :])

    # ===== Phase 2: forward pass =====
    # Recompute S, G. Compute dq via Neumann backward.
    b_S = tl.zeros([DR, BDV], dtype=tl.float32)
    b_G = tl.zeros([DR, DR], dtype=tl.float32)
    b_ds2_accum = 0.0

    if USE_INITIAL_STATE:
        offs_S_init = tl.arange(0, DR)[:, None] * DV + offs_dv[None, :]
        b_S = tl.load(h0_S + i_nh * DR * DV + offs_S_init,
                       mask=mask_dv[None, :], other=0.0).to(tl.float32)
        b_G = tl.load(h0_G + i_nh * DR * DR + tl.arange(0, DR * DR)).to(tl.float32)
        b_G = tl.reshape(b_G, [DR, DR])

    p_qr = q_r + (bos * H + i_h) * DR + offs_dr
    p_kr = k_r + (bos * H + i_h) * DR + offs_dr
    p_v = v + (bos * H + i_h) * DV + offs_dv
    p_qn = q_n_saved + (bos * H + i_h) * DR + offs_dr
    p_do = do_grad + (bos * H + i_h) * DV + offs_dv
    p_alpha = alpha + bos * H + i_h
    p_beta = beta + bos * H + i_h
    p_dq = dq + (i_dv * all_T + bos) * H * DR + i_h * DR + offs_dr

    for _ in range(T):
        b_q = tl.load(p_qr).to(tl.float32)
        b_k = tl.load(p_kr).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_dv, other=0.0).to(tl.float32)
        b_qn = tl.load(p_qn).to(tl.float32)
        b_do_val = tl.load(p_do, mask=mask_dv, other=0.0).to(tl.float32)
        b_a = tl.load(p_alpha).to(tl.float32)
        b_b = tl.load(p_beta).to(tl.float32)

        b_S = b_a * b_S + b_b * b_k[:, None] * b_v[None, :]
        b_G = b_a * b_G + b_b * b_k[:, None] * b_k[None, :]

        # dq_n partial from this block
        b_dqn_partial = tl.sum(b_S * b_do_val[None, :], axis=1)

        # Recompute q_w and norm for L2norm backward
        b_Gq = tl.sum(b_G * b_q[None, :], axis=1)
        b_GGq = tl.sum(b_G * b_Gq[None, :], axis=1)
        b_qw = b_q / b_s2 - b_Gq / b_s2_sq + b_GGq / b_s2_cu
        b_qw_norm = tl.sqrt(tl.sum(b_qw * b_qw) + 1e-12)

        # L2norm backward (linear in dq_n_partial)
        b_proj = tl.sum(b_qn * b_dqn_partial)
        b_dqw_partial = (b_dqn_partial - b_qn * b_proj) / b_qw_norm

        # Neumann backward (same series applied to dq_w, linear)
        b_Gdqw = tl.sum(b_G * b_dqw_partial[None, :], axis=1)
        b_GGdqw = tl.sum(b_G * b_Gdqw[None, :], axis=1)
        b_dq_partial = b_dqw_partial / b_s2 - b_Gdqw / b_s2_sq + b_GGdqw / b_s2_cu

        tl.store(p_dq, b_dq_partial.to(p_dq.dtype.element_ty))

        # dsigma2: dq_w · ∂q_w/∂s2
        b_dqw_ds2 = -b_q / b_s2_sq + 2.0 * b_Gq / b_s2_cu - 3.0 * b_GGq / (b_s2_cu * b_s2)
        b_ds2_accum = b_ds2_accum + tl.sum(b_dqw_partial * b_dqw_ds2)

        p_qr += H * DR
        p_kr += H * DR
        p_v += H * DV
        p_qn += H * DR
        p_do += H * DV
        p_alpha += H
        p_beta += H
        p_dq += H * DR

    tl.store(dsigma2_out + i_dv * B * H + i_nh, b_ds2_accum)


def fused_recurrent_wla_fwd(
    q_r: torch.Tensor,
    k_r: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    sigma2: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor,
           torch.Tensor | None, torch.Tensor | None]:
    B, T, H, DR = q_r.shape
    DV = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BDV = min(triton.next_power_of_2(DV), 4)
    NV = triton.cdiv(DV, BDV)

    o = torch.empty_like(v)
    q_n_saved = q_r.new_empty(B, T, H, DR, dtype=torch.float32)

    h0_S, h0_G = None, None
    if initial_state is not None:
        S0, G0 = initial_state
        h0_S = S0.reshape(N * H, DR, DV).contiguous().float()
        h0_G = G0.reshape(N * H, DR * DR).contiguous().float()

    ht_S = q_r.new_empty(N * H, DR, DV, dtype=torch.float32) if output_final_state else None
    ht_G = q_r.new_empty(N * H, DR * DR, dtype=torch.float32) if output_final_state else None

    grid = (NV, N * H)
    _fused_recurrent_wla_fwd_kernel[grid](
        q_r, k_r, v, alpha, beta, sigma2,
        o, q_n_saved,
        h0_S, h0_G,
        ht_S, ht_G,
        cu_seqlens,
        T=T, B=B, H=H, DR=DR, DV=DV, BDV=BDV,
        num_warps=1, num_stages=1,
    )
    return o, q_n_saved, ht_S, ht_G


def fused_recurrent_wla_bwd(
    q_r: torch.Tensor,
    k_r: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    sigma2: torch.Tensor,
    q_n_saved: torch.Tensor,
    do_grad: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None,
    dht: tuple[torch.Tensor, ...] | None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor | None]:
    B, T, H, DR = q_r.shape
    DV = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BDV = min(triton.next_power_of_2(DV), 4)
    NV = triton.cdiv(DV, BDV)

    dq = q_r.new_empty(NV, B, T, H, DR)
    dk = q_r.new_empty(NV, B, T, H, DR)
    dv_out = torch.empty_like(v)
    dbeta_out = q_r.new_empty(NV, B, T, H)
    dalpha = alpha.new_zeros(B, T, H)
    dsigma2_out = q_r.new_empty(NV, B * H)

    h0_S, h0_G = None, None
    if initial_state is not None:
        S0, G0 = initial_state
        h0_S = S0.reshape(N * H, DR, DV).contiguous().float()
        h0_G = G0.reshape(N * H, DR * DR).contiguous().float()

    need_dh0 = initial_state is not None and any(
        t is not None and t.requires_grad for t in initial_state
    )
    dh0_S = torch.empty(N * H, DR, DV, device=q_r.device, dtype=torch.float32) if need_dh0 else None

    dht_S = None
    if dht is not None:
        dS_final = dht[0]
        if dS_final is not None:
            dht_S = dS_final.reshape(N * H, DR, DV).contiguous().float()

    grid = (NV, N * H)
    _fused_recurrent_wla_bwd_kernel[grid](
        q_r, k_r, v,
        alpha, beta, sigma2,
        q_n_saved,
        do_grad,
        h0_S, h0_G,
        dh0_S,
        dht_S,
        dq, dk, dv_out, dbeta_out, dsigma2_out,
        cu_seqlens,
        T=T, B=B, H=H, DR=DR, DV=DV, BDV=BDV,
        num_warps=1, num_stages=1,
    )

    dq = dq.sum(0)
    dk = dk.sum(0)
    dbeta = dbeta_out.sum(0)
    dsigma2 = dsigma2_out.sum(0).reshape(B, H).sum(0)

    return dq, dk, dv_out, dalpha, dbeta, dsigma2, dh0_S


class FusedRecurrentWLAFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q_r: torch.Tensor,
        k_r: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        sigma2: torch.Tensor,
        initial_state: tuple[torch.Tensor, ...] | None,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        o, q_n, ht_S, ht_G = fused_recurrent_wla_fwd(
            q_r, k_r, v, alpha, beta, sigma2,
            initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(q_r, k_r, v, alpha, beta, sigma2, q_n)
        if initial_state is not None:
            ctx._initial_state = initial_state
        else:
            ctx._initial_state = None
        ctx.cu_seqlens = cu_seqlens
        ctx._output_final_state = output_final_state

        final_state = None
        if output_final_state:
            N, H, DR = q_r.shape[0], q_r.shape[2], q_r.shape[3]
            DV = v.shape[-1]
            n_out = cu_seqlens.numel() - 1 if cu_seqlens is not None else q_r.shape[0]
            final_state = (
                ht_S.reshape(n_out, H, DR, DV).contiguous(),
                ht_G.reshape(n_out, H, DR, DR).contiguous(),
            )
        return o, final_state

    @staticmethod
    def backward(ctx, do, d_final_state):
        q_r, k_r, v, alpha, beta, sigma2, q_n = ctx.saved_tensors
        initial_state = ctx._initial_state

        dht = None
        if d_final_state is not None:
            dht = d_final_state

        dq, dk, dv, dalpha, dbeta, dsigma2, dh0_S = fused_recurrent_wla_bwd(
            q_r, k_r, v, alpha, beta, sigma2, q_n,
            do,
            initial_state, dht,
            cu_seqlens=ctx.cu_seqlens,
        )

        dh0 = None
        if dh0_S is not None:
            B, H, DR = q_r.shape[0], q_r.shape[2], q_r.shape[3]
            DV = v.shape[-1]
            n_out = ctx.cu_seqlens.numel() - 1 if ctx.cu_seqlens is not None else B
            dh0 = (
                dh0_S.reshape(n_out, H, DR, DV),
                None,
            )

        return (dq.to(q_r), dk.to(k_r), dv.to(v),
                dalpha.to(alpha), dbeta.to(beta), dsigma2.to(sigma2),
                dh0, None, None)


@torch.compiler.disable
def fused_recurrent_wla(
    q_r: torch.Tensor,
    k_r: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    sigma2: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    if cu_seqlens is not None and q_r.shape[0] != 1:
        raise ValueError(f"Batch must be 1 with cu_seqlens, got {q_r.shape[0]}.")
    o, final_state = FusedRecurrentWLAFunction.apply(
        q_r, k_r, v, alpha, beta, sigma2,
        initial_state, output_final_state, cu_seqlens,
    )
    return o, final_state
