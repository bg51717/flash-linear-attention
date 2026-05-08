"""Triton fused recurrent kernels for SOAM linear attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def _fused_recurrent_soam_fwd_kernel(
    q_r,
    k_r,
    v,
    alpha,
    beta,
    o,
    err,
    h0,
    ht,
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
    p_err = err + (bos * H + i_h) * DV + offs_dv
    p_alpha = alpha + bos * H + i_h
    p_beta = beta + bos * H + i_h

    b_T = tl.zeros([DR, DR, BDV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        offs_st = tl.arange(0, DR * DR)[:, None] * DV + offs_dv[None, :]
        b_T_flat = tl.load(h0 + i_nh * DR * DR * DV + offs_st,
                           mask=mask_dv[None, :], other=0.0).to(tl.float32)
        b_T = tl.reshape(b_T_flat, [DR, DR, BDV])

    for _ in range(0, T):
        b_q = tl.load(p_qr).to(tl.float32)
        b_k = tl.load(p_kr).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_dv, other=0.0).to(tl.float32)
        b_a = tl.load(p_alpha).to(tl.float32)
        b_b = tl.load(p_beta).to(tl.float32)

        b_Tk = tl.sum(b_T * b_k[None, :, None], axis=1)
        b_ret = tl.sum(b_Tk * b_k[:, None], axis=0)

        b_err = b_v - b_ret
        tl.store(p_err, b_err.to(p_err.dtype.element_ty), mask=mask_dv)

        b_kk = b_k[:, None] * b_k[None, :]
        b_T = b_a * b_T + b_b * b_kk[:, :, None] * b_err[None, None, :]

        b_Tq = tl.sum(b_T * b_q[None, :, None], axis=1)
        b_out = tl.sum(b_Tq * b_q[:, None], axis=0)
        tl.store(p_o, b_out.to(p_o.dtype.element_ty), mask=mask_dv)

        p_qr += H * DR
        p_kr += H * DR
        p_v += H * DV
        p_o += H * DV
        p_err += H * DV
        p_alpha += H
        p_beta += H

    if STORE_FINAL_STATE:
        offs_st = tl.arange(0, DR * DR)[:, None] * DV + offs_dv[None, :]
        b_T_flat = tl.reshape(b_T, [DR * DR, BDV])
        tl.store(ht + i_nh * DR * DR * DV + offs_st,
                 b_T_flat.to(ht.dtype.element_ty), mask=mask_dv[None, :])


# ---------------------------------------------------------------------------
# Backward: two-phase, register-only (no global T buffer)
# Phase 1 (reverse): propagate dT backward, compute dv, dk_partial, dbeta
# Phase 2 (forward): recompute T from h0, correct dk, compute dq
# dalpha is NOT computed (decay gate kept fixed) — eliminates 4.5 GB T_buf
# ---------------------------------------------------------------------------

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_DH0': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def _fused_recurrent_soam_bwd_kernel(
    q_r,
    k_r,
    err,
    alpha,
    beta,
    h0,
    dh0,
    dht,
    do,
    dq,
    dk,
    dv,
    dbeta_out,
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
    offs_st = tl.arange(0, DR * DR)[:, None] * DV + offs_dv[None, :]
    mask_st = mask_dv[None, :]

    last = T - 1

    # ===== Phase 1: reverse pass =====
    # Propagate dT backward. Compute dv, partial dk (kk*err term), dbeta.
    p_qr = q_r + (bos * H + i_h) * DR + offs_dr + last * H * DR
    p_kr = k_r + (bos * H + i_h) * DR + offs_dr + last * H * DR
    p_err = err + (bos * H + i_h) * DV + offs_dv + last * H * DV
    p_do = do + (bos * H + i_h) * DV + offs_dv + last * H * DV
    p_alpha = alpha + bos * H + i_h + last * H
    p_beta = beta + bos * H + i_h + last * H
    p_dk = dk + (i_dv * all_T + bos + last) * H * DR + i_h * DR + offs_dr
    p_dv = dv + (bos * H + i_h) * DV + offs_dv + last * H * DV
    p_dbeta = dbeta_out + (i_dv * all_T + bos + last) * H + i_h

    b_dT = tl.zeros([DR, DR, BDV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        b_dT_flat = tl.load(dht + i_nh * DR * DR * DV + offs_st,
                            mask=mask_st, other=0.0).to(tl.float32)
        b_dT = tl.reshape(b_dT_flat, [DR, DR, BDV])

    for _ in range(T):
        b_q = tl.load(p_qr).to(tl.float32)
        b_k = tl.load(p_kr).to(tl.float32)
        b_err = tl.load(p_err, mask=mask_dv, other=0.0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_dv, other=0.0).to(tl.float32)
        b_a = tl.load(p_alpha).to(tl.float32)
        b_b = tl.load(p_beta).to(tl.float32)

        b_qq = b_q[:, None] * b_q[None, :]
        b_dT = b_dT + b_qq[:, :, None] * b_do[None, None, :]

        b_dTk = tl.sum(b_dT * b_k[None, :, None], axis=1)
        b_dTk_t = tl.sum(b_dT * b_k[:, None, None], axis=0)
        b_dT_kk = tl.sum(b_dTk * b_k[:, None], axis=0)

        b_dv_val = b_b * b_dT_kk
        tl.store(p_dv, b_dv_val.to(p_dv.dtype.element_ty), mask=mask_dv)

        b_dk_val = b_b * tl.sum((b_dTk + b_dTk_t) * b_err[None, :], axis=1)
        tl.store(p_dk, b_dk_val.to(p_dk.dtype.element_ty))

        b_dbeta_val = tl.sum(b_dT_kk * b_err)
        tl.store(p_dbeta, b_dbeta_val.to(p_dbeta.dtype.element_ty))

        b_kk = b_k[:, None] * b_k[None, :]
        b_dT = b_a * b_dT - b_kk[:, :, None] * b_dv_val[None, None, :]

        p_qr -= H * DR
        p_kr -= H * DR
        p_err -= H * DV
        p_do -= H * DV
        p_alpha -= H
        p_beta -= H
        p_dk -= H * DR
        p_dv -= H * DV
        p_dbeta -= H

    if USE_DH0:
        b_dT_flat = tl.reshape(b_dT, [DR * DR, BDV])
        tl.store(dh0 + i_nh * DR * DR * DV + offs_st,
                 b_dT_flat.to(dh0.dtype.element_ty), mask=mask_st)

    # ===== Phase 2: forward pass =====
    # Recompute T from h0. Correct dk (retrieval term), compute dq.
    b_T = tl.zeros([DR, DR, BDV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_T_flat = tl.load(h0 + i_nh * DR * DR * DV + offs_st,
                           mask=mask_st, other=0.0).to(tl.float32)
        b_T = tl.reshape(b_T_flat, [DR, DR, BDV])

    p_qr = q_r + (bos * H + i_h) * DR + offs_dr
    p_kr = k_r + (bos * H + i_h) * DR + offs_dr
    p_err = err + (bos * H + i_h) * DV + offs_dv
    p_do = do + (bos * H + i_h) * DV + offs_dv
    p_alpha = alpha + bos * H + i_h
    p_beta = beta + bos * H + i_h
    p_dk = dk + (i_dv * all_T + bos) * H * DR + i_h * DR + offs_dr
    p_dv = dv + (bos * H + i_h) * DV + offs_dv
    p_dq = dq + (i_dv * all_T + bos) * H * DR + i_h * DR + offs_dr

    for _ in range(T):
        b_q = tl.load(p_qr).to(tl.float32)
        b_k = tl.load(p_kr).to(tl.float32)
        b_err = tl.load(p_err, mask=mask_dv, other=0.0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_dv, other=0.0).to(tl.float32)
        b_a = tl.load(p_alpha).to(tl.float32)
        b_b = tl.load(p_beta).to(tl.float32)

        b_dk_partial = tl.load(p_dk).to(tl.float32)
        b_dv_val = tl.load(p_dv, mask=mask_dv, other=0.0).to(tl.float32)

        b_Tk = tl.sum(b_T * b_k[None, :, None], axis=1)
        b_Tk_t = tl.sum(b_T * b_k[:, None, None], axis=0)
        b_dk_partial = b_dk_partial - tl.sum(b_dv_val[None, :] * (b_Tk + b_Tk_t), axis=1)
        tl.store(p_dk, b_dk_partial.to(p_dk.dtype.element_ty))

        b_kk = b_k[:, None] * b_k[None, :]
        b_T = b_a * b_T + b_b * b_kk[:, :, None] * b_err[None, None, :]

        b_Tq = tl.sum(b_T * b_q[None, :, None], axis=1)
        b_Tq_t = tl.sum(b_T * b_q[:, None, None], axis=0)
        b_dq_val = tl.sum((b_Tq + b_Tq_t) * b_do[None, :], axis=1)
        tl.store(p_dq, b_dq_val.to(p_dq.dtype.element_ty))

        p_qr += H * DR
        p_kr += H * DR
        p_err += H * DV
        p_do += H * DV
        p_alpha += H
        p_beta += H
        p_dk += H * DR
        p_dv += H * DV
        p_dq += H * DR


def fused_recurrent_soam_fwd(
    q_r: torch.Tensor,
    k_r: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, DR = q_r.shape
    DV = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BDV = min(triton.next_power_of_2(DV), 4)
    NV = triton.cdiv(DV, BDV)

    o = torch.empty_like(v)
    err = torch.empty_like(v)

    h0 = initial_state
    ht = q_r.new_empty(N * H, DR * DR, DV, dtype=torch.float32) if output_final_state else None

    grid = (NV, N * H)
    _fused_recurrent_soam_fwd_kernel[grid](
        q_r, k_r, v, alpha, beta,
        o, err, h0, ht,
        cu_seqlens,
        T=T, B=B, H=H, DR=DR, DV=DV, BDV=BDV,
        num_warps=1, num_stages=1,
    )
    return o, err, ht


def fused_recurrent_soam_bwd(
    q_r: torch.Tensor,
    k_r: torch.Tensor,
    err: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    dht: torch.Tensor | None,
    do: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, DR = q_r.shape
    DV = err.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BDV = min(triton.next_power_of_2(DV), 4)
    NV = triton.cdiv(DV, BDV)

    dq = q_r.new_empty(NV, B, T, H, DR)
    dk = q_r.new_empty(NV, B, T, H, DR)
    dv = torch.empty_like(err)
    dbeta_out = q_r.new_empty(NV, B, T, H)
    dalpha = alpha.new_zeros(B, T, H)

    if initial_state is not None and initial_state.requires_grad:
        dh0 = torch.empty_like(initial_state, dtype=torch.float32)
    else:
        dh0 = None

    grid = (NV, N * H)
    _fused_recurrent_soam_bwd_kernel[grid](
        q_r, k_r, err, alpha, beta,
        initial_state, dh0, dht, do,
        dq, dk, dv, dbeta_out,
        cu_seqlens,
        T=T, B=B, H=H, DR=DR, DV=DV, BDV=BDV,
        num_warps=1, num_stages=1,
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dbeta = dbeta_out.sum(0)
    return dq, dk, dv, dalpha, dbeta, dh0


class FusedRecurrentSOAMFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q_r: torch.Tensor,
        k_r: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        o, err, final_state = fused_recurrent_soam_fwd(
            q_r, k_r, v, alpha, beta,
            initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q_r, k_r, err, alpha, beta, initial_state)
        ctx.cu_seqlens = cu_seqlens
        ctx._output_final_state = output_final_state
        return o, final_state if output_final_state else None

    @staticmethod
    def backward(ctx, do, dht):
        q_r, k_r, err, alpha, beta, initial_state = ctx.saved_tensors
        dq, dk, dv, dalpha, dbeta, dh0 = fused_recurrent_soam_bwd(
            q_r, k_r, err, alpha, beta,
            initial_state, dht, do,
            cu_seqlens=ctx.cu_seqlens,
        )
        return dq.to(q_r), dk.to(k_r), dv.to(err), dalpha.to(alpha), dbeta.to(beta), dh0, None, None


@torch.compiler.disable
def fused_recurrent_soam(
    q_r: torch.Tensor,
    k_r: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Triton-accelerated SOAM fused recurrence.

    Args:
        q_r: [B, T, H, DR]
        k_r: [B, T, H, DR]
        v:   [B, T, H, DV]
        alpha: [B, T, H] per-timestep decay gate
        beta:  [B, T, H] per-timestep write gate
        initial_state: [N, H, DR*DR, DV] or None
        output_final_state: bool
        cu_seqlens: [N+1] for varlen

    Returns:
        o: [B, T, H, DV]
        final_state: [N*H, DR*DR, DV] or None
    """
    if cu_seqlens is not None and q_r.shape[0] != 1:
        raise ValueError(f"Batch must be 1 with cu_seqlens, got {q_r.shape[0]}.")
    o, final_state = FusedRecurrentSOAMFunction.apply(
        q_r, k_r, v, alpha, beta,
        initial_state, output_final_state, cu_seqlens,
    )
    return o, final_state
