from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - runtime fallback
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def _torch_recompute_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    hist_eps: float,
    score_clip: float,
) -> torch.Tensor:
    qf = q.float()
    kf = k.float()
    vf = v.float()
    n, t, k_dim = q.shape
    v_dim = v.shape[-1]

    s_sum = torch.zeros((n, v_dim, k_dim), device=q.device, dtype=torch.float32)
    k_sum = torch.zeros((n, k_dim), device=q.device, dtype=torch.float32)
    o_prev = torch.zeros((n, v_dim), device=q.device, dtype=torch.float32)
    q_prev = torch.zeros((n, k_dim), device=q.device, dtype=torch.float32)
    count = torch.zeros((n,), device=q.device, dtype=torch.float32)

    out = torch.empty((n, t, v_dim), device=q.device, dtype=torch.float32)
    for i in range(t):
        q_i = qf[:, i]
        k_i = kf[:, i]
        v_i = vf[:, i]

        valid_hist = count > 0
        o_hist = torch.zeros_like(o_prev)
        beta = torch.ones_like(count)
        if valid_hist.any():
            inv_count = torch.zeros_like(count)
            inv_count[valid_hist] = count[valid_hist].reciprocal()
            s_mean = s_sum * inv_count[:, None, None]
            k_mean = k_sum * inv_count[:, None]
            dq = q_i - q_prev
            hist_delta = torch.einsum("nvk,nk->nv", s_mean, dq)
            hist_delta = hist_delta - o_prev * torch.sum(k_mean * dq, dim=-1, keepdim=True)
            o_hist = o_prev + hist_delta

            hist_mass = torch.sum(k_sum * q_i, dim=-1)
            hist_log = torch.log(torch.nn.functional.softplus(hist_mass) + hist_eps)
            score = torch.sum(k_i * q_i, dim=-1).clamp(min=-score_clip, max=score_clip)
            beta_hist = torch.sigmoid(score - hist_log)
            beta = torch.where(valid_hist, beta_hist, beta)

        o_i = (1.0 - beta[:, None]) * o_hist + beta[:, None] * v_i
        out[:, i] = o_i

        s_sum = s_sum + torch.einsum("nv,nk->nvk", v_i, k_i)
        k_sum = k_sum + k_i
        o_prev = o_i
        q_prev = q_i
        count = count + 1.0

    return out.to(q.dtype)


def _bwd_torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    hist_eps: float,
    score_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        o_ = _torch_recompute_forward(
            q=q_,
            k=k_,
            v=v_,
            hist_eps=hist_eps,
            score_clip=score_clip,
        )
        dq, dk, dv = torch.autograd.grad(o_, (q_, k_, v_), do)
    return dq, dk, dv


if _TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=["T"])
    def _pdf_final_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        ssum_ptr,
        ksum_ptr,
        oprev_ptr,
        qprev_ptr,
        count_ptr,
        T,
        HIST_EPS,
        SCORE_CLIP,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        n = tl.program_id(0).to(tl.int64)

        offs_k = tl.arange(0, BK)
        mask_k = offs_k < K

        p_ssum_n = ssum_ptr + n * V * K
        p_ksum_n = ksum_ptr + n * K
        p_oprev_n = oprev_ptr + n * V
        p_qprev_n = qprev_ptr + n * K
        p_count_n = count_ptr + n

        b_ksum = tl.load(p_ksum_n + offs_k, mask=mask_k, other=0).to(tl.float32)
        b_qprev = tl.load(p_qprev_n + offs_k, mask=mask_k, other=0).to(tl.float32)
        count = tl.load(p_count_n).to(tl.float32)

        for t in range(0, T):
            q_base = (n * T + t) * K
            v_base = (n * T + t) * V

            b_q = tl.load(q_ptr + q_base + offs_k, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(k_ptr + q_base + offs_k, mask=mask_k, other=0).to(tl.float32)
            b_dq = b_q - b_qprev

            valid_hist = count > 0
            inv_count = tl.where(valid_hist, 1.0 / count, 0.0)
            hist_dot_dq = inv_count * tl.sum(b_ksum * b_dq, axis=0)

            hist_mass = tl.sum(b_ksum * b_q, axis=0)
            hist_abs = tl.abs(hist_mass)
            hist_softplus = tl.maximum(hist_mass, 0.0) + tl.log(1.0 + tl.exp(-hist_abs))
            hist_log = tl.log(hist_softplus + HIST_EPS)

            score = tl.sum(b_k * b_q, axis=0)
            score = tl.minimum(tl.maximum(score, -SCORE_CLIP), SCORE_CLIP)
            beta_hist = tl.sigmoid(score - hist_log)
            beta = tl.where(valid_hist, beta_hist, 1.0)

            for v0 in range(0, V, BV):
                offs_v = v0 + tl.arange(0, BV)
                mask_v = offs_v < V
                mask_vk = mask_v[:, None] & mask_k[None, :]

                p_ssum = p_ssum_n + offs_v[:, None] * K + offs_k[None, :]
                p_oprev = p_oprev_n + offs_v

                b_ssum = tl.load(p_ssum, mask=mask_vk, other=0).to(tl.float32)
                b_oprev = tl.load(p_oprev, mask=mask_v, other=0).to(tl.float32)
                b_v = tl.load(v_ptr + v_base + offs_v, mask=mask_v, other=0).to(tl.float32)

                hist_delta = inv_count * tl.sum(b_ssum * b_dq[None, :], axis=1)
                hist_delta = hist_delta - b_oprev * hist_dot_dq
                o_hist = b_oprev + hist_delta
                o_t = (1.0 - beta) * o_hist + beta * b_v

                tl.store(o_ptr + v_base + offs_v, o_t.to(o_ptr.dtype.element_ty), mask=mask_v)
                tl.store(p_oprev, o_t.to(oprev_ptr.dtype.element_ty), mask=mask_v)

                b_ssum = b_ssum + b_v[:, None] * b_k[None, :]
                tl.store(p_ssum, b_ssum.to(ssum_ptr.dtype.element_ty), mask=mask_vk)

            b_ksum = b_ksum + b_k
            b_qprev = b_q
            count = count + 1.0

        tl.store(p_ksum_n + offs_k, b_ksum.to(ksum_ptr.dtype.element_ty), mask=mask_k)
        tl.store(p_qprev_n + offs_k, b_qprev.to(qprev_ptr.dtype.element_ty), mask=mask_k)
        tl.store(p_count_n, count.to(count_ptr.dtype.element_ty))


def _fwd_triton_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    hist_eps: float,
    score_clip: float,
) -> torch.Tensor:
    n, t, k_dim = q.shape
    v_dim = v.shape[-1]

    s_sum = torch.zeros((n, v_dim, k_dim), device=q.device, dtype=torch.float32)
    k_sum = torch.zeros((n, k_dim), device=q.device, dtype=torch.float32)
    o_prev = torch.zeros((n, v_dim), device=q.device, dtype=torch.float32)
    q_prev = torch.zeros((n, k_dim), device=q.device, dtype=torch.float32)
    count = torch.zeros((n,), device=q.device, dtype=torch.float32)
    out = torch.empty((n, t, v_dim), device=q.device, dtype=q.dtype)

    bk = triton.next_power_of_2(k_dim)
    if bk > 256:
        raise ValueError(f"Unsupported K={k_dim}. This kernel supports K <= 256.")
    bv = min(64, triton.next_power_of_2(v_dim))

    _pdf_final_fwd_kernel[(n,)](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        o_ptr=out,
        ssum_ptr=s_sum,
        ksum_ptr=k_sum,
        oprev_ptr=o_prev,
        qprev_ptr=q_prev,
        count_ptr=count,
        T=t,
        HIST_EPS=float(hist_eps),
        SCORE_CLIP=float(score_clip),
        K=k_dim,
        V=v_dim,
        BK=bk,
        BV=bv,
        num_warps=4,
        num_stages=2,
    )
    return out


class _PDFFinalDenseTritonFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hist_eps: float,
        score_clip: float,
    ) -> torch.Tensor:
        out = _fwd_triton_dense(q.contiguous(), k.contiguous(), v.contiguous(), hist_eps, score_clip)
        ctx.save_for_backward(q, k, v)
        ctx.hist_eps = float(hist_eps)
        ctx.score_clip = float(score_clip)
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v = ctx.saved_tensors
        dq, dk, dv = _bwd_torch_reference(
            q=q,
            k=k,
            v=v,
            do=do.contiguous(),
            hist_eps=ctx.hist_eps,
            score_clip=ctx.score_clip,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def pdf_final_linear_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    hist_eps: float = 1e-4,
    score_clip: float = 20.0,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, dict[str, float]]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape [B, T, H, D].")
    if initial_state is not None or output_final_state:
        raise RuntimeError("Triton PDF-final kernel currently supports training forward only (no cache/final state).")
    if not _TRITON_AVAILABLE:
        raise RuntimeError("PDF-final Triton kernel is not available in current environment.")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("PDF-final Triton kernel requires CUDA tensors.")

    batch, seqlen, n_heads, k_dim = q.shape
    v_dim = v.shape[-1]
    qf = q.permute(0, 2, 1, 3).contiguous().reshape(batch * n_heads, seqlen, k_dim)
    kf = k.permute(0, 2, 1, 3).contiguous().reshape(batch * n_heads, seqlen, k_dim)
    vf = v.permute(0, 2, 1, 3).contiguous().reshape(batch * n_heads, seqlen, v_dim)

    if cu_seqlens is None:
        out_flat = _PDFFinalDenseTritonFunction.apply(qf, kf, vf, hist_eps, score_clip)
    else:
        if batch != 1:
            raise ValueError("When cu_seqlens is provided, expected flattened varlen input with batch=1.")
        cu = cu_seqlens.tolist()
        parts = []
        for i in range(len(cu) - 1):
            bos, eos = int(cu[i]), int(cu[i + 1])
            if eos > bos:
                parts.append(
                    _PDFFinalDenseTritonFunction.apply(
                        qf[:, bos:eos, :],
                        kf[:, bos:eos, :],
                        vf[:, bos:eos, :],
                        hist_eps,
                        score_clip,
                    ),
                )
        out_flat = torch.cat(parts, dim=1) if parts else torch.empty_like(vf)

    out = out_flat.reshape(batch, n_heads, seqlen, v_dim).permute(0, 2, 1, 3).contiguous()
    return out, None, {}


__all__ = ["pdf_final_linear_attention_triton", "_TRITON_AVAILABLE"]
