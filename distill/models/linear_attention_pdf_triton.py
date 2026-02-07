from __future__ import annotations

import torch

try:
    from .linear_attention_pdf_triton_kernels import triton_first_order_linear_attention

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - runtime fallback
    triton_first_order_linear_attention = None
    _TRITON_AVAILABLE = False


_EPS = 1e-6


def _prepare_initial_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, _, k_dim = q.shape
    v_dim = v.shape[-1]
    device = q.device
    dtype = torch.float32

    if initial_state is None:
        a = torch.zeros((n, v_dim, k_dim), device=device, dtype=dtype)
        b = torch.zeros((n, k_dim), device=device, dtype=dtype)
        s = torch.zeros((n, v_dim), device=device, dtype=dtype)
        z = torch.zeros((n,), device=device, dtype=dtype)
        m = torch.full((n,), -float('inf'), device=device, dtype=dtype)
        q_prev = torch.zeros((n, k_dim), device=device, dtype=dtype)
        return a, b, s, z, m, q_prev

    if len(initial_state) != 6:
        raise ValueError("initial_state must be a 6-tuple: (A, b, s, z, m, q_prev).")

    a, b, s, z, m, q_prev = initial_state
    a = a.reshape(n, v_dim, k_dim).to(device=device, dtype=dtype).contiguous()
    b = b.reshape(n, k_dim).to(device=device, dtype=dtype).contiguous()
    s = s.reshape(n, v_dim).to(device=device, dtype=dtype).contiguous()
    z = z.reshape(n).to(device=device, dtype=dtype).contiguous()
    m = m.reshape(n).to(device=device, dtype=dtype).contiguous()
    q_prev = q_prev.reshape(n, k_dim).to(device=device, dtype=dtype).contiguous()
    return a, b, s, z, m, q_prev


def _torch_first_order_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    n, t, _ = q.shape
    qf = q.float()
    kf = k.float()
    vf = v.float()

    a, b, s, z, m, q_prev = _prepare_initial_state(qf, kf, vf, initial_state)

    o = torch.empty((n, t, vf.shape[-1]), device=q.device, dtype=torch.float32)
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

    final_state = (a, b, s, z, m, q_prev) if output_final_state else None
    return o.to(q.dtype), final_state


def _run_dense(
    q_dense: torch.Tensor,
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    init_dense: tuple[torch.Tensor, ...] | None,
    need_final: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    q_dense = q_dense.contiguous()
    k_dense = k_dense.contiguous()
    v_dense = v_dense.contiguous()

    if (not _TRITON_AVAILABLE) or (not q_dense.is_cuda):
        return _torch_first_order_linear_attention(
            q_dense,
            k_dense,
            v_dense,
            initial_state=init_dense,
            output_final_state=need_final,
        )

    a0, b0, s0, z0, m0, q0 = _prepare_initial_state(q_dense, k_dense, v_dense, init_dense)
    return triton_first_order_linear_attention(
        q_dense,
        k_dense,
        v_dense,
        a0,
        b0,
        s0,
        z0,
        m0,
        q0,
        need_final,
    )


def _flatten_initial_state(
    initial_state: tuple[torch.Tensor, ...] | None,
    batch: int,
    heads: int,
    v_dim: int,
    k_dim: int,
) -> tuple[torch.Tensor, ...] | None:
    if initial_state is None:
        return None
    a0, b0, s0, z0, m0, q0 = initial_state
    return (
        a0.reshape(batch * heads, v_dim, k_dim).contiguous(),
        b0.reshape(batch * heads, k_dim).contiguous(),
        s0.reshape(batch * heads, v_dim).contiguous(),
        z0.reshape(batch * heads).contiguous(),
        m0.reshape(batch * heads).contiguous(),
        q0.reshape(batch * heads, k_dim).contiguous(),
    )


def _reshape_final_state(
    final_state: tuple[torch.Tensor, ...],
    out_batch: int,
    n_heads: int,
    v_dim: int,
    k_dim: int,
) -> tuple[torch.Tensor, ...]:
    a, b_, s_, z_, m_, q_prev_ = final_state
    return (
        a.reshape(out_batch, n_heads, v_dim, k_dim).contiguous(),
        b_.reshape(out_batch, n_heads, k_dim).contiguous(),
        s_.reshape(out_batch, n_heads, v_dim).contiguous(),
        z_.reshape(out_batch, n_heads).contiguous(),
        m_.reshape(out_batch, n_heads).contiguous(),
        q_prev_.reshape(out_batch, n_heads, k_dim).contiguous(),
    )


def _run_varlen(
    qf: torch.Tensor,
    kf: torch.Tensor,
    vf: torch.Tensor,
    n_heads: int,
    k_dim: int,
    v_dim: int,
    cu_seqlens: torch.LongTensor,
    initial_state: tuple[torch.Tensor, ...] | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    n_seq = cu_seqlens.numel() - 1
    cu = cu_seqlens.tolist()

    if initial_state is not None:
        a0, b0, s0, z0, m0, q0 = initial_state
        init_flat = (
            a0.reshape(n_seq, n_heads, v_dim, k_dim).contiguous(),
            b0.reshape(n_seq, n_heads, k_dim).contiguous(),
            s0.reshape(n_seq, n_heads, v_dim).contiguous(),
            z0.reshape(n_seq, n_heads).contiguous(),
            m0.reshape(n_seq, n_heads).contiguous(),
            q0.reshape(n_seq, n_heads, k_dim).contiguous(),
        )
    else:
        init_flat = None

    o = torch.empty_like(vf)
    final_chunks = [[], [], [], [], [], []] if output_final_state else None

    for i in range(n_seq):
        bos, eos = int(cu[i]), int(cu[i + 1])
        seg_len = eos - bos
        if seg_len < 0:
            raise ValueError("`cu_seqlens` must be non-decreasing.")

        if init_flat is None:
            init_seg = None
        else:
            init_seg = (
                init_flat[0][i].contiguous(),
                init_flat[1][i].contiguous(),
                init_flat[2][i].contiguous(),
                init_flat[3][i].contiguous(),
                init_flat[4][i].contiguous(),
                init_flat[5][i].contiguous(),
            )

        if seg_len == 0:
            if output_final_state:
                if init_seg is None:
                    zeros_a = torch.zeros((n_heads, v_dim, k_dim), device=qf.device, dtype=torch.float32)
                    zeros_b = torch.zeros((n_heads, k_dim), device=qf.device, dtype=torch.float32)
                    zeros_s = torch.zeros((n_heads, v_dim), device=qf.device, dtype=torch.float32)
                    zeros_z = torch.zeros((n_heads,), device=qf.device, dtype=torch.float32)
                    zeros_m = torch.full((n_heads,), -float('inf'), device=qf.device, dtype=torch.float32)
                    zeros_q = torch.zeros((n_heads, k_dim), device=qf.device, dtype=torch.float32)
                    end_seg = (zeros_a, zeros_b, zeros_s, zeros_z, zeros_m, zeros_q)
                else:
                    end_seg = init_seg
                for idx in range(6):
                    final_chunks[idx].append(end_seg[idx])
            continue

        o_seg, st_seg = _run_dense(
            qf[:, bos:eos, :],
            kf[:, bos:eos, :],
            vf[:, bos:eos, :],
            init_seg,
            output_final_state,
        )
        o[:, bos:eos, :] = o_seg

        if output_final_state:
            for idx in range(6):
                final_chunks[idx].append(st_seg[idx])

    if output_final_state:
        final_state = tuple(torch.stack(chunks, dim=0) for chunks in final_chunks)
    else:
        final_state = None

    return o, final_state


def first_order_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape [B, T, H, D].")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, v must share [B, T, H] dimensions.")

    bsz, seq_len, n_heads, k_dim = q.shape
    v_dim = v.shape[-1]

    qf = q.permute(0, 2, 1, 3).contiguous().reshape(bsz * n_heads, seq_len, k_dim)
    kf = k.permute(0, 2, 1, 3).contiguous().reshape(bsz * n_heads, seq_len, k_dim)
    vf = v.permute(0, 2, 1, 3).contiguous().reshape(bsz * n_heads, seq_len, v_dim)

    if cu_seqlens is None:
        init_flat = _flatten_initial_state(initial_state, bsz, n_heads, v_dim, k_dim)
        o, final_state = _run_dense(qf, kf, vf, init_flat, output_final_state)
        out_batch = bsz
    else:
        if bsz != 1:
            raise ValueError(f"Expected batch size 1 for variable-length mode, but got {bsz}.")
        if cu_seqlens.ndim != 1:
            raise ValueError("`cu_seqlens` must be a 1D tensor of cumulative sequence offsets.")
        if int(cu_seqlens[0].item()) != 0 or int(cu_seqlens[-1].item()) != seq_len:
            raise ValueError("`cu_seqlens` must start with 0 and end with total token length.")

        o, final_state = _run_varlen(
            qf=qf,
            kf=kf,
            vf=vf,
            n_heads=n_heads,
            k_dim=k_dim,
            v_dim=v_dim,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        out_batch = cu_seqlens.numel() - 1

    o = o.reshape(bsz, n_heads, seq_len, v_dim).permute(0, 2, 1, 3).contiguous()
    if final_state is None:
        return o, None

    return o, _reshape_final_state(final_state, out_batch, n_heads, v_dim, k_dim)


__all__ = [
    'first_order_linear_attention',
]
