from __future__ import annotations

import torch

try:
    from fla.ops.linear_attn import fused_recurrent_linear_attn

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - runtime fallback
    fused_recurrent_linear_attn = None
    _TRITON_AVAILABLE = False


def _segmentwise_cumsum(x: torch.Tensor, cu_seqlens: torch.LongTensor) -> torch.Tensor:
    # x: [1, T, H, D]
    if x.shape[0] != 1:
        raise ValueError("Segmentwise cumsum expects flattened varlen inputs with batch size 1.")
    out = torch.empty_like(x)
    cu = cu_seqlens.tolist()
    for i in range(len(cu) - 1):
        start, end = int(cu[i]), int(cu[i + 1])
        if end > start:
            out[:, start:end] = x[:, start:end].cumsum(dim=1)
    return out


def _segmentwise_last(
    x_cumsum: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    initial_state_k: torch.Tensor | None,
) -> torch.Tensor:
    # x_cumsum: [1, T, H, D]
    cu = cu_seqlens.tolist()
    n_seq = len(cu) - 1
    finals = []
    for i in range(n_seq):
        start, end = int(cu[i]), int(cu[i + 1])
        if end > start:
            finals.append(x_cumsum[:, end - 1].squeeze(0))
        else:
            if initial_state_k is None:
                finals.append(torch.zeros_like(x_cumsum[:, 0].squeeze(0)))
            else:
                finals.append(initial_state_k[i])
    return torch.stack(finals, dim=0)


def performer_causal_linear_attention_triton(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """
    Triton-backed causal Performer attention.
    Numerator uses FLA fused recurrent Triton kernel.
    Denominator and k-prefix state are computed with differentiable torch ops.
    """
    if q_prime.ndim != 4 or k_prime.ndim != 4 or v.ndim != 4:
        raise ValueError("q_prime, k_prime, v must have shape [B, T, H, D].")
    if q_prime.shape[:3] != k_prime.shape[:3] or q_prime.shape[:3] != v.shape[:3]:
        raise ValueError("Leading dimensions of q_prime, k_prime, v must match.")

    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton performer kernel is not available in current environment.")
    if not (q_prime.is_cuda and k_prime.is_cuda and v.is_cuda):
        raise RuntimeError("Triton performer kernel requires CUDA tensors.")

    q_prime = q_prime.contiguous()
    k_prime = k_prime.contiguous()
    v = v.contiguous()

    initial_state_kv = None
    initial_state_k = None
    if initial_state is not None:
        if len(initial_state) != 2:
            raise ValueError("initial_state must be (kv_state, k_state).")
        initial_state_kv = initial_state[0].contiguous()
        initial_state_k = initial_state[1].contiguous().to(torch.float32)

    # Triton numerator: sum_{i<=t} q_t^T k_i * v_i
    num, kv_final = fused_recurrent_linear_attn(
        q=q_prime,
        k=k_prime,
        v=v,
        scale=1.0,
        initial_state=initial_state_kv,
        output_final_state=output_final_state,
        normalize=False,
        cu_seqlens=cu_seqlens,
    )

    # Denominator: sum_{i<=t} q_t^T k_i, with support for initial k-state.
    if cu_seqlens is None:
        k_prefix = k_prime.to(torch.float32).cumsum(dim=1)
        if initial_state_k is not None:
            k_prefix = k_prefix + initial_state_k.unsqueeze(1)
        k_final = k_prefix[:, -1]
    else:
        k_prefix = _segmentwise_cumsum(k_prime.to(torch.float32), cu_seqlens)
        if initial_state_k is not None:
            cu = cu_seqlens.tolist()
            for i in range(len(cu) - 1):
                start, end = int(cu[i]), int(cu[i + 1])
                if end > start:
                    k_prefix[:, start:end] = k_prefix[:, start:end] + initial_state_k[i:i + 1].unsqueeze(1)
        k_final = _segmentwise_last(k_prefix, cu_seqlens, initial_state_k)

    den = (q_prime.to(torch.float32) * k_prefix).sum(dim=-1, keepdim=True)
    out = num.to(torch.float32) / (den + eps)

    final_state = (kv_final, k_final) if output_final_state else None
    return out, final_state


__all__ = ['performer_causal_linear_attention_triton', '_TRITON_AVAILABLE']
