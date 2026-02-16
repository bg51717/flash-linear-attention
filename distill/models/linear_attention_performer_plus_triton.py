from __future__ import annotations

import torch

try:
    from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - runtime fallback
    fused_recurrent_simple_gla = None
    _TRITON_AVAILABLE = False


def performer_plus_causal_linear_attention_triton(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """
    Triton-backed causal Performer+ operator.

    Recurrence:
        S_t = lambda_t * S_{t-1} + k'_t v_t^T
        z_t = lambda_t * z_{t-1} + k'_t
        o_t = (q'_t^T S_t) / (q'_t^T z_t + eps)
    where lambda_t = exp(log_decay_t) if `log_decay` is provided, otherwise 1.

    This function launches Triton recurrent kernels for both numerator and
    denominator recurrences via FLA fused recurrent simple GLA op.
    """
    if q_prime.ndim != 4 or k_prime.ndim != 4 or v.ndim != 4:
        raise ValueError("q_prime, k_prime, v must have shape [B, T, H, D].")
    if q_prime.shape[:3] != k_prime.shape[:3] or q_prime.shape[:3] != v.shape[:3]:
        raise ValueError("Leading dimensions of q_prime, k_prime, v must match.")
    if log_decay is not None:
        if log_decay.ndim == 4 and log_decay.shape[-1] == 1:
            log_decay = log_decay.squeeze(-1)
        if log_decay.ndim != 3:
            raise ValueError("log_decay must have shape [B, T, H] or [B, T, H, 1].")
        if log_decay.shape != q_prime.shape[:3]:
            raise ValueError("log_decay shape must match q_prime[:, :, :, 0].")

    if not _TRITON_AVAILABLE:
        raise RuntimeError("Performer+ Triton operator is not available in current environment.")
    if not (q_prime.is_cuda and k_prime.is_cuda and v.is_cuda):
        raise RuntimeError("Performer+ Triton operator requires CUDA tensors.")

    q_prime = q_prime.contiguous()
    k_prime = k_prime.contiguous()
    v = v.contiguous()
    if log_decay is not None:
        log_decay = log_decay.contiguous().to(torch.float32)

    initial_state_kv = None
    initial_state_k = None
    if initial_state is not None:
        if len(initial_state) != 2:
            raise ValueError("initial_state must be (kv_state, k_state).")
        initial_state_kv = initial_state[0].contiguous()
        initial_state_k = initial_state[1].contiguous().to(torch.float32)
        if initial_state_k.ndim != 3:
            raise ValueError("initial_state k_state must have shape [N, H, M].")
        if initial_state_kv.ndim != 4:
            raise ValueError("initial_state kv_state must have shape [N, H, M, V].")
        if initial_state_kv.shape[:3] != initial_state_k.shape:
            raise ValueError("initial_state kv_state and k_state leading dims must match.")

    # Numerator recurrence: q'^T * S_t
    num, kv_final = fused_recurrent_simple_gla(
        q=q_prime,
        k=k_prime,
        v=v,
        g=log_decay,
        scale=1.0,
        initial_state=initial_state_kv,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    # Denominator recurrence: q'^T * z_t, implemented as value-dim=1 recurrence.
    ones = torch.ones((*v.shape[:3], 1), device=v.device, dtype=v.dtype)
    den_init = None if initial_state_k is None else initial_state_k.unsqueeze(-1)
    den, z_final = fused_recurrent_simple_gla(
        q=q_prime,
        k=k_prime,
        v=ones,
        g=log_decay,
        scale=1.0,
        initial_state=den_init,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    out = num.to(torch.float32) / (den.to(torch.float32) + eps)
    final_state = None
    if output_final_state:
        final_state = (kv_final, z_final.squeeze(-1).to(torch.float32))
    return out, final_state


__all__ = ['performer_plus_causal_linear_attention_triton', '_TRITON_AVAILABLE']
