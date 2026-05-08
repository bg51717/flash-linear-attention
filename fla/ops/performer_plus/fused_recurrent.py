from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from fla.ops.delta_rule import fused_recurrent_delta_rule
    from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule
    from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - runtime fallback
    fused_recurrent_delta_rule = None
    chunk_dplr_delta_rule = None
    fused_recurrent_simple_gla = None
    _TRITON_AVAILABLE = False


def _to_kernel_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != dtype:
        x = x.to(dtype)
    return x


def performer_plus_pdf_delta_attention_triton(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    rho: torch.Tensor | None = None,
    q_prime_den: torch.Tensor | None = None,
    k_prime_den: torch.Tensor | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    eps: float = 1e-6,
    denom_eps: float = 1e-6,
    denom_stopgrad: bool = True,
    forget_norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """
    Triton implementation for rho-Delta Performer+ update:
        W_t = W_{t-1}
              - (1-rho_t) * (W_{t-1} phi_t phi_t^T / ||phi_t||^2)
              + v_t phi_t^T
        z_t = z_{t-1} + phi_t
        o_t = (W_t psi_t) / (z_t^T psi_t + eps)

    We realize this with two fused Triton passes:
      1) Numerator recurrence via generalized DPLR delta-rule:
           S_t = S_{t-1} @ (I + a_t b_t^T) + v_t k_t^T
         with `a_t = -beta_t * phi_t`, `b_t = phi_t`, `k_t = phi_t`,
         `beta_t = (1-rho_t)/(||phi_t||^2 + forget_norm_eps)`.
      2) Denominator `z_t^T psi_t` via simple additive recurrence (`z_t = z_{t-1}+phi_t`).
    """
    if q_prime.ndim != 4 or k_prime.ndim != 4 or v.ndim != 4:
        raise ValueError("q_prime, k_prime, v must have shape [B, T, H, D].")
    if q_prime.shape[:3] != k_prime.shape[:3] or q_prime.shape[:3] != v.shape[:3]:
        raise ValueError("Leading dimensions of q_prime, k_prime, v must match.")
    if q_prime_den is None:
        q_prime_den = q_prime
    if k_prime_den is None:
        k_prime_den = k_prime
    if q_prime_den.ndim != 4 or k_prime_den.ndim != 4:
        raise ValueError("q_prime_den and k_prime_den must have shape [B, T, H, D].")
    if q_prime_den.shape[:3] != q_prime.shape[:3] or k_prime_den.shape[:3] != q_prime.shape[:3]:
        raise ValueError("Leading dimensions of denominator maps must match q_prime.")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}.")
    if denom_eps <= 0:
        raise ValueError(f"denom_eps must be > 0, got {denom_eps}.")
    if forget_norm_eps <= 0:
        raise ValueError(f"forget_norm_eps must be > 0, got {forget_norm_eps}.")
    if chunk_dplr_delta_rule is None or fused_recurrent_simple_gla is None:
        raise RuntimeError("PDF delta Triton operator requires chunk_dplr_delta_rule and fused_recurrent_simple_gla.")
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Performer+ Triton operator is not available in current environment.")
    if not (q_prime.is_cuda and k_prime.is_cuda and q_prime_den.is_cuda and k_prime_den.is_cuda and v.is_cuda):
        raise RuntimeError("Performer+ Triton operator requires CUDA tensors.")

    kernel_dtype = v.dtype
    if kernel_dtype == torch.float32:
        kernel_dtype = torch.bfloat16
    q_num_kernel = _to_kernel_dtype(q_prime, kernel_dtype)
    k_num_kernel = _to_kernel_dtype(k_prime, kernel_dtype)
    q_den_kernel = _to_kernel_dtype(q_prime_den, kernel_dtype)
    k_den_kernel = _to_kernel_dtype(k_prime_den, kernel_dtype)
    v_kernel = _to_kernel_dtype(v, kernel_dtype)
    if rho is None:
        rho_kernel = torch.zeros(
            q_prime.shape[:3],
            device=q_prime.device,
            dtype=torch.float32,
        )
    else:
        if rho.ndim != 3:
            raise ValueError("rho must have shape [B, T, H].")
        if rho.shape != q_prime.shape[:3]:
            raise ValueError("rho shape must match q_prime[:, :, :, 0].")
        rho_kernel = rho.float().clamp(0.0, 1.0).contiguous()

    initial_state_kv = None
    initial_state_k = None
    if initial_state is not None:
        if len(initial_state) != 2:
            raise ValueError("initial_state must be (kv_state, k_state).")
        initial_state_kv = initial_state[0].contiguous()
        initial_state_k = initial_state[1].contiguous().to(torch.float32)
        if initial_state_kv.ndim != 4:
            raise ValueError("initial_state kv_state must have shape [N, H, M, V].")
        if initial_state_k.ndim != 3:
            raise ValueError("initial_state k_state must have shape [N, H, M].")
        if initial_state_kv.shape[:2] != initial_state_k.shape[:2]:
            raise ValueError("initial_state leading [N, H] dims must match.")
        if initial_state_kv.shape[2] != k_prime.shape[-1]:
            raise ValueError("initial_state kv_state feature dim must match k_prime.")
        if initial_state_k.shape[2] != k_prime_den.shape[-1]:
            raise ValueError("initial_state k_state feature dim must match k_prime_den.")
        if initial_state_kv.dtype != kernel_dtype:
            initial_state_kv = initial_state_kv.to(kernel_dtype)

    # Pass 1: numerator update via generalized DPLR delta recurrence.
    k_norm_sq = k_num_kernel.float().square().sum(dim=-1).clamp_min(float(forget_norm_eps))
    beta_pdf = ((1.0 - rho_kernel) / k_norm_sq).to(torch.float32)
    a = -(beta_pdf.unsqueeze(-1).to(k_num_kernel.dtype) * k_num_kernel)
    b = k_num_kernel
    gk = torch.zeros_like(k_num_kernel)
    num, kv_final = chunk_dplr_delta_rule(
        q=q_num_kernel,
        k=k_num_kernel,
        v=v_kernel,
        a=a,
        b=b,
        gk=gk,
        scale=1.0,
        initial_state=initial_state_kv,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    # Pass 2: denominator z_t^T psi_t with z_t = z_{t-1} + phi_t.
    ones = torch.ones((*v_kernel.shape[:3], 1), device=v_kernel.device, dtype=kernel_dtype)
    den_q_out = q_den_kernel.detach() if denom_stopgrad else q_den_kernel
    den_k_out = k_den_kernel.detach() if denom_stopgrad else k_den_kernel
    den_init = None if initial_state_k is None else initial_state_k.unsqueeze(-1).to(kernel_dtype)
    den_state_out = den_init.detach() if (denom_stopgrad and den_init is not None) else den_init
    den_out, z_final = fused_recurrent_simple_gla(
        q=den_q_out,
        k=den_k_out,
        v=ones,
        g=None,
        scale=1.0,
        initial_state=den_state_out,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    den_used = den_out.detach() if denom_stopgrad else den_out
    # Keep the final ratio in kernel dtype (bf16/fp16) to reduce peak memory
    # and avoid materializing an extra fp32-sized activation tensor.
    out = num / den_used.clamp_min(float(eps))
    final_state = None
    if output_final_state:
        final_state = (kv_final, z_final.squeeze(-1).to(torch.float32))
    return out, final_state


def performer_plus_causal_linear_attention_triton(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    q_prime_den: torch.Tensor | None = None,
    k_prime_den: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    beta_den: torch.Tensor | None = None,
    log_decay: torch.Tensor | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    update_rule: str = "sum",
    eps: float = 1e-6,
    delta_denom_eps: float = 1e-3,
    delta_smooth_denom: bool = False,
    delta_denom_tau: float = 1e-2,
    delta_log_decay: torch.Tensor | None = None,
    delta_denominator_update: str = "delta",
    delta_denominator_stopgrad: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """
    Triton-backed causal Performer+ operator.

    Recurrence (`update_rule='sum'`):
        S_t = lambda_t * S_{t-1} + k'_t v_t^T
        z_t = lambda_t * z_{t-1} + k'_t
        o_t = (q'_t^T S_t) / (q'_t^T z_t + eps)
    where lambda_t = exp(log_decay_t) if `log_decay` is provided, otherwise 1.

    Recurrence (`update_rule='delta'`):
        S_t = S_{t-1} + beta_t * (v_t - S_{t-1} k'_t) k'_t^T
        z_t = z_{t-1} + beta_t * (1 - z_{t-1} · k'_t) k'_t
        o_t = (q'_t^T S_t) / (q'_t^T z_t + eps)

    `q_prime_den` and `k_prime_den` optionally define a separate feature map for
    denominator recurrence. When omitted, numerator maps are reused.

    This function launches Triton recurrent kernels for both numerator and
    denominator recurrences via FLA fused recurrent operators.
    For `delta` update, denominator is sign-preserving clamped with
    `delta_denom_eps` for stability.
    """
    if q_prime.ndim != 4 or k_prime.ndim != 4 or v.ndim != 4:
        raise ValueError("q_prime, k_prime, v must have shape [B, T, H, D].")
    if q_prime.shape[:3] != k_prime.shape[:3] or q_prime.shape[:3] != v.shape[:3]:
        raise ValueError("Leading dimensions of q_prime, k_prime, v must match.")
    if q_prime_den is None:
        q_prime_den = q_prime
    if k_prime_den is None:
        k_prime_den = k_prime
    if q_prime_den.ndim != 4 or k_prime_den.ndim != 4:
        raise ValueError("q_prime_den and k_prime_den must have shape [B, T, H, D].")
    if q_prime_den.shape[:3] != q_prime.shape[:3] or k_prime_den.shape[:3] != q_prime.shape[:3]:
        raise ValueError("Leading dimensions of denominator maps must match q_prime.")
    if update_rule not in ("sum", "delta"):
        raise ValueError(f"Unsupported update_rule={update_rule}. Expected one of ('sum', 'delta').")
    if delta_denominator_update not in ("delta", "sum"):
        raise ValueError(
            f"Unsupported delta_denominator_update={delta_denominator_update}. Expected one of ('delta', 'sum').",
        )
    if delta_denom_eps <= 0:
        raise ValueError(f"delta_denom_eps must be > 0, got {delta_denom_eps}.")
    if delta_smooth_denom and delta_denom_tau <= 0:
        raise ValueError(f"delta_denom_tau must be > 0 when delta_smooth_denom=True, got {delta_denom_tau}.")
    if log_decay is not None:
        if log_decay.ndim == 4 and log_decay.shape[-1] == 1:
            log_decay = log_decay.squeeze(-1)
        if log_decay.ndim != 3:
            raise ValueError("log_decay must have shape [B, T, H] or [B, T, H, 1].")
        if log_decay.shape != q_prime.shape[:3]:
            raise ValueError("log_decay shape must match q_prime[:, :, :, 0].")
    if delta_log_decay is not None:
        if delta_log_decay.ndim == 4 and delta_log_decay.shape[-1] == 1:
            delta_log_decay = delta_log_decay.squeeze(-1)
        if delta_log_decay.ndim != 3:
            raise ValueError("delta_log_decay must have shape [B, T, H] or [B, T, H, 1].")
        if delta_log_decay.shape != q_prime.shape[:3]:
            raise ValueError("delta_log_decay shape must match q_prime[:, :, :, 0].")

    if not _TRITON_AVAILABLE:
        raise RuntimeError("Performer+ Triton operator is not available in current environment.")
    if not (q_prime.is_cuda and k_prime.is_cuda and q_prime_den.is_cuda and k_prime_den.is_cuda and v.is_cuda):
        raise RuntimeError("Performer+ Triton operator requires CUDA tensors.")

    q_prime = q_prime.contiguous()
    k_prime = k_prime.contiguous()
    q_prime_den = q_prime_den.contiguous()
    k_prime_den = k_prime_den.contiguous()
    v = v.contiguous()
    if beta is not None:
        if beta.ndim == 4 and beta.shape[-1] == 1:
            beta = beta.squeeze(-1)
        if beta.ndim != 3:
            raise ValueError("beta must have shape [B, T, H] or [B, T, H, 1].")
        if beta.shape != q_prime.shape[:3]:
            raise ValueError("beta shape must match q_prime[:, :, :, 0].")
        beta = beta.contiguous().to(torch.float32)
    if beta_den is not None:
        if beta_den.ndim == 4 and beta_den.shape[-1] == 1:
            beta_den = beta_den.squeeze(-1)
        if beta_den.ndim != 3:
            raise ValueError("beta_den must have shape [B, T, H] or [B, T, H, 1].")
        if beta_den.shape != q_prime.shape[:3]:
            raise ValueError("beta_den shape must match q_prime[:, :, :, 0].")
        beta_den = beta_den.contiguous().to(torch.float32)
    if log_decay is not None:
        log_decay = log_decay.contiguous().to(torch.float32)
    if delta_log_decay is not None:
        delta_log_decay = delta_log_decay.contiguous().to(torch.float32)

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
        if initial_state_kv.shape[:2] != initial_state_k.shape[:2]:
            raise ValueError("initial_state kv_state and k_state leading [N, H] dims must match.")
        if initial_state_kv.shape[2] != k_prime.shape[-1]:
            raise ValueError("initial_state kv_state feature size must match numerator key feature dim.")
        if initial_state_k.shape[2] != k_prime_den.shape[-1]:
            raise ValueError("initial_state k_state feature size must match denominator key feature dim.")

    if update_rule == "sum":
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
            q=q_prime_den,
            k=k_prime_den,
            v=ones,
            g=log_decay,
            scale=1.0,
            initial_state=den_init,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
    else:
        if log_decay is not None:
            raise ValueError("log_decay is only supported with update_rule='sum'.")
        if beta is None:
            beta = torch.ones_like(q_prime[..., 0], dtype=torch.float32)
        if beta_den is None:
            beta_den = beta

        use_dplr_leaky = (
            delta_log_decay is not None
            and q_prime.shape[-1] >= 16
            and q_prime_den.shape[-1] >= 16
            and v.shape[-1] >= 16
        )
        if not use_dplr_leaky:
            num, kv_final = fused_recurrent_delta_rule(
                q=q_prime,
                k=k_prime,
                v=v,
                beta=beta,
                scale=1.0,
                initial_state=initial_state_kv,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,
            )

            ones = torch.ones((*v.shape[:3], 1), device=v.device, dtype=v.dtype)
            den_init = None if initial_state_k is None else initial_state_k.unsqueeze(-1)
            if delta_denominator_update == "delta":
                den, z_final = fused_recurrent_delta_rule(
                    q=q_prime_den,
                    k=k_prime_den,
                    v=ones,
                    beta=beta_den,
                    scale=1.0,
                    initial_state=den_init,
                    output_final_state=output_final_state,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=False,
                )
            else:
                den, z_final = fused_recurrent_simple_gla(
                    q=q_prime_den,
                    k=k_prime_den,
                    v=ones,
                    g=None,
                    scale=1.0,
                    initial_state=den_init,
                    output_final_state=output_final_state,
                    cu_seqlens=cu_seqlens,
                )
        else:
            if chunk_dplr_delta_rule is None:
                raise RuntimeError(
                    "delta_log_decay was provided but chunk_dplr_delta_rule is unavailable.",
                )
            beta_t = beta.to(q_prime.dtype).unsqueeze(-1)

            # Leaky delta as a DPLR recurrence:
            #   S_t = lambda_t S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T
            # by choosing:
            #   a_t = -beta_t k_t, b_t = k_t, v'_t = beta_t v_t, gk_t = log(lambda_t).
            a_num = -beta_t * k_prime
            b_num = k_prime
            v_num = v * beta_t
            gk_num = delta_log_decay.to(q_prime.dtype).unsqueeze(-1).expand_as(k_prime)

            num, kv_final = chunk_dplr_delta_rule(
                q=q_prime,
                k=k_prime,
                v=v_num,
                a=a_num,
                b=b_num,
                gk=gk_num,
                scale=1.0,
                initial_state=initial_state_kv,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )

            # Keep denominator on fused kernels (value-dim=1 is not currently
            # robust on the DPLR chunk path across all shapes).
            ones = torch.ones((*v.shape[:3], 1), device=v.device, dtype=v.dtype)
            den_init = None if initial_state_k is None else initial_state_k.unsqueeze(-1)
            if delta_denominator_update == "delta":
                den, z_final = fused_recurrent_delta_rule(
                    q=q_prime_den,
                    k=k_prime_den,
                    v=ones,
                    beta=beta_den,
                    scale=1.0,
                    initial_state=den_init,
                    output_final_state=output_final_state,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=False,
                )
            else:
                den, z_final = fused_recurrent_simple_gla(
                    q=q_prime_den,
                    k=k_prime_den,
                    v=ones,
                    g=delta_log_decay,
                    scale=1.0,
                    initial_state=den_init,
                    output_final_state=output_final_state,
                    cu_seqlens=cu_seqlens,
                )

    num32 = num.to(torch.float32)
    den32 = den.to(torch.float32)
    if update_rule == "delta" and delta_denominator_update == "delta":
        # Delta update does not guarantee strictly positive denominator; protect
        # against tiny values that cause gradient explosions.
        #
        # Optional smooth barrier:
        #   |d|_safe = softplus(|d| / tau) * tau
        # provides a differentiable alternative to hard clamping and reduces
        # gradient spikes around |d| ~= delta_denom_eps.
        den_sign = torch.where(den32 >= 0, torch.ones_like(den32), -torch.ones_like(den32))
        den_abs = den32.abs()
        if delta_smooth_denom:
            den_abs = F.softplus(den_abs / float(delta_denom_tau)) * float(delta_denom_tau)
        den_safe = den_sign * den_abs.clamp_min(float(delta_denom_eps))
        out = num32 / den_safe
    elif update_rule == "delta" and delta_denominator_update == "sum":
        # Sum-denominator path is positive by construction; still keep a floor
        # to avoid large ratio spikes when the accumulated mass is tiny.
        den_used = den32.detach() if delta_denominator_stopgrad else den32
        den_safe = den_used.clamp_min(float(delta_denom_eps))
        out = num32 / den_safe
    else:
        out = num32 / (den32 + eps)
    final_state = None
    if output_final_state:
        final_state = (kv_final, z_final.squeeze(-1).to(torch.float32))
    return out, final_state


__all__ = [
    'performer_plus_causal_linear_attention_triton',
    'performer_plus_pdf_delta_attention_triton',
    '_TRITON_AVAILABLE',
]
