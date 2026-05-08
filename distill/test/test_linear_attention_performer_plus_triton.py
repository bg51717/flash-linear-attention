from __future__ import annotations

import os
import sys

import torch

try:
    import pytest
except Exception:  # pragma: no cover
    pytest = None

# Allow running this file directly without installing `distill` as a package.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_IMPORT_ERROR: Exception | None = None
try:
    from fla.ops.performer_plus.fused_recurrent import (
        _TRITON_AVAILABLE,
        performer_plus_causal_linear_attention_triton,
    )
except Exception as exc:  # pragma: no cover - dependency/runtime guard
    _IMPORT_ERROR = exc
    _TRITON_AVAILABLE = False
    performer_plus_causal_linear_attention_triton = None


def _is_triton_test_ready() -> bool:
    return bool(_IMPORT_ERROR is None and torch.cuda.is_available() and _TRITON_AVAILABLE)


def _reference_causal_attention_with_decay(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor | None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    b, t, h, m = q_prime.shape
    vdim = v.shape[-1]

    out = torch.empty(b, t, h, vdim, dtype=q_prime.dtype, device=q_prime.device)
    kv_state = torch.zeros(b, h, m, vdim, dtype=q_prime.dtype, device=q_prime.device)
    k_state = torch.zeros(b, h, m, dtype=q_prime.dtype, device=q_prime.device)

    for i in range(t):
        if log_decay is not None:
            lam = torch.exp(log_decay[:, i]).unsqueeze(-1)  # [B, H, 1]
            kv_state = kv_state * lam.unsqueeze(-1)
            k_state = k_state * lam
        kv_state = kv_state + torch.einsum('bhm,bhv->bhmv', k_prime[:, i], v[:, i])
        k_state = k_state + k_prime[:, i]
        numerator = torch.einsum('bhm,bhmv->bhv', q_prime[:, i], kv_state)
        denominator = (q_prime[:, i] * k_state).sum(dim=-1, keepdim=True)
        out[:, i] = numerator / (denominator + eps)

    return out, (kv_state, k_state)


def _reference_varlen_with_decay(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor | None,
    cu_seqlens: torch.LongTensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    out = torch.empty_like(v)
    kv_states = []
    k_states = []
    cu = cu_seqlens.tolist()
    for i in range(len(cu) - 1):
        start, end = int(cu[i]), int(cu[i + 1])
        if end == start:
            continue
        log_decay_seg = None if log_decay is None else log_decay[:, start:end]
        o_seg, st_seg = _reference_causal_attention_with_decay(
            q_prime[:, start:end],
            k_prime[:, start:end],
            v[:, start:end],
            log_decay=log_decay_seg,
            eps=eps,
        )
        out[:, start:end] = o_seg
        kv_states.append(st_seg[0].squeeze(0))
        k_states.append(st_seg[1].squeeze(0))
    return out, (torch.stack(kv_states, dim=0), torch.stack(k_states, dim=0))


def _reference_delta_normalized(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    q_prime_den: torch.Tensor | None = None,
    k_prime_den: torch.Tensor | None = None,
    eps: float = 1e-6,
    delta_denom_eps: float = 1e-3,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    b, t, h, m = q_prime.shape
    vdim = v.shape[-1]
    if q_prime_den is None:
        q_prime_den = q_prime
    if k_prime_den is None:
        k_prime_den = k_prime
    m_den = q_prime_den.shape[-1]
    out = torch.empty(b, t, h, vdim, dtype=q_prime.dtype, device=q_prime.device)
    kv_state = torch.zeros(b, h, m, vdim, dtype=q_prime.dtype, device=q_prime.device)
    k_state = torch.zeros(b, h, m_den, dtype=q_prime.dtype, device=q_prime.device)

    for i in range(t):
        pred_v = torch.einsum('bhmv,bhm->bhv', kv_state, k_prime[:, i])
        err_v = v[:, i] - pred_v
        kv_state = kv_state + torch.einsum('bhm,bhv->bhmv', k_prime[:, i], beta[:, i].unsqueeze(-1) * err_v)

        pred_1 = (k_state * k_prime_den[:, i]).sum(dim=-1, keepdim=True)
        err_1 = 1.0 - pred_1
        k_state = k_state + beta[:, i].unsqueeze(-1) * k_prime_den[:, i] * err_1

        num = torch.einsum('bhm,bhmv->bhv', q_prime[:, i], kv_state)
        den = (q_prime_den[:, i] * k_state).sum(dim=-1, keepdim=True)
        den_sign = torch.where(den >= 0, torch.ones_like(den), -torch.ones_like(den))
        den_safe = den_sign * den.abs().clamp_min(float(delta_denom_eps))
        out[:, i] = num / den_safe

    return out, (kv_state, k_state)


def _reference_delta_normalized_separate_beta(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    beta_num: torch.Tensor,
    beta_den: torch.Tensor,
    q_prime_den: torch.Tensor,
    k_prime_den: torch.Tensor,
    delta_denom_eps: float = 1e-3,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    b, t, h, m = q_prime.shape
    vdim = v.shape[-1]
    m_den = q_prime_den.shape[-1]
    out = torch.empty(b, t, h, vdim, dtype=q_prime.dtype, device=q_prime.device)
    kv_state = torch.zeros(b, h, m, vdim, dtype=q_prime.dtype, device=q_prime.device)
    k_state = torch.zeros(b, h, m_den, dtype=q_prime.dtype, device=q_prime.device)

    for i in range(t):
        pred_v = torch.einsum('bhmv,bhm->bhv', kv_state, k_prime[:, i])
        err_v = v[:, i] - pred_v
        kv_state = kv_state + torch.einsum(
            'bhm,bhv->bhmv',
            k_prime[:, i],
            beta_num[:, i].unsqueeze(-1) * err_v,
        )

        pred_1 = (k_state * k_prime_den[:, i]).sum(dim=-1, keepdim=True)
        err_1 = 1.0 - pred_1
        k_state = k_state + beta_den[:, i].unsqueeze(-1) * k_prime_den[:, i] * err_1

        num = torch.einsum('bhm,bhmv->bhv', q_prime[:, i], kv_state)
        den = (q_prime_den[:, i] * k_state).sum(dim=-1, keepdim=True)
        den_sign = torch.where(den >= 0, torch.ones_like(den), -torch.ones_like(den))
        den_safe = den_sign * den.abs().clamp_min(float(delta_denom_eps))
        out[:, i] = num / den_safe

    return out, (kv_state, k_state)


def _reference_delta_normalized_varlen(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    q_prime_den: torch.Tensor | None = None,
    k_prime_den: torch.Tensor | None = None,
    eps: float = 1e-6,
    delta_denom_eps: float = 1e-3,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if q_prime_den is None:
        q_prime_den = q_prime
    if k_prime_den is None:
        k_prime_den = k_prime
    out = torch.empty_like(v)
    kv_states = []
    k_states = []
    cu = cu_seqlens.tolist()
    for i in range(len(cu) - 1):
        start, end = int(cu[i]), int(cu[i + 1])
        if end == start:
            continue
        o_seg, st_seg = _reference_delta_normalized(
            q_prime[:, start:end],
            k_prime[:, start:end],
            v[:, start:end],
            beta[:, start:end],
            q_prime_den=q_prime_den[:, start:end],
            k_prime_den=k_prime_den[:, start:end],
            eps=eps,
            delta_denom_eps=delta_denom_eps,
        )
        out[:, start:end] = o_seg
        kv_states.append(st_seg[0].squeeze(0))
        k_states.append(st_seg[1].squeeze(0))
    return out, (torch.stack(kv_states, dim=0), torch.stack(k_states, dim=0))


def _reference_delta_leaky(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    delta_log_decay: torch.Tensor,
    q_prime_den: torch.Tensor | None = None,
    k_prime_den: torch.Tensor | None = None,
    delta_denom_eps: float = 1e-3,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    b, t, h, m = q_prime.shape
    vdim = v.shape[-1]
    if q_prime_den is None:
        q_prime_den = q_prime
    if k_prime_den is None:
        k_prime_den = k_prime
    m_den = q_prime_den.shape[-1]
    out = torch.empty(b, t, h, vdim, dtype=q_prime.dtype, device=q_prime.device)
    kv_state = torch.zeros(b, h, m, vdim, dtype=q_prime.dtype, device=q_prime.device)
    k_state = torch.zeros(b, h, m_den, dtype=q_prime.dtype, device=q_prime.device)

    for i in range(t):
        lam = torch.exp(delta_log_decay[:, i]).unsqueeze(-1)

        pred_v = torch.einsum('bhmv,bhm->bhv', kv_state, k_prime[:, i])
        err_v = v[:, i] - pred_v
        kv_state = (
            kv_state * lam.unsqueeze(-1)
            + torch.einsum('bhm,bhv->bhmv', k_prime[:, i], beta[:, i].unsqueeze(-1) * err_v)
        )

        pred_1 = (k_state * k_prime_den[:, i]).sum(dim=-1, keepdim=True)
        err_1 = 1.0 - pred_1
        k_state = k_state + beta[:, i].unsqueeze(-1) * k_prime_den[:, i] * err_1

        num = torch.einsum('bhm,bhmv->bhv', q_prime[:, i], kv_state)
        den = (q_prime_den[:, i] * k_state).sum(dim=-1, keepdim=True)
        den_sign = torch.where(den >= 0, torch.ones_like(den), -torch.ones_like(den))
        den_safe = den_sign * den.abs().clamp_min(float(delta_denom_eps))
        out[:, i] = num / den_safe

    return out, (kv_state, k_state)


def _run_dense_case() -> None:
    torch.manual_seed(2026)
    device = torch.device('cuda')

    b, t, h, m, vdim = 2, 48, 4, 32, 24
    q = (torch.rand(b, t, h, m, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k = (torch.rand(b, t, h, m, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    val = torch.randn(b, t, h, vdim, device=device, dtype=torch.float32).detach().requires_grad_(True)
    log_decay = (-torch.rand(b, t, h, device=device, dtype=torch.float32) * 0.3).detach().requires_grad_(True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    val_ref = val.detach().clone().requires_grad_(True)
    log_decay_ref = log_decay.detach().clone().requires_grad_(True)

    o_tri, st_tri = performer_plus_causal_linear_attention_triton(
        q_prime=q,
        k_prime=k,
        v=val,
        log_decay=log_decay,
        output_final_state=True,
    )
    o_ref, st_ref = _reference_causal_attention_with_decay(q_ref, k_ref, val_ref, log_decay_ref)

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=6e-3, atol=6e-3)
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(val.grad, val_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(log_decay.grad, log_decay_ref.grad, rtol=1e-2, atol=1e-2)

    torch.testing.assert_close(st_tri[0], st_ref[0], rtol=6e-3, atol=6e-3)
    torch.testing.assert_close(st_tri[1], st_ref[1], rtol=6e-3, atol=6e-3)


def _run_delta_dense_case() -> None:
    torch.manual_seed(2007)
    device = torch.device('cuda')

    b, t, h, m, vdim = 2, 40, 4, 24, 20
    q = (torch.rand(b, t, h, m, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k = (torch.rand(b, t, h, m, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    val = torch.randn(b, t, h, vdim, device=device, dtype=torch.float32).detach().requires_grad_(True)
    beta = torch.sigmoid(torch.randn(b, t, h, device=device, dtype=torch.float32)).detach().requires_grad_(True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    val_ref = val.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)
    eps = 1e-4

    o_tri, st_tri = performer_plus_causal_linear_attention_triton(
        q_prime=q,
        k_prime=k,
        v=val,
        beta=beta,
        output_final_state=True,
        update_rule='delta',
        eps=eps,
        delta_denom_eps=eps,
    )
    o_ref, st_ref = _reference_delta_normalized(
        q_ref,
        k_ref,
        val_ref,
        beta_ref,
        eps=eps,
        delta_denom_eps=eps,
    )

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1.2e-2, atol=1.2e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1.2e-2, atol=1.2e-2)
    torch.testing.assert_close(val.grad, val_ref.grad, rtol=1.2e-2, atol=1.2e-2)
    torch.testing.assert_close(beta.grad, beta_ref.grad, rtol=1.2e-2, atol=1.2e-2)

    torch.testing.assert_close(st_tri[0], st_ref[0], rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(st_tri[1], st_ref[1], rtol=8e-3, atol=8e-3)


def _run_delta_dual_map_dense_case() -> None:
    torch.manual_seed(2326)
    device = torch.device('cuda')

    b, t, h, m_num, m_den, vdim = 2, 36, 3, 18, 24, 12
    q_num = (torch.rand(b, t, h, m_num, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k_num = (torch.rand(b, t, h, m_num, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    q_den = (torch.rand(b, t, h, m_den, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k_den = (torch.rand(b, t, h, m_den, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    val = torch.randn(b, t, h, vdim, device=device, dtype=torch.float32).detach().requires_grad_(True)
    beta = torch.sigmoid(torch.randn(b, t, h, device=device, dtype=torch.float32)).detach().requires_grad_(True)

    q_num_ref = q_num.detach().clone().requires_grad_(True)
    k_num_ref = k_num.detach().clone().requires_grad_(True)
    q_den_ref = q_den.detach().clone().requires_grad_(True)
    k_den_ref = k_den.detach().clone().requires_grad_(True)
    val_ref = val.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)

    o_tri, st_tri = performer_plus_causal_linear_attention_triton(
        q_prime=q_num,
        k_prime=k_num,
        q_prime_den=q_den,
        k_prime_den=k_den,
        v=val,
        beta=beta,
        output_final_state=True,
        update_rule='delta',
    )
    o_ref, st_ref = _reference_delta_normalized(
        q_num_ref,
        k_num_ref,
        val_ref,
        beta_ref,
        q_prime_den=q_den_ref,
        k_prime_den=k_den_ref,
    )

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(q_num.grad, q_num_ref.grad, rtol=1.2e-2, atol=1.2e-2)
    torch.testing.assert_close(k_num.grad, k_num_ref.grad, rtol=1.2e-2, atol=1.2e-2)
    torch.testing.assert_close(q_den.grad, q_den_ref.grad, rtol=1.2e-2, atol=1.2e-2)
    torch.testing.assert_close(k_den.grad, k_den_ref.grad, rtol=1.2e-2, atol=1.2e-2)
    torch.testing.assert_close(val.grad, val_ref.grad, rtol=1.2e-2, atol=1.2e-2)
    torch.testing.assert_close(beta.grad, beta_ref.grad, rtol=1.2e-2, atol=1.2e-2)

    torch.testing.assert_close(st_tri[0], st_ref[0], rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(st_tri[1], st_ref[1], rtol=8e-3, atol=8e-3)


def _run_delta_dual_map_separate_beta_dense_case() -> None:
    torch.manual_seed(2628)
    device = torch.device('cuda')

    b, t, h, m_num, m_den, vdim = 2, 36, 3, 18, 24, 12
    q_num = (torch.rand(b, t, h, m_num, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k_num = (torch.rand(b, t, h, m_num, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    q_den = (torch.rand(b, t, h, m_den, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k_den = (torch.rand(b, t, h, m_den, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    val = torch.randn(b, t, h, vdim, device=device, dtype=torch.float32).detach().requires_grad_(True)
    beta_num = torch.sigmoid(torch.randn(b, t, h, device=device, dtype=torch.float32)).detach().requires_grad_(True)
    beta_den = (0.5 * torch.sigmoid(torch.randn(b, t, h, device=device, dtype=torch.float32))).detach().requires_grad_(True)

    q_num_ref = q_num.detach().clone().requires_grad_(True)
    k_num_ref = k_num.detach().clone().requires_grad_(True)
    q_den_ref = q_den.detach().clone().requires_grad_(True)
    k_den_ref = k_den.detach().clone().requires_grad_(True)
    val_ref = val.detach().clone().requires_grad_(True)
    beta_num_ref = beta_num.detach().clone().requires_grad_(True)
    beta_den_ref = beta_den.detach().clone().requires_grad_(True)

    o_tri, st_tri = performer_plus_causal_linear_attention_triton(
        q_prime=q_num,
        k_prime=k_num,
        q_prime_den=q_den,
        k_prime_den=k_den,
        v=val,
        beta=beta_num,
        beta_den=beta_den,
        output_final_state=True,
        update_rule='delta',
    )
    o_ref, st_ref = _reference_delta_normalized_separate_beta(
        q_num_ref,
        k_num_ref,
        val_ref,
        beta_num_ref,
        beta_den_ref,
        q_prime_den=q_den_ref,
        k_prime_den=k_den_ref,
    )

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=1.2e-2, atol=1.2e-2)
    assert torch.isfinite(q_num.grad).all()
    assert torch.isfinite(k_num.grad).all()
    assert torch.isfinite(q_den.grad).all()
    assert torch.isfinite(k_den.grad).all()
    assert torch.isfinite(val.grad).all()
    assert torch.isfinite(beta_num.grad).all()
    assert torch.isfinite(beta_den.grad).all()
    assert torch.isfinite(st_tri[0]).all()
    assert torch.isfinite(st_tri[1]).all()
    assert torch.isfinite(st_ref[0]).all()
    assert torch.isfinite(st_ref[1]).all()


def _run_delta_leaky_dense_case() -> None:
    torch.manual_seed(2528)
    device = torch.device('cuda')

    b, t, h, m_num, m_den, vdim = 2, 36, 3, 20, 26, 20
    q_num = (torch.rand(b, t, h, m_num, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k_num = (torch.rand(b, t, h, m_num, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    q_den = (torch.rand(b, t, h, m_den, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k_den = (torch.rand(b, t, h, m_den, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    val = torch.randn(b, t, h, vdim, device=device, dtype=torch.float32).detach().requires_grad_(True)
    beta = (0.2 * torch.sigmoid(torch.randn(b, t, h, device=device, dtype=torch.float32))).detach().requires_grad_(True)
    delta_log_decay = (-1.2 * beta.detach()).requires_grad_(True)

    q_num_ref = q_num.detach().clone().requires_grad_(True)
    k_num_ref = k_num.detach().clone().requires_grad_(True)
    q_den_ref = q_den.detach().clone().requires_grad_(True)
    k_den_ref = k_den.detach().clone().requires_grad_(True)
    val_ref = val.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)
    delta_log_decay_ref = delta_log_decay.detach().clone().requires_grad_(True)

    o_tri, st_tri = performer_plus_causal_linear_attention_triton(
        q_prime=q_num,
        k_prime=k_num,
        q_prime_den=q_den,
        k_prime_den=k_den,
        v=val,
        beta=beta,
        delta_log_decay=delta_log_decay,
        output_final_state=True,
        update_rule='delta',
        delta_denom_eps=1e-4,
    )
    o_ref, st_ref = _reference_delta_leaky(
        q_num_ref,
        k_num_ref,
        val_ref,
        beta_ref,
        delta_log_decay_ref,
        q_prime_den=q_den_ref,
        k_prime_den=k_den_ref,
        delta_denom_eps=1e-4,
    )

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=1.2e-2, atol=1.2e-2)
    assert torch.isfinite(q_num.grad).all()
    assert torch.isfinite(k_num.grad).all()
    assert torch.isfinite(q_den.grad).all()
    assert torch.isfinite(k_den.grad).all()
    assert torch.isfinite(val.grad).all()
    assert torch.isfinite(beta.grad).all()
    assert torch.isfinite(delta_log_decay.grad).all()

    torch.testing.assert_close(st_tri[0], st_ref[0], rtol=1.2e-2, atol=1.2e-2)
    torch.testing.assert_close(st_tri[1], st_ref[1], rtol=1.2e-2, atol=1.2e-2)


def _run_varlen_case() -> None:
    torch.manual_seed(2027)
    device = torch.device('cuda')

    b, h, m, vdim = 1, 3, 16, 12
    cu = torch.tensor([0, 17, 39, 64], dtype=torch.int32, device=device)
    t = int(cu[-1].item())

    q = (torch.rand(b, t, h, m, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k = (torch.rand(b, t, h, m, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    val = torch.randn(b, t, h, vdim, device=device, dtype=torch.float32).detach().requires_grad_(True)
    log_decay = (-torch.rand(b, t, h, device=device, dtype=torch.float32) * 0.2).detach().requires_grad_(True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    val_ref = val.detach().clone().requires_grad_(True)
    log_decay_ref = log_decay.detach().clone().requires_grad_(True)

    o_tri, st_tri = performer_plus_causal_linear_attention_triton(
        q_prime=q,
        k_prime=k,
        v=val,
        log_decay=log_decay,
        output_final_state=True,
        cu_seqlens=cu,
    )
    o_ref, st_ref = _reference_varlen_with_decay(q_ref, k_ref, val_ref, log_decay_ref, cu)

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1.2e-2, atol=1.2e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1.2e-2, atol=1.2e-2)
    torch.testing.assert_close(val.grad, val_ref.grad, rtol=1.2e-2, atol=1.2e-2)
    torch.testing.assert_close(log_decay.grad, log_decay_ref.grad, rtol=1.2e-2, atol=1.2e-2)

    torch.testing.assert_close(st_tri[0], st_ref[0], rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(st_tri[1], st_ref[1], rtol=8e-3, atol=8e-3)


def _run_delta_varlen_case() -> None:
    torch.manual_seed(2127)
    device = torch.device('cuda')

    b, h, m, vdim = 1, 3, 20, 10
    cu = torch.tensor([0, 13, 31, 55], dtype=torch.int32, device=device)
    t = int(cu[-1].item())

    q = (torch.rand(b, t, h, m, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k = (torch.rand(b, t, h, m, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    val = torch.randn(b, t, h, vdim, device=device, dtype=torch.float32).detach().requires_grad_(True)
    beta = torch.sigmoid(torch.randn(b, t, h, device=device, dtype=torch.float32)).detach().requires_grad_(True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    val_ref = val.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)

    o_tri, st_tri = performer_plus_causal_linear_attention_triton(
        q_prime=q,
        k_prime=k,
        v=val,
        beta=beta,
        output_final_state=True,
        cu_seqlens=cu,
        update_rule='delta',
    )
    o_ref, st_ref = _reference_delta_normalized_varlen(q_ref, k_ref, val_ref, beta_ref, cu)

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=1.1e-2, atol=1.1e-2)
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1.5e-2, atol=1.5e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1.5e-2, atol=1.5e-2)
    torch.testing.assert_close(val.grad, val_ref.grad, rtol=1.5e-2, atol=1.5e-2)
    torch.testing.assert_close(beta.grad, beta_ref.grad, rtol=1.5e-2, atol=1.5e-2)

    torch.testing.assert_close(st_tri[0], st_ref[0], rtol=1.1e-2, atol=1.1e-2)
    torch.testing.assert_close(st_tri[1], st_ref[1], rtol=1.1e-2, atol=1.1e-2)


def _run_delta_dual_map_varlen_case() -> None:
    torch.manual_seed(2427)
    device = torch.device('cuda')

    b, h, m_num, m_den, vdim = 1, 3, 18, 22, 10
    cu = torch.tensor([0, 13, 31, 55], dtype=torch.int32, device=device)
    t = int(cu[-1].item())

    q_num = (torch.rand(b, t, h, m_num, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k_num = (torch.rand(b, t, h, m_num, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    q_den = (torch.rand(b, t, h, m_den, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k_den = (torch.rand(b, t, h, m_den, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    val = torch.randn(b, t, h, vdim, device=device, dtype=torch.float32).detach().requires_grad_(True)
    beta = torch.sigmoid(torch.randn(b, t, h, device=device, dtype=torch.float32)).detach().requires_grad_(True)

    q_num_ref = q_num.detach().clone().requires_grad_(True)
    k_num_ref = k_num.detach().clone().requires_grad_(True)
    q_den_ref = q_den.detach().clone().requires_grad_(True)
    k_den_ref = k_den.detach().clone().requires_grad_(True)
    val_ref = val.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)

    o_tri, st_tri = performer_plus_causal_linear_attention_triton(
        q_prime=q_num,
        k_prime=k_num,
        q_prime_den=q_den,
        k_prime_den=k_den,
        v=val,
        beta=beta,
        output_final_state=True,
        cu_seqlens=cu,
        update_rule='delta',
    )
    o_ref, st_ref = _reference_delta_normalized_varlen(
        q_num_ref,
        k_num_ref,
        val_ref,
        beta_ref,
        cu,
        q_prime_den=q_den_ref,
        k_prime_den=k_den_ref,
    )

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=1.1e-2, atol=1.1e-2)
    torch.testing.assert_close(q_num.grad, q_num_ref.grad, rtol=1.5e-2, atol=1.5e-2)
    torch.testing.assert_close(k_num.grad, k_num_ref.grad, rtol=1.5e-2, atol=1.5e-2)
    torch.testing.assert_close(q_den.grad, q_den_ref.grad, rtol=1.5e-2, atol=1.5e-2)
    torch.testing.assert_close(k_den.grad, k_den_ref.grad, rtol=1.5e-2, atol=1.5e-2)
    torch.testing.assert_close(val.grad, val_ref.grad, rtol=1.5e-2, atol=1.5e-2)
    torch.testing.assert_close(beta.grad, beta_ref.grad, rtol=1.5e-2, atol=1.5e-2)

    torch.testing.assert_close(st_tri[0], st_ref[0], rtol=1.1e-2, atol=1.1e-2)
    torch.testing.assert_close(st_tri[1], st_ref[1], rtol=1.1e-2, atol=1.1e-2)


if pytest is not None:
    if _IMPORT_ERROR is not None:
        pytestmark = pytest.mark.skip(reason=f'Performer+ test import failed: {_IMPORT_ERROR}')

    @pytest.mark.skipif(not _is_triton_test_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_triton_dense() -> None:
        _run_dense_case()


    @pytest.mark.skipif(not _is_triton_test_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_triton_varlen() -> None:
        _run_varlen_case()

    @pytest.mark.skipif(not _is_triton_test_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_triton_delta_dense() -> None:
        _run_delta_dense_case()

    @pytest.mark.skipif(not _is_triton_test_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_triton_delta_dual_map_dense() -> None:
        _run_delta_dual_map_dense_case()

    @pytest.mark.skipif(not _is_triton_test_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_triton_delta_dual_map_separate_beta_dense() -> None:
        _run_delta_dual_map_separate_beta_dense_case()

    @pytest.mark.skipif(not _is_triton_test_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_triton_delta_leaky_dense() -> None:
        _run_delta_leaky_dense_case()

    @pytest.mark.skipif(not _is_triton_test_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_triton_delta_varlen() -> None:
        _run_delta_varlen_case()

    @pytest.mark.skipif(not _is_triton_test_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_triton_delta_dual_map_varlen() -> None:
        _run_delta_dual_map_varlen_case()


def main() -> None:
    if _IMPORT_ERROR is not None:
        print(f'skip: import failed: {_IMPORT_ERROR}')
        return
    if not _is_triton_test_ready():
        print('skip: CUDA + Triton are required.')
        return
    _run_dense_case()
    _run_varlen_case()
    _run_delta_dense_case()
    _run_delta_dual_map_dense_case()
    _run_delta_dual_map_separate_beta_dense_case()
    _run_delta_leaky_dense_case()
    _run_delta_varlen_case()
    _run_delta_dual_map_varlen_case()
    print('ok: performer+ triton dense + varlen + delta passed')


if __name__ == '__main__':
    main()
