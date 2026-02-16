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
    from distill.models.linear_attention_performer_triton import (
        _TRITON_AVAILABLE,
        performer_causal_linear_attention_triton,
    )
except Exception as exc:  # pragma: no cover - dependency/runtime guard
    _IMPORT_ERROR = exc
    _TRITON_AVAILABLE = False
    performer_causal_linear_attention_triton = None


def _is_triton_test_ready() -> bool:
    return bool(_IMPORT_ERROR is None and torch.cuda.is_available() and _TRITON_AVAILABLE)


def _reference_causal_attention(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    b, t, h, m = q_prime.shape
    vdim = v.shape[-1]

    out = torch.empty(b, t, h, vdim, dtype=q_prime.dtype, device=q_prime.device)
    kv_state = torch.zeros(b, h, m, vdim, dtype=q_prime.dtype, device=q_prime.device)
    k_state = torch.zeros(b, h, m, dtype=q_prime.dtype, device=q_prime.device)

    for i in range(t):
        kv_state = kv_state + torch.einsum('bhm,bhv->bhmv', k_prime[:, i], v[:, i])
        k_state = k_state + k_prime[:, i]
        numerator = torch.einsum('bhm,bhmv->bhv', q_prime[:, i], kv_state)
        denominator = (q_prime[:, i] * k_state).sum(dim=-1, keepdim=True)
        out[:, i] = numerator / (denominator + eps)

    return out, (kv_state, k_state)


def _reference_varlen(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
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
        o_seg, st_seg = _reference_causal_attention(
            q_prime[:, start:end],
            k_prime[:, start:end],
            v[:, start:end],
            eps=eps,
        )
        out[:, start:end] = o_seg
        kv_states.append(st_seg[0].squeeze(0))
        k_states.append(st_seg[1].squeeze(0))
    return out, (torch.stack(kv_states, dim=0), torch.stack(k_states, dim=0))


def _run_dense_case() -> None:
    torch.manual_seed(1234)
    device = torch.device('cuda')

    b, t, h, m, v = 2, 64, 4, 32, 24
    q = (torch.rand(b, t, h, m, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k = (torch.rand(b, t, h, m, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    val = torch.randn(b, t, h, v, device=device, dtype=torch.float32).detach().requires_grad_(True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    val_ref = val.detach().clone().requires_grad_(True)

    o_tri, st_tri = performer_causal_linear_attention_triton(
        q_prime=q,
        k_prime=k,
        v=val,
        output_final_state=True,
    )
    o_ref, st_ref = _reference_causal_attention(q_ref, k_ref, val_ref)

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(val.grad, val_ref.grad, rtol=8e-3, atol=8e-3)

    torch.testing.assert_close(st_tri[0], st_ref[0], rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(st_tri[1], st_ref[1], rtol=5e-3, atol=5e-3)


def _run_varlen_case() -> None:
    torch.manual_seed(5678)
    device = torch.device('cuda')

    b, h, m, v = 1, 3, 16, 16
    cu = torch.tensor([0, 21, 40, 64], dtype=torch.int32, device=device)
    t = int(cu[-1].item())

    q = (torch.rand(b, t, h, m, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    k = (torch.rand(b, t, h, m, device=device, dtype=torch.float32) + 0.05).detach().requires_grad_(True)
    val = torch.randn(b, t, h, v, device=device, dtype=torch.float32).detach().requires_grad_(True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    val_ref = val.detach().clone().requires_grad_(True)

    o_tri, st_tri = performer_causal_linear_attention_triton(
        q_prime=q,
        k_prime=k,
        v=val,
        output_final_state=True,
        cu_seqlens=cu,
    )
    o_ref, st_ref = _reference_varlen(q_ref, k_ref, val_ref, cu)

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=6e-3, atol=6e-3)
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(val.grad, val_ref.grad, rtol=1e-2, atol=1e-2)

    torch.testing.assert_close(st_tri[0], st_ref[0], rtol=6e-3, atol=6e-3)
    torch.testing.assert_close(st_tri[1], st_ref[1], rtol=6e-3, atol=6e-3)


if pytest is not None:
    if _IMPORT_ERROR is not None:
        pytestmark = pytest.mark.skip(reason=f'Performer test import failed: {_IMPORT_ERROR}')

    @pytest.mark.skipif(not _is_triton_test_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_triton_dense() -> None:
        _run_dense_case()


    @pytest.mark.skipif(not _is_triton_test_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_triton_varlen() -> None:
        _run_varlen_case()


def main() -> None:
    if _IMPORT_ERROR is not None:
        print(f'skip: import failed: {_IMPORT_ERROR}')
        return
    if not _is_triton_test_ready():
        print('skip: CUDA + Triton are required.')
        return
    _run_dense_case()
    _run_varlen_case()
    print('ok: performer triton dense + varlen passed')


if __name__ == '__main__':
    main()
