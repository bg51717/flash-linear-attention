from __future__ import annotations

import os
import sys

import torch

try:
    import pytest
except Exception:  # pragma: no cover
    pytest = None

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fla.layers.pdf_final import pdf_final_linear_attention
from fla.ops.pdf_final.fused_recurrent import (
    _TRITON_AVAILABLE,
    pdf_final_linear_attention_triton,
)


def _is_ready() -> bool:
    return bool(torch.cuda.is_available() and _TRITON_AVAILABLE)


def _run_dense_case() -> None:
    torch.manual_seed(123)
    device = torch.device("cuda")

    q = torch.randn(2, 33, 4, 32, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 33, 4, 32, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(2, 33, 4, 24, device=device, dtype=torch.float32, requires_grad=True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    o_tri, _, _ = pdf_final_linear_attention_triton(q, k, v)
    o_ref, _, _ = pdf_final_linear_attention(q_ref, k_ref, v_ref)

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v.grad, v_ref.grad, rtol=1e-3, atol=1e-3)


def _run_varlen_case() -> None:
    torch.manual_seed(456)
    device = torch.device("cuda")
    cu = torch.tensor([0, 9, 20, 31], dtype=torch.int32, device=device)
    t = int(cu[-1].item())

    q = torch.randn(1, t, 3, 16, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(1, t, 3, 16, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(1, t, 3, 12, device=device, dtype=torch.float32, requires_grad=True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    o_tri, _, _ = pdf_final_linear_attention_triton(q, k, v, cu_seqlens=cu)
    o_ref, _, _ = pdf_final_linear_attention(q_ref, k_ref, v_ref, cu_seqlens=cu)

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v.grad, v_ref.grad, rtol=1e-3, atol=1e-3)


if pytest is not None:

    @pytest.mark.skipif(not _is_ready(), reason="CUDA + Triton are required for this test.")
    def test_pdf_final_triton_dense() -> None:
        _run_dense_case()


    @pytest.mark.skipif(not _is_ready(), reason="CUDA + Triton are required for this test.")
    def test_pdf_final_triton_varlen() -> None:
        _run_varlen_case()


def main() -> None:
    if not _is_ready():
        print("skip: CUDA + Triton are required.")
        return
    _run_dense_case()
    _run_varlen_case()
    print("ok: pdf_final_linear_attention triton tests passed")


if __name__ == "__main__":
    main()
