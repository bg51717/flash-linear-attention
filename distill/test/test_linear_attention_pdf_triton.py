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

from distill.models import linear_attention_pdf_triton as la


def _is_triton_test_ready() -> bool:
    return bool(torch.cuda.is_available() and la._TRITON_AVAILABLE)


def _to_flat_heads(x: torch.Tensor) -> torch.Tensor:
    # [B, T, H, D] -> [B*H, T, D]
    b, t, h, d = x.shape
    return x.permute(0, 2, 1, 3).contiguous().reshape(b * h, t, d)


def _from_flat_heads(x: torch.Tensor, b: int, h: int) -> torch.Tensor:
    # [B*H, T, D] -> [B, T, H, D]
    _, t, d = x.shape
    return x.reshape(b, h, t, d).permute(0, 2, 1, 3).contiguous()


def _build_varlen_reference(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    # flat tensors: [H, T, D]
    seg_outputs = []
    state_chunks = [[] for _ in range(6)]
    cu = cu_seqlens.tolist()

    for i in range(len(cu) - 1):
        bos, eos = int(cu[i]), int(cu[i + 1])
        o_seg, st_seg = la._torch_first_order_linear_attention(
            q_flat[:, bos:eos, :],
            k_flat[:, bos:eos, :],
            v_flat[:, bos:eos, :],
            initial_state=None,
            output_final_state=True,
        )
        seg_outputs.append(o_seg)
        for j in range(6):
            state_chunks[j].append(st_seg[j])

    o_full = torch.cat(seg_outputs, dim=1)
    state_full = tuple(torch.stack(chunks, dim=0) for chunks in state_chunks)
    return o_full, state_full


def _run_dense_case() -> None:
    torch.manual_seed(123)
    device = torch.device('cuda')

    b, t, h, k, v = 2, 64, 4, 32, 24

    q = torch.randn(b, t, h, k, device=device, dtype=torch.float32, requires_grad=True)
    k_t = torch.randn(b, t, h, k, device=device, dtype=torch.float32, requires_grad=True)
    v_t = torch.randn(b, t, h, v, device=device, dtype=torch.float32, requires_grad=True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k_t.detach().clone().requires_grad_(True)
    v_ref = v_t.detach().clone().requires_grad_(True)

    o_tri, st_tri = la.first_order_linear_attention(q, k_t, v_t, output_final_state=True)

    qf_ref = _to_flat_heads(q_ref)
    kf_ref = _to_flat_heads(k_ref)
    vf_ref = _to_flat_heads(v_ref)
    o_ref_flat, st_ref_flat = la._torch_first_order_linear_attention(
        qf_ref,
        kf_ref,
        vf_ref,
        initial_state=None,
        output_final_state=True,
    )
    o_ref = _from_flat_heads(o_ref_flat, b, h)

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=5e-3, atol=5e-3)
    # dq is the most cancellation-sensitive path in this recurrence.
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=6e-3, atol=4e-1)
    torch.testing.assert_close(k_t.grad, k_ref.grad, rtol=6e-3, atol=6e-3)
    torch.testing.assert_close(v_t.grad, v_ref.grad, rtol=6e-3, atol=6e-3)

    st_ref = (
        st_ref_flat[0].reshape(b, h, v, k),
        st_ref_flat[1].reshape(b, h, k),
        st_ref_flat[2].reshape(b, h, v),
        st_ref_flat[3].reshape(b, h),
        st_ref_flat[4].reshape(b, h),
        st_ref_flat[5].reshape(b, h, k),
    )
    for x, y in zip(st_tri, st_ref, strict=False):
        torch.testing.assert_close(x, y, rtol=5e-3, atol=5e-3)


def _run_varlen_case() -> None:
    torch.manual_seed(456)
    device = torch.device('cuda')

    b, h, k, v = 1, 3, 16, 16
    cu = torch.tensor([0, 21, 40, 64], dtype=torch.int32, device=device)
    t = int(cu[-1].item())

    q = torch.randn(b, t, h, k, device=device, dtype=torch.float32, requires_grad=True)
    k_t = torch.randn(b, t, h, k, device=device, dtype=torch.float32, requires_grad=True)
    v_t = torch.randn(b, t, h, v, device=device, dtype=torch.float32, requires_grad=True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k_t.detach().clone().requires_grad_(True)
    v_ref = v_t.detach().clone().requires_grad_(True)

    o_tri, st_tri = la.first_order_linear_attention(
        q,
        k_t,
        v_t,
        output_final_state=True,
        cu_seqlens=cu,
    )

    qf_ref = _to_flat_heads(q_ref)
    kf_ref = _to_flat_heads(k_ref)
    vf_ref = _to_flat_heads(v_ref)
    o_ref_flat, st_ref_flat = _build_varlen_reference(qf_ref, kf_ref, vf_ref, cu)
    o_ref = _from_flat_heads(o_ref_flat, b, h)

    do = torch.randn_like(o_tri)
    (o_tri * do).sum().backward()
    (o_ref * do).sum().backward()

    torch.testing.assert_close(o_tri, o_ref, rtol=5e-3, atol=5e-3)
    # dq is the most cancellation-sensitive path in this recurrence.
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=6e-3, atol=4e-1)
    torch.testing.assert_close(k_t.grad, k_ref.grad, rtol=6e-3, atol=6e-3)
    torch.testing.assert_close(v_t.grad, v_ref.grad, rtol=6e-3, atol=6e-3)

    st_ref = (
        st_ref_flat[0].reshape(cu.numel() - 1, h, v, k),
        st_ref_flat[1].reshape(cu.numel() - 1, h, k),
        st_ref_flat[2].reshape(cu.numel() - 1, h, v),
        st_ref_flat[3].reshape(cu.numel() - 1, h),
        st_ref_flat[4].reshape(cu.numel() - 1, h),
        st_ref_flat[5].reshape(cu.numel() - 1, h, k),
    )
    for x, y in zip(st_tri, st_ref, strict=False):
        torch.testing.assert_close(x, y, rtol=5e-3, atol=5e-3)


if pytest is not None:

    @pytest.mark.skipif(not _is_triton_test_ready(), reason='CUDA + Triton are required for this test.')
    def test_linear_attention_pdf_triton_dense() -> None:
        _run_dense_case()


    @pytest.mark.skipif(not _is_triton_test_ready(), reason='CUDA + Triton are required for this test.')
    def test_linear_attention_pdf_triton_varlen() -> None:
        _run_varlen_case()


def main() -> None:
    if not _is_triton_test_ready():
        print('skip: CUDA + Triton are required.')
        return
    _run_dense_case()
    _run_varlen_case()
    print('ok: linear_attention_pdf triton dense + varlen passed')


if __name__ == '__main__':
    main()
