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

_IMPORT_ERROR: Exception | None = None
try:
    from fla.layers.approxnet_v3 import (
        ApproxNetV3LinearAttention,
        approxnet_v3_linear_attention,
    )
    from fla.ops.approxnet_v3.fused_recurrent import (
        _TRITON_AVAILABLE as _APPROXNET_V3_TRITON_AVAILABLE,
        approxnet_v3_linear_attention_triton,
    )
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    ApproxNetV3LinearAttention = None
    approxnet_v3_linear_attention = None
    approxnet_v3_linear_attention_triton = None
    _APPROXNET_V3_TRITON_AVAILABLE = False


def _run_dense_chunk_equivalence_case() -> None:
    torch.manual_seed(20260415)
    q = torch.randn(2, 15, 3, 12, dtype=torch.float32) * 0.5
    k = torch.randn(2, 15, 3, 12, dtype=torch.float32) * 0.5
    v = torch.randn(2, 15, 3, 8, dtype=torch.float32)

    out_full, st_full, _ = approxnet_v3_linear_attention(
        q=q,
        k=k,
        v=v,
        output_final_state=True,
        beta_denom_eps=1e-5,
        score_clip=12.0,
    )
    out_a, st_a, _ = approxnet_v3_linear_attention(
        q=q[:, :6],
        k=k[:, :6],
        v=v[:, :6],
        output_final_state=True,
        beta_denom_eps=1e-5,
        score_clip=12.0,
    )
    out_b, st_b, _ = approxnet_v3_linear_attention(
        q=q[:, 6:],
        k=k[:, 6:],
        v=v[:, 6:],
        initial_state=st_a,
        output_final_state=True,
        beta_denom_eps=1e-5,
        score_clip=12.0,
    )

    assert st_full is not None and st_b is not None
    assert torch.allclose(out_full[:, :6], out_a, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_full[:, 6:], out_b, atol=1e-5, rtol=1e-5)
    for x, y in zip(st_full, st_b):
        assert torch.allclose(x, y, atol=1e-5, rtol=1e-5)


def _run_varlen_equivalence_case() -> None:
    torch.manual_seed(20260415 + 1)
    q = torch.randn(1, 17, 2, 10, dtype=torch.float32) * 0.4
    k = torch.randn(1, 17, 2, 10, dtype=torch.float32) * 0.4
    v = torch.randn(1, 17, 2, 6, dtype=torch.float32)
    cu = torch.tensor([0, 5, 11, 17], dtype=torch.long)

    out_packed, st_packed, _ = approxnet_v3_linear_attention(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu,
        output_final_state=True,
        beta_denom_eps=1e-5,
        score_clip=12.0,
    )

    parts = []
    states = [[], [], [], [], [], [], []]
    for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
        o_i, st_i, _ = approxnet_v3_linear_attention(
            q=q[:, s:e],
            k=k[:, s:e],
            v=v[:, s:e],
            output_final_state=True,
            beta_denom_eps=1e-5,
            score_clip=12.0,
        )
        parts.append(o_i)
        assert st_i is not None
        for idx in range(7):
            states[idx].append(st_i[idx][0])

    out_ref = torch.cat(parts, dim=1)
    assert st_packed is not None
    assert torch.allclose(out_packed, out_ref, atol=1e-5, rtol=1e-5)
    for idx in range(7):
        assert torch.allclose(st_packed[idx], torch.stack(states[idx], dim=0), atol=1e-5, rtol=1e-5)


def _run_triton_vs_torch_case() -> None:
    if not torch.cuda.is_available() or not _APPROXNET_V3_TRITON_AVAILABLE:
        return
    torch.manual_seed(20260415 + 2)
    device = torch.device("cuda")
    q_ref = (torch.randn(2, 19, 4, 16, device=device, dtype=torch.float32) * 0.10).requires_grad_(True)
    k_ref = (torch.randn(2, 19, 4, 16, device=device, dtype=torch.float32) * 0.10).requires_grad_(True)
    v_ref = torch.randn(2, 19, 4, 8, device=device, dtype=torch.float32, requires_grad=True)

    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)
    v_tri = v_ref.detach().clone().requires_grad_(True)

    out_ref, _, _ = approxnet_v3_linear_attention(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        beta_denom_eps=1e-1,
        score_clip=12.0,
    )
    out_tri, _, _ = approxnet_v3_linear_attention_triton(
        q=q_tri,
        k=k_tri,
        v=v_tri,
        beta_denom_eps=1e-1,
        score_clip=12.0,
        recompute_chunk_size=64,
    )
    assert torch.allclose(out_ref.float(), out_tri.float(), atol=6e-4, rtol=6e-4), \
        f"fwd max diff: {(out_ref.float() - out_tri.float()).abs().max().item()}"

    loss_ref = out_ref.float().pow(2).mean()
    loss_tri = out_tri.float().pow(2).mean()
    loss_ref.backward()
    loss_tri.backward()

    assert q_ref.grad is not None and k_ref.grad is not None and v_ref.grad is not None
    assert q_tri.grad is not None and k_tri.grad is not None and v_tri.grad is not None
    assert torch.isfinite(q_tri.grad).all() and torch.isfinite(k_tri.grad).all() and torch.isfinite(v_tri.grad).all()
    assert torch.allclose(q_ref.grad.float(), q_tri.grad.float(), atol=8e-4, rtol=2e-3), \
        f"dq max diff: {(q_ref.grad.float() - q_tri.grad.float()).abs().max().item()}"
    assert torch.allclose(k_ref.grad.float(), k_tri.grad.float(), atol=8e-4, rtol=2e-3), \
        f"dk max diff: {(k_ref.grad.float() - k_tri.grad.float()).abs().max().item()}"
    assert torch.allclose(v_ref.grad.float(), v_tri.grad.float(), atol=8e-4, rtol=2e-3), \
        f"dv max diff: {(v_ref.grad.float() - v_tri.grad.float()).abs().max().item()}"


def _run_triton_stateful_dense_vs_torch_case() -> None:
    if not torch.cuda.is_available() or not _APPROXNET_V3_TRITON_AVAILABLE:
        return
    torch.manual_seed(20260415 + 4)
    device = torch.device("cuda")

    b, t, h, k_dim, v_dim = 2, 17, 3, 16, 8
    q = torch.randn(b, t, h, k_dim, device=device, dtype=torch.float32) * 0.15
    k = torch.randn(b, t, h, k_dim, device=device, dtype=torch.float32) * 0.15
    v = torch.randn(b, t, h, v_dim, device=device, dtype=torch.float32)

    init_state = (
        torch.randn(b, h, k_dim, v_dim, device=device, dtype=torch.float32) * 0.05,
        torch.randn(b, h, v_dim, device=device, dtype=torch.float32) * 0.05,
        torch.randn(b, h, k_dim, device=device, dtype=torch.float32) * 0.05,
        torch.randn(b, h, k_dim, device=device, dtype=torch.float32) * 0.05,
        torch.randn(b, h, v_dim, device=device, dtype=torch.float32) * 0.05,
        torch.randn(b, h, k_dim, device=device, dtype=torch.float32) * 0.05,
        torch.rand(b, h, device=device, dtype=torch.float32) * 5.0,
    )

    out_ref, st_ref, _ = approxnet_v3_linear_attention(
        q=q,
        k=k,
        v=v,
        initial_state=tuple(x.clone() for x in init_state),
        output_final_state=True,
        beta_denom_eps=1e-5,
        score_clip=12.0,
    )
    out_tri, st_tri, _ = approxnet_v3_linear_attention_triton(
        q=q,
        k=k,
        v=v,
        initial_state=tuple(x.clone() for x in init_state),
        output_final_state=True,
        beta_denom_eps=1e-5,
        score_clip=12.0,
        recompute_chunk_size=64,
    )
    assert st_ref is not None and st_tri is not None
    assert torch.allclose(out_ref.float(), out_tri.float(), atol=8e-4, rtol=8e-4), \
        f"stateful fwd max diff: {(out_ref.float() - out_tri.float()).abs().max().item()}"
    for x, y in zip(st_ref, st_tri):
        assert torch.allclose(x.float(), y.float(), atol=1e-3, rtol=1e-3)


def _run_triton_stateful_varlen_vs_torch_case() -> None:
    if not torch.cuda.is_available() or not _APPROXNET_V3_TRITON_AVAILABLE:
        return
    torch.manual_seed(20260415 + 5)
    device = torch.device("cuda")

    h, k_dim, v_dim = 2, 12, 6
    cu = torch.tensor([0, 5, 11, 17], dtype=torch.long, device=device)
    n_seq = cu.numel() - 1
    total = int(cu[-1].item())
    q = torch.randn(1, total, h, k_dim, device=device, dtype=torch.float32) * 0.12
    k = torch.randn(1, total, h, k_dim, device=device, dtype=torch.float32) * 0.12
    v = torch.randn(1, total, h, v_dim, device=device, dtype=torch.float32)

    init_state = (
        torch.randn(n_seq, h, k_dim, v_dim, device=device, dtype=torch.float32) * 0.05,
        torch.randn(n_seq, h, v_dim, device=device, dtype=torch.float32) * 0.05,
        torch.randn(n_seq, h, k_dim, device=device, dtype=torch.float32) * 0.05,
        torch.randn(n_seq, h, k_dim, device=device, dtype=torch.float32) * 0.05,
        torch.randn(n_seq, h, v_dim, device=device, dtype=torch.float32) * 0.05,
        torch.randn(n_seq, h, k_dim, device=device, dtype=torch.float32) * 0.05,
        torch.rand(n_seq, h, device=device, dtype=torch.float32) * 3.0,
    )

    parts = []
    st_ref_chunks = [[] for _ in range(7)]
    for i, (s, e) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
        o_i, st_i, _ = approxnet_v3_linear_attention(
            q=q[:, s:e],
            k=k[:, s:e],
            v=v[:, s:e],
            initial_state=tuple(x[i : i + 1].clone() for x in init_state),
            output_final_state=True,
            beta_denom_eps=1e-5,
            score_clip=12.0,
        )
        parts.append(o_i)
        assert st_i is not None
        for idx in range(7):
            st_ref_chunks[idx].append(st_i[idx][0])
    out_ref = torch.cat(parts, dim=1)
    st_ref = tuple(torch.stack(chunks, dim=0) for chunks in st_ref_chunks)

    out_tri, st_tri, _ = approxnet_v3_linear_attention_triton(
        q=q,
        k=k,
        v=v,
        initial_state=tuple(x.clone() for x in init_state),
        output_final_state=True,
        cu_seqlens=cu,
        beta_denom_eps=1e-5,
        score_clip=12.0,
        recompute_chunk_size=64,
    )
    assert st_ref is not None and st_tri is not None
    assert torch.allclose(out_ref.float(), out_tri.float(), atol=1e-3, rtol=1e-3), \
        f"varlen fwd max diff: {(out_ref.float() - out_tri.float()).abs().max().item()}"
    for x, y in zip(st_ref, st_tri):
        assert torch.allclose(x.float(), y.float(), atol=1e-3, rtol=1e-3)


def _run_module_smoke_case() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(20260415 + 3)
    device = torch.device("cuda")
    module = ApproxNetV3LinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=4,
        num_kv_heads=4,
        use_short_conv=False,
        output_norm="identity",
        beta_denom_eps=1e-5,
        score_clip=12.0,
        use_triton=True,
        recompute_chunk_size=64,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 21, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.tensor(
        [
            [1] * 21,
            [1] * 14 + [0] * 7,
        ],
        device=device,
        dtype=torch.int64,
    )
    out, attn_weights, cache = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert out.shape == hidden_states.shape
    assert attn_weights is None
    assert cache is None
    assert hidden_states.grad is not None
    assert torch.isfinite(out).all() and torch.isfinite(hidden_states.grad).all()


if pytest is not None:
    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="approxnet-v3 import failed.")
    def test_approxnet_v3_dense_chunk_equivalence() -> None:
        _run_dense_chunk_equivalence_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="approxnet-v3 import failed.")
    def test_approxnet_v3_varlen_equivalence() -> None:
        _run_varlen_equivalence_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="approxnet-v3 import failed.")
    def test_approxnet_v3_triton_vs_torch() -> None:
        _run_triton_vs_torch_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="approxnet-v3 import failed.")
    def test_approxnet_v3_module_smoke() -> None:
        _run_module_smoke_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="approxnet-v3 import failed.")
    def test_approxnet_v3_triton_stateful_dense_vs_torch() -> None:
        _run_triton_stateful_dense_vs_torch_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="approxnet-v3 import failed.")
    def test_approxnet_v3_triton_stateful_varlen_vs_torch() -> None:
        _run_triton_stateful_varlen_vs_torch_case()


def _run_sigmoid_gate_dense_chunk_equivalence_case() -> None:
    torch.manual_seed(20260416)
    q = torch.randn(2, 15, 3, 12, dtype=torch.float32) * 0.5
    k = torch.randn(2, 15, 3, 12, dtype=torch.float32) * 0.5
    v = torch.randn(2, 15, 3, 8, dtype=torch.float32)

    out_full, st_full, _ = approxnet_v3_linear_attention(
        q=q, k=k, v=v, output_final_state=True, use_sigmoid_gate=True,
    )
    out_a, st_a, _ = approxnet_v3_linear_attention(
        q=q[:, :6], k=k[:, :6], v=v[:, :6], output_final_state=True, use_sigmoid_gate=True,
    )
    out_b, st_b, _ = approxnet_v3_linear_attention(
        q=q[:, 6:], k=k[:, 6:], v=v[:, 6:], initial_state=st_a, output_final_state=True, use_sigmoid_gate=True,
    )
    out_cat = torch.cat([out_a, out_b], dim=1)
    assert torch.allclose(out_full, out_cat, atol=1e-5), f"sigmoid gate chunk mismatch: {(out_full - out_cat).abs().max()}"
    for i in range(7):
        assert torch.allclose(st_full[i], st_b[i], atol=1e-5), f"sigmoid gate state[{i}] mismatch"


def _run_sigmoid_gate_gradient_case() -> None:
    torch.manual_seed(20260416)
    q = (torch.randn(2, 10, 3, 12, dtype=torch.float32) * 0.5).requires_grad_(True)
    k = (torch.randn(2, 10, 3, 12, dtype=torch.float32) * 0.5).requires_grad_(True)
    v = torch.randn(2, 10, 3, 8, dtype=torch.float32).requires_grad_(True)
    out, _, _ = approxnet_v3_linear_attention(q=q, k=k, v=v, use_sigmoid_gate=True)
    loss = out.sum()
    loss.backward()
    assert q.grad is not None and not torch.isnan(q.grad).any(), "q.grad has NaN"
    assert k.grad is not None and not torch.isnan(k.grad).any(), "k.grad has NaN"
    assert v.grad is not None and not torch.isnan(v.grad).any(), "v.grad has NaN"
    assert (q.grad.abs().max() > 0), "q.grad is all zeros"


def main() -> None:
    if _IMPORT_ERROR is not None:
        print(f"skip: import failed: {_IMPORT_ERROR}")
        return
    _run_dense_chunk_equivalence_case()
    print("  [pass] dense chunk equivalence")
    _run_varlen_equivalence_case()
    print("  [pass] varlen equivalence")
    _run_triton_vs_torch_case()
    print("  [pass] triton vs torch (fwd+bwd)")
    _run_triton_stateful_dense_vs_torch_case()
    print("  [pass] triton stateful dense vs torch")
    _run_triton_stateful_varlen_vs_torch_case()
    print("  [pass] triton stateful varlen vs torch")
    _run_module_smoke_case()
    print("  [pass] module smoke")
    _run_sigmoid_gate_dense_chunk_equivalence_case()
    print("  [pass] sigmoid gate dense chunk equivalence")
    _run_sigmoid_gate_gradient_case()
    print("  [pass] sigmoid gate gradient")
    print("ok: approxnet-v3 tests passed")


if __name__ == "__main__":
    main()
