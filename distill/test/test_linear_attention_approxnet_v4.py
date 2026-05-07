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
    from distill.models.linear_attention_approxnet_v4 import (
        ApproxNetV4LinearAttention,
        approxnet_v4_linear_attention,
    )
    from distill.models.linear_attention_approxnet_v4_triton import (
        _TRITON_AVAILABLE as _APPROXNET_V4_TRITON_AVAILABLE,
        approxnet_v4_linear_attention_triton,
    )
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    ApproxNetV4LinearAttention = None
    approxnet_v4_linear_attention = None
    approxnet_v4_linear_attention_triton = None
    _APPROXNET_V4_TRITON_AVAILABLE = False


def _run_dense_chunk_equivalence_case() -> None:
    torch.manual_seed(20260427)
    B, T, H, K, V = 2, 15, 3, 12, 8
    q = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
    k = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
    v = torch.randn(B, T, H, V, dtype=torch.float32)
    ga = torch.randn(H, dtype=torch.float32) * 0.3 + 1.0
    gb = torch.randn(H, dtype=torch.float32) * 0.1

    out_full, st_full, _ = approxnet_v4_linear_attention(
        q=q, k=k, v=v, gate_alpha=ga, gate_bias=gb,
        output_final_state=True, z_score_eps=1.0,
    )
    out_a, st_a, _ = approxnet_v4_linear_attention(
        q=q[:, :6], k=k[:, :6], v=v[:, :6], gate_alpha=ga, gate_bias=gb,
        output_final_state=True, z_score_eps=1.0,
    )
    out_b, st_b, _ = approxnet_v4_linear_attention(
        q=q[:, 6:], k=k[:, 6:], v=v[:, 6:], gate_alpha=ga, gate_bias=gb,
        initial_state=st_a, output_final_state=True, z_score_eps=1.0,
    )

    assert st_full is not None and st_b is not None
    assert torch.allclose(out_full[:, :6], out_a, atol=1e-5, rtol=1e-5), \
        f"chunk A mismatch: {(out_full[:, :6] - out_a).abs().max().item()}"
    assert torch.allclose(out_full[:, 6:], out_b, atol=1e-5, rtol=1e-5), \
        f"chunk B mismatch: {(out_full[:, 6:] - out_b).abs().max().item()}"
    for i, (x, y) in enumerate(zip(st_full, st_b)):
        assert torch.allclose(x, y, atol=1e-5, rtol=1e-5), f"state[{i}] mismatch"


def _run_varlen_equivalence_case() -> None:
    torch.manual_seed(20260427 + 1)
    H, K, V = 2, 10, 6
    q = torch.randn(1, 17, H, K, dtype=torch.float32) * 0.4
    k = torch.randn(1, 17, H, K, dtype=torch.float32) * 0.4
    v = torch.randn(1, 17, H, V, dtype=torch.float32)
    cu = torch.tensor([0, 5, 11, 17], dtype=torch.long)
    ga = torch.ones(H, dtype=torch.float32)
    gb = torch.zeros(H, dtype=torch.float32)

    out_packed, st_packed, _ = approxnet_v4_linear_attention(
        q=q, k=k, v=v, gate_alpha=ga, gate_bias=gb,
        cu_seqlens=cu, output_final_state=True, z_score_eps=1.0,
    )

    parts = []
    states = [[] for _ in range(8)]
    for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
        o_i, st_i, _ = approxnet_v4_linear_attention(
            q=q[:, s:e], k=k[:, s:e], v=v[:, s:e],
            gate_alpha=ga, gate_bias=gb,
            output_final_state=True, z_score_eps=1.0,
        )
        parts.append(o_i)
        assert st_i is not None
        for idx in range(8):
            states[idx].append(st_i[idx][0])

    out_ref = torch.cat(parts, dim=1)
    assert st_packed is not None
    assert torch.allclose(out_packed, out_ref, atol=1e-5, rtol=1e-5)
    for idx in range(8):
        assert torch.allclose(st_packed[idx], torch.stack(states[idx], dim=0), atol=1e-5, rtol=1e-5)


def _run_triton_vs_torch_case() -> None:
    if not torch.cuda.is_available() or not _APPROXNET_V4_TRITON_AVAILABLE:
        return
    torch.manual_seed(20260427 + 2)
    device = torch.device("cuda")
    B, T, H, K, V = 2, 19, 4, 16, 8

    q_ref = (torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.10).requires_grad_(True)
    k_ref = (torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.10).requires_grad_(True)
    v_ref = torch.randn(B, T, H, V, device=device, dtype=torch.float32, requires_grad=True)
    ga_ref = (torch.randn(H, device=device, dtype=torch.float32) * 0.3 + 1.0).requires_grad_(True)
    gb_ref = (torch.randn(H, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)

    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)
    v_tri = v_ref.detach().clone().requires_grad_(True)
    ga_tri = ga_ref.detach().clone().requires_grad_(True)
    gb_tri = gb_ref.detach().clone().requires_grad_(True)

    out_ref, _, _ = approxnet_v4_linear_attention(
        q=q_ref, k=k_ref, v=v_ref, gate_alpha=ga_ref, gate_bias=gb_ref,
        z_score_eps=1.0,
    )
    out_tri, _, _ = approxnet_v4_linear_attention_triton(
        q=q_tri, k=k_tri, v=v_tri, gate_alpha=ga_tri, gate_bias=gb_tri,
        z_score_eps=1.0, recompute_chunk_size=64,
    )
    assert torch.allclose(out_ref.float(), out_tri.float(), atol=6e-4, rtol=6e-4), \
        f"fwd max diff: {(out_ref.float() - out_tri.float()).abs().max().item()}"

    loss_ref = out_ref.float().pow(2).mean()
    loss_tri = out_tri.float().pow(2).mean()
    loss_ref.backward()
    loss_tri.backward()

    for name, ref_g, tri_g in [
        ("dq", q_ref.grad, q_tri.grad),
        ("dk", k_ref.grad, k_tri.grad),
        ("dv", v_ref.grad, v_tri.grad),
        ("d_alpha", ga_ref.grad, ga_tri.grad),
        ("d_bias", gb_ref.grad, gb_tri.grad),
    ]:
        assert ref_g is not None and tri_g is not None, f"{name} grad is None"
        assert torch.isfinite(tri_g).all(), f"{name} has non-finite values"
        assert torch.allclose(ref_g.float(), tri_g.float(), atol=8e-4, rtol=2e-3), \
            f"{name} max diff: {(ref_g.float() - tri_g.float()).abs().max().item()}"


def _run_triton_stateful_dense_vs_torch_case() -> None:
    if not torch.cuda.is_available() or not _APPROXNET_V4_TRITON_AVAILABLE:
        return
    torch.manual_seed(20260427 + 4)
    device = torch.device("cuda")

    B, T, H, K, V = 2, 17, 3, 16, 8
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.15
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float32) * 0.15
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float32)
    ga = torch.ones(H, device=device, dtype=torch.float32)
    gb = torch.zeros(H, device=device, dtype=torch.float32)

    init_state = (
        torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.05,
        torch.randn(B, H, V, device=device, dtype=torch.float32) * 0.05,
        torch.randn(B, H, K, device=device, dtype=torch.float32) * 0.05,
        torch.randn(B, H, V, device=device, dtype=torch.float32) * 0.05,
        torch.randn(B, H, K, device=device, dtype=torch.float32) * 0.05,
        torch.rand(B, H, device=device, dtype=torch.float32) * 5.0,
        torch.randn(B, H, device=device, dtype=torch.float32) * 0.1,
        torch.rand(B, H, device=device, dtype=torch.float32) * 0.5,
    )

    out_ref, st_ref, _ = approxnet_v4_linear_attention(
        q=q, k=k, v=v, gate_alpha=ga, gate_bias=gb,
        initial_state=tuple(x.clone() for x in init_state),
        output_final_state=True, z_score_eps=1.0,
    )
    out_tri, st_tri, _ = approxnet_v4_linear_attention_triton(
        q=q, k=k, v=v, gate_alpha=ga, gate_bias=gb,
        initial_state=tuple(x.clone() for x in init_state),
        output_final_state=True, z_score_eps=1.0, recompute_chunk_size=64,
    )
    assert st_ref is not None and st_tri is not None
    assert torch.allclose(out_ref.float(), out_tri.float(), atol=8e-4, rtol=8e-4), \
        f"stateful fwd max diff: {(out_ref.float() - out_tri.float()).abs().max().item()}"
    for i, (x, y) in enumerate(zip(st_ref, st_tri)):
        assert torch.allclose(x.float(), y.float(), atol=1e-3, rtol=1e-3), \
            f"state[{i}] max diff: {(x.float() - y.float()).abs().max().item()}"


def _run_triton_stateful_varlen_vs_torch_case() -> None:
    if not torch.cuda.is_available() or not _APPROXNET_V4_TRITON_AVAILABLE:
        return
    torch.manual_seed(20260427 + 5)
    device = torch.device("cuda")

    H, K, V = 2, 12, 6
    cu = torch.tensor([0, 5, 11, 17], dtype=torch.long, device=device)
    n_seq = cu.numel() - 1
    total = int(cu[-1].item())
    q = torch.randn(1, total, H, K, device=device, dtype=torch.float32) * 0.12
    k = torch.randn(1, total, H, K, device=device, dtype=torch.float32) * 0.12
    v = torch.randn(1, total, H, V, device=device, dtype=torch.float32)
    ga = torch.ones(H, device=device, dtype=torch.float32)
    gb = torch.zeros(H, device=device, dtype=torch.float32)

    init_state = (
        torch.randn(n_seq, H, K, V, device=device, dtype=torch.float32) * 0.05,
        torch.randn(n_seq, H, V, device=device, dtype=torch.float32) * 0.05,
        torch.randn(n_seq, H, K, device=device, dtype=torch.float32) * 0.05,
        torch.randn(n_seq, H, V, device=device, dtype=torch.float32) * 0.05,
        torch.randn(n_seq, H, K, device=device, dtype=torch.float32) * 0.05,
        torch.rand(n_seq, H, device=device, dtype=torch.float32) * 3.0,
        torch.randn(n_seq, H, device=device, dtype=torch.float32) * 0.1,
        torch.rand(n_seq, H, device=device, dtype=torch.float32) * 0.5,
    )

    parts = []
    st_ref_chunks = [[] for _ in range(8)]
    for i, (s, e) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
        o_i, st_i, _ = approxnet_v4_linear_attention(
            q=q[:, s:e], k=k[:, s:e], v=v[:, s:e],
            gate_alpha=ga, gate_bias=gb,
            initial_state=tuple(x[i : i + 1].clone() for x in init_state),
            output_final_state=True, z_score_eps=1.0,
        )
        parts.append(o_i)
        assert st_i is not None
        for idx in range(8):
            st_ref_chunks[idx].append(st_i[idx][0])
    out_ref = torch.cat(parts, dim=1)
    st_ref = tuple(torch.stack(chunks, dim=0) for chunks in st_ref_chunks)

    out_tri, st_tri, _ = approxnet_v4_linear_attention_triton(
        q=q, k=k, v=v, gate_alpha=ga, gate_bias=gb,
        initial_state=tuple(x.clone() for x in init_state),
        output_final_state=True, cu_seqlens=cu,
        z_score_eps=1.0, recompute_chunk_size=64,
    )
    assert st_ref is not None and st_tri is not None
    assert torch.allclose(out_ref.float(), out_tri.float(), atol=1e-3, rtol=1e-3), \
        f"varlen fwd max diff: {(out_ref.float() - out_tri.float()).abs().max().item()}"
    for i, (x, y) in enumerate(zip(st_ref, st_tri)):
        assert torch.allclose(x.float(), y.float(), atol=1e-3, rtol=1e-3), \
            f"state[{i}] max diff: {(x.float() - y.float()).abs().max().item()}"


def _run_module_smoke_case() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(20260427 + 3)
    device = torch.device("cuda")
    module = ApproxNetV4LinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=4,
        num_kv_heads=4,
        use_short_conv=False,
        output_norm="identity",
        z_score_eps=1.0,
        gate_alpha_init=1.0,
        gate_bias_init=0.0,
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
    assert module.gate_alpha.grad is not None, "gate_alpha did not receive gradient"
    assert module.gate_bias.grad is not None, "gate_bias did not receive gradient"


def _run_gradient_case() -> None:
    torch.manual_seed(20260427 + 6)
    q = (torch.randn(2, 10, 3, 12, dtype=torch.float32) * 0.5).requires_grad_(True)
    k = (torch.randn(2, 10, 3, 12, dtype=torch.float32) * 0.5).requires_grad_(True)
    v = torch.randn(2, 10, 3, 8, dtype=torch.float32).requires_grad_(True)
    ga = (torch.ones(3, dtype=torch.float32)).requires_grad_(True)
    gb = (torch.zeros(3, dtype=torch.float32)).requires_grad_(True)

    out, _, _ = approxnet_v4_linear_attention(
        q=q, k=k, v=v, gate_alpha=ga, gate_bias=gb, z_score_eps=1.0,
    )
    loss = out.sum()
    loss.backward()
    assert q.grad is not None and not torch.isnan(q.grad).any(), "q.grad has NaN"
    assert k.grad is not None and not torch.isnan(k.grad).any(), "k.grad has NaN"
    assert v.grad is not None and not torch.isnan(v.grad).any(), "v.grad has NaN"
    assert ga.grad is not None and not torch.isnan(ga.grad).any(), "gate_alpha.grad has NaN"
    assert gb.grad is not None and not torch.isnan(gb.grad).any(), "gate_bias.grad has NaN"
    assert (q.grad.abs().max() > 0), "q.grad is all zeros"


if pytest is not None:
    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="approxnet-v4 import failed.")
    def test_approxnet_v4_dense_chunk_equivalence() -> None:
        _run_dense_chunk_equivalence_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="approxnet-v4 import failed.")
    def test_approxnet_v4_varlen_equivalence() -> None:
        _run_varlen_equivalence_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="approxnet-v4 import failed.")
    def test_approxnet_v4_triton_vs_torch() -> None:
        _run_triton_vs_torch_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="approxnet-v4 import failed.")
    def test_approxnet_v4_module_smoke() -> None:
        _run_module_smoke_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="approxnet-v4 import failed.")
    def test_approxnet_v4_triton_stateful_dense_vs_torch() -> None:
        _run_triton_stateful_dense_vs_torch_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="approxnet-v4 import failed.")
    def test_approxnet_v4_triton_stateful_varlen_vs_torch() -> None:
        _run_triton_stateful_varlen_vs_torch_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="approxnet-v4 import failed.")
    def test_approxnet_v4_gradient() -> None:
        _run_gradient_case()


def main() -> None:
    if _IMPORT_ERROR is not None:
        print(f"skip: import failed: {_IMPORT_ERROR}")
        return
    _run_dense_chunk_equivalence_case()
    print("  [pass] dense chunk equivalence")
    _run_varlen_equivalence_case()
    print("  [pass] varlen equivalence")
    _run_gradient_case()
    print("  [pass] gradient (PyTorch)")
    _run_triton_vs_torch_case()
    print("  [pass] triton vs torch (fwd+bwd)")
    _run_triton_stateful_dense_vs_torch_case()
    print("  [pass] triton stateful dense vs torch")
    _run_triton_stateful_varlen_vs_torch_case()
    print("  [pass] triton stateful varlen vs torch")
    _run_module_smoke_case()
    print("  [pass] module smoke")
    print("ok: approxnet-v4 tests passed")


if __name__ == "__main__":
    main()
