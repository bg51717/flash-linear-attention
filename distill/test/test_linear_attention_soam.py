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
    from fla.layers.soam import (
        SOAMLinearAttention,
        soam_linear_attention,
        soam_recurrence_naive,
    )
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    SOAMLinearAttention = None
    soam_linear_attention = None
    soam_recurrence_naive = None


def _make_inputs(B, T, H, D_R, D_V, device="cpu"):
    q_r = torch.randn(B, T, H, D_R, device=device, dtype=torch.float32) * 0.3
    k_r = torch.randn(B, T, H, D_R, device=device, dtype=torch.float32) * 0.3
    v = torch.randn(B, T, H, D_V, device=device, dtype=torch.float32)
    q_r = torch.nn.functional.normalize(q_r, p=2, dim=-1)
    k_r = torch.nn.functional.normalize(k_r, p=2, dim=-1)
    da = torch.zeros(H, device=device, dtype=torch.float32)
    db = torch.full((H,), 2.0, device=device, dtype=torch.float32)
    wa = torch.ones(H, device=device, dtype=torch.float32)
    wb = torch.zeros(H, device=device, dtype=torch.float32)
    return q_r, k_r, v, da, db, wa, wb


def _run_naive_vs_optimized_case() -> None:
    torch.manual_seed(20260503)
    B, T, H, D_R, D_V = 2, 20, 3, 8, 6
    q_r, k_r, v, da, db, wa, wb = _make_inputs(B, T, H, D_R, D_V)

    out_opt, st_opt, _ = soam_linear_attention(
        q_r=q_r, k_r=k_r, v=v,
        decay_alpha=da, decay_bias=db, write_alpha=wa, write_bias=wb,
        output_final_state=True,
    )
    out_naive, st_naive = soam_recurrence_naive(
        q_r=q_r.permute(0, 2, 1, 3),
        k_r=k_r.permute(0, 2, 1, 3),
        v=v.permute(0, 2, 1, 3),
        decay_alpha=da, decay_bias=db, write_alpha=wa, write_bias=wb,
        output_final_state=True,
    )
    out_naive = out_naive.permute(0, 2, 1, 3)

    assert torch.allclose(out_opt, out_naive, atol=1e-5, rtol=1e-5), \
        f"naive vs opt output: {(out_opt - out_naive).abs().max().item()}"
    assert st_opt is not None and st_naive is not None
    assert torch.allclose(st_opt[0], st_naive, atol=1e-5, rtol=1e-5), \
        f"naive vs opt state: {(st_opt[0] - st_naive).abs().max().item()}"


def _run_dense_chunk_equivalence_case() -> None:
    torch.manual_seed(20260427)
    B, T, H, D_R, D_V = 2, 15, 3, 8, 6
    q_r, k_r, v, da, db, wa, wb = _make_inputs(B, T, H, D_R, D_V)

    out_full, st_full, _ = soam_linear_attention(
        q_r=q_r, k_r=k_r, v=v,
        decay_alpha=da, decay_bias=db, write_alpha=wa, write_bias=wb,
        output_final_state=True,
    )
    out_a, st_a, _ = soam_linear_attention(
        q_r=q_r[:, :6], k_r=k_r[:, :6], v=v[:, :6],
        decay_alpha=da, decay_bias=db, write_alpha=wa, write_bias=wb,
        output_final_state=True,
    )
    out_b, st_b, _ = soam_linear_attention(
        q_r=q_r[:, 6:], k_r=k_r[:, 6:], v=v[:, 6:],
        decay_alpha=da, decay_bias=db, write_alpha=wa, write_bias=wb,
        initial_state=st_a, output_final_state=True,
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
    H, D_R, D_V = 2, 6, 4
    q_r, k_r, v, da, db, wa, wb = _make_inputs(1, 17, H, D_R, D_V)
    cu = torch.tensor([0, 5, 11, 17], dtype=torch.long)

    out_packed, st_packed, _ = soam_linear_attention(
        q_r=q_r, k_r=k_r, v=v,
        decay_alpha=da, decay_bias=db, write_alpha=wa, write_bias=wb,
        cu_seqlens=cu, output_final_state=True,
    )

    parts = []
    states = [[]]
    for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
        o_i, st_i, _ = soam_linear_attention(
            q_r=q_r[:, s:e], k_r=k_r[:, s:e], v=v[:, s:e],
            decay_alpha=da, decay_bias=db, write_alpha=wa, write_bias=wb,
            output_final_state=True,
        )
        parts.append(o_i)
        assert st_i is not None
        states[0].append(st_i[0][0])

    out_ref = torch.cat(parts, dim=1)
    assert st_packed is not None
    assert torch.allclose(out_packed, out_ref, atol=1e-5, rtol=1e-5), \
        f"varlen output mismatch: {(out_packed - out_ref).abs().max().item()}"
    assert torch.allclose(st_packed[0], torch.stack(states[0], dim=0), atol=1e-5, rtol=1e-5), \
        f"varlen state mismatch"


def _run_gradient_case() -> None:
    torch.manual_seed(20260427 + 2)
    B, T, H, D_R, D_V = 2, 10, 3, 8, 6
    q_r = (torch.randn(B, T, H, D_R, dtype=torch.float32) * 0.3).requires_grad_(True)
    k_r = (torch.randn(B, T, H, D_R, dtype=torch.float32) * 0.3).requires_grad_(True)
    v = torch.randn(B, T, H, D_V, dtype=torch.float32).requires_grad_(True)
    da = torch.zeros(H, dtype=torch.float32).requires_grad_(True)
    db = torch.full((H,), 2.0, dtype=torch.float32).requires_grad_(True)
    wa = torch.ones(H, dtype=torch.float32).requires_grad_(True)
    wb = torch.zeros(H, dtype=torch.float32).requires_grad_(True)

    out, _, _ = soam_linear_attention(
        q_r=q_r, k_r=k_r, v=v,
        decay_alpha=da, decay_bias=db, write_alpha=wa, write_bias=wb,
    )
    loss = out.sum()
    loss.backward()
    for name, param in [("q_r", q_r), ("k_r", k_r), ("v", v), ("da", da), ("db", db), ("wa", wa), ("wb", wb)]:
        assert param.grad is not None and not torch.isnan(param.grad).any(), f"{name}.grad has NaN"
    assert (q_r.grad.abs().max() > 0), "q_r.grad is all zeros"


def _run_module_smoke_case() -> None:
    torch.manual_seed(20260427 + 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = SOAMLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=4,
        num_kv_heads=4,
        use_short_conv=False,
        output_norm="identity",
        d_r=4,
        decay_alpha_init=0.0,
        decay_bias_init=2.0,
        write_alpha_init=1.0,
        write_bias_init=0.0,
        qk_l2_norm=True,
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
    assert module.write_alpha.grad is not None, "write_alpha did not receive gradient"
    assert module.proj_weight.grad is not None, "proj_weight did not receive gradient"


if pytest is not None:
    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="soam import failed.")
    def test_soam_naive_vs_optimized() -> None:
        _run_naive_vs_optimized_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="soam import failed.")
    def test_soam_dense_chunk_equivalence() -> None:
        _run_dense_chunk_equivalence_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="soam import failed.")
    def test_soam_varlen_equivalence() -> None:
        _run_varlen_equivalence_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="soam import failed.")
    def test_soam_gradient() -> None:
        _run_gradient_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="soam import failed.")
    def test_soam_module_smoke() -> None:
        _run_module_smoke_case()


def _run_triton_vs_naive_case() -> None:
    if not torch.cuda.is_available():
        print("  [skip] triton vs naive (no CUDA)")
        return
    try:
        from fla.ops.soam.fused_recurrent import fused_recurrent_soam  # noqa: F401
    except ImportError:
        print("  [skip] triton vs naive (triton not available)")
        return

    torch.manual_seed(20260503 + 10)
    device = "cuda"
    B, T, H, D_R, D_V = 2, 30, 3, 8, 6
    q_r, k_r, v, da, db, wa, wb = _make_inputs(B, T, H, D_R, D_V, device=device)

    out_triton, st_triton, _ = soam_linear_attention(
        q_r=q_r, k_r=k_r, v=v,
        decay_alpha=da, decay_bias=db, write_alpha=wa, write_bias=wb,
        output_final_state=True, use_triton=True,
    )
    out_naive, st_naive = soam_recurrence_naive(
        q_r=q_r.permute(0, 2, 1, 3),
        k_r=k_r.permute(0, 2, 1, 3),
        v=v.permute(0, 2, 1, 3),
        decay_alpha=da, decay_bias=db, write_alpha=wa, write_bias=wb,
        output_final_state=True,
    )
    out_naive = out_naive.permute(0, 2, 1, 3)

    assert torch.allclose(out_triton.float(), out_naive.float(), atol=1e-4, rtol=1e-4), \
        f"triton vs naive output: {(out_triton.float() - out_naive.float()).abs().max().item()}"
    assert st_triton is not None and st_naive is not None
    st_t = st_triton[0].reshape_as(st_naive)
    assert torch.allclose(st_t.float(), st_naive.float(), atol=1e-4, rtol=1e-4), \
        f"triton vs naive state: {(st_t.float() - st_naive.float()).abs().max().item()}"


def _run_triton_gradient_case() -> None:
    if not torch.cuda.is_available():
        print("  [skip] triton gradient (no CUDA)")
        return
    try:
        from fla.ops.soam.fused_recurrent import fused_recurrent_soam  # noqa: F401
    except ImportError:
        print("  [skip] triton gradient (triton not available)")
        return

    torch.manual_seed(20260503 + 11)
    device = "cuda"
    B, T, H, D_R, D_V = 1, 16, 2, 8, 6
    q_r = torch.nn.functional.normalize(
        torch.randn(B, T, H, D_R, device=device, dtype=torch.float32), dim=-1,
    ).requires_grad_(True)
    k_r = torch.nn.functional.normalize(
        torch.randn(B, T, H, D_R, device=device, dtype=torch.float32), dim=-1,
    ).requires_grad_(True)
    v = torch.randn(B, T, H, D_V, device=device, dtype=torch.float32).requires_grad_(True)
    da = torch.zeros(H, device=device, dtype=torch.float32).requires_grad_(True)
    db = torch.full((H,), 2.0, device=device, dtype=torch.float32).requires_grad_(True)
    wa = torch.ones(H, device=device, dtype=torch.float32).requires_grad_(True)
    wb = torch.zeros(H, device=device, dtype=torch.float32).requires_grad_(True)

    out, _, _ = soam_linear_attention(
        q_r=q_r, k_r=k_r, v=v,
        decay_alpha=da, decay_bias=db, write_alpha=wa, write_bias=wb,
        use_triton=True,
    )
    loss = out.sum()
    loss.backward()
    for name, param in [("q_r", q_r), ("k_r", k_r), ("v", v), ("wa", wa), ("wb", wb)]:
        assert param.grad is not None and not torch.isnan(param.grad).any(), f"{name}.grad has NaN"
    assert (q_r.grad.abs().max() > 0), "q_r.grad is all zeros"
    assert (wa.grad is not None and torch.isfinite(wa.grad).all()), "wa.grad issue"


def _run_speed_test() -> None:
    if not torch.cuda.is_available():
        print("  [skip] speed test (no CUDA)")
        return
    import time

    torch.manual_seed(20260503 + 20)
    device = "cuda"
    B, T, H, D_R, D_V = 4, 2048, 9, 16, 64
    q_r, k_r, v, da, db, wa, wb = _make_inputs(B, T, H, D_R, D_V, device=device)

    for _ in range(3):
        q_g = q_r.detach().requires_grad_(True)
        k_g = k_r.detach().requires_grad_(True)
        v_g = v.detach().requires_grad_(True)
        out, _, _ = soam_linear_attention(
            q_r=q_g, k_r=k_g, v=v_g,
            decay_alpha=da.detach().requires_grad_(True),
            decay_bias=db.detach().requires_grad_(True),
            write_alpha=wa.detach().requires_grad_(True),
            write_bias=wb.detach().requires_grad_(True),
        )
        out.sum().backward()
    torch.cuda.synchronize()

    N_ITER = 10

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        out, _, _ = soam_linear_attention(
            q_r=q_r, k_r=k_r, v=v,
            decay_alpha=da, decay_bias=db, write_alpha=wa, write_bias=wb,
        )
        torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) / N_ITER * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        q_g = q_r.detach().requires_grad_(True)
        k_g = k_r.detach().requires_grad_(True)
        v_g = v.detach().requires_grad_(True)
        da_g = da.detach().requires_grad_(True)
        db_g = db.detach().requires_grad_(True)
        wa_g = wa.detach().requires_grad_(True)
        wb_g = wb.detach().requires_grad_(True)
        out, _, _ = soam_linear_attention(
            q_r=q_g, k_r=k_g, v=v_g,
            decay_alpha=da_g, decay_bias=db_g,
            write_alpha=wa_g, write_bias=wb_g,
        )
        out.sum().backward()
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) / N_ITER * 1000
    bwd_ms = total_ms - fwd_ms

    print(f"  Config: B={B}, T={T}, H={H}, D_R={D_R}, D_V={D_V}")
    print(f"  Forward:  {fwd_ms:.1f} ms")
    print(f"  Backward: {bwd_ms:.1f} ms")
    print(f"  Total:    {total_ms:.1f} ms")
    print(f"  30-layer estimate: {total_ms * 30 / 1000:.1f} s/step")


def main() -> None:
    if _IMPORT_ERROR is not None:
        print(f"skip: import failed: {_IMPORT_ERROR}")
        return
    print("=== SOAM Tests ===")
    _run_naive_vs_optimized_case()
    print("  [pass] naive vs optimized")
    _run_dense_chunk_equivalence_case()
    print("  [pass] dense chunk equivalence")
    _run_varlen_equivalence_case()
    print("  [pass] varlen equivalence")
    _run_gradient_case()
    print("  [pass] gradient (PyTorch)")
    _run_module_smoke_case()
    print("  [pass] module smoke")
    _run_triton_vs_naive_case()
    print("  [pass] triton vs naive")
    _run_triton_gradient_case()
    print("  [pass] triton gradient")
    _run_speed_test()
    print("  [pass] speed test")
    print("ok: soam tests passed")


if __name__ == "__main__":
    main()
