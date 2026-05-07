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
    from distill.models.mean_delta_net import MeanDeltaNet
    from distill.models.mean_delta_rule import fused_recurrent_mean_delta_rule
    from distill.models.mean_delta_rule_naive import mean_delta_rule_recurrence
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    MeanDeltaNet = None
    fused_recurrent_mean_delta_rule = None
    mean_delta_rule_recurrence = None


def _run_dense_chunk_equivalence_case() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(20260409)
    device = torch.device("cuda")
    q = torch.randn(2, 17, 3, 32, device=device, dtype=torch.float32)
    k = torch.randn(2, 17, 3, 32, device=device, dtype=torch.float32)
    v = torch.randn(2, 17, 3, 24, device=device, dtype=torch.float32)

    out_full, state_full = fused_recurrent_mean_delta_rule(
        q=q,
        k=k,
        v=v,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    out_a, state_a = fused_recurrent_mean_delta_rule(
        q=q[:, :7],
        k=k[:, :7],
        v=v[:, :7],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    out_b, state_b = fused_recurrent_mean_delta_rule(
        q=q[:, 7:],
        k=k[:, 7:],
        v=v[:, 7:],
        initial_state=state_a,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    assert state_full is not None and state_b is not None
    assert torch.allclose(out_full[:, :7], out_a, atol=3e-4, rtol=2e-3)
    assert torch.allclose(out_full[:, 7:], out_b, atol=3e-4, rtol=2e-3)
    for x, y in zip(state_full, state_b):
        assert torch.allclose(x, y, atol=5e-4, rtol=3e-3)


def _run_varlen_equivalence_case() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(20260410)
    device = torch.device("cuda")
    q = torch.randn(1, 19, 4, 24, device=device, dtype=torch.float32)
    k = torch.randn(1, 19, 4, 24, device=device, dtype=torch.float32)
    v = torch.randn(1, 19, 4, 16, device=device, dtype=torch.float32)
    cu = torch.tensor([0, 4, 11, 19], dtype=torch.long, device=device)

    out_packed, state_packed = fused_recurrent_mean_delta_rule(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    out_ref_parts = []
    kv_ref, mv_ref, mk_ref, c_ref = [], [], [], []
    for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
        out_seg, st_seg = fused_recurrent_mean_delta_rule(
            q=q[:, s:e],
            k=k[:, s:e],
            v=v[:, s:e],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        out_ref_parts.append(out_seg)
        assert st_seg is not None
        kv_ref.append(st_seg[0][0])
        mv_ref.append(st_seg[1][0])
        mk_ref.append(st_seg[2][0])
        c_ref.append(st_seg[3][0])
    out_ref = torch.cat(out_ref_parts, dim=1)

    assert state_packed is not None
    assert torch.allclose(out_packed, out_ref, atol=4e-4, rtol=3e-3)
    assert torch.allclose(state_packed[0], torch.stack(kv_ref), atol=5e-4, rtol=3e-3)
    assert torch.allclose(state_packed[1], torch.stack(mv_ref), atol=5e-4, rtol=3e-3)
    assert torch.allclose(state_packed[2], torch.stack(mk_ref), atol=5e-4, rtol=3e-3)
    assert torch.allclose(state_packed[3], torch.stack(c_ref), atol=1e-5, rtol=1e-5)


def _run_triton_vs_naive_grad_case() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(20260411)
    device = torch.device("cuda")

    q_ref = torch.randn(2, 13, 3, 24, device=device, dtype=torch.float32, requires_grad=True)
    k_ref = torch.randn(2, 13, 3, 24, device=device, dtype=torch.float32, requires_grad=True)
    v_ref = torch.randn(2, 13, 3, 16, device=device, dtype=torch.float32, requires_grad=True)

    q_opt = q_ref.detach().clone().requires_grad_(True)
    k_opt = k_ref.detach().clone().requires_grad_(True)
    v_opt = v_ref.detach().clone().requires_grad_(True)

    out_ref, st_ref = mean_delta_rule_recurrence(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        output_final_state=True,
    )
    out_opt, st_opt = fused_recurrent_mean_delta_rule(
        q=q_opt,
        k=k_opt,
        v=v_opt,
        output_final_state=True,
    )

    assert st_ref is not None and st_opt is not None
    assert torch.allclose(out_ref.float(), out_opt.float(), atol=5e-4, rtol=4e-3)
    for x, y in zip(st_ref, st_opt):
        assert torch.allclose(x.float(), y.float(), atol=6e-4, rtol=5e-3)

    loss_ref = out_ref.float().pow(2).mean()
    loss_ref = loss_ref + st_ref[0].float().pow(2).mean() * 1e-4
    loss_ref = loss_ref + st_ref[1].float().pow(2).mean() * 1e-4
    loss_ref = loss_ref + st_ref[2].float().pow(2).mean() * 1e-4

    loss_opt = out_opt.float().pow(2).mean()
    loss_opt = loss_opt + st_opt[0].float().pow(2).mean() * 1e-4
    loss_opt = loss_opt + st_opt[1].float().pow(2).mean() * 1e-4
    loss_opt = loss_opt + st_opt[2].float().pow(2).mean() * 1e-4

    loss_ref.backward()
    loss_opt.backward()

    assert q_ref.grad is not None and k_ref.grad is not None and v_ref.grad is not None
    assert q_opt.grad is not None and k_opt.grad is not None and v_opt.grad is not None
    assert torch.isfinite(q_opt.grad).all() and torch.isfinite(k_opt.grad).all() and torch.isfinite(v_opt.grad).all()
    assert torch.allclose(q_ref.grad.float(), q_opt.grad.float(), atol=3e-4, rtol=5e-2)
    assert torch.allclose(k_ref.grad.float(), k_opt.grad.float(), atol=3e-4, rtol=5e-2)
    assert torch.allclose(v_ref.grad.float(), v_opt.grad.float(), atol=3e-4, rtol=5e-2)


def _run_module_smoke_case() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(20260412)
    device = torch.device("cuda")
    module = MeanDeltaNet(
        hidden_size=64,
        num_heads=4,
        use_gate=False,
        use_short_conv=False,
        value_l2_norm=True,
        qk_norm="l2",
        mean_recompute_chunk_size=64,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 23, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.tensor(
        [
            [1] * 23,
            [1] * 17 + [0] * 6,
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
    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()


if pytest is not None:
    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="mean-delta import failed.")
    def test_mean_delta_dense_chunk_equivalence() -> None:
        _run_dense_chunk_equivalence_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="mean-delta import failed.")
    def test_mean_delta_varlen_equivalence() -> None:
        _run_varlen_equivalence_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="mean-delta import failed.")
    def test_mean_delta_triton_vs_naive_grad() -> None:
        _run_triton_vs_naive_grad_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason="mean-delta import failed.")
    def test_mean_delta_module_smoke() -> None:
        _run_module_smoke_case()


def main() -> None:
    if _IMPORT_ERROR is not None:
        print(f"skip: import failed: {_IMPORT_ERROR}")
        return
    _run_dense_chunk_equivalence_case()
    _run_varlen_equivalence_case()
    _run_triton_vs_naive_grad_case()
    _run_module_smoke_case()
    print("ok: mean-delta tests passed")


if __name__ == "__main__":
    main()
