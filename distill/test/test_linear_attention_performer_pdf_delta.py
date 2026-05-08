from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

try:
    import pytest
except Exception:  # pragma: no cover
    pytest = None

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_IMPORT_ERROR: Exception | None = None
try:
    from fla.layers.performer_plus import (
        _PERFORMER_PLUS_TRITON_AVAILABLE,
        PerformerPlusLinearAttention,
        performer_plus_pdf_delta_attention,
    )
    from fla.ops.performer_plus.fused_recurrent import (
        performer_plus_pdf_delta_attention_triton,
    )
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    _PERFORMER_PLUS_TRITON_AVAILABLE = False
    PerformerPlusLinearAttention = None
    performer_plus_pdf_delta_attention = None
    performer_plus_pdf_delta_attention_triton = None


def _run_dense_chunk_equivalence_case() -> None:
    torch.manual_seed(20260406)
    q = torch.randn(2, 9, 3, 7, dtype=torch.float32)
    k = torch.randn(2, 9, 3, 7, dtype=torch.float32)
    v = torch.randn(2, 9, 3, 5, dtype=torch.float32)

    out_full, state_full = performer_plus_pdf_delta_attention(
        q, k, v, output_final_state=True, eps=1e-6
    )
    out_a, state_a = performer_plus_pdf_delta_attention(
        q[:, :4], k[:, :4], v[:, :4], output_final_state=True, eps=1e-6
    )
    out_b, state_b = performer_plus_pdf_delta_attention(
        q[:, 4:],
        k[:, 4:],
        v[:, 4:],
        initial_state=state_a,
        output_final_state=True,
        eps=1e-6,
    )

    assert state_full is not None and state_b is not None
    assert torch.allclose(out_full[:, :4], out_a, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_full[:, 4:], out_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(state_full[0], state_b[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(state_full[1], state_b[1], atol=1e-5, rtol=1e-5)


def _run_varlen_equivalence_case() -> None:
    torch.manual_seed(20260407)
    q = torch.randn(1, 11, 2, 6, dtype=torch.float32)
    k = torch.randn(1, 11, 2, 6, dtype=torch.float32)
    v = torch.randn(1, 11, 2, 4, dtype=torch.float32)
    cu = torch.tensor([0, 5, 8, 11], dtype=torch.long)

    out_packed, state_packed = performer_plus_pdf_delta_attention(
        q,
        k,
        v,
        cu_seqlens=cu,
        output_final_state=True,
        eps=1e-6,
    )

    parts = []
    kv_parts = []
    z_parts = []
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        out_seg, state_seg = performer_plus_pdf_delta_attention(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            output_final_state=True,
            eps=1e-6,
        )
        parts.append(out_seg)
        assert state_seg is not None
        kv_parts.append(state_seg[0][0])
        z_parts.append(state_seg[1][0])

    out_ref = torch.cat(parts, dim=1)
    kv_ref = torch.stack(kv_parts, dim=0)
    z_ref = torch.stack(z_parts, dim=0)

    assert state_packed is not None
    assert torch.allclose(out_packed, out_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(state_packed[0], kv_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(state_packed[1], z_ref, atol=1e-5, rtol=1e-5)


def _run_module_case() -> None:
    if not torch.cuda.is_available() or not _PERFORMER_PLUS_TRITON_AVAILABLE:
        return
    torch.manual_seed(20260408)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=48,
        head_dim=8,
        num_heads=6,
        num_v_heads=6,
        use_short_conv=False,
        performer_nb_features=16,
        performer_state_update='pdf_delta',
        performer_use_triton=True,
        performer_use_control_variate=True,
        performer_use_beta=True,
        performer_use_output_gate=False,
        performer_use_decay=True,
        layer_idx=0,
    ).to(device)
    module.train()

    assert module.pdf_delta_update
    assert not module.use_control_variate
    assert module.use_beta
    assert not module.use_decay

    hidden_states = torch.randn(2, 13, 48, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.tensor(
        [
            [1] * 13,
            [1] * 9 + [0] * 4,
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
    assert torch.isfinite(out).all()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def _run_pdf_delta_triton_vs_torch_case() -> None:
    if not torch.cuda.is_available() or not _PERFORMER_PLUS_TRITON_AVAILABLE:
        return
    torch.manual_seed(20260409)
    device = torch.device('cuda')
    q_ref = (F.softplus(torch.randn(2, 21, 6, 16, device=device, dtype=torch.float32)) + 1e-3).requires_grad_(True)
    k_ref = (F.softplus(torch.randn(2, 21, 6, 16, device=device, dtype=torch.float32)) + 1e-3).requires_grad_(True)
    v_ref = torch.randn(2, 21, 6, 8, device=device, dtype=torch.float32, requires_grad=True)
    q_tri = q_ref.detach().clone().to(torch.bfloat16).requires_grad_(True)
    k_tri = k_ref.detach().clone().to(torch.bfloat16).requires_grad_(True)
    v_tri = v_ref.detach().clone().to(torch.bfloat16).requires_grad_(True)
    rho_ref = torch.sigmoid(torch.randn(2, 21, 6, device=device, dtype=torch.float32)).requires_grad_(True)
    rho_tri = rho_ref.detach().clone().to(torch.bfloat16).requires_grad_(True)

    out_ref, state_ref = performer_plus_pdf_delta_attention(
        q_ref,
        k_ref,
        v_ref,
        rho=rho_ref,
        output_final_state=True,
        eps=1e-6,
    )
    out_tri, state_tri = performer_plus_pdf_delta_attention_triton(
        q_prime=q_tri,
        k_prime=k_tri,
        v=v_tri,
        rho=rho_tri,
        output_final_state=True,
        eps=1e-6,
        denom_eps=1e-6,
        denom_stopgrad=False,
    )

    assert torch.allclose(out_ref.float(), out_tri.float(), atol=5e-3, rtol=5e-2)
    assert state_ref is not None and state_tri is not None
    assert torch.allclose(state_ref[0].float(), state_tri[0].float(), atol=5e-2, rtol=8e-2)
    assert torch.allclose(state_ref[1].float(), state_tri[1].float(), atol=5e-2, rtol=8e-2)

    loss_ref = out_ref.float().pow(2).mean() + state_ref[0].float().pow(2).mean() * 1e-4
    loss_tri = out_tri.float().pow(2).mean() + state_tri[0].float().pow(2).mean() * 1e-4
    loss_ref.backward()
    loss_tri.backward()
    assert q_ref.grad is not None and k_ref.grad is not None and v_ref.grad is not None and rho_ref.grad is not None
    assert q_tri.grad is not None and k_tri.grad is not None and v_tri.grad is not None and rho_tri.grad is not None
    assert torch.isfinite(q_ref.grad).all() and torch.isfinite(k_ref.grad).all() and torch.isfinite(v_ref.grad).all() and torch.isfinite(rho_ref.grad).all()
    assert torch.isfinite(q_tri.grad).all() and torch.isfinite(k_tri.grad).all() and torch.isfinite(v_tri.grad).all() and torch.isfinite(rho_tri.grad).all()
    assert torch.allclose(
        q_ref.grad.float(),
        q_tri.grad.float(),
        atol=2e-4,
        rtol=1e-1,
    )
    assert torch.allclose(
        k_ref.grad.float(),
        k_tri.grad.float(),
        atol=2e-4,
        rtol=1e-1,
    )
    assert torch.allclose(
        v_ref.grad.float(),
        v_tri.grad.float(),
        atol=2e-4,
        rtol=1e-1,
    )
    assert torch.allclose(
        rho_ref.grad.float(),
        rho_tri.grad.float(),
        atol=2e-4,
        rtol=1e-1,
    )


if pytest is not None:
    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason='performer+ pdf-delta import failed.')
    def test_performer_pdf_delta_dense_chunk_equivalence() -> None:
        _run_dense_chunk_equivalence_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason='performer+ pdf-delta import failed.')
    def test_performer_pdf_delta_varlen_equivalence() -> None:
        _run_varlen_equivalence_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason='performer+ pdf-delta import failed.')
    def test_performer_pdf_delta_module() -> None:
        _run_module_case()

    @pytest.mark.skipif(_IMPORT_ERROR is not None, reason='performer+ pdf-delta import failed.')
    def test_performer_pdf_delta_triton_vs_torch() -> None:
        _run_pdf_delta_triton_vs_torch_case()


def main() -> None:
    if _IMPORT_ERROR is not None:
        print(f'skip: import failed: {_IMPORT_ERROR}')
        return
    _run_dense_chunk_equivalence_case()
    _run_varlen_equivalence_case()
    _run_module_case()
    _run_pdf_delta_triton_vs_torch_case()
    print('ok: performer+ pdf-delta tests passed')


if __name__ == '__main__':
    main()
