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
    from fla.layers.performer import (
        PerformerLinearAttention,
    )
    from fla.ops.performer.fused_recurrent import (
        _TRITON_AVAILABLE,
        performer_causal_linear_attention_triton,
    )
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    PerformerLinearAttention = None
    performer_causal_linear_attention_triton = None
    _TRITON_AVAILABLE = False


def _is_ready() -> bool:
    return bool(_IMPORT_ERROR is None and _TRITON_AVAILABLE and torch.cuda.is_available())


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


def _run_core_case() -> None:
    torch.manual_seed(7)
    device = torch.device('cuda')
    q_prime = torch.rand(2, 19, 3, 11, device=device, dtype=torch.float32) + 0.05
    k_prime = torch.rand(2, 19, 3, 11, device=device, dtype=torch.float32) + 0.05
    v = torch.randn(2, 19, 3, 5, device=device, dtype=torch.float32)

    out, state = performer_causal_linear_attention_triton(
        q_prime=q_prime,
        k_prime=k_prime,
        v=v,
        output_final_state=True,
    )
    ref_out, ref_state = _reference_causal_attention(q_prime, k_prime, v)

    torch.testing.assert_close(out, ref_out, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(state[0], ref_state[0], rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(state[1], ref_state[1], rtol=5e-3, atol=5e-3)

def _run_module_case() -> None:
    torch.manual_seed(13)
    device = torch.device('cuda')
    module = PerformerLinearAttention(
        hidden_size=48,
        head_dim=8,
        num_heads=6,
        num_v_heads=6,
        expand_v=1.0,
        use_short_conv=False,
        performer_nb_features=16,
        performer_redraw_projection=False,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.eval()

    hidden_states = torch.randn(2, 17, 48, device=device)
    attention_mask = torch.tensor(
        [
            [1] * 17,
            [1] * 13 + [0] * 4,
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

    assert out.shape == hidden_states.shape
    assert attn_weights is None
    assert cache is None
    assert torch.isfinite(out).all()


if pytest is not None:
    if _IMPORT_ERROR is not None:
        pytestmark = pytest.mark.skip(reason=f'Performer import failed: {_IMPORT_ERROR}')

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_triton_core_matches_reference() -> None:
        _run_core_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_module_forward_cuda() -> None:
        _run_module_case()


def main() -> None:
    if _IMPORT_ERROR is not None:
        print(f'skip: import failed: {_IMPORT_ERROR}')
        return
    if not _is_ready():
        print('skip: CUDA + Triton are required.')
        return
    _run_core_case()
    _run_module_case()
    print('ok: performer core + module triton tests passed')


if __name__ == '__main__':
    main()
