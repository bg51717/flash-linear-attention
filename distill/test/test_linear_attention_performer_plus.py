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
    from distill.models.linear_attention_performer_plus import (
        PerformerPlusLinearAttention,
    )
    from distill.models.linear_attention_performer_plus_triton import (
        _TRITON_AVAILABLE,
    )
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    PerformerPlusLinearAttention = None
    _TRITON_AVAILABLE = False


def _is_ready() -> bool:
    return bool(_IMPORT_ERROR is None and _TRITON_AVAILABLE and torch.cuda.is_available())


def _run_module_case() -> None:
    torch.manual_seed(1314)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=48,
        head_dim=8,
        num_heads=6,
        num_v_heads=6,
        expand_v=1.0,
        use_short_conv=False,
        performer_nb_features=24,
        performer_use_decay=True,
        performer_projection_seed=0,
        performer_per_layer_projection=True,
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


def _run_projection_diversity_case() -> None:
    torch.manual_seed(2028)
    device = torch.device('cuda')
    layer0 = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=16,
        performer_projection_seed=7,
        performer_per_layer_projection=True,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    layer1 = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=16,
        performer_projection_seed=7,
        performer_per_layer_projection=True,
        performer_use_triton=True,
        layer_idx=1,
    ).to(device)
    assert not torch.allclose(layer0.projection_matrix, layer1.projection_matrix)


if pytest is not None:
    if _IMPORT_ERROR is not None:
        pytestmark = pytest.mark.skip(reason=f'Performer+ import failed: {_IMPORT_ERROR}')

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_module_forward_cuda() -> None:
        _run_module_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_projection_diversity() -> None:
        _run_projection_diversity_case()


def main() -> None:
    if _IMPORT_ERROR is not None:
        print(f'skip: import failed: {_IMPORT_ERROR}')
        return
    if not _is_ready():
        print('skip: CUDA + Triton are required.')
        return
    _run_module_case()
    _run_projection_diversity_case()
    print('ok: performer+ module tests passed')


if __name__ == '__main__':
    main()
