from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
from einops import rearrange

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
    from fla.layers.taylor import TaylorLinearAttention
    from distill.models.utils import init_attention_module
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    TaylorLinearAttention = None
    init_attention_module = None


def _is_ready() -> bool:
    return bool(_IMPORT_ERROR is None and torch.cuda.is_available())


def _reference_taylor_output(
    module: TaylorLinearAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    q_all = rearrange(module.q_proj(hidden_states), 'b t (h d) -> b t h d', h=module.num_heads)
    k_all = rearrange(module.k_proj(hidden_states), 'b t (h d) -> b t h d', h=module.num_kv_heads)
    v_all = rearrange(module.v_proj(hidden_states), 'b t (h d) -> b t h d', h=module.num_kv_heads)

    refs = torch.zeros_like(hidden_states)
    for b in range(hidden_states.shape[0]):
        valid = hidden_states.shape[1] if attention_mask is None else int(attention_mask[b].sum().item())
        q = q_all[b:b + 1, :valid]
        k = k_all[b:b + 1, :valid]
        v = v_all[b:b + 1, :valid]
        q, k = module.rotary(q, k, seqlen_offset=0, max_seqlen=valid)
        q = q * module.qk_scale
        k = k * module.qk_scale
        if module.num_kv_groups > 1:
            k = k.repeat_interleave(module.num_kv_groups, dim=2)
            v = v.repeat_interleave(module.num_kv_groups, dim=2)

        ones = torch.ones_like(q[..., :1])
        q_aug = torch.cat([ones, q], dim=-1)
        k_aug = torch.cat([ones, k], dim=-1)

        num_state = torch.zeros(
            1, module.num_heads, module.head_k_dim + 1, module.head_v_dim,
            device=hidden_states.device, dtype=torch.float32,
        )
        den_state = torch.zeros(
            1, module.num_heads, module.head_k_dim + 1, 1,
            device=hidden_states.device, dtype=torch.float32,
        )
        outs = []
        for t in range(valid):
            num_state = num_state + torch.einsum('bhk,bhv->bhkv', k_aug[:, t].float(), v[:, t].float())
            den_state = den_state + k_aug[:, t].float().unsqueeze(-1)
            num_t = torch.einsum('bhkv,bhk->bhv', num_state, q_aug[:, t].float())
            den_t = torch.einsum('bhkv,bhk->bhv', den_state, q_aug[:, t].float())
            den_t = den_t.clamp_min(module.denom_eps)
            outs.append(num_t / den_t)

        ref = torch.stack(outs, dim=1)
        ref = rearrange(ref.to(hidden_states.dtype), 'b t h d -> b t (h d)')
        ref = module.o_proj(ref)
        refs[b, :valid] = ref[0]
    return refs


def _run_formula_case() -> None:
    torch.manual_seed(11)
    device = torch.device('cuda')

    module = TaylorLinearAttention(
        hidden_size=32,
        head_dim=8,
        num_heads=4,
        num_kv_heads=2,
        mode='fused_recurrent',
        output_norm='identity',
        denom_eps=1e-4,
    ).to(device=device, dtype=torch.float32)
    module.eval()

    hidden_states = torch.randn(2, 5, 32, device=device, dtype=torch.float32)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
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
    ref = _reference_taylor_output(module, hidden_states, attention_mask)

    assert attn_weights is None
    assert cache is None
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def _run_init_case() -> None:
    device = torch.device('cuda')
    cfg = SimpleNamespace(
        linear_attention_type='taylor_linear_attention',
        hidden_size=72,
        num_attention_heads=9,
        num_key_value_heads=3,
        rms_norm_eps=1e-6,
        head_dim=8,
        expand_v=1.0,
        linear_attn_mode='fused_recurrent',
        taylor_denom_eps=1e-4,
        taylor_output_norm='identity',
        attention_bias=False,
        rope_theta=10000.0,
        max_position_embeddings=2048,
    )
    class SourceAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(72, 72, bias=False)
            self.k_proj = nn.Linear(72, 24, bias=False)
            self.v_proj = nn.Linear(72, 24, bias=False)
            self.o_proj = nn.Linear(72, 72, bias=False)

    source_attn = SourceAttention().to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        source_attn.q_proj.weight.fill_(0.125)
        source_attn.k_proj.weight.fill_(0.25)
        source_attn.v_proj.weight.fill_(0.375)
        source_attn.o_proj.weight.fill_(0.5)

    module = init_attention_module(cfg, layer_idx=0, source_attn=source_attn).to(device=device, dtype=torch.bfloat16)
    hidden_states = torch.randn(2, 16, 72, device=device, dtype=torch.bfloat16)
    attention_mask = torch.ones(2, 16, device=device, dtype=torch.int64)
    out, attn_weights = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        use_cache=False,
    )

    assert out.shape == hidden_states.shape
    assert attn_weights is None
    assert torch.isfinite(out).all()
    torch.testing.assert_close(module.q_proj.weight, source_attn.q_proj.weight)
    torch.testing.assert_close(module.k_proj.weight, source_attn.k_proj.weight)
    torch.testing.assert_close(module.v_proj.weight, source_attn.v_proj.weight)
    torch.testing.assert_close(module.o_proj.weight, source_attn.o_proj.weight)


if pytest is not None:
    if _IMPORT_ERROR is not None:
        pytestmark = pytest.mark.skip(reason=f'Taylor attention import failed: {_IMPORT_ERROR}')

    @pytest.mark.skipif(not _is_ready(), reason='CUDA is required for this test.')
    def test_taylor_linear_attention_matches_formula() -> None:
        _run_formula_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA is required for this test.')
    def test_taylor_linear_attention_init_forward_cuda() -> None:
        _run_init_case()


def main() -> None:
    if _IMPORT_ERROR is not None:
        print(f'skip: import failed: {_IMPORT_ERROR}')
        return
    if not _is_ready():
        print('skip: CUDA is required.')
        return
    _run_formula_case()
    _run_init_case()
    print('ok: taylor linear attention tests passed')


if __name__ == '__main__':
    main()
