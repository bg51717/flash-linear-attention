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
    from fla.layers.performer_plus import (
        PerformerPlusLinearAttention,
    )
    from fla.ops.performer_plus.fused_recurrent import (
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


def _run_module_delta_dual_map_case() -> None:
    torch.manual_seed(2128)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=48,
        head_dim=8,
        num_heads=6,
        num_v_heads=6,
        expand_v=1.0,
        use_short_conv=False,
        performer_nb_features=20,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_control_variate=True,
        performer_use_dual_map=True,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 23, 48, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 23, device=device, dtype=torch.int64)

    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()


def _run_learnable_projection_grad_case() -> None:
    torch.manual_seed(3030)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=16,
        performer_use_beta=True,
        performer_use_value_gate=True,
        performer_use_decay=True,
        performer_learnable_projection=True,
        performer_learnable_projection_scale=True,
        performer_learnable_kernel_scale=True,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 19, 64, device=device, dtype=torch.float32)
    attention_mask = torch.ones(2, 19, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert module.projection_matrix.grad is not None
    assert torch.isfinite(module.projection_matrix.grad).all()
    assert module.projection_log_scale.grad is not None
    assert torch.isfinite(module.projection_log_scale.grad).all()
    assert module.b_proj is not None and module.b_proj.weight.grad is not None
    assert torch.isfinite(module.b_proj.weight.grad).all()
    assert module.beta_bias is not None and module.beta_bias.grad is not None
    assert torch.isfinite(module.beta_bias.grad).all()
    assert module.v_gate_proj is not None and module.v_gate_proj.weight.grad is not None
    assert torch.isfinite(module.v_gate_proj.weight.grad).all()
    assert module.v_gate_bias is not None and module.v_gate_bias.grad is not None
    assert torch.isfinite(module.v_gate_bias.grad).all()
    assert module.o_gate_proj is not None and module.o_gate_proj.weight.grad is not None
    assert torch.isfinite(module.o_gate_proj.weight.grad).all()
    assert module.o_gate_bias is not None and module.o_gate_bias.grad is not None
    assert torch.isfinite(module.o_gate_bias.grad).all()
    assert module.kernel_scale_log.grad is not None
    assert torch.isfinite(module.kernel_scale_log.grad).all()


def _run_projection_ensemble_case() -> None:
    torch.manual_seed(4242)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_projection_ensemble=True,
        performer_projection_ensemble_groups=2,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 17, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 17, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()


def _run_jackknife_debias_case() -> None:
    torch.manual_seed(5252)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_jackknife_debias=True,
        performer_jackknife_groups=2,
        performer_jackknife_min_per_group=4,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()


def _run_jackknife_adaptive_shrinkage_case() -> None:
    torch.manual_seed(5353)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_jackknife_debias=True,
        performer_jackknife_groups=2,
        performer_jackknife_min_per_group=4,
        performer_use_jackknife_adaptive_shrinkage=True,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()


def _run_qmc_sampling_case() -> None:
    torch.manual_seed(5454)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_qmc_gaussian_sampling=True,
        performer_qmc_scramble=True,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()


def _run_landmark_sampling_case() -> None:
    torch.manual_seed(6161)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_control_variate=True,
        performer_use_dual_map=True,
        performer_use_landmark_sampling=True,
        performer_landmark_ratio=0.5,
        performer_landmark_alpha_init=0.35,
        performer_landmark_in_denominator=True,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()
    assert module.landmark_projection is not None
    assert module.landmark_projection.grad is not None
    assert torch.isfinite(module.landmark_projection.grad).all()


def _run_delta_abs_denominator_case() -> None:
    torch.manual_seed(7171)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_control_variate=True,
        performer_delta_denominator_map='abs',
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()


def _run_feature_rms_norm_case() -> None:
    torch.manual_seed(7272)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_control_variate=True,
        performer_feature_rms_norm=True,
        performer_feature_rms_norm_eps=1e-4,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()


def _run_delta_positive_linear_denominator_case() -> None:
    torch.manual_seed(7282)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_control_variate=True,
        performer_delta_denominator_map='positive_linear',
        performer_den_poly_constant=2.0,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()


def _run_feature_pairwise_balance_case() -> None:
    torch.manual_seed(7292)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_control_variate=True,
        performer_feature_pairwise_balance=True,
        performer_feature_pairwise_balance_eps=1e-4,
        performer_feature_pairwise_balance_log_clip=2.0,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()


def _run_dual_precondition_sampling_case() -> None:
    torch.manual_seed(7302)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_control_variate=True,
        performer_use_dual_precondition_sampling=True,
        performer_precondition_momentum=0.9,
        performer_precondition_eps=1e-4,
        performer_precondition_log_clip=2.0,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()
    assert bool(module.precondition_initialized.item())
    assert torch.isfinite(module.precondition_log_scale_ema).all()


def _run_full_precondition_case() -> None:
    torch.manual_seed(7312)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_control_variate=True,
        performer_use_dual_precondition_sampling=True,
        performer_precondition_mode='full',
        performer_precondition_momentum=0.9,
        performer_precondition_eps=1e-4,
        performer_precondition_log_clip=2.0,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()
    assert bool(module.precondition_initialized.item())
    assert torch.isfinite(module.precondition_cov_q_ema).all()
    assert torch.isfinite(module.precondition_cov_k_ema).all()


def _run_third_order_cv_case() -> None:
    torch.manual_seed(7373)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_control_variate=True,
        performer_use_second_order_cv=True,
        performer_use_third_order_cv=True,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()


def _run_decoupled_second_order_cv_case() -> None:
    torch.manual_seed(7383)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_control_variate=True,
        performer_use_second_order_cv=True,
        performer_use_cv_decoupled_second_order=True,
        performer_cv_decoupled_h2_ratio=0.25,
        performer_cv_split_feature_budget=True,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()


def _run_decoupled_second_order_cv_deterministic_h2_case() -> None:
    torch.manual_seed(7393)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_control_variate=True,
        performer_use_second_order_cv=True,
        performer_use_cv_decoupled_second_order=True,
        performer_cv_decoupled_h2_ratio=0.25,
        performer_cv_decoupled_h2_deterministic=True,
        performer_cv_split_feature_budget=True,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()


def _run_decoupled_second_order_cv_adaptive_ratio_case() -> None:
    torch.manual_seed(7403)
    device = torch.device('cuda')
    module = PerformerPlusLinearAttention(
        hidden_size=64,
        head_dim=8,
        num_heads=8,
        num_v_heads=8,
        use_short_conv=False,
        performer_nb_features=24,
        performer_state_update='delta',
        performer_use_beta=True,
        performer_use_decay=False,
        performer_use_control_variate=True,
        performer_use_second_order_cv=True,
        performer_use_cv_decoupled_second_order=True,
        performer_cv_decoupled_h2_ratio=0.25,
        performer_use_cv_decoupled_adaptive_h2_ratio=True,
        performer_cv_decoupled_ratio_ema_momentum=0.9,
        performer_cv_decoupled_ratio_min=0.05,
        performer_cv_decoupled_ratio_max=0.5,
        performer_cv_split_feature_budget=True,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    hidden_states = torch.randn(2, 15, 64, device=device, dtype=torch.float32, requires_grad=True)
    attention_mask = torch.ones(2, 15, device=device, dtype=torch.int64)
    out, _, _ = module(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
    )
    loss = out.float().pow(2).mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(hidden_states.grad).all()
    ratio = float(module.cv_decoupled_h2_ratio_ema.item())
    assert 0.05 <= ratio <= 0.5


if pytest is not None:
    if _IMPORT_ERROR is not None:
        pytestmark = pytest.mark.skip(reason=f'Performer+ import failed: {_IMPORT_ERROR}')

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_module_forward_cuda() -> None:
        _run_module_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_projection_diversity() -> None:
        _run_projection_diversity_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_module_delta_dual_map() -> None:
        _run_module_delta_dual_map_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_learnable_projection_grad() -> None:
        _run_learnable_projection_grad_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_projection_ensemble() -> None:
        _run_projection_ensemble_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_jackknife_debias() -> None:
        _run_jackknife_debias_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_jackknife_adaptive_shrinkage() -> None:
        _run_jackknife_adaptive_shrinkage_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_qmc_sampling() -> None:
        _run_qmc_sampling_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_landmark_sampling() -> None:
        _run_landmark_sampling_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_delta_abs_denominator() -> None:
        _run_delta_abs_denominator_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_feature_rms_norm() -> None:
        _run_feature_rms_norm_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_delta_positive_linear_denominator() -> None:
        _run_delta_positive_linear_denominator_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_feature_pairwise_balance() -> None:
        _run_feature_pairwise_balance_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_dual_precondition_sampling() -> None:
        _run_dual_precondition_sampling_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_full_precondition() -> None:
        _run_full_precondition_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_third_order_cv() -> None:
        _run_third_order_cv_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_decoupled_second_order_cv() -> None:
        _run_decoupled_second_order_cv_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_decoupled_second_order_cv_deterministic_h2() -> None:
        _run_decoupled_second_order_cv_deterministic_h2_case()

    @pytest.mark.skipif(not _is_ready(), reason='CUDA + Triton are required for this test.')
    def test_performer_plus_decoupled_second_order_cv_adaptive_ratio() -> None:
        _run_decoupled_second_order_cv_adaptive_ratio_case()


def main() -> None:
    if _IMPORT_ERROR is not None:
        print(f'skip: import failed: {_IMPORT_ERROR}')
        return
    if not _is_ready():
        print('skip: CUDA + Triton are required.')
        return
    _run_module_case()
    _run_projection_diversity_case()
    _run_module_delta_dual_map_case()
    _run_learnable_projection_grad_case()
    _run_projection_ensemble_case()
    _run_jackknife_debias_case()
    _run_jackknife_adaptive_shrinkage_case()
    _run_qmc_sampling_case()
    _run_landmark_sampling_case()
    _run_delta_abs_denominator_case()
    _run_feature_rms_norm_case()
    _run_delta_positive_linear_denominator_case()
    _run_feature_pairwise_balance_case()
    _run_dual_precondition_sampling_case()
    _run_full_precondition_case()
    _run_third_order_cv_case()
    _run_decoupled_second_order_cv_case()
    _run_decoupled_second_order_cv_deterministic_h2_case()
    _run_decoupled_second_order_cv_adaptive_ratio_case()
    print('ok: performer+ module tests passed')


if __name__ == '__main__':
    main()
