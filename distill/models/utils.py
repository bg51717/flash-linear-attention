from fla.layers import delta_net, gated_deltanet
import torch

from .linear_attention_pdf import FirstOrderLinearAttention
from .linear_attention_performer import PerformerLinearAttention
from .linear_attention_performer_plus import PerformerPlusLinearAttention

TYPE2ATTN = {
    "gated_deltanet": gated_deltanet.GatedDeltaNet,
    "delta_net": delta_net.DeltaNet,
    "pdf_linear_attention": FirstOrderLinearAttention,
    "first_order_linear_attention": FirstOrderLinearAttention,
    "performer_linear_attention": PerformerLinearAttention,
    "performer_plus_linear_attention": PerformerPlusLinearAttention,
}

CONFIG2KWARGS = {
    "gated_deltanet": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_v_heads": lambda cfg: (
            getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            if getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            >= cfg.num_attention_heads
            else cfg.num_attention_heads
        ),
    },
    "delta_net": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "mode": lambda cfg: getattr(cfg, "delta_net_mode", "chunk"),
        "expand_k": lambda cfg: getattr(cfg, "expand_k", 1.0),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_beta": lambda cfg: getattr(cfg, "use_beta", True),
        "use_gate": lambda cfg: getattr(cfg, "use_gate", False),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "allow_neg_eigval": lambda cfg: getattr(cfg, "allow_neg_eigval", False),
        "qk_activation": lambda cfg: getattr(cfg, "qk_activation", "silu"),
        "qk_norm": lambda cfg: getattr(cfg, "qk_norm", "l2"),
    },
    "pdf_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_v_heads": lambda cfg: (
            getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            if getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            >= cfg.num_attention_heads
            else cfg.num_attention_heads
        ),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
    },
    "first_order_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_v_heads": lambda cfg: (
            getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            if getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            >= cfg.num_attention_heads
            else cfg.num_attention_heads
        ),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
    },
    "performer_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_v_heads": lambda cfg: (
            getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            if getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            >= cfg.num_attention_heads
            else cfg.num_attention_heads
        ),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "performer_nb_features": lambda cfg: getattr(cfg, "performer_nb_features", None),
        "performer_feature_eps": lambda cfg: getattr(cfg, "performer_feature_eps", 1e-4),
        "performer_ortho_scaling": lambda cfg: getattr(cfg, "performer_ortho_scaling", 0),
        "performer_redraw_projection": lambda cfg: getattr(cfg, "performer_redraw_projection", False),
        "performer_projection_seed": lambda cfg: getattr(cfg, "performer_projection_seed", 0),
        "performer_use_triton": lambda cfg: getattr(cfg, "performer_use_triton", True),
    },
    "performer_plus_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_v_heads": lambda cfg: (
            getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            if getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            >= cfg.num_attention_heads
            else cfg.num_attention_heads
        ),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "performer_nb_features": lambda cfg: getattr(cfg, "performer_nb_features", None),
        "performer_feature_eps": lambda cfg: getattr(cfg, "performer_feature_eps", 1e-4),
        "performer_ortho_scaling": lambda cfg: getattr(cfg, "performer_ortho_scaling", 0),
        "performer_redraw_projection": lambda cfg: getattr(cfg, "performer_redraw_projection", False),
        "performer_projection_seed": lambda cfg: getattr(cfg, "performer_projection_seed", 0),
        "performer_use_triton": lambda cfg: getattr(cfg, "performer_use_triton", True),
        "performer_antithetic_features": lambda cfg: getattr(cfg, "performer_antithetic_features", True),
        "performer_qk_l2_norm": lambda cfg: getattr(cfg, "performer_qk_l2_norm", True),
        "performer_use_decay": lambda cfg: getattr(cfg, "performer_use_decay", True),
        "performer_decay_init": lambda cfg: getattr(cfg, "performer_decay_init", 2.0),
        "performer_per_layer_projection": lambda cfg: getattr(cfg, "performer_per_layer_projection", True),
    },
}


def config_to_kwargs(config, mapping: dict) -> dict:
    out = {}
    for init_kw, src in mapping.items():
        if callable(src):
            out[init_kw] = src(config)
        else:
            if hasattr(config, src):
                out[init_kw] = getattr(config, src)
    return out


def init_attention_module(config, layer_idx: int):
    attn_type = config.linear_attention_type
    attn_class = TYPE2ATTN[attn_type]
    attn_kwargs = config_to_kwargs(config, CONFIG2KWARGS[attn_type])

    class LinearAttn(attn_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, *args, **kwargs):
            attention_mask = kwargs.get("attention_mask", None)
            if attention_mask is not None and attention_mask.dim() == 4:
                # HF eager/sdpa can pass a 4D causal mask [B, 1, Q, K]. The FLA
                # layers expect a 2D padding mask [B, K] (True for valid tokens).
                attention_mask_2d = attention_mask[:, 0, -1, :]
                if attention_mask_2d.dtype != torch.bool:
                    # For eager masks, valid positions are 0 and masked are -inf.
                    attention_mask_2d = attention_mask_2d.eq(0)
                kwargs = {**kwargs}
                kwargs.update({"attention_mask": attention_mask_2d})

            outputs = super().forward(*args, **kwargs)
            assert len(outputs) == 3, (
                "Expected linear attention output to be a tuple of (attn_output, attn_weights, extra_info)"
            )
            return outputs[0], outputs[1]

    return LinearAttn(**attn_kwargs, layer_idx=layer_idx)
