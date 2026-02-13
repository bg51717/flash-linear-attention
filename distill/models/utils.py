from fla.layers import gated_deltanet
import torch

from .linear_attention_pdf import FirstOrderLinearAttention

TYPE2ATTN = {
    "gated_deltanet": gated_deltanet.GatedDeltaNet,
    "pdf_linear_attention": FirstOrderLinearAttention,
    "first_order_linear_attention": FirstOrderLinearAttention,
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
