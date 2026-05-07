import torch
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaForCausalLM,
)
from fla.models.utils import FLAGenerationMixin, Cache as FLACache

from .configuration_llamala import LlamaLAConfig
from .utils import init_attention_module

# Explicit relative imports so `transformers` dynamic module loader copies these
# files when loading from a local checkpoint path with `trust_remote_code=True`.
try:  # pragma: no cover
    from .linear_attention_pdf import FirstOrderLinearAttention  # noqa: F401
    from .linear_attention_pdf_final import PDFFinalLinearAttention  # noqa: F401
    from .linear_attention_pdf_final_triton import pdf_final_linear_attention_triton  # noqa: F401
    from .linear_attention_pdf_triton import first_order_linear_attention  # noqa: F401
    from .linear_attention_pdf_triton_kernels import triton_first_order_linear_attention  # noqa: F401
    from .linear_attention_taylor import TaylorLinearAttention  # noqa: F401
    from .linear_attention_approxnet_v2 import ApproxNetV2LinearAttention  # noqa: F401
    from .linear_attention_approxnet_v2_triton import approxnet_v2_linear_attention_triton  # noqa: F401
    from .linear_attention_performer import PerformerLinearAttention  # noqa: F401
    from .linear_attention_performer_triton import performer_causal_linear_attention_triton  # noqa: F401
    from .linear_attention_performer_plus import PerformerPlusLinearAttention  # noqa: F401
    from .linear_attention_performer_plus_triton import performer_plus_causal_linear_attention_triton  # noqa: F401
    from .dual_delta_net import DualDeltaNet  # noqa: F401
    from .dual_delta_rule import fused_recurrent_dual_delta_rule  # noqa: F401
    from .dual_delta_rule_naive import dual_delta_rule_naive  # noqa: F401
    from .mean_delta_net import MeanDeltaNet  # noqa: F401
    from .mean_delta_rule import fused_recurrent_mean_delta_rule  # noqa: F401
    from .mean_delta_rule_naive import mean_delta_rule_recurrence  # noqa: F401
    from .linear_attention_approxnet_v3 import ApproxNetV3LinearAttention  # noqa: F401
    from .linear_attention_approxnet_v3_triton import approxnet_v3_linear_attention_triton  # noqa: F401
    from .linear_attention_approxnet_v4 import ApproxNetV4LinearAttention  # noqa: F401
    from .linear_attention_approxnet_v4_triton import approxnet_v4_linear_attention_triton  # noqa: F401
    from .linear_attention_soam import SOAMLinearAttention  # noqa: F401
    from .linear_attention_soam_triton import fused_recurrent_soam  # noqa: F401
    from .linear_attention_wla import WLALinearAttention  # noqa: F401
    from .linear_attention_wla_triton import fused_recurrent_wla  # noqa: F401
    from .linear_attention_sisa import SiSALinearAttention  # noqa: F401
    from .linear_attention_sisa_triton import fused_recurrent_sisa  # noqa: F401
except Exception:
    pass


class LlamaLADecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        src_attn = self.self_attn
        self.self_attn = init_attention_module(
            config, layer_idx=layer_idx, source_attn=src_attn
        )

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        outputs = super().forward(hidden_states, attention_mask=attention_mask, **kwargs)
        # Linear attention uses unpadding (pad_input fills zeros at pad positions),
        # but the residual connections accumulate non-zero values at those positions
        # across layers, eventually overflowing bfloat16 and producing NaN.
        # Zero out pad positions after each layer to prevent this accumulation.
        if attention_mask is not None and attention_mask.dim() == 4 and outputs[0].shape[1] > 1:
            padding_mask = attention_mask[:, 0, -1, :]
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask.eq(0)
            if not padding_mask.all():
                outputs = (outputs[0] * padding_mask.unsqueeze(-1).to(outputs[0].dtype),) + outputs[1:]
        return outputs


class LlamaLAPreTrainedModel(LlamaPreTrainedModel):
    config_class = LlamaLAConfig
    _no_split_modules = ["LlamaLADecoderLayer"]


class LlamaLAModel(LlamaLAPreTrainedModel, LlamaModel):
    def __init__(self, config):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [
                LlamaLADecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **flash_attn_kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Align with FLA official models: use FLA cache class for recurrent
        # states instead of HF DynamicCache key/value tensors.
        if use_cache and not isinstance(past_key_values, FLACache):
            past_key_values = FLACache.from_legacy_cache(past_key_values)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )


class LlamaLAForCausalLM(FLAGenerationMixin, LlamaLAPreTrainedModel, LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaLAModel(config)

LlamaForCausalLM.register_for_auto_class("AutoModelForCausalLM")

__all__ = [
    "LlamaLADecoderLayer",
    "LlamaLAPreTrainedModel",
    "LlamaLAModel",
    "LlamaLAForCausalLM",
]
