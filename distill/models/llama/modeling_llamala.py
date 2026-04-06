from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaForCausalLM,
)
from fla.models.utils import FLAGenerationMixin

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
    from .linear_attention_performer import PerformerLinearAttention  # noqa: F401
    from .linear_attention_performer_triton import performer_causal_linear_attention_triton  # noqa: F401
    from .linear_attention_performer_plus import PerformerPlusLinearAttention  # noqa: F401
    from .linear_attention_performer_plus_triton import performer_plus_causal_linear_attention_triton  # noqa: F401
except Exception:
    pass


class LlamaLADecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        src_attn = self.self_attn
        self.self_attn = init_attention_module(
            config, layer_idx=layer_idx, source_attn=src_attn
        )


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
