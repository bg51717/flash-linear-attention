from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaForCausalLM,
)

from .configuration_llamala import LlamaLAConfig
from .utils import init_attention_module


class LlamaLADecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = init_attention_module(config, layer_idx=layer_idx)


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


class LlamaLAForCausalLM(LlamaLAPreTrainedModel, LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaLAModel(config)


__all__ = [
    "LlamaLADecoderLayer",
    "LlamaLAPreTrainedModel",
    "LlamaLAModel",
    "LlamaLAForCausalLM",
]
