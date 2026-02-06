from transformers.models.llama.configuration_llama import LlamaConfig


class LlamaLAConfig(LlamaConfig):
    model_type = "llama_la"

    def __init__(
        self,
        linear_attention_type: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.linear_attention_type = linear_attention_type

LlamaLAConfig.register_for_auto_class()
