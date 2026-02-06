from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    linear_attention_type: str = field(
        default="gated_deltanet",
        metadata={"help": "Type of linear attention mechanism to use."}
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    seq_len: int = field(
        default=2048,
        metadata={"help": "The sequence length for training."}
    )
