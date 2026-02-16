from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    linear_attention_type: str = field(
        default="gated_deltanet",
        metadata={"help": "Type of linear attention mechanism to use, e.g. gated_deltanet, delta_net, first_order_linear_attention, performer_linear_attention, performer_plus_linear_attention."}
    )
    teacher_model: str = field(
        default=None,
        metadata={"help": "Path to the teacher model for distillation."}
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
