from simple_parsing import ArgumentParser
import torch
import torch.nn as nn
from typing import Any, Union, Optional
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .args import ModelArguments, DataArguments
from .data import prapare_dataset, DataCollatorWithFlattening


class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        assert return_outputs is False, (
            "return_outputs is not supported in DistillationTrainer"
        )
        assert num_items_in_batch is not None, "num_items_in_batch must be provided"
        student_outputs = model(**inputs)
        logits_student = student_outputs.logits.flatten(0, 1)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            logits_teacher = teacher_outputs.logits.detach().flatten(0, 1)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.flatten(0, 1)
            active_positions = attention_mask.eq(1)
            logits_student = logits_student[active_positions]
            logits_teacher = logits_teacher[active_positions]
        loss_fct = nn.KLDivLoss(reduction="sum")
        loss = (
            loss_fct(
                nn.functional.log_softmax(logits_student / 1.0, dim=-1),
                nn.functional.softmax(logits_teacher / 1.0, dim=-1),
            )
            / num_items_in_batch.detach()
        )
        return loss


def load_model_tokenizer(ckpt_path):
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, trust_remote_code=True
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main():
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(ModelArguments, dest="model")
    parser.add_arguments(TrainingArguments, dest="train")
    parser.add_arguments(DataArguments, dest="data")
    args = parser.parse_args()
    model, tokenizer = load_model_tokenizer(args.model.model_name_or_path)
    for name, param in model.named_parameters():
        if "attn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    train_dataset = prapare_dataset(tokenizer, args.data, split="train")
    # Train
    teacher_model, _ = load_model_tokenizer(args.model.teacher_model)
    trainer = DistillationTrainer(
        model=model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        args=args.train,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithFlattening(
            max_len=args.data.seq_len,
            pad_token_id=tokenizer.pad_token_id,
            return_position_ids=False,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
