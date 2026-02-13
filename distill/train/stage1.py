from simple_parsing import ArgumentParser
import torch
import torch.nn as nn
from pathlib import Path
import shutil
import os
from typing import Any, Union, Optional
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

from .args import ModelArguments, DataArguments
from .data import prapare_dataset, DataCollatorWithFlattening

from ..models.utils import init_attention_module

settings = {
    "llama": {
        "auto_map": {
            "AutoConfig": "configuration_llamala.LlamaLAConfig",
            "AutoModel": "modeling_llamala.LlamaLAModel",
            "AutoModelForCausalLM": "modeling_llamala.LlamaLAForCausalLM",
        },
        "architectures": ["LlamaLAForCausalLM"],
        "model_type": "llama_la",
    }
}

remote_code_dirs = {
    "llama": "models/llama",
}

utils_files = {
    "models/utils.py",
    "models/linear_attention_pdf.py",
    "models/linear_attention_pdf_triton.py",
    "models/linear_attention_pdf_triton_kernels.py",
}

class MSETrainer(Trainer):
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        assert return_outputs is False, "return_outputs=True not supported"
        assert num_items_in_batch is not None, "num_items_in_batch must be provided"
        base_model = model.module if hasattr(model, "module") else model
        num_layers = len(base_model.model.layers)
        loss = torch.zeros(1, device=base_model.device)
        attention_mask = (
            inputs["attention_mask"] if "attention_mask" in inputs else None
        )  # expected shape [B, T]

        def hook(module, args, kwargs, output):
            nonlocal loss
            attn_output = output[0]
            # Attention Mask
            kwargs = {k: v for k, v in kwargs.items() if k != "attention_mask"}
            linear_attn_output = module.linear_attn(
                *args, attention_mask=attention_mask, **kwargs
            )
            diff = linear_attn_output[0] - attn_output.detach()
            if attention_mask is not None:
                diff = diff * attention_mask.unsqueeze(-1)
            loss += diff.pow(2).sum() / diff.shape[-1] / num_items_in_batch / num_layers

        handles = []
        for layer in base_model.model.layers:
            handle = layer.self_attn.register_forward_hook(hook, with_kwargs=True)
            handles.append(handle)
        base_model(**inputs)
        for handle in handles:
            handle.remove()
        return loss


def load_model_tokenizer(model_args):
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = model.config
    config.linear_attention_type = model_args.linear_attention_type
    for layer_idx in range(len(model.model.layers)):
        model.model.layers[layer_idx].self_attn.linear_attn = init_attention_module(
            config, layer_idx
        )
    return model, tokenizer


def main():
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(ModelArguments, dest="model")
    parser.add_arguments(TrainingArguments, dest="train")
    parser.add_arguments(DataArguments, dest="data")
    args = parser.parse_args()
    model, tokenizer = load_model_tokenizer(args.model)
    train_dataset = prapare_dataset(tokenizer, args.data, split="train")
    for name, param in model.named_parameters():
        if "linear_attn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    # MSE Train
    trainer = MSETrainer(
        model=model,
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
    # Save and Convert
    if trainer.is_world_process_zero():
        for layer in model.model.layers:
            layer.self_attn = layer.self_attn.linear_attn
        save_dir = Path(args.train.output_dir).absolute()
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        config = AutoConfig.from_pretrained(save_dir)
        model_type = config.model_type
        for key, value in settings[model_type].items():
            setattr(config, key, value)
        config.linear_attention_type = args.model.linear_attention_type
        config.save_pretrained(save_dir)
        current_dir = Path(__file__).resolve().parent.parent
        for item in os.listdir(current_dir / remote_code_dirs[model_type]):
            if not item.endswith(".py"):
                continue
            source_path = current_dir / remote_code_dirs[model_type] / item
            target_path = save_dir / item
            shutil.copy(source_path, target_path)
        for utils_file in utils_files:
            file_name = utils_file.split("/")[-1]
            shutil.copy(current_dir / utils_file, save_dir / file_name)
        print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
