from simple_parsing import ArgumentParser
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .args import ModelArguments, DataArguments
from .data import prapare_dataset, DataCollatorWithFlattening


def load_model_tokenizer(model_args):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
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
    model, tokenizer = load_model_tokenizer(args.model)
    train_dataset = prapare_dataset(tokenizer, args.data, split="train")
    for name, param in model.named_parameters():
        if "linear_attn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    # Train
    trainer = Trainer(
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


if __name__ == "__main__":
    main()
