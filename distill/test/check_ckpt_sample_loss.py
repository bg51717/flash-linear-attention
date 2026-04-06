#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Large language models can be distilled into efficient architectures.",
    "Linear attention approximates softmax attention with kernel feature maps.",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a checkpoint and report sample CE loss.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpts/135m-performer_plus-stage-1",
        help="Checkpoint directory path.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max token length for each sample.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model dtype on CUDA.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Custom sample text. Can be specified multiple times.",
    )
    return parser


def parse_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model_dtype = torch.float32 if device.type == "cpu" else parse_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt),
        trust_remote_code=True,
        torch_dtype=model_dtype,
    ).to(device)
    model.eval()

    samples = args.text if args.text else DEFAULT_SAMPLES
    print(f"ckpt: {ckpt}")
    print(f"device: {device}")
    print(f"model_class: {model.__class__.__module__}.{model.__class__.__name__}")
    print(f"linear_attention_type: {getattr(model.config, 'linear_attention_type', None)}")
    print("-" * 80)

    losses = []
    for idx, text in enumerate(samples, start=1):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(device)
        out = model(**enc, labels=enc["input_ids"])
        loss = float(out.loss.detach().float().item())
        ppl = math.exp(loss)
        token_count = int(enc["input_ids"].shape[-1])
        losses.append(loss)
        print(f"[{idx}] tokens={token_count:4d} loss={loss:.6f} ppl={ppl:.3f}")
        print(f"    text: {text[:120]}")

    mean_loss = sum(losses) / max(1, len(losses))
    print("-" * 80)
    print(f"mean_loss={mean_loss:.6f} mean_ppl={math.exp(mean_loss):.3f}")


if __name__ == "__main__":
    main()

