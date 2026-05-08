from __future__ import annotations

import os
import sys
import time

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fla.layers.performer_plus import (  # noqa: E402
    PerformerPlusLinearAttention,
    _PERFORMER_PLUS_TRITON_AVAILABLE,
)


def _bench_case(*, iters: int = 6, warmup: int = 2) -> tuple[float, float, float]:
    device = torch.device("cuda")
    bsz, seqlen, num_heads, head_dim = 2, 1024, 9, 64
    hidden_size = num_heads * head_dim

    module = PerformerPlusLinearAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_v_heads=num_heads,
        use_short_conv=False,
        performer_nb_features=head_dim,
        performer_state_update="pdf_delta",
        performer_qk_l2_norm=True,
        performer_use_output_gate=False,
        performer_learnable_kernel_scale=False,
        performer_dim_aware_kernel_scale=False,
        performer_use_triton=True,
        layer_idx=0,
    ).to(device)
    module.train()

    x = torch.randn(bsz, seqlen, hidden_size, device=device, dtype=torch.bfloat16).float().requires_grad_(True)
    attention_mask = torch.ones(bsz, seqlen, device=device, dtype=torch.long)

    for _ in range(warmup):
        out, _, _ = module(hidden_states=x, attention_mask=attention_mask, past_key_values=None, use_cache=False)
        loss = out.float().pow(2).mean()
        loss.backward()
        module.zero_grad(set_to_none=True)
        x.grad = None
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    fw_ms = 0.0
    bw_ms = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        out, _, _ = module(hidden_states=x, attention_mask=attention_mask, past_key_values=None, use_cache=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss = out.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        module.zero_grad(set_to_none=True)
        x.grad = None
        fw_ms += (t1 - t0) * 1000.0
        bw_ms += (t2 - t1) * 1000.0
    peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    return fw_ms / iters, bw_ms / iters, peak_mb


def main() -> None:
    if not torch.cuda.is_available():
        print("skip: CUDA is required")
        return
    if not _PERFORMER_PLUS_TRITON_AVAILABLE:
        print("skip: Triton operator unavailable")
        return

    torch.manual_seed(20260407)
    fw_tri, bw_tri, mem_tri = _bench_case()
    print(
        f"triton_pdf_delta fw_ms={fw_tri:.2f} bw_ms={bw_tri:.2f} peak_mb={mem_tri:.1f}",
    )


if __name__ == "__main__":
    main()
