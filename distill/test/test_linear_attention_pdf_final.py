from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

try:
    import pytest
except Exception:  # pragma: no cover
    pytest = None

# Allow running this file directly without installing `distill` as a package.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fla.layers.pdf_final import (
    _torch_pdf_final_linear_attention,
    pdf_final_linear_attention,
)
from distill.models.utils import init_attention_module


def _reference_pdf_final(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None = None,
    hist_eps: float = 1e-4,
    score_clip: float = 20.0,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    n, t, k_dim = q.shape
    v_dim = v.shape[-1]
    qf = q.float()
    kf = k.float()
    vf = v.float()

    if initial_state is None:
        s_sum = torch.zeros((n, v_dim, k_dim), dtype=torch.float32, device=q.device)
        k_sum = torch.zeros((n, k_dim), dtype=torch.float32, device=q.device)
        o_prev = torch.zeros((n, v_dim), dtype=torch.float32, device=q.device)
        q_prev = torch.zeros((n, k_dim), dtype=torch.float32, device=q.device)
        count = torch.zeros((n,), dtype=torch.float32, device=q.device)
    else:
        s_sum, k_sum, o_prev, q_prev, count = [x.clone().float() for x in initial_state]

    out = torch.empty((n, t, v_dim), dtype=torch.float32, device=q.device)
    for i in range(t):
        q_i = qf[:, i]
        k_i = kf[:, i]
        v_i = vf[:, i]
        o_i = torch.empty_like(o_prev)
        for row in range(n):
            if count[row] <= 0:
                beta = 1.0
                o_hist = torch.zeros_like(o_prev[row])
            else:
                s_mean = s_sum[row] / count[row]
                k_mean = k_sum[row] / count[row]
                dq = q_i[row] - q_prev[row]
                o_hist = o_prev[row] + s_mean @ dq - o_prev[row] * torch.dot(k_mean, dq)
                hist_mass = torch.dot(k_sum[row], q_i[row])
                hist_mass = torch.log(torch.nn.functional.softplus(hist_mass) + hist_eps)
                score = torch.dot(k_i[row], q_i[row]).clamp(min=-score_clip, max=score_clip)
                beta = torch.sigmoid(score - hist_mass)
            o_i[row] = (1.0 - beta) * o_hist + beta * v_i[row]
        out[:, i] = o_i
        s_sum = s_sum + torch.einsum("nv,nk->nvk", v_i, k_i)
        k_sum = k_sum + k_i
        o_prev = o_i
        q_prev = q_i
        count = count + 1.0
    return out.to(q.dtype), (s_sum, k_sum, o_prev, q_prev, count)


def _run_dense_reference_case() -> None:
    torch.manual_seed(123)
    q = torch.randn(5, 17, 16, dtype=torch.float32)
    k = torch.randn(5, 17, 16, dtype=torch.float32)
    v = torch.randn(5, 17, 12, dtype=torch.float32)

    out, state, _ = _torch_pdf_final_linear_attention(
        q,
        k,
        v,
        output_final_state=True,
    )
    ref_out, ref_state = _reference_pdf_final(q, k, v)

    torch.testing.assert_close(out, ref_out, rtol=1e-6, atol=1e-6)
    for got, ref in zip(state, ref_state, strict=False):
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)


def _run_chunk_equivalence_case() -> None:
    torch.manual_seed(456)
    q = torch.randn(2, 19, 3, 10, dtype=torch.float32)
    k = torch.randn(2, 19, 3, 10, dtype=torch.float32)
    v = torch.randn(2, 19, 3, 14, dtype=torch.float32)

    out_full, state_full, _ = pdf_final_linear_attention(
        q,
        k,
        v,
        output_final_state=True,
    )
    out_a, state_a, _ = pdf_final_linear_attention(
        q[:, :7],
        k[:, :7],
        v[:, :7],
        output_final_state=True,
    )
    out_b, state_b, _ = pdf_final_linear_attention(
        q[:, 7:],
        k[:, 7:],
        v[:, 7:],
        initial_state=state_a,
        output_final_state=True,
    )
    out_cat = torch.cat([out_a, out_b], dim=1)

    torch.testing.assert_close(out_cat, out_full, rtol=1e-6, atol=1e-6)
    for got, ref in zip(state_b, state_full, strict=False):
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)


def _run_varlen_case() -> None:
    torch.manual_seed(789)
    cu = torch.tensor([0, 5, 11, 18], dtype=torch.int32)
    q = torch.randn(1, 18, 2, 8, dtype=torch.float32)
    k = torch.randn(1, 18, 2, 8, dtype=torch.float32)
    v = torch.randn(1, 18, 2, 6, dtype=torch.float32)

    out, final_state, _ = pdf_final_linear_attention(
        q,
        k,
        v,
        cu_seqlens=cu,
        output_final_state=True,
    )

    qf = q.permute(0, 2, 1, 3).reshape(2, 18, 8)
    kf = k.permute(0, 2, 1, 3).reshape(2, 18, 8)
    vf = v.permute(0, 2, 1, 3).reshape(2, 18, 6)
    ref_parts = []
    ref_states = [[] for _ in range(5)]
    cu_list = cu.tolist()
    for i in range(len(cu_list) - 1):
        bos, eos = cu_list[i], cu_list[i + 1]
        out_seg, st_seg = _reference_pdf_final(qf[:, bos:eos], kf[:, bos:eos], vf[:, bos:eos])
        ref_parts.append(out_seg)
        for idx in range(5):
            ref_states[idx].append(st_seg[idx])
    ref_flat = torch.cat(ref_parts, dim=1)
    ref = ref_flat.reshape(1, 2, 18, 6).permute(0, 2, 1, 3).contiguous()
    ref_state = tuple(torch.stack(chunks, dim=0) for chunks in ref_states)

    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
    for got, ref in zip(final_state, ref_state, strict=False):
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)


def _run_module_case() -> None:
    if not torch.cuda.is_available():
        print("skip: CUDA is required for module integration test.")
        return

    cfg = SimpleNamespace(
        linear_attention_type="pdf_final_linear_attention",
        hidden_size=72,
        num_attention_heads=9,
        num_key_value_heads=3,
        rms_norm_eps=1e-5,
        head_dim=8,
        expand_v=1.0,
        use_short_conv=False,
        conv_size=4,
        conv_bias=False,
        pdf_final_output_norm="identity",
        pdf_final_hist_eps=1e-4,
        pdf_final_score_clip=20.0,
        attention_bias=False,
        rope_theta=10000.0,
        max_position_embeddings=128,
    )

    module = init_attention_module(cfg, layer_idx=0, source_attn=None)
    module = module.to("cuda")
    hidden_states = torch.randn(2, 13, 72, dtype=torch.float32, device="cuda")
    attention_mask = torch.ones(2, 13, dtype=torch.bool, device="cuda")
    out, _ = module(hidden_states, attention_mask=attention_mask)

    assert out.shape == (2, 13, 72)
    assert torch.isfinite(out).all()


if pytest is not None:

    def test_linear_attention_pdf_final_dense_reference() -> None:
        _run_dense_reference_case()


    def test_linear_attention_pdf_final_chunk_equivalence() -> None:
        _run_chunk_equivalence_case()


    def test_linear_attention_pdf_final_varlen() -> None:
        _run_varlen_case()


    def test_linear_attention_pdf_final_module() -> None:
        _run_module_case()


def main() -> None:
    _run_dense_reference_case()
    _run_chunk_equivalence_case()
    _run_varlen_case()
    _run_module_case()
    print("ok: pdf_final_linear_attention tests passed")


if __name__ == "__main__":
    main()
