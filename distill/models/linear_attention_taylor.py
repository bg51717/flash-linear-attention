from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.linear_attn import chunk_linear_attn, fused_recurrent_linear_attn
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def _safe_denominator(x: torch.Tensor, eps: float) -> torch.Tensor:
    # The true softmax partition is always positive. Once the linearized
    # denominator crosses zero, the approximation has already left its valid
    # regime, so keep only the positive branch for stability.
    return x.clamp_min(eps)


class TaylorLinearAttention(nn.Module):
    """
    Implements the first-order softmax approximation

        exp(q^T k) ~= 1 + q^T k

    by augmenting query/key features with a constant channel:

        q' = [1; q], k' = [1; k].
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        mode: str = "fused_recurrent",
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        denom_eps: float = 1e-4,
        output_norm: str = "identity",
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,
        max_position_embeddings: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.mode = mode
        self.layer_idx = layer_idx
        self.denom_eps = denom_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.qkv_bias = qkv_bias

        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = self.num_heads * self.head_k_dim
        self.kv_key_dim = self.num_kv_heads * self.head_k_dim
        self.kv_value_dim = self.num_kv_heads * self.head_v_dim
        self.value_dim = self.num_heads * self.head_v_dim
        self.qk_scale = self.head_k_dim ** -0.25

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads={self.num_heads} must be divisible by num_kv_heads={self.num_kv_heads}.",
            )
        if not math.isclose(self.head_v_dim, head_dim * expand_v, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce integer head_v_dim for head_dim={head_dim}.",
            )
        if mode not in {"fused_recurrent", "chunk"}:
            raise ValueError(f"Unsupported mode `{mode}`.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=self.qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_key_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_value_dim, bias=self.qkv_bias)

        if output_norm == "rmsnorm":
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)
        elif output_norm == "identity":
            self.o_norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported output_norm `{output_norm}`.")
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(dim=self.head_k_dim, base=self.rope_theta)
        self.last_error_stats: dict[str, float] = {}

    def _run_linear_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cu_seqlens is not None:
            return fused_recurrent_linear_attn(
                q=q,
                k=k,
                v=v,
                scale=1.0,
                initial_state=initial_state,
                output_final_state=output_final_state,
                normalize=False,
                cu_seqlens=cu_seqlens,
            )
        if self.mode == "chunk":
            return chunk_linear_attn(
                q=q,
                k=k,
                v=v,
                scale=1.0,
                initial_state=initial_state,
                output_final_state=output_final_state,
                normalize=False,
            )
        return fused_recurrent_linear_attn(
            q=q,
            k=k,
            v=v,
            scale=1.0,
            initial_state=initial_state,
            output_final_state=output_final_state,
            normalize=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        del output_attentions

        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch_size, seq_len].")

        batch_size, q_len, _ = hidden_states.shape
        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        num_state = None
        den_state = None
        if last_state is not None and last_state["recurrent_state"] is not None:
            num_state, den_state = last_state["recurrent_state"]

        cu_seqlens = kwargs.get("cu_seqlens")
        indices = None
        max_seqlen = q_len
        seqlen_offset = 0
        if attention_mask is not None and past_key_values is None:
            indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b t d -> (b t) d"),
                indices,
            ).unsqueeze(0)
            max_seqlen = max(max_seqlen, hidden_states.shape[1])
        elif past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q_len + seqlen_offset
            if attention_mask is not None:
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q_len + int(max(seqlen_offset).item())
        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, "b t (h d) -> b t h d", d=self.head_k_dim)
        k = rearrange(k, "b t (h d) -> b t h d", d=self.head_k_dim)
        v = rearrange(v, "b t (h d) -> b t h d", d=self.head_v_dim)

        q, k = self.rotary(
            q,
            k,
            seqlen_offset=seqlen_offset,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
        )

        q = q * self.qk_scale
        k = k * self.qk_scale

        if self.num_kv_groups > 1:
            k = repeat(k, "b t h d -> b t (h g) d", g=self.num_kv_groups)
            v = repeat(v, "b t h d -> b t (h g) d", g=self.num_kv_groups)

        ones = torch.ones_like(q[..., :1])
        q_aug = torch.cat([ones, q], dim=-1)
        k_aug = torch.cat([ones, k], dim=-1)

        num, num_state = self._run_linear_attn(
            q=q_aug,
            k=k_aug,
            v=v,
            initial_state=num_state,
            output_final_state=bool(use_cache),
            cu_seqlens=cu_seqlens,
        )
        den, den_state = self._run_linear_attn(
            q=q_aug,
            k=k_aug,
            v=torch.ones_like(v[..., :1]),
            initial_state=den_state,
            output_final_state=bool(use_cache),
            cu_seqlens=cu_seqlens,
        )

        den_safe = _safe_denominator(den, self.denom_eps)
        o = num / den_safe
        with torch.no_grad():
            self.last_error_stats = {
                "taylor_den_negative_frac": float((den < 0).float().mean().item()),
                "taylor_den_small_frac": float((den.abs() < self.denom_eps).float().mean().item()),
                "taylor_den_min": float(den.min().item()),
                "taylor_den_max": float(den.max().item()),
            }

        if past_key_values is not None and self.layer_idx is not None:
            past_key_values.update(
                recurrent_state=(num_state, den_state),
                conv_state=None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        if indices is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o, None, past_key_values


__all__ = ["TaylorLinearAttention"]
