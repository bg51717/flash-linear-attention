from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, RotaryEmbedding, ShortConvolution
from fla.ops.simple_gla import fused_recurrent_simple_gla
from fla.ops.utils.index import prepare_lens_from_mask

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def sqk_feature_map(
    x: torch.Tensor,
    proj_weight: torch.Tensor,
    triu_i: torch.Tensor,
    triu_j: torch.Tensor,
    triu_scale: torch.Tensor,
) -> torch.Tensor:
    """Squared Kernel feature map.

    φ(x) = vech_scaled(g g^T),  g = Wx
    Guarantees φ(q)^T φ(k) = (g(q)^T g(k))^2 >= 0.

    Args:
        x: [*, d_head]
        proj_weight: [r, d_head]
        triu_i: [M] row indices of upper triangle
        triu_j: [M] col indices of upper triangle
        triu_scale: [M] 1.0 for diag, sqrt(2) for off-diag
    Returns:
        [*, M] where M = r(r+1)/2
    """
    g = F.linear(x, proj_weight)
    return g[..., triu_i] * g[..., triu_j] * triu_scale


@torch.compiler.disable
def sqk_linear_attention(
    phi_q: torch.Tensor,
    phi_k: torch.Tensor,
    v: torch.Tensor,
    log_gamma: torch.Tensor,
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """SQK linear attention via v-augmentation + fused_recurrent_simple_gla."""
    B, T, H, M = phi_q.shape
    DV = v.shape[-1]

    ones = v.new_ones(B, T, H, 1)
    v_aug = torch.cat([v, ones], dim=-1)

    h0 = None
    if initial_state is not None:
        S0, z0 = initial_state
        h0 = torch.cat([S0, z0.unsqueeze(-1)], dim=-1)

    g = log_gamma.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

    out_aug, ht_aug = fused_recurrent_simple_gla(
        q=phi_q,
        k=phi_k,
        v=v_aug,
        g=g,
        scale=1.0,
        initial_state=h0,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    num = out_aug[..., :DV]
    den = out_aug[..., DV:] + eps
    o = num / den

    final_state = None
    if output_final_state and ht_aug is not None:
        final_state = (ht_aug[..., :DV], ht_aug[..., DV:].squeeze(-1))

    return o, final_state


class SQKLinearAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'fused_recurrent',
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        output_norm: str = "identity",
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,
        max_position_embeddings: int | None = None,
        proj_rank: int = 16,
        decay_init: float = 4.0,
        denom_eps: float = 1e-6,
        qk_l2_norm: bool = True,
        qk_l2_norm_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx
        self.proj_rank = proj_rank
        self.feature_dim = proj_rank * (proj_rank + 1) // 2
        self.denom_eps = denom_eps
        self.qk_l2_norm = bool(qk_l2_norm)
        self.qk_l2_norm_eps = float(qk_l2_norm_eps)
        self.max_position_embeddings = max_position_embeddings

        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = self.num_heads * self.head_k_dim
        self.kv_key_dim = self.num_kv_heads * self.head_k_dim
        self.kv_value_dim = self.num_kv_heads * self.head_v_dim
        self.value_dim = self.num_heads * self.head_v_dim

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads={self.num_heads} must be divisible by num_kv_heads={self.num_kv_heads}.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_key_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_value_dim, bias=qkv_bias)

        self.proj_weight = nn.Parameter(torch.empty(proj_rank, self.head_k_dim))
        nn.init.xavier_normal_(self.proj_weight)

        idx_i, idx_j = torch.triu_indices(proj_rank, proj_rank)
        self.register_buffer('triu_idx_i', idx_i)
        self.register_buffer('triu_idx_j', idx_j)
        scale = torch.where(idx_i == idx_j, 1.0, math.sqrt(2))
        self.register_buffer('triu_scale', scale)

        self.raw_decay = nn.Parameter(torch.full((num_heads,), decay_init))

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.kv_key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.kv_value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        if output_norm == "rmsnorm":
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)
        elif output_norm == "identity":
            self.o_norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported output_norm `{output_norm}`.")
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(dim=self.head_k_dim, base=rope_theta)

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
        recurrent_state = last_state["recurrent_state"] if last_state is not None else None

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

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))
            conv_state_q, conv_state_k, conv_state_v = None, None, None

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
        if self.qk_l2_norm:
            q = F.normalize(q, p=2, dim=-1, eps=self.qk_l2_norm_eps)
            k = F.normalize(k, p=2, dim=-1, eps=self.qk_l2_norm_eps)

        if self.num_kv_groups > 1:
            k = repeat(k, "b t h d -> b t (h g) d", g=self.num_kv_groups)
            v = repeat(v, "b t h d -> b t (h g) d", g=self.num_kv_groups)

        phi_q = sqk_feature_map(q, self.proj_weight, self.triu_idx_i, self.triu_idx_j, self.triu_scale)
        phi_k = sqk_feature_map(k, self.proj_weight, self.triu_idx_i, self.triu_idx_j, self.triu_scale)

        log_gamma = F.logsigmoid(self.raw_decay)

        o, recurrent_state = sqk_linear_attention(
            phi_q=phi_q,
            phi_k=phi_k,
            v=v,
            log_gamma=log_gamma,
            eps=self.denom_eps,
            initial_state=recurrent_state,
            output_final_state=bool(use_cache),
            cu_seqlens=cu_seqlens,
        )

        if past_key_values is not None and self.layer_idx is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if indices is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o, None, past_key_values


__all__ = [
    "SQKLinearAttention",
    "sqk_linear_attention",
    "sqk_feature_map",
]
