from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, ShortConvolution

from .linear_attention_performer import (
    _build_gaussian_orthogonal_random_matrix,
    performer_softmax_feature_map,
)
from .linear_attention_performer_plus_triton import (
    _TRITON_AVAILABLE as _PERFORMER_PLUS_TRITON_AVAILABLE,
)
from .linear_attention_performer_plus_triton import (
    performer_plus_causal_linear_attention_triton,
)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def _build_antithetic_orthogonal_random_matrix(
    num_heads: int,
    num_features: int,
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    scaling: int,
    antithetic: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not antithetic:
        return _build_gaussian_orthogonal_random_matrix(
            num_heads=num_heads,
            num_features=num_features,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            scaling=scaling,
            generator=generator,
        )

    half = (num_features + 1) // 2
    base = _build_gaussian_orthogonal_random_matrix(
        num_heads=num_heads,
        num_features=half,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        scaling=scaling,
        generator=generator,
    )
    neg_count = num_features - half
    if neg_count <= 0:
        return base
    return torch.cat([base, -base[:, :neg_count]], dim=1)


class PerformerPlusLinearAttention(nn.Module):
    """
    Causal Performer+ linear attention with:
    1) antithetic orthogonal random features (variance reduction),
    2) optional adaptive forget gate in recurrent state updates.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 8,
        num_v_heads: int | None = None,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        performer_nb_features: int | None = None,
        performer_feature_eps: float = 1e-4,
        performer_ortho_scaling: int = 0,
        performer_redraw_projection: bool = False,
        performer_projection_seed: int | None = 0,
        performer_use_triton: bool = True,
        performer_antithetic_features: bool = True,
        performer_qk_l2_norm: bool = True,
        performer_use_decay: bool = True,
        performer_decay_init: float = 2.0,
        performer_per_layer_projection: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.layer_idx = layer_idx

        self.feature_eps = performer_feature_eps
        self.ortho_scaling = performer_ortho_scaling
        self.redraw_projection = performer_redraw_projection
        self.use_triton_kernel = performer_use_triton
        self.antithetic_features = performer_antithetic_features
        self.qk_l2_norm = performer_qk_l2_norm
        self.use_decay = performer_use_decay
        self.num_features = (
            performer_nb_features
            if performer_nb_features is not None
            else 2 * self.head_k_dim
        )

        self.projection_seed = performer_projection_seed
        if (
            self.projection_seed is not None
            and performer_per_layer_projection
            and layer_idx is not None
        ):
            # Keep each layer deterministic while avoiding shared projections.
            self.projection_seed = int(self.projection_seed) + int(layer_idx) * 100003

        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )
        if self.num_v_heads < self.num_heads:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} is not supported. Please use num_v_heads >= num_heads.",
            )
        if not math.isclose(self.head_v_dim, head_dim * expand_v, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce integer head_v_dim for head_dim={head_dim}.",
            )
        if self.num_features <= 0:
            raise ValueError(f"performer_nb_features must be > 0, got {self.num_features}.")
        if not self.use_triton_kernel:
            raise ValueError(
                "PerformerPlusLinearAttention is configured to run Triton-only. "
                "Set performer_use_triton=True.",
            )

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
        else:
            warnings.warn(
                "ShortConvolution is usually helpful. Disable only if you know the trade-off.",
            )

        if self.use_decay:
            self.decay_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
            self.decay_bias = nn.Parameter(
                torch.full((self.num_v_heads,), float(performer_decay_init), dtype=torch.float32)
            )
            self.decay_bias._no_weight_decay = True
        else:
            self.decay_proj = None
            self.decay_bias = None

        self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.register_buffer(
            "projection_matrix",
            torch.empty(self.num_heads, self.num_features, self.head_k_dim, dtype=torch.float32),
        )
        self._redraw_projection_matrix()

    @torch.no_grad()
    def _redraw_projection_matrix(self) -> None:
        generator = None
        if self.projection_seed is not None:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(int(self.projection_seed))
            self.projection_seed += 1
        matrix = _build_antithetic_orthogonal_random_matrix(
            num_heads=self.num_heads,
            num_features=self.num_features,
            head_dim=self.head_k_dim,
            device=torch.device('cpu'),
            dtype=self.projection_matrix.dtype,
            scaling=self.ortho_scaling,
            antithetic=self.antithetic_features,
            generator=generator,
        )
        self.projection_matrix.copy_(matrix.to(self.projection_matrix.device))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch_size, seq_len].")

        batch_size, q_len, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None and self.layer_idx is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
        cu_seqlens = kwargs.get('cu_seqlens')
        indices = None

        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, 'b t d -> (b t) d'),
                indices,
            ).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

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

        q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_k_dim)
        k = rearrange(k, 'b t (h d) -> b t h d', d=self.head_k_dim)
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        if self.qk_l2_norm:
            q = F.normalize(q.float(), dim=-1).to(q)
            k = F.normalize(k.float(), dim=-1).to(k)

        if self.num_v_heads > self.num_heads:
            groups = self.num_v_heads // self.num_heads
            q = repeat(q, 'b t h d -> b t (h g) d', g=groups)
            k = repeat(k, 'b t h d -> b t (h g) d', g=groups)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        if self.redraw_projection and self.training:
            self._redraw_projection_matrix()

        projection_matrix = self.projection_matrix
        if projection_matrix.shape[0] != q.shape[2]:
            if q.shape[2] % projection_matrix.shape[0] != 0:
                raise ValueError(
                    f"Head mismatch: q has {q.shape[2]} heads but projection has {projection_matrix.shape[0]} heads.",
                )
            groups = q.shape[2] // projection_matrix.shape[0]
            projection_matrix = repeat(projection_matrix, 'h m d -> (h g) m d', g=groups)

        q_prime = performer_softmax_feature_map(
            q,
            projection_matrix=projection_matrix,
            is_query=True,
            eps=self.feature_eps,
            cu_seqlens=cu_seqlens,
        )
        k_prime = performer_softmax_feature_map(
            k,
            projection_matrix=projection_matrix,
            is_query=False,
            eps=self.feature_eps,
            cu_seqlens=cu_seqlens,
        )

        if not _PERFORMER_PLUS_TRITON_AVAILABLE:
            raise RuntimeError("PerformerPlusLinearAttention requires Triton, but Triton kernel is unavailable.")
        if not q_prime.is_cuda:
            raise RuntimeError("PerformerPlusLinearAttention Triton path requires CUDA tensors.")

        log_decay = None
        if self.use_decay:
            decay_logits = self.decay_proj(hidden_states).float()
            decay_logits = decay_logits + self.decay_bias.view(1, 1, -1)
            log_decay = F.logsigmoid(decay_logits)

        o, recurrent_state = performer_plus_causal_linear_attention_triton(
            q_prime=q_prime,
            k_prime=k_prime,
            v=v,
            log_decay=log_decay,
            initial_state=recurrent_state,
            output_final_state=use_cache,
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
        o = rearrange(o, 'b t h d -> b t (h d)')
        if o.dtype != self.o_proj.weight.dtype:
            o = o.to(self.o_proj.weight.dtype)
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values


__all__ = ['PerformerPlusLinearAttention']
