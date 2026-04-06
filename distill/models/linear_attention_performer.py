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

from .linear_attention_performer_triton import (
    _TRITON_AVAILABLE as _PERFORMER_TRITON_AVAILABLE,
)
from .linear_attention_performer_triton import (
    performer_causal_linear_attention_triton,
)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def _build_gaussian_orthogonal_random_matrix(
    num_heads: int,
    num_features: int,
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    scaling: int = 0,
    stratified_norm_sampling: bool = False,
    stratified_jitter: bool = True,
    qmc_gaussian_sampling: bool = False,
    qmc_scramble: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if num_features <= 0:
        raise ValueError(f"num_features must be > 0, got {num_features}.")
    if head_dim <= 0:
        raise ValueError(f"head_dim must be > 0, got {head_dim}.")
    if scaling not in (0, 1):
        raise ValueError(f"scaling must be 0 or 1, got {scaling}.")

    def _draw_qmc_uniform(num_draws: int, dim: int) -> torch.Tensor | None:
        # Torch SobolEngine supports dimensions up to 21201.
        if dim <= 0 or dim > 21201:
            return None
        seed = None
        if generator is not None:
            seed = int(
                torch.randint(
                    0,
                    2**31 - 1,
                    (1,),
                    generator=generator,
                    device='cpu',
                    dtype=torch.int64,
                ).item()
            )
        engine = torch.quasirandom.SobolEngine(
            dimension=dim,
            scramble=bool(qmc_scramble),
            seed=seed,
        )
        return engine.draw(num_draws).to(device=device, dtype=torch.float32)

    def _draw_unstructured_block() -> torch.Tensor:
        if qmc_gaussian_sampling:
            # Prefer row-wise Sobol points in R^D over a single Sobol point in
            # R^(D^2). This keeps discrepancy control in the same ambient
            # dimension as each Gaussian row vector and empirically improves
            # finite-sample stability for orthogonal block construction.
            u = _draw_qmc_uniform(num_heads * head_dim, head_dim)
            if u is not None:
                u = u.clamp(1e-6, 1.0 - 1e-6)
                z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
                return z.view(num_heads, head_dim, head_dim)
            # Fallback for unsupported Sobol dimensions.
            dim = head_dim * head_dim
            u = _draw_qmc_uniform(num_heads, dim)
            if u is not None:
                u = u.clamp(1e-6, 1.0 - 1e-6)
                z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
                return z.view(num_heads, head_dim, head_dim)
        return torch.randn(
            num_heads,
            head_dim,
            head_dim,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )

    blocks: list[torch.Tensor] = []
    num_full_blocks = num_features // head_dim
    remaining_rows = num_features - num_full_blocks * head_dim

    for _ in range(num_full_blocks):
        unstructured = _draw_unstructured_block()
        q, r = torch.linalg.qr(unstructured, mode='reduced')
        # Sign-correct so each row direction is uniformly random.
        signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        q = q * signs.unsqueeze(-2)
        blocks.append(q.transpose(-2, -1))

    if remaining_rows > 0:
        unstructured = _draw_unstructured_block()
        q, r = torch.linalg.qr(unstructured, mode='reduced')
        signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        q = q * signs.unsqueeze(-2)
        blocks.append(q.transpose(-2, -1)[:, :remaining_rows, :])

    final_matrix = torch.cat(blocks, dim=1)

    if scaling == 0:
        if stratified_norm_sampling:
            # Stratified radial sampling (randomized quasi-Monte Carlo):
            # r_i ~= F^{-1}_{chi_d}(u_i), u_i in equal-probability bins.
            # This reduces variance of the random-feature kernel estimator while
            # keeping the expected radius profile close to chi(d).
            base_u = (torch.arange(num_features, device=device, dtype=torch.float32) + 0.5) / float(num_features)
            base_u = base_u.unsqueeze(0).expand(num_heads, -1)
            if stratified_jitter:
                rand_kwargs = {}
                if generator is not None and device.type == 'cpu':
                    rand_kwargs["generator"] = generator
                shift = torch.rand(num_heads, 1, device=device, dtype=torch.float32, **rand_kwargs)
                u = (base_u + shift) % 1.0
            else:
                u = base_u
            u = u.clamp(1e-6, 1.0 - 1e-6)

            # Wilson-Hilferty chi-square quantile approximation.
            # chi_d = sqrt(chi2_d), chi2_q(p) ~= d*(1 - 2/(9d) + z_p*sqrt(2/(9d)))^3
            z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
            d = float(head_dim)
            a = 1.0 - 2.0 / (9.0 * d)
            b = math.sqrt(2.0 / (9.0 * d))
            chi2_q = (d * (a + b * z).clamp_min(1e-4).pow(3)).clamp_min(1e-8)
            multiplier = chi2_q.sqrt()
        elif qmc_gaussian_sampling:
            u = _draw_qmc_uniform(num_heads, num_features)
            if u is not None:
                u = u.clamp(1e-6, 1.0 - 1e-6)
                z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
                d = float(head_dim)
                a = 1.0 - 2.0 / (9.0 * d)
                b = math.sqrt(2.0 / (9.0 * d))
                chi2_q = (d * (a + b * z).clamp_min(1e-4).pow(3)).clamp_min(1e-8)
                multiplier = chi2_q.sqrt()
            else:
                multiplier = torch.linalg.norm(
                    torch.randn(
                        num_heads,
                        num_features,
                        head_dim,
                        device=device,
                        dtype=torch.float32,
                        generator=generator,
                    ),
                    dim=-1,
                )
        else:
            multiplier = torch.linalg.norm(
                torch.randn(num_heads, num_features, head_dim, device=device, dtype=torch.float32, generator=generator),
                dim=-1,
            )
    else:
        multiplier = math.sqrt(float(head_dim)) * torch.ones(
            num_heads, num_features, device=device, dtype=torch.float32,
        )

    final_matrix = final_matrix * multiplier.unsqueeze(-1)
    return final_matrix.to(dtype=dtype)


def _segmentwise_key_max(data_dash: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    # data_dash: [1, T, H, M], cu_seqlens: [N+1]
    if data_dash.shape[0] != 1:
        raise ValueError("Segmentwise max currently expects batch size 1 in unpadded mode.")

    max_values = torch.empty(
        data_dash.shape[0], data_dash.shape[1], data_dash.shape[2], 1,
        device=data_dash.device,
        dtype=data_dash.dtype,
    )

    cu = cu_seqlens.tolist()
    for i in range(len(cu) - 1):
        start, end = int(cu[i]), int(cu[i + 1])
        seg = data_dash[:, start:end]
        seg_max = seg.amax(dim=(1, 3), keepdim=True)  # [1, 1, H, 1]
        max_values[:, start:end] = seg_max
    return max_values


def performer_softmax_feature_map(
    x: torch.Tensor,
    projection_matrix: torch.Tensor,
    *,
    is_query: bool,
    eps: float,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    # x: [B, T, H, D], projection_matrix: [H, M, D]
    x = x.to(torch.float32)
    projection_matrix = projection_matrix.to(torch.float32)

    data_normalizer = x.shape[-1] ** -0.25
    x = x * data_normalizer
    ratio = projection_matrix.shape[1] ** -0.5

    data_dash = torch.einsum('bthd,hmd->bthm', x, projection_matrix)
    diag_data = (x.square().sum(dim=-1, keepdim=True)) * 0.5

    if is_query:
        stabilizer = data_dash.amax(dim=-1, keepdim=True)
    else:
        if cu_seqlens is None:
            stabilizer = data_dash.amax(dim=(1, 3), keepdim=True)
        else:
            stabilizer = _segmentwise_key_max(data_dash, cu_seqlens)

    return ratio * (torch.exp(data_dash - diag_data - stabilizer) + eps)


def performer_causal_linear_attention(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    # q_prime, k_prime: [B, T, H, M]
    # v: [B, T, H, V]
    q_prime = q_prime.to(torch.float32)
    k_prime = k_prime.to(torch.float32)
    v = v.to(torch.float32)

    if cu_seqlens is None:
        kv = torch.einsum('bthm,bthv->bthmv', k_prime, v).cumsum(dim=1)
        k_cumsum = k_prime.cumsum(dim=1)

        if initial_state is not None:
            init_kv, init_k = initial_state
            kv = kv + init_kv.unsqueeze(1).to(torch.float32)
            k_cumsum = k_cumsum + init_k.unsqueeze(1).to(torch.float32)

        numerator = torch.einsum('bthm,bthmv->bthv', q_prime, kv)
        denominator = (q_prime * k_cumsum).sum(dim=-1, keepdim=True)
        out = numerator / (denominator + eps)

        final_state = None
        if output_final_state:
            final_state = (kv[:, -1], k_cumsum[:, -1])
        return out, final_state

    # Variable-length path (unpadded, batch dimension kept as 1).
    if q_prime.shape[0] != 1:
        raise ValueError("When cu_seqlens is provided, expected unpadded batch with leading dim=1.")

    out = torch.empty_like(v)
    final_kv_states: list[torch.Tensor] = []
    final_k_states: list[torch.Tensor] = []
    cu = cu_seqlens.tolist()
    num_seqs = len(cu) - 1

    for i in range(num_seqs):
        start, end = int(cu[i]), int(cu[i + 1])
        q_seg = q_prime[:, start:end]
        k_seg = k_prime[:, start:end]
        v_seg = v[:, start:end]

        kv_seg = torch.einsum('bthm,bthv->bthmv', k_seg, v_seg).cumsum(dim=1)
        k_seg_cumsum = k_seg.cumsum(dim=1)

        if initial_state is not None:
            init_kv, init_k = initial_state
            kv_seg = kv_seg + init_kv[i:i + 1].unsqueeze(1).to(torch.float32)
            k_seg_cumsum = k_seg_cumsum + init_k[i:i + 1].unsqueeze(1).to(torch.float32)

        num_seg = torch.einsum('bthm,bthmv->bthv', q_seg, kv_seg)
        den_seg = (q_seg * k_seg_cumsum).sum(dim=-1, keepdim=True)
        out[:, start:end] = num_seg / (den_seg + eps)

        if output_final_state:
            final_kv_states.append(kv_seg[:, -1].squeeze(0))
            final_k_states.append(k_seg_cumsum[:, -1].squeeze(0))

    final_state = None
    if output_final_state:
        final_state = (
            torch.stack(final_kv_states, dim=0),
            torch.stack(final_k_states, dim=0),
        )
    return out, final_state


class PerformerLinearAttention(nn.Module):
    """
    Causal Performer linear attention (FAVOR+ style) with recurrent prefix states.
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
        performer_qmc_gaussian_sampling: bool = False,
        performer_qmc_scramble: bool = True,
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
        self.projection_seed = performer_projection_seed
        self.use_triton_kernel = performer_use_triton
        self.qmc_gaussian_sampling = performer_qmc_gaussian_sampling
        self.qmc_scramble = performer_qmc_scramble
        self.num_features = performer_nb_features if performer_nb_features is not None else self.head_k_dim

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
            raise ValueError("PerformerLinearAttention is configured to run Triton-only. Set performer_use_triton=True.")

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
            # Keep subsequent redraws different but deterministic.
            self.projection_seed += 1
        matrix = _build_gaussian_orthogonal_random_matrix(
            num_heads=self.num_heads,
            num_features=self.num_features,
            head_dim=self.head_k_dim,
            device=torch.device('cpu'),
            dtype=self.projection_matrix.dtype,
            scaling=self.ortho_scaling,
            qmc_gaussian_sampling=self.qmc_gaussian_sampling,
            qmc_scramble=self.qmc_scramble,
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

        if not _PERFORMER_TRITON_AVAILABLE:
            raise RuntimeError("PerformerLinearAttention requires Triton, but Triton kernel is unavailable.")
        if not q_prime.is_cuda:
            raise RuntimeError("PerformerLinearAttention Triton path requires CUDA tensors.")

        o, recurrent_state = performer_causal_linear_attention_triton(
            q_prime=q_prime,
            k_prime=k_prime,
            v=v,
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


__all__ = [
    'PerformerLinearAttention',
    'performer_softmax_feature_map',
    'performer_causal_linear_attention',
]
