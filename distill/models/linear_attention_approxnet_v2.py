from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, RotaryEmbedding, ShortConvolution
from fla.ops.utils.index import prepare_lens_from_mask

try:
    from .linear_attention_approxnet_v2_triton import (
        _TRITON_AVAILABLE as _APPROXNET_V2_TRITON_AVAILABLE,
        approxnet_v2_linear_attention_triton,
    )
except Exception:  # pragma: no cover
    _APPROXNET_V2_TRITON_AVAILABLE = False
    approxnet_v2_linear_attention_triton = None

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def _prepare_initial_state(
    q: torch.Tensor,
    v: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, _, k_dim = q.shape
    v_dim = v.shape[-1]
    device = q.device
    dtype = torch.float32
    if initial_state is None:
        d_state = torch.zeros((n, v_dim, k_dim), device=device, dtype=dtype)
        nu = torch.zeros((n, v_dim), device=device, dtype=dtype)
        kap = torch.zeros((n, k_dim), device=device, dtype=dtype)
        r_sum = torch.zeros((n, k_dim), device=device, dtype=dtype)
        o_prev = torch.zeros((n, v_dim), device=device, dtype=dtype)
        q_prev = torch.zeros((n, k_dim), device=device, dtype=dtype)
        count = torch.zeros((n,), device=device, dtype=dtype)
        return d_state, nu, kap, r_sum, o_prev, q_prev, count

    if len(initial_state) != 7:
        raise ValueError(
            "initial_state must be a 7-tuple: (D, nu, kappa, r_sum, o_prev, q_prev, count).",
        )
    d_state, nu, kap, r_sum, o_prev, q_prev, count = initial_state
    d_state = d_state.reshape(n, v_dim, k_dim).to(device=device, dtype=dtype).contiguous()
    nu = nu.reshape(n, v_dim).to(device=device, dtype=dtype).contiguous()
    kap = kap.reshape(n, k_dim).to(device=device, dtype=dtype).contiguous()
    r_sum = r_sum.reshape(n, k_dim).to(device=device, dtype=dtype).contiguous()
    o_prev = o_prev.reshape(n, v_dim).to(device=device, dtype=dtype).contiguous()
    q_prev = q_prev.reshape(n, k_dim).to(device=device, dtype=dtype).contiguous()
    count = count.reshape(n).to(device=device, dtype=dtype).contiguous()
    return d_state, nu, kap, r_sum, o_prev, q_prev, count


def _torch_approxnet_v2_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None = None,
    output_final_state: bool = False,
    beta_denom_eps: float = 1e-6,
    score_clip: float = 20.0,
    use_sigmoid_gate: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, dict[str, float]]:
    # q/k/v: [N, T, D]
    n, t, _ = q.shape
    qf = q.float()
    kf = k.float()
    vf = v.float()

    d_state, nu, kap, r_sum, o_prev, q_prev, count = _prepare_initial_state(qf, vf, initial_state)
    out = torch.empty((n, t, vf.shape[-1]), device=q.device, dtype=torch.float32)

    beta_mean_sum = 0.0
    den_min = float("inf")
    den_neg_sum = 0.0

    for i in range(t):
        q_i = qf[:, i]
        k_i = kf[:, i]
        v_i = vf[:, i]

        c_t = count + 1.0
        alpha = c_t.reciprocal()
        lam = count * alpha

        x_i = v_i - nu
        y_i = k_i - kap
        d_state = d_state + lam[:, None, None] * torch.einsum("nv,nk->nvk", x_i, y_i)

        dq_i = q_i - q_prev
        o_hist = o_prev + torch.einsum("nvk,nk->nv", d_state, dq_i)

        score = torch.sum(k_i * q_i, dim=-1)
        if use_sigmoid_gate:
            beta = torch.sigmoid(score)
        else:
            if score_clip > 0:
                score = score.clamp(min=-score_clip, max=score_clip)
            den = count + torch.sum(r_sum * q_i, dim=-1) + beta_denom_eps
            beta = torch.exp(score) / den
        o_i = o_hist + beta[:, None] * (v_i - o_hist)
        out[:, i] = o_i

        nu = nu + alpha[:, None] * (v_i - nu)
        kap = kap + alpha[:, None] * (k_i - kap)
        if not use_sigmoid_gate:
            r_sum = r_sum + k_i
        o_prev = o_i
        q_prev = q_i
        count = c_t

        beta_mean_sum += float(beta.mean().item())
        if not use_sigmoid_gate:
            den_min = min(den_min, float(den.min().item()))
            den_neg_sum += float((den < 0).float().mean().item())

    final_state = (d_state, nu, kap, r_sum, o_prev, q_prev, count) if output_final_state else None
    stats = {
        "approxnet_v2_beta_mean": beta_mean_sum / max(t, 1),
        "approxnet_v2_den_min": den_min if t > 0 else 0.0,
        "approxnet_v2_den_neg_frac": den_neg_sum / max(t, 1),
    }
    return out.to(q.dtype), final_state, stats


def _flatten_initial_state(
    initial_state: tuple[torch.Tensor, ...] | None,
    batch: int,
    heads: int,
    v_dim: int,
    k_dim: int,
) -> tuple[torch.Tensor, ...] | None:
    if initial_state is None:
        return None
    d_state, nu, kap, r_sum, o_prev, q_prev, count = initial_state
    return (
        d_state.reshape(batch * heads, k_dim, v_dim).transpose(-2, -1).contiguous(),
        nu.reshape(batch * heads, v_dim).contiguous(),
        kap.reshape(batch * heads, k_dim).contiguous(),
        r_sum.reshape(batch * heads, k_dim).contiguous(),
        o_prev.reshape(batch * heads, v_dim).contiguous(),
        q_prev.reshape(batch * heads, k_dim).contiguous(),
        count.reshape(batch * heads).contiguous(),
    )


def _reshape_final_state(
    final_state: tuple[torch.Tensor, ...],
    out_batch: int,
    n_heads: int,
    v_dim: int,
    k_dim: int,
) -> tuple[torch.Tensor, ...]:
    d_state, nu, kap, r_sum, o_prev, q_prev, count = final_state
    return (
        d_state.reshape(out_batch, n_heads, v_dim, k_dim).transpose(-2, -1).contiguous(),
        nu.reshape(out_batch, n_heads, v_dim).contiguous(),
        kap.reshape(out_batch, n_heads, k_dim).contiguous(),
        r_sum.reshape(out_batch, n_heads, k_dim).contiguous(),
        o_prev.reshape(out_batch, n_heads, v_dim).contiguous(),
        q_prev.reshape(out_batch, n_heads, k_dim).contiguous(),
        count.reshape(out_batch, n_heads).contiguous(),
    )


def _run_varlen(
    qf: torch.Tensor,
    kf: torch.Tensor,
    vf: torch.Tensor,
    n_heads: int,
    k_dim: int,
    v_dim: int,
    cu_seqlens: torch.LongTensor,
    initial_state: tuple[torch.Tensor, ...] | None,
    output_final_state: bool,
    beta_denom_eps: float,
    score_clip: float,
    use_sigmoid_gate: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, dict[str, float]]:
    n_seq = cu_seqlens.numel() - 1
    cu = cu_seqlens.tolist()
    if initial_state is not None:
        d0, nu0, kap0, r0, o0, qp0, c0 = initial_state
        init_flat = (
            d0.reshape(n_seq, n_heads, k_dim, v_dim).transpose(-2, -1).contiguous(),
            nu0.reshape(n_seq, n_heads, v_dim).contiguous(),
            kap0.reshape(n_seq, n_heads, k_dim).contiguous(),
            r0.reshape(n_seq, n_heads, k_dim).contiguous(),
            o0.reshape(n_seq, n_heads, v_dim).contiguous(),
            qp0.reshape(n_seq, n_heads, k_dim).contiguous(),
            c0.reshape(n_seq, n_heads).contiguous(),
        )
    else:
        init_flat = None

    out = torch.empty_like(vf)
    final_chunks = [[], [], [], [], [], [], []] if output_final_state else None
    beta_means = []
    den_negs = []
    den_mins = []

    for i in range(n_seq):
        bos, eos = int(cu[i]), int(cu[i + 1])
        seg_len = eos - bos
        init_seg = None if init_flat is None else (
            init_flat[0][i].contiguous(),
            init_flat[1][i].contiguous(),
            init_flat[2][i].contiguous(),
            init_flat[3][i].contiguous(),
            init_flat[4][i].contiguous(),
            init_flat[5][i].contiguous(),
            init_flat[6][i].contiguous(),
        )
        if seg_len == 0:
            if output_final_state:
                if init_seg is None:
                    end_seg = (
                        torch.zeros((n_heads, v_dim, k_dim), device=qf.device, dtype=torch.float32),
                        torch.zeros((n_heads, v_dim), device=qf.device, dtype=torch.float32),
                        torch.zeros((n_heads, k_dim), device=qf.device, dtype=torch.float32),
                        torch.zeros((n_heads, k_dim), device=qf.device, dtype=torch.float32),
                        torch.zeros((n_heads, v_dim), device=qf.device, dtype=torch.float32),
                        torch.zeros((n_heads, k_dim), device=qf.device, dtype=torch.float32),
                        torch.zeros((n_heads,), device=qf.device, dtype=torch.float32),
                    )
                else:
                    end_seg = init_seg
                for idx in range(7):
                    final_chunks[idx].append(end_seg[idx])
            continue

        out_seg, st_seg, stats_seg = _torch_approxnet_v2_dense(
            q=qf[:, bos:eos, :],
            k=kf[:, bos:eos, :],
            v=vf[:, bos:eos, :],
            initial_state=init_seg,
            output_final_state=output_final_state,
            beta_denom_eps=beta_denom_eps,
            score_clip=score_clip,
            use_sigmoid_gate=use_sigmoid_gate,
        )
        out[:, bos:eos, :] = out_seg
        beta_means.append(stats_seg["approxnet_v2_beta_mean"])
        den_negs.append(stats_seg["approxnet_v2_den_neg_frac"])
        den_mins.append(stats_seg["approxnet_v2_den_min"])
        if output_final_state:
            for idx in range(7):
                final_chunks[idx].append(st_seg[idx])

    final_state = tuple(torch.stack(chunks, dim=0) for chunks in final_chunks) if output_final_state else None
    stats = {
        "approxnet_v2_beta_mean": float(sum(beta_means) / max(len(beta_means), 1)),
        "approxnet_v2_den_neg_frac": float(sum(den_negs) / max(len(den_negs), 1)),
        "approxnet_v2_den_min": float(min(den_mins)) if den_mins else 0.0,
    }
    return out, final_state, stats


def approxnet_v2_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    beta_denom_eps: float = 1e-6,
    score_clip: float = 20.0,
    use_sigmoid_gate: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, dict[str, float]]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape [B, T, H, D].")
    batch, seqlen, n_heads, k_dim = q.shape
    v_dim = v.shape[-1]
    if k.shape[:3] != (batch, seqlen, n_heads):
        raise ValueError("q and k must share [B, T, H].")
    if v.shape[:3] != (batch, seqlen, n_heads):
        raise ValueError("q and v must share [B, T, H].")

    qf = q.permute(0, 2, 1, 3).contiguous().reshape(batch * n_heads, seqlen, k_dim)
    kf = k.permute(0, 2, 1, 3).contiguous().reshape(batch * n_heads, seqlen, k_dim)
    vf = v.permute(0, 2, 1, 3).contiguous().reshape(batch * n_heads, seqlen, v_dim)

    init_flat = _flatten_initial_state(initial_state, batch, n_heads, v_dim, k_dim)
    if cu_seqlens is None:
        out_flat, final_flat, stats = _torch_approxnet_v2_dense(
            q=qf,
            k=kf,
            v=vf,
            initial_state=init_flat,
            output_final_state=output_final_state,
            beta_denom_eps=beta_denom_eps,
            score_clip=score_clip,
            use_sigmoid_gate=use_sigmoid_gate,
        )
    else:
        out_flat, final_flat, stats = _run_varlen(
            qf=qf,
            kf=kf,
            vf=vf,
            n_heads=n_heads,
            k_dim=k_dim,
            v_dim=v_dim,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            output_final_state=output_final_state,
            beta_denom_eps=beta_denom_eps,
            score_clip=score_clip,
            use_sigmoid_gate=use_sigmoid_gate,
        )

    out = out_flat.reshape(batch, n_heads, seqlen, v_dim).permute(0, 2, 1, 3).contiguous()
    final_state = (
        _reshape_final_state(final_flat, cu_seqlens.numel() - 1 if cu_seqlens is not None else batch, n_heads, v_dim, k_dim)
        if output_final_state else None
    )
    return out, final_state, stats


class ApproxNetV2LinearAttention(nn.Module):
    def __init__(
        self,
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
        beta_denom_eps: float = 1e-6,
        score_clip: float = 20.0,
        qk_l2_norm: bool = False,
        qk_l2_norm_eps: float = 1e-6,
        use_triton: bool = True,
        recompute_chunk_size: int = 128,
        use_sigmoid_gate: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx
        self.beta_denom_eps = float(beta_denom_eps)
        self.score_clip = float(score_clip)
        self.qk_l2_norm = bool(qk_l2_norm)
        self.qk_l2_norm_eps = float(qk_l2_norm_eps)
        self.use_triton = bool(use_triton)
        self.recompute_chunk_size = int(recompute_chunk_size)
        self.max_position_embeddings = max_position_embeddings
        self.use_sigmoid_gate = bool(use_sigmoid_gate)

        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = self.num_heads * self.head_k_dim
        self.kv_key_dim = self.num_kv_heads * self.head_k_dim
        self.kv_value_dim = self.num_kv_heads * self.head_v_dim
        self.value_dim = self.num_heads * self.head_v_dim
        self.qk_scale = self.head_k_dim ** -0.25

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads={self.num_heads} must be divisible by num_kv_heads={self.num_kv_heads}.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_key_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_value_dim, bias=qkv_bias)

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
        self.last_error_stats: dict[str, float] = {}

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
        q = q * self.qk_scale
        k = k * self.qk_scale

        if self.num_kv_groups > 1:
            k = repeat(k, "b t h d -> b t (h g) d", g=self.num_kv_groups)
            v = repeat(v, "b t h d -> b t (h g) d", g=self.num_kv_groups)

        stateful = recurrent_state is not None or bool(use_cache)
        grad_enabled = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
        use_triton_path = bool(
            self.use_triton
            and _APPROXNET_V2_TRITON_AVAILABLE
            and q.is_cuda
            and not (stateful and grad_enabled)
        )
        if use_triton_path:
            o, recurrent_state, stats = approxnet_v2_linear_attention_triton(
                q=q,
                k=k,
                v=v,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens,
                beta_denom_eps=self.beta_denom_eps,
                score_clip=self.score_clip,
                recompute_chunk_size=self.recompute_chunk_size,
                use_sigmoid_gate=self.use_sigmoid_gate,
            )
        else:
            o, recurrent_state, stats = approxnet_v2_linear_attention(
                q=q,
                k=k,
                v=v,
                initial_state=recurrent_state,
                output_final_state=bool(use_cache),
                cu_seqlens=cu_seqlens,
                beta_denom_eps=self.beta_denom_eps,
                score_clip=self.score_clip,
                use_sigmoid_gate=self.use_sigmoid_gate,
            )
        self.last_error_stats = stats

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
    "ApproxNetV2LinearAttention",
    "approxnet_v2_linear_attention",
]
