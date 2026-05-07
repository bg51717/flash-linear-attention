from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, RotaryEmbedding, ShortConvolution
from fla.ops.utils.index import prepare_lens_from_mask

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache

_TRITON_AVAILABLE = False
try:
    from .linear_attention_sisa_triton import fused_recurrent_sisa
    _TRITON_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Naive reference (for testing only, not for training)
# Matches fla convention: inputs are [B, H, T, D]
# ---------------------------------------------------------------------------

def sisa_recurrence_naive(
    q_r: torch.Tensor,
    k_r: torch.Tensor,
    v: torch.Tensor,
    decay_alpha: torch.Tensor,
    decay_bias: torch.Tensor,
    write_alpha: torch.Tensor,
    write_bias: torch.Tensor,
    log_beta: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    """
    Pure PyTorch reference. For correctness verification only.

    Args:
        q_r: [B, H, T, d_r]
        k_r: [B, H, T, d_r]
        v:   [B, H, T, d_v]
        decay_alpha, decay_bias, write_alpha, write_bias: [H]
        log_beta: [H]
        initial_state: (S[B,H,d_r,d_v], G[B,H,d_r,d_r]) or None
    Returns:
        o: [B, H, T, d_v]
        final_state: (S, G) or None
    """
    orig_dtype = q_r.dtype
    b, h, t, d_r = q_r.shape
    d_v = v.shape[-1]
    q_r, k_r, v = q_r.float(), k_r.float(), v.float()
    da, db = decay_alpha.float(), decay_bias.float()
    wa, wb = write_alpha.float(), write_bias.float()
    sm_beta = torch.exp(log_beta.float())

    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_r, d_v, device=q_r.device, dtype=torch.float32)
    G = torch.zeros(b, h, d_r, d_r, device=q_r.device, dtype=torch.float32)
    if initial_state is not None:
        S = S + initial_state[0].float()
        G = G + initial_state[1].float()

    for i in range(t):
        q_i = q_r[:, :, i]
        k_i = k_r[:, :, i]
        v_i = v[:, :, i]

        score = (k_i * q_i).sum(-1)
        alpha = torch.sigmoid(da * score + db)
        beta_gate = torch.sigmoid(wa * score + wb)

        a_S = alpha[:, :, None, None]
        b_S = beta_gate[:, :, None, None]
        S = a_S * S + b_S * (k_i[:, :, :, None] * v_i[:, :, None, :])

        a_G = alpha[:, :, None, None]
        b_G = beta_gate[:, :, None, None]
        G = a_G * G + b_G * (k_i[:, :, :, None] * k_i[:, :, None, :])

        # SiSA readout: softmax gating on Gram-query response
        Gq = torch.einsum('bhrc,bhc->bhr', G.detach(), q_i)
        w = F.softmax(sm_beta[None, :, None] * Gq, dim=-1)
        q_tilde = q_i * w
        q_n = F.normalize(q_tilde, p=2, dim=-1)

        o[:, :, i] = torch.einsum('bhrd,bhr->bhd', S, q_n)

    final = (S, G) if output_final_state else None
    return o.to(orig_dtype), final


# ---------------------------------------------------------------------------
# Optimized recurrent (for training)
# Uses gradient checkpointing + torch.compile on the inner step
# ---------------------------------------------------------------------------

def _sisa_step(
    q_i: torch.Tensor,
    k_i: torch.Tensor,
    v_i: torch.Tensor,
    da: torch.Tensor,
    db: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    sm_beta: torch.Tensor,
    S: torch.Tensor,
    G: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    score = (k_i * q_i).sum(-1)
    alpha = torch.sigmoid(da * score + db)[:, None, None]
    beta_gate = torch.sigmoid(wa * score + wb)[:, None, None]

    S = alpha * S + beta_gate * (k_i.unsqueeze(-1) * v_i.unsqueeze(-2))
    G = alpha * G + beta_gate * (k_i.unsqueeze(-1) * k_i.unsqueeze(-2))

    Gq = torch.einsum('nrc,nc->nr', G.detach(), q_i)
    w = F.softmax(sm_beta[:, None] * Gq, dim=-1)
    q_tilde = q_i * w
    q_n = F.normalize(q_tilde, p=2, dim=-1)

    o_i = torch.einsum('nrv,nr->nv', S, q_n)

    return o_i, S, G


try:
    _sisa_step_compiled = torch.compile(_sisa_step)
except Exception:
    _sisa_step_compiled = _sisa_step


def _sisa_segment_fn(
    q_r_seg: torch.Tensor,
    k_r_seg: torch.Tensor,
    v_seg: torch.Tensor,
    da: torch.Tensor,
    db: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    sm_beta: torch.Tensor,
    S: torch.Tensor,
    G: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, t_seg, _ = q_r_seg.shape
    d_v = v_seg.shape[-1]
    outs = torch.empty(n, t_seg, d_v, device=S.device, dtype=S.dtype)
    for i in range(t_seg):
        outs[:, i], S, G = _sisa_step_compiled(
            q_r_seg[:, i], k_r_seg[:, i], v_seg[:, i],
            da, db, wa, wb, sm_beta, S, G,
        )
    return outs, S, G


def _fused_recurrent_sisa(
    q_r: torch.Tensor,
    k_r: torch.Tensor,
    v: torch.Tensor,
    decay_alpha: torch.Tensor,
    decay_bias: torch.Tensor,
    write_alpha: torch.Tensor,
    write_bias: torch.Tensor,
    sm_beta: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, dict[str, float]]:
    """Optimized recurrent with gradient checkpointing per segment."""
    n, t, d_r = q_r.shape
    d_v = v.shape[-1]
    q_rf = q_r.float()
    k_rf = k_r.float()
    vf = v.float()
    da = decay_alpha.float()
    db = decay_bias.float()
    wa = write_alpha.float()
    wb = write_bias.float()
    smb = sm_beta.float()

    if initial_state is None:
        S = torch.zeros((n, d_r, d_v), device=q_r.device, dtype=torch.float32)
        G = torch.zeros((n, d_r, d_r), device=q_r.device, dtype=torch.float32)
    else:
        S_init, G_init = initial_state
        S = S_init.reshape(n, d_r, d_v).to(device=q_r.device, dtype=torch.float32).contiguous()
        G = G_init.reshape(n, d_r, d_r).to(device=q_r.device, dtype=torch.float32).contiguous()

    use_checkpoint = torch.is_grad_enabled() and (
        q_r.requires_grad or k_r.requires_grad or v.requires_grad
    )
    out = torch.empty(n, t, d_v, device=q_r.device, dtype=torch.float32)
    for seg_start in range(0, t, chunk_size):
        seg_end = min(seg_start + chunk_size, t)
        q_seg = q_rf[:, seg_start:seg_end]
        k_seg = k_rf[:, seg_start:seg_end]
        v_seg = vf[:, seg_start:seg_end]

        if use_checkpoint:
            out_seg, S, G = torch.utils.checkpoint.checkpoint(
                _sisa_segment_fn, q_seg, k_seg, v_seg,
                da, db, wa, wb, smb, S, G,
                use_reentrant=False,
            )
        else:
            out_seg, S, G = _sisa_segment_fn(
                q_seg, k_seg, v_seg, da, db, wa, wb, smb, S, G,
            )
        out[:, seg_start:seg_end] = out_seg

    final_state = (S, G) if output_final_state else None

    with torch.no_grad():
        mid = min(t // 2, t - 1)
        score_sample = (k_rf[:, mid] * q_rf[:, mid]).sum(-1)
        decay_sample = torch.sigmoid(da * score_sample + db).mean().item()
        write_sample = torch.sigmoid(wa * score_sample + wb).mean().item()
    stats = {
        "sisa_decay_mean": float(decay_sample),
        "sisa_write_mean": float(write_sample),
    }
    return out.to(q_r.dtype), final_state, stats


# ---------------------------------------------------------------------------
# Entry point: handles [B,T,H,D] reshape and varlen dispatch
# ---------------------------------------------------------------------------

def _run_varlen(
    q_rf: torch.Tensor,
    k_rf: torch.Tensor,
    vf: torch.Tensor,
    decay_alpha: torch.Tensor,
    decay_bias: torch.Tensor,
    write_alpha: torch.Tensor,
    write_bias: torch.Tensor,
    sm_beta: torch.Tensor,
    n_heads: int,
    d_r: int,
    d_v: int,
    cu_seqlens: torch.LongTensor,
    initial_state: tuple[torch.Tensor, ...] | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, dict[str, float]]:
    n_seq = cu_seqlens.numel() - 1
    cu = cu_seqlens.tolist()

    if initial_state is not None:
        S0, G0 = initial_state
        init_reshaped = (
            S0.reshape(n_seq, n_heads, d_r, d_v).contiguous(),
            G0.reshape(n_seq, n_heads, d_r, d_r).contiguous(),
        )
    else:
        init_reshaped = None

    out = torch.empty_like(vf)
    final_chunks: list[list[torch.Tensor]] | None = [[], []] if output_final_state else None
    decay_means: list[float] = []
    write_means: list[float] = []

    for i in range(n_seq):
        bos, eos = int(cu[i]), int(cu[i + 1])
        seg_len = eos - bos

        init_seg = None
        if init_reshaped is not None:
            init_seg = (
                init_reshaped[0][i].contiguous(),
                init_reshaped[1][i].contiguous(),
            )

        if seg_len == 0:
            if output_final_state:
                if init_seg is None:
                    final_chunks[0].append(torch.zeros((n_heads, d_r, d_v), device=q_rf.device, dtype=torch.float32))
                    final_chunks[1].append(torch.zeros((n_heads, d_r, d_r), device=q_rf.device, dtype=torch.float32))
                else:
                    final_chunks[0].append(init_seg[0])
                    final_chunks[1].append(init_seg[1])
            continue

        out_seg, st_seg, stats_seg = _fused_recurrent_sisa(
            q_r=q_rf[:, bos:eos, :],
            k_r=k_rf[:, bos:eos, :],
            v=vf[:, bos:eos, :],
            decay_alpha=decay_alpha,
            decay_bias=decay_bias,
            write_alpha=write_alpha,
            write_bias=write_bias,
            sm_beta=sm_beta,
            initial_state=init_seg,
            output_final_state=output_final_state,
        )
        out[:, bos:eos, :] = out_seg
        decay_means.append(stats_seg["sisa_decay_mean"])
        write_means.append(stats_seg["sisa_write_mean"])
        if output_final_state:
            final_chunks[0].append(st_seg[0])
            final_chunks[1].append(st_seg[1])

    final_state = None
    if output_final_state:
        final_state = (
            torch.stack(final_chunks[0], dim=0),
            torch.stack(final_chunks[1], dim=0),
        )
    stats = {
        "sisa_decay_mean": float(sum(decay_means) / max(len(decay_means), 1)),
        "sisa_write_mean": float(sum(write_means) / max(len(write_means), 1)),
    }
    return out, final_state, stats


def _triton_sisa_path(
    q_r: torch.Tensor,
    k_r: torch.Tensor,
    v: torch.Tensor,
    decay_alpha: torch.Tensor,
    decay_bias: torch.Tensor,
    write_alpha: torch.Tensor,
    write_bias: torch.Tensor,
    sm_beta: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, dict[str, float]]:
    batch, seqlen, n_heads, d_r = q_r.shape
    d_v = v.shape[-1]

    q_c = q_r.contiguous()
    k_c = k_r.contiguous()
    v_c = v.contiguous()

    score = (q_c.float() * k_c.float()).sum(-1)
    alpha = torch.sigmoid(decay_alpha * score + decay_bias).detach()
    beta_gate = torch.sigmoid(write_alpha * score + write_bias)

    triton_init = None
    if initial_state is not None:
        S0, G0 = initial_state
        n_init = S0.shape[0]
        triton_init = (
            S0.reshape(n_init * n_heads, d_r, d_v).contiguous().float(),
            G0.reshape(n_init * n_heads, d_r * d_r).contiguous().float(),
        )

    o, final_state = fused_recurrent_sisa(
        q_r=q_c.float(), k_r=k_c.float(), v=v_c.float(),
        alpha=alpha, beta=beta_gate, sm_beta=sm_beta,
        initial_state=triton_init,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    final_out = None
    if output_final_state and final_state is not None:
        final_out = final_state

    with torch.no_grad():
        mid = min(seqlen // 2, seqlen - 1)
        stats = {
            "sisa_decay_mean": alpha[:, mid].mean().item(),
            "sisa_write_mean": beta_gate[:, mid].mean().item(),
        }
    return o.to(q_r.dtype), final_out, stats


@torch.compiler.disable
def sisa_linear_attention(
    q_r: torch.Tensor,
    k_r: torch.Tensor,
    v: torch.Tensor,
    decay_alpha: torch.Tensor,
    decay_bias: torch.Tensor,
    write_alpha: torch.Tensor,
    write_bias: torch.Tensor,
    sm_beta: torch.Tensor,
    initial_state: tuple[torch.Tensor, ...] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_triton: bool | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, dict[str, float]]:
    """
    Args:
        q_r: [B, T, H, d_r]
        k_r: [B, T, H, d_r]
        v:   [B, T, H, d_v]
        decay_alpha, decay_bias, write_alpha, write_bias: [H]
        sm_beta: [H] per-head softmax temperature (exp of learnable log_beta)
        initial_state: (S[N,H,d_r,d_v], G[N,H,d_r,d_r]) or None
        use_triton: None=auto, True=force Triton, False=force PyTorch
    Returns:
        o: [B, T, H, d_v]
        final_state: tuple or None
        stats: dict
    """
    if q_r.ndim != 4 or k_r.ndim != 4 or v.ndim != 4:
        raise ValueError("q_r, k_r, v must have shape [B, T, H, D].")

    if use_triton is None:
        use_triton = _TRITON_AVAILABLE and q_r.is_cuda
    if use_triton:
        return _triton_sisa_path(
            q_r, k_r, v,
            decay_alpha, decay_bias, write_alpha, write_bias, sm_beta,
            initial_state, output_final_state, cu_seqlens,
        )

    batch, seqlen, n_heads, d_r = q_r.shape
    d_v = v.shape[-1]

    q_f = q_r.permute(0, 2, 1, 3).contiguous().reshape(batch * n_heads, seqlen, d_r)
    k_f = k_r.permute(0, 2, 1, 3).contiguous().reshape(batch * n_heads, seqlen, d_r)
    v_f = v.permute(0, 2, 1, 3).contiguous().reshape(batch * n_heads, seqlen, d_v)

    da = decay_alpha.repeat(batch)
    db = decay_bias.repeat(batch)
    wa = write_alpha.repeat(batch)
    wb = write_bias.repeat(batch)
    smb = sm_beta.repeat(batch)

    def _flatten_init(state):
        if state is None:
            return None
        S, G = state
        return (
            S.reshape(batch * n_heads, d_r, d_v).contiguous(),
            G.reshape(batch * n_heads, d_r, d_r).contiguous(),
        )

    init_flat = _flatten_init(initial_state)
    if cu_seqlens is None:
        out_flat, final_flat, stats = _fused_recurrent_sisa(
            q_r=q_f, k_r=k_f, v=v_f,
            decay_alpha=da, decay_bias=db,
            write_alpha=wa, write_bias=wb,
            sm_beta=smb,
            initial_state=init_flat,
            output_final_state=output_final_state,
        )
    else:
        out_flat, final_flat, stats = _run_varlen(
            q_rf=q_f, k_rf=k_f, vf=v_f,
            decay_alpha=da, decay_bias=db,
            write_alpha=wa, write_bias=wb,
            sm_beta=smb,
            n_heads=n_heads, d_r=d_r, d_v=d_v,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    out = out_flat.reshape(batch, n_heads, seqlen, d_v).permute(0, 2, 1, 3).contiguous()
    final_state = None
    if output_final_state and final_flat is not None:
        S_final, G_final = final_flat
        n_out = cu_seqlens.numel() - 1 if cu_seqlens is not None else batch
        final_state = (
            S_final.reshape(n_out, n_heads, d_r, d_v).contiguous(),
            G_final.reshape(n_out, n_heads, d_r, d_r).contiguous(),
        )
    return out, final_state, stats


# ---------------------------------------------------------------------------
# nn.Module layer — matches fla/layers/delta_net.py pattern
# ---------------------------------------------------------------------------

class SiSALinearAttention(nn.Module):

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
        d_r: int = 16,
        decay_alpha_init: float = 0.0,
        decay_bias_init: float = 2.0,
        write_alpha_init: float = 1.0,
        write_bias_init: float = 0.0,
        qk_l2_norm: bool = True,
        qk_l2_norm_eps: float = 1e-6,
        beta_init: float = 1.0,
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
        self.d_r = int(d_r)
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

        self.proj_weight = nn.Parameter(torch.empty(self.d_r, self.head_k_dim))
        nn.init.orthogonal_(self.proj_weight)

        self.decay_alpha = nn.Parameter(torch.full((num_heads,), decay_alpha_init))
        self.decay_bias = nn.Parameter(torch.full((num_heads,), decay_bias_init))
        self.write_alpha = nn.Parameter(torch.full((num_heads,), write_alpha_init))
        self.write_bias = nn.Parameter(torch.full((num_heads,), write_bias_init))

        self.log_beta = nn.Parameter(torch.full((num_heads,), math.log(beta_init)))

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

        q_r = F.linear(q, self.proj_weight)
        k_r = F.linear(k, self.proj_weight)
        q_r = F.normalize(q_r, p=2, dim=-1, eps=self.qk_l2_norm_eps)
        k_r = F.normalize(k_r, p=2, dim=-1, eps=self.qk_l2_norm_eps)

        if self.num_kv_groups > 1:
            k_r = repeat(k_r, "b t h d -> b t (h g) d", g=self.num_kv_groups)
            v = repeat(v, "b t h d -> b t (h g) d", g=self.num_kv_groups)

        sm_beta = torch.exp(self.log_beta)

        o, recurrent_state, stats = sisa_linear_attention(
            q_r=q_r,
            k_r=k_r,
            v=v,
            decay_alpha=self.decay_alpha,
            decay_bias=self.decay_bias,
            write_alpha=self.write_alpha,
            write_bias=self.write_bias,
            sm_beta=sm_beta,
            initial_state=recurrent_state,
            output_final_state=bool(use_cache),
            cu_seqlens=cu_seqlens,
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
    "SiSALinearAttention",
    "sisa_linear_attention",
    "sisa_recurrence_naive",
]
