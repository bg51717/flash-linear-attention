import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers import delta_net, gated_deltanet
from fla.layers.linear_attn import LinearAttention as FLALinearAttention
from fla.modules import RMSNorm, RotaryEmbedding
from typing import Optional
try:
    from fla.layers.attn import Attention as FLAAttention
    _FLA_ATTENTION_AVAILABLE = True
except Exception:  # pragma: no cover
    FLAAttention = None
    _FLA_ATTENTION_AVAILABLE = False

from fla.layers.pdf import FirstOrderLinearAttention
from fla.layers.pdf_final import PDFFinalLinearAttention
from fla.layers.performer import PerformerLinearAttention
from fla.layers.performer_plus import PerformerPlusLinearAttention
from fla.layers.taylor import TaylorLinearAttention
from fla.layers.approxnet_v2 import ApproxNetV2LinearAttention
from fla.layers.approxnet_v3 import ApproxNetV3LinearAttention
from fla.layers.approxnet_v4 import ApproxNetV4LinearAttention
from fla.layers.soam import SOAMLinearAttention
from fla.layers.wla import WLALinearAttention
from fla.layers.sisa import SiSALinearAttention
from fla.layers.dual_delta_net import DualDeltaNet
from fla.layers.hpk import HPKLinearAttention
from fla.layers.sqk import SQKLinearAttention
from fla.layers.css import CSSLinearAttention
from fla.layers.mean_delta_net import MeanDeltaNet


class TorchMHAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        qk_unit_norm: bool = False,
        qk_unit_norm_eps: float = 1e-6,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = 10000.0,
        max_position_embeddings: Optional[int] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.qk_unit_norm = qk_unit_norm
        self.qk_unit_norm_eps = qk_unit_norm_eps
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, dtype=torch.float32)
            self.k_norm = RMSNorm(self.head_dim, dtype=torch.float32)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def _project(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        return x.view(*x.shape[:-1], heads, self.head_dim)

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_groups == 1:
            return x
        return x.repeat_interleave(self.num_kv_groups, dim=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        del use_cache

        if attention_mask is not None:
            assert attention_mask.dim() == 2, (
                "Expected attention_mask with shape [batch_size, seq_len]."
            )

        bsz, q_len, _ = hidden_states.size()
        q = self._project(self.q_proj(hidden_states), self.num_heads)
        k = self._project(self.k_proj(hidden_states), self.num_kv_heads)
        v = self._project(self.v_proj(hidden_states), self.num_kv_heads)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.qk_unit_norm:
            q = F.normalize(q, p=2, dim=-1, eps=self.qk_unit_norm_eps)
            k = F.normalize(k, p=2, dim=-1, eps=self.qk_unit_norm_eps)

        cu_seqlens = kwargs.get("cu_seqlens", None)
        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q_len + seqlen_offset

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        q, k = self.rotary(
            q,
            k,
            seqlen_offset=seqlen_offset,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
        )

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )["attn_state"]
            if cache_has_content:
                k = k_cached.view(*k_cached.shape[:-1], self.num_kv_heads, self.head_dim)
                v = v_cached.view(*v_cached.shape[:-1], self.num_kv_heads, self.head_dim)

        k = self._expand_kv(k)
        v = self._expand_kv(v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_len = k.shape[-2]
        attn_mask = None
        if attention_mask is not None:
            key_mask = attention_mask
            if key_mask.shape[-1] != k_len:
                key_mask = key_mask[:, -k_len:]
            key_mask = key_mask.to(torch.bool)[:, None, None, :]
            neg_inf = torch.finfo(q.dtype).min
            attn_mask = torch.zeros(
                (bsz, 1, q_len, k_len), dtype=q.dtype, device=q.device
            )
            attn_mask = attn_mask.masked_fill(~key_mask, neg_inf)

        # In incremental decoding (q_len=1, cache exists), `is_causal=False` is
        # correct because keys only contain valid history and current token.
        use_causal = not (past_key_values is not None and q_len == 1)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=use_causal
        )
        o = o.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        o = self.o_proj(o)

        attentions = None
        return o, attentions, past_key_values


TYPE2ATTN = {
    "gated_deltanet": gated_deltanet.GatedDeltaNet,
    "delta_net": delta_net.DeltaNet,
    "dual_delta_net": DualDeltaNet,
    "dual_deltanet": DualDeltaNet,
    "mean_delta_net": MeanDeltaNet,
    "mean_deltanet": MeanDeltaNet,
    "pdf_linear_attention": FirstOrderLinearAttention,
    "first_order_linear_attention": FirstOrderLinearAttention,
    "pdf_final_linear_attention": PDFFinalLinearAttention,
    "pdf_refined_linear_attention": PDFFinalLinearAttention,
    "taylor_linear_attention": TaylorLinearAttention,
    "softmax_taylor_linear_attention": TaylorLinearAttention,
    "approxnet_v2_linear_attention": ApproxNetV2LinearAttention,
    "approxnet_v2": ApproxNetV2LinearAttention,
    "approxnet_v3_linear_attention": ApproxNetV3LinearAttention,
    "approxnet_v3": ApproxNetV3LinearAttention,
    "approxnet_v4_linear_attention": ApproxNetV4LinearAttention,
    "approxnet_v4": ApproxNetV4LinearAttention,
    "soam_linear_attention": SOAMLinearAttention,
    "soam": SOAMLinearAttention,
    "wla_linear_attention": WLALinearAttention,
    "wla": WLALinearAttention,
    "sisa_linear_attention": SiSALinearAttention,
    "sisa": SiSALinearAttention,
    "original_linear_attention": FLALinearAttention,
    "vanilla_linear_attention": FLALinearAttention,
    "mha_torch_attention": TorchMHAAttention,
    "mha_torch": TorchMHAAttention,
    "mha_torch_qk_unit_norm_attention": TorchMHAAttention,
    "mha_torch_qk_unit_norm": TorchMHAAttention,
    "performer_linear_attention": PerformerLinearAttention,
    "performer_plus_linear_attention": PerformerPlusLinearAttention,
    "hpk_linear_attention": HPKLinearAttention,
    "hpk": HPKLinearAttention,
    "sqk_linear_attention": SQKLinearAttention,
    "sqk": SQKLinearAttention,
    "css_linear_attention": CSSLinearAttention,
    "css": CSSLinearAttention,
}

CONFIG2KWARGS = {
    "gated_deltanet": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_v_heads": lambda cfg: (
            getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            if getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            >= cfg.num_attention_heads
            else cfg.num_attention_heads
        ),
    },
    "delta_net": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "mode": lambda cfg: getattr(cfg, "delta_net_mode", "chunk"),
        "expand_k": lambda cfg: getattr(cfg, "expand_k", 1.0),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_beta": lambda cfg: getattr(cfg, "use_beta", True),
        "use_gate": lambda cfg: getattr(cfg, "use_gate", False),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "allow_neg_eigval": lambda cfg: getattr(cfg, "allow_neg_eigval", False),
        "qk_activation": lambda cfg: getattr(cfg, "qk_activation", "silu"),
        "qk_norm": lambda cfg: getattr(cfg, "qk_norm", "l2"),
    },
    "dual_delta_net": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "mode": lambda cfg: getattr(cfg, "dual_delta_net_mode", "fused_recurrent"),
        "expand_k": lambda cfg: getattr(cfg, "expand_k", 1.0),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_gate": lambda cfg: getattr(cfg, "use_gate", False),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "qk_activation": lambda cfg: getattr(cfg, "qk_activation", "silu"),
        "qk_norm": lambda cfg: getattr(cfg, "qk_norm", "l2"),
        "value_l2_norm": lambda cfg: getattr(cfg, "dual_delta_value_l2_norm", True),
        "value_norm_eps": lambda cfg: getattr(cfg, "dual_delta_value_norm_eps", 1e-6),
        "dual_recompute_chunk_size": lambda cfg: getattr(cfg, "dual_recompute_chunk_size", 128),
    },
    "dual_deltanet": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "mode": lambda cfg: getattr(cfg, "dual_delta_net_mode", "fused_recurrent"),
        "expand_k": lambda cfg: getattr(cfg, "expand_k", 1.0),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_gate": lambda cfg: getattr(cfg, "use_gate", False),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "qk_activation": lambda cfg: getattr(cfg, "qk_activation", "silu"),
        "qk_norm": lambda cfg: getattr(cfg, "qk_norm", "l2"),
        "value_l2_norm": lambda cfg: getattr(cfg, "dual_delta_value_l2_norm", True),
        "value_norm_eps": lambda cfg: getattr(cfg, "dual_delta_value_norm_eps", 1e-6),
        "dual_recompute_chunk_size": lambda cfg: getattr(cfg, "dual_recompute_chunk_size", 128),
    },
    "mean_delta_net": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "mode": lambda cfg: getattr(cfg, "mean_delta_net_mode", "fused_recurrent"),
        "expand_k": lambda cfg: getattr(cfg, "expand_k", 1.0),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_gate": lambda cfg: getattr(cfg, "use_gate", False),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "qk_activation": lambda cfg: getattr(cfg, "qk_activation", "silu"),
        "qk_norm": lambda cfg: getattr(cfg, "qk_norm", "l2"),
        "value_l2_norm": lambda cfg: getattr(cfg, "mean_delta_value_l2_norm", True),
        "value_norm_eps": lambda cfg: getattr(cfg, "mean_delta_value_norm_eps", 1e-6),
        "mean_recompute_chunk_size": lambda cfg: getattr(cfg, "mean_recompute_chunk_size", 128),
    },
    "mean_deltanet": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "mode": lambda cfg: getattr(cfg, "mean_delta_net_mode", "fused_recurrent"),
        "expand_k": lambda cfg: getattr(cfg, "expand_k", 1.0),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_gate": lambda cfg: getattr(cfg, "use_gate", False),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "qk_activation": lambda cfg: getattr(cfg, "qk_activation", "silu"),
        "qk_norm": lambda cfg: getattr(cfg, "qk_norm", "l2"),
        "value_l2_norm": lambda cfg: getattr(cfg, "mean_delta_value_l2_norm", True),
        "value_norm_eps": lambda cfg: getattr(cfg, "mean_delta_value_norm_eps", 1e-6),
        "mean_recompute_chunk_size": lambda cfg: getattr(cfg, "mean_recompute_chunk_size", 128),
    },
    "pdf_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_v_heads": lambda cfg: (
            getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            if getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            >= cfg.num_attention_heads
            else cfg.num_attention_heads
        ),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
    },
    "first_order_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_v_heads": lambda cfg: (
            getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            if getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            >= cfg.num_attention_heads
            else cfg.num_attention_heads
        ),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
    },
    "pdf_final_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "pdf_final_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "hist_eps": lambda cfg: getattr(cfg, "pdf_final_hist_eps", 1e-4),
        "score_clip": lambda cfg: getattr(cfg, "pdf_final_score_clip", 20.0),
        "use_triton": lambda cfg: getattr(cfg, "pdf_final_use_triton", True),
    },
    "pdf_refined_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "pdf_final_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "hist_eps": lambda cfg: getattr(cfg, "pdf_final_hist_eps", 1e-4),
        "score_clip": lambda cfg: getattr(cfg, "pdf_final_score_clip", 20.0),
        "use_triton": lambda cfg: getattr(cfg, "pdf_final_use_triton", True),
    },
    "taylor_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "mode": lambda cfg: getattr(cfg, "linear_attn_mode", "fused_recurrent"),
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "denom_eps": lambda cfg: getattr(cfg, "taylor_denom_eps", 1e-4),
        "output_norm": lambda cfg: getattr(cfg, "taylor_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
    },
    "softmax_taylor_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "mode": lambda cfg: getattr(cfg, "linear_attn_mode", "fused_recurrent"),
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "denom_eps": lambda cfg: getattr(cfg, "taylor_denom_eps", 1e-4),
        "output_norm": lambda cfg: getattr(cfg, "taylor_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
    },
    "approxnet_v2_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "approxnet_v2_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "beta_denom_eps": lambda cfg: getattr(cfg, "approxnet_v2_beta_denom_eps", 1e-6),
        "score_clip": lambda cfg: getattr(cfg, "approxnet_v2_score_clip", 20.0),
        "qk_l2_norm": lambda cfg: getattr(cfg, "approxnet_v2_qk_l2_norm", False),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "approxnet_v2_qk_l2_norm_eps", 1e-6),
        "use_triton": lambda cfg: getattr(cfg, "approxnet_v2_use_triton", True),
        "recompute_chunk_size": lambda cfg: getattr(cfg, "approxnet_v2_recompute_chunk_size", 128),
        "use_sigmoid_gate": lambda cfg: getattr(cfg, "approxnet_v2_use_sigmoid_gate", False),
    },
    "approxnet_v2": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "approxnet_v2_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "beta_denom_eps": lambda cfg: getattr(cfg, "approxnet_v2_beta_denom_eps", 1e-6),
        "score_clip": lambda cfg: getattr(cfg, "approxnet_v2_score_clip", 20.0),
        "qk_l2_norm": lambda cfg: getattr(cfg, "approxnet_v2_qk_l2_norm", False),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "approxnet_v2_qk_l2_norm_eps", 1e-6),
        "use_triton": lambda cfg: getattr(cfg, "approxnet_v2_use_triton", True),
        "recompute_chunk_size": lambda cfg: getattr(cfg, "approxnet_v2_recompute_chunk_size", 128),
        "use_sigmoid_gate": lambda cfg: getattr(cfg, "approxnet_v2_use_sigmoid_gate", False),
    },
    "approxnet_v3_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "approxnet_v3_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "beta_denom_eps": lambda cfg: getattr(cfg, "approxnet_v3_beta_denom_eps", 1e-6),
        "score_clip": lambda cfg: getattr(cfg, "approxnet_v3_score_clip", 20.0),
        "qk_l2_norm": lambda cfg: getattr(cfg, "approxnet_v3_qk_l2_norm", False),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "approxnet_v3_qk_l2_norm_eps", 1e-6),
        "use_triton": lambda cfg: getattr(cfg, "approxnet_v3_use_triton", True),
        "recompute_chunk_size": lambda cfg: getattr(cfg, "approxnet_v3_recompute_chunk_size", 128),
        "use_sigmoid_gate": lambda cfg: getattr(cfg, "approxnet_v3_use_sigmoid_gate", False),
    },
    "approxnet_v3": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "approxnet_v3_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "beta_denom_eps": lambda cfg: getattr(cfg, "approxnet_v3_beta_denom_eps", 1e-6),
        "score_clip": lambda cfg: getattr(cfg, "approxnet_v3_score_clip", 20.0),
        "qk_l2_norm": lambda cfg: getattr(cfg, "approxnet_v3_qk_l2_norm", False),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "approxnet_v3_qk_l2_norm_eps", 1e-6),
        "use_triton": lambda cfg: getattr(cfg, "approxnet_v3_use_triton", True),
        "recompute_chunk_size": lambda cfg: getattr(cfg, "approxnet_v3_recompute_chunk_size", 128),
        "use_sigmoid_gate": lambda cfg: getattr(cfg, "approxnet_v3_use_sigmoid_gate", False),
    },
    "approxnet_v4_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "approxnet_v4_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "z_score_eps": lambda cfg: getattr(cfg, "approxnet_v4_z_score_eps", 1.0),
        "gate_alpha_init": lambda cfg: getattr(cfg, "approxnet_v4_gate_alpha_init", 1.0),
        "gate_bias_init": lambda cfg: getattr(cfg, "approxnet_v4_gate_bias_init", 0.0),
        "qk_l2_norm": lambda cfg: getattr(cfg, "approxnet_v4_qk_l2_norm", False),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "approxnet_v4_qk_l2_norm_eps", 1e-6),
        "use_triton": lambda cfg: getattr(cfg, "approxnet_v4_use_triton", True),
        "recompute_chunk_size": lambda cfg: getattr(cfg, "approxnet_v4_recompute_chunk_size", 128),
    },
    "approxnet_v4": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "approxnet_v4_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "z_score_eps": lambda cfg: getattr(cfg, "approxnet_v4_z_score_eps", 1.0),
        "gate_alpha_init": lambda cfg: getattr(cfg, "approxnet_v4_gate_alpha_init", 1.0),
        "gate_bias_init": lambda cfg: getattr(cfg, "approxnet_v4_gate_bias_init", 0.0),
        "qk_l2_norm": lambda cfg: getattr(cfg, "approxnet_v4_qk_l2_norm", False),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "approxnet_v4_qk_l2_norm_eps", 1e-6),
        "use_triton": lambda cfg: getattr(cfg, "approxnet_v4_use_triton", True),
        "recompute_chunk_size": lambda cfg: getattr(cfg, "approxnet_v4_recompute_chunk_size", 128),
    },
    "soam_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "soam_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "d_r": lambda cfg: getattr(cfg, "soam_d_r", 16),
        "decay_alpha_init": lambda cfg: getattr(cfg, "soam_decay_alpha_init", 0.0),
        "decay_bias_init": lambda cfg: getattr(cfg, "soam_decay_bias_init", 2.0),
        "write_alpha_init": lambda cfg: getattr(cfg, "soam_write_alpha_init", 1.0),
        "write_bias_init": lambda cfg: getattr(cfg, "soam_write_bias_init", 0.0),
        "qk_l2_norm": lambda cfg: getattr(cfg, "soam_qk_l2_norm", True),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "soam_qk_l2_norm_eps", 1e-6),
    },
    "soam": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "soam_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "d_r": lambda cfg: getattr(cfg, "soam_d_r", 16),
        "decay_alpha_init": lambda cfg: getattr(cfg, "soam_decay_alpha_init", 0.0),
        "decay_bias_init": lambda cfg: getattr(cfg, "soam_decay_bias_init", 2.0),
        "write_alpha_init": lambda cfg: getattr(cfg, "soam_write_alpha_init", 1.0),
        "write_bias_init": lambda cfg: getattr(cfg, "soam_write_bias_init", 0.0),
        "qk_l2_norm": lambda cfg: getattr(cfg, "soam_qk_l2_norm", True),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "soam_qk_l2_norm_eps", 1e-6),
    },
    "wla_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "wla_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "d_r": lambda cfg: getattr(cfg, "wla_d_r", 16),
        "decay_alpha_init": lambda cfg: getattr(cfg, "wla_decay_alpha_init", 0.0),
        "decay_bias_init": lambda cfg: getattr(cfg, "wla_decay_bias_init", 2.0),
        "write_alpha_init": lambda cfg: getattr(cfg, "wla_write_alpha_init", 1.0),
        "write_bias_init": lambda cfg: getattr(cfg, "wla_write_bias_init", 0.0),
        "qk_l2_norm": lambda cfg: getattr(cfg, "wla_qk_l2_norm", True),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "wla_qk_l2_norm_eps", 1e-6),
        "sigma2_init": lambda cfg: getattr(cfg, "wla_sigma2_init", 5.0),
    },
    "wla": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "wla_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "d_r": lambda cfg: getattr(cfg, "wla_d_r", 16),
        "decay_alpha_init": lambda cfg: getattr(cfg, "wla_decay_alpha_init", 0.0),
        "decay_bias_init": lambda cfg: getattr(cfg, "wla_decay_bias_init", 2.0),
        "write_alpha_init": lambda cfg: getattr(cfg, "wla_write_alpha_init", 1.0),
        "write_bias_init": lambda cfg: getattr(cfg, "wla_write_bias_init", 0.0),
        "qk_l2_norm": lambda cfg: getattr(cfg, "wla_qk_l2_norm", True),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "wla_qk_l2_norm_eps", 1e-6),
        "sigma2_init": lambda cfg: getattr(cfg, "wla_sigma2_init", 5.0),
    },
    "sisa_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "sisa_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "d_r": lambda cfg: getattr(cfg, "sisa_d_r", 16),
        "decay_alpha_init": lambda cfg: getattr(cfg, "sisa_decay_alpha_init", 0.0),
        "decay_bias_init": lambda cfg: getattr(cfg, "sisa_decay_bias_init", 2.0),
        "write_alpha_init": lambda cfg: getattr(cfg, "sisa_write_alpha_init", 1.0),
        "write_bias_init": lambda cfg: getattr(cfg, "sisa_write_bias_init", 0.0),
        "qk_l2_norm": lambda cfg: getattr(cfg, "sisa_qk_l2_norm", True),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "sisa_qk_l2_norm_eps", 1e-6),
        "beta_init": lambda cfg: getattr(cfg, "sisa_beta_init", 1.0),
    },
    "sisa": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "sisa_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "d_r": lambda cfg: getattr(cfg, "sisa_d_r", 16),
        "decay_alpha_init": lambda cfg: getattr(cfg, "sisa_decay_alpha_init", 0.0),
        "decay_bias_init": lambda cfg: getattr(cfg, "sisa_decay_bias_init", 2.0),
        "write_alpha_init": lambda cfg: getattr(cfg, "sisa_write_alpha_init", 1.0),
        "write_bias_init": lambda cfg: getattr(cfg, "sisa_write_bias_init", 0.0),
        "qk_l2_norm": lambda cfg: getattr(cfg, "sisa_qk_l2_norm", True),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "sisa_qk_l2_norm_eps", 1e-6),
        "beta_init": lambda cfg: getattr(cfg, "sisa_beta_init", 1.0),
    },
    "original_linear_attention": {
        "mode": lambda cfg: getattr(cfg, "linear_attn_mode", "fused_recurrent"),
        "hidden_size": "hidden_size",
        "expand_k": lambda cfg: getattr(cfg, "expand_k", 1.0),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "num_heads": "num_attention_heads",
        "num_kv_heads": lambda cfg: getattr(cfg, "num_attention_heads"),
        "feature_map": lambda cfg: "identity",
        "tie_feature_map_qk": lambda cfg: True,
        "output_norm": lambda cfg: getattr(cfg, "linear_attn_output_norm", "rmsnorm"),
        "norm_q": lambda cfg: False,
        "norm_k": lambda cfg: False,
        "do_feature_map_norm": lambda cfg: False,
        "elementwise_affine": lambda cfg: True,
        "norm_eps": "rms_norm_eps",
    },
    "vanilla_linear_attention": {
        "mode": lambda cfg: getattr(cfg, "linear_attn_mode", "fused_recurrent"),
        "hidden_size": "hidden_size",
        "expand_k": lambda cfg: getattr(cfg, "expand_k", 1.0),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "num_heads": "num_attention_heads",
        "num_kv_heads": lambda cfg: getattr(cfg, "num_attention_heads"),
        "feature_map": lambda cfg: "identity",
        "tie_feature_map_qk": lambda cfg: True,
        "output_norm": lambda cfg: getattr(cfg, "linear_attn_output_norm", "rmsnorm"),
        "norm_q": lambda cfg: False,
        "norm_k": lambda cfg: False,
        "do_feature_map_norm": lambda cfg: False,
        "elementwise_affine": lambda cfg: True,
        "norm_eps": "rms_norm_eps",
    },
    "mha_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "num_kv_heads": lambda cfg: getattr(cfg, "num_attention_heads"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "qk_norm": lambda cfg: getattr(cfg, "qk_norm", False),
        "window_size": lambda cfg: getattr(cfg, "window_size", None),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
    },
    "mha": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "num_kv_heads": lambda cfg: getattr(cfg, "num_attention_heads"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "qk_norm": lambda cfg: getattr(cfg, "qk_norm", False),
        "window_size": lambda cfg: getattr(cfg, "window_size", None),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
    },
    "mha_torch_qk_unit_norm_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "num_kv_heads": lambda cfg: getattr(cfg, "num_attention_heads"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "qk_norm": lambda cfg: False,
        "qk_unit_norm": lambda cfg: True,
        "qk_unit_norm_eps": lambda cfg: getattr(cfg, "qk_unit_norm_eps", 1e-6),
        "window_size": lambda cfg: getattr(cfg, "window_size", None),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
    },
    "mha_torch_qk_unit_norm": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "num_kv_heads": lambda cfg: getattr(cfg, "num_attention_heads"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "qk_norm": lambda cfg: False,
        "qk_unit_norm": lambda cfg: True,
        "qk_unit_norm_eps": lambda cfg: getattr(cfg, "qk_unit_norm_eps", 1e-6),
        "window_size": lambda cfg: getattr(cfg, "window_size", None),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
    },
    "performer_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_v_heads": lambda cfg: (
            getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            if getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            >= cfg.num_attention_heads
            else cfg.num_attention_heads
        ),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "performer_nb_features": lambda cfg: getattr(cfg, "performer_nb_features", None),
        "performer_feature_eps": lambda cfg: getattr(cfg, "performer_feature_eps", 1e-4),
        "performer_ortho_scaling": lambda cfg: getattr(cfg, "performer_ortho_scaling", 0),
        "performer_redraw_projection": lambda cfg: getattr(cfg, "performer_redraw_projection", False),
        "performer_projection_seed": lambda cfg: getattr(cfg, "performer_projection_seed", 0),
        "performer_adaptive_center_sampling": lambda cfg: getattr(cfg, "performer_adaptive_center_sampling", False),
        "performer_adaptive_center_momentum": lambda cfg: getattr(cfg, "performer_adaptive_center_momentum", 0.9),
        "performer_adaptive_center_log_clip": lambda cfg: getattr(cfg, "performer_adaptive_center_log_clip", 12.0),
        "performer_use_triton": lambda cfg: getattr(cfg, "performer_use_triton", True),
        "performer_qmc_gaussian_sampling": lambda cfg: getattr(cfg, "performer_qmc_gaussian_sampling", False),
        "performer_qmc_scramble": lambda cfg: getattr(cfg, "performer_qmc_scramble", True),
    },
    "performer_plus_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_v_heads": lambda cfg: (
            getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            if getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            >= cfg.num_attention_heads
            else cfg.num_attention_heads
        ),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "performer_nb_features": lambda cfg: getattr(cfg, "performer_nb_features", None),
        "performer_feature_eps": lambda cfg: getattr(cfg, "performer_feature_eps", 1e-4),
        "performer_ortho_scaling": lambda cfg: getattr(cfg, "performer_ortho_scaling", 0),
        "performer_redraw_projection": lambda cfg: getattr(cfg, "performer_redraw_projection", False),
        "performer_projection_seed": lambda cfg: getattr(cfg, "performer_projection_seed", 0),
        "performer_adaptive_center_sampling": lambda cfg: getattr(cfg, "performer_adaptive_center_sampling", False),
        "performer_adaptive_center_momentum": lambda cfg: getattr(cfg, "performer_adaptive_center_momentum", 0.9),
        "performer_adaptive_center_log_clip": lambda cfg: getattr(cfg, "performer_adaptive_center_log_clip", 12.0),
        "performer_use_dual_precondition_sampling": lambda cfg: getattr(cfg, "performer_use_dual_precondition_sampling", False),
        "performer_precondition_momentum": lambda cfg: getattr(cfg, "performer_precondition_momentum", 0.9),
        "performer_precondition_eps": lambda cfg: getattr(cfg, "performer_precondition_eps", 1e-4),
        "performer_precondition_log_clip": lambda cfg: getattr(cfg, "performer_precondition_log_clip", 2.0),
        "performer_precondition_mode": lambda cfg: getattr(cfg, "performer_precondition_mode", "diag"),
        "performer_use_triton": lambda cfg: getattr(cfg, "performer_use_triton", True),
        "performer_antithetic_features": lambda cfg: getattr(cfg, "performer_antithetic_features", True),
        "performer_stratified_norm_sampling": lambda cfg: getattr(cfg, "performer_stratified_norm_sampling", False),
        "performer_stratified_jitter": lambda cfg: getattr(cfg, "performer_stratified_jitter", True),
        "performer_qmc_gaussian_sampling": lambda cfg: getattr(cfg, "performer_qmc_gaussian_sampling", False),
        "performer_qmc_scramble": lambda cfg: getattr(cfg, "performer_qmc_scramble", True),
        "performer_use_landmark_sampling": lambda cfg: getattr(cfg, "performer_use_landmark_sampling", False),
        "performer_landmark_ratio": lambda cfg: getattr(cfg, "performer_landmark_ratio", 0.25),
        "performer_landmark_alpha_init": lambda cfg: getattr(cfg, "performer_landmark_alpha_init", 0.35),
        "performer_landmark_in_denominator": lambda cfg: getattr(cfg, "performer_landmark_in_denominator", True),
        "performer_use_projection_ensemble": lambda cfg: getattr(cfg, "performer_use_projection_ensemble", False),
        "performer_projection_ensemble_groups": lambda cfg: getattr(cfg, "performer_projection_ensemble_groups", 2),
        "performer_use_deterministic_nodes": lambda cfg: getattr(cfg, "performer_use_deterministic_nodes", False),
        "performer_deterministic_ratio": lambda cfg: getattr(cfg, "performer_deterministic_ratio", 0.25),
        "performer_qk_l2_norm": lambda cfg: getattr(cfg, "performer_qk_l2_norm", True),
        "performer_use_beta": lambda cfg: getattr(cfg, "performer_use_beta", False),
        "performer_beta_init": lambda cfg: getattr(cfg, "performer_beta_init", 1.0),
        "performer_use_value_gate": lambda cfg: getattr(cfg, "performer_use_value_gate", False),
        "performer_value_gate_init": lambda cfg: getattr(cfg, "performer_value_gate_init", 0.0),
        "performer_use_output_gate": lambda cfg: getattr(cfg, "performer_use_output_gate", True),
        "performer_output_gate_init": lambda cfg: getattr(cfg, "performer_output_gate_init", -1.0),
        "performer_use_decay": lambda cfg: getattr(cfg, "performer_use_decay", True),
        "performer_decay_init": lambda cfg: getattr(cfg, "performer_decay_init", 1.0),
        "performer_state_update": lambda cfg: getattr(cfg, "performer_state_update", "sum"),
        "performer_delta_beta_norm": lambda cfg: getattr(cfg, "performer_delta_beta_norm", True),
        "performer_delta_beta_norm_eps": lambda cfg: getattr(cfg, "performer_delta_beta_norm_eps", 1e-3),
        "performer_delta_denom_eps": lambda cfg: getattr(cfg, "performer_delta_denom_eps", 1e-3),
        "performer_delta_smooth_denom": lambda cfg: getattr(cfg, "performer_delta_smooth_denom", False),
        "performer_delta_denom_tau": lambda cfg: getattr(cfg, "performer_delta_denom_tau", 1e-2),
        "performer_delta_beta_cap": lambda cfg: getattr(cfg, "performer_delta_beta_cap", 0.25),
        "performer_delta_safe_denom_floor": lambda cfg: getattr(cfg, "performer_delta_safe_denom_floor", 1e-3),
        "performer_delta_decouple_beta": lambda cfg: getattr(cfg, "performer_delta_decouple_beta", False),
        "performer_delta_denominator_update": lambda cfg: getattr(cfg, "performer_delta_denominator_update", "delta"),
        "performer_delta_denominator_map": lambda cfg: getattr(cfg, "performer_delta_denominator_map", "auto"),
        "performer_delta_denominator_stopgrad": lambda cfg: getattr(cfg, "performer_delta_denominator_stopgrad", False),
        "performer_delta_use_leaky_dplr": lambda cfg: getattr(cfg, "performer_delta_use_leaky_dplr", False),
        "performer_delta_leaky_rho_init": lambda cfg: getattr(cfg, "performer_delta_leaky_rho_init", 1.0),
        "performer_delta_leaky_min_lambda": lambda cfg: getattr(cfg, "performer_delta_leaky_min_lambda", 0.5),
        "performer_use_hybrid_numerator": lambda cfg: getattr(cfg, "performer_use_hybrid_numerator", False),
        "performer_hybrid_num_ratio": lambda cfg: getattr(cfg, "performer_hybrid_num_ratio", 0.25),
        "performer_hybrid_alpha_init": lambda cfg: getattr(cfg, "performer_hybrid_alpha_init", 0.75),
        "performer_use_cv_residual_shrinkage": lambda cfg: getattr(cfg, "performer_use_cv_residual_shrinkage", False),
        "performer_cv_residual_init": lambda cfg: getattr(cfg, "performer_cv_residual_init", 0.75),
        "performer_cv_finite_sample_orthogonalize": lambda cfg: getattr(cfg, "performer_cv_finite_sample_orthogonalize", False),
        "performer_cv_finite_sample_orth_eps": lambda cfg: getattr(cfg, "performer_cv_finite_sample_orth_eps", 1e-4),
        "performer_use_jackknife_debias": lambda cfg: getattr(cfg, "performer_use_jackknife_debias", False),
        "performer_jackknife_groups": lambda cfg: getattr(cfg, "performer_jackknife_groups", 2),
        "performer_jackknife_min_per_group": lambda cfg: getattr(cfg, "performer_jackknife_min_per_group", 8),
        "performer_use_jackknife_adaptive_shrinkage": lambda cfg: getattr(cfg, "performer_use_jackknife_adaptive_shrinkage", False),
        "performer_jackknife_shrinkage_eps": lambda cfg: getattr(cfg, "performer_jackknife_shrinkage_eps", 1e-5),
        "performer_use_second_order_cv": lambda cfg: getattr(cfg, "performer_use_second_order_cv", False),
        "performer_use_cv_decoupled_second_order": lambda cfg: getattr(cfg, "performer_use_cv_decoupled_second_order", False),
        "performer_cv_decoupled_h2_ratio": lambda cfg: getattr(cfg, "performer_cv_decoupled_h2_ratio", 0.25),
        "performer_cv_decoupled_h2_deterministic": lambda cfg: getattr(cfg, "performer_cv_decoupled_h2_deterministic", False),
        "performer_use_cv_decoupled_adaptive_h2_ratio": lambda cfg: getattr(cfg, "performer_use_cv_decoupled_adaptive_h2_ratio", False),
        "performer_cv_decoupled_ratio_ema_momentum": lambda cfg: getattr(cfg, "performer_cv_decoupled_ratio_ema_momentum", 0.9),
        "performer_cv_decoupled_ratio_min": lambda cfg: getattr(cfg, "performer_cv_decoupled_ratio_min", 0.05),
        "performer_cv_decoupled_ratio_max": lambda cfg: getattr(cfg, "performer_cv_decoupled_ratio_max", 0.5),
        "performer_use_third_order_cv": lambda cfg: getattr(cfg, "performer_use_third_order_cv", False),
        "performer_use_diag2_term": lambda cfg: getattr(cfg, "performer_use_diag2_term", False),
        "performer_diag2_ratio": lambda cfg: getattr(cfg, "performer_diag2_ratio", 0.25),
        "performer_diag2_alpha_init": lambda cfg: getattr(cfg, "performer_diag2_alpha_init", 0.1),
        "performer_use_dual_map": lambda cfg: getattr(cfg, "performer_use_dual_map", True),
        "performer_dual_map_den_ratio": lambda cfg: getattr(cfg, "performer_dual_map_den_ratio", 0.25),
        "performer_dual_map_row_selection": lambda cfg: getattr(cfg, "performer_dual_map_row_selection", "auto"),
        "performer_use_layerwise_den_ratio": lambda cfg: getattr(cfg, "performer_use_layerwise_den_ratio", False),
        "performer_layerwise_den_ratio_tau": lambda cfg: getattr(cfg, "performer_layerwise_den_ratio_tau", 8.0),
        "performer_use_error_feedback_den_ratio": lambda cfg: getattr(cfg, "performer_use_error_feedback_den_ratio", False),
        "performer_error_feedback_den_ratio_momentum": lambda cfg: getattr(cfg, "performer_error_feedback_den_ratio_momentum", 0.9),
        "performer_error_feedback_den_ratio_gain": lambda cfg: getattr(cfg, "performer_error_feedback_den_ratio_gain", 0.5),
        "performer_use_adaptive_den_mix": lambda cfg: getattr(cfg, "performer_use_adaptive_den_mix", False),
        "performer_adaptive_den_mix_init": lambda cfg: getattr(cfg, "performer_adaptive_den_mix_init", 0.0),
        "performer_dual_map_low_precision": lambda cfg: getattr(cfg, "performer_dual_map_low_precision", True),
        "performer_use_den_poly_kernel": lambda cfg: getattr(cfg, "performer_use_den_poly_kernel", False),
        "performer_den_poly_alpha_init": lambda cfg: getattr(cfg, "performer_den_poly_alpha_init", 0.15),
        "performer_den_poly_constant": lambda cfg: getattr(cfg, "performer_den_poly_constant", 2.0),
        "performer_den_poly_ratio": lambda cfg: getattr(cfg, "performer_den_poly_ratio", 0.5),
        "performer_per_layer_projection": lambda cfg: getattr(cfg, "performer_per_layer_projection", True),
        "performer_learnable_projection": lambda cfg: getattr(cfg, "performer_learnable_projection", False),
        "performer_learnable_projection_scale": lambda cfg: getattr(cfg, "performer_learnable_projection_scale", False),
        "performer_use_control_variate": lambda cfg: getattr(cfg, "performer_use_control_variate", True),
        "performer_use_adaptive_linear_cv": lambda cfg: getattr(cfg, "performer_use_adaptive_linear_cv", False),
        "performer_adaptive_linear_cv_init": lambda cfg: getattr(cfg, "performer_adaptive_linear_cv_init", 1.0),
        "performer_adaptive_linear_cv_eps": lambda cfg: getattr(cfg, "performer_adaptive_linear_cv_eps", 1e-4),
        "performer_feature_rms_norm": lambda cfg: getattr(cfg, "performer_feature_rms_norm", False),
        "performer_feature_rms_norm_eps": lambda cfg: getattr(cfg, "performer_feature_rms_norm_eps", 1e-4),
        "performer_feature_pairwise_balance": lambda cfg: getattr(cfg, "performer_feature_pairwise_balance", False),
        "performer_feature_pairwise_balance_eps": lambda cfg: getattr(cfg, "performer_feature_pairwise_balance_eps", 1e-4),
        "performer_feature_pairwise_balance_log_clip": lambda cfg: getattr(cfg, "performer_feature_pairwise_balance_log_clip", 2.0),
        "performer_control_variate_exp_clip": lambda cfg: getattr(cfg, "performer_control_variate_exp_clip", 20.0),
        "performer_enable_error_observability": lambda cfg: getattr(cfg, "performer_enable_error_observability", True),
        "performer_error_observe_interval": lambda cfg: getattr(cfg, "performer_error_observe_interval", 10),
        "performer_error_observe_max_tokens": lambda cfg: getattr(cfg, "performer_error_observe_max_tokens", 32),
        "performer_error_observe_max_heads": lambda cfg: getattr(cfg, "performer_error_observe_max_heads", 2),
        "performer_cv_split_feature_budget": lambda cfg: getattr(cfg, "performer_cv_split_feature_budget", True),
        "performer_dim_aware_kernel_scale": lambda cfg: getattr(cfg, "performer_dim_aware_kernel_scale", True),
        "performer_learnable_kernel_scale": lambda cfg: getattr(cfg, "performer_learnable_kernel_scale", True),
        "performer_kernel_scale_init": lambda cfg: getattr(cfg, "performer_kernel_scale_init", 1.0),
    },
    "hpk_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "hpk_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "feature_dim": lambda cfg: getattr(cfg, "hpk_feature_dim", 64),
        "power_order": lambda cfg: getattr(cfg, "hpk_power_order", 2),
        "decay_init": lambda cfg: getattr(cfg, "hpk_decay_init", 4.0),
        "denom_eps": lambda cfg: getattr(cfg, "hpk_denom_eps", 1e-6),
        "qk_l2_norm": lambda cfg: getattr(cfg, "hpk_qk_l2_norm", True),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "hpk_qk_l2_norm_eps", 1e-6),
    },
    "hpk": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "hpk_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "feature_dim": lambda cfg: getattr(cfg, "hpk_feature_dim", 64),
        "power_order": lambda cfg: getattr(cfg, "hpk_power_order", 2),
        "decay_init": lambda cfg: getattr(cfg, "hpk_decay_init", 4.0),
        "denom_eps": lambda cfg: getattr(cfg, "hpk_denom_eps", 1e-6),
        "qk_l2_norm": lambda cfg: getattr(cfg, "hpk_qk_l2_norm", True),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "hpk_qk_l2_norm_eps", 1e-6),
    },
    "sqk_linear_attention": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "sqk_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "proj_rank": lambda cfg: getattr(cfg, "sqk_proj_rank", 16),
        "decay_init": lambda cfg: getattr(cfg, "sqk_decay_init", 4.0),
        "denom_eps": lambda cfg: getattr(cfg, "sqk_denom_eps", 1e-6),
        "qk_l2_norm": lambda cfg: getattr(cfg, "sqk_qk_l2_norm", True),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "sqk_qk_l2_norm_eps", 1e-6),
    },
    "sqk": {
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "sqk_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "proj_rank": lambda cfg: getattr(cfg, "sqk_proj_rank", 16),
        "decay_init": lambda cfg: getattr(cfg, "sqk_decay_init", 4.0),
        "denom_eps": lambda cfg: getattr(cfg, "sqk_denom_eps", 1e-6),
        "qk_l2_norm": lambda cfg: getattr(cfg, "sqk_qk_l2_norm", True),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "sqk_qk_l2_norm_eps", 1e-6),
    },
    "css_linear_attention": {
        "mode": lambda cfg: getattr(cfg, "css_mode", "fused_chunk"),
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "css_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "proj_rank": lambda cfg: getattr(cfg, "css_proj_rank", 22),
        "decay_init": lambda cfg: getattr(cfg, "css_decay_init", 4.0),
        "denom_eps": lambda cfg: getattr(cfg, "css_denom_eps", 1e-6),
        "proj_normalize": lambda cfg: getattr(cfg, "css_proj_normalize", True),
        "proj_norm_eps": lambda cfg: getattr(cfg, "css_proj_norm_eps", 1e-6),
        "qk_l2_norm": lambda cfg: getattr(cfg, "css_qk_l2_norm", True),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "css_qk_l2_norm_eps", 1e-6),
    },
    "css": {
        "mode": lambda cfg: getattr(cfg, "css_mode", "fused_chunk"),
        "hidden_size": "hidden_size",
        "num_heads": "num_attention_heads",
        "norm_eps": "rms_norm_eps",
        "head_dim": lambda cfg: getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        ),
        "num_kv_heads": lambda cfg: getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "expand_v": lambda cfg: getattr(cfg, "expand_v", 1.0),
        "use_short_conv": lambda cfg: getattr(cfg, "use_short_conv", True),
        "conv_size": lambda cfg: getattr(cfg, "conv_size", 4),
        "conv_bias": lambda cfg: getattr(cfg, "conv_bias", False),
        "output_norm": lambda cfg: getattr(cfg, "css_output_norm", "identity"),
        "qkv_bias": lambda cfg: getattr(cfg, "attention_bias", False),
        "rope_theta": lambda cfg: getattr(cfg, "rope_theta", 10000.0),
        "max_position_embeddings": lambda cfg: getattr(cfg, "max_position_embeddings", None),
        "proj_rank": lambda cfg: getattr(cfg, "css_proj_rank", 22),
        "decay_init": lambda cfg: getattr(cfg, "css_decay_init", 4.0),
        "denom_eps": lambda cfg: getattr(cfg, "css_denom_eps", 1e-6),
        "proj_normalize": lambda cfg: getattr(cfg, "css_proj_normalize", True),
        "proj_norm_eps": lambda cfg: getattr(cfg, "css_proj_norm_eps", 1e-6),
        "qk_l2_norm": lambda cfg: getattr(cfg, "css_qk_l2_norm", True),
        "qk_l2_norm_eps": lambda cfg: getattr(cfg, "css_qk_l2_norm_eps", 1e-6),
    },
}


def config_to_kwargs(config, mapping: dict) -> dict:
    out = {}
    for init_kw, src in mapping.items():
        if callable(src):
            out[init_kw] = src(config)
        else:
            if hasattr(config, src):
                out[init_kw] = getattr(config, src)
    return out


def _attention_has_norm(attn_module: nn.Module | None) -> bool:
    if attn_module is None:
        return False
    for attr in ("norm", "o_norm", "q_norm", "k_norm"):
        mod = getattr(attn_module, attr, None)
        if isinstance(mod, nn.Module):
            return True
    for name, mod in attn_module.named_modules():
        if not name:
            continue
        if "norm" in mod.__class__.__name__.lower():
            return True
    return False


def _copy_linear_if_compatible(dst_module: nn.Module, src_module: nn.Module, name: str) -> None:
    dst = getattr(dst_module, name, None)
    src = getattr(src_module, name, None)
    if not isinstance(dst, nn.Linear) or not isinstance(src, nn.Linear):
        return
    if dst.weight.shape == src.weight.shape:
        dst.weight.data.copy_(src.weight.data.to(device=dst.weight.device, dtype=dst.weight.dtype))
    if dst.bias is not None and src.bias is not None and dst.bias.shape == src.bias.shape:
        dst.bias.data.copy_(src.bias.data.to(device=dst.bias.device, dtype=dst.bias.dtype))


def _infer_module_device_dtype(module: nn.Module | None) -> tuple[torch.device | None, torch.dtype | None]:
    if module is None:
        return None, None
    try:
        param = next(module.parameters())
        return param.device, param.dtype
    except StopIteration:
        return None, None


def _initialize_from_source(
    attn_type: str,
    module: nn.Module,
    source_attn: nn.Module | None,
) -> None:
    if source_attn is None:
        return
    if attn_type in {
        "taylor_linear_attention",
        "softmax_taylor_linear_attention",
        "pdf_final_linear_attention",
        "pdf_refined_linear_attention",
        "approxnet_v2_linear_attention",
        "approxnet_v2",
        "approxnet_v3_linear_attention",
        "approxnet_v3",
        "approxnet_v4_linear_attention",
        "approxnet_v4",
        "soam_linear_attention",
        "soam",
        "wla_linear_attention",
        "wla",
        "sisa_linear_attention",
        "sisa",
        "hpk_linear_attention",
        "hpk",
        "sqk_linear_attention",
        "sqk",
        "css_linear_attention",
        "css",
    }:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _copy_linear_if_compatible(module, source_attn, name)


def init_attention_module(config, layer_idx: int, source_attn: nn.Module | None = None):
    attn_type = config.linear_attention_type
    if attn_type in {"mha_attention", "mha"}:
        if not _FLA_ATTENTION_AVAILABLE:
            raise ImportError(
                "linear_attention_type='mha_attention' requires FLA official Attention "
                "(flash-attn dependency). Install flash-attn or use "
                "linear_attention_type='mha_torch_attention'."
            )
        attn_class = FLAAttention
        attn_kwargs = config_to_kwargs(config, CONFIG2KWARGS["mha_attention"])
    else:
        attn_class = TYPE2ATTN[attn_type]
        attn_kwargs = config_to_kwargs(config, CONFIG2KWARGS[attn_type])
        if attn_type in {
            "original_linear_attention",
            "vanilla_linear_attention",
            "pdf_final_linear_attention",
            "pdf_refined_linear_attention",
            "approxnet_v2_linear_attention",
            "approxnet_v2",
            "approxnet_v3_linear_attention",
            "approxnet_v3",
            "approxnet_v4_linear_attention",
            "approxnet_v4",
            "soam_linear_attention",
            "soam",
            "wla_linear_attention",
            "wla",
            "sisa_linear_attention",
            "sisa",
            "hpk_linear_attention",
            "hpk",
            "sqk_linear_attention",
            "sqk",
            "css_linear_attention",
            "css",
        }:
            keep_norm = _attention_has_norm(source_attn)
            attn_kwargs["output_norm"] = "rmsnorm" if keep_norm else "identity"

    class LinearAttn(attn_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, *args, **kwargs):
            attention_mask = kwargs.get("attention_mask", None)
            if attention_mask is not None and attention_mask.dim() == 4:
                # HF eager/sdpa can pass a 4D causal mask [B, 1, Q, K]. The FLA
                # layers expect a 2D padding mask [B, K] (True for valid tokens).
                attention_mask_2d = attention_mask[:, 0, -1, :]
                if attention_mask_2d.dtype != torch.bool:
                    # For eager masks, valid positions are 0 and masked are -inf.
                    attention_mask_2d = attention_mask_2d.eq(0)
                kwargs = {**kwargs}
                kwargs.update({"attention_mask": attention_mask_2d})

            # HF LlamaDecoderLayer passes `past_key_value` (singular), while the
            # custom linear-attention modules in this repo consume
            # `past_key_values` (plural).
            if "past_key_value" in kwargs and "past_key_values" not in kwargs:
                kwargs = {**kwargs}
                kwargs["past_key_values"] = kwargs.pop("past_key_value")

            outputs = super().forward(*args, **kwargs)
            if isinstance(outputs, torch.Tensor):
                return outputs, None
            if isinstance(outputs, (tuple, list)):
                if len(outputs) >= 2:
                    return outputs[0], outputs[1]
                if len(outputs) == 1:
                    return outputs[0], None
            raise TypeError(
                f"Unexpected output type from attention module: {type(outputs)}",
            )

    module = LinearAttn(**attn_kwargs, layer_idx=layer_idx)
    source_device, source_dtype = _infer_module_device_dtype(source_attn)
    source_is_meta = source_device is not None and source_device.type == "meta"
    # Align with Transformers meta-init flow: avoid moving freshly created
    # module to meta device and skip source-weight copy in meta stage.
    if source_device is not None and not source_is_meta:
        module = module.to(device=source_device, dtype=source_dtype)
    _initialize_from_source(attn_type, module, None if source_is_meta else source_attn)
    return module
