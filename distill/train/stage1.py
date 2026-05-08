from simple_parsing import ArgumentParser
import math
from contextlib import nullcontext
import torch
import torch.nn as nn
from pathlib import Path
import shutil
import os
import re
from typing import Any, Union, Optional
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

from .args import ModelArguments, DataArguments
from .data import prapare_dataset, DataCollatorWithFlattening

from ..models.utils import init_attention_module

settings = {
    "llama": {
        "auto_map": {
            "AutoConfig": "configuration_llamala.LlamaLAConfig",
            "AutoModel": "modeling_llamala.LlamaLAModel",
            "AutoModelForCausalLM": "modeling_llamala.LlamaLAForCausalLM",
        },
        "architectures": ["LlamaLAForCausalLM"],
        "model_type": "llama_la",
    }
}

remote_code_dirs = {
    "llama": "models/llama",
}

utils_files = {
    "models/utils.py",
    "models/linear_attention_pdf.py",
    "models/linear_attention_pdf_final.py",
    "models/linear_attention_pdf_final_triton.py",
    "models/linear_attention_taylor.py",
    "models/linear_attention_approxnet_v2.py",
    "models/linear_attention_approxnet_v2_triton.py",
    "models/linear_attention_approxnet_v3.py",
    "models/linear_attention_approxnet_v3_triton.py",
    "models/linear_attention_approxnet_v4.py",
    "models/linear_attention_approxnet_v4_triton.py",
    "models/linear_attention_soam.py",
    "models/linear_attention_soam_triton.py",
    "models/linear_attention_wla.py",
    "models/linear_attention_wla_triton.py",
    "models/linear_attention_sisa.py",
    "models/linear_attention_sisa_triton.py",
    "models/linear_attention_performer.py",
    "models/linear_attention_performer_triton.py",
    "models/linear_attention_performer_plus.py",
    "models/linear_attention_performer_plus_triton.py",
    "models/linear_attention_pdf_triton.py",
    "models/linear_attention_pdf_triton_kernels.py",
    "models/dual_delta_rule_naive.py",
    "models/dual_delta_rule.py",
    "models/dual_delta_net.py",
    "models/mean_delta_rule_naive.py",
    "models/mean_delta_rule.py",
    "models/mean_delta_net.py",
}


def _collect_relative_import_files(module_file: Path) -> set[Path]:
    """Collect recursive relative-import files (e.g. from .foo import bar -> foo.py)."""
    pattern_import = re.compile(r"^\s*import\s+\.(\S+)\s*$", flags=re.MULTILINE)
    pattern_from = re.compile(r"^\s*from\s+\.(\S+)\s+import", flags=re.MULTILINE)
    module_file = module_file.resolve()
    module_dir = module_file.parent
    pending = [module_file]
    seen: set[Path] = set()
    while pending:
        cur = pending.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if not cur.exists():
            continue
        text = cur.read_text(encoding="utf-8")
        rels = set(pattern_import.findall(text) + pattern_from.findall(text))
        for rel in rels:
            rel_path = module_dir / f"{rel}.py"
            if rel_path not in seen:
                pending.append(rel_path)
    return seen


def _validate_remote_code_export(save_dir: Path, entry: str = "modeling_llamala.py") -> None:
    entry_file = save_dir / entry
    if not entry_file.exists():
        raise RuntimeError(f"Missing remote-code entry file: {entry_file}")
    needed = _collect_relative_import_files(entry_file)
    missing = [str(p) for p in sorted(needed) if not p.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise RuntimeError(
            "Remote-code export incomplete. Missing recursively imported files:\n"
            f"  - {joined}"
        )

class MSETrainer(Trainer):
    OBSERVE_INTERVAL = 10

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        obs_logs = getattr(self, "_last_observability_logs", None)
        if obs_logs:
            for key, value in obs_logs.items():
                logs.setdefault(key, value)
        return super().log(logs, *args, **kwargs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        assert return_outputs is False, "return_outputs=True not supported"
        assert num_items_in_batch is not None, "num_items_in_batch must be provided"
        base_model = model.module if hasattr(model, "module") else model
        num_layers = len(base_model.model.layers)
        loss = torch.zeros(1, device=base_model.device)
        collect_obs = (int(self.state.global_step) % int(self.OBSERVE_INTERVAL)) == 0
        layer_mse_sum = torch.zeros(1, device=base_model.device) if collect_obs else None
        layer_rel_sum = torch.zeros(1, device=base_model.device) if collect_obs else None
        layer_absmax_sum = torch.zeros(1, device=base_model.device) if collect_obs else None
        layer_linear_peak_mb_sum = 0.0
        la_stats_sum: dict[str, float] = {} if collect_obs else {}
        la_stats_count = 0 if collect_obs else 0
        layer_den_frac_sum: dict[int, float] = {} if collect_obs else {}
        layer_den_frac_cnt: dict[int, int] = {} if collect_obs else {}
        layer_den_mse_sum: dict[int, float] = {} if collect_obs else {}
        layer_num_mse_sum: dict[int, float] = {} if collect_obs else {}
        step_peak_mb = None
        step_mem_before_mb = None
        device = base_model.device
        if collect_obs and torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            step_mem_before_mb = float(torch.cuda.memory_allocated(device)) / (1024.0 * 1024.0)
        attention_mask = (
            inputs["attention_mask"] if "attention_mask" in inputs else None
        )  # expected shape [B, T]

        def hook(module, args, kwargs, output):
            nonlocal loss
            nonlocal layer_mse_sum, layer_rel_sum, layer_absmax_sum
            nonlocal layer_linear_peak_mb_sum
            nonlocal la_stats_sum, la_stats_count
            attn_output = output[0]
            # Attention Mask
            kwargs = {k: v for k, v in kwargs.items() if k != "attention_mask"}
            # Teacher forward runs under no_grad for memory, but linear_attn branch
            # must keep grad for distillation.
            dev = attn_output.device
            autocast_ctx = nullcontext()
            if dev.type == "cuda":
                # Some FLA kernels (e.g. chunk DeltaNet) reject fp32. Force
                # mixed precision in the hook path even when surrounding context
                # falls back to fp32.
                autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                autocast_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype)
            mem_before_hook = None
            if collect_obs and attn_output.is_cuda:
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
                mem_before_hook = float(torch.cuda.memory_allocated(dev)) / (1024.0 * 1024.0)
            with torch.enable_grad():
                with autocast_ctx:
                    linear_attn_output = module.linear_attn(
                        *args, attention_mask=attention_mask, **kwargs
                    )
                diff = linear_attn_output[0] - attn_output.detach()
                if attention_mask is not None:
                    diff = diff * attention_mask.unsqueeze(-1)
                loss = loss + diff.pow(2).sum() / diff.shape[-1] / num_items_in_batch / num_layers
            if collect_obs and attn_output.is_cuda and mem_before_hook is not None:
                torch.cuda.synchronize(dev)
                hook_peak = float(torch.cuda.max_memory_allocated(dev)) / (1024.0 * 1024.0)
                layer_linear_peak_mb_sum += max(0.0, hook_peak - mem_before_hook)

            if collect_obs:
                diff_obs = diff.detach()
                layer_mse = diff_obs.square().mean()
                layer_mse_sum += layer_mse
                # Use unmasked teacher energy here to avoid additional large temporary
                # tensors in the observability path.
                teacher_energy = attn_output.detach().square().mean().clamp_min(1e-8)
                layer_rel_sum += layer_mse / teacher_energy
                layer_absmax_sum += diff_obs.abs().amax()

                linear_stats = getattr(module.linear_attn, "last_error_stats", None)
                if isinstance(linear_stats, dict) and len(linear_stats) > 0:
                    la_stats_count += 1
                    for key, value in linear_stats.items():
                        if isinstance(value, (float, int)):
                            val = float(value)
                            if math.isfinite(val):
                                la_stats_sum[key] = la_stats_sum.get(key, 0.0) + val
                    layer_idx = int(getattr(module, "layer_idx", -1))
                    if layer_idx >= 0:
                        den_frac = linear_stats.get("obs_kernel_err_den_frac", None)
                        if isinstance(den_frac, (float, int)) and math.isfinite(float(den_frac)):
                            layer_den_frac_sum[layer_idx] = layer_den_frac_sum.get(layer_idx, 0.0) + float(den_frac)
                            layer_den_frac_cnt[layer_idx] = layer_den_frac_cnt.get(layer_idx, 0) + 1
                        den_mse = linear_stats.get("obs_den_kernel_mse", None)
                        num_mse = linear_stats.get("obs_num_kernel_mse", None)
                        if isinstance(den_mse, (float, int)) and math.isfinite(float(den_mse)):
                            layer_den_mse_sum[layer_idx] = layer_den_mse_sum.get(layer_idx, 0.0) + float(den_mse)
                        if isinstance(num_mse, (float, int)) and math.isfinite(float(num_mse)):
                            layer_num_mse_sum[layer_idx] = layer_num_mse_sum.get(layer_idx, 0.0) + float(num_mse)

        handles = []
        for layer in base_model.model.layers:
            handle = layer.self_attn.register_forward_hook(hook, with_kwargs=True)
            handles.append(handle)
        # Stage1 only needs teacher attention activations. Avoid lm_head logits and
        # CE loss path to reduce memory peak.
        model_inputs = {}
        for k in ("input_ids", "attention_mask", "position_ids", "inputs_embeds", "cache_position"):
            if k in inputs:
                model_inputs[k] = inputs[k]
        with torch.no_grad():
            base_model.model(
                **model_inputs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
        if collect_obs and torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
            step_peak_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        for handle in handles:
            handle.remove()
        if collect_obs:
            obs_logs = {
                "obs_layer_mse": float((layer_mse_sum / num_layers).detach().item()),
                "obs_layer_rel_mse": float((layer_rel_sum / num_layers).detach().item()),
                "obs_layer_absmax": float((layer_absmax_sum / num_layers).detach().item()),
                "obs_layer_linear_hook_peak_mb": float(layer_linear_peak_mb_sum / num_layers),
            }
            if step_peak_mb is not None:
                obs_logs["obs_step_peak_mb"] = float(step_peak_mb)
            if step_mem_before_mb is not None and step_peak_mb is not None:
                obs_logs["obs_step_peak_delta_mb"] = float(max(0.0, step_peak_mb - step_mem_before_mb))
            if la_stats_count > 0:
                inv_count = 1.0 / float(la_stats_count)
                for key, total in la_stats_sum.items():
                    obs_logs[key] = total * inv_count
            if layer_den_frac_cnt:
                per_layer_den_frac: list[tuple[int, float]] = []
                for lid, cnt in layer_den_frac_cnt.items():
                    if cnt > 0:
                        per_layer_den_frac.append((lid, layer_den_frac_sum[lid] / float(cnt)))
                if per_layer_den_frac:
                    vals = torch.tensor([v for _, v in per_layer_den_frac], device=base_model.device, dtype=torch.float32)
                    obs_logs["obs_layer_den_frac_mean"] = float(vals.mean().item())
                    obs_logs["obs_layer_den_frac_std"] = float(vals.std(unbiased=False).item())
                    sorted_layers = sorted(per_layer_den_frac, key=lambda x: x[1], reverse=True)
                    topk = min(3, len(sorted_layers))
                    for rank in range(topk):
                        lid, frac = sorted_layers[rank]
                        obs_logs[f"obs_layer_den_frac_top{rank+1}_idx"] = float(lid)
                        obs_logs[f"obs_layer_den_frac_top{rank+1}_val"] = float(frac)
                        if lid in layer_den_mse_sum and lid in layer_num_mse_sum:
                            num_v = max(layer_num_mse_sum[lid], 1e-12)
                            obs_logs[f"obs_layer_den_over_num_mse_top{rank+1}"] = float(layer_den_mse_sum[lid] / num_v)
            self._last_observability_logs = obs_logs
        else:
            self._last_observability_logs = {}
        return loss


def load_model_tokenizer(model_args):
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = model.config
    config.linear_attention_type = model_args.linear_attention_type
    config.dual_delta_net_mode = model_args.dual_delta_net_mode
    config.dual_recompute_chunk_size = model_args.dual_recompute_chunk_size
    config.dual_delta_value_l2_norm = model_args.dual_delta_value_l2_norm
    config.dual_delta_value_norm_eps = model_args.dual_delta_value_norm_eps
    config.mean_delta_net_mode = model_args.mean_delta_net_mode
    config.mean_recompute_chunk_size = model_args.mean_recompute_chunk_size
    config.mean_delta_value_l2_norm = model_args.mean_delta_value_l2_norm
    config.mean_delta_value_norm_eps = model_args.mean_delta_value_norm_eps
    config.approxnet_v2_use_triton = model_args.approxnet_v2_use_triton
    config.approxnet_v2_beta_denom_eps = model_args.approxnet_v2_beta_denom_eps
    config.approxnet_v2_score_clip = model_args.approxnet_v2_score_clip
    config.approxnet_v2_qk_l2_norm = model_args.approxnet_v2_qk_l2_norm
    config.approxnet_v2_qk_l2_norm_eps = model_args.approxnet_v2_qk_l2_norm_eps
    config.approxnet_v2_output_norm = model_args.approxnet_v2_output_norm
    config.approxnet_v2_recompute_chunk_size = model_args.approxnet_v2_recompute_chunk_size
    config.approxnet_v2_use_sigmoid_gate = model_args.approxnet_v2_use_sigmoid_gate
    config.approxnet_v3_use_triton = model_args.approxnet_v3_use_triton
    config.approxnet_v3_beta_denom_eps = model_args.approxnet_v3_beta_denom_eps
    config.approxnet_v3_score_clip = model_args.approxnet_v3_score_clip
    config.approxnet_v3_qk_l2_norm = model_args.approxnet_v3_qk_l2_norm
    config.approxnet_v3_qk_l2_norm_eps = model_args.approxnet_v3_qk_l2_norm_eps
    config.approxnet_v3_output_norm = model_args.approxnet_v3_output_norm
    config.approxnet_v3_recompute_chunk_size = model_args.approxnet_v3_recompute_chunk_size
    config.approxnet_v3_use_sigmoid_gate = model_args.approxnet_v3_use_sigmoid_gate
    config.approxnet_v4_use_triton = model_args.approxnet_v4_use_triton
    config.approxnet_v4_z_score_eps = model_args.approxnet_v4_z_score_eps
    config.approxnet_v4_gate_alpha_init = model_args.approxnet_v4_gate_alpha_init
    config.approxnet_v4_gate_bias_init = model_args.approxnet_v4_gate_bias_init
    config.approxnet_v4_qk_l2_norm = model_args.approxnet_v4_qk_l2_norm
    config.approxnet_v4_qk_l2_norm_eps = model_args.approxnet_v4_qk_l2_norm_eps
    config.approxnet_v4_output_norm = model_args.approxnet_v4_output_norm
    config.approxnet_v4_recompute_chunk_size = model_args.approxnet_v4_recompute_chunk_size
    config.soam_d_r = model_args.soam_d_r
    config.soam_decay_alpha_init = model_args.soam_decay_alpha_init
    config.soam_decay_bias_init = model_args.soam_decay_bias_init
    config.soam_write_alpha_init = model_args.soam_write_alpha_init
    config.soam_write_bias_init = model_args.soam_write_bias_init
    config.soam_qk_l2_norm = model_args.soam_qk_l2_norm
    config.soam_qk_l2_norm_eps = model_args.soam_qk_l2_norm_eps
    config.soam_output_norm = model_args.soam_output_norm
    config.wla_d_r = model_args.wla_d_r
    config.wla_decay_alpha_init = model_args.wla_decay_alpha_init
    config.wla_decay_bias_init = model_args.wla_decay_bias_init
    config.wla_write_alpha_init = model_args.wla_write_alpha_init
    config.wla_write_bias_init = model_args.wla_write_bias_init
    config.wla_qk_l2_norm = model_args.wla_qk_l2_norm
    config.wla_qk_l2_norm_eps = model_args.wla_qk_l2_norm_eps
    config.wla_output_norm = model_args.wla_output_norm
    config.wla_sigma2_init = model_args.wla_sigma2_init
    config.sisa_d_r = model_args.sisa_d_r
    config.sisa_decay_alpha_init = model_args.sisa_decay_alpha_init
    config.sisa_decay_bias_init = model_args.sisa_decay_bias_init
    config.sisa_write_alpha_init = model_args.sisa_write_alpha_init
    config.sisa_write_bias_init = model_args.sisa_write_bias_init
    config.sisa_qk_l2_norm = model_args.sisa_qk_l2_norm
    config.sisa_qk_l2_norm_eps = model_args.sisa_qk_l2_norm_eps
    config.sisa_output_norm = model_args.sisa_output_norm
    config.sisa_beta_init = model_args.sisa_beta_init
    config.performer_nb_features = model_args.performer_nb_features
    config.performer_redraw_projection = model_args.performer_redraw_projection
    config.performer_projection_seed = model_args.performer_projection_seed
    config.performer_adaptive_center_sampling = model_args.performer_adaptive_center_sampling
    config.performer_adaptive_center_momentum = model_args.performer_adaptive_center_momentum
    config.performer_adaptive_center_log_clip = model_args.performer_adaptive_center_log_clip
    config.performer_use_dual_precondition_sampling = model_args.performer_use_dual_precondition_sampling
    config.performer_precondition_momentum = model_args.performer_precondition_momentum
    config.performer_precondition_eps = model_args.performer_precondition_eps
    config.performer_precondition_log_clip = model_args.performer_precondition_log_clip
    config.performer_precondition_mode = model_args.performer_precondition_mode
    config.performer_learnable_kernel_scale = model_args.performer_learnable_kernel_scale
    config.performer_dim_aware_kernel_scale = model_args.performer_dim_aware_kernel_scale
    config.performer_kernel_scale_init = model_args.performer_kernel_scale_init
    config.performer_use_deterministic_nodes = model_args.performer_use_deterministic_nodes
    config.performer_stratified_norm_sampling = model_args.performer_stratified_norm_sampling
    config.performer_stratified_jitter = model_args.performer_stratified_jitter
    config.performer_qmc_gaussian_sampling = model_args.performer_qmc_gaussian_sampling
    config.performer_qmc_scramble = model_args.performer_qmc_scramble
    config.performer_use_landmark_sampling = model_args.performer_use_landmark_sampling
    config.performer_landmark_ratio = model_args.performer_landmark_ratio
    config.performer_landmark_alpha_init = model_args.performer_landmark_alpha_init
    config.performer_landmark_in_denominator = model_args.performer_landmark_in_denominator
    config.performer_learnable_projection = model_args.performer_learnable_projection
    config.performer_learnable_projection_scale = model_args.performer_learnable_projection_scale
    config.performer_deterministic_ratio = model_args.performer_deterministic_ratio
    config.performer_qk_l2_norm = model_args.performer_qk_l2_norm
    config.performer_use_control_variate = model_args.performer_use_control_variate
    config.performer_use_adaptive_linear_cv = model_args.performer_use_adaptive_linear_cv
    config.performer_adaptive_linear_cv_init = model_args.performer_adaptive_linear_cv_init
    config.performer_adaptive_linear_cv_eps = model_args.performer_adaptive_linear_cv_eps
    config.performer_feature_rms_norm = model_args.performer_feature_rms_norm
    config.performer_feature_rms_norm_eps = model_args.performer_feature_rms_norm_eps
    config.performer_feature_pairwise_balance = model_args.performer_feature_pairwise_balance
    config.performer_feature_pairwise_balance_eps = model_args.performer_feature_pairwise_balance_eps
    config.performer_feature_pairwise_balance_log_clip = model_args.performer_feature_pairwise_balance_log_clip
    config.performer_control_variate_exp_clip = model_args.performer_control_variate_exp_clip
    config.performer_enable_error_observability = model_args.performer_enable_error_observability
    config.performer_error_observe_interval = model_args.performer_error_observe_interval
    config.performer_error_observe_max_tokens = model_args.performer_error_observe_max_tokens
    config.performer_error_observe_max_heads = model_args.performer_error_observe_max_heads
    config.performer_ortho_scaling = model_args.performer_ortho_scaling
    config.performer_use_projection_ensemble = model_args.performer_use_projection_ensemble
    config.performer_projection_ensemble_groups = model_args.performer_projection_ensemble_groups
    config.performer_state_update = model_args.performer_state_update
    config.performer_use_beta = model_args.performer_use_beta
    config.performer_use_value_gate = model_args.performer_use_value_gate
    config.performer_use_output_gate = model_args.performer_use_output_gate
    config.performer_use_decay = model_args.performer_use_decay
    config.performer_delta_beta_norm = model_args.performer_delta_beta_norm
    config.performer_delta_beta_norm_eps = model_args.performer_delta_beta_norm_eps
    config.performer_delta_denom_eps = model_args.performer_delta_denom_eps
    config.performer_delta_smooth_denom = model_args.performer_delta_smooth_denom
    config.performer_delta_denom_tau = model_args.performer_delta_denom_tau
    config.performer_delta_beta_cap = model_args.performer_delta_beta_cap
    config.performer_delta_safe_denom_floor = model_args.performer_delta_safe_denom_floor
    config.performer_delta_decouple_beta = model_args.performer_delta_decouple_beta
    config.performer_delta_denominator_update = model_args.performer_delta_denominator_update
    config.performer_delta_denominator_map = model_args.performer_delta_denominator_map
    config.performer_delta_denominator_stopgrad = model_args.performer_delta_denominator_stopgrad
    config.performer_delta_use_leaky_dplr = model_args.performer_delta_use_leaky_dplr
    config.performer_delta_leaky_rho_init = model_args.performer_delta_leaky_rho_init
    config.performer_delta_leaky_min_lambda = model_args.performer_delta_leaky_min_lambda
    config.performer_use_hybrid_numerator = model_args.performer_use_hybrid_numerator
    config.performer_hybrid_num_ratio = model_args.performer_hybrid_num_ratio
    config.performer_hybrid_alpha_init = model_args.performer_hybrid_alpha_init
    config.performer_use_cv_residual_shrinkage = model_args.performer_use_cv_residual_shrinkage
    config.performer_cv_residual_init = model_args.performer_cv_residual_init
    config.performer_cv_finite_sample_orthogonalize = model_args.performer_cv_finite_sample_orthogonalize
    config.performer_cv_finite_sample_orth_eps = model_args.performer_cv_finite_sample_orth_eps
    config.performer_use_jackknife_debias = model_args.performer_use_jackknife_debias
    config.performer_jackknife_groups = model_args.performer_jackknife_groups
    config.performer_jackknife_min_per_group = model_args.performer_jackknife_min_per_group
    config.performer_use_jackknife_adaptive_shrinkage = model_args.performer_use_jackknife_adaptive_shrinkage
    config.performer_jackknife_shrinkage_eps = model_args.performer_jackknife_shrinkage_eps
    config.performer_use_second_order_cv = model_args.performer_use_second_order_cv
    config.performer_use_cv_decoupled_second_order = model_args.performer_use_cv_decoupled_second_order
    config.performer_cv_decoupled_h2_ratio = model_args.performer_cv_decoupled_h2_ratio
    config.performer_cv_decoupled_h2_deterministic = model_args.performer_cv_decoupled_h2_deterministic
    config.performer_use_cv_decoupled_adaptive_h2_ratio = model_args.performer_use_cv_decoupled_adaptive_h2_ratio
    config.performer_cv_decoupled_ratio_ema_momentum = model_args.performer_cv_decoupled_ratio_ema_momentum
    config.performer_cv_decoupled_ratio_min = model_args.performer_cv_decoupled_ratio_min
    config.performer_cv_decoupled_ratio_max = model_args.performer_cv_decoupled_ratio_max
    config.performer_cv_split_feature_budget = model_args.performer_cv_split_feature_budget
    config.performer_use_third_order_cv = model_args.performer_use_third_order_cv
    config.performer_use_diag2_term = model_args.performer_use_diag2_term
    config.performer_diag2_ratio = model_args.performer_diag2_ratio
    config.performer_diag2_alpha_init = model_args.performer_diag2_alpha_init
    config.performer_use_dual_map = model_args.performer_use_dual_map
    config.performer_dual_map_den_ratio = model_args.performer_dual_map_den_ratio
    config.performer_dual_map_row_selection = model_args.performer_dual_map_row_selection
    config.performer_use_layerwise_den_ratio = model_args.performer_use_layerwise_den_ratio
    config.performer_layerwise_den_ratio_tau = model_args.performer_layerwise_den_ratio_tau
    config.performer_use_error_feedback_den_ratio = model_args.performer_use_error_feedback_den_ratio
    config.performer_error_feedback_den_ratio_momentum = model_args.performer_error_feedback_den_ratio_momentum
    config.performer_error_feedback_den_ratio_gain = model_args.performer_error_feedback_den_ratio_gain
    config.performer_use_adaptive_den_mix = model_args.performer_use_adaptive_den_mix
    config.performer_adaptive_den_mix_init = model_args.performer_adaptive_den_mix_init
    config.performer_dual_map_low_precision = model_args.performer_dual_map_low_precision
    config.performer_use_den_poly_kernel = model_args.performer_use_den_poly_kernel
    config.performer_den_poly_alpha_init = model_args.performer_den_poly_alpha_init
    config.performer_den_poly_constant = model_args.performer_den_poly_constant
    config.performer_den_poly_ratio = model_args.performer_den_poly_ratio
    for layer_idx in range(len(model.model.layers)):
        src_attn = model.model.layers[layer_idx].self_attn
        model.model.layers[layer_idx].self_attn.linear_attn = init_attention_module(
            config, layer_idx, source_attn=src_attn
        )
    return model, tokenizer


def main():
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(ModelArguments, dest="model")
    parser.add_arguments(TrainingArguments, dest="train")
    parser.add_arguments(DataArguments, dest="data")
    args = parser.parse_args()
    model, tokenizer = load_model_tokenizer(args.model)
    train_dataset = prapare_dataset(tokenizer, args.data, split="train")
    for name, param in model.named_parameters():
        if "linear_attn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    # MSE Train
    trainer = MSETrainer(
        model=model,
        tokenizer=tokenizer,
        args=args.train,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithFlattening(
            max_len=args.data.seq_len,
            pad_token_id=tokenizer.pad_token_id,
            return_position_ids=False,
        ),
    )
    trainer.train()
    # Save and Convert
    if trainer.is_world_process_zero():
        for layer in model.model.layers:
            layer.self_attn = layer.self_attn.linear_attn
        save_dir = Path(args.train.output_dir).absolute()
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        config = AutoConfig.from_pretrained(save_dir)
        model_type = config.model_type
        for key, value in settings[model_type].items():
            setattr(config, key, value)
        config.linear_attention_type = args.model.linear_attention_type
        config.dual_delta_net_mode = args.model.dual_delta_net_mode
        config.dual_recompute_chunk_size = args.model.dual_recompute_chunk_size
        config.dual_delta_value_l2_norm = args.model.dual_delta_value_l2_norm
        config.dual_delta_value_norm_eps = args.model.dual_delta_value_norm_eps
        config.mean_delta_net_mode = args.model.mean_delta_net_mode
        config.mean_recompute_chunk_size = args.model.mean_recompute_chunk_size
        config.mean_delta_value_l2_norm = args.model.mean_delta_value_l2_norm
        config.mean_delta_value_norm_eps = args.model.mean_delta_value_norm_eps
        config.approxnet_v2_use_triton = args.model.approxnet_v2_use_triton
        config.approxnet_v2_beta_denom_eps = args.model.approxnet_v2_beta_denom_eps
        config.approxnet_v2_score_clip = args.model.approxnet_v2_score_clip
        config.approxnet_v2_qk_l2_norm = args.model.approxnet_v2_qk_l2_norm
        config.approxnet_v2_qk_l2_norm_eps = args.model.approxnet_v2_qk_l2_norm_eps
        config.approxnet_v2_output_norm = args.model.approxnet_v2_output_norm
        config.approxnet_v2_recompute_chunk_size = args.model.approxnet_v2_recompute_chunk_size
        config.approxnet_v2_use_sigmoid_gate = args.model.approxnet_v2_use_sigmoid_gate
        config.approxnet_v3_use_triton = args.model.approxnet_v3_use_triton
        config.approxnet_v3_beta_denom_eps = args.model.approxnet_v3_beta_denom_eps
        config.approxnet_v3_score_clip = args.model.approxnet_v3_score_clip
        config.approxnet_v3_qk_l2_norm = args.model.approxnet_v3_qk_l2_norm
        config.approxnet_v3_qk_l2_norm_eps = args.model.approxnet_v3_qk_l2_norm_eps
        config.approxnet_v3_output_norm = args.model.approxnet_v3_output_norm
        config.approxnet_v3_recompute_chunk_size = args.model.approxnet_v3_recompute_chunk_size
        config.approxnet_v3_use_sigmoid_gate = args.model.approxnet_v3_use_sigmoid_gate
        config.approxnet_v4_use_triton = args.model.approxnet_v4_use_triton
        config.approxnet_v4_z_score_eps = args.model.approxnet_v4_z_score_eps
        config.approxnet_v4_gate_alpha_init = args.model.approxnet_v4_gate_alpha_init
        config.approxnet_v4_gate_bias_init = args.model.approxnet_v4_gate_bias_init
        config.approxnet_v4_qk_l2_norm = args.model.approxnet_v4_qk_l2_norm
        config.approxnet_v4_qk_l2_norm_eps = args.model.approxnet_v4_qk_l2_norm_eps
        config.approxnet_v4_output_norm = args.model.approxnet_v4_output_norm
        config.approxnet_v4_recompute_chunk_size = args.model.approxnet_v4_recompute_chunk_size
        config.soam_d_r = args.model.soam_d_r
        config.soam_decay_alpha_init = args.model.soam_decay_alpha_init
        config.soam_decay_bias_init = args.model.soam_decay_bias_init
        config.soam_write_alpha_init = args.model.soam_write_alpha_init
        config.soam_write_bias_init = args.model.soam_write_bias_init
        config.soam_qk_l2_norm = args.model.soam_qk_l2_norm
        config.soam_qk_l2_norm_eps = args.model.soam_qk_l2_norm_eps
        config.soam_output_norm = args.model.soam_output_norm
        config.performer_nb_features = args.model.performer_nb_features
        config.performer_redraw_projection = args.model.performer_redraw_projection
        config.performer_projection_seed = args.model.performer_projection_seed
        config.performer_adaptive_center_sampling = args.model.performer_adaptive_center_sampling
        config.performer_adaptive_center_momentum = args.model.performer_adaptive_center_momentum
        config.performer_adaptive_center_log_clip = args.model.performer_adaptive_center_log_clip
        config.performer_use_dual_precondition_sampling = args.model.performer_use_dual_precondition_sampling
        config.performer_precondition_momentum = args.model.performer_precondition_momentum
        config.performer_precondition_eps = args.model.performer_precondition_eps
        config.performer_precondition_log_clip = args.model.performer_precondition_log_clip
        config.performer_precondition_mode = args.model.performer_precondition_mode
        config.performer_use_deterministic_nodes = args.model.performer_use_deterministic_nodes
        config.performer_stratified_norm_sampling = args.model.performer_stratified_norm_sampling
        config.performer_stratified_jitter = args.model.performer_stratified_jitter
        config.performer_qmc_gaussian_sampling = args.model.performer_qmc_gaussian_sampling
        config.performer_qmc_scramble = args.model.performer_qmc_scramble
        config.performer_use_landmark_sampling = args.model.performer_use_landmark_sampling
        config.performer_landmark_ratio = args.model.performer_landmark_ratio
        config.performer_landmark_alpha_init = args.model.performer_landmark_alpha_init
        config.performer_landmark_in_denominator = args.model.performer_landmark_in_denominator
        config.performer_learnable_projection = args.model.performer_learnable_projection
        config.performer_learnable_projection_scale = args.model.performer_learnable_projection_scale
        config.performer_deterministic_ratio = args.model.performer_deterministic_ratio
        config.performer_qk_l2_norm = args.model.performer_qk_l2_norm
        config.performer_use_control_variate = args.model.performer_use_control_variate
        config.performer_use_adaptive_linear_cv = args.model.performer_use_adaptive_linear_cv
        config.performer_adaptive_linear_cv_init = args.model.performer_adaptive_linear_cv_init
        config.performer_adaptive_linear_cv_eps = args.model.performer_adaptive_linear_cv_eps
        config.performer_feature_rms_norm = args.model.performer_feature_rms_norm
        config.performer_feature_rms_norm_eps = args.model.performer_feature_rms_norm_eps
        config.performer_feature_pairwise_balance = args.model.performer_feature_pairwise_balance
        config.performer_feature_pairwise_balance_eps = args.model.performer_feature_pairwise_balance_eps
        config.performer_feature_pairwise_balance_log_clip = args.model.performer_feature_pairwise_balance_log_clip
        config.performer_control_variate_exp_clip = args.model.performer_control_variate_exp_clip
        config.performer_enable_error_observability = args.model.performer_enable_error_observability
        config.performer_error_observe_interval = args.model.performer_error_observe_interval
        config.performer_error_observe_max_tokens = args.model.performer_error_observe_max_tokens
        config.performer_error_observe_max_heads = args.model.performer_error_observe_max_heads
        config.performer_ortho_scaling = args.model.performer_ortho_scaling
        config.performer_use_projection_ensemble = args.model.performer_use_projection_ensemble
        config.performer_projection_ensemble_groups = args.model.performer_projection_ensemble_groups
        config.performer_state_update = args.model.performer_state_update
        config.performer_use_beta = args.model.performer_use_beta
        config.performer_use_value_gate = args.model.performer_use_value_gate
        config.performer_use_output_gate = args.model.performer_use_output_gate
        config.performer_use_decay = args.model.performer_use_decay
        config.performer_delta_beta_norm = args.model.performer_delta_beta_norm
        config.performer_delta_beta_norm_eps = args.model.performer_delta_beta_norm_eps
        config.performer_delta_denom_eps = args.model.performer_delta_denom_eps
        config.performer_delta_smooth_denom = args.model.performer_delta_smooth_denom
        config.performer_delta_denom_tau = args.model.performer_delta_denom_tau
        config.performer_delta_beta_cap = args.model.performer_delta_beta_cap
        config.performer_delta_safe_denom_floor = args.model.performer_delta_safe_denom_floor
        config.performer_delta_decouple_beta = args.model.performer_delta_decouple_beta
        config.performer_delta_denominator_update = args.model.performer_delta_denominator_update
        config.performer_delta_denominator_map = args.model.performer_delta_denominator_map
        config.performer_delta_denominator_stopgrad = args.model.performer_delta_denominator_stopgrad
        config.performer_delta_use_leaky_dplr = args.model.performer_delta_use_leaky_dplr
        config.performer_delta_leaky_rho_init = args.model.performer_delta_leaky_rho_init
        config.performer_delta_leaky_min_lambda = args.model.performer_delta_leaky_min_lambda
        config.performer_use_hybrid_numerator = args.model.performer_use_hybrid_numerator
        config.performer_hybrid_num_ratio = args.model.performer_hybrid_num_ratio
        config.performer_hybrid_alpha_init = args.model.performer_hybrid_alpha_init
        config.performer_use_cv_residual_shrinkage = args.model.performer_use_cv_residual_shrinkage
        config.performer_cv_residual_init = args.model.performer_cv_residual_init
        config.performer_cv_finite_sample_orthogonalize = args.model.performer_cv_finite_sample_orthogonalize
        config.performer_cv_finite_sample_orth_eps = args.model.performer_cv_finite_sample_orth_eps
        config.performer_use_jackknife_debias = args.model.performer_use_jackknife_debias
        config.performer_jackknife_groups = args.model.performer_jackknife_groups
        config.performer_jackknife_min_per_group = args.model.performer_jackknife_min_per_group
        config.performer_use_jackknife_adaptive_shrinkage = args.model.performer_use_jackknife_adaptive_shrinkage
        config.performer_jackknife_shrinkage_eps = args.model.performer_jackknife_shrinkage_eps
        config.performer_use_second_order_cv = args.model.performer_use_second_order_cv
        config.performer_use_cv_decoupled_second_order = args.model.performer_use_cv_decoupled_second_order
        config.performer_cv_decoupled_h2_ratio = args.model.performer_cv_decoupled_h2_ratio
        config.performer_cv_decoupled_h2_deterministic = args.model.performer_cv_decoupled_h2_deterministic
        config.performer_use_cv_decoupled_adaptive_h2_ratio = args.model.performer_use_cv_decoupled_adaptive_h2_ratio
        config.performer_cv_decoupled_ratio_ema_momentum = args.model.performer_cv_decoupled_ratio_ema_momentum
        config.performer_cv_decoupled_ratio_min = args.model.performer_cv_decoupled_ratio_min
        config.performer_cv_decoupled_ratio_max = args.model.performer_cv_decoupled_ratio_max
        config.performer_cv_split_feature_budget = args.model.performer_cv_split_feature_budget
        config.performer_use_third_order_cv = args.model.performer_use_third_order_cv
        config.performer_use_diag2_term = args.model.performer_use_diag2_term
        config.performer_diag2_ratio = args.model.performer_diag2_ratio
        config.performer_diag2_alpha_init = args.model.performer_diag2_alpha_init
        config.performer_use_dual_map = args.model.performer_use_dual_map
        config.performer_dual_map_den_ratio = args.model.performer_dual_map_den_ratio
        config.performer_dual_map_row_selection = args.model.performer_dual_map_row_selection
        config.performer_use_layerwise_den_ratio = args.model.performer_use_layerwise_den_ratio
        config.performer_layerwise_den_ratio_tau = args.model.performer_layerwise_den_ratio_tau
        config.performer_use_error_feedback_den_ratio = args.model.performer_use_error_feedback_den_ratio
        config.performer_error_feedback_den_ratio_momentum = args.model.performer_error_feedback_den_ratio_momentum
        config.performer_error_feedback_den_ratio_gain = args.model.performer_error_feedback_den_ratio_gain
        config.performer_use_adaptive_den_mix = args.model.performer_use_adaptive_den_mix
        config.performer_adaptive_den_mix_init = args.model.performer_adaptive_den_mix_init
        config.performer_dual_map_low_precision = args.model.performer_dual_map_low_precision
        config.performer_use_den_poly_kernel = args.model.performer_use_den_poly_kernel
        config.performer_den_poly_alpha_init = args.model.performer_den_poly_alpha_init
        config.performer_den_poly_constant = args.model.performer_den_poly_constant
        config.performer_den_poly_ratio = args.model.performer_den_poly_ratio
        config.performer_learnable_kernel_scale = args.model.performer_learnable_kernel_scale
        config.performer_dim_aware_kernel_scale = args.model.performer_dim_aware_kernel_scale
        config.performer_kernel_scale_init = args.model.performer_kernel_scale_init
        config.save_pretrained(save_dir)
        current_dir = Path(__file__).resolve().parent.parent
        for item in os.listdir(current_dir / remote_code_dirs[model_type]):
            if not item.endswith(".py"):
                continue
            source_path = current_dir / remote_code_dirs[model_type] / item
            target_path = save_dir / item
            shutil.copy(source_path, target_path)
        for utils_file in utils_files:
            src_path = current_dir / utils_file
            if not src_path.exists():
                continue
            file_name = utils_file.split("/")[-1]
            shutil.copy(src_path, save_dir / file_name)
        _validate_remote_code_export(save_dir)
        print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
