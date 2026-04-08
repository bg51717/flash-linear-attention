from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules.l2norm import l2norm_fwd
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution

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
from .linear_attention_performer_plus_triton import (
    performer_plus_pdf_delta_attention_triton,
)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


_PDF_DELTA_STATE_UPDATES = {"pdf_delta", "overwrite"}


def _build_antithetic_orthogonal_random_matrix(
    num_heads: int,
    num_features: int,
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    scaling: int,
    antithetic: bool,
    stratified_norm_sampling: bool,
    stratified_jitter: bool,
    qmc_gaussian_sampling: bool,
    qmc_scramble: bool,
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
            stratified_norm_sampling=stratified_norm_sampling,
            stratified_jitter=stratified_jitter,
            qmc_gaussian_sampling=qmc_gaussian_sampling,
            qmc_scramble=qmc_scramble,
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
        stratified_norm_sampling=stratified_norm_sampling,
        stratified_jitter=stratified_jitter,
        qmc_gaussian_sampling=qmc_gaussian_sampling,
        qmc_scramble=qmc_scramble,
        generator=generator,
    )
    neg_count = num_features - half
    if neg_count <= 0:
        return base
    return torch.cat([base, -base[:, :neg_count]], dim=1)


def _build_deterministic_orthogonal_nodes(
    num_heads: int,
    num_features: int,
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_features <= 0:
        return torch.empty(num_heads, 0, head_dim, device=device, dtype=dtype)
    if head_dim <= 0:
        raise ValueError(f"head_dim must be > 0, got {head_dim}.")

    # Deterministic orthoplex nodes: +/- sqrt(d) * e_i, cycled per head.
    eye = torch.eye(head_dim, device=device, dtype=torch.float32)
    rows: list[torch.Tensor] = []
    for row_idx in range(num_features):
        idx = row_idx % head_dim
        phase = row_idx // head_dim
        sign = -1.0 if (phase % 2 == 1) else 1.0
        rows.append(sign * eye[idx])
    base = torch.stack(rows, dim=0)  # [M, D]
    base = base.unsqueeze(0).repeat(num_heads, 1, 1)

    if num_heads > 1:
        shifts = torch.arange(num_heads, device=device) % head_dim
        # Head-wise cyclic shift to avoid identical deterministic rows across heads.
        for h in range(num_heads):
            if shifts[h] != 0:
                base[h] = torch.roll(base[h], shifts=int(shifts[h].item()), dims=-1)

    scale = math.sqrt(float(head_dim))
    return (base * scale).to(dtype=dtype)


def _build_mixed_projection_matrix(
    num_heads: int,
    num_features: int,
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    scaling: int,
    antithetic: bool,
    deterministic_ratio: float,
    stratified_norm_sampling: bool,
    stratified_jitter: bool,
    qmc_gaussian_sampling: bool,
    qmc_scramble: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if deterministic_ratio <= 0:
        return _build_antithetic_orthogonal_random_matrix(
            num_heads=num_heads,
            num_features=num_features,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            scaling=scaling,
            antithetic=antithetic,
            stratified_norm_sampling=stratified_norm_sampling,
            stratified_jitter=stratified_jitter,
            qmc_gaussian_sampling=qmc_gaussian_sampling,
            qmc_scramble=qmc_scramble,
            generator=generator,
        )

    num_det = min(num_features, max(1, int(round(num_features * deterministic_ratio))))
    num_rand = max(0, num_features - num_det)

    det = _build_deterministic_orthogonal_nodes(
        num_heads=num_heads,
        num_features=num_det,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )
    if num_rand == 0:
        return det
    rnd = _build_antithetic_orthogonal_random_matrix(
        num_heads=num_heads,
        num_features=num_rand,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        scaling=scaling,
        antithetic=antithetic,
        stratified_norm_sampling=stratified_norm_sampling,
        stratified_jitter=stratified_jitter,
        qmc_gaussian_sampling=qmc_gaussian_sampling,
        qmc_scramble=qmc_scramble,
        generator=generator,
    )
    return torch.cat([det, rnd], dim=1)


def _performer_control_variate_feature_map(
    x: torch.Tensor,
    projection_matrix: torch.Tensor,
    *,
    center: torch.Tensor | None = None,
    exp_clip: float | None = 20.0,
    residual_weight: torch.Tensor | None = None,
    linear_cv_coef: torch.Tensor | None = None,
    linear_cv_coef_eps: float = 1e-4,
    use_second_order_cv: bool = True,
    use_third_order_cv: bool = False,
    split_feature_budget: bool = True,
    finite_sample_orthogonalize: bool = False,
    finite_sample_orth_eps: float = 1e-4,
) -> torch.Tensor:
    """
    Control-variate FAVOR+ feature map.

    For s = <x_q, x_k> with x already scaled by d^{-1/4}, we use:
        exp(s) = 1 + s + E_omega[(f_omega(x_q) - 1 - <omega, x_q>) * (f_omega(x_k) - 1 - <omega, x_k>)]
    where f_omega(x) = exp(<omega, x> - ||x||^2 / 2).

    With `use_second_order_cv=True`, we use a second-order Hermite control variate:
        f(omega, x) = exp(<omega, x> - ||x||^2 / 2)
        h2(omega, x) = 0.5 * ((<omega, x>)^2 - ||x||^2)
        r2 = f - 1 - <omega, x> - h2
    and features [1, x, h2, r2].

    With `use_third_order_cv=True`, we additionally remove the cubic Hermite term:
        h3(omega, x) = ( (<omega, x>)^3 - 3<omega, x>||x||^2 ) / 6
        r3 = f - 1 - <omega, x> - h2 - h3
    and features [1, x, h2, h3, r3].

    This removes quadratic components from the stochastic residual in expectation,
    reducing estimator variance for moderate dot-product regimes.

    Optional first-order generalized control variate (`linear_cv_coef=b`):
        exp(s) = 1 + (2b-b^2)s + E[(f-1-bz)_q (f-1-bz)_k],  for b in (0, 2)
    where z=<w,x>. We realize this by scaling deterministic linear channels by
    a = sqrt(2b-b^2) and replacing residual with (f-1-bz). b=1 recovers the
    standard first-order CV decomposition.
    """
    x = x.to(torch.float32)
    projection_matrix = projection_matrix.to(torch.float32)
    if use_third_order_cv and split_feature_budget:
        # Keep total feature width close to first-order map by splitting rows
        # across h2/h3/residual stochastic parts.
        used_rows = max(1, projection_matrix.shape[1] // 3)
        projection_matrix = projection_matrix[:, :used_rows, :]
    elif use_second_order_cv and split_feature_budget:
        # Keep total feature width close to the first-order map by using half of
        # projection rows for each of the second-order and residual stochastic parts.
        used_rows = max(1, projection_matrix.shape[1] // 2)
        projection_matrix = projection_matrix[:, :used_rows, :]

    data_normalizer = x.shape[-1] ** -0.25
    x_scaled = x * data_normalizer
    amplitude = None
    if center is not None:
        center_scaled = center.to(torch.float32) * data_normalizer
        center_scaled = center_scaled.view(1, 1, center_scaled.shape[0], center_scaled.shape[1])
        # Shifted-CV identity:
        #   exp(<w,x>-||x||^2/2) = A(x,c) * exp(<w,x-c>-||x-c||^2/2)
        # where A(x,c)=exp(<c,x>-||c||^2/2), valid for any fixed c.
        amplitude = torch.exp(
            (x_scaled * center_scaled).sum(dim=-1, keepdim=True)
            - 0.5 * center_scaled.square().sum(dim=-1, keepdim=True)
        )
        x_scaled = x_scaled - center_scaled
    data_dash = torch.einsum('bthd,hmd->bthm', x_scaled, projection_matrix)
    x_norm_sq = x_scaled.square().sum(dim=-1, keepdim=True)
    diag_data = x_norm_sq * 0.5

    linear_coef = None
    linear_scale = None
    if (
        linear_cv_coef is not None
        and not use_second_order_cv
        and not use_third_order_cv
    ):
        linear_coef = linear_cv_coef.to(torch.float32).view(1, 1, -1, 1)
        linear_coef = linear_coef.clamp(
            min=float(linear_cv_coef_eps),
            max=2.0 - float(linear_cv_coef_eps),
        )
        linear_scale = (2.0 * linear_coef - linear_coef.square()).clamp_min(float(linear_cv_coef_eps)).sqrt()

    exponent = data_dash - diag_data
    if exp_clip is not None:
        exponent = exponent.clamp_max(float(exp_clip))
    exp_part = torch.exp(exponent)

    ratio = projection_matrix.shape[1] ** -0.5
    third_order = None
    if use_third_order_cv:
        second_order = 0.5 * (data_dash.square() - x_norm_sq)
        third_order = (data_dash.pow(3) - 3.0 * data_dash * x_norm_sq) / 6.0
        residual = ratio * (exp_part - 1.0 - data_dash - second_order - third_order)
        second_order = ratio * second_order
        third_order = ratio * third_order
    elif use_second_order_cv:
        second_order = 0.5 * (data_dash.square() - x_norm_sq)
        residual = ratio * (exp_part - 1.0 - data_dash - second_order)
        second_order = ratio * second_order
    else:
        second_order = None
        if linear_coef is None:
            residual = ratio * (exp_part - 1.0 - data_dash)
        else:
            residual = ratio * (exp_part - 1.0 - linear_coef * data_dash)
    if finite_sample_orthogonalize:
        # Finite-sample orthogonalization: force stochastic residual channels to
        # be sample-orthogonal to deterministic low-order bases (1, <w,x>, h2),
        # reducing leakage and variance under small feature budgets.
        residual = residual - residual.mean(dim=-1, keepdim=True)
        dash_centered = data_dash - data_dash.mean(dim=-1, keepdim=True)
        dash_denom = dash_centered.square().mean(dim=-1, keepdim=True).clamp_min(finite_sample_orth_eps)
        dash_coef = (residual * dash_centered).mean(dim=-1, keepdim=True) / dash_denom
        residual = residual - dash_coef * dash_centered
        if second_order is not None:
            h2_centered = second_order - second_order.mean(dim=-1, keepdim=True)
            h2_denom = h2_centered.square().mean(dim=-1, keepdim=True).clamp_min(finite_sample_orth_eps)
            h2_coef = (residual * h2_centered).mean(dim=-1, keepdim=True) / h2_denom
            residual = residual - h2_coef * h2_centered
        if third_order is not None:
            h3_centered = third_order - third_order.mean(dim=-1, keepdim=True)
            h3_denom = h3_centered.square().mean(dim=-1, keepdim=True).clamp_min(finite_sample_orth_eps)
            h3_coef = (residual * h3_centered).mean(dim=-1, keepdim=True) / h3_denom
            residual = residual - h3_coef * h3_centered

    if residual_weight is not None:
        residual = residual * residual_weight.view(1, 1, -1, 1).to(residual.dtype)
    ones = torch.ones(
        (*x_scaled.shape[:3], 1),
        device=x_scaled.device,
        dtype=x_scaled.dtype,
    )
    if linear_scale is not None:
        x_scaled = x_scaled * linear_scale
    if amplitude is not None:
        amp = amplitude.to(x_scaled.dtype)
        ones = ones * amp
        x_scaled = x_scaled * amp
        residual = residual * amp
        if second_order is not None:
            second_order = second_order * amp
        if third_order is not None:
            third_order = third_order * amp
    if third_order is not None:
        return torch.cat([ones, x_scaled, second_order, third_order, residual], dim=-1)
    if second_order is not None:
        return torch.cat([ones, x_scaled, second_order, residual], dim=-1)
    return torch.cat([ones, x_scaled, residual], dim=-1)


def _performer_control_variate_residual_map(
    x: torch.Tensor,
    projection_matrix: torch.Tensor,
    *,
    center: torch.Tensor | None = None,
    exp_clip: float | None = 20.0,
    linear_cv_coef: torch.Tensor | None = None,
    linear_cv_coef_eps: float = 1e-4,
    use_second_order_cv: bool = False,
    use_third_order_cv: bool = False,
    split_feature_budget: bool = True,
) -> torch.Tensor:
    """
    Residual-only control-variate channels.

    This keeps deterministic low-order terms outside of the sampled branch and
    is useful when mixing multiple sampling schemes:
        exp = (1 + linear + optional h2) + residual.
    """
    x = x.to(torch.float32)
    projection_matrix = projection_matrix.to(torch.float32)
    if use_third_order_cv and split_feature_budget:
        used_rows = max(1, projection_matrix.shape[1] // 3)
        projection_matrix = projection_matrix[:, :used_rows, :]
    elif use_second_order_cv and split_feature_budget:
        used_rows = max(1, projection_matrix.shape[1] // 2)
        projection_matrix = projection_matrix[:, :used_rows, :]

    data_normalizer = x.shape[-1] ** -0.25
    x_scaled = x * data_normalizer
    amplitude = None
    if center is not None:
        center_scaled = center.to(torch.float32) * data_normalizer
        center_scaled = center_scaled.view(1, 1, center_scaled.shape[0], center_scaled.shape[1])
        amplitude = torch.exp(
            (x_scaled * center_scaled).sum(dim=-1, keepdim=True)
            - 0.5 * center_scaled.square().sum(dim=-1, keepdim=True)
        )
        x_scaled = x_scaled - center_scaled
    data_dash = torch.einsum('bthd,hmd->bthm', x_scaled, projection_matrix)
    x_norm_sq = x_scaled.square().sum(dim=-1, keepdim=True)
    diag_data = x_norm_sq * 0.5

    linear_coef = None
    if (
        linear_cv_coef is not None
        and not use_second_order_cv
        and not use_third_order_cv
    ):
        linear_coef = linear_cv_coef.to(torch.float32).view(1, 1, -1, 1)
        linear_coef = linear_coef.clamp(
            min=float(linear_cv_coef_eps),
            max=2.0 - float(linear_cv_coef_eps),
        )

    exponent = data_dash - diag_data
    if exp_clip is not None:
        exponent = exponent.clamp_max(float(exp_clip))
    exp_part = torch.exp(exponent)

    ratio = projection_matrix.shape[1] ** -0.5
    if use_third_order_cv:
        second_order = 0.5 * (data_dash.square() - x_norm_sq)
        third_order = (data_dash.pow(3) - 3.0 * data_dash * x_norm_sq) / 6.0
        residual = ratio * (exp_part - 1.0 - data_dash - second_order - third_order)
    elif use_second_order_cv:
        second_order = 0.5 * (data_dash.square() - x_norm_sq)
        residual = ratio * (exp_part - 1.0 - data_dash - second_order)
    else:
        if linear_coef is None:
            residual = ratio * (exp_part - 1.0 - data_dash)
        else:
            residual = ratio * (exp_part - 1.0 - linear_coef * data_dash)
    if amplitude is not None:
        residual = residual * amplitude.to(residual.dtype)
    return residual.to(x_scaled.dtype)


def _performer_control_variate_prefix_map(
    x: torch.Tensor,
    *,
    center: torch.Tensor | None = None,
    linear_cv_coef: torch.Tensor | None = None,
    linear_cv_coef_eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.to(torch.float32)
    data_normalizer = x.shape[-1] ** -0.25
    x_scaled = x * data_normalizer
    if linear_cv_coef is not None:
        linear_coef = linear_cv_coef.to(torch.float32).view(1, 1, -1, 1)
        linear_coef = linear_coef.clamp(
            min=float(linear_cv_coef_eps),
            max=2.0 - float(linear_cv_coef_eps),
        )
        linear_scale = (2.0 * linear_coef - linear_coef.square()).clamp_min(float(linear_cv_coef_eps)).sqrt()
        x_scaled = x_scaled * linear_scale
    amplitude = None
    if center is not None:
        center_scaled = center.to(torch.float32) * data_normalizer
        center_scaled = center_scaled.view(1, 1, center_scaled.shape[0], center_scaled.shape[1])
        amplitude = torch.exp(
            (x_scaled * center_scaled).sum(dim=-1, keepdim=True)
            - 0.5 * center_scaled.square().sum(dim=-1, keepdim=True)
        )
        x_scaled = x_scaled - center_scaled
    ones = torch.ones(
        (*x_scaled.shape[:3], 1),
        device=x_scaled.device,
        dtype=x_scaled.dtype,
    )
    if amplitude is not None:
        amp = amplitude.to(x_scaled.dtype)
        ones = ones * amp
        x_scaled = x_scaled * amp
    return ones, x_scaled


def _performer_second_order_feature_map(
    x: torch.Tensor,
    projection_matrix: torch.Tensor,
    *,
    center: torch.Tensor | None = None,
) -> torch.Tensor:
    x = x.to(torch.float32)
    projection_matrix = projection_matrix.to(torch.float32)
    data_normalizer = x.shape[-1] ** -0.25
    x_scaled = x * data_normalizer
    amplitude = None
    if center is not None:
        center_scaled = center.to(torch.float32) * data_normalizer
        center_scaled = center_scaled.view(1, 1, center_scaled.shape[0], center_scaled.shape[1])
        amplitude = torch.exp(
            (x_scaled * center_scaled).sum(dim=-1, keepdim=True)
            - 0.5 * center_scaled.square().sum(dim=-1, keepdim=True)
        )
        x_scaled = x_scaled - center_scaled
    data_dash = torch.einsum('bthd,hmd->bthm', x_scaled, projection_matrix)
    x_norm_sq = x_scaled.square().sum(dim=-1, keepdim=True)
    ratio = projection_matrix.shape[1] ** -0.5
    second_order = 0.5 * (data_dash.square() - x_norm_sq)
    if amplitude is not None:
        second_order = second_order * amplitude.to(second_order.dtype)
    return (ratio * second_order).to(x_scaled.dtype)


def _performer_positive_linear_feature_map(
    x: torch.Tensor,
    projection_matrix: torch.Tensor,
    *,
    constant: float,
) -> torch.Tensor:
    """
    Deterministic positive kernel features:
        K_lin+(x, y) = c + <x, y> / c
    realized by:
        phi_lin+(x) = [sqrt(c), (P x) / sqrt(c)].

    When c upper-bounds |<x, y>| in the operating regime, K_lin+ stays non-negative.
    """
    if constant <= 0:
        raise ValueError(f"constant must be > 0, got {constant}.")
    x = x.to(torch.float32)
    projection_matrix = F.normalize(projection_matrix.to(torch.float32), dim=-1)
    x_proj = torch.einsum('bthd,hrd->bthr', x, projection_matrix)
    c_sqrt = math.sqrt(float(constant))
    const = torch.full(
        (*x_proj.shape[:3], 1),
        fill_value=c_sqrt,
        device=x_proj.device,
        dtype=x_proj.dtype,
    )
    return torch.cat([const, x_proj / c_sqrt], dim=-1)


def _performer_shared_softmax_feature_map(
    x: torch.Tensor,
    projection_matrix: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """
    Symmetric FAVOR+ positive feature map used by the PDF delta update.

    Unlike the standard Performer query/key pair, this map uses the same
    per-token stabilizer for both q and k so the PDF delta derivation can treat
    `phi(x)` consistently when the stored key is queried again.
    """
    x = x.to(torch.float32)
    projection_matrix = projection_matrix.to(torch.float32)

    data_normalizer = x.shape[-1] ** -0.25
    x = x * data_normalizer
    ratio = projection_matrix.shape[1] ** -0.5

    data_dash = torch.einsum('bthd,hmd->bthm', x, projection_matrix)
    diag_data = 0.5 * x.square().sum(dim=-1, keepdim=True)
    stabilizer = data_dash.amax(dim=-1, keepdim=True)
    return ratio * (torch.exp(data_dash - diag_data - stabilizer) + eps)


def performer_plus_pdf_delta_attention(
    q_prime: torch.Tensor,
    k_prime: torch.Tensor,
    v: torch.Tensor,
    *,
    q_prime_den: torch.Tensor | None = None,
    k_prime_den: torch.Tensor | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """
    Performer+ delta update inspired by `performer+delta.pdf`.

    The updated PDF defines an intermediate state via one standard Performer
    update that writes the negative current output at the current key:

        S_t^- = Update_perf(S_{t-1}, k_t, -Output_perf(S_{t-1}, k_t))

    followed by a normal Performer write with `v_t`. With shared positive
    features `phi_t = phi(k_t)` and `psi_t = phi(q_t)`, this yields:

        W_t = W_{t-1}
              - (W_{t-1} phi_t / (z_{t-1}^T phi_t)) phi_t^T
              + v_t phi_t^T
        z_t = z_{t-1} + 2 phi_t

    In the transposed state layout used here, `kv_state` has shape [N, H, M, V]
    and stores W^T. The recurrence is:

        y_t = kv_state_{t-1}^T phi_t
        d_t = z_{t-1}^T phi_t
        o_t^- = y_t / d_t
        kv_state_t = kv_state_{t-1} + phi_t (v_t - o_t^-)^T
        z_t = z_{t-1} + 2 phi_t
        o_t = kv_state_t^T psi_t / (z_t^T psi_t + eps)
    """
    if q_prime.ndim != 4 or k_prime.ndim != 4 or v.ndim != 4:
        raise ValueError("q_prime, k_prime, v must have shape [B, T, H, D].")
    if q_prime.shape[:3] != k_prime.shape[:3] or q_prime.shape[:3] != v.shape[:3]:
        raise ValueError("Leading dimensions of q_prime, k_prime, v must match.")
    if q_prime_den is None:
        q_prime_den = q_prime
    if k_prime_den is None:
        k_prime_den = k_prime
    if q_prime_den.ndim != 4 or k_prime_den.ndim != 4:
        raise ValueError("q_prime_den and k_prime_den must have shape [B, T, H, D].")
    if q_prime_den.shape[:3] != q_prime.shape[:3] or k_prime_den.shape[:3] != q_prime.shape[:3]:
        raise ValueError("Leading dimensions of denominator maps must match q_prime.")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}.")

    q_num = q_prime.float()
    k_num = k_prime.float()
    q_den = q_prime_den.float()
    k_den = k_prime_den.float()
    v32 = v.float()

    feature_dim = k_num.shape[-1]
    den_feature_dim = k_den.shape[-1]
    value_dim = v32.shape[-1]

    def _run_segment(
        q_seg: torch.Tensor,
        k_seg: torch.Tensor,
        v_seg: torch.Tensor,
        q_den_seg: torch.Tensor,
        k_den_seg: torch.Tensor,
        kv_state: torch.Tensor,
        z_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outs: list[torch.Tensor] = []
        for t in range(q_seg.shape[1]):
            phi_num = k_seg[:, t]
            phi_den = k_den_seg[:, t]
            psi_num = q_seg[:, t]
            psi_den = q_den_seg[:, t]
            value_t = v_seg[:, t]

            current_num_at_key = torch.einsum('nhmv,nhm->nhv', kv_state, phi_num)
            current_den_at_key = (z_state * phi_den).sum(dim=-1, keepdim=True).clamp_min(float(eps))
            current_output_at_key = current_num_at_key / current_den_at_key
            delta_value = value_t - current_output_at_key

            kv_state = kv_state + torch.einsum(
                'nhm,nhv->nhmv',
                phi_num,
                delta_value,
            )
            z_state = z_state + 2.0 * phi_den

            num_t = torch.einsum('nhmv,nhm->nhv', kv_state, psi_num)
            den_t = (z_state * psi_den).sum(dim=-1, keepdim=True).clamp_min(float(eps))
            outs.append((num_t / den_t).to(v.dtype))

        return torch.stack(outs, dim=1), kv_state, z_state

    if cu_seqlens is None:
        batch = q_num.shape[0]
        if initial_state is None:
            kv_state = torch.zeros(
                batch,
                q_num.shape[2],
                feature_dim,
                value_dim,
                device=q_num.device,
                dtype=torch.float32,
            )
            z_state = torch.zeros(
                batch,
                q_num.shape[2],
                den_feature_dim,
                device=q_num.device,
                dtype=torch.float32,
            )
        else:
            kv_state, z_state = initial_state
            kv_state = kv_state.contiguous().float()
            z_state = z_state.contiguous().float()
            if kv_state.shape != (batch, q_num.shape[2], feature_dim, value_dim):
                raise ValueError("initial_state kv_state shape does not match dense input.")
            if z_state.shape != (batch, q_num.shape[2], den_feature_dim):
                raise ValueError("initial_state k_state shape does not match dense input.")
        out, kv_state, z_state = _run_segment(q_num, k_num, v32, q_den, k_den, kv_state, z_state)
        final_state = (kv_state, z_state) if output_final_state else None
        return out, final_state

    if q_num.shape[0] != 1:
        raise ValueError("Packed varlen mode expects q_prime batch dimension to be 1.")
    num_segments = int(cu_seqlens.numel() - 1)
    if num_segments <= 0:
        raise ValueError("cu_seqlens must contain at least one segment.")

    if initial_state is None:
        kv_init = torch.zeros(
            num_segments,
            q_num.shape[2],
            feature_dim,
            value_dim,
            device=q_num.device,
            dtype=torch.float32,
        )
        z_init = torch.zeros(
            num_segments,
            q_num.shape[2],
            den_feature_dim,
            device=q_num.device,
            dtype=torch.float32,
        )
    else:
        kv_init, z_init = initial_state
        kv_init = kv_init.contiguous().float()
        z_init = z_init.contiguous().float()
        if kv_init.shape != (num_segments, q_num.shape[2], feature_dim, value_dim):
            raise ValueError("initial_state kv_state shape does not match packed varlen input.")
        if z_init.shape != (num_segments, q_num.shape[2], den_feature_dim):
            raise ValueError("initial_state k_state shape does not match packed varlen input.")

    out = torch.empty(
        q_num.shape[0],
        q_num.shape[1],
        q_num.shape[2],
        value_dim,
        device=q_num.device,
        dtype=v.dtype,
    )
    kv_final: list[torch.Tensor] = []
    z_final: list[torch.Tensor] = []
    for seg_idx in range(num_segments):
        start = int(cu_seqlens[seg_idx].item())
        end = int(cu_seqlens[seg_idx + 1].item())
        out_seg, kv_seg, z_seg = _run_segment(
            q_num[:, start:end],
            k_num[:, start:end],
            v32[:, start:end],
            q_den[:, start:end],
            k_den[:, start:end],
            kv_init[seg_idx:seg_idx + 1],
            z_init[seg_idx:seg_idx + 1],
        )
        out[:, start:end] = out_seg
        kv_final.append(kv_seg.squeeze(0))
        z_final.append(z_seg.squeeze(0))
    final_state = None
    if output_final_state:
        final_state = (torch.stack(kv_final, dim=0), torch.stack(z_final, dim=0))
    return out, final_state


class PerformerPlusLinearAttention(nn.Module):
    """
    Causal Performer+ linear attention with:
    1) antithetic orthogonal random features and optional deterministic nodes,
    2) control-variate kernel decomposition (exact constant + linear terms),
    3) selectable state update rule (`sum`, `delta`, or `pdf_delta`/`overwrite`),
    4) optional hybrid numerator kernel in `delta` mode:
       concat([sqrt(alpha) * CV, sqrt(1-alpha) * positive]) to reduce variance,
    5) optional dual-map denominator in `delta` mode (positive denominator map),
    6) optional adaptive forget gate for `sum` updates,
    7) optional jackknife debiasing on the ratio estimator to cancel leading
       O(1/M) finite-feature bias,
    8) optional adaptive landmark sampling:
       replace a subset of random features with learnable deterministic nodes.
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
        performer_adaptive_center_sampling: bool = False,
        performer_adaptive_center_momentum: float = 0.9,
        performer_adaptive_center_log_clip: float = 12.0,
        performer_use_dual_precondition_sampling: bool = False,
        performer_precondition_momentum: float = 0.9,
        performer_precondition_eps: float = 1e-4,
        performer_precondition_log_clip: float = 2.0,
        performer_precondition_mode: str = "diag",
        performer_use_triton: bool = True,
        performer_antithetic_features: bool = True,
        performer_stratified_norm_sampling: bool = False,
        performer_stratified_jitter: bool = True,
        performer_qmc_gaussian_sampling: bool = False,
        performer_qmc_scramble: bool = True,
        performer_use_landmark_sampling: bool = False,
        performer_landmark_ratio: float = 0.25,
        performer_landmark_alpha_init: float = 0.35,
        performer_landmark_in_denominator: bool = True,
        performer_use_projection_ensemble: bool = False,
        performer_projection_ensemble_groups: int = 2,
        performer_use_deterministic_nodes: bool = False,
        performer_deterministic_ratio: float = 0.25,
        performer_qk_l2_norm: bool = True,
        performer_use_beta: bool = False,
        performer_beta_init: float = 1.0,
        performer_use_value_gate: bool = False,
        performer_value_gate_init: float = 0.0,
        performer_use_output_gate: bool = True,
        performer_output_gate_init: float = -1.0,
        performer_use_decay: bool = True,
        performer_decay_init: float = 1.0,
        performer_state_update: str = "sum",
        performer_delta_beta_norm: bool = True,
        performer_delta_beta_norm_eps: float = 1e-3,
        performer_delta_denom_eps: float = 1e-3,
        performer_delta_smooth_denom: bool = False,
        performer_delta_denom_tau: float = 1e-2,
        performer_delta_beta_cap: float = 0.25,
        performer_delta_safe_denom_floor: float = 1e-3,
        performer_delta_decouple_beta: bool = False,
        performer_delta_denominator_update: str = "delta",
        performer_delta_denominator_map: str = "auto",
        performer_delta_denominator_stopgrad: bool = False,
        performer_pdf_delta_denom_stopgrad: bool = True,
        performer_pdf_delta_feature_low_precision: bool = True,
        performer_delta_use_leaky_dplr: bool = False,
        performer_delta_leaky_rho_init: float = 1.0,
        performer_delta_leaky_min_lambda: float = 0.5,
        performer_use_hybrid_numerator: bool = False,
        performer_hybrid_num_ratio: float = 0.25,
        performer_hybrid_alpha_init: float = 0.75,
        performer_use_dual_map: bool = True,
        performer_dual_map_den_ratio: float = 0.25,
        performer_dual_map_row_selection: str = "auto",
        performer_use_layerwise_den_ratio: bool = False,
        performer_layerwise_den_ratio_tau: float = 8.0,
        performer_use_error_feedback_den_ratio: bool = False,
        performer_error_feedback_den_ratio_momentum: float = 0.9,
        performer_error_feedback_den_ratio_gain: float = 0.5,
        performer_use_adaptive_den_mix: bool = False,
        performer_adaptive_den_mix_init: float = 0.0,
        performer_dual_map_low_precision: bool = True,
        performer_use_den_poly_kernel: bool = False,
        performer_den_poly_alpha_init: float = 0.15,
        performer_den_poly_constant: float = 2.0,
        performer_den_poly_ratio: float = 0.5,
        performer_per_layer_projection: bool = True,
        performer_learnable_projection: bool = False,
        performer_learnable_projection_scale: bool = False,
        performer_use_control_variate: bool = True,
        performer_use_adaptive_linear_cv: bool = False,
        performer_adaptive_linear_cv_init: float = 1.0,
        performer_adaptive_linear_cv_eps: float = 1e-4,
        performer_feature_rms_norm: bool = False,
        performer_feature_rms_norm_eps: float = 1e-4,
        performer_feature_pairwise_balance: bool = False,
        performer_feature_pairwise_balance_eps: float = 1e-4,
        performer_feature_pairwise_balance_log_clip: float = 2.0,
        performer_control_variate_exp_clip: float | None = 20.0,
        performer_use_cv_residual_shrinkage: bool = False,
        performer_cv_residual_init: float = 0.75,
        performer_cv_finite_sample_orthogonalize: bool = False,
        performer_cv_finite_sample_orth_eps: float = 1e-4,
        performer_use_jackknife_debias: bool = False,
        performer_jackknife_groups: int = 2,
        performer_jackknife_min_per_group: int = 8,
        performer_use_jackknife_adaptive_shrinkage: bool = False,
        performer_jackknife_shrinkage_eps: float = 1e-5,
        performer_use_second_order_cv: bool = False,
        performer_use_cv_decoupled_second_order: bool = False,
        performer_cv_decoupled_h2_ratio: float = 0.25,
        performer_cv_decoupled_h2_deterministic: bool = False,
        performer_use_cv_decoupled_adaptive_h2_ratio: bool = False,
        performer_cv_decoupled_ratio_ema_momentum: float = 0.9,
        performer_cv_decoupled_ratio_min: float = 0.05,
        performer_cv_decoupled_ratio_max: float = 0.5,
        performer_use_third_order_cv: bool = False,
        performer_cv_split_feature_budget: bool = True,
        performer_use_diag2_term: bool = False,
        performer_diag2_ratio: float = 0.25,
        performer_diag2_alpha_init: float = 0.1,
        performer_dim_aware_kernel_scale: bool = True,
        performer_learnable_kernel_scale: bool = True,
        performer_kernel_scale_init: float = 1.0,
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
        self.adaptive_center_sampling = performer_adaptive_center_sampling
        self.adaptive_center_momentum = performer_adaptive_center_momentum
        self.adaptive_center_log_clip = performer_adaptive_center_log_clip
        self.use_dual_precondition_sampling = performer_use_dual_precondition_sampling
        self.precondition_momentum = performer_precondition_momentum
        self.precondition_eps = performer_precondition_eps
        self.precondition_log_clip = performer_precondition_log_clip
        self.precondition_mode = performer_precondition_mode
        self.use_triton_kernel = performer_use_triton
        self.antithetic_features = performer_antithetic_features
        self.stratified_norm_sampling = performer_stratified_norm_sampling
        self.stratified_jitter = performer_stratified_jitter
        self.qmc_gaussian_sampling = performer_qmc_gaussian_sampling
        self.qmc_scramble = performer_qmc_scramble
        self.use_landmark_sampling = performer_use_landmark_sampling
        self.landmark_ratio = performer_landmark_ratio
        self.landmark_alpha_init = performer_landmark_alpha_init
        self.landmark_in_denominator = performer_landmark_in_denominator
        self.use_projection_ensemble = performer_use_projection_ensemble
        self.projection_ensemble_groups = performer_projection_ensemble_groups
        self.use_deterministic_nodes = performer_use_deterministic_nodes
        self.deterministic_ratio = performer_deterministic_ratio
        self.qk_l2_norm = performer_qk_l2_norm
        self.use_beta = performer_use_beta
        self.use_value_gate = performer_use_value_gate
        self.use_output_gate = performer_use_output_gate
        self.use_decay = performer_use_decay
        self.state_update = performer_state_update
        self.pdf_delta_update = self.state_update in _PDF_DELTA_STATE_UPDATES
        self.delta_beta_norm = performer_delta_beta_norm
        self.delta_beta_norm_eps = performer_delta_beta_norm_eps
        self.delta_denom_eps = performer_delta_denom_eps
        self.delta_smooth_denom = performer_delta_smooth_denom
        self.delta_denom_tau = performer_delta_denom_tau
        self.delta_beta_cap = performer_delta_beta_cap
        self.delta_safe_denom_floor = performer_delta_safe_denom_floor
        self.delta_decouple_beta = performer_delta_decouple_beta
        self.delta_denominator_update = performer_delta_denominator_update
        self.delta_denominator_map = performer_delta_denominator_map
        self.delta_denominator_stopgrad = performer_delta_denominator_stopgrad
        self.pdf_delta_denom_stopgrad = performer_pdf_delta_denom_stopgrad
        self.pdf_delta_feature_low_precision = performer_pdf_delta_feature_low_precision
        self.delta_use_leaky_dplr = performer_delta_use_leaky_dplr
        self.delta_leaky_rho_init = performer_delta_leaky_rho_init
        self.delta_leaky_min_lambda = performer_delta_leaky_min_lambda
        self.use_hybrid_numerator = performer_use_hybrid_numerator
        self.hybrid_num_ratio = performer_hybrid_num_ratio
        self.hybrid_alpha_init = performer_hybrid_alpha_init
        self.use_dual_map = performer_use_dual_map
        self.dual_map_den_ratio = performer_dual_map_den_ratio
        self.dual_map_row_selection = performer_dual_map_row_selection
        self.use_layerwise_den_ratio = performer_use_layerwise_den_ratio
        self.layerwise_den_ratio_tau = performer_layerwise_den_ratio_tau
        self.use_error_feedback_den_ratio = performer_use_error_feedback_den_ratio
        self.error_feedback_den_ratio_momentum = performer_error_feedback_den_ratio_momentum
        self.error_feedback_den_ratio_gain = performer_error_feedback_den_ratio_gain
        self.use_adaptive_den_mix = performer_use_adaptive_den_mix
        self.adaptive_den_mix_init = performer_adaptive_den_mix_init
        self.dual_map_low_precision = performer_dual_map_low_precision
        self.use_den_poly_kernel = performer_use_den_poly_kernel
        self.den_poly_alpha_init = performer_den_poly_alpha_init
        self.den_poly_constant = performer_den_poly_constant
        self.den_poly_ratio = performer_den_poly_ratio
        self.learnable_projection = performer_learnable_projection
        self.learnable_projection_scale = performer_learnable_projection_scale
        self.use_control_variate = performer_use_control_variate
        self.use_adaptive_linear_cv = performer_use_adaptive_linear_cv
        self.adaptive_linear_cv_init = performer_adaptive_linear_cv_init
        self.adaptive_linear_cv_eps = performer_adaptive_linear_cv_eps
        self.feature_rms_norm = performer_feature_rms_norm
        self.feature_rms_norm_eps = performer_feature_rms_norm_eps
        self.feature_pairwise_balance = performer_feature_pairwise_balance
        self.feature_pairwise_balance_eps = performer_feature_pairwise_balance_eps
        self.feature_pairwise_balance_log_clip = performer_feature_pairwise_balance_log_clip
        self.control_variate_exp_clip = performer_control_variate_exp_clip
        self.use_cv_residual_shrinkage = performer_use_cv_residual_shrinkage
        self.cv_residual_init = performer_cv_residual_init
        self.cv_finite_sample_orthogonalize = performer_cv_finite_sample_orthogonalize
        self.cv_finite_sample_orth_eps = performer_cv_finite_sample_orth_eps
        self.use_jackknife_debias = performer_use_jackknife_debias
        self.jackknife_groups = performer_jackknife_groups
        self.jackknife_min_per_group = performer_jackknife_min_per_group
        self.use_jackknife_adaptive_shrinkage = performer_use_jackknife_adaptive_shrinkage
        self.jackknife_shrinkage_eps = performer_jackknife_shrinkage_eps
        self.use_second_order_cv = performer_use_second_order_cv
        self.use_cv_decoupled_second_order = performer_use_cv_decoupled_second_order
        self.cv_decoupled_h2_ratio = performer_cv_decoupled_h2_ratio
        self.cv_decoupled_h2_deterministic = performer_cv_decoupled_h2_deterministic
        self.use_cv_decoupled_adaptive_h2_ratio = performer_use_cv_decoupled_adaptive_h2_ratio
        self.cv_decoupled_ratio_ema_momentum = performer_cv_decoupled_ratio_ema_momentum
        self.cv_decoupled_ratio_min = performer_cv_decoupled_ratio_min
        self.cv_decoupled_ratio_max = performer_cv_decoupled_ratio_max
        self.use_third_order_cv = performer_use_third_order_cv
        self.cv_split_feature_budget = performer_cv_split_feature_budget
        self.use_diag2_term = performer_use_diag2_term
        self.diag2_ratio = performer_diag2_ratio
        self.diag2_alpha_init = performer_diag2_alpha_init
        self.dim_aware_kernel_scale = performer_dim_aware_kernel_scale
        self.learnable_kernel_scale = performer_learnable_kernel_scale
        self.kernel_scale_init = performer_kernel_scale_init
        self.num_features = (
            performer_nb_features
            if performer_nb_features is not None
            else self.head_k_dim
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
        if self.use_deterministic_nodes and (self.deterministic_ratio <= 0 or self.deterministic_ratio > 1):
            raise ValueError("performer_deterministic_ratio must be in (0, 1].")
        if self.landmark_ratio <= 0 or self.landmark_ratio > 1:
            raise ValueError("performer_landmark_ratio must be in (0, 1].")
        if self.landmark_alpha_init <= 0 or self.landmark_alpha_init >= 1:
            raise ValueError("performer_landmark_alpha_init must be in (0, 1).")
        if self.projection_ensemble_groups < 1:
            raise ValueError("performer_projection_ensemble_groups must be >= 1.")
        if self.adaptive_center_momentum < 0 or self.adaptive_center_momentum >= 1:
            raise ValueError("performer_adaptive_center_momentum must be in [0, 1).")
        if self.adaptive_center_log_clip < 0:
            raise ValueError("performer_adaptive_center_log_clip must be >= 0.")
        if self.precondition_momentum < 0 or self.precondition_momentum >= 1:
            raise ValueError("performer_precondition_momentum must be in [0, 1).")
        if self.precondition_eps <= 0:
            raise ValueError("performer_precondition_eps must be > 0.")
        if self.precondition_log_clip < 0:
            raise ValueError("performer_precondition_log_clip must be >= 0.")
        if self.precondition_mode not in ("diag", "full"):
            raise ValueError("performer_precondition_mode must be one of ('diag', 'full').")
        if self.state_update not in ("sum", "delta", "pdf_delta", "overwrite"):
            raise ValueError(
                f"performer_state_update={self.state_update} is not supported. "
                "Expected one of ('sum', 'delta', 'pdf_delta', 'overwrite').",
            )
        if self.control_variate_exp_clip is not None and self.control_variate_exp_clip <= 0:
            raise ValueError(
                "performer_control_variate_exp_clip must be > 0 when provided.",
            )
        if self.feature_rms_norm_eps <= 0:
            raise ValueError("performer_feature_rms_norm_eps must be > 0.")
        if self.feature_pairwise_balance_eps <= 0:
            raise ValueError("performer_feature_pairwise_balance_eps must be > 0.")
        if self.feature_pairwise_balance_log_clip < 0:
            raise ValueError("performer_feature_pairwise_balance_log_clip must be >= 0.")
        if self.cv_residual_init <= 0 or self.cv_residual_init > 1:
            raise ValueError("performer_cv_residual_init must be in (0, 1].")
        if self.adaptive_linear_cv_eps <= 0:
            raise ValueError("performer_adaptive_linear_cv_eps must be > 0.")
        if self.use_adaptive_linear_cv:
            if self.adaptive_linear_cv_init <= 0 or self.adaptive_linear_cv_init >= 2:
                raise ValueError("performer_adaptive_linear_cv_init must be in (0, 2).")
        if self.cv_finite_sample_orth_eps <= 0:
            raise ValueError("performer_cv_finite_sample_orth_eps must be > 0.")
        if self.jackknife_groups < 2:
            raise ValueError("performer_jackknife_groups must be >= 2.")
        if self.jackknife_min_per_group < 1:
            raise ValueError("performer_jackknife_min_per_group must be >= 1.")
        if self.jackknife_shrinkage_eps <= 0:
            raise ValueError("performer_jackknife_shrinkage_eps must be > 0.")
        if self.cv_decoupled_h2_ratio <= 0 or self.cv_decoupled_h2_ratio >= 1:
            raise ValueError("performer_cv_decoupled_h2_ratio must be in (0, 1).")
        if self.cv_decoupled_ratio_ema_momentum < 0 or self.cv_decoupled_ratio_ema_momentum >= 1:
            raise ValueError("performer_cv_decoupled_ratio_ema_momentum must be in [0, 1).")
        if self.cv_decoupled_ratio_min <= 0 or self.cv_decoupled_ratio_max >= 1:
            raise ValueError("performer_cv_decoupled_ratio_min/max must lie in (0, 1).")
        if self.cv_decoupled_ratio_min >= self.cv_decoupled_ratio_max:
            raise ValueError("performer_cv_decoupled_ratio_min must be < performer_cv_decoupled_ratio_max.")
        if self.diag2_ratio <= 0 or self.diag2_ratio > 1:
            raise ValueError("performer_diag2_ratio must be in (0, 1].")
        if self.diag2_alpha_init <= 0 or self.diag2_alpha_init >= 1:
            raise ValueError("performer_diag2_alpha_init must be in (0, 1).")
        if self.kernel_scale_init <= 0:
            raise ValueError("performer_kernel_scale_init must be > 0.")
        if self.delta_beta_norm_eps <= 0:
            raise ValueError("performer_delta_beta_norm_eps must be > 0.")
        if self.delta_denom_eps <= 0:
            raise ValueError("performer_delta_denom_eps must be > 0.")
        if self.delta_denom_tau <= 0:
            raise ValueError("performer_delta_denom_tau must be > 0.")
        if self.delta_beta_cap <= 0 or self.delta_beta_cap > 2:
            raise ValueError("performer_delta_beta_cap must be in (0, 2].")
        if self.delta_safe_denom_floor <= 0:
            raise ValueError("performer_delta_safe_denom_floor must be > 0.")
        if self.delta_denominator_update not in ("delta", "sum"):
            raise ValueError("performer_delta_denominator_update must be one of ('delta', 'sum').")
        if self.delta_denominator_map not in (
            "auto",
            "dual_softmax",
            "abs",
            "softplus",
            "numerator",
            "positive_linear",
        ):
            raise ValueError(
                "performer_delta_denominator_map must be one of "
                "('auto', 'dual_softmax', 'abs', 'softplus', 'numerator', 'positive_linear').",
            )
        if self.delta_leaky_rho_init <= 0:
            raise ValueError("performer_delta_leaky_rho_init must be > 0.")
        if self.delta_leaky_min_lambda <= 0 or self.delta_leaky_min_lambda > 1:
            raise ValueError("performer_delta_leaky_min_lambda must be in (0, 1].")
        if self.hybrid_num_ratio <= 0 or self.hybrid_num_ratio > 1:
            raise ValueError("performer_hybrid_num_ratio must be in (0, 1].")
        if self.hybrid_alpha_init <= 0 or self.hybrid_alpha_init >= 1:
            raise ValueError("performer_hybrid_alpha_init must be in (0, 1).")
        if self.dual_map_den_ratio <= 0 or self.dual_map_den_ratio > 1:
            raise ValueError("performer_dual_map_den_ratio must be in (0, 1].")
        if self.dual_map_row_selection not in ("auto", "head", "strided", "antithetic_balanced"):
            raise ValueError(
                "performer_dual_map_row_selection must be one of "
                "('auto', 'head', 'strided', 'antithetic_balanced')."
            )
        if self.layerwise_den_ratio_tau <= 0:
            raise ValueError("performer_layerwise_den_ratio_tau must be > 0.")
        if self.error_feedback_den_ratio_momentum < 0 or self.error_feedback_den_ratio_momentum >= 1:
            raise ValueError("performer_error_feedback_den_ratio_momentum must be in [0, 1).")
        if self.error_feedback_den_ratio_gain <= 0:
            raise ValueError("performer_error_feedback_den_ratio_gain must be > 0.")
        if self.adaptive_den_mix_init < 0 or self.adaptive_den_mix_init > 1:
            raise ValueError("performer_adaptive_den_mix_init must be in [0, 1].")
        if self.den_poly_alpha_init <= 0 or self.den_poly_alpha_init >= 1:
            raise ValueError("performer_den_poly_alpha_init must be in (0, 1).")
        if self.den_poly_constant <= 0:
            raise ValueError("performer_den_poly_constant must be > 0.")
        if self.den_poly_ratio <= 0 or self.den_poly_ratio > 1:
            raise ValueError("performer_den_poly_ratio must be in (0, 1].")
        if self.state_update == "delta" and self.use_decay:
            warnings.warn(
                "`performer_use_decay=True` is ignored when performer_state_update='delta'. "
                "Delta update does not support explicit exponential decay.",
            )
            self.use_decay = False
        if self.pdf_delta_update and self.use_decay:
            warnings.warn(
                "`performer_use_decay=True` is ignored when performer_state_update in "
                "{'pdf_delta','overwrite'}. The PDF delta update uses its own additive denominator state.",
            )
            self.use_decay = False
        if self.state_update != "delta" and self.delta_denominator_map != "auto":
            warnings.warn(
                "`performer_delta_denominator_map` is ignored unless performer_state_update='delta'. Resetting to 'auto'.",
            )
            self.delta_denominator_map = "auto"
        if self.state_update != "delta" and self.delta_use_leaky_dplr:
            self.delta_use_leaky_dplr = False
        if self.delta_denominator_map == "dual_softmax" and not self.use_dual_map:
            warnings.warn(
                "`performer_delta_denominator_map='dual_softmax'` requires `performer_use_dual_map=True`. Enabling it.",
            )
            self.use_dual_map = True
        if self.use_hybrid_numerator and not self.use_control_variate:
            warnings.warn(
                "`performer_use_hybrid_numerator=True` is ignored when control variate is disabled.",
            )
            self.use_hybrid_numerator = False
        if self.use_cv_residual_shrinkage and not self.use_control_variate:
            warnings.warn(
                "`performer_use_cv_residual_shrinkage=True` is ignored when control variate is disabled.",
            )
            self.use_cv_residual_shrinkage = False
        if self.use_adaptive_linear_cv and not self.use_control_variate:
            warnings.warn(
                "`performer_use_adaptive_linear_cv=True` is ignored when control variate is disabled.",
            )
            self.use_adaptive_linear_cv = False
        if self.use_second_order_cv and not self.use_control_variate:
            warnings.warn(
                "`performer_use_second_order_cv=True` is ignored when control variate is disabled.",
            )
            self.use_second_order_cv = False
        if self.use_cv_decoupled_second_order and not self.use_second_order_cv:
            warnings.warn(
                "`performer_use_cv_decoupled_second_order=True` requires `performer_use_second_order_cv=True`. Disabling it.",
            )
            self.use_cv_decoupled_second_order = False
        if self.cv_decoupled_h2_deterministic and not self.use_cv_decoupled_second_order:
            warnings.warn(
                "`performer_cv_decoupled_h2_deterministic=True` requires `performer_use_cv_decoupled_second_order=True`. Disabling it.",
            )
            self.cv_decoupled_h2_deterministic = False
        if self.use_cv_decoupled_adaptive_h2_ratio and not self.use_cv_decoupled_second_order:
            warnings.warn(
                "`performer_use_cv_decoupled_adaptive_h2_ratio=True` requires `performer_use_cv_decoupled_second_order=True`. Disabling it.",
            )
            self.use_cv_decoupled_adaptive_h2_ratio = False
        if self.use_third_order_cv and not self.use_control_variate:
            warnings.warn(
                "`performer_use_third_order_cv=True` is ignored when control variate is disabled.",
            )
            self.use_third_order_cv = False
        if self.use_third_order_cv and not self.use_second_order_cv:
            warnings.warn(
                "`performer_use_third_order_cv=True` implies second-order control variate. Enabling `performer_use_second_order_cv`.",
            )
            self.use_second_order_cv = True
        if self.use_cv_decoupled_second_order and self.use_third_order_cv:
            warnings.warn(
                "`performer_use_cv_decoupled_second_order=True` is incompatible with third-order CV. Disabling it.",
            )
            self.use_cv_decoupled_second_order = False
        if self.use_cv_decoupled_second_order and self.use_projection_ensemble:
            warnings.warn(
                "`performer_use_cv_decoupled_second_order=True` is incompatible with projection ensemble. Disabling it.",
            )
            self.use_cv_decoupled_second_order = False
        if self.use_adaptive_linear_cv and (self.use_second_order_cv or self.use_third_order_cv):
            warnings.warn(
                "`performer_use_adaptive_linear_cv=True` is currently only applied to first-order CV. Disabling it for higher-order CV.",
            )
            self.use_adaptive_linear_cv = False
        if self.use_cv_decoupled_second_order and not self.cv_split_feature_budget:
            warnings.warn(
                "`performer_use_cv_decoupled_second_order=True` currently requires `performer_cv_split_feature_budget=True`. Enabling split budget.",
            )
            self.cv_split_feature_budget = True
        if self.use_cv_decoupled_second_order and self.cv_finite_sample_orthogonalize:
            warnings.warn(
                "`performer_cv_finite_sample_orthogonalize=True` is ignored for decoupled CV2 path.",
            )
            self.cv_finite_sample_orthogonalize = False
        if self.use_diag2_term and not self.use_control_variate:
            warnings.warn(
                "`performer_use_diag2_term=True` is ignored when control variate is disabled.",
            )
            self.use_diag2_term = False
        if self.use_den_poly_kernel and self.state_update != "delta":
            warnings.warn(
                "`performer_use_den_poly_kernel=True` is ignored unless performer_state_update='delta'.",
            )
            self.use_den_poly_kernel = False
        if self.use_den_poly_kernel and not self.use_dual_map:
            warnings.warn(
                "`performer_use_den_poly_kernel=True` is ignored unless performer_use_dual_map=True.",
            )
            self.use_den_poly_kernel = False
        if self.use_den_poly_kernel and self.delta_denominator_map in ("abs", "softplus", "numerator", "positive_linear"):
            warnings.warn(
                "`performer_use_den_poly_kernel=True` is ignored for performer_delta_denominator_map in "
                "{'abs','softplus','numerator','positive_linear'}.",
            )
            self.use_den_poly_kernel = False
        if self.use_den_poly_kernel and not self.qk_l2_norm:
            warnings.warn(
                "`performer_use_den_poly_kernel=True` requires performer_qk_l2_norm=True for positivity assumptions.",
            )
            self.use_den_poly_kernel = False
        if self.use_adaptive_den_mix and not self.use_dual_map:
            warnings.warn(
                "`performer_use_adaptive_den_mix=True` requires performer_use_dual_map=True. Disabling it.",
            )
            self.use_adaptive_den_mix = False
        if self.use_adaptive_den_mix and self.dual_map_den_ratio >= 1.0:
            warnings.warn(
                "`performer_use_adaptive_den_mix=True` is only effective when performer_dual_map_den_ratio<1. Disabling it.",
            )
            self.use_adaptive_den_mix = False
        if self.use_adaptive_den_mix and self.use_den_poly_kernel:
            warnings.warn(
                "`performer_use_adaptive_den_mix=True` is incompatible with performer_use_den_poly_kernel=True. Disabling adaptive mix.",
            )
            self.use_adaptive_den_mix = False
        if self.delta_denominator_map == "positive_linear" and not self.qk_l2_norm:
            warnings.warn(
                "`performer_delta_denominator_map='positive_linear'` is most reliable with performer_qk_l2_norm=True.",
            )
        if self.use_landmark_sampling and self.num_features < 2:
            warnings.warn(
                "`performer_use_landmark_sampling=True` requires performer_nb_features>=2. Disabling it.",
            )
            self.use_landmark_sampling = False
        if self.use_cv_decoupled_second_order and self.num_features < 2:
            warnings.warn(
                "`performer_use_cv_decoupled_second_order=True` requires performer_nb_features>=2. Disabling it.",
            )
            self.use_cv_decoupled_second_order = False
        if self.use_landmark_sampling and self.landmark_in_denominator and self.use_den_poly_kernel:
            warnings.warn(
                "`performer_landmark_in_denominator=True` is ignored when performer_use_den_poly_kernel=True.",
            )
            self.landmark_in_denominator = False
        if self.use_adaptive_den_mix and self.use_landmark_sampling and self.landmark_in_denominator:
            warnings.warn(
                "`performer_use_adaptive_den_mix=True` is incompatible with landmark denominator mixing. Disabling adaptive mix.",
            )
            self.use_adaptive_den_mix = False
        if self.use_layerwise_den_ratio and self.dual_map_den_ratio >= 1.0:
            warnings.warn(
                "`performer_use_layerwise_den_ratio=True` is only effective when performer_dual_map_den_ratio<1. Disabling it.",
            )
            self.use_layerwise_den_ratio = False
        if self.use_error_feedback_den_ratio and not self.use_dual_map:
            warnings.warn(
                "`performer_use_error_feedback_den_ratio=True` requires performer_use_dual_map=True. Disabling it.",
            )
            self.use_error_feedback_den_ratio = False
        if self.use_error_feedback_den_ratio and self.dual_map_den_ratio >= 1.0:
            warnings.warn(
                "`performer_use_error_feedback_den_ratio=True` is only effective when performer_dual_map_den_ratio<1. Disabling it.",
            )
            self.use_error_feedback_den_ratio = False
        if self.use_error_feedback_den_ratio and self.use_layerwise_den_ratio:
            warnings.warn(
                "`performer_use_error_feedback_den_ratio=True` overrides layerwise den-ratio schedule. Disabling layerwise schedule.",
            )
            self.use_layerwise_den_ratio = False
        if self.stratified_norm_sampling and self.ortho_scaling != 0:
            warnings.warn(
                "`performer_stratified_norm_sampling=True` only affects performer_ortho_scaling=0. Disabling it.",
            )
            self.stratified_norm_sampling = False
        if self.stratified_jitter and not self.stratified_norm_sampling:
            self.stratified_jitter = False
        if self.use_projection_ensemble and self.projection_ensemble_groups == 1:
            self.use_projection_ensemble = False
        if self.use_dual_precondition_sampling and self.adaptive_center_sampling:
            warnings.warn(
                "`performer_use_dual_precondition_sampling=True` currently uses an asymmetric q/k transform "
                "and is incompatible with shared adaptive-center correction. Disabling adaptive center.",
            )
            self.adaptive_center_sampling = False
        if self.use_projection_ensemble and self.projection_ensemble_groups > self.num_features:
            warnings.warn(
                "`performer_projection_ensemble_groups` exceeds `performer_nb_features`; clamping to feature count.",
            )
            self.projection_ensemble_groups = self.num_features
            if self.projection_ensemble_groups <= 1:
                self.use_projection_ensemble = False
        if self.use_jackknife_debias and self.jackknife_groups > self.num_features:
            warnings.warn(
                "`performer_jackknife_groups` exceeds `performer_nb_features`; clamping to feature count.",
            )
            self.jackknife_groups = self.num_features
            if self.jackknife_groups < 2:
                self.use_jackknife_debias = False
        if self.use_jackknife_adaptive_shrinkage and not self.use_jackknife_debias:
            warnings.warn(
                "`performer_use_jackknife_adaptive_shrinkage=True` requires `performer_use_jackknife_debias=True`. Disabling it.",
            )
            self.use_jackknife_adaptive_shrinkage = False
        if self.dim_aware_kernel_scale and not self.qk_l2_norm:
            warnings.warn(
                "`performer_dim_aware_kernel_scale=True` is disabled when performer_qk_l2_norm=False.",
            )
            self.dim_aware_kernel_scale = False
        if self.pdf_delta_update and self.use_beta:
            warnings.warn(
                "`performer_use_beta=True` is ignored for performer_state_update in "
                "{'pdf_delta','overwrite'}. The PDF delta rule uses a closed-form correction write.",
            )
            self.use_beta = False
        if self.pdf_delta_update and self.use_control_variate:
            warnings.warn(
                "`performer_use_control_variate=True` is ignored for performer_state_update in "
                "{'pdf_delta','overwrite'}. This mode uses the plain shared FAVOR+ map from performer+delta.pdf.",
            )
            self.use_control_variate = False
        if self.pdf_delta_update:
            self.use_adaptive_linear_cv = False
            self.use_second_order_cv = False
            self.use_third_order_cv = False
            self.use_cv_decoupled_second_order = False
            self.use_cv_decoupled_adaptive_h2_ratio = False
            self.use_diag2_term = False
            self.use_hybrid_numerator = False
        if self.pdf_delta_update and self.use_projection_ensemble:
            warnings.warn(
                "`performer_use_projection_ensemble=True` is ignored for performer_state_update in "
                "{'pdf_delta','overwrite'}.",
            )
            self.use_projection_ensemble = False
        if self.pdf_delta_update and self.use_landmark_sampling:
            warnings.warn(
                "`performer_use_landmark_sampling=True` is ignored for performer_state_update in "
                "{'pdf_delta','overwrite'}.",
            )
            self.use_landmark_sampling = False
        if self.pdf_delta_update and self.use_dual_map:
            warnings.warn(
                "`performer_use_dual_map=True` is ignored for performer_state_update in "
                "{'pdf_delta','overwrite'}. This mode shares a single positive feature map for numerator/denominator.",
            )
            self.use_dual_map = False
        if self.pdf_delta_update and self.use_jackknife_debias:
            warnings.warn(
                "`performer_use_jackknife_debias=True` is ignored for performer_state_update in "
                "{'pdf_delta','overwrite'}.",
            )
            self.use_jackknife_debias = False
            self.use_jackknife_adaptive_shrinkage = False
        if self.pdf_delta_update and self.delta_use_leaky_dplr:
            warnings.warn(
                "`performer_delta_use_leaky_dplr=True` is ignored for performer_state_update in "
                "{'pdf_delta','overwrite'}.",
            )
            self.delta_use_leaky_dplr = False
        if self.pdf_delta_update and self.use_den_poly_kernel:
            warnings.warn(
                "`performer_use_den_poly_kernel=True` is ignored for performer_state_update in "
                "{'pdf_delta','overwrite'}.",
            )
            self.use_den_poly_kernel = False
        if self.pdf_delta_update and self.use_adaptive_den_mix:
            warnings.warn(
                "`performer_use_adaptive_den_mix=True` is ignored for performer_state_update in "
                "{'pdf_delta','overwrite'}.",
            )
            self.use_adaptive_den_mix = False
        if self.pdf_delta_update and self.use_error_feedback_den_ratio:
            warnings.warn(
                "`performer_use_error_feedback_den_ratio=True` is ignored for performer_state_update in "
                "{'pdf_delta','overwrite'}.",
            )
            self.use_error_feedback_den_ratio = False
        if self.pdf_delta_update and self.use_layerwise_den_ratio:
            warnings.warn(
                "`performer_use_layerwise_den_ratio=True` is ignored for performer_state_update in "
                "{'pdf_delta','overwrite'}.",
            )
            self.use_layerwise_den_ratio = False
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

        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
            self.beta_bias = nn.Parameter(
                torch.full((self.num_v_heads,), float(performer_beta_init), dtype=torch.float32)
            )
            self.beta_bias._no_weight_decay = True
        else:
            self.b_proj = None
            self.beta_bias = None

        if self.state_update == "delta" and self.delta_use_leaky_dplr:
            self.delta_leaky_rho_log = nn.Parameter(
                torch.full((self.num_v_heads,), math.log(float(self.delta_leaky_rho_init)), dtype=torch.float32)
            )
            self.delta_leaky_rho_log._no_weight_decay = True
        else:
            self.delta_leaky_rho_log = None

        if self.use_value_gate:
            self.v_gate_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
            self.v_gate_bias = nn.Parameter(
                torch.full((self.num_v_heads,), float(performer_value_gate_init), dtype=torch.float32)
            )
            self.v_gate_bias._no_weight_decay = True
        else:
            self.v_gate_proj = None
            self.v_gate_bias = None

        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        else:
            self.g_proj = None

        if self.use_hybrid_numerator:
            alpha_init = torch.full(
                (self.num_heads,),
                float(self.hybrid_alpha_init),
                dtype=torch.float32,
            )
            alpha_init = alpha_init.clamp(1e-4, 1 - 1e-4)
            self.hybrid_alpha_logit = nn.Parameter(torch.logit(alpha_init))
        else:
            self.hybrid_alpha_logit = None

        if self.use_den_poly_kernel:
            den_alpha_init = torch.full(
                (self.num_heads,),
                float(self.den_poly_alpha_init),
                dtype=torch.float32,
            )
            den_alpha_init = den_alpha_init.clamp(1e-4, 1 - 1e-4)
            self.den_poly_alpha_logit = nn.Parameter(torch.logit(den_alpha_init))
        else:
            self.den_poly_alpha_logit = None

        if self.use_adaptive_den_mix:
            den_mix_init = torch.full(
                (self.num_heads,),
                float(self.adaptive_den_mix_init),
                dtype=torch.float32,
            ).clamp_(1e-4, 1 - 1e-4)
            self.adaptive_den_mix_logit = nn.Parameter(torch.logit(den_mix_init))
        else:
            self.adaptive_den_mix_logit = None

        if self.use_cv_residual_shrinkage:
            cv_gamma_init = torch.full(
                (self.num_heads,),
                float(self.cv_residual_init),
                dtype=torch.float32,
            )
            cv_gamma_init = cv_gamma_init.clamp(1e-4, 1.0 - 1e-4)
            self.cv_residual_logit = nn.Parameter(torch.logit(cv_gamma_init))
        else:
            self.cv_residual_logit = None

        if self.use_adaptive_linear_cv:
            # Parameterize b in (0, 2) as b = 2 * sigmoid(logit).
            # b=1 recovers the standard first-order control variate.
            cv_linear_init = torch.full(
                (self.num_heads,),
                float(self.adaptive_linear_cv_init) * 0.5,
                dtype=torch.float32,
            )
            cv_linear_init = cv_linear_init.clamp(1e-4, 1.0 - 1e-4)
            self.cv_linear_coef_logit = nn.Parameter(torch.logit(cv_linear_init))
        else:
            self.cv_linear_coef_logit = None

        if self.use_diag2_term:
            diag2_alpha_init = torch.full(
                (self.num_heads,),
                float(self.diag2_alpha_init),
                dtype=torch.float32,
            )
            diag2_alpha_init = diag2_alpha_init.clamp(1e-4, 1 - 1e-4)
            self.diag2_alpha_logit = nn.Parameter(torch.logit(diag2_alpha_init))
        else:
            self.diag2_alpha_logit = None

        if self.use_decay:
            self.decay_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
            self.decay_bias = nn.Parameter(
                torch.full((self.num_v_heads,), float(performer_decay_init), dtype=torch.float32)
            )
            self.decay_bias._no_weight_decay = True
        else:
            self.decay_proj = None
            self.decay_bias = None

        if self.use_output_gate:
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Quarter-power base scaling is a stable dimension-aware compensation for
        # FAVOR+ with orthogonal projections and q/k L2 normalization.
        kernel_scale_base = (float(self.head_k_dim) ** 0.25) if self.dim_aware_kernel_scale else 1.0
        if self.learnable_kernel_scale:
            self.kernel_scale_log = nn.Parameter(
                torch.full((self.num_heads,), math.log(float(self.kernel_scale_init)), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "kernel_scale",
                torch.full((self.num_heads,), float(self.kernel_scale_init), dtype=torch.float32),
            )
        self.register_buffer(
            "kernel_scale_base",
            torch.full((self.num_heads,), float(kernel_scale_base), dtype=torch.float32),
        )
        self.register_buffer(
            "adaptive_center_ema",
            torch.zeros(self.num_heads, self.head_k_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "adaptive_center_initialized",
            torch.tensor(False, dtype=torch.bool),
        )
        self.register_buffer(
            "precondition_log_scale_ema",
            torch.zeros(self.num_heads, self.head_k_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "precondition_cov_q_ema",
            torch.eye(self.head_k_dim, dtype=torch.float32).unsqueeze(0).repeat(self.num_heads, 1, 1),
        )
        self.register_buffer(
            "precondition_cov_k_ema",
            torch.eye(self.head_k_dim, dtype=torch.float32).unsqueeze(0).repeat(self.num_heads, 1, 1),
        )
        self.register_buffer(
            "precondition_initialized",
            torch.tensor(False, dtype=torch.bool),
        )

        projection_init = torch.empty(
            self.num_heads,
            self.num_features,
            self.head_k_dim,
            dtype=torch.float32,
        )
        if self.learnable_projection:
            self.projection_matrix = nn.Parameter(projection_init)
            if self.learnable_projection_scale:
                self.projection_log_scale = nn.Parameter(
                    torch.zeros(self.num_heads, self.num_features, dtype=torch.float32)
                )
            else:
                self.register_buffer(
                    "projection_row_scale",
                    torch.ones(self.num_heads, self.num_features, dtype=torch.float32),
                )
        else:
            self.register_buffer("projection_matrix", projection_init)
        self._redraw_projection_matrix()
        self.register_buffer(
            "cv_decoupled_h2_ratio_ema",
            torch.tensor(float(self.cv_decoupled_h2_ratio), dtype=torch.float32),
        )
        self.register_buffer(
            "error_feedback_den_ratio_ema",
            torch.tensor(float(self.dual_map_den_ratio), dtype=torch.float32),
        )

        if self.use_landmark_sampling:
            self.landmark_features = min(
                self.num_features - 1,
                max(1, int(round(self.num_features * self.landmark_ratio))),
            )
            self.landmark_rf_features = max(1, self.num_features - self.landmark_features)
            landmark_init = torch.randn(
                self.num_heads,
                self.landmark_features,
                self.head_k_dim,
                dtype=torch.float32,
            ) / math.sqrt(float(self.head_k_dim))
            self.landmark_projection = nn.Parameter(landmark_init)
            landmark_alpha = torch.full(
                (self.num_heads,),
                float(self.landmark_alpha_init),
                dtype=torch.float32,
            ).clamp_(1e-4, 1 - 1e-4)
            self.landmark_alpha_logit = nn.Parameter(torch.logit(landmark_alpha))
        else:
            self.landmark_features = 0
            self.landmark_rf_features = self.num_features
            self.landmark_projection = None
            self.landmark_alpha_logit = None

        # Lightweight approximation-error observability.
        self.enable_error_observability = bool(kwargs.pop("performer_enable_error_observability", False))
        self.error_observe_interval = max(1, int(kwargs.pop("performer_error_observe_interval", 10)))
        self.error_observe_max_tokens = max(4, int(kwargs.pop("performer_error_observe_max_tokens", 32)))
        self.error_observe_max_heads = max(1, int(kwargs.pop("performer_error_observe_max_heads", 2)))
        self.last_error_stats: dict[str, float] = {}
        self._error_observe_counter = 0

    @staticmethod
    def _split_feature_ranges(feature_dim: int, groups: int) -> list[tuple[int, int]]:
        groups = max(1, min(int(groups), int(feature_dim)))
        base = feature_dim // groups
        rem = feature_dim % groups
        ranges: list[tuple[int, int]] = []
        start = 0
        for group_idx in range(groups):
            size = base + (1 if group_idx < rem else 0)
            end = start + size
            ranges.append((start, end))
            start = end
        return ranges

    @staticmethod
    def _slice_recurrent_state(
        recurrent_state: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        num_start: int,
        num_end: int,
        den_start: int,
        den_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if recurrent_state is None:
            return None
        kv_state, k_state = recurrent_state
        kv_chunk = kv_state[:, :, num_start:num_end, :].contiguous()
        k_chunk = k_state[:, :, den_start:den_end].contiguous()
        return (kv_chunk, k_chunk)

    @staticmethod
    def _expand_head_vector(vec: torch.Tensor, target_heads: int, *, name: str) -> torch.Tensor:
        if vec.shape[0] == target_heads:
            return vec
        if target_heads % vec.shape[0] != 0:
            raise ValueError(
                f"{name} head mismatch: target has {target_heads} heads but vector has {vec.shape[0]}."
            )
        groups = target_heads // vec.shape[0]
        return repeat(vec, 'h -> (h g)', g=groups)

    def _select_dual_map_rows(
        self,
        projection_matrix: torch.Tensor,
        num_rows: int,
    ) -> torch.Tensor:
        m = int(projection_matrix.shape[1])
        if num_rows >= m:
            return projection_matrix
        mode = self.dual_map_row_selection
        if mode == "auto":
            if self.antithetic_features and (m % 2 == 0) and num_rows >= 2:
                mode = "antithetic_balanced"
            else:
                mode = "strided"

        device = projection_matrix.device
        if mode == "head":
            idx = torch.arange(num_rows, device=device, dtype=torch.long)
            return projection_matrix.index_select(1, idx)

        if mode == "strided":
            # Uniformly cover the full row set instead of taking only the first rows.
            idx = torch.div(
                torch.arange(num_rows, device=device, dtype=torch.long) * m,
                num_rows,
                rounding_mode="floor",
            )
            return projection_matrix.index_select(1, idx)

        # antithetic_balanced
        if mode == "antithetic_balanced" and (m % 2 == 0) and num_rows >= 2:
            half = m // 2
            pair_count = num_rows // 2
            if pair_count > 0:
                pos_idx = torch.div(
                    torch.arange(pair_count, device=device, dtype=torch.long) * half,
                    pair_count,
                    rounding_mode="floor",
                )
                neg_idx = pos_idx + half
                idx = torch.cat([pos_idx, neg_idx], dim=0)
            else:
                idx = torch.empty(0, device=device, dtype=torch.long)
            if num_rows % 2 == 1:
                extra = torch.tensor([half // 2], device=device, dtype=torch.long)
                idx = torch.cat([idx, extra], dim=0)
            return projection_matrix.index_select(1, idx)

        # Fallback to strided when antithetic assumptions do not hold.
        idx = torch.div(
            torch.arange(num_rows, device=device, dtype=torch.long) * m,
            num_rows,
            rounding_mode="floor",
        )
        return projection_matrix.index_select(1, idx)

    @staticmethod
    def _pairwise_balance_features(
        q_feat: torch.Tensor,
        k_feat: torch.Tensor,
        *,
        eps: float,
        log_clip: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Channel-wise preconditioning that preserves q·k exactly:
        #   q' = q * s, k' = k / s.
        q32 = q_feat.float()
        k32 = k_feat.float()
        q_rms = (q32.square().mean(dim=(0, 1), keepdim=True) + float(eps)).sqrt().detach()
        k_rms = (k32.square().mean(dim=(0, 1), keepdim=True) + float(eps)).sqrt().detach()
        log_scale = 0.5 * (k_rms.log() - q_rms.log())
        if log_clip > 0:
            log_scale = log_scale.clamp(min=-float(log_clip), max=float(log_clip))
        scale = log_scale.exp()
        q_bal = (q32 * scale).to(q_feat.dtype)
        k_bal = (k32 / scale).to(k_feat.dtype)
        return q_bal, k_bal

    @torch.no_grad()
    def _update_adaptive_center(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Head-wise shared center: c_h ~= 0.5 * (E[q_h] + E[k_h]).
        batch_center = 0.5 * (q.float().mean(dim=(0, 1)) + k.float().mean(dim=(0, 1)))
        if not bool(self.adaptive_center_initialized.item()):
            self.adaptive_center_ema.copy_(batch_center)
            self.adaptive_center_initialized.fill_(True)
        elif self.training:
            momentum = float(self.adaptive_center_momentum)
            self.adaptive_center_ema.mul_(momentum).add_(batch_center * (1.0 - momentum))
        return self.adaptive_center_ema

    @torch.no_grad()
    def _update_dual_precondition_scale(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if self.precondition_mode != "diag":
            raise RuntimeError("_update_dual_precondition_scale is only valid when performer_precondition_mode='diag'.")
        # Variance-aware kernel-preserving preconditioner:
        #   q' = P q, k' = P^{-1} k, so <q', k'> = <q, k>.
        # For diagonal P=diag(p_i), minimizing E||Pq||^2 + E||P^{-1}k||^2 gives
        #   p_i = (E[k_i^2] / E[q_i^2])^(1/4).
        q_second = q.float().square().mean(dim=(0, 1)).clamp_min(float(self.precondition_eps))
        k_second = k.float().square().mean(dim=(0, 1)).clamp_min(float(self.precondition_eps))
        target_log = 0.25 * (k_second.log() - q_second.log())
        if self.precondition_log_clip > 0:
            clip = float(self.precondition_log_clip)
            target_log = target_log.clamp(min=-clip, max=clip)

        if not bool(self.precondition_initialized.item()):
            self.precondition_log_scale_ema.copy_(target_log)
            self.precondition_initialized.fill_(True)
        elif self.training:
            momentum = float(self.precondition_momentum)
            self.precondition_log_scale_ema.mul_(momentum).add_(target_log * (1.0 - momentum))

        return self.precondition_log_scale_ema.exp()

    @staticmethod
    def _symmetric_psd_sqrt_and_invsqrt(
        mat: torch.Tensor,
        *,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eigvals, eigvecs = torch.linalg.eigh(mat)
        eigvals = eigvals.clamp_min(float(eps))
        sqrt_vals = eigvals.sqrt()
        invsqrt_vals = eigvals.rsqrt()
        sqrt_mat = eigvecs @ torch.diag_embed(sqrt_vals) @ eigvecs.transpose(-1, -2)
        invsqrt_mat = eigvecs @ torch.diag_embed(invsqrt_vals) @ eigvecs.transpose(-1, -2)
        return sqrt_mat, invsqrt_mat

    @torch.no_grad()
    def _update_dual_precondition_matrix(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.precondition_mode != "full":
            raise RuntimeError("_update_dual_precondition_matrix is only valid when performer_precondition_mode='full'.")

        with torch.autocast(device_type=q.device.type, enabled=False):
            q32 = q.float()
            k32 = k.float()
            num_samples = max(1, q32.shape[0] * q32.shape[1])
            cov_q = torch.einsum('bthd,bthe->hde', q32, q32) / float(num_samples)
            cov_k = torch.einsum('bthd,bthe->hde', k32, k32) / float(num_samples)
            eye = torch.eye(self.head_k_dim, device=q32.device, dtype=torch.float32).unsqueeze(0)
            cov_q = cov_q + eye * float(self.precondition_eps)
            cov_k = cov_k + eye * float(self.precondition_eps)

            if not bool(self.precondition_initialized.item()):
                self.precondition_cov_q_ema.copy_(cov_q)
                self.precondition_cov_k_ema.copy_(cov_k)
                self.precondition_initialized.fill_(True)
            elif self.training:
                momentum = float(self.precondition_momentum)
                self.precondition_cov_q_ema.mul_(momentum).add_(cov_q * (1.0 - momentum))
                self.precondition_cov_k_ema.mul_(momentum).add_(cov_k * (1.0 - momentum))

            cov_q_ema = self.precondition_cov_q_ema.to(device=q32.device, dtype=torch.float32)
            cov_k_ema = self.precondition_cov_k_ema.to(device=q32.device, dtype=torch.float32)

            cov_q_sqrt, cov_q_invsqrt = self._symmetric_psd_sqrt_and_invsqrt(
                cov_q_ema,
                eps=self.precondition_eps,
            )
            geom_arg = cov_q_sqrt @ cov_k_ema @ cov_q_sqrt
            geom_sqrt, _ = self._symmetric_psd_sqrt_and_invsqrt(
                geom_arg,
                eps=self.precondition_eps,
            )
            # B solves B C_q B = C_k; choose symmetric A = B^{1/2}.
            b_mat = cov_q_invsqrt @ geom_sqrt @ cov_q_invsqrt
            eig_b, vec_b = torch.linalg.eigh(b_mat)
            eig_b = eig_b.clamp_min(float(self.precondition_eps))
            a_log = 0.5 * eig_b.log()
            if self.precondition_log_clip > 0:
                clip = float(self.precondition_log_clip)
                a_log = a_log.clamp(min=-clip, max=clip)
            a_vals = a_log.exp()
            a_inv_vals = (-a_log).exp()
            a_mat = vec_b @ torch.diag_embed(a_vals) @ vec_b.transpose(-1, -2)
            a_inv = vec_b @ torch.diag_embed(a_inv_vals) @ vec_b.transpose(-1, -2)
        return a_mat, a_inv

    def _prepare_centered_kernel_input(
        self,
        x: torch.Tensor,
        center: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if center is None:
            return x, None

        x32 = x.float()
        c32 = center.float().view(1, 1, center.shape[0], center.shape[1])
        x_shifted = (x32 - c32).to(x.dtype)

        # Change-of-measure correction:
        # exp(c·x - ||c||^2 / 2), computed in the same d^{-1/4} scaled space
        # used by FAVOR to preserve kernel consistency.
        data_normalizer = x.shape[-1] ** -0.25
        x_scaled = x32 * data_normalizer
        c_scaled = c32 * data_normalizer
        log_correction = (x_scaled * c_scaled).sum(dim=-1, keepdim=True) - 0.5 * c_scaled.square().sum(
            dim=-1,
            keepdim=True,
        )
        if self.adaptive_center_log_clip > 0:
            clip = float(self.adaptive_center_log_clip)
            log_correction = log_correction.clamp(min=-clip, max=clip)
        correction = torch.exp(log_correction)
        return x_shifted, correction

    @staticmethod
    def _apply_feature_correction(feature: torch.Tensor, correction: torch.Tensor | None) -> torch.Tensor:
        if correction is None:
            return feature
        return feature * correction.to(feature.dtype)

    @torch.no_grad()
    def _redraw_projection_matrix(self) -> None:
        generator = None
        if self.projection_seed is not None:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(int(self.projection_seed))
            self.projection_seed += 1
        if self.use_deterministic_nodes:
            matrix = _build_mixed_projection_matrix(
                num_heads=self.num_heads,
                num_features=self.num_features,
                head_dim=self.head_k_dim,
                device=torch.device('cpu'),
                dtype=self.projection_matrix.dtype,
                scaling=self.ortho_scaling,
                antithetic=self.antithetic_features,
                deterministic_ratio=self.deterministic_ratio,
                stratified_norm_sampling=self.stratified_norm_sampling,
                stratified_jitter=self.stratified_jitter,
                qmc_gaussian_sampling=self.qmc_gaussian_sampling,
                qmc_scramble=self.qmc_scramble,
                generator=generator,
            )
        else:
            matrix = _build_antithetic_orthogonal_random_matrix(
                num_heads=self.num_heads,
                num_features=self.num_features,
                head_dim=self.head_k_dim,
                device=torch.device('cpu'),
                dtype=self.projection_matrix.dtype,
                scaling=self.ortho_scaling,
                antithetic=self.antithetic_features,
                stratified_norm_sampling=self.stratified_norm_sampling,
                stratified_jitter=self.stratified_jitter,
                qmc_gaussian_sampling=self.qmc_gaussian_sampling,
                qmc_scramble=self.qmc_scramble,
                generator=generator,
            )
        matrix = matrix.to(self.projection_matrix.device)
        self.projection_matrix.copy_(matrix)
        row_scale = matrix.float().norm(dim=-1).clamp_min(1e-6)
        if self.learnable_projection:
            if self.learnable_projection_scale:
                self.projection_log_scale.copy_(row_scale.log())
            else:
                self.projection_row_scale.copy_(row_scale)

    def _should_collect_error_stats(self) -> bool:
        if not self.enable_error_observability:
            return False
        self._error_observe_counter += 1
        # Align with trainer-side observability that typically probes at step 0, N, 2N...
        # so the first forward should be observable when interval >= 1.
        return ((self._error_observe_counter - 1) % self.error_observe_interval) == 0

    @torch.no_grad()
    def _collect_error_stats(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        q_prime: torch.Tensor,
        k_prime: torch.Tensor,
        q_prime_den: torch.Tensor | None,
        k_prime_den: torch.Tensor | None,
        beta: torch.Tensor | None,
        adaptive_center: torch.Tensor | None,
        dual_precondition: torch.Tensor | None,
        adaptive_den_mix_alpha: torch.Tensor | None,
        den_ratio_effective: float | None,
        den_map: str,
        delta_denom_eps: float | None,
    ) -> dict[str, float]:
        stats: dict[str, float] = {}
        def _tensor_mb(x: torch.Tensor | None) -> float:
            if x is None:
                return 0.0
            return float(x.numel() * x.element_size()) / (1024.0 * 1024.0)

        stats["obs_mem_q_mb"] = _tensor_mb(q)
        stats["obs_mem_k_mb"] = _tensor_mb(k)
        stats["obs_mem_q_prime_mb"] = _tensor_mb(q_prime)
        stats["obs_mem_k_prime_mb"] = _tensor_mb(k_prime)
        stats["obs_mem_q_prime_den_mb"] = _tensor_mb(q_prime_den)
        stats["obs_mem_k_prime_den_mb"] = _tensor_mb(k_prime_den)

        sample_t = min(int(q.shape[1]), int(self.error_observe_max_tokens))
        sample_h = min(int(q.shape[2]), int(self.error_observe_max_heads))
        sample_t_feat = min(sample_t, int(q_prime.shape[1]))
        sample_h_feat = min(sample_h, int(q_prime.shape[2]))
        sample_t = sample_t_feat
        sample_h = sample_h_feat
        if sample_t < 1 or sample_h < 1:
            return stats
        q_feat_sub = q_prime[0, :sample_t_feat, :sample_h_feat, :]
        k_feat_sub = k_prime[0, :sample_t_feat, :sample_h_feat, :]
        q_feat_sub32 = q_feat_sub.float()
        k_feat_sub32 = k_feat_sub.float()
        stats["obs_num_feat_q_rms"] = float(q_feat_sub32.square().mean().sqrt().item())
        stats["obs_num_feat_k_rms"] = float(k_feat_sub32.square().mean().sqrt().item())
        stats["obs_num_feat_nonfinite_frac"] = float(
            0.5
            * (
                (~torch.isfinite(q_feat_sub)).float().mean()
                + (~torch.isfinite(k_feat_sub)).float().mean()
            ).item()
        )

        sample_t_den_eff = 0
        sample_h_den_eff = 0
        if q_prime_den is not None and k_prime_den is not None:
            sample_t_den = min(sample_t, int(q_prime_den.shape[1]))
            sample_h_den = min(sample_h, int(q_prime_den.shape[2]))
            sample_t_den_eff = sample_t_den
            sample_h_den_eff = sample_h_den
            q_den_sub = q_prime_den[0, :sample_t_den, :sample_h_den, :]
            k_den_sub = k_prime_den[0, :sample_t_den, :sample_h_den, :]
            q_den_sub32 = q_den_sub.float()
            k_den_sub32 = k_den_sub.float()
            stats["obs_den_feat_q_rms"] = float(q_den_sub32.square().mean().sqrt().item())
            stats["obs_den_feat_k_rms"] = float(k_den_sub32.square().mean().sqrt().item())
            stats["obs_den_feat_nonfinite_frac"] = float(
                0.5
                * (
                    (~torch.isfinite(q_den_sub)).float().mean()
                    + (~torch.isfinite(k_den_sub)).float().mean()
                ).item()
            )
        else:
            q_den_sub32 = None
            k_den_sub32 = None

        if beta is not None:
            beta_sub = beta[0, :sample_t, :sample_h].float()
            stats["obs_beta_mean"] = float(beta_sub.mean().item())
            stats["obs_beta_max"] = float(beta_sub.max().item())

        if adaptive_center is not None:
            center_h = min(sample_h, int(adaptive_center.shape[0]))
            stats["obs_adaptive_center_rms"] = float(adaptive_center[:center_h].float().square().mean().sqrt().item())
        if dual_precondition is not None:
            if dual_precondition.ndim == 2:
                precond_h = min(sample_h, int(dual_precondition.shape[0]))
                precond_log = dual_precondition[:precond_h].float().clamp_min(1e-12).log()
                stats["obs_precond_log_std"] = float(precond_log.std(unbiased=False).item())
                stats["obs_precond_log_absmax"] = float(precond_log.abs().max().item())
            else:
                precond_h = min(sample_h, int(dual_precondition.shape[0]))
                precond_norm = dual_precondition[:precond_h].float().norm(dim=(-1, -2))
                stats["obs_precond_mat_norm_mean"] = float(precond_norm.mean().item())
                stats["obs_precond_mat_norm_max"] = float(precond_norm.max().item())
        if adaptive_den_mix_alpha is not None:
            alpha_h = min(sample_h, int(adaptive_den_mix_alpha.shape[0]))
            alpha_sub = adaptive_den_mix_alpha[:alpha_h].float()
            stats["obs_adaptive_den_mix_alpha_mean"] = float(alpha_sub.mean().item())
            stats["obs_adaptive_den_mix_alpha_max"] = float(alpha_sub.max().item())
        if den_ratio_effective is not None and math.isfinite(float(den_ratio_effective)):
            stats["obs_den_ratio_effective"] = float(den_ratio_effective)

        if delta_denom_eps is not None:
            stats["obs_delta_denom_eps"] = float(delta_denom_eps)

        den_map_code = {
            "disabled": 0.0,
            "numerator": 1.0,
            "dual_softmax": 2.0,
            "abs": 3.0,
            "softplus": 4.0,
            "positive_linear": 5.0,
        }.get(den_map, -1.0)
        stats["obs_den_map_code"] = float(den_map_code)

        if sample_t < 2 or sample_h < 1:
            return stats

        q_sub = q.float()[0, :sample_t, :sample_h, :]
        k_sub = k.float()[0, :sample_t, :sample_h, :]
        logits = torch.einsum("thd,shd->hts", q_sub, k_sub)
        target_kernel = torch.exp(logits.clamp(min=-12.0, max=12.0))
        approx_kernel = torch.einsum("thm,shm->hts", q_feat_sub32[:sample_t, :sample_h, :], k_feat_sub32[:sample_t, :sample_h, :])

        tril = torch.tril(
            torch.ones(sample_t, sample_t, device=q.device, dtype=torch.float32)
        ).unsqueeze(0)
        denom_count = float(sample_h) * float(tril.sum().item())
        denom_count = max(1.0, denom_count)

        def _kernel_metrics(prefix: str, approx: torch.Tensor) -> None:
            diff = (approx - target_kernel) * tril
            rel = (diff.abs() / target_kernel.clamp_min(1e-6)) * tril
            prefix_sum = (approx * tril).sum(dim=-1)
            stats[f"obs_{prefix}_kernel_mse"] = float((diff.square().sum() / denom_count).item())
            stats[f"obs_{prefix}_kernel_rel_l1"] = float((rel.sum() / denom_count).item())
            stats[f"obs_{prefix}_kernel_neg_frac"] = float(
                (((approx < 0).float() * tril).sum() / denom_count).item()
            )
            stats[f"obs_{prefix}_prefix_min"] = float(prefix_sum.min().item())
            stats[f"obs_{prefix}_prefix_neg_frac"] = float((prefix_sum < 0).float().mean().item())

        _kernel_metrics("num", approx_kernel)
        stats["obs_qk_logit_std"] = float(logits.std(unbiased=False).item())
        stats["obs_qk_logit_absmax"] = float(logits.abs().max().item())

        if q_den_sub32 is not None and k_den_sub32 is not None and sample_t_den_eff >= 2 and sample_h_den_eff >= 1:
            target_den = target_kernel[:, :sample_t_den_eff, :sample_t_den_eff]
            tril_den = torch.tril(
                torch.ones(sample_t_den_eff, sample_t_den_eff, device=q.device, dtype=torch.float32)
            ).unsqueeze(0)
            denom_count_den = max(1.0, float(sample_h_den_eff) * float(tril_den.sum().item()))
            approx_den_kernel = torch.einsum(
                "thm,shm->hts",
                q_den_sub32[:sample_t_den_eff, :sample_h_den_eff, :],
                k_den_sub32[:sample_t_den_eff, :sample_h_den_eff, :],
            )
            diff_den = (approx_den_kernel - target_den) * tril_den
            rel_den = (diff_den.abs() / target_den.clamp_min(1e-6)) * tril_den
            prefix_den = (approx_den_kernel * tril_den).sum(dim=-1)
            stats["obs_den_kernel_mse"] = float((diff_den.square().sum() / denom_count_den).item())
            stats["obs_den_kernel_rel_l1"] = float((rel_den.sum() / denom_count_den).item())
            stats["obs_den_kernel_neg_frac"] = float(
                (((approx_den_kernel < 0).float() * tril_den).sum() / denom_count_den).item()
            )
            stats["obs_den_prefix_min"] = float(prefix_den.min().item())
            stats["obs_den_prefix_neg_frac"] = float((prefix_den < 0).float().mean().item())

        # Kernel approximation error attribution between numerator and denominator
        # branches. This is a lightweight proxy for ratio-estimator error source:
        # if denominator branch dominates here, prioritize denominator map/budget;
        # otherwise prioritize numerator feature-map variance reduction.
        num_mse = stats.get("obs_num_kernel_mse", None)
        den_mse = stats.get("obs_den_kernel_mse", None)
        if num_mse is not None and den_mse is not None:
            total_mse = float(num_mse) + float(den_mse)
            total_mse = max(total_mse, 1e-12)
            stats["obs_kernel_err_num_frac"] = float(num_mse) / total_mse
            stats["obs_kernel_err_den_frac"] = float(den_mse) / total_mse
            stats["obs_kernel_err_num_over_den"] = float(num_mse) / max(float(den_mse), 1e-12)

        num_rel = stats.get("obs_num_kernel_rel_l1", None)
        den_rel = stats.get("obs_den_kernel_rel_l1", None)
        if num_rel is not None and den_rel is not None:
            total_rel = float(num_rel) + float(den_rel)
            total_rel = max(total_rel, 1e-12)
            stats["obs_kernel_rel_num_frac"] = float(num_rel) / total_rel
            stats["obs_kernel_rel_den_frac"] = float(den_rel) / total_rel
            stats["obs_kernel_rel_num_over_den"] = float(num_rel) / max(float(den_rel), 1e-12)

        return stats

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
        self.last_error_stats = {}
        collect_error_stats = self._should_collect_error_stats()

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

        if self.use_value_gate:
            v_gate_logits = self.v_gate_proj(hidden_states).float()
            v_gate = torch.sigmoid(v_gate_logits + self.v_gate_bias.view(1, 1, -1)).to(v.dtype)
            v = v * v_gate.unsqueeze(-1)

        if self.qk_l2_norm:
            # Use the fused Triton L2Norm path (same family used by DeltaNet ops)
            # to reduce intermediate activations compared with F.normalize(...float()).
            q, _ = l2norm_fwd(q, eps=1e-6, output_dtype=q.dtype)
            k, _ = l2norm_fwd(k, eps=1e-6, output_dtype=k.dtype)

        dual_precondition = None
        dual_precondition_inv = None
        if self.use_dual_precondition_sampling:
            if self.precondition_mode == "diag":
                dual_precondition = self._update_dual_precondition_scale(q, k)
            else:
                dual_precondition, dual_precondition_inv = self._update_dual_precondition_matrix(q, k)

        adaptive_center = None
        if self.adaptive_center_sampling:
            adaptive_center = self._update_adaptive_center(q, k)

        if self.num_v_heads > self.num_heads:
            groups = self.num_v_heads // self.num_heads
            q = repeat(q, 'b t h d -> b t (h g) d', g=groups)
            k = repeat(k, 'b t h d -> b t (h g) d', g=groups)
            if adaptive_center is not None:
                adaptive_center = repeat(adaptive_center, 'h d -> (h g) d', g=groups)
            if dual_precondition is not None:
                if dual_precondition.ndim == 2:
                    dual_precondition = repeat(dual_precondition, 'h d -> (h g) d', g=groups)
                else:
                    dual_precondition = repeat(dual_precondition, 'h d e -> (h g) d e', g=groups)
                    dual_precondition_inv = repeat(dual_precondition_inv, 'h d e -> (h g) d e', g=groups)

        if self.learnable_kernel_scale:
            kernel_scale_log = self.kernel_scale_log.float()
            if self.state_update == "delta":
                # Delta recurrence is more sensitive to kernel temperature explosions.
                # Clamp the learnable multiplier instead of the absolute scale.
                kernel_scale_log = kernel_scale_log.clamp(
                    min=math.log(0.5),
                    max=math.log(2.0),
                )
            kernel_scale = kernel_scale_log.exp()
        else:
            kernel_scale = self.kernel_scale.float()
        kernel_scale = kernel_scale * self.kernel_scale_base.float()
        if q.shape[2] != kernel_scale.shape[0]:
            if q.shape[2] % kernel_scale.shape[0] != 0:
                raise ValueError(
                    f"Kernel scale head mismatch: q has {q.shape[2]} heads but kernel_scale has {kernel_scale.shape[0]}."
                )
            groups = q.shape[2] // kernel_scale.shape[0]
            kernel_scale = repeat(kernel_scale, 'h -> (h g)', g=groups)
        scale = kernel_scale.sqrt().view(1, 1, -1, 1).to(q.dtype)
        q = q * scale
        k = k * scale
        if adaptive_center is not None:
            if adaptive_center.shape[0] != q.shape[2]:
                if q.shape[2] % adaptive_center.shape[0] != 0:
                    raise ValueError(
                        f"Adaptive center head mismatch: q has {q.shape[2]} heads but center has {adaptive_center.shape[0]} heads.",
                    )
                groups = q.shape[2] // adaptive_center.shape[0]
                adaptive_center = repeat(adaptive_center, 'h d -> (h g) d', g=groups)
            head_scale = kernel_scale.sqrt().view(-1, 1).to(adaptive_center.dtype)
            adaptive_center = adaptive_center * head_scale
        if dual_precondition is not None:
            if dual_precondition.shape[0] != q.shape[2]:
                if q.shape[2] % dual_precondition.shape[0] != 0:
                    raise ValueError(
                        f"Dual precondition head mismatch: q has {q.shape[2]} heads but precondition has {dual_precondition.shape[0]} heads.",
                    )
                groups = q.shape[2] // dual_precondition.shape[0]
                if dual_precondition.ndim == 2:
                    dual_precondition = repeat(dual_precondition, 'h d -> (h g) d', g=groups)
                else:
                    dual_precondition = repeat(dual_precondition, 'h d e -> (h g) d e', g=groups)
                    dual_precondition_inv = repeat(dual_precondition_inv, 'h d e -> (h g) d e', g=groups)
            if dual_precondition.ndim == 2:
                precond = dual_precondition.to(q.dtype).view(1, 1, q.shape[2], q.shape[3])
                q = q * precond
                k = k / precond
            else:
                q = torch.einsum('bthd,hde->bthe', q.float(), dual_precondition.float()).to(q.dtype)
                k = torch.einsum('bthd,hde->bthe', k.float(), dual_precondition_inv.float()).to(k.dtype)

        q_kernel, q_kernel_correction = self._prepare_centered_kernel_input(q, adaptive_center)
        k_kernel, k_kernel_correction = self._prepare_centered_kernel_input(k, adaptive_center)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        if self.redraw_projection and self.training:
            if self.learnable_projection:
                warnings.warn(
                    "`performer_redraw_projection=True` is ignored when projection is learnable.",
                )
            else:
                self._redraw_projection_matrix()

        projection_matrix = self.projection_matrix
        if self.learnable_projection:
            proj = F.normalize(projection_matrix.float(), dim=-1)
            if self.learnable_projection_scale:
                row_scale = self.projection_log_scale.float().exp().unsqueeze(-1)
            else:
                row_scale = self.projection_row_scale.float().unsqueeze(-1)
            projection_matrix = (proj * row_scale).to(projection_matrix.dtype)
        if projection_matrix.shape[0] != q.shape[2]:
            if q.shape[2] % projection_matrix.shape[0] != 0:
                raise ValueError(
                    f"Head mismatch: q has {q.shape[2]} heads but projection has {projection_matrix.shape[0]} heads.",
                )
            groups = q.shape[2] // projection_matrix.shape[0]
            projection_matrix = repeat(projection_matrix, 'h m d -> (h g) m d', g=groups)

        projection_matrix_base = projection_matrix
        landmark_projection = None
        landmark_base_weight = None
        landmark_weight = None
        if self.use_landmark_sampling:
            rf_features = max(1, min(projection_matrix.shape[1] - 1, self.landmark_rf_features))
            projection_matrix_base = projection_matrix_base[:, :rf_features, :]
            landmark_projection = F.normalize(self.landmark_projection.float(), dim=-1) * math.sqrt(float(self.head_k_dim))
            landmark_projection = landmark_projection.to(projection_matrix.dtype)
            if landmark_projection.shape[0] != q.shape[2]:
                if q.shape[2] % landmark_projection.shape[0] != 0:
                    raise ValueError(
                        f"Landmark projection head mismatch: q has {q.shape[2]} heads but landmarks have {landmark_projection.shape[0]}."
                    )
                groups = q.shape[2] // landmark_projection.shape[0]
                landmark_projection = repeat(landmark_projection, 'h m d -> (h g) m d', g=groups)
            landmark_alpha = torch.sigmoid(self.landmark_alpha_logit.float()).clamp(0.02, 0.98)
            landmark_alpha = self._expand_head_vector(landmark_alpha, q.shape[2], name="Landmark alpha")
            landmark_base_weight = (1.0 - landmark_alpha).sqrt().view(1, 1, -1, 1)
            landmark_weight = landmark_alpha.sqrt().view(1, 1, -1, 1)

        residual_weight = None
        if self.use_control_variate and self.use_cv_residual_shrinkage:
            cv_gamma = torch.sigmoid(self.cv_residual_logit.float()).clamp(0.05, 1.0)
            cv_gamma = self._expand_head_vector(cv_gamma, q.shape[2], name="CV residual gamma")
            residual_weight = cv_gamma.sqrt().to(q.dtype)
        linear_cv_coef = None
        if self.use_control_variate and self.use_adaptive_linear_cv:
            coef = 2.0 * torch.sigmoid(self.cv_linear_coef_logit.float()).clamp(1e-4, 1.0 - 1e-4)
            coef = self._expand_head_vector(coef, q.shape[2], name="CV linear coefficient")
            linear_cv_coef = coef.to(q.dtype)

        if self.pdf_delta_update:
            q_prime = _performer_shared_softmax_feature_map(
                q_kernel,
                projection_matrix=projection_matrix_base,
                eps=self.feature_eps,
            )
            k_prime = _performer_shared_softmax_feature_map(
                k_kernel,
                projection_matrix=projection_matrix_base,
                eps=self.feature_eps,
            )
        elif self.use_control_variate:
            use_decoupled_cv2 = (
                self.use_cv_decoupled_second_order
                and self.use_second_order_cv
                and not self.use_third_order_cv
                and not self.use_projection_ensemble
                and projection_matrix_base.shape[1] >= 2
            )
            if use_decoupled_cv2:
                total_rows = int(projection_matrix_base.shape[1])
                h2_ratio = float(self.cv_decoupled_h2_ratio)
                if self.use_cv_decoupled_adaptive_h2_ratio:
                    h2_ratio = float(self.cv_decoupled_h2_ratio_ema.item())
                h2_rows = min(
                    total_rows - 1,
                    max(1, int(round(float(total_rows) * h2_ratio))),
                )
                residual_rows = max(1, total_rows - h2_rows)
                if self.cv_decoupled_h2_deterministic:
                    projection_matrix_h2 = _build_deterministic_orthogonal_nodes(
                        num_heads=projection_matrix_base.shape[0],
                        num_features=h2_rows,
                        head_dim=self.head_k_dim,
                        device=projection_matrix_base.device,
                        dtype=projection_matrix_base.dtype,
                    )
                else:
                    projection_matrix_h2 = projection_matrix_base[:, :h2_rows, :]
                projection_matrix_res = projection_matrix_base[:, h2_rows:h2_rows + residual_rows, :]
                if projection_matrix_res.shape[1] == 0:
                    projection_matrix_res = projection_matrix_base[:, -1:, :]

                q_ones, q_linear = _performer_control_variate_prefix_map(
                    q_kernel,
                    linear_cv_coef=linear_cv_coef,
                    linear_cv_coef_eps=self.adaptive_linear_cv_eps,
                )
                k_ones, k_linear = _performer_control_variate_prefix_map(
                    k_kernel,
                    linear_cv_coef=linear_cv_coef,
                    linear_cv_coef_eps=self.adaptive_linear_cv_eps,
                )
                q_h2 = _performer_second_order_feature_map(q_kernel, projection_matrix_h2)
                k_h2 = _performer_second_order_feature_map(k_kernel, projection_matrix_h2)
                q_residual = _performer_control_variate_residual_map(
                    q_kernel,
                    projection_matrix=projection_matrix_res,
                    exp_clip=self.control_variate_exp_clip,
                    linear_cv_coef=linear_cv_coef,
                    linear_cv_coef_eps=self.adaptive_linear_cv_eps,
                    use_second_order_cv=True,
                    use_third_order_cv=False,
                    split_feature_budget=False,
                )
                k_residual = _performer_control_variate_residual_map(
                    k_kernel,
                    projection_matrix=projection_matrix_res,
                    exp_clip=self.control_variate_exp_clip,
                    linear_cv_coef=linear_cv_coef,
                    linear_cv_coef_eps=self.adaptive_linear_cv_eps,
                    use_second_order_cv=True,
                    use_third_order_cv=False,
                    split_feature_budget=False,
                )
                if residual_weight is not None:
                    rw = residual_weight.view(1, 1, -1, 1).to(q_residual.dtype)
                    q_residual = q_residual * rw
                    k_residual = k_residual * rw

                if self.training and self.use_cv_decoupled_adaptive_h2_ratio:
                    with torch.no_grad():
                        h2_energy = 0.5 * (
                            q_h2.float().square().mean() + k_h2.float().square().mean()
                        )
                        res_energy = 0.5 * (
                            q_residual.float().square().mean() + k_residual.float().square().mean()
                        )
                        h2_std = torch.sqrt(h2_energy + 1e-12)
                        res_std = torch.sqrt(res_energy + 1e-12)
                        target_ratio = h2_std / (h2_std + res_std + 1e-12)
                        target_ratio = target_ratio.clamp(
                            min=float(self.cv_decoupled_ratio_min),
                            max=float(self.cv_decoupled_ratio_max),
                        )
                        momentum = float(self.cv_decoupled_ratio_ema_momentum)
                        self.cv_decoupled_h2_ratio_ema.copy_(
                            self.cv_decoupled_h2_ratio_ema * momentum + target_ratio * (1.0 - momentum)
                        )

                q_prime = torch.cat([q_ones, q_linear, q_h2, q_residual], dim=-1)
                k_prime = torch.cat([k_ones, k_linear, k_h2, k_residual], dim=-1)
            elif self.use_projection_ensemble:
                num_groups = min(self.projection_ensemble_groups, projection_matrix_base.shape[1])
                if num_groups <= 1:
                    q_prime = _performer_control_variate_feature_map(
                        q_kernel,
                        projection_matrix=projection_matrix_base,
                        exp_clip=self.control_variate_exp_clip,
                        residual_weight=residual_weight,
                        linear_cv_coef=linear_cv_coef,
                        linear_cv_coef_eps=self.adaptive_linear_cv_eps,
                        use_second_order_cv=self.use_second_order_cv,
                        use_third_order_cv=self.use_third_order_cv,
                        split_feature_budget=self.cv_split_feature_budget,
                    )
                    k_prime = _performer_control_variate_feature_map(
                        k_kernel,
                        projection_matrix=projection_matrix_base,
                        exp_clip=self.control_variate_exp_clip,
                        residual_weight=residual_weight,
                        linear_cv_coef=linear_cv_coef,
                        linear_cv_coef_eps=self.adaptive_linear_cv_eps,
                        use_second_order_cv=self.use_second_order_cv,
                        use_third_order_cv=self.use_third_order_cv,
                        split_feature_budget=self.cv_split_feature_budget,
                    )
                else:
                    # Projection ensemble: share deterministic [1, x] terms and
                    # average independent stochastic residual estimators by
                    # concatenating group residuals with 1/sqrt(G) scaling.
                    group_sizes = [projection_matrix_base.shape[1] // num_groups] * num_groups
                    for i in range(projection_matrix_base.shape[1] % num_groups):
                        group_sizes[i] += 1
                    det_dim = 1 + q.shape[-1]
                    q_det = None
                    k_det = None
                    q_stoch_parts: list[torch.Tensor] = []
                    k_stoch_parts: list[torch.Tensor] = []
                    offset = 0
                    for size in group_sizes:
                        pm_chunk = projection_matrix_base[:, offset:offset + size, :]
                        offset += size
                        q_chunk = _performer_control_variate_feature_map(
                            q_kernel,
                            projection_matrix=pm_chunk,
                            exp_clip=self.control_variate_exp_clip,
                            residual_weight=residual_weight,
                            linear_cv_coef=linear_cv_coef,
                            linear_cv_coef_eps=self.adaptive_linear_cv_eps,
                            use_second_order_cv=self.use_second_order_cv,
                            use_third_order_cv=self.use_third_order_cv,
                            split_feature_budget=self.cv_split_feature_budget,
                            finite_sample_orthogonalize=self.cv_finite_sample_orthogonalize,
                            finite_sample_orth_eps=self.cv_finite_sample_orth_eps,
                        )
                        k_chunk = _performer_control_variate_feature_map(
                            k_kernel,
                            projection_matrix=pm_chunk,
                            exp_clip=self.control_variate_exp_clip,
                            residual_weight=residual_weight,
                            linear_cv_coef=linear_cv_coef,
                            linear_cv_coef_eps=self.adaptive_linear_cv_eps,
                            use_second_order_cv=self.use_second_order_cv,
                            use_third_order_cv=self.use_third_order_cv,
                            split_feature_budget=self.cv_split_feature_budget,
                            finite_sample_orthogonalize=self.cv_finite_sample_orthogonalize,
                            finite_sample_orth_eps=self.cv_finite_sample_orth_eps,
                        )
                        if q_det is None:
                            q_det = q_chunk[..., :det_dim]
                            k_det = k_chunk[..., :det_dim]
                        q_stoch_parts.append(q_chunk[..., det_dim:])
                        k_stoch_parts.append(k_chunk[..., det_dim:])

                    group_scale = float(num_groups) ** -0.5
                    q_prime = torch.cat(
                        [q_det] + [part * group_scale for part in q_stoch_parts],
                        dim=-1,
                    )
                    k_prime = torch.cat(
                        [k_det] + [part * group_scale for part in k_stoch_parts],
                        dim=-1,
                    )
            else:
                q_prime = _performer_control_variate_feature_map(
                    q_kernel,
                    projection_matrix=projection_matrix_base,
                    exp_clip=self.control_variate_exp_clip,
                    residual_weight=residual_weight,
                    linear_cv_coef=linear_cv_coef,
                    linear_cv_coef_eps=self.adaptive_linear_cv_eps,
                    use_second_order_cv=self.use_second_order_cv,
                    use_third_order_cv=self.use_third_order_cv,
                    split_feature_budget=self.cv_split_feature_budget,
                    finite_sample_orthogonalize=self.cv_finite_sample_orthogonalize,
                    finite_sample_orth_eps=self.cv_finite_sample_orth_eps,
                )
                k_prime = _performer_control_variate_feature_map(
                    k_kernel,
                    projection_matrix=projection_matrix_base,
                    exp_clip=self.control_variate_exp_clip,
                    residual_weight=residual_weight,
                    linear_cv_coef=linear_cv_coef,
                    linear_cv_coef_eps=self.adaptive_linear_cv_eps,
                    use_second_order_cv=self.use_second_order_cv,
                    use_third_order_cv=self.use_third_order_cv,
                    split_feature_budget=self.cv_split_feature_budget,
                    finite_sample_orthogonalize=self.cv_finite_sample_orthogonalize,
                    finite_sample_orth_eps=self.cv_finite_sample_orth_eps,
                )
        else:
            q_prime = performer_softmax_feature_map(
                q_kernel,
                projection_matrix=projection_matrix_base,
                is_query=True,
                eps=self.feature_eps,
                cu_seqlens=cu_seqlens,
            )
            k_prime = performer_softmax_feature_map(
                k_kernel,
                projection_matrix=projection_matrix_base,
                is_query=False,
                eps=self.feature_eps,
                cu_seqlens=cu_seqlens,
            )
        q_prime = self._apply_feature_correction(q_prime, q_kernel_correction)
        k_prime = self._apply_feature_correction(k_prime, k_kernel_correction)

        if landmark_projection is not None:
            if self.use_control_variate and not self.use_second_order_cv:
                # Keep exact low-order terms [1, x] intact and only mix sampled
                # high-order residual channels.
                det_dim = 1 + q.shape[-1]
                q_det = q_prime[..., :det_dim]
                k_det = k_prime[..., :det_dim]
                q_res = q_prime[..., det_dim:]
                k_res = k_prime[..., det_dim:]
                q_landmark_res = _performer_control_variate_residual_map(
                    q_kernel,
                    projection_matrix=landmark_projection,
                    exp_clip=self.control_variate_exp_clip,
                    linear_cv_coef=linear_cv_coef,
                    linear_cv_coef_eps=self.adaptive_linear_cv_eps,
                    use_second_order_cv=False,
                    use_third_order_cv=False,
                )
                k_landmark_res = _performer_control_variate_residual_map(
                    k_kernel,
                    projection_matrix=landmark_projection,
                    exp_clip=self.control_variate_exp_clip,
                    linear_cv_coef=linear_cv_coef,
                    linear_cv_coef_eps=self.adaptive_linear_cv_eps,
                    use_second_order_cv=False,
                    use_third_order_cv=False,
                )
                q_landmark_res = self._apply_feature_correction(q_landmark_res, q_kernel_correction)
                k_landmark_res = self._apply_feature_correction(k_landmark_res, k_kernel_correction)
                q_prime = torch.cat(
                    [
                        q_det,
                        q_res * landmark_base_weight.to(q_res.dtype),
                        q_landmark_res.to(q_res.dtype) * landmark_weight.to(q_res.dtype),
                    ],
                    dim=-1,
                )
                k_prime = torch.cat(
                    [
                        k_det,
                        k_res * landmark_base_weight.to(k_res.dtype),
                        k_landmark_res.to(k_res.dtype) * landmark_weight.to(k_res.dtype),
                    ],
                    dim=-1,
                )
            else:
                q_landmark = performer_softmax_feature_map(
                    q_kernel,
                    projection_matrix=landmark_projection,
                    is_query=True,
                    eps=self.feature_eps,
                    cu_seqlens=cu_seqlens,
                )
                k_landmark = performer_softmax_feature_map(
                    k_kernel,
                    projection_matrix=landmark_projection,
                    is_query=False,
                    eps=self.feature_eps,
                    cu_seqlens=cu_seqlens,
                )
                q_landmark = self._apply_feature_correction(q_landmark, q_kernel_correction)
                k_landmark = self._apply_feature_correction(k_landmark, k_kernel_correction)
                q_prime = torch.cat(
                    [
                        q_prime * landmark_base_weight.to(q_prime.dtype),
                        q_landmark.to(q_prime.dtype) * landmark_weight.to(q_prime.dtype),
                    ],
                    dim=-1,
                )
                k_prime = torch.cat(
                    [
                        k_prime * landmark_base_weight.to(k_prime.dtype),
                        k_landmark.to(k_prime.dtype) * landmark_weight.to(k_prime.dtype),
                    ],
                    dim=-1,
                )

        if self.use_diag2_term:
            # Add a low-rank deterministic diagonal second-order component:
            # K_diag2(q, k) = 0.5 * <q^2, k^2> over a subspace.
            # The blended kernel is (1-alpha) * K_cv + alpha * K_diag2.
            quad_rank = max(1, int(round(self.head_k_dim * self.diag2_ratio)))
            q_diag2 = q[..., :quad_rank].float().square() * (2.0 ** -0.5)
            k_diag2 = k[..., :quad_rank].float().square() * (2.0 ** -0.5)
            diag2_alpha = torch.sigmoid(self.diag2_alpha_logit.float()).clamp(0.01, 0.95)
            if q.shape[2] != diag2_alpha.shape[0]:
                if q.shape[2] % diag2_alpha.shape[0] != 0:
                    raise ValueError(
                        f"Diag2 alpha head mismatch: q has {q.shape[2]} heads but alpha has {diag2_alpha.shape[0]}."
                    )
                groups = q.shape[2] // diag2_alpha.shape[0]
                diag2_alpha = repeat(diag2_alpha, 'h -> (h g)', g=groups)
            base_weight = (1.0 - diag2_alpha).sqrt().view(1, 1, -1, 1)
            diag2_weight = diag2_alpha.sqrt().view(1, 1, -1, 1)
            q_prime = torch.cat(
                [
                    q_prime * base_weight.to(q_prime.dtype),
                    q_diag2.to(q_prime.dtype) * diag2_weight.to(q_prime.dtype),
                ],
                dim=-1,
            )
            k_prime = torch.cat(
                [
                    k_prime * base_weight.to(k_prime.dtype),
                    k_diag2.to(k_prime.dtype) * diag2_weight.to(k_prime.dtype),
                ],
                dim=-1,
            )
        if self.pdf_delta_update and self.pdf_delta_feature_low_precision:
            if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
                target_dtype = q.dtype
                if q_prime.dtype != target_dtype:
                    q_prime = q_prime.to(target_dtype)
                if k_prime.dtype != target_dtype:
                    k_prime = k_prime.to(target_dtype)
        q_prime_den = None
        k_prime_den = None
        q_prime_den_fp32 = None
        k_prime_den_fp32 = None
        k_prime_den_for_beta = None
        adaptive_den_mix_alpha = None
        den_ratio_effective = None
        den_map_effective = "disabled"
        if self.pdf_delta_update:
            q_prime_den = q_prime
            k_prime_den = k_prime
            den_map_effective = "shared_softmax"
        elif self.state_update == "delta":
            den_map = self.delta_denominator_map
            if den_map == "auto":
                den_map = "dual_softmax" if self.use_dual_map and self.use_control_variate else "numerator"
            den_map_effective = den_map

            if den_map == "abs":
                q_prime_den_fp32 = q_prime.float().abs().clamp_min(self.feature_eps)
                k_prime_den_fp32 = k_prime.float().abs().clamp_min(self.feature_eps)
                q_prime_den = q_prime_den_fp32
                k_prime_den = k_prime_den_fp32
            elif den_map == "softplus":
                q_prime_den_fp32 = F.softplus(q_prime.float()) + self.feature_eps
                k_prime_den_fp32 = F.softplus(k_prime.float()) + self.feature_eps
                q_prime_den = q_prime_den_fp32
                k_prime_den = k_prime_den_fp32
            elif den_map == "dual_softmax":
                # In delta mode, numerator can use control-variate features while denominator
                # stays strictly positive to prevent sign-flip instabilities.
                projection_matrix_den = projection_matrix
                den_ratio = float(self.dual_map_den_ratio)
                if self.use_error_feedback_den_ratio and den_ratio < 1.0:
                    den_ratio = float(self.error_feedback_den_ratio_ema.item())
                elif (
                    self.use_layerwise_den_ratio
                    and self.layer_idx is not None
                    and den_ratio < 1.0
                ):
                    depth = max(0.0, float(self.layer_idx))
                    tau = float(self.layerwise_den_ratio_tau)
                    den_ratio = 1.0 - (1.0 - den_ratio) * math.exp(-depth / tau)
                den_ratio = min(1.0, max(float(self.dual_map_den_ratio), den_ratio))
                den_ratio_effective = den_ratio
                if den_ratio < 1.0:
                    den_features = max(1, int(round(self.num_features * den_ratio)))
                    projection_matrix_den = self._select_dual_map_rows(
                        projection_matrix,
                        den_features,
                    )
                den_features = projection_matrix_den.shape[1]
                projection_matrix_den_softmax = projection_matrix_den
                projection_matrix_den_landmark = None
                if (
                    landmark_projection is not None
                    and self.landmark_in_denominator
                    and not self.use_den_poly_kernel
                    and den_features >= 2
                ):
                    den_landmark_features = min(landmark_projection.shape[1], den_features - 1)
                    den_rf_features = max(1, den_features - den_landmark_features)
                    projection_matrix_den_softmax = projection_matrix_den[:, :den_rf_features, :]
                    projection_matrix_den_landmark = landmark_projection[:, :den_landmark_features, :]
                projection_matrix_den_poly = None
                if self.use_den_poly_kernel and den_features >= 4:
                    # Keep denominator state width unchanged: allocate a low-rank positive linear
                    # branch and reduce RF denominator features accordingly.
                    poly_rank = max(1, int(round(den_features * self.den_poly_ratio)))
                    poly_rank = min(poly_rank, den_features - 2)
                    rf_features = max(1, den_features - (poly_rank + 1))
                    projection_matrix_den_softmax = projection_matrix_den[:, :rf_features, :]
                    projection_matrix_den_poly = projection_matrix_den[:, rf_features:rf_features + poly_rank, :]
                q_prime_den_fp32 = performer_softmax_feature_map(
                    q_kernel,
                    projection_matrix=projection_matrix_den_softmax,
                    is_query=True,
                    eps=self.feature_eps,
                    cu_seqlens=cu_seqlens,
                )
                k_prime_den_fp32 = performer_softmax_feature_map(
                    k_kernel,
                    projection_matrix=projection_matrix_den_softmax,
                    is_query=False,
                    eps=self.feature_eps,
                    cu_seqlens=cu_seqlens,
                )
                q_prime_den_fp32 = self._apply_feature_correction(q_prime_den_fp32, q_kernel_correction)
                k_prime_den_fp32 = self._apply_feature_correction(k_prime_den_fp32, k_kernel_correction)
                if (
                    self.use_adaptive_den_mix
                    and projection_matrix_den_softmax.shape[1] < projection_matrix.shape[1]
                    and projection_matrix_den_landmark is None
                    and not self.use_den_poly_kernel
                ):
                    projection_matrix_den_extra = projection_matrix[:, projection_matrix_den_softmax.shape[1]:, :]
                    if projection_matrix_den_extra.shape[1] > 0:
                        q_den_extra = performer_softmax_feature_map(
                            q_kernel,
                            projection_matrix=projection_matrix_den_extra,
                            is_query=True,
                            eps=self.feature_eps,
                            cu_seqlens=cu_seqlens,
                        )
                        k_den_extra = performer_softmax_feature_map(
                            k_kernel,
                            projection_matrix=projection_matrix_den_extra,
                            is_query=False,
                            eps=self.feature_eps,
                            cu_seqlens=cu_seqlens,
                        )
                        q_den_extra = self._apply_feature_correction(q_den_extra, q_kernel_correction)
                        k_den_extra = self._apply_feature_correction(k_den_extra, k_kernel_correction)
                        adaptive_den_mix_alpha = torch.sigmoid(self.adaptive_den_mix_logit.float()).clamp(1e-4, 1 - 1e-4)
                        if q.shape[2] != adaptive_den_mix_alpha.shape[0]:
                            if q.shape[2] % adaptive_den_mix_alpha.shape[0] != 0:
                                raise ValueError(
                                    f"Adaptive denominator mix alpha head mismatch: q has {q.shape[2]} heads but alpha has {adaptive_den_mix_alpha.shape[0]}."
                                )
                            groups = q.shape[2] // adaptive_den_mix_alpha.shape[0]
                            adaptive_den_mix_alpha = repeat(adaptive_den_mix_alpha, 'h -> (h g)', g=groups)
                        den_base_weight = (1.0 - adaptive_den_mix_alpha).sqrt().view(1, 1, -1, 1)
                        den_extra_weight = adaptive_den_mix_alpha.sqrt().view(1, 1, -1, 1)
                        q_prime_den_fp32 = torch.cat(
                            [
                                q_prime_den_fp32 * den_base_weight.to(q_prime_den_fp32.dtype),
                                q_den_extra * den_extra_weight.to(q_den_extra.dtype),
                            ],
                            dim=-1,
                        )
                        k_prime_den_fp32 = torch.cat(
                            [
                                k_prime_den_fp32 * den_base_weight.to(k_prime_den_fp32.dtype),
                                k_den_extra * den_extra_weight.to(k_den_extra.dtype),
                            ],
                            dim=-1,
                        )
                if self.use_den_poly_kernel and projection_matrix_den_poly is not None:
                    q_den_poly = _performer_positive_linear_feature_map(
                        q,
                        projection_matrix=projection_matrix_den_poly,
                        constant=self.den_poly_constant,
                    )
                    k_den_poly = _performer_positive_linear_feature_map(
                        k,
                        projection_matrix=projection_matrix_den_poly,
                        constant=self.den_poly_constant,
                    )
                    den_alpha = torch.sigmoid(self.den_poly_alpha_logit.float()).clamp(0.01, 0.95)
                    if q.shape[2] != den_alpha.shape[0]:
                        if q.shape[2] % den_alpha.shape[0] != 0:
                            raise ValueError(
                                f"Denominator poly alpha head mismatch: q has {q.shape[2]} heads but alpha has {den_alpha.shape[0]}."
                            )
                        groups = q.shape[2] // den_alpha.shape[0]
                        den_alpha = repeat(den_alpha, 'h -> (h g)', g=groups)
                    rf_weight = (1.0 - den_alpha).sqrt().view(1, 1, -1, 1)
                    poly_weight = den_alpha.sqrt().view(1, 1, -1, 1)
                    q_prime_den_fp32 = torch.cat(
                        [
                            q_prime_den_fp32 * rf_weight.to(q_prime_den_fp32.dtype),
                            q_den_poly * poly_weight.to(q_den_poly.dtype),
                        ],
                        dim=-1,
                    )
                    k_prime_den_fp32 = torch.cat(
                        [
                            k_prime_den_fp32 * rf_weight.to(k_prime_den_fp32.dtype),
                            k_den_poly * poly_weight.to(k_den_poly.dtype),
                        ],
                        dim=-1,
                    )
                if projection_matrix_den_landmark is not None:
                    q_den_landmark = performer_softmax_feature_map(
                        q_kernel,
                        projection_matrix=projection_matrix_den_landmark,
                        is_query=True,
                        eps=self.feature_eps,
                        cu_seqlens=cu_seqlens,
                    )
                    k_den_landmark = performer_softmax_feature_map(
                        k_kernel,
                        projection_matrix=projection_matrix_den_landmark,
                        is_query=False,
                        eps=self.feature_eps,
                        cu_seqlens=cu_seqlens,
                    )
                    q_den_landmark = self._apply_feature_correction(q_den_landmark, q_kernel_correction)
                    k_den_landmark = self._apply_feature_correction(k_den_landmark, k_kernel_correction)
                    q_prime_den_fp32 = torch.cat(
                        [
                            q_prime_den_fp32 * landmark_base_weight.to(q_prime_den_fp32.dtype),
                            q_den_landmark.to(q_prime_den_fp32.dtype) * landmark_weight.to(q_prime_den_fp32.dtype),
                        ],
                        dim=-1,
                    )
                    k_prime_den_fp32 = torch.cat(
                        [
                            k_prime_den_fp32 * landmark_base_weight.to(k_prime_den_fp32.dtype),
                            k_den_landmark.to(k_prime_den_fp32.dtype) * landmark_weight.to(k_prime_den_fp32.dtype),
                        ],
                        dim=-1,
                    )
                q_prime_den = q_prime_den_fp32
                k_prime_den = k_prime_den_fp32
            elif den_map == "positive_linear":
                # Deterministic positive denominator kernel:
                #   K_den(q, k) = c + <q, k> / c
                # implemented by phi_den(x) = [sqrt(c), x / sqrt(c)].
                c = float(self.den_poly_constant)
                c_sqrt = math.sqrt(c)
                q32 = q.float()
                k32 = k.float()
                q_const = torch.full(
                    (*q32.shape[:3], 1),
                    fill_value=c_sqrt,
                    device=q32.device,
                    dtype=q32.dtype,
                )
                k_const = torch.full(
                    (*k32.shape[:3], 1),
                    fill_value=c_sqrt,
                    device=k32.device,
                    dtype=k32.dtype,
                )
                q_prime_den_fp32 = torch.cat([q_const, q32 / c_sqrt], dim=-1)
                k_prime_den_fp32 = torch.cat([k_const, k32 / c_sqrt], dim=-1)
                q_prime_den = q_prime_den_fp32
                k_prime_den = k_prime_den_fp32

            if k_prime_den_fp32 is not None:
                # Keep an fp32 copy for stable NLMS-style beta normalization.
                k_prime_den_for_beta = k_prime_den_fp32
            if q_prime_den is not None and self.dual_map_low_precision:
                target_dtype = v.dtype
                if target_dtype in (torch.float16, torch.bfloat16):
                    q_prime_den = q_prime_den.to(target_dtype)
                    k_prime_den = k_prime_den.to(target_dtype)

        if self.state_update == "delta" and self.use_control_variate and self.use_hybrid_numerator:
            # Hybrid numerator map:
            # phi_num = [sqrt(alpha) * phi_cv, sqrt(1-alpha) * phi_pos]
            # The induced kernel is alpha*K_cv + (1-alpha)*K_pos (no cross terms).
            hybrid_features = max(1, int(round(self.num_features * self.hybrid_num_ratio)))
            if q_prime_den_fp32 is not None and q_prime_den_fp32.shape[-1] == hybrid_features:
                q_prime_pos_hybrid = q_prime_den_fp32
                k_prime_pos_hybrid = k_prime_den_fp32
            else:
                projection_matrix_hybrid = projection_matrix[:, :hybrid_features, :]
                q_prime_pos_hybrid = performer_softmax_feature_map(
                    q_kernel,
                    projection_matrix=projection_matrix_hybrid,
                    is_query=True,
                    eps=self.feature_eps,
                    cu_seqlens=cu_seqlens,
                )
                k_prime_pos_hybrid = performer_softmax_feature_map(
                    k_kernel,
                    projection_matrix=projection_matrix_hybrid,
                    is_query=False,
                    eps=self.feature_eps,
                    cu_seqlens=cu_seqlens,
                )
                q_prime_pos_hybrid = self._apply_feature_correction(q_prime_pos_hybrid, q_kernel_correction)
                k_prime_pos_hybrid = self._apply_feature_correction(k_prime_pos_hybrid, k_kernel_correction)

            hybrid_alpha = torch.sigmoid(self.hybrid_alpha_logit.float()).clamp(0.05, 0.95)
            if q.shape[2] != hybrid_alpha.shape[0]:
                if q.shape[2] % hybrid_alpha.shape[0] != 0:
                    raise ValueError(
                        f"Hybrid alpha head mismatch: q has {q.shape[2]} heads but alpha has {hybrid_alpha.shape[0]}."
                    )
                groups = q.shape[2] // hybrid_alpha.shape[0]
                hybrid_alpha = repeat(hybrid_alpha, 'h -> (h g)', g=groups)
            hybrid_cv_weight = hybrid_alpha.sqrt().view(1, 1, -1, 1).to(q_prime.dtype)
            hybrid_pos_weight = (1.0 - hybrid_alpha).sqrt().view(1, 1, -1, 1).to(q_prime.dtype)
            q_prime = torch.cat(
                [
                    q_prime * hybrid_cv_weight,
                    q_prime_pos_hybrid.to(q_prime.dtype) * hybrid_pos_weight,
                ],
                dim=-1,
            )
            k_prime = torch.cat(
                [
                    k_prime * hybrid_cv_weight,
                    k_prime_pos_hybrid.to(k_prime.dtype) * hybrid_pos_weight,
                ],
                dim=-1,
            )
        if self.feature_rms_norm:
            def _feature_rms_normalize(x: torch.Tensor) -> torch.Tensor:
                x32 = x.float()
                inv_rms = torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + float(self.feature_rms_norm_eps))
                return (x32 * inv_rms).to(x.dtype)

            q_prime = _feature_rms_normalize(q_prime)
            k_prime = _feature_rms_normalize(k_prime)
            if q_prime_den is not None and k_prime_den is not None:
                q_prime_den = _feature_rms_normalize(q_prime_den)
                k_prime_den = _feature_rms_normalize(k_prime_den)
                if q_prime_den_fp32 is not None:
                    q_prime_den_fp32 = q_prime_den.float()
                if k_prime_den_fp32 is not None:
                    k_prime_den_fp32 = k_prime_den.float()
                if k_prime_den_for_beta is not None:
                    k_prime_den_for_beta = k_prime_den.float()
        beta = None
        beta_den = None
        if self.use_beta:
            beta_logits = self.b_proj(hidden_states).float()
            beta = torch.sigmoid(beta_logits + self.beta_bias.view(1, 1, -1)).to(k_prime.dtype)
            if self.state_update == "sum":
                k_prime = k_prime * beta.unsqueeze(-1)
        if self.state_update == "delta":
            if beta is None:
                beta = torch.ones_like(k_prime[..., 0], dtype=k_prime.dtype)
            if self.delta_beta_norm:
                beta_base = beta.float()
                k_norm_sq_num = k_prime.float().square().sum(dim=-1)
                beta_num = (
                    beta_base / (k_norm_sq_num + self.delta_beta_norm_eps)
                ).clamp_(0.0, self.delta_beta_cap)
                if self.delta_decouple_beta:
                    k_norm_source_den = k_prime_den_for_beta if k_prime_den_for_beta is not None else k_prime.float()
                    k_norm_sq_den = k_norm_source_den.float().square().sum(dim=-1)
                    beta_den = (
                        beta_base / (k_norm_sq_den + self.delta_beta_norm_eps)
                    ).clamp_(0.0, self.delta_beta_cap).to(k_prime.dtype)
                beta = beta_num.to(k_prime.dtype)
            if beta_den is None:
                beta_den = beta

        if self.feature_pairwise_balance:
            q_prime, k_prime = self._pairwise_balance_features(
                q_prime,
                k_prime,
                eps=self.feature_pairwise_balance_eps,
                log_clip=self.feature_pairwise_balance_log_clip,
            )
            if q_prime_den is not None and k_prime_den is not None:
                q_prime_den, k_prime_den = self._pairwise_balance_features(
                    q_prime_den,
                    k_prime_den,
                    eps=self.feature_pairwise_balance_eps,
                    log_clip=self.feature_pairwise_balance_log_clip,
                )

        delta_log_decay = None
        if self.state_update == "delta" and self.delta_use_leaky_dplr:
            if beta is None:
                raise RuntimeError("delta_use_leaky_dplr requires beta to be defined.")
            rho = self.delta_leaky_rho_log.float().exp().clamp(1e-3, 8.0)
            rho = self._expand_head_vector(rho, beta.shape[-1], name="Delta leaky rho")
            delta_log_decay = -rho.view(1, 1, -1) * beta.float()
            min_log_lambda = math.log(float(self.delta_leaky_min_lambda))
            if min_log_lambda < 0:
                delta_log_decay = delta_log_decay.clamp(min=min_log_lambda, max=0.0)
            else:
                delta_log_decay = delta_log_decay.clamp(max=0.0)

        delta_denom_eps_effective = max(self.delta_denom_eps, self.delta_safe_denom_floor)

        if collect_error_stats:
            self.last_error_stats = self._collect_error_stats(
                q=q,
                k=k,
                q_prime=q_prime,
                k_prime=k_prime,
                q_prime_den=q_prime_den,
                k_prime_den=k_prime_den,
                beta=beta,
                adaptive_center=adaptive_center,
                dual_precondition=dual_precondition,
                adaptive_den_mix_alpha=adaptive_den_mix_alpha,
                den_ratio_effective=den_ratio_effective,
                den_map=den_map_effective,
                delta_denom_eps=delta_denom_eps_effective if self.state_update == "delta" else None,
            )
            if (
                self.use_error_feedback_den_ratio
                and self.training
                and self.state_update == "delta"
                and den_map_effective == "dual_softmax"
                and den_ratio_effective is not None
                and float(self.dual_map_den_ratio) < 1.0
            ):
                den_frac = self.last_error_stats.get("obs_kernel_err_den_frac", None)
                if isinstance(den_frac, (float, int)) and math.isfinite(float(den_frac)):
                    den_frac_val = min(1.0, max(0.0, float(den_frac)))
                    curr_ratio = float(self.error_feedback_den_ratio_ema.item())
                    ratio_target = curr_ratio * math.exp(
                        float(self.error_feedback_den_ratio_gain) * (den_frac_val - 0.5),
                    )
                    ratio_target = min(1.0, max(float(self.dual_map_den_ratio), ratio_target))
                    mom = float(self.error_feedback_den_ratio_momentum)
                    ratio_updated = mom * curr_ratio + (1.0 - mom) * ratio_target
                    ratio_updated = min(1.0, max(float(self.dual_map_den_ratio), ratio_updated))
                    self.error_feedback_den_ratio_ema.fill_(float(ratio_updated))
                    self.last_error_stats["obs_error_feedback_den_ratio_ema"] = float(ratio_updated)
                    self.last_error_stats["obs_error_feedback_den_frac"] = float(den_frac_val)

        log_decay = None
        if self.use_decay:
            decay_logits = self.decay_proj(hidden_states).float()
            decay_logits = decay_logits + self.decay_bias.view(1, 1, -1)
            log_decay = F.logsigmoid(decay_logits)

        initial_recurrent_state = recurrent_state
        if self.pdf_delta_update:
            if not _PERFORMER_PLUS_TRITON_AVAILABLE:
                raise RuntimeError("PerformerPlusLinearAttention requires Triton, but Triton kernel is unavailable.")
            if not q_prime.is_cuda:
                raise RuntimeError("PerformerPlusLinearAttention Triton path requires CUDA tensors.")
            o, recurrent_state = performer_plus_pdf_delta_attention_triton(
                q_prime=q_prime,
                k_prime=k_prime,
                v=v,
                q_prime_den=q_prime_den,
                k_prime_den=k_prime_den,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                eps=1e-6,
                denom_eps=delta_denom_eps_effective,
                denom_stopgrad=self.pdf_delta_denom_stopgrad,
            )
        else:
            if not _PERFORMER_PLUS_TRITON_AVAILABLE:
                raise RuntimeError("PerformerPlusLinearAttention requires Triton, but Triton kernel is unavailable.")
            if not q_prime.is_cuda:
                raise RuntimeError("PerformerPlusLinearAttention Triton path requires CUDA tensors.")
            o, recurrent_state = performer_plus_causal_linear_attention_triton(
                q_prime=q_prime,
                k_prime=k_prime,
                v=v,
                q_prime_den=q_prime_den,
                k_prime_den=k_prime_den,
                beta=beta,
                beta_den=beta_den,
                log_decay=log_decay,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                update_rule=self.state_update,
                delta_denom_eps=delta_denom_eps_effective,
                delta_smooth_denom=self.delta_smooth_denom,
                delta_denom_tau=self.delta_denom_tau,
                delta_log_decay=delta_log_decay,
                delta_denominator_update=self.delta_denominator_update,
                delta_denominator_stopgrad=self.delta_denominator_stopgrad,
            )
        if self.use_jackknife_debias and not self.pdf_delta_update:
            den_q_base = q_prime_den if q_prime_den is not None else q_prime
            den_k_base = k_prime_den if k_prime_den is not None else k_prime
            num_ranges = self._split_feature_ranges(q_prime.shape[-1], self.jackknife_groups)
            den_ranges = self._split_feature_ranges(den_q_base.shape[-1], self.jackknife_groups)
            split_groups = min(len(num_ranges), len(den_ranges))
            min_num_group = min((end - start) for start, end in num_ranges[:split_groups])
            min_den_group = min((end - start) for start, end in den_ranges[:split_groups])
            if (
                split_groups >= 2
                and min_num_group >= self.jackknife_min_per_group
                and min_den_group >= self.jackknife_min_per_group
            ):
                split_sum = None
                split_sq_sum = None
                for group_idx in range(split_groups):
                    n_start, n_end = num_ranges[group_idx]
                    d_start, d_end = den_ranges[group_idx]
                    q_num_chunk = q_prime[..., n_start:n_end]
                    k_num_chunk = k_prime[..., n_start:n_end]
                    q_den_chunk = den_q_base[..., d_start:d_end]
                    k_den_chunk = den_k_base[..., d_start:d_end]
                    state_chunk = self._slice_recurrent_state(
                        initial_recurrent_state,
                        num_start=n_start,
                        num_end=n_end,
                        den_start=d_start,
                        den_end=d_end,
                    )
                    o_chunk, _ = performer_plus_causal_linear_attention_triton(
                        q_prime=q_num_chunk,
                        k_prime=k_num_chunk,
                        v=v,
                        q_prime_den=q_den_chunk,
                        k_prime_den=k_den_chunk,
                        beta=beta,
                        beta_den=beta_den,
                        log_decay=log_decay,
                        initial_state=state_chunk,
                        output_final_state=False,
                        cu_seqlens=cu_seqlens,
                        update_rule=self.state_update,
                        delta_denom_eps=delta_denom_eps_effective,
                        delta_smooth_denom=self.delta_smooth_denom,
                        delta_denom_tau=self.delta_denom_tau,
                        delta_log_decay=delta_log_decay,
                        delta_denominator_update=self.delta_denominator_update,
                        delta_denominator_stopgrad=self.delta_denominator_stopgrad,
                    )
                    o_chunk32 = o_chunk.float()
                    if split_sum is None:
                        split_sum = o_chunk32
                    else:
                        split_sum = split_sum + o_chunk32
                    if self.use_jackknife_adaptive_shrinkage:
                        sq = o_chunk32.square()
                        if split_sq_sum is None:
                            split_sq_sum = sq
                        else:
                            split_sq_sum = split_sq_sum + sq
                if split_sum is not None:
                    split_mean = split_sum / float(split_groups)
                    coeff_full = float(split_groups) / float(split_groups - 1)
                    coeff_split = 1.0 / float(split_groups - 1)
                    o_full = o.float()
                    o_jk = o_full * coeff_full - split_mean * coeff_split
                    if self.use_jackknife_adaptive_shrinkage and split_sq_sum is not None:
                        split_second_moment = split_sq_sum / float(split_groups)
                        split_var = (split_second_moment - split_mean.square()).clamp_min_(0.0)
                        split_mean_var = split_var / float(split_groups)
                        correction = o_jk - o_full
                        correction_sq = correction.square()
                        shrink = correction_sq / (
                            correction_sq + split_mean_var + float(self.jackknife_shrinkage_eps)
                        )
                        o = o_full + shrink * correction
                    else:
                        o = o_jk
        if self.use_output_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)

        if past_key_values is not None and self.layer_idx is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if not self.use_output_gate:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        if o.dtype != self.o_proj.weight.dtype:
            o = o.to(self.o_proj.weight.dtype)
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values


__all__ = ['PerformerPlusLinearAttention']
