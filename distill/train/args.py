from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    linear_attention_type: str = field(
        default="gated_deltanet",
        metadata={"help": "Type of attention mechanism to use, e.g. gated_deltanet, delta_net, first_order_linear_attention, pdf_final_linear_attention (or pdf_refined_linear_attention, final simplified recurrence from Approximating Self-attention.pdf), taylor_linear_attention (or softmax_taylor_linear_attention, using exp(q^T k) ~= 1 + q^T k), original_linear_attention (or vanilla_linear_attention, backed by FLA official LinearAttention with identity feature map), mha_attention (or mha, strict FLA official), mha_torch_attention (fallback), mha_torch_qk_unit_norm_attention (or mha_torch_qk_unit_norm, L2-normalize q/k to unit norm after q_proj/k_proj), performer_linear_attention, performer_plus_linear_attention."}
    )
    pdf_final_use_triton: bool = field(
        default=True,
        metadata={"help": "Use the Triton forward kernel for pdf_final_linear_attention when training without cache."}
    )
    performer_nb_features: int = field(
        default=None,
        metadata={"help": "Number of random features for Performer/Performer+ kernels. None keeps model default."}
    )
    performer_redraw_projection: bool = field(
        default=False,
        metadata={"help": "Redraw Performer/Performer+ projection matrix during training."}
    )
    performer_projection_seed: int = field(
        default=0,
        metadata={"help": "Base random seed used to initialize Performer/Performer+ projection matrices."}
    )
    performer_adaptive_center_sampling: bool = field(
        default=False,
        metadata={"help": "Enable input-adaptive centered random-feature sampling for Performer+ (shared head-wise shift c)."}
    )
    performer_adaptive_center_momentum: float = field(
        default=0.9,
        metadata={"help": "EMA momentum for adaptive center c in [0,1). Lower means more per-batch adaptation."}
    )
    performer_adaptive_center_log_clip: float = field(
        default=12.0,
        metadata={"help": "Clamp range for adaptive-center log-correction before exp; 0 disables clipping."}
    )
    performer_use_dual_precondition_sampling: bool = field(
        default=False,
        metadata={"help": "Enable dual-side input preconditioning for Performer+ sampling: q->Pq, k->P^{-1}k (kernel-preserving)."}
    )
    performer_precondition_momentum: float = field(
        default=0.9,
        metadata={"help": "EMA momentum for Performer+ dual-precondition scale statistics in [0,1)."}
    )
    performer_precondition_eps: float = field(
        default=1.0e-4,
        metadata={"help": "Epsilon for Performer+ dual-precondition RMS statistics."}
    )
    performer_precondition_log_clip: float = field(
        default=2.0,
        metadata={"help": "Clamp range for Performer+ dual-precondition log-scales; 0 disables clipping."}
    )
    performer_precondition_mode: str = field(
        default="diag",
        metadata={"help": "Performer+ dual-precondition mode: 'diag' (channel-wise) or 'full' (full SPD covariance balancing)."}
    )
    performer_learnable_kernel_scale: bool = field(
        default=True,
        metadata={"help": "Enable per-head learnable kernel temperature for Performer+."}
    )
    performer_dim_aware_kernel_scale: bool = field(
        default=True,
        metadata={"help": "Use d^(1/4) base scaling for Performer+ kernel temperature when q/k are L2-normalized."}
    )
    performer_kernel_scale_init: float = field(
        default=1.0,
        metadata={"help": "Initial kernel temperature for Performer+ (alpha > 0)."}
    )
    performer_use_deterministic_nodes: bool = field(
        default=False,
        metadata={"help": "Enable deterministic orthogonal nodes mixed with random projections for Performer+."}
    )
    performer_stratified_norm_sampling: bool = field(
        default=False,
        metadata={"help": "Use stratified (low-variance) chi-radius sampling for Performer+ random features when performer_ortho_scaling=0."}
    )
    performer_stratified_jitter: bool = field(
        default=True,
        metadata={"help": "Enable per-head random jitter for stratified Performer+ radial sampling."}
    )
    performer_qmc_gaussian_sampling: bool = field(
        default=False,
        metadata={"help": "Use Sobol quasi-Monte Carlo Gaussian sampling for Performer/Performer+ projection construction."}
    )
    performer_qmc_scramble: bool = field(
        default=True,
        metadata={"help": "Enable Owen-style scrambling for Sobol QMC Gaussian sampling."}
    )
    performer_use_landmark_sampling: bool = field(
        default=False,
        metadata={"help": "Use learnable deterministic landmark features mixed with random features (adaptive sampling)." }
    )
    performer_landmark_ratio: float = field(
        default=0.25,
        metadata={"help": "Feature ratio allocated to landmark nodes when adaptive sampling is enabled (0, 1]."}
    )
    performer_landmark_alpha_init: float = field(
        default=0.35,
        metadata={"help": "Initial mixture weight alpha in (0,1) for landmark-kernel blending."}
    )
    performer_landmark_in_denominator: bool = field(
        default=True,
        metadata={"help": "Also blend landmark features into Performer+ dual-map denominator."}
    )
    performer_qk_l2_norm: bool = field(
        default=True,
        metadata={"help": "Apply L2 normalization to q/k before Performer+ feature mapping."}
    )
    performer_use_control_variate: bool = field(
        default=True,
        metadata={"help": "Enable control-variate kernel decomposition in Performer+."}
    )
    performer_use_adaptive_linear_cv: bool = field(
        default=False,
        metadata={"help": "Enable generalized first-order control variate with learnable per-head linear coefficient b in (0,2)."}
    )
    performer_adaptive_linear_cv_init: float = field(
        default=1.0,
        metadata={"help": "Initial per-head linear CV coefficient b for generalized first-order CV (0,2). b=1 recovers standard CV."}
    )
    performer_adaptive_linear_cv_eps: float = field(
        default=1.0e-4,
        metadata={"help": "Epsilon clamp for adaptive linear CV coefficient and induced deterministic scale."}
    )
    performer_feature_rms_norm: bool = field(
        default=False,
        metadata={"help": "Apply RMS normalization over Performer+ feature channels for q/k feature maps."}
    )
    performer_feature_rms_norm_eps: float = field(
        default=1.0e-4,
        metadata={"help": "Epsilon for Performer+ feature RMS normalization."}
    )
    performer_feature_pairwise_balance: bool = field(
        default=False,
        metadata={"help": "Apply channel-wise pairwise balancing to (q', k') features while preserving dot-product kernel exactly."}
    )
    performer_feature_pairwise_balance_eps: float = field(
        default=1.0e-4,
        metadata={"help": "Epsilon for pairwise feature balancing statistics."}
    )
    performer_feature_pairwise_balance_log_clip: float = field(
        default=2.0,
        metadata={"help": "Clamp range for pairwise balancing log-scales; 0 disables clipping."}
    )
    performer_control_variate_exp_clip: float = field(
        default=20.0,
        metadata={"help": "Upper clip applied to control-variate exp exponent for Performer+ stability."}
    )
    performer_enable_error_observability: bool = field(
        default=False,
        metadata={"help": "Enable Performer+ approximation-error observability metrics in forward pass."}
    )
    performer_error_observe_interval: int = field(
        default=10,
        metadata={"help": "Collect Performer+ observability stats every N forwards (>=1)."}
    )
    performer_error_observe_max_tokens: int = field(
        default=32,
        metadata={"help": "Maximum token count used for lightweight kernel-error probes."}
    )
    performer_error_observe_max_heads: int = field(
        default=2,
        metadata={"help": "Maximum head count used for lightweight kernel-error probes."}
    )
    performer_learnable_projection: bool = field(
        default=False,
        metadata={"help": "Enable learnable Performer+ projection matrix (data-adaptive kernel features)."}
    )
    performer_learnable_projection_scale: bool = field(
        default=False,
        metadata={"help": "Enable per-row learnable scale for Performer+ learnable projection."}
    )
    performer_deterministic_ratio: float = field(
        default=0.25,
        metadata={"help": "Feature ratio allocated to deterministic nodes when enabled (0, 1]."}
    )
    performer_ortho_scaling: int = field(
        default=0,
        metadata={"help": "Performer orthogonal random-feature scaling mode: 0=chi-radial, 1=fixed sqrt(d) radial."}
    )
    performer_use_projection_ensemble: bool = field(
        default=False,
        metadata={"help": "Use grouped projection ensemble for Performer+ control-variate stochastic features."}
    )
    performer_projection_ensemble_groups: int = field(
        default=2,
        metadata={"help": "Number of grouped projection ensembles when enabled (>=2)."}
    )
    performer_state_update: str = field(
        default="sum",
        metadata={"help": "Performer+ state update rule: 'sum', 'delta', or 'pdf_delta'/'overwrite' (PDF delta rule from performer+delta.pdf)."}
    )
    performer_use_beta: bool = field(
        default=False,
        metadata={"help": "Enable beta gating for Performer+ state update."}
    )
    performer_use_value_gate: bool = field(
        default=False,
        metadata={"help": "Enable value gate for Performer+ output values."}
    )
    performer_use_output_gate: bool = field(
        default=True,
        metadata={"help": "Enable output gate for Performer+ attention output."}
    )
    performer_use_decay: bool = field(
        default=True,
        metadata={"help": "Enable explicit decay for Performer+ (only effective when performer_state_update='sum')."}
    )
    performer_delta_beta_norm: bool = field(
        default=True,
        metadata={"help": "Normalize delta beta by key-feature norm for Performer+ delta update."}
    )
    performer_delta_beta_norm_eps: float = field(
        default=1.0e-3,
        metadata={"help": "Epsilon used in Performer+ delta beta normalization denominator."}
    )
    performer_delta_denom_eps: float = field(
        default=1.0e-3,
        metadata={"help": "Denominator clamp epsilon for Performer+ delta recurrence output."}
    )
    performer_delta_smooth_denom: bool = field(
        default=False,
        metadata={"help": "Use smooth softplus barrier instead of hard clamp for Performer+ delta denominator stabilization."}
    )
    performer_delta_denom_tau: float = field(
        default=1.0e-2,
        metadata={"help": "Softplus barrier temperature tau for Performer+ delta denominator stabilization."}
    )
    performer_delta_beta_cap: float = field(
        default=0.25,
        metadata={"help": "Upper bound for normalized delta step size (NLMS-style cap) in Performer+, typically in (0, 2]."}
    )
    performer_delta_safe_denom_floor: float = field(
        default=1.0e-3,
        metadata={"help": "Safety floor applied to delta denominator epsilon for numerical stability."}
    )
    performer_delta_decouple_beta: bool = field(
        default=False,
        metadata={"help": "Use separate NLMS-normalized beta for numerator and denominator recurrences in Performer+ delta mode."}
    )
    performer_delta_denominator_update: str = field(
        default="delta",
        metadata={"help": "Performer+ delta denominator recurrence: 'delta' (error-correcting) or 'sum' (positive accumulation)."}
    )
    performer_delta_denominator_map: str = field(
        default="auto",
        metadata={
            "help": "Performer+ delta denominator feature map: 'auto' (legacy), 'dual_softmax', 'abs', 'softplus', 'numerator', or 'positive_linear'."
        }
    )
    performer_delta_denominator_stopgrad: bool = field(
        default=False,
        metadata={"help": "When denominator update is 'sum', stop gradients through denominator in the final ratio for stability."}
    )
    performer_delta_use_leaky_dplr: bool = field(
        default=False,
        metadata={"help": "Use leaky delta recurrence implemented via DPLR: S_t=lambda_t S_{t-1}+beta_t(v_t-S_{t-1}k_t)k_t^T."}
    )
    performer_delta_leaky_rho_init: float = field(
        default=1.0,
        metadata={"help": "Initial per-head leaky coefficient rho (>0) for delta lambda_t=exp(-rho*beta_t)."}
    )
    performer_delta_leaky_min_lambda: float = field(
        default=0.5,
        metadata={"help": "Minimum allowed leaky delta contraction lambda in (0,1]."}
    )
    performer_use_hybrid_numerator: bool = field(
        default=False,
        metadata={"help": "Use hybrid delta numerator kernel: concat weighted control-variate and positive maps."}
    )
    performer_hybrid_num_ratio: float = field(
        default=0.25,
        metadata={"help": "Positive-feature ratio used by hybrid delta numerator map (0, 1]."}
    )
    performer_hybrid_alpha_init: float = field(
        default=0.75,
        metadata={"help": "Initial control-variate mixture weight alpha for hybrid delta numerator, in (0, 1)."}
    )
    performer_use_cv_residual_shrinkage: bool = field(
        default=False,
        metadata={"help": "Apply learnable shrinkage on control-variate random residual features."}
    )
    performer_cv_residual_init: float = field(
        default=0.75,
        metadata={"help": "Initial residual shrinkage gamma in (0,1] for control-variate features."}
    )
    performer_cv_finite_sample_orthogonalize: bool = field(
        default=False,
        metadata={"help": "Apply finite-sample orthogonalization to control-variate residual channels against low-order deterministic bases."}
    )
    performer_cv_finite_sample_orth_eps: float = field(
        default=1.0e-4,
        metadata={"help": "Epsilon for finite-sample orthogonalization denominators."}
    )
    performer_use_jackknife_debias: bool = field(
        default=False,
        metadata={"help": "Apply jackknife debiasing on the Performer+ ratio estimator to cancel leading finite-feature bias."}
    )
    performer_jackknife_groups: int = field(
        default=2,
        metadata={"help": "Number of feature partitions for jackknife debiasing (>=2)."}
    )
    performer_jackknife_min_per_group: int = field(
        default=8,
        metadata={"help": "Minimum feature channels per partition required to enable jackknife debiasing."}
    )
    performer_use_jackknife_adaptive_shrinkage: bool = field(
        default=False,
        metadata={"help": "Apply adaptive shrinkage on jackknife correction based on split-estimator variance."}
    )
    performer_jackknife_shrinkage_eps: float = field(
        default=1.0e-5,
        metadata={"help": "Epsilon for jackknife adaptive-shrinkage denominator."}
    )
    performer_use_second_order_cv: bool = field(
        default=False,
        metadata={"help": "Use second-order Hermite control variate decomposition for Performer+ kernel features."}
    )
    performer_use_cv_decoupled_second_order: bool = field(
        default=False,
        metadata={"help": "Use independent feature budgets for second-order Hermite and residual branches in CV2."}
    )
    performer_cv_decoupled_h2_ratio: float = field(
        default=0.25,
        metadata={"help": "In decoupled CV2, fraction of RF rows allocated to the h2 branch (0,1)."}
    )
    performer_cv_decoupled_h2_deterministic: bool = field(
        default=False,
        metadata={"help": "Use deterministic orthogonal nodes for the decoupled CV2 h2 branch instead of random rows."}
    )
    performer_use_cv_decoupled_adaptive_h2_ratio: bool = field(
        default=False,
        metadata={"help": "Use adaptive EMA-updated h2/residual budget ratio in decoupled CV2."}
    )
    performer_cv_decoupled_ratio_ema_momentum: float = field(
        default=0.9,
        metadata={"help": "EMA momentum for adaptive decoupled CV2 h2 ratio in [0,1)."}
    )
    performer_cv_decoupled_ratio_min: float = field(
        default=0.05,
        metadata={"help": "Lower clamp for adaptive decoupled CV2 h2 ratio."}
    )
    performer_cv_decoupled_ratio_max: float = field(
        default=0.5,
        metadata={"help": "Upper clamp for adaptive decoupled CV2 h2 ratio."}
    )
    performer_cv_split_feature_budget: bool = field(
        default=True,
        metadata={"help": "When second/third-order CV is enabled, split RF rows to keep total feature width near baseline. Disable to keep full RF rows for lower-variance CV."}
    )
    performer_use_third_order_cv: bool = field(
        default=False,
        metadata={"help": "Use third-order Hermite control variate (includes h2+h3) for Performer+ kernel features."}
    )
    performer_use_diag2_term: bool = field(
        default=False,
        metadata={"help": "Blend a low-rank deterministic diagonal second-order term into Performer+ numerator kernel."}
    )
    performer_diag2_ratio: float = field(
        default=0.25,
        metadata={"help": "Feature-space ratio for diagonal second-order subspace (0,1]."}
    )
    performer_diag2_alpha_init: float = field(
        default=0.1,
        metadata={"help": "Initial blend weight alpha in (0,1) for diagonal second-order kernel component."}
    )
    performer_use_dual_map: bool = field(
        default=True,
        metadata={"help": "Use positive dual-map denominator in Performer+ delta mode when control variate is enabled."}
    )
    performer_dual_map_den_ratio: float = field(
        default=0.25,
        metadata={"help": "Feature ratio for dual-map denominator (0, 1]. Smaller value reduces memory."}
    )
    performer_dual_map_row_selection: str = field(
        default="auto",
        metadata={"help": "Row-selection strategy when sub-sampling dual-map denominator features: auto|head|strided|antithetic_balanced."}
    )
    performer_use_layerwise_den_ratio: bool = field(
        default=False,
        metadata={"help": "Use layer-dependent denominator ratio schedule in Performer+ delta dual-map: deeper layers receive higher den feature budget."}
    )
    performer_layerwise_den_ratio_tau: float = field(
        default=8.0,
        metadata={"help": "Depth time-constant (in layers) for layerwise denominator ratio schedule; larger means slower increase."}
    )
    performer_use_error_feedback_den_ratio: bool = field(
        default=False,
        metadata={"help": "Enable error-feedback adaptation of Performer+ denominator feature ratio using observed num/den approximation error split."}
    )
    performer_error_feedback_den_ratio_momentum: float = field(
        default=0.9,
        metadata={"help": "EMA momentum for error-feedback denominator ratio adaptation in [0,1)."}
    )
    performer_error_feedback_den_ratio_gain: float = field(
        default=0.5,
        metadata={"help": "Multiplicative gain for error-feedback denominator ratio updates (>0)."}
    )
    performer_use_adaptive_den_mix: bool = field(
        default=False,
        metadata={"help": "Enable adaptive mixing for extra denominator softmax features beyond performer_dual_map_den_ratio in Performer+ delta mode."}
    )
    performer_adaptive_den_mix_init: float = field(
        default=0.0,
        metadata={"help": "Initial per-head mix weight alpha in [0,1] for adaptive extra denominator features."}
    )
    performer_dual_map_low_precision: bool = field(
        default=True,
        metadata={"help": "Cast dual-map denominator features to activation dtype to reduce memory."}
    )
    performer_use_den_poly_kernel: bool = field(
        default=False,
        metadata={"help": "Blend positive deterministic linear kernel features into Performer+ dual-map denominator."}
    )
    performer_den_poly_alpha_init: float = field(
        default=0.15,
        metadata={"help": "Initial mixture weight alpha in (0,1) for denominator positive linear-kernel blend."}
    )
    performer_den_poly_constant: float = field(
        default=2.0,
        metadata={"help": "Positive linear-kernel constant c (>0) for denominator map: K=c+<q,k>/c."}
    )
    performer_den_poly_ratio: float = field(
        default=0.5,
        metadata={"help": "Fraction of denominator channels used by low-rank positive linear kernel branch (0,1]."}
    )
    teacher_model: str = field(
        default=None,
        metadata={"help": "Path to the teacher model for distillation."}
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    seq_len: int = field(
        default=2048,
        metadata={"help": "The sequence length for training."}
    )
