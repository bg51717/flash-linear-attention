import torch


def _dual_step(state_vk: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
    """
    Args:
        state_vk: [B, H, V, K]
        k_t: [B, H, K]
        v_t: [B, H, V]
    """
    a_t = torch.einsum("bhvk,bhk->bhv", state_vk, k_t)
    b_t = torch.einsum("bhvk,bhv->bhk", state_vk, v_t)
    x_t = v_t - a_t
    y_t = k_t - b_t
    return (
        state_vk
        + torch.einsum("bhv,bhk->bhvk", x_t, k_t)
        + torch.einsum("bhv,bhk->bhvk", v_t, y_t)
    )


def _dual_forward_fixed_len(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # q/k/v: [B, T, H, K]/[B, T, H, K]/[B, T, H, V]
    B, T, H, K = q.shape
    V = v.shape[-1]

    qf = q.float()
    kf = k.float()
    vf = v.float()

    state = qf.new_zeros(B, H, V, K)
    if initial_state is not None:
        # external state is [B, H, K, V]
        state = state + initial_state.float().transpose(-2, -1)

    out = vf.new_zeros(B, T, H, V)
    for t in range(T):
        state = _dual_step(state, kf[:, t], vf[:, t])
        out[:, t] = torch.einsum("bhvk,bhk->bhv", state, qf[:, t]) * scale

    final_state = state.transpose(-2, -1).contiguous() if output_final_state else None
    return out.to(q.dtype), final_state


def dual_delta_rule_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Reference implementation for the dual-symmetric Delta rule:
      S_t = S_{t-1} + (v_t - S_{t-1} k_t) k_t^T + v_t (k_t^T - v_t^T S_{t-1})
    with output:
      o_t = S_t q_t

    State shape follows FLA DeltaNet convention externally: [N, H, K, V].
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        if scale <= 0:
            raise ValueError("scale must be positive.")

    if cu_seqlens is None:
        return _dual_forward_fixed_len(
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    if q.shape[0] != 1:
        raise ValueError(
            f"Expected batch size 1 when cu_seqlens is provided, got {q.shape[0]}."
        )

    cu = cu_seqlens.to(torch.long)
    n_seq = cu.numel() - 1
    out = torch.zeros_like(v)
    final_state = None
    if output_final_state:
        final_state = q.new_empty(n_seq, q.shape[2], q.shape[-1], v.shape[-1], dtype=torch.float32)

    for i in range(n_seq):
        bos = int(cu[i].item())
        eos = int(cu[i + 1].item())
        init_i = None if initial_state is None else initial_state[i: i + 1]
        out_i, ht_i = _dual_forward_fixed_len(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            scale=scale,
            initial_state=init_i,
            output_final_state=output_final_state,
        )
        out[:, bos:eos] = out_i
        if output_final_state:
            final_state[i] = ht_i[0]

    return out, final_state

