import torch


def _mean_step(
    state_vk: torch.Tensor,
    mean_v: torch.Tensor,
    mean_k: torch.Tensor,
    count: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        state_vk: [B, H, V, K]
        mean_v: [B, H, V]
        mean_k: [B, H, K]
        count: [B, H]
        k_t: [B, H, K]
        v_t: [B, H, V]
    """
    delta_v = v_t - mean_v
    delta_k = k_t - mean_k
    state_vk = (
        state_vk
        + torch.einsum("bhv,bhk->bhvk", delta_v, k_t)
        + torch.einsum("bhv,bhk->bhvk", v_t, delta_k)
    )

    count_new = count + 1.0
    inv = count_new.reciprocal().unsqueeze(-1)
    mean_v = mean_v + (v_t - mean_v) * inv
    mean_k = mean_k + (k_t - mean_k) * inv
    return state_vk, mean_v, mean_k, count_new


def _mean_forward_fixed_len(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    # q/k/v: [B, T, H, K]/[B, T, H, K]/[B, T, H, V]
    B, T, H, K = q.shape
    V = v.shape[-1]

    qf = q.float()
    kf = k.float()
    vf = v.float()

    state = qf.new_zeros(B, H, V, K)
    mean_v = qf.new_zeros(B, H, V)
    mean_k = qf.new_zeros(B, H, K)
    count = qf.new_zeros(B, H)
    if initial_state is not None:
        state_in, mean_v_in, mean_k_in, count_in = initial_state
        state = state + state_in.float().transpose(-2, -1)
        mean_v = mean_v + mean_v_in.float()
        mean_k = mean_k + mean_k_in.float()
        count = count + count_in.float()

    out = vf.new_zeros(B, T, H, V)
    for t in range(T):
        state, mean_v, mean_k, count = _mean_step(
            state_vk=state,
            mean_v=mean_v,
            mean_k=mean_k,
            count=count,
            k_t=kf[:, t],
            v_t=vf[:, t],
        )
        out[:, t] = torch.einsum("bhvk,bhk->bhv", state, qf[:, t]) * scale

    final_state = None
    if output_final_state:
        final_state = (
            state.transpose(-2, -1).contiguous(),
            mean_v.contiguous(),
            mean_k.contiguous(),
            count.contiguous(),
        )
    return out.to(q.dtype), final_state


def mean_delta_rule_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """
    Reference implementation for mean-delta recurrence:
      S_t = S_{t-1} + (v_t - mean_{v,t-1}) k_t^T + v_t (k_t^T - mean_{k,t-1}^T)
    with output:
      o_t = S_t q_t

    External state layout follows DeltaNet convention for matrix state:
      kv_state: [N, H, K, V]
      mean_v: [N, H, V]
      mean_k: [N, H, K]
      count: [N, H]
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        if scale <= 0:
            raise ValueError("scale must be positive.")

    if cu_seqlens is None:
        return _mean_forward_fixed_len(
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
        final_state = (
            q.new_empty(n_seq, q.shape[2], q.shape[-1], v.shape[-1], dtype=torch.float32),
            q.new_empty(n_seq, q.shape[2], v.shape[-1], dtype=torch.float32),
            q.new_empty(n_seq, q.shape[2], q.shape[-1], dtype=torch.float32),
            q.new_empty(n_seq, q.shape[2], dtype=torch.float32),
        )

    for i in range(n_seq):
        bos = int(cu[i].item())
        eos = int(cu[i + 1].item())
        init_i = None
        if initial_state is not None:
            init_i = tuple(x[i:i + 1] for x in initial_state)
        out_i, st_i = _mean_forward_fixed_len(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            scale=scale,
            initial_state=init_i,
            output_final_state=output_final_state,
        )
        out[:, bos:eos] = out_i
        if output_final_state:
            assert st_i is not None and final_state is not None
            final_state[0][i] = st_i[0][0]
            final_state[1][i] = st_i[1][0]
            final_state[2][i] = st_i[2][0]
            final_state[3][i] = st_i[3][0]

    return out, final_state
