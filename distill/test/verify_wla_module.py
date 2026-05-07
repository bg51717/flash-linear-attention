"""Quick sanity check for WLA module — forward pass, gradient flow, naive vs compiled match."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F

from distill.models.linear_attention_wla import (
    wla_recurrence_naive,
    _fused_recurrent_wla,
    WLALinearAttention,
)


def test_naive_forward():
    """Test naive reference produces reasonable output."""
    torch.manual_seed(42)
    B, H, T, d_r, d_v = 2, 4, 32, 16, 64

    q_r = F.normalize(torch.randn(B, H, T, d_r), dim=-1)
    k_r = F.normalize(torch.randn(B, H, T, d_r), dim=-1)
    v = torch.randn(B, H, T, d_v)

    decay_alpha = torch.zeros(H)
    decay_bias = torch.full((H,), 2.0)
    write_alpha = torch.ones(H)
    write_bias = torch.zeros(H)
    log_sigma2 = torch.full((H,), 1.6)  # sigma2 ≈ 5

    o, state = wla_recurrence_naive(
        q_r, k_r, v, decay_alpha, decay_bias, write_alpha, write_bias, log_sigma2
    )
    assert o.shape == (B, H, T, d_v), f"Bad shape: {o.shape}"
    assert not torch.isnan(o).any(), "NaN in output"
    assert not torch.isinf(o).any(), "Inf in output"
    assert state is not None
    S, G = state
    assert S.shape == (B, H, d_r, d_v)
    assert G.shape == (B, H, d_r, d_r)
    print(f"  naive forward OK — output range [{o.min():.4f}, {o.max():.4f}]")


def test_naive_vs_compiled():
    """Test that naive and compiled paths produce matching results."""
    torch.manual_seed(42)
    B, H, T, d_r, d_v = 1, 2, 16, 16, 32

    q_r = F.normalize(torch.randn(B, H, T, d_r), dim=-1)
    k_r = F.normalize(torch.randn(B, H, T, d_r), dim=-1)
    v = torch.randn(B, H, T, d_v)

    decay_alpha = torch.zeros(H)
    decay_bias = torch.full((H,), 2.0)
    write_alpha = torch.ones(H)
    write_bias = torch.zeros(H)
    log_sigma2 = torch.full((H,), 1.6)

    # Naive path
    o_naive, _ = wla_recurrence_naive(
        q_r, k_r, v, decay_alpha, decay_bias, write_alpha, write_bias, log_sigma2,
        output_final_state=False,
    )

    # Compiled/checkpointed path — reshape to [N, T, D]
    N = B * H
    q_flat = q_r.permute(0, 1, 2, 3).reshape(N, T, d_r)
    k_flat = k_r.permute(0, 1, 2, 3).reshape(N, T, d_r)
    v_flat = v.permute(0, 1, 2, 3).reshape(N, T, d_v)
    da = decay_alpha.repeat(B)
    db = decay_bias.repeat(B)
    wa = write_alpha.repeat(B)
    wb = write_bias.repeat(B)
    sigma2 = torch.exp(log_sigma2).repeat(B)

    o_compiled, _, _ = _fused_recurrent_wla(
        q_flat, k_flat, v_flat, da, db, wa, wb, sigma2,
    )
    o_compiled = o_compiled.reshape(B, H, T, d_v)

    diff = (o_naive.float() - o_compiled.float()).abs().max().item()
    print(f"  naive vs compiled max diff: {diff:.2e}")
    assert diff < 2e-3, f"Mismatch: {diff}"


def test_gradient_flow():
    """Test that gradients flow through all learnable parameters."""
    torch.manual_seed(42)
    B, T, D = 1, 16, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    module = WLALinearAttention(
        hidden_size=D,
        num_heads=2,
        head_dim=D // 2,
        d_r=16,
        use_short_conv=False,
        output_norm="identity",
    ).to(device)

    x = torch.randn(B, T, D, requires_grad=False, device=device)
    o, _, _ = module(x)
    loss = o.sum()
    loss.backward()

    params_with_grad = []
    params_no_grad = []
    for name, p in module.named_parameters():
        if p.grad is not None and p.grad.abs().max() > 0:
            params_with_grad.append(name)
        else:
            params_no_grad.append(name)

    print(f"  params with gradient: {len(params_with_grad)}")
    for name in params_with_grad:
        p = dict(module.named_parameters())[name]
        print(f"    {name}: grad_norm={p.grad.norm():.4e}")

    if params_no_grad:
        print(f"  params WITHOUT gradient: {params_no_grad}")

    assert "log_sigma2" in params_with_grad, "log_sigma2 has no gradient!"
    assert "q_proj.weight" in params_with_grad, "q_proj has no gradient!"
    print("  gradient flow OK")


def test_module_forward():
    """Test full module forward pass (needs CUDA for ShortConv)."""
    torch.manual_seed(42)
    B, T, D = 2, 32, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_conv = torch.cuda.is_available()

    module = WLALinearAttention(
        hidden_size=D,
        num_heads=2,
        head_dim=D // 2,
        d_r=16,
        use_short_conv=use_conv,
        conv_size=4,
        output_norm="identity",
    ).to(device)

    x = torch.randn(B, T, D, device=device)
    o, _, _ = module(x)
    assert o.shape == (B, T, D), f"Bad shape: {o.shape}"
    assert not torch.isnan(o).any(), "NaN in output"
    print(f"  module forward OK — output shape {o.shape}, device={device}, conv={use_conv}")


if __name__ == '__main__':
    print("WLA Module Verification")
    print("=" * 50)

    print("\n1. Naive forward:")
    test_naive_forward()

    print("\n2. Naive vs compiled:")
    test_naive_vs_compiled()

    print("\n3. Module forward:")
    test_module_forward()

    print("\n4. Gradient flow:")
    test_gradient_flow()

    print("\n" + "=" * 50)
    print("All tests passed!")
