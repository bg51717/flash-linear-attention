from __future__ import annotations

import torch


def _pair_optimal_precondition(q: torch.Tensor, k: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    # Diagonal minimizer of ||Pq||^2 + ||P^{-1}k||^2 for fixed (q, k):
    # p_i = (k_i^2 / q_i^2)^(1/4).
    return ((k.square() + eps) / (q.square() + eps)).pow(0.25)


def _rf_estimates(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_features: int,
    num_trials: int,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator(device='cpu').manual_seed(seed)
    estimates = torch.empty(num_trials, dtype=torch.float64)
    q_norm_sq = 0.5 * torch.dot(q, q)
    k_norm_sq = 0.5 * torch.dot(k, k)
    for idx in range(num_trials):
        omega = torch.randn(num_features, q.numel(), generator=g, dtype=torch.float64)
        oq = omega @ q
        ok = omega @ k
        estimates[idx] = torch.exp(oq - q_norm_sq).mul(torch.exp(ok - k_norm_sq)).mean()
    return estimates


def test_dual_precondition_preserves_dot_product() -> None:
    torch.manual_seed(0)
    d = 48
    q = torch.randn(d, dtype=torch.float64)
    k = torch.randn(d, dtype=torch.float64)
    p = _pair_optimal_precondition(q, k)
    lhs = torch.dot(q * p, k / p)
    rhs = torch.dot(q, k)
    assert torch.allclose(lhs, rhs, atol=1e-10, rtol=1e-10)


def test_dual_precondition_reduces_rf_variance_anisotropic() -> None:
    torch.manual_seed(1)
    d = 64
    ratio = torch.logspace(-2, 2, d, dtype=torch.float64)
    ratio = ratio[torch.randperm(d)]

    q = torch.randn(d, dtype=torch.float64) * ratio
    k = torch.randn(d, dtype=torch.float64) / ratio
    q = q / q.norm()
    k = k / k.norm()
    p = _pair_optimal_precondition(q, k)

    plain = _rf_estimates(q, k, num_features=32, num_trials=1024, seed=2026)
    precond = _rf_estimates(q * p, k / p, num_features=32, num_trials=1024, seed=2026)

    target = torch.exp(torch.dot(q, k))
    assert abs(float(plain.mean() - target)) < 5e-2
    assert abs(float(precond.mean() - target)) < 5e-2

    plain_var = float(plain.var(unbiased=True))
    precond_var = float(precond.var(unbiased=True))
    assert precond_var < plain_var * 0.3


def main() -> None:
    test_dual_precondition_preserves_dot_product()
    test_dual_precondition_reduces_rf_variance_anisotropic()
    print('ok: dual-precondition math tests passed')


if __name__ == '__main__':
    main()
