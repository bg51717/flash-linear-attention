from __future__ import annotations

import torch


def _sample_kernel_estimates(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_features: int,
    num_trials: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device='cpu').manual_seed(seed)
    plain = torch.empty(num_trials, dtype=torch.float64)
    control_variate = torch.empty(num_trials, dtype=torch.float64)

    q_norm_sq = 0.5 * torch.dot(q, q)
    k_norm_sq = 0.5 * torch.dot(k, k)
    qk = torch.dot(q, k)

    for idx in range(num_trials):
        omega = torch.randn(num_features, q.numel(), generator=g, dtype=torch.float64)
        oq = omega @ q
        ok = omega @ k

        fq = torch.exp(oq - q_norm_sq)
        fk = torch.exp(ok - k_norm_sq)
        plain[idx] = (fq * fk).mean()

        psi_q = fq - 1.0 - oq
        psi_k = fk - 1.0 - ok
        control_variate[idx] = 1.0 + qk + (psi_q * psi_k).mean()

    return plain, control_variate


def _sample_generalized_cv_estimates(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_features: int,
    num_trials: int,
    b: float,
    seed: int,
) -> torch.Tensor:
    if not (0.0 < b < 2.0):
        raise ValueError("b must be in (0, 2).")
    g = torch.Generator(device='cpu').manual_seed(seed)
    estimate = torch.empty(num_trials, dtype=torch.float64)

    q_norm_sq = 0.5 * torch.dot(q, q)
    k_norm_sq = 0.5 * torch.dot(k, k)
    qk = torch.dot(q, k)
    a_sq = 2.0 * b - b * b

    for idx in range(num_trials):
        omega = torch.randn(num_features, q.numel(), generator=g, dtype=torch.float64)
        oq = omega @ q
        ok = omega @ k

        fq = torch.exp(oq - q_norm_sq)
        fk = torch.exp(ok - k_norm_sq)
        psi_q = fq - 1.0 - b * oq
        psi_k = fk - 1.0 - b * ok
        estimate[idx] = 1.0 + a_sq * qk + (psi_q * psi_k).mean()

    return estimate


def test_control_variate_softmax_estimator_unbiased() -> None:
    torch.manual_seed(0)
    d = 32
    q = torch.randn(d, dtype=torch.float64)
    k = torch.randn(d, dtype=torch.float64)

    # Keep dot-product in a realistic range for attention after scaling.
    q = q / q.norm() * 0.9
    k = k / k.norm() * 0.8

    plain, control_variate = _sample_kernel_estimates(
        q=q,
        k=k,
        num_features=64,
        num_trials=4096,
        seed=2026,
    )
    target = torch.exp(torch.dot(q, k))

    # Both estimators are unbiased in expectation; finite-sample tolerance is loose.
    assert abs(float(plain.mean() - target)) < 3e-2
    assert abs(float(control_variate.mean() - target)) < 3e-2


def test_control_variate_reduces_estimator_variance() -> None:
    torch.manual_seed(1)
    d = 32
    q = torch.randn(d, dtype=torch.float64)
    k = torch.randn(d, dtype=torch.float64)
    q = q / q.norm() * 0.9
    k = k / k.norm() * 0.8

    plain, control_variate = _sample_kernel_estimates(
        q=q,
        k=k,
        num_features=32,
        num_trials=1024,
        seed=2027,
    )

    plain_var = float(plain.var(unbiased=True))
    control_variate_var = float(control_variate.var(unbiased=True))
    assert control_variate_var < plain_var * 0.2


def test_generalized_first_order_cv_unbiased() -> None:
    torch.manual_seed(2)
    d = 32
    q = torch.randn(d, dtype=torch.float64)
    k = torch.randn(d, dtype=torch.float64)
    q = q / q.norm() * 0.9
    k = k / k.norm() * 0.8
    target = torch.exp(torch.dot(q, k))

    for b in (0.6, 1.0, 1.4):
        est = _sample_generalized_cv_estimates(
            q=q,
            k=k,
            num_features=64,
            num_trials=4096,
            b=b,
            seed=2030 + int(100 * b),
        )
        assert abs(float(est.mean() - target)) < 3e-2


def main() -> None:
    test_control_variate_softmax_estimator_unbiased()
    test_control_variate_reduces_estimator_variance()
    test_generalized_first_order_cv_unbiased()
    print('ok: control-variate math tests passed')


if __name__ == '__main__':
    main()
