"""
Synthetic retrieval test for Wiener Linear Attention.
No training needed — directly tests whether whitened readout
improves retrieval from compressed state.
"""
import torch
import torch.nn.functional as F
import math


def build_state(keys, values, decay=0.99):
    """Accumulate keys/values into compressed state with exponential decay."""
    n, d_r = keys.shape
    d_v = values.shape[1]
    S = torch.zeros(d_r, d_v, dtype=keys.dtype, device=keys.device)
    G = torch.zeros(d_r, d_r, dtype=keys.dtype, device=keys.device)
    z = torch.zeros(d_r, dtype=keys.dtype, device=keys.device)
    for i in range(n):
        k = keys[i]
        v = values[i]
        S = decay * S + torch.outer(k, v)
        G = decay * G + torch.outer(k, k)
        z = decay * z + k
    return S, G, z


def readout_vanilla(S, G, z, q):
    """Standard linear attention readout."""
    return S.T @ q


def readout_wiener(S, G, z, q, sigma2):
    """Wiener-whitened readout: S^T (G + σ²I)^{-1} q"""
    d_r = G.shape[0]
    G_reg = G + sigma2 * torch.eye(d_r, dtype=G.dtype, device=G.device)
    q_w = torch.linalg.solve(G_reg, q)
    q_w = F.normalize(q_w, dim=0)
    return S.T @ q_w


def readout_wiener_neumann(S, G, z, q, sigma2, steps=3):
    """Wiener readout approximated via Neumann series (no matrix inverse).
    Convergence requires sigma2 > ||G||_op. We use spectral norm estimate."""
    G_spec = torch.linalg.eigvalsh(G)[-1].item()
    s2 = max(sigma2, G_spec * 1.2)
    q_w = q / s2
    Gq_power = q.clone()
    for m in range(1, steps + 1):
        Gq_power = G @ Gq_power
        q_w = q_w + ((-1) ** m) * Gq_power / (s2 ** (m + 1))
    q_w = F.normalize(q_w, dim=0)
    return S.T @ q_w


def readout_gala(S, G, z, q, lam):
    """GALA: q_tilde = norm(q + λ Gq)"""
    q_tilde = F.normalize(q + lam * (G @ q), dim=0)
    return S.T @ q_tilde


def run_experiment(n_tokens, d_r, d_v, noise_std=0.1, decay=0.99,
                   key_correlation=0.0, n_trials=200, device='cpu'):
    """
    Store n_tokens key-value pairs, then retrieve each one.
    Compare retrieval error across methods.

    key_correlation: 0.0 = random keys, higher = more correlated (harder)
    """
    results = {name: [] for name in ['vanilla', 'wiener', 'wiener_neumann', 'gala']}

    for trial in range(n_trials):
        torch.manual_seed(trial)

        if key_correlation > 0:
            base = torch.randn(max(1, int(d_r * (1 - key_correlation))), d_r, device=device)
            indices = torch.randint(0, base.shape[0], (n_tokens,))
            keys = base[indices] + torch.randn(n_tokens, d_r, device=device) * 0.3
        else:
            keys = torch.randn(n_tokens, d_r, device=device)
        keys = F.normalize(keys, dim=1)
        values = torch.randn(n_tokens, d_v, device=device)

        S, G, z = build_state(keys, values, decay=decay)

        # Pick a random target from the last 20% tokens (recent, less decayed)
        target_idx = torch.randint(int(n_tokens * 0.8), n_tokens, (1,)).item()
        k_target = keys[target_idx]
        v_target = values[target_idx]

        # Query = target key + noise (simulating imperfect query)
        q = F.normalize(k_target + torch.randn(d_r, device=device) * noise_std, dim=0)

        # Sigma^2 heuristic: trace(G) / d_r gives average eigenvalue
        sigma2 = G.trace().item() / d_r * 0.5

        o_vanilla = readout_vanilla(S, G, z, q)
        o_wiener = readout_wiener(S, G, z, q, sigma2)
        o_neumann = readout_wiener_neumann(S, G, z, q, sigma2, steps=3)
        o_gala = readout_gala(S, G, z, q, lam=1.0)

        # Measure cosine similarity with target value (direction matters more than scale)
        for name, o in [('vanilla', o_vanilla), ('wiener', o_wiener),
                        ('wiener_neumann', o_neumann), ('gala', o_gala)]:
            cos_sim = F.cosine_similarity(o.unsqueeze(0), v_target.unsqueeze(0)).item()
            results[name].append(cos_sim)

    return results


def print_results(results, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'Method':<20} {'Mean CosSim':>12} {'Std':>10} {'vs Vanilla':>12}")
    print(f"  {'-'*54}")
    vanilla_mean = sum(results['vanilla']) / len(results['vanilla'])
    for name in ['vanilla', 'wiener', 'wiener_neumann', 'gala']:
        vals = results[name]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean)**2 for v in vals) / len(vals)) ** 0.5
        diff = mean - vanilla_mean
        sign = '+' if diff >= 0 else ''
        print(f"  {name:<20} {mean:>12.4f} {std:>10.4f} {sign}{diff:>11.4f}")


if __name__ == '__main__':
    print("Synthetic retrieval test for Wiener Linear Attention")
    print("=" * 60)

    configs = [
        {"label": "Easy: 32 tokens, d_r=16, low correlation",
         "n_tokens": 32, "d_r": 16, "d_v": 64, "noise_std": 0.1,
         "key_correlation": 0.0},

        {"label": "Medium: 64 tokens, d_r=16, low correlation",
         "n_tokens": 64, "d_r": 16, "d_v": 64, "noise_std": 0.1,
         "key_correlation": 0.0},

        {"label": "Hard: 128 tokens, d_r=16, high correlation",
         "n_tokens": 128, "d_r": 16, "d_v": 64, "noise_std": 0.1,
         "key_correlation": 0.5},

        {"label": "Very hard: 256 tokens, d_r=16, high correlation",
         "n_tokens": 256, "d_r": 16, "d_v": 64, "noise_std": 0.1,
         "key_correlation": 0.5},

        {"label": "Extreme: 256 tokens, d_r=16, very high correlation",
         "n_tokens": 256, "d_r": 16, "d_v": 64, "noise_std": 0.1,
         "key_correlation": 0.8},

        {"label": "Noisy query: 64 tokens, noise=0.5",
         "n_tokens": 64, "d_r": 16, "d_v": 64, "noise_std": 0.5,
         "key_correlation": 0.0},

        {"label": "Large d_r: 128 tokens, d_r=64",
         "n_tokens": 128, "d_r": 64, "d_v": 64, "noise_std": 0.1,
         "key_correlation": 0.0},
    ]

    for cfg in configs:
        label = cfg.pop("label")
        res = run_experiment(**cfg)
        print_results(res, label)
