#!/usr/bin/env python3.12
"""D3: Stabilization raw convergence matrix + Cochran's Q test.
D4: Convergence trajectory analysis + what predicts convergence at init."""
import json, numpy as np
from phasegrad.kuramoto import make_network, kuramoto_jacobian

# D3: Extract binary convergence matrix from stabilization data
def d3_convergence_matrix():
    """20×5 binary convergence matrix + Cochran's Q test."""
    with open('experiments/stabilization_results.json') as f:
        data = json.load(f)

    variants = list(data.keys())
    n_seeds = len(data[variants[0]])

    matrix = np.zeros((n_seeds, len(variants)), dtype=int)
    for j, v in enumerate(variants):
        for i, acc in enumerate(data[v]):
            matrix[i, j] = 1 if acc > 0.60 else 0

    print("D3: Binary Convergence Matrix (1=converged, 0=failed)")
    print(f"    {'':>6s} " + " ".join(f"{v[:10]:>10s}" for v in variants))
    for i in range(n_seeds):
        row = " ".join(f"{matrix[i,j]:>10d}" for j in range(len(variants)))
        print(f"    seed {i:2d} {row}")

    # Check if all columns are identical
    all_same = all(np.array_equal(matrix[:, 0], matrix[:, j])
                   for j in range(1, len(variants)))
    print(f"\n    All columns identical: {all_same}")

    # Cochran's Q test (manual — scipy doesn't have it)
    # Q = (k-1) * (k * Σ_j T_j² - T²) / (k * T - Σ_i L_i²)
    # where T_j = column sum, L_i = row sum, T = grand total, k = n_variants
    k = len(variants)
    T_j = matrix.sum(axis=0)  # column sums
    L_i = matrix.sum(axis=1)  # row sums
    T = matrix.sum()

    num = (k - 1) * (k * np.sum(T_j**2) - T**2)
    den = k * T - np.sum(L_i**2)
    if den > 0:
        Q = num / den
        # Under H0 (all variants same), Q ~ chi²(k-1)
        from scipy.stats import chi2
        p = 1 - chi2.cdf(Q, k - 1)
        print(f"    Cochran's Q = {Q:.4f}, df = {k-1}, p = {p:.4f}")
        print(f"    {'Variants differ' if p < 0.05 else 'No difference between variants'}")
    else:
        Q, p = 0.0, 1.0
        print(f"    Cochran's Q = 0 (degenerate — all identical)")

    return {'matrix': matrix.tolist(), 'variants': variants,
            'all_identical': bool(all_same), 'Q': float(Q), 'p': float(p)}


# D4: What distinguishes converged from failed seeds at initialization?
def d4_init_features():
    """Compute initialization features and correlate with convergence."""
    with open('experiments/convergence_diagnosis.json') as f:
        diag = json.load(f)

    features = []
    for r in diag:
        seed = r['seed']
        converged = r['converged']
        net = make_network(n_input=2, n_hidden=5, n_output=2,
                           K_scale=2.0, input_scale=1.5, seed=seed)

        K = net.K
        # Graph Laplacian of the coupling matrix
        K_binary = (K > 0).astype(float)
        degree = K_binary.sum(axis=1)
        L = np.diag(degree) - K_binary
        evals = np.sort(np.linalg.eigvalsh(L))
        algebraic_connectivity = float(evals[1])  # 2nd smallest eigenvalue (Fiedler)
        spectral_gap = float(evals[1] / evals[-1]) if evals[-1] > 0 else 0

        # Weighted Laplacian (using actual K values)
        K_sym = (K + K.T) / 2  # symmetrize
        degree_w = K_sym.sum(axis=1)
        L_w = np.diag(degree_w) - K_sym
        evals_w = np.sort(np.linalg.eigvalsh(L_w))
        alg_conn_weighted = float(evals_w[1])

        # Frequency features
        omega_learnable = [net.omega[i] for i in net.learnable_ids]
        omega_spread = float(np.std(omega_learnable))
        omega_range = float(np.ptp(omega_learnable))

        # Coupling features
        K_nonzero = K[K > 0]
        mean_K = float(K_nonzero.mean())
        min_K = float(K_nonzero.min())
        n_edges = int((K > 0).sum()) // 2

        # Initial loss + gradient norm (need to run one forward pass)
        from phasegrad.data import load_hillenbrand
        from phasegrad.losses import mse_loss, mse_target
        from phasegrad.gradient import analytical_gradient
        tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
        x, cls = tr[0]
        net.set_input(x)
        theta, res = net.equilibrium()
        target = mse_target(net.N, net.output_ids, cls, margin=0.2)
        init_loss = mse_loss(theta, target, net.output_ids)
        grad = analytical_gradient(net, theta, target)
        grad_norm = float(np.linalg.norm(grad[1:]))

        features.append({
            'seed': seed, 'converged': converged,
            'algebraic_connectivity': algebraic_connectivity,
            'alg_conn_weighted': alg_conn_weighted,
            'spectral_gap': spectral_gap,
            'omega_spread': omega_spread,
            'omega_range': omega_range,
            'mean_K': mean_K, 'min_K': min_K,
            'n_edges': n_edges,
            'init_loss': float(init_loss),
            'init_grad_norm': grad_norm,
            'init_residual': float(res),
        })

    # Correlation with convergence (point-biserial)
    from scipy.stats import pointbiserialr
    conv_binary = np.array([f['converged'] for f in features], dtype=float)

    print("\nD4: Feature-Convergence Correlations (40 seeds)")
    print(f"    {'feature':30s} {'conv_mean':>10s} {'fail_mean':>10s} {'r_pb':>8s} {'p':>8s}")

    feature_names = ['algebraic_connectivity', 'alg_conn_weighted', 'spectral_gap',
                     'omega_spread', 'omega_range', 'mean_K', 'min_K',
                     'n_edges', 'init_loss', 'init_grad_norm']
    corr_results = {}

    for fname in feature_names:
        vals = np.array([f[fname] for f in features])
        conv_vals = vals[conv_binary == 1]
        fail_vals = vals[conv_binary == 0]
        r, p = pointbiserialr(conv_binary, vals)
        print(f"    {fname:30s} {np.mean(conv_vals):10.4f} {np.mean(fail_vals):10.4f} "
              f"{r:+8.4f} {p:8.4f}{'*' if p < 0.05 else ' '}")
        corr_results[fname] = {'r': float(r), 'p': float(p),
                               'conv_mean': float(np.mean(conv_vals)),
                               'fail_mean': float(np.mean(fail_vals))}

    return {'features': features, 'correlations': corr_results}


if __name__ == '__main__':
    d3_result = d3_convergence_matrix()
    d4_result = d4_init_features()

    with open('experiments/convergence_matrix_results.json', 'w') as f:
        json.dump({'d3': d3_result, 'd4': d4_result}, f, indent=2, default=str)
    print(f"\nSaved to experiments/convergence_matrix_results.json")
