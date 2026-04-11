#!/usr/bin/env python3.12
"""Softmax equivalence: does the Kuramoto equilibrium compute softmax?

Hypothesis: All-to-all coupled Kuramoto equilibrium phases are a
monotonic, normalized function of input frequencies — structurally
equivalent to softmax with coupling K playing the role of 1/temperature.

Parts:
  1. Mathematical verification: sweep K, compare θ* to softmax(ω/T)
  2. Energy comparison (from existing SPICE data)
  3. Gradient verification: backpropagate through the normalization
  4. Hybrid layer: digital matmul → oscillator softmax → gradient
"""
import json, time
import numpy as np
from scipy.optimize import minimize_scalar

from phasegrad.kuramoto import KuramotoNetwork, kuramoto_jacobian
from phasegrad.gradient import verify_gradients, two_phase_gradient
from phasegrad.losses import mse_loss


# ── Utilities ────────────────────────────────────────────────────

def softmax(z, T=1.0):
    """Stable softmax: exp(z/T) / Σ exp(z/T)."""
    z_scaled = z / T
    z_shifted = z_scaled - np.max(z_scaled)
    e = np.exp(z_shifted)
    return e / e.sum()


def make_alltoall(N, K_coupling, omega, seed=42):
    """Create all-to-all Kuramoto network with uniform coupling."""
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                K[i, j] = K_coupling
    return KuramotoNetwork(
        omega=omega.copy(), K=K,
        input_ids=[], output_ids=list(range(N)),
        input_scale=1.0,
    )


def rank_correlation(a, b):
    """Spearman rank correlation."""
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return r


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize_01(x):
    """Normalize to [0, 1]."""
    r = x.max() - x.min()
    if r < 1e-15:
        return np.ones_like(x) / len(x)
    return (x - x.min()) / r


# ── Part 1: Mathematical verification ────────────────────────────

def fit_temperature(theta_star, omega):
    """Find T* that minimizes ||normalize(θ*) - softmax(ω/T)||."""
    theta_norm = normalize_01(theta_star)

    def objective(log_T):
        T = np.exp(log_T)
        sm = softmax(omega, T)
        sm_norm = normalize_01(sm)
        return float(np.mean((theta_norm - sm_norm)**2))

    result = minimize_scalar(objective, bounds=(-5, 10), method='bounded')
    T_star = np.exp(result.x)
    mse = result.fun
    return T_star, mse


def compare_mappings(theta_star, omega, K):
    """Compare θ* to softmax(ω/T) under various mappings."""
    N = len(omega)
    omega_c = omega - omega.mean()
    theta_c = theta_star - theta_star.mean()

    # 1. Direct linear: θ ∝ ω?
    linear_cos = cosine_sim(theta_c, omega_c)

    # 2. Best-fit temperature for normalize(θ) ≈ softmax(ω/T)
    T_star, fit_mse = fit_temperature(theta_star, omega)
    sm_best = softmax(omega, T_star)
    sm_norm = normalize_01(sm_best)
    theta_norm = normalize_01(theta_star)
    fit_cos = cosine_sim(theta_norm, sm_norm)

    # 3. Rank correlation (monotonicity)
    rank_corr = rank_correlation(theta_star, omega)

    # 4. Phase-based softmax: exp(θ_i) / Σ exp(θ_j) vs softmax(ω/T)
    phase_sm = softmax(theta_star, T=1.0)
    # Find T such that phase_sm ≈ softmax(ω/T)
    def obj_phase(log_T):
        T = np.exp(log_T)
        return float(np.mean((phase_sm - softmax(omega, T))**2))
    res2 = minimize_scalar(obj_phase, bounds=(-5, 10), method='bounded')
    T_phase = np.exp(res2.x)
    phase_fit_mse = res2.fun

    # 5. Phase spread ratio: range(θ) / range(ω)
    theta_range = theta_star.max() - theta_star.min()
    omega_range = omega.max() - omega.min()
    compression = theta_range / omega_range if omega_range > 0 else 0

    return {
        'linear_cos': linear_cos,
        'T_star': T_star,
        'fit_mse': fit_mse,
        'fit_cos': fit_cos,
        'rank_correlation': rank_corr,
        'T_phase': T_phase,
        'phase_fit_mse': phase_fit_mse,
        'compression_ratio': compression,
        'theta_range': theta_range,
    }


def part1_sweep():
    """Sweep K for multiple input vectors, compare to softmax."""
    print("=" * 65)
    print("Part 1: Softmax Equivalence — K Sweep")
    print("=" * 65)

    N = 8
    input_vectors = {
        'uniform': np.linspace(1, 8, N),
        'random1': np.array([3.2, 1.1, 5.7, 2.3, 7.8, 4.5, 6.1, 0.9]),
        'peaked':  np.array([1.0, 1.0, 1.0, 8.0, 1.0, 1.0, 1.0, 1.0]),
        'bimodal': np.array([1.0, 1.0, 1.0, 1.0, 7.0, 7.0, 7.0, 7.0]),
        'equal':   np.array([4.0, 4.1, 4.0, 4.1, 4.0, 4.1, 4.0, 4.1]),
    }

    K_values = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0, 100.0]

    all_results = {}

    for vec_name, omega in input_vectors.items():
        print(f"\n  Input: {vec_name} = {omega.round(1)}")
        vec_results = []

        for K in K_values:
            net = make_alltoall(N, K, omega)
            theta, res = net.equilibrium()

            if res > 0.01:
                print(f"    K={K:6.1f}: FAILED (res={res:.2e})")
                continue

            m = compare_mappings(theta, omega, K)
            vec_results.append({
                'K': K, 'residual': res,
                'theta': theta.tolist(),
                **m,
            })

            print(f"    K={K:6.1f}: rank={m['rank_correlation']:+.4f} "
                  f"linear_cos={m['linear_cos']:.4f} "
                  f"T*={m['T_star']:.3f} fit_cos={m['fit_cos']:.4f} "
                  f"compress={m['compression_ratio']:.4f}")

        all_results[vec_name] = vec_results

    # Key finding: is it linear or softmax?
    print(f"\n{'─'*65}")
    print("KEY FINDING: Linear or softmax?")
    print("  For all-to-all coupling, θ_i = ω_i^c / (K·N) (linear).")
    print("  Compression ratio = 1/(K·N):")
    for r in all_results.get('uniform', []):
        predicted = 1.0 / (r['K'] * N)
        actual = r['compression_ratio']
        print(f"    K={r['K']:6.1f}: predicted={predicted:.4f}, "
              f"actual={actual:.4f}, ratio={actual/predicted:.4f}")

    print(f"\n  Verdict: NOT softmax. The equilibrium is a LINEAR compression.")
    print(f"  θ* ∝ ω/K — monotonic, bounded, differentiable, but linear.")
    print(f"  The nonlinearity (sin) only matters near K_critical ≈ max(|ω^c|)/N")
    print(f"  where equilibrium is unreliable anyway.")

    return all_results


# ── Part 3: Gradient verification ────────────────────────────────

def part3_gradient():
    """Verify gradients through the all-to-all softmax-like layer."""
    print(f"\n{'='*65}")
    print("Part 3: Gradient Verification Through Normalization Layer")
    print(f"{'='*65}")

    N = 8
    K = 5.0
    omega = np.linspace(1, 8, N)
    net = make_alltoall(N, K, omega)

    theta, res = net.equilibrium()
    print(f"  Equilibrium residual: {res:.2e}")

    # Target: push oscillator 3 to have larger phase, others smaller
    target = np.zeros(N)
    for i in range(N):
        target[i] = -0.1 if i == 3 else 0.1

    results = verify_gradients(net, theta, target, beta=1e-4, eps=1e-5)
    print(f"\n  Gradient cosine similarities:")
    print(f"    Two-phase ↔ FD:        {results['cos_tp_fd']:.8f}")
    print(f"    Analytical ↔ FD:       {results['cos_an_fd']:.8f}")
    print(f"    Two-phase ↔ Analytical: {results['cos_tp_an']:.8f}")

    # Multiple K values
    print(f"\n  Gradient verification across K values:")
    grad_results = []
    for K in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        net = make_alltoall(N, K, omega)
        theta, res = net.equilibrium()
        if res > 0.01:
            print(f"    K={K:5.1f}: FAILED")
            continue
        r = verify_gradients(net, theta, target, beta=1e-4, eps=1e-5)
        print(f"    K={K:5.1f}: tp↔fd={r['cos_tp_fd']:.8f} "
              f"an↔fd={r['cos_an_fd']:.8f}")
        grad_results.append({
            'K': K, 'cos_tp_fd': r['cos_tp_fd'],
            'cos_an_fd': r['cos_an_fd'], 'cos_tp_an': r['cos_tp_an'],
        })

    return grad_results


# ── Part 4: Hybrid layer ────────────────────────────────────────

def part4_hybrid():
    """End-to-end hybrid: digital matmul → oscillator normalization → gradient."""
    print(f"\n{'='*65}")
    print("Part 4: Hybrid Digital-Analog Layer")
    print(f"{'='*65}")

    N = 8
    K = 5.0
    rng = np.random.default_rng(42)

    # Random weight matrix and input
    W = rng.standard_normal((N, N)) * 0.3
    x = rng.standard_normal(N) * 0.5

    # Forward pass: z = Wx (digital)
    z = W @ x
    print(f"  x = {x.round(3)}")
    print(f"  z = Wx = {z.round(3)}")

    # Feed z as frequencies into Kuramoto (analog)
    # Shift to positive range for valid frequencies
    omega = z - z.min() + 1.0  # ensure all positive
    net = make_alltoall(N, K, omega)
    theta, res = net.equilibrium()
    print(f"  θ* = {theta.round(4)}")
    print(f"  Equilibrium residual: {res:.2e}")

    # Compare to softmax(z/T)
    T_star, mse = fit_temperature(theta, z)
    sm = softmax(z, T_star)
    print(f"  Best-fit T={T_star:.3f}, MSE={mse:.6f}")
    print(f"  softmax(z/{T_star:.1f}) = {sm.round(4)}")

    # Loss: MSE of phases to target
    target_phases = np.zeros(N)
    correct_class = int(np.argmax(z))  # class with highest logit
    for i in range(N):
        target_phases[i] = -0.2 if i == correct_class else 0.2

    loss = mse_loss(theta, target_phases, list(range(N)))
    print(f"\n  Target class: {correct_class} (highest logit)")
    print(f"  Loss: {loss:.4f}")

    # Gradient via two-phase EP
    grad_omega, _, clamp_res = two_phase_gradient(
        net, theta, target_phases, beta=1e-4)
    print(f"  Clamped residual: {clamp_res:.2e}")

    # ∂L/∂z = gradient w.r.t. frequencies (= ω-gradient)
    # Since ω = z - min(z) + 1, ∂ω/∂z = I (identity), so ∂L/∂z = ∂L/∂ω
    dL_dz = grad_omega
    print(f"  ∂L/∂z (ω-gradient): {dL_dz.round(4)}")

    # ∂L/∂W = (∂L/∂z) × x^T
    dL_dW = np.outer(dL_dz, x)
    print(f"  ∂L/∂W shape: {dL_dW.shape}")
    print(f"  |∂L/∂W| = {np.linalg.norm(dL_dW):.4f}")

    # Verify gradient numerically using centered frequencies
    eps = 1e-5
    omega_c = net.omega_centered.copy()
    fd_grad_z = np.zeros(N)
    for k in range(1, N):  # skip pinned node
        oc_p = omega_c.copy(); oc_p[k] += eps
        theta_p, _ = net.equilibrium(theta_init=theta.copy(), omega_c=oc_p)
        L_p = mse_loss(theta_p, target_phases, list(range(N)))

        oc_m = omega_c.copy(); oc_m[k] -= eps
        theta_m, _ = net.equilibrium(theta_init=theta.copy(), omega_c=oc_m)
        L_m = mse_loss(theta_m, target_phases, list(range(N)))

        fd_grad_z[k] = (L_p - L_m) / (2 * eps)

    ep_fd_cos = cosine_sim(grad_omega[1:], fd_grad_z[1:])
    print(f"\n  EP gradient ↔ FD gradient cosine: {ep_fd_cos:.8f}")
    print(f"  Gradient is {'EXACT' if ep_fd_cos > 0.9999 else 'APPROXIMATE'}")

    return {
        'W': W.tolist(), 'x': x.tolist(), 'z': z.tolist(),
        'omega': omega.tolist(), 'theta': theta.tolist(),
        'T_star': T_star, 'fit_mse': mse,
        'loss': loss, 'correct_class': correct_class,
        'grad_omega': grad_omega.tolist(),
        'dL_dW_norm': float(np.linalg.norm(dL_dW)),
        'ep_fd_cosine': ep_fd_cos,
    }


# ── Main ─────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    results = {}

    # Part 1: Mathematical verification
    results['part1_sweep'] = part1_sweep()

    # Part 3: Gradient verification
    results['part3_gradient'] = part3_gradient()

    # Part 4: Hybrid layer
    results['part4_hybrid'] = part4_hybrid()

    # Part 2: Energy comparison (from existing measurements)
    print(f"\n{'='*65}")
    print("Part 2: Energy Comparison")
    print(f"{'='*65}")

    energy = {
        'oscillator': {
            'per_osc_uW': 180,
            'settling_ns': 30,
            'N8_pJ': 8 * 180e-6 * 30e-9 * 1e12,   # 43.2 pJ
            'N64_pJ': 64 * 180e-6 * 30e-9 * 1e12,  # 345.6 pJ
            'N512_pJ': 512 * 180e-6 * 30e-9 * 1e12, # 2764.8 pJ
            'note': 'from SPICE measured sensor bank, all-to-all O(1) settling',
        },
        'digital_softmax': {
            'N8_pJ': 65,    # ROM exp + sum + div, 8-bit, 45nm
            'N64_pJ': 520,
            'N512_pJ': 4200,
            'note': 'estimate: N×(5 exp + 0.1 add + 3 div) pJ, 8-bit, 45nm',
        },
    }

    for N in [8, 64, 512]:
        osc = energy['oscillator'][f'N{N}_pJ']
        dig = energy['digital_softmax'][f'N{N}_pJ']
        ratio = dig / osc if osc > 0 else float('inf')
        winner = 'oscillator' if osc < dig else 'digital'
        print(f"  N={N:4d}: osc={osc:.1f} pJ, digital={dig:.0f} pJ, "
              f"ratio={ratio:.1f}x → {winner} wins")

    results['part2_energy'] = energy

    # Summary
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")

    # Check if the equivalence holds
    uniform_results = results['part1_sweep'].get('uniform', [])
    if uniform_results:
        avg_rank = np.mean([r['rank_correlation'] for r in uniform_results
                            if r.get('rank_correlation') is not None])
        avg_linear = np.mean([r['linear_cos'] for r in uniform_results])
        avg_fit = np.mean([r['fit_cos'] for r in uniform_results])

        print(f"\n  Softmax equivalence (average over K sweep, 'uniform' input):")
        print(f"    Rank correlation (monotonicity): {avg_rank:.4f}")
        print(f"    Linear cosine (θ ∝ ω):           {avg_linear:.4f}")
        print(f"    Softmax fit cosine:               {avg_fit:.4f}")

    hybrid = results['part4_hybrid']
    print(f"\n  Hybrid layer (digital matmul → oscillator norm):")
    print(f"    EP ↔ FD gradient cosine: {hybrid['ep_fd_cosine']:.8f}")
    print(f"    Gradient is {'EXACT' if hybrid['ep_fd_cosine'] > 0.9999 else 'APPROXIMATE'}")

    results['wall_time_s'] = time.time() - t_start
    print(f"\nTotal wall time: {results['wall_time_s']:.0f}s")

    outfile = 'experiments/softmax_equivalence_results.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {outfile}")

    return results


if __name__ == '__main__':
    main()
