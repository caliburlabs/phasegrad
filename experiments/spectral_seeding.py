#!/usr/bin/env python3.12
"""Spectral seeding experiment: fix the ~50% convergence problem.

The convergence failure is a basin problem (Cochran's Q = 0 across 5 training
configs). The same seeds always fail. This experiment tests whether intelligent
ω initialization can push convergence from ~50% to 90%+.

Key insight: in the 2+5+2 architecture, both output nodes connect to ALL
hidden nodes. They are topologically symmetric. The ONLY way to break this
symmetry is through natural frequencies. Random ω init gives ~50% chance
of landing in a useful basin. Spectral seeding aims to guarantee it.

Strategies:
  1. random       — baseline (current make_network behavior)
  2. output_split — outputs get ±δ, hidden gets 0. Simplest symmetry break.
  3. greens       — ω maximizes initial output separation via Green's function
                    of the graph Laplacian: ω ∝ L̃⁻¹ e where e is the output
                    difference vector.
  4. best_eigen   — use the eigenvector of L̃ that best separates the two
                    output nodes. Not necessarily Fiedler — could be any mode.
  5. multi_start  — try 10 random ω inits, screen by initial output
                    separation across training samples, pick the best one.
"""
import json
import sys
import time

import numpy as np

from phasegrad.kuramoto import make_network, KuramotoNetwork
from phasegrad.data import load_hillenbrand
from phasegrad.training import train
from phasegrad.seeding import spectral_seed, graph_laplacian_reduced


N_SEEDS = 100
EPOCHS = 200
LR = 0.001
BETA = 0.1


# ---------------------------------------------------------------------------
#  Seeding strategies
# ---------------------------------------------------------------------------

def seed_random(net: KuramotoNetwork, rng: np.random.Generator) -> None:
    """Baseline: keep the random ω from make_network (do nothing)."""
    pass


def seed_output_split(net: KuramotoNetwork, rng: np.random.Generator,
                      delta: float = 0.3) -> None:
    """Set output ω to ±δ, hidden ω to small random values."""
    for i in net.learnable_ids:
        if i in net.output_ids:
            idx_in_out = net.output_ids.index(i)
            net.omega[i] = -delta if idx_in_out == 0 else +delta
        else:
            net.omega[i] = rng.uniform(-0.05, 0.05)


def seed_greens(net: KuramotoNetwork, rng: np.random.Generator,
                scale: float = 0.3) -> None:
    """Set ω to maximize initial output separation via Green's function.

    At θ=0, the equilibrium response is θ* ≈ -L̃⁻¹ ω. We want to
    maximize |e^T L̃⁻¹ ω| where e is the output difference vector.
    By Cauchy-Schwarz, optimal ω ∝ L̃⁻¹ e (since L̃ is symmetric).
    """
    N = net.N
    L_red = graph_laplacian_reduced(net)

    # Output difference vector in reduced system
    e = np.zeros(N - 1)
    out_red = [o - 1 for o in net.output_ids]
    e[out_red[0]] = -1.0
    e[out_red[1]] = +1.0

    opt_omega = np.linalg.solve(L_red, e)

    # Only set learnable frequencies; zero out inputs
    new_omega = np.zeros(N)
    new_omega[1:] = opt_omega
    for inp in net.input_ids:
        new_omega[inp] = 0.0

    # Normalize to desired scale
    max_abs = np.max(np.abs(new_omega))
    if max_abs > 1e-10:
        new_omega = new_omega / max_abs * scale

    net.omega = new_omega


def seed_best_eigen(net: KuramotoNetwork, rng: np.random.Generator,
                    scale: float = 0.3) -> None:
    """Use the eigenvector of L̃ that best separates the two output nodes."""
    N = net.N
    L_red = graph_laplacian_reduced(net)
    evals, evecs = np.linalg.eigh(L_red)

    out_red = [o - 1 for o in net.output_ids]

    # Find eigenvector with maximum output separation
    best_idx = 0
    best_sep = 0.0
    for i in range(len(evals)):
        sep = abs(evecs[out_red[0], i] - evecs[out_red[1], i])
        if sep > best_sep:
            best_sep = sep
            best_idx = i

    chosen = evecs[:, best_idx].copy()

    # Ensure output nodes have opposite signs
    if (chosen[out_red[0]] > 0) == (chosen[out_red[1]] > 0):
        # Same sign — flip the one with smaller magnitude
        if abs(chosen[out_red[0]]) < abs(chosen[out_red[1]]):
            chosen[out_red[0]] *= -1
        else:
            chosen[out_red[1]] *= -1

    new_omega = np.zeros(N)
    new_omega[1:] = chosen
    for inp in net.input_ids:
        new_omega[inp] = 0.0

    max_abs = np.max(np.abs(new_omega))
    if max_abs > 1e-10:
        new_omega = new_omega / max_abs * scale

    net.omega = new_omega


def seed_multi_start(net: KuramotoNetwork, rng: np.random.Generator,
                     n_starts: int = 10, n_probe: int = 10) -> None:
    """Try n_starts random ω inits, screen by initial output separation,
    pick the best one.

    For each candidate, solve equilibrium on n_probe training samples
    and measure average |θ_out1 - θ_out2|. Select the candidate with
    the largest average separation.
    """
    from phasegrad.data import load_hillenbrand

    # We need a few training samples to probe — use a fixed small set
    tr, _, _ = load_hillenbrand(vowels=['a', 'i'], seed=42)
    probe_samples = tr[:n_probe]

    best_omega = net.omega.copy()
    best_sep = -1.0
    original_omega = net.omega.copy()

    for trial in range(n_starts):
        # Generate candidate ω
        candidate = original_omega.copy()
        for i in net.learnable_ids:
            candidate[i] = rng.uniform(-0.3, 0.3)
        net.omega = candidate

        # Measure output separation across probe samples
        seps = []
        for x, cls in probe_samples:
            net.set_input(x)
            theta, res = net.equilibrium()
            if res < 0.01:
                out_phases = [theta[o] for o in net.output_ids]
                seps.append(abs(out_phases[0] - out_phases[1]))

        mean_sep = np.mean(seps) if seps else 0.0
        if mean_sep > best_sep:
            best_sep = mean_sep
            best_omega = candidate.copy()

    net.omega = best_omega


def seed_multi_eigen(net: KuramotoNetwork, rng: np.random.Generator,
                     scale: float = 0.3) -> None:
    """Linear combination of eigenvectors, weighted by output separation / λ.

    Delegates to phasegrad.seeding.spectral_seed (the library implementation).
    """
    spectral_seed(net, scale=scale)


def seed_best_eigen_noisy(net: KuramotoNetwork, rng: np.random.Generator,
                          scale: float = 0.3, noise: float = 0.05) -> None:
    """Best eigenvector + small perturbation to break exact symmetry."""
    seed_best_eigen(net, rng, scale=scale)
    for i in net.learnable_ids:
        net.omega[i] += rng.uniform(-noise, noise)


STRATEGIES = {
    'random': seed_random,
    'best_eigen': seed_best_eigen,
    'best_eigen_noisy': seed_best_eigen_noisy,
    'multi_eigen': seed_multi_eigen,
    'output_split': seed_output_split,
    'multi_start_10': seed_multi_start,
}


# ---------------------------------------------------------------------------
#  Experiment runner
# ---------------------------------------------------------------------------

def run_one(seed: int, strategy_name: str, strategy_fn) -> dict:
    """Train one seed with one strategy, return results."""
    # Build network (K topology determined by seed)
    net = make_network(seed=seed, K_scale=2.0, input_scale=1.5)

    # Load data (split determined by seed)
    tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)

    # Apply seeding strategy
    rng = np.random.default_rng(seed + 10000)  # separate rng for seeding
    strategy_fn(net, rng)

    # Record initial state
    init_omega = net.omega.copy()

    # Measure initial output separation
    init_seps = []
    for x, cls in tr[:20]:
        net.set_input(x)
        theta, res = net.equilibrium()
        if res < 0.01:
            out = [theta[o] for o in net.output_ids]
            init_seps.append(abs(out[0] - out[1]))
    init_sep = float(np.mean(init_seps)) if init_seps else 0.0

    # Train ω-only
    hist = train(net, tr, te,
                 lr_omega=LR, lr_K=0.0,  # ω-only
                 beta=BETA, epochs=EPOCHS, verbose=False,
                 eval_every=EPOCHS, seed=seed)

    final_acc = hist[-1]['acc']
    test_success = final_acc > 0.60

    return {
        'seed': seed,
        'strategy': strategy_name,
        'final_acc': final_acc,
        'test_success': test_success,
        'init_sep': init_sep,
    }


if __name__ == '__main__':
    print(f"Spectral Seeding Experiment: {N_SEEDS} seeds, {EPOCHS} epochs, ω-only")
    print(f"Strategies: {list(STRATEGIES.keys())}")
    print()

    all_results = {name: [] for name in STRATEGIES}
    t0 = time.time()

    for seed in range(N_SEEDS):
        row = {}
        for name, fn in STRATEGIES.items():
            r = run_one(seed, name, fn)
            all_results[name].append(r)
            row[name] = r['final_acc']

        elapsed = time.time() - t0
        conv_str = " ".join(
            f"{n[:7]}={'C' if row[n] > 0.6 else 'F'}"
            for n in STRATEGIES
        )
        print(f"  seed {seed:3d}: {conv_str}  ({elapsed:.0f}s)", flush=True)

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Strategy':20s} {'Success':>10s} {'Succ acc':>10s} "
          f"{'Overall acc':>12s} {'Init sep':>10s}")
    print(f"{'-'*80}")

    for name in STRATEGIES:
        results = all_results[name]
        accs = [r['final_acc'] for r in results]
        conv = [r for r in results if r['test_success']]
        conv_accs = [r['final_acc'] for r in conv]
        init_seps = [r['init_sep'] for r in results]

        conv_rate = f"{len(conv)}/{N_SEEDS}"
        conv_acc = f"{np.mean(conv_accs):.1%}" if conv_accs else "N/A"
        overall = f"{np.mean(accs):.1%}"
        isep = f"{np.mean(init_seps):.4f}"

        print(f"{name:20s} {conv_rate:>10s} {conv_acc:>10s} "
              f"{overall:>12s} {isep:>10s}")

    # Cross-strategy analysis: which seeds changed?
    print(f"\n{'='*80}")
    print("Seeds that changed test-success status (vs random baseline):")
    for name in STRATEGIES:
        if name == 'random':
            continue
        rescued = []
        lost = []
        for i in range(N_SEEDS):
            rand_conv = all_results['random'][i]['test_success']
            this_conv = all_results[name][i]['test_success']
            if not rand_conv and this_conv:
                rescued.append(i)
            elif rand_conv and not this_conv:
                lost.append(i)
        print(f"  {name:20s}: rescued {len(rescued):2d}, lost {len(lost):2d}  "
              f"(net: {len(rescued) - len(lost):+d})")

    # Save
    out_path = 'experiments/spectral_seeding_100seed_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
