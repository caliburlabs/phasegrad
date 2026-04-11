"""Parameter-matched ablation: ω-only vs K-only at equal parameter counts.

The original ablation compared 7 ω parameters against 24 K parameters.
This experiment eliminates the parameter-count confound by:

  Experiment 1 — Fixed architecture (2+5+2), matched params:
    ω-only:     7 learnable frequencies, 0 learnable edges
    K-matched:  0 learnable frequencies, 7 randomly selected edges
    K-full:     0 learnable frequencies, all 24 edges (original baseline)

  Experiment 2 — Architecture sweep (varying n_hidden):
    At each size, run ω-only and K-only with all params,
    reporting accuracy vs parameter count.

100 seeds per condition. Same seed controls data split, network init,
and (for K-matched) edge selection.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.losses import mse_loss, mse_target


# ---------------------------------------------------------------------------
#  Training loop with edge mask support
# ---------------------------------------------------------------------------

def train_with_mask(net, train_data, test_data, *,
                    lr_omega=0.001, lr_K=0.001,
                    learnable_edges=None,
                    beta=0.1, epochs=200, margin=0.2,
                    grad_clip=2.0, seed=42, eval_every=20):
    """Train loop identical to training.py but with an edge mask.

    Args:
        learnable_edges: if not None, only these (i,j) pairs get K updates.
                         All other edges are frozen at their initial values.
    """
    rng = np.random.default_rng(seed)
    all_edges = net.edges
    active_edges = learnable_edges if learnable_edges is not None else all_edges

    theta_warm = None
    final_train_acc = 0.0
    final_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(train_data))
        correct_train = n_train = 0
        for idx in indices:
            x, cls = train_data[idx]
            net.set_input(x)
            target = mse_target(net.N, net.output_ids, cls, margin)

            theta_free, res_free = net.equilibrium(theta_warm)
            if res_free > 0.01:
                continue

            theta_clamp, res_clamp = net.clamped_equilibrium(
                beta, target, theta_init=theta_free.copy())
            if res_clamp > 0.01:
                continue

            pred = net.classify(theta_free)
            correct_train += (pred == cls)
            n_train += 1

            # ω update
            for i in net.learnable_ids:
                g = -(theta_clamp[i] - theta_free[i]) / beta
                g = np.clip(g, -grad_clip, grad_clip)
                net.omega[i] -= lr_omega * g

            # K update (only active edges)
            for (i, j) in active_edges:
                cos_free = np.cos(theta_free[j] - theta_free[i])
                cos_clamp = np.cos(theta_clamp[j] - theta_clamp[i])
                g = (cos_free - cos_clamp) / beta
                g = np.clip(g, -grad_clip, grad_clip)
                net.K[i, j] -= lr_K * g
                net.K[j, i] = net.K[i, j]

            # Physical bounds: clip existing edges only, don't create new ones
            for i in net.learnable_ids:
                net.omega[i] = np.clip(net.omega[i], -3.0, 3.0)
            existing = net.K > 0
            net.K = np.where(existing, np.clip(net.K, 0.01, 8.0), 0.0)

            theta_warm = theta_free

        final_train_acc = correct_train / max(n_train, 1)

        # Evaluate every eval_every epochs
        if epoch % eval_every == 0 or epoch == epochs:
            correct = n = 0
            theta_warm_eval = None
            for x, cls in test_data:
                net.set_input(x)
                theta, res = net.equilibrium(theta_warm_eval)
                if res > 0.1:
                    continue
                pred = net.classify(theta)
                correct += (pred == cls)
                n += 1
                theta_warm_eval = theta
            final_test_acc = correct / max(n, 1)

    return {'test_acc': final_test_acc, 'train_acc': final_train_acc}


# ---------------------------------------------------------------------------
#  Experiment 1: Parameter-matched comparison
# ---------------------------------------------------------------------------

def run_param_matched(n_seeds=100, epochs=200):
    """ω-only (7 params) vs K-matched (7 random edges) vs K-full (24 edges)."""
    print("=" * 70)
    print("EXPERIMENT 1: Parameter-matched ablation (7 vs 7 vs 24)")
    print("=" * 70)

    results = {"omega_only": [], "K_matched_7": [], "K_full_24": []}
    t0 = time.time()

    for seed in range(n_seeds):
        tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)

        # --- ω-only (7 params) ---
        net = make_network(n_input=2, n_hidden=5, n_output=2,
                           K_scale=2.0, input_scale=1.5, seed=seed)
        r_w = train_with_mask(net, tr, te, lr_omega=0.001, lr_K=0.0,
                              epochs=epochs, seed=seed)
        results["omega_only"].append(r_w)

        # --- K-matched: 7 random edges (7 params) ---
        net = make_network(n_input=2, n_hidden=5, n_output=2,
                           K_scale=2.0, input_scale=1.5, seed=seed)
        all_edges = net.edges
        edge_rng = np.random.default_rng(seed + 10000)
        selected = edge_rng.choice(len(all_edges), size=7, replace=False)
        learnable_edges = [all_edges[i] for i in selected]
        r_km = train_with_mask(net, tr, te, lr_omega=0.0, lr_K=0.001,
                               learnable_edges=learnable_edges,
                               epochs=epochs, seed=seed)
        results["K_matched_7"].append(r_km)

        # --- K-full: all 24 edges (24 params) ---
        net = make_network(n_input=2, n_hidden=5, n_output=2,
                           K_scale=2.0, input_scale=1.5, seed=seed)
        r_kf = train_with_mask(net, tr, te, lr_omega=0.0, lr_K=0.001,
                               epochs=epochs, seed=seed)
        results["K_full_24"].append(r_kf)

        elapsed = time.time() - t0
        if (seed + 1) % 10 == 0:
            w_test = [r['test_acc'] for r in results['omega_only']]
            km_test = [r['test_acc'] for r in results['K_matched_7']]
            kf_test = [r['test_acc'] for r in results['K_full_24']]
            print(f"  Seed {seed+1:3d}/{n_seeds} ({elapsed:.0f}s)  "
                  f"ω={np.mean(w_test):.1%}  "
                  f"K7={np.mean(km_test):.1%}  "
                  f"K24={np.mean(kf_test):.1%}")

    return results


# ---------------------------------------------------------------------------
#  Experiment 2: Architecture sweep
# ---------------------------------------------------------------------------

def run_architecture_sweep(n_seeds=50, epochs=200):
    """Sweep n_hidden, reporting accuracy and parameter counts."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Architecture sweep (varying n_hidden)")
    print("=" * 70)

    hidden_sizes = [3, 5, 7, 10, 15]
    results = []
    t0 = time.time()

    for n_hidden in hidden_sizes:
        n_omega_params = n_hidden + 2  # hidden + output frequencies
        # Edge count: input→hidden + hidden→output + hidden chain
        n_K_params = 2 * n_hidden + 5 * 2 + (n_hidden - 1)
        # More precisely:
        n_K_params = 2 * n_hidden + n_hidden * 2 + (n_hidden - 1)

        results_w = []
        results_k = []

        for seed in range(n_seeds):
            tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)

            # ω-only
            net = make_network(n_input=2, n_hidden=n_hidden, n_output=2,
                               K_scale=2.0, input_scale=1.5, seed=seed)
            r_w = train_with_mask(net, tr, te, lr_omega=0.001, lr_K=0.0,
                                  epochs=epochs, seed=seed)
            results_w.append(r_w)

            # K-only (all edges)
            net = make_network(n_input=2, n_hidden=n_hidden, n_output=2,
                               K_scale=2.0, input_scale=1.5, seed=seed)
            r_k = train_with_mask(net, tr, te, lr_omega=0.0, lr_K=0.001,
                                  epochs=epochs, seed=seed)
            results_k.append(r_k)

        # Actual edge count from network
        net_check = make_network(n_input=2, n_hidden=n_hidden, n_output=2,
                                 K_scale=2.0, input_scale=1.5, seed=0)
        n_K_params = len(net_check.edges)

        # Convergence from TRAIN accuracy
        w_test = [r['test_acc'] for r in results_w]
        k_test = [r['test_acc'] for r in results_k]
        w_conv = [r['test_acc'] for r in results_w if r['train_acc'] > 0.6]
        k_conv = [r['test_acc'] for r in results_k if r['train_acc'] > 0.6]

        elapsed = time.time() - t0
        row = {
            "n_hidden": n_hidden,
            "n_total": 2 + n_hidden + 2,
            "n_omega_params": n_omega_params,
            "n_K_params": n_K_params,
            "omega_results": results_w,
            "K_results": results_k,
            "omega_mean": float(np.mean(w_test)),
            "K_mean": float(np.mean(k_test)),
            "omega_converged_mean": float(np.mean(w_conv)) if w_conv else 0.0,
            "K_converged_mean": float(np.mean(k_conv)) if k_conv else 0.0,
            "omega_conv_rate": len(w_conv) / len(results_w),
            "K_conv_rate": len(k_conv) / len(results_k),
        }
        results.append(row)
        print(f"  n_hidden={n_hidden:2d}  N={row['n_total']:2d}  "
              f"ω_params={n_omega_params:2d}  K_params={n_K_params:2d}  "
              f"ω_acc={row['omega_mean']:.1%} ({row['omega_conv_rate']:.0%} conv)  "
              f"K_acc={row['K_mean']:.1%} ({row['K_conv_rate']:.0%} conv)  "
              f"({elapsed:.0f}s)")

    return results


# ---------------------------------------------------------------------------
#  Statistics
# ---------------------------------------------------------------------------

def analyze_param_matched(results):
    """Print statistics for the parameter-matched experiment.

    Convergence defined by final-epoch TRAIN accuracy > 0.6.
    Reports final-epoch TEST accuracy for converged seeds.
    """
    from scipy import stats

    print("\n" + "=" * 70)
    print("EXPERIMENT 1 RESULTS")
    print("Convergence defined by final-epoch TRAIN acc > 0.6")
    print("=" * 70)

    for key in ["omega_only", "K_matched_7", "K_full_24"]:
        test_accs = np.array([r['test_acc'] for r in results[key]])
        train_accs = np.array([r['train_acc'] for r in results[key]])
        conv_mask = train_accs > 0.6
        conv_test = test_accs[conv_mask]
        print(f"\n  {key}:")
        print(f"    All seeds (test):  {test_accs.mean():.1%} +/- {test_accs.std():.1%}  (n={len(test_accs)})")
        if len(conv_test) > 0:
            print(f"    Converged (test):  {conv_test.mean():.1%} +/- {conv_test.std():.1%}  "
                  f"(n={len(conv_test)}, rate={conv_mask.sum()}/{len(test_accs)})")

    # Head-to-head: ω-only vs K-matched (both 7 params)
    w_test = np.array([r['test_acc'] for r in results["omega_only"]])
    w_train = np.array([r['train_acc'] for r in results["omega_only"]])
    km_test = np.array([r['test_acc'] for r in results["K_matched_7"]])
    km_train = np.array([r['train_acc'] for r in results["K_matched_7"]])

    w_conv_test = w_test[w_train > 0.6]
    km_conv_test = km_test[km_train > 0.6]

    if len(w_conv_test) > 1 and len(km_conv_test) > 1:
        t_stat, p_val = stats.ttest_ind(w_conv_test, km_conv_test, equal_var=False)
        print(f"\n  ω-only vs K-matched (train-converged, Welch's t-test on TEST acc):")
        print(f"    ω: {w_conv_test.mean():.1%} (n={len(w_conv_test)})  vs  "
              f"K7: {km_conv_test.mean():.1%} (n={len(km_conv_test)})")
        print(f"    t={t_stat:.2f}, p={p_val:.2e}")

    # Convergence rate comparison (train-defined)
    w_conv_n = int((w_train > 0.6).sum())
    km_conv_n = int((km_train > 0.6).sum())
    table = np.array([[w_conv_n, len(w_test) - w_conv_n],
                      [km_conv_n, len(km_test) - km_conv_n]])
    _, fisher_p = stats.fisher_exact(table)
    print(f"\n  Convergence rate (train-defined, Fisher's exact test):")
    print(f"    ω: {w_conv_n}/{len(w_test)}  vs  K7: {km_conv_n}/{len(km_test)}")
    print(f"    p={fisher_p:.4f}")

    # Per-seed paired comparison on test acc (all seeds, no filtering)
    wins_w = int(np.sum(w_test > km_test))
    wins_k = int(np.sum(km_test > w_test))
    ties = int(np.sum(w_test == km_test))
    print(f"\n  Per-seed head-to-head on test acc (all seeds, no filtering):")
    print(f"    ω wins: {wins_w}  K7 wins: {wins_k}  ties: {ties}")

    # Wilcoxon signed-rank on test acc (paired, non-parametric, all seeds)
    diffs = w_test - km_test
    diffs_nonzero = diffs[diffs != 0]
    if len(diffs_nonzero) > 10:
        wsr_stat, wsr_p = stats.wilcoxon(diffs_nonzero)
        print(f"    Wilcoxon signed-rank: W={wsr_stat:.0f}, p={wsr_p:.2e}")

    # All-seed test comparison (no filtering at all)
    t_all, p_all = stats.ttest_ind(w_test, km_test, equal_var=False)
    print(f"\n  All-seed test accuracy (no filtering):")
    print(f"    ω: {w_test.mean():.1%}  vs  K7: {km_test.mean():.1%}")
    print(f"    Welch's t={t_all:.2f}, p={p_all:.2e}")


def analyze_sweep(results):
    """Print architecture sweep summary."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 RESULTS: Accuracy vs Parameter Count")
    print("=" * 70)
    print(f"\n  {'n_hid':>5} {'N':>3} {'ω_p':>4} {'K_p':>4} "
          f"{'ω_acc':>7} {'K_acc':>7} {'ω_conv':>7} {'K_conv':>7} "
          f"{'ω_c_acc':>8} {'K_c_acc':>8}")
    print("  " + "-" * 75)

    for r in results:
        print(f"  {r['n_hidden']:5d} {r['n_total']:3d} "
              f"{r['n_omega_params']:4d} {r['n_K_params']:4d} "
              f"{r['omega_mean']:7.1%} {r['K_mean']:7.1%} "
              f"{r['omega_conv_rate']:7.0%} {r['K_conv_rate']:7.0%} "
              f"{r['omega_converged_mean']:8.1%} {r['K_converged_mean']:8.1%}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds1", type=int, default=100,
                        help="Seeds for Experiment 1 (param-matched)")
    parser.add_argument("--seeds2", type=int, default=50,
                        help="Seeds for Experiment 2 (arch sweep)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--exp", choices=["1", "2", "both"], default="both")
    args = parser.parse_args()

    if args.exp in ("1", "both"):
        r1 = run_param_matched(n_seeds=args.seeds1, epochs=args.epochs)
        analyze_param_matched(r1)
        pm_results = {
            "param_matched": {k: vals for k, vals in r1.items()},
            "convergence_criterion": "final_epoch_train_acc > 0.6",
        }
        outfile1 = Path(__file__).with_name("param_matched_results.json")
        with open(outfile1, "w") as f:
            json.dump(pm_results, f, indent=2)
        print(f"\nParam-matched results saved to {outfile1}")

    if args.exp in ("2", "both"):
        r2 = run_architecture_sweep(n_seeds=args.seeds2, epochs=args.epochs)
        analyze_sweep(r2)
        sweep_results = {
            "architecture_sweep": [
                {k: v for k, v in r.items()
                 if k not in ("omega_results", "K_results")}
                for r in r2
            ],
            "convergence_criterion": "final_epoch_train_acc > 0.6",
        }
        outfile2 = Path(__file__).with_name("architecture_sweep_results.json")
        with open(outfile2, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nArchitecture sweep results saved to {outfile2}")
