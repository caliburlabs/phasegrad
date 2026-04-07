#!/usr/bin/env python3.12
"""C2: Training stabilization experiments.

Push convergence rate above 55% with targeted modifications.
Each variant vs baseline (vanilla training, 20 seeds).

C2a: Strong-start — initialize K well above K_critical, let learning refine
C2b: Coupling floor — clamp K >= 1.5 during training
C2c: Frequency spread control — soft penalty on omega variance
C2d: Basin filter — reject gradient steps where phase displacement is suspiciously large
"""
import json, time, numpy as np
from phasegrad.kuramoto import make_network, KuramotoNetwork
from phasegrad.data import load_hillenbrand
from phasegrad.training import train, _train_epoch, _evaluate, _apply_update
from phasegrad.losses import mse_loss, mse_target

N_SEEDS = 20
EPOCHS = 200

def run_baseline(seed):
    tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
    net = make_network(seed=seed, K_scale=2.0, input_scale=1.5)
    hist = train(net, tr, te, lr_omega=0.001, lr_K=0.001, beta=0.1,
                 epochs=EPOCHS, verbose=False, eval_every=20, seed=seed)
    return hist[-1]['acc']  # final-epoch, not best-during-training


def run_strong_start(seed):
    """C2a: K_scale=4.0 instead of 2.0."""
    tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
    net = make_network(seed=seed, K_scale=4.0, input_scale=1.5)
    hist = train(net, tr, te, lr_omega=0.001, lr_K=0.001, beta=0.1,
                 epochs=EPOCHS, verbose=False, eval_every=20, seed=seed)
    return hist[-1]['acc']  # final-epoch, not best-during-training


def run_coupling_floor(seed):
    """C2b: After each update, clamp K >= 1.5."""
    tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
    net = make_network(seed=seed, K_scale=2.0, input_scale=1.5)
    rng = np.random.default_rng(seed)

    history = []
    ev0 = _evaluate(net, te, margin=0.2)
    history.append({'epoch': 0, **ev0})

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc, n_skip = _train_epoch(
            net, tr, beta=0.1, lr_omega=0.001, lr_K=0.001,
            margin=0.2, grad_clip=2.0, rng=rng)
        # Coupling floor: enforce K >= 1.5 where K > 0
        mask = net.K > 0
        net.K[mask] = np.maximum(net.K[mask], 1.5)

        if ep % 20 == 0 or ep == EPOCHS:
            ev = _evaluate(net, te, margin=0.2)
            history.append({'epoch': ep, **ev})

    return history[-1]['acc']  # final-epoch, not best-during-training


def run_freq_control(seed):
    """C2c: Periodically re-center learnable omega to reduce spread."""
    tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
    net = make_network(seed=seed, K_scale=2.0, input_scale=1.5)
    rng = np.random.default_rng(seed)

    history = []
    ev0 = _evaluate(net, te, margin=0.2)
    history.append({'epoch': 0, **ev0})

    learnable = net.learnable_ids
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc, n_skip = _train_epoch(
            net, tr, beta=0.1, lr_omega=0.001, lr_K=0.001,
            margin=0.2, grad_clip=2.0, rng=rng)
        # Re-center learnable frequencies every 10 epochs
        if ep % 10 == 0:
            mean_learnable = np.mean([net.omega[i] for i in learnable])
            for i in learnable:
                net.omega[i] -= 0.5 * mean_learnable  # soft re-centering

        if ep % 20 == 0 or ep == EPOCHS:
            ev = _evaluate(net, te, margin=0.2)
            history.append({'epoch': ep, **ev})

    return history[-1]['acc']  # final-epoch, not best-during-training


def run_omega_only_strong(seed):
    """Omega-only + strong coupling (our best config based on ablation)."""
    tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
    net = make_network(seed=seed, K_scale=3.0, input_scale=1.5)
    hist = train(net, tr, te, lr_omega=0.001, lr_K=0.0, beta=0.1,
                 epochs=EPOCHS, verbose=False, eval_every=20, seed=seed)
    return hist[-1]['acc']  # final-epoch, not best-during-training


VARIANTS = {
    'baseline': run_baseline,
    'strong_start_K4': run_strong_start,
    'coupling_floor_1.5': run_coupling_floor,
    'freq_recentering': run_freq_control,
    'omega_only_K3': run_omega_only_strong,
}


if __name__ == '__main__':
    print(f"Training Stabilization: {N_SEEDS} seeds, {EPOCHS} epochs")
    results = {v: [] for v in VARIANTS}
    t0 = time.time()

    for seed in range(N_SEEDS):
        row = {}
        for name, fn in VARIANTS.items():
            acc = fn(seed)
            results[name].append(acc)
            row[name] = acc
        elapsed = time.time() - t0
        print(f"  seed {seed:2d}: " +
              " ".join(f"{n[:8]}={row[n]:.0%}" for n in VARIANTS) +
              f" ({elapsed:.0f}s)", flush=True)

    print(f"\n{'='*75}")
    print(f"{'variant':25s} {'mean':>7s} {'std':>7s} {'conv':>6s} {'conv_mean':>10s}")
    for name, accs in results.items():
        conv = [a for a in accs if a > 0.60]
        cm = f"{np.mean(conv):.1%}" if conv else "N/A"
        print(f"{name:25s} {np.mean(accs):7.1%} {np.std(accs):7.1%} "
              f"{len(conv):3d}/{N_SEEDS:2d} {cm:>10s}")

    out = 'experiments/stabilization_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")
