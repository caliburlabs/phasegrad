#!/usr/bin/env python3.12
"""D5: Clean gradient validation — omega-only two-phase vs omega-only FD.
Same parameter set, same update rule, different gradient method.
Reports exact numbers for the paper."""
import json, time, numpy as np
from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.training import train, _evaluate
from phasegrad.gradient import finite_difference_gradient
from phasegrad.losses import mse_loss, mse_target

EPOCHS = 200
N_SEEDS = 10

def train_fd_omega_only(net, train_data, test_data, lr=0.001, epochs=200,
                         margin=0.2, seed=42):
    """Train with FD gradients, omega only."""
    rng = np.random.default_rng(seed)
    history = []
    ev0 = _evaluate(net, test_data, margin)
    history.append({'epoch': 0, **ev0})

    for ep in range(1, epochs + 1):
        indices = rng.permutation(len(train_data))
        total_loss = correct = n = 0
        theta_warm = None

        for idx in indices:
            x, cls = train_data[idx]
            net.set_input(x)
            target = mse_target(net.N, net.output_ids, cls, margin)
            theta, res = net.equilibrium(theta_warm)
            if res > 0.01:
                continue

            loss = mse_loss(theta, target, net.output_ids)
            pred = net.classify(theta)
            total_loss += loss
            correct += (pred == cls)
            n += 1

            grad_fd = finite_difference_gradient(net, theta, target, eps=1e-5)
            for i in net.learnable_ids:
                net.omega[i] -= lr * np.clip(grad_fd[i], -2.0, 2.0)
            for i in net.learnable_ids:
                net.omega[i] = np.clip(net.omega[i], -3.0, 3.0)
            theta_warm = theta

        if ep % 20 == 0 or ep == epochs:
            ev = _evaluate(net, test_data, margin)
            history.append({'epoch': ep, **ev,
                           'train_loss': total_loss / max(n, 1),
                           'train_acc': correct / max(n, 1)})
    return history

if __name__ == '__main__':
    print(f"D5: Clean gradient validation (omega-only, {N_SEEDS} seeds, {EPOCHS} ep)")
    tp_accs, fd_accs = [], []
    t0 = time.time()

    for seed in range(N_SEEDS):
        tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)

        # Two-phase (omega only)
        net_tp = make_network(seed=seed, K_scale=2.0, input_scale=1.5)
        hist_tp = train(net_tp, tr, te, lr_omega=0.001, lr_K=0.0, beta=0.1,
                        epochs=EPOCHS, verbose=False, eval_every=20, seed=seed)
        best_tp = hist_tp[-1]['acc']  # final-epoch, not best-during-training

        # FD (omega only)
        net_fd = make_network(seed=seed, K_scale=2.0, input_scale=1.5)
        hist_fd = train_fd_omega_only(net_fd, tr, te, lr=0.001,
                                       epochs=EPOCHS, seed=seed)
        best_fd = hist_fd[-1]['acc']  # final-epoch, not best-during-training

        tp_accs.append(best_tp)
        fd_accs.append(best_fd)
        gap = abs(best_tp - best_fd)
        print(f"  seed {seed:2d}: TP={best_tp:.1%} FD={best_fd:.1%} gap={gap:.1%} "
              f"({time.time()-t0:.0f}s)", flush=True)

    gaps = [abs(t - f) for t, f in zip(tp_accs, fd_accs)]
    print(f"\n  Mean gap: {np.mean(gaps):.1%} ± {np.std(gaps):.1%}")
    print(f"  Max gap:  {max(gaps):.1%}")
    print(f"  TP mean:  {np.mean(tp_accs):.1%} ± {np.std(tp_accs):.1%}")
    print(f"  FD mean:  {np.mean(fd_accs):.1%} ± {np.std(fd_accs):.1%}")

    with open('experiments/gradient_validation_clean_results.json', 'w') as f:
        json.dump({'tp_accs': tp_accs, 'fd_accs': fd_accs,
                   'gaps': gaps, 'mean_gap': float(np.mean(gaps))},
                  f, indent=2, default=str)
