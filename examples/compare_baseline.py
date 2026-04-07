#!/usr/bin/env python3
"""Compare Kuramoto oscillator network vs logistic regression baseline.

Runs both on the same train/test splits across multiple random seeds.
Reports accuracy with mean ± std.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.training import train


def run_comparison(n_seeds: int = 10, epochs: int = 200):
    """Run both models across multiple seeds."""
    kuramoto_accs = []
    logreg_accs = []

    for seed in range(n_seeds):
        train_data, test_data, info = load_hillenbrand(
            vowels=['a', 'i'], seed=seed)

        # Logistic regression baseline
        X_train = np.array([x for x, _ in train_data])
        y_train = np.array([y for _, y in train_data])
        X_test = np.array([x for x, _ in test_data])
        y_test = np.array([y for _, y in test_data])

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        lr_acc = float(lr.score(X_test, y_test))
        logreg_accs.append(lr_acc)

        # Kuramoto network
        net = make_network(n_input=2, n_hidden=5, n_output=2,
                           K_scale=2.0, input_scale=1.5, seed=seed)
        history = train(net, train_data, test_data,
                        lr_omega=0.001, lr_K=0.001, beta=0.1,
                        epochs=epochs, verbose=False, eval_every=20,
                        seed=seed)
        best_acc = max(h["acc"] for h in history)
        kuramoto_accs.append(best_acc)

        print(f"  Seed {seed:2d}: Kuramoto={best_acc:.1%}  LogReg={lr_acc:.1%}")

    k_mean, k_std = np.mean(kuramoto_accs), np.std(kuramoto_accs)
    l_mean, l_std = np.mean(logreg_accs), np.std(logreg_accs)

    print(f"\n{'='*50}")
    print(f"  Kuramoto (two-phase EP): {k_mean:.1%} ± {k_std:.1%}")
    print(f"  Logistic Regression:     {l_mean:.1%} ± {l_std:.1%}")
    print(f"  Chance baseline:         50.0%")
    print(f"{'='*50}")

    return {
        "kuramoto": {"mean": k_mean, "std": k_std, "values": kuramoto_accs},
        "logreg": {"mean": l_mean, "std": l_std, "values": logreg_accs},
    }


if __name__ == "__main__":
    print("Baseline Comparison: Kuramoto vs Logistic Regression\n")
    run_comparison(n_seeds=10, epochs=200)
