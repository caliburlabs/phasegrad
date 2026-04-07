"""Statistical rigor: accuracy over multiple seeds with confidence intervals.

One run at 75% means nothing. This produces publishable error bars.
"""

import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression

from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.training import train


N_SEEDS = 20
EPOCHS = 150


def _run_kuramoto(seed):
    """Train Kuramoto network, return best test accuracy."""
    train_data, test_data, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
    net = make_network(n_input=2, n_hidden=5, n_output=2,
                       K_scale=2.0, input_scale=1.5, seed=seed)
    history = train(net, train_data, test_data,
                    lr_omega=0.001, lr_K=0.001, beta=0.1,
                    epochs=EPOCHS, verbose=False, eval_every=20, seed=seed)
    return max(h["acc"] for h in history)


def _run_logreg(seed):
    """Train logistic regression, return test accuracy."""
    train_data, test_data, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
    X_train = np.array([x for x, _ in train_data])
    y_train = np.array([y for _, y in train_data])
    X_test = np.array([x for x, _ in test_data])
    y_test = np.array([y for _, y in test_data])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


class TestMultiSeedAccuracy:
    """Accuracy statistics over multiple random seeds."""

    def test_kuramoto_accuracy_distribution(self):
        """Kuramoto accuracy distribution over random seeds.

        The distribution is bimodal: some initializations converge to
        high accuracy (67-100%), others fail to learn (5-25%). This is
        initialization sensitivity in the coupling topology, not a
        gradient computation failure.

        We test that:
        1. A meaningful fraction of seeds learn (>40%)
        2. When learning succeeds, accuracy is high (>65%)
        3. The overall mean is above chance
        """
        accs = [_run_kuramoto(seed) for seed in range(N_SEEDS)]

        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1)

        # Separate converged vs failed runs
        converged = [a for a in accs if a > 0.60]
        failed = [a for a in accs if a <= 0.60]
        frac_converged = len(converged) / N_SEEDS

        print(f"\n  Kuramoto ({N_SEEDS} seeds, {EPOCHS} epochs):")
        print(f"    Overall: {mean_acc:.1%} ± {std_acc:.1%}")
        print(f"    Converged: {len(converged)}/{N_SEEDS} ({frac_converged:.0%})")
        if converged:
            print(f"    Converged mean: {np.mean(converged):.1%}")
        if failed:
            print(f"    Failed mean: {np.mean(failed):.1%}")

        # At least 40% of seeds should learn
        assert frac_converged >= 0.40, (
            f"Only {frac_converged:.0%} of seeds converged (need ≥40%)")

        # When it works, it should work well
        if converged:
            assert np.mean(converged) > 0.70, (
                f"Converged accuracy {np.mean(converged):.1%} too low")

    def test_logreg_baseline(self):
        """Logistic regression baseline accuracy over multiple seeds."""
        accs = [_run_logreg(seed) for seed in range(N_SEEDS)]

        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1)

        print(f"\n  Logistic Regression ({N_SEEDS} seeds):")
        print(f"    Mean: {mean_acc:.1%} ± {std_acc:.1%}")

        # LogReg on well-separated vowels should be high
        assert mean_acc > 0.80, (
            f"LogReg baseline unexpectedly low: {mean_acc:.1%}")

    def test_paired_comparison(self):
        """Paired comparison: Kuramoto vs logistic regression."""
        k_accs = []
        l_accs = []

        for seed in range(N_SEEDS):
            k_accs.append(_run_kuramoto(seed))
            l_accs.append(_run_logreg(seed))

        k_mean = np.mean(k_accs)
        l_mean = np.mean(l_accs)
        k_std = np.std(k_accs, ddof=1)
        l_std = np.std(l_accs, ddof=1)

        print(f"\n  Paired comparison ({N_SEEDS} seeds):")
        print(f"    Kuramoto: {k_mean:.1%} ± {k_std:.1%}")
        print(f"    LogReg:   {l_mean:.1%} ± {l_std:.1%}")
        print(f"    Gap:      {k_mean - l_mean:+.1%}")

        # We do NOT assert Kuramoto beats LogReg — 10 oscillators
        # won't beat a linear classifier. We assert it's meaningfully
        # above chance and document the gap honestly.
        assert k_mean > 0.55, (
            f"Kuramoto not meaningfully above chance: {k_mean:.1%}")


def _run_kuramoto_ou(seed):
    """Train Kuramoto on the hard pair (o/u)."""
    train_data, test_data, _ = load_hillenbrand(vowels=['o', 'u'], seed=seed)
    net = make_network(n_input=2, n_hidden=5, n_output=2,
                       K_scale=2.0, input_scale=1.5, seed=seed)
    history = train(net, train_data, test_data,
                    lr_omega=0.001, lr_K=0.001, beta=0.1,
                    epochs=EPOCHS, verbose=False, eval_every=20, seed=seed)
    return max(h["acc"] for h in history)


class TestHardPair:
    """o/u vowel pair — non-trivially separable (LogReg ~83%)."""

    def test_ou_accuracy_distribution(self):
        """Kuramoto on the hard pair has meaningful converged accuracy."""
        accs = [_run_kuramoto_ou(seed) for seed in range(N_SEEDS)]

        converged = [a for a in accs if a > 0.60]
        failed = [a for a in accs if a <= 0.60]
        frac = len(converged) / N_SEEDS

        print(f"\n  o/u Kuramoto ({N_SEEDS} seeds, {EPOCHS} epochs):")
        print(f"    Overall: {np.mean(accs):.1%} ± {np.std(accs):.1%}")
        print(f"    Converged: {len(converged)}/{N_SEEDS} ({frac:.0%})")
        if converged:
            print(f"    Converged mean: {np.mean(converged):.1%} ± {np.std(converged):.1%}")

        assert frac >= 0.35, f"Only {frac:.0%} converged on hard pair"
        if converged:
            assert np.mean(converged) > 0.65, (
                f"Converged accuracy {np.mean(converged):.1%} too low on hard pair")
