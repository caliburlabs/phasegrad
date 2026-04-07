"""Test: finite-difference training as ground truth for the two-phase rule.

If the two-phase gradient is computing the correct gradient, then a network
trained with FD gradients should achieve similar accuracy. This validates
the gradient rule end-to-end through learning.
"""

import numpy as np
import pytest

from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.gradient import finite_difference_gradient, mse_loss
from phasegrad.losses import mse_target
from phasegrad.training import train, _evaluate


def _train_with_fd(net, train_data, test_data, lr_omega=0.001, beta=0.1,
                   epochs=100, margin=0.2, seed=42):
    """Train using finite-difference gradients (no two-phase rule).

    Same architecture, same data, same loss — only the gradient method differs.
    This is the ground truth for what the learning rule should achieve.

    Note: only updates omega (not K), since FD for K is prohibitively slow.
    """
    rng = np.random.default_rng(seed)
    history = []

    ev0 = _evaluate(net, test_data, margin)
    history.append({"epoch": 0, **ev0})

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(train_data))
        total_loss = correct = n = 0
        theta_warm = None

        for idx in indices:
            x, cls = train_data[idx]
            net.set_input(x)
            target = mse_target(net.N, net.output_ids, cls, margin)

            theta_free, res = net.equilibrium(theta_warm)
            if res > 0.01:
                continue

            loss = mse_loss(theta_free, target, net.output_ids)
            pred = net.classify(theta_free)
            total_loss += loss
            correct += (pred == cls)
            n += 1

            # FD gradient for omega only
            grad_fd = finite_difference_gradient(
                net, theta_free, target, eps=1e-5)

            # SGD update
            for i in net.learnable_ids:
                net.omega[i] -= lr_omega * np.clip(grad_fd[i], -2.0, 2.0)
            net.omega[list(net.learnable_ids)] = np.clip(
                net.omega[list(net.learnable_ids)], -3.0, 3.0)

            theta_warm = theta_free

        if epoch % 20 == 0 or epoch == epochs:
            ev = _evaluate(net, test_data, margin)
            history.append({"epoch": epoch, **ev,
                           "train_loss": total_loss / max(n, 1),
                           "train_acc": correct / max(n, 1)})

    return history


class TestFDTraining:
    """Finite-difference training as ground truth."""

    def test_fd_training_above_chance(self):
        """FD-trained network achieves above-chance accuracy."""
        train_data, test_data, _ = load_hillenbrand(vowels=['a', 'i'], seed=42)
        net = make_network(n_input=2, n_hidden=5, n_output=2,
                           K_scale=2.0, input_scale=1.5, seed=42)

        history = _train_with_fd(net, train_data, test_data,
                                 lr_omega=0.001, epochs=80, seed=42)
        best = max(history, key=lambda h: h["acc"])

        print(f"\n  FD training: best accuracy = {best['acc']:.1%} "
              f"@ epoch {best['epoch']}")

        assert best["acc"] > 0.55, (
            f"FD training accuracy {best['acc']:.1%} not above 55%")

    def test_twophase_matches_fd_training(self):
        """Two-phase and FD training achieve similar accuracy."""
        train_data, test_data, _ = load_hillenbrand(vowels=['a', 'i'], seed=42)

        # Two-phase training
        net_tp = make_network(n_input=2, n_hidden=5, n_output=2,
                              K_scale=2.0, input_scale=1.5, seed=42)
        hist_tp = train(net_tp, train_data, test_data,
                        lr_omega=0.001, lr_K=0.001, beta=0.1,
                        epochs=100, verbose=False, eval_every=20, seed=42)
        best_tp = max(hist_tp, key=lambda h: h["acc"])

        # FD training (same initial weights via same seed)
        net_fd = make_network(n_input=2, n_hidden=5, n_output=2,
                              K_scale=2.0, input_scale=1.5, seed=42)
        hist_fd = _train_with_fd(net_fd, train_data, test_data,
                                 lr_omega=0.001, epochs=100, seed=42)
        best_fd = max(hist_fd, key=lambda h: h["acc"])

        print(f"\n  Two-phase: {best_tp['acc']:.1%}, FD: {best_fd['acc']:.1%}, "
              f"gap: {abs(best_tp['acc'] - best_fd['acc']):.1%}")

        # They should be within 15% of each other
        # (FD only updates omega, two-phase updates omega + K, so
        # two-phase may actually do better)
        gap = abs(best_tp["acc"] - best_fd["acc"])
        assert gap < 0.15, (
            f"Two-phase ({best_tp['acc']:.1%}) and FD ({best_fd['acc']:.1%}) "
            f"diverge by {gap:.1%}")
