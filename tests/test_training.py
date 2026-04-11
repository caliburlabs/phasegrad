"""Test suite: training produces above-chance accuracy on vowel classification."""

import pytest
from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.training import train


def test_binary_vowel_above_chance():
    """Binary (a vs i) classifier trained with two-phase gradients beats chance."""
    train_data, test_data, info = load_hillenbrand(vowels=['a', 'i'], seed=42)
    net = make_network(n_input=2, n_hidden=5, n_output=2,
                       K_scale=2.0, input_scale=1.5, seed=42)

    history = train(net, train_data, test_data,
                    lr_omega=0.001, lr_K=0.001, beta=0.1,
                    epochs=100, verbose=False, eval_every=20)

    best_acc = max(h["acc"] for h in history)

    # Must beat chance (50%) by a meaningful margin
    assert best_acc > 0.60, (
        f"Best accuracy {best_acc:.1%} not above 60% threshold")


def test_loss_decreases():
    """Training loss should decrease over epochs."""
    train_data, test_data, info = load_hillenbrand(vowels=['a', 'i'], seed=42)
    net = make_network(n_input=2, n_hidden=5, n_output=2,
                       K_scale=2.0, input_scale=1.5, seed=42)

    history = train(net, train_data, test_data,
                    lr_omega=0.001, lr_K=0.001, beta=0.1,
                    epochs=50, verbose=False, eval_every=10)

    losses = [h["loss"] for h in history if h["loss"] is not None]
    # At least the last loss should be lower than the first
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}")
