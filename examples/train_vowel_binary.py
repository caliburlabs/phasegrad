#!/usr/bin/env python3
"""Train a 10-oscillator Kuramoto network on binary vowel classification.

Uses only the two-phase gradient rule — no backpropagation.

Expected: ~65-75% test accuracy (50% chance baseline) using equilibrium
propagation to learn natural frequencies and coupling strengths.
"""

from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.training import train

if __name__ == "__main__":
    print("Kuramoto Binary Vowel Classifier — Equilibrium Propagation\n")

    train_data, test_data, info = load_hillenbrand(vowels=['a', 'i'], seed=42)
    print(f"  Data: {info['n_train']} train, {info['n_test']} test, "
          f"{info['n_classes']} classes ({info['vowels']})\n")

    net = make_network(n_input=2, n_hidden=5, n_output=2,
                       K_scale=2.0, input_scale=1.5, seed=42)
    print(f"  Network: {net.N} oscillators, {len(net.edges)} edges\n")

    history = train(net, train_data, test_data,
                    lr_omega=0.001, lr_K=0.001, beta=0.1,
                    epochs=300, eval_every=10)

    best = max(history, key=lambda h: h["acc"])
    print(f"\n  Best test accuracy: {best['acc']:.1%} @ epoch {best['epoch']}")
    print(f"  Chance baseline: 50.0%")
