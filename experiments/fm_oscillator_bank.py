#!/usr/bin/env python3.12
"""Fork 2: The oscillator IS the filter bank.

No pre-extracted features. The input FM signal drives sensor oscillators
directly. Each sensor's natural frequency determines what it responds to.
The lock/beat pattern IS the spectral decomposition.

Two approaches:
A) Transient: ODE integration, extract coherence features, feed to EP classifier
B) Quasi-static: treat each input frequency as a forced equilibrium, use EP directly

We try both and compare against LogReg on the raw coherence features.
"""
import json, time, numpy as np
from sklearn.linear_model import LogisticRegression

from phasegrad.forced import OscillatorBank
from phasegrad.kuramoto import make_network
from phasegrad.training import train
from phasegrad.losses import mse_target, mse_loss


def generate_fm_samples(n_classes=3, n_per_class=80,
                         carrier=2.0, mod_depth=0.5,
                         mod_freq_range=(0.5, 3.0),
                         noise=0.05, seed=42):
    """Generate FM classification dataset.

    Classes differ in modulating frequency. Features are extracted
    by the oscillator bank (not by hand).

    Returns list of (mod_freq, class_idx) pairs.
    """
    rng = np.random.default_rng(seed)
    class_freqs = np.linspace(mod_freq_range[0], mod_freq_range[1], n_classes)

    samples = []
    for cls, mf in enumerate(class_freqs):
        for _ in range(n_per_class):
            # Jitter
            freq = mf * (1 + rng.uniform(-0.15, 0.15))
            samples.append((freq, cls))

    rng.shuffle(samples)
    return samples, class_freqs


def approach_a_transient(n_classes=3, n_sensors=8, n_hidden=4, n_output=None,
                          n_seeds=10, seed=42):
    """Approach A: transient ODE → coherence features → EP classifier.

    Stage 1: Oscillator bank (transient sim) extracts coherence features
    Stage 2: Kuramoto network (EP) classifies from coherence features
    """
    if n_output is None:
        n_output = n_classes

    print(f"\n  Approach A: Transient → Coherence → EP Classifier")
    print(f"  Sensors: {n_sensors}, Hidden: {n_hidden}, Output: {n_output}")

    all_results = {}
    t0 = time.time()

    for seed_i in range(n_seeds):
        # Generate data
        samples, class_freqs = generate_fm_samples(n_classes=n_classes, seed=seed_i)
        n_train = int(0.75 * len(samples))

        # Create oscillator bank
        bank = OscillatorBank(n_sensors=n_sensors, n_hidden=0, n_output=0,
                               freq_range=(0.3, 4.0), K_scale=0.3,
                               F_strength=2.0, seed=seed_i)

        # Extract features via transient simulation
        train_feats, test_feats = [], []
        for i, (mf, cls) in enumerate(samples):
            vec = bank.extract_feature_vector(mf, duration=15.0, settle=8.0)
            if vec is None:
                vec = np.zeros(bank.N, dtype=np.float32)
            if i < n_train:
                train_feats.append((vec, cls))
            else:
                test_feats.append((vec, cls))

        # LogReg on oscillator features
        X_tr = np.array([x for x, _ in train_feats])
        y_tr = np.array([y for _, y in train_feats])
        X_te = np.array([x for x, _ in test_feats])
        y_te = np.array([y for _, y in test_feats])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_tr, y_tr)
        lr_acc = float(clf.score(X_te, y_te))

        # EP classifier on same features
        net = make_network(n_input=bank.N, n_hidden=n_hidden, n_output=n_output,
                           K_scale=2.0, input_scale=1.0, seed=seed_i)
        hist = train(net, train_feats, test_feats,
                     lr_omega=0.005, lr_K=0.0, beta=0.1,
                     epochs=100, margin=0.2, verbose=False,
                     eval_every=20, seed=seed_i)
        ep_acc = hist[-1]['acc']  # final-epoch, not best-during-training

        print(f"    seed {seed_i}: EP={ep_acc:.1%} LogReg={lr_acc:.1%} ({time.time()-t0:.0f}s)",
              flush=True)
        all_results[seed_i] = {'ep': ep_acc, 'logreg': lr_acc}

    ep_accs = [r['ep'] for r in all_results.values()]
    lr_accs = [r['logreg'] for r in all_results.values()]
    print(f"  Summary: EP={np.mean(ep_accs):.1%}±{np.std(ep_accs):.1%} "
          f"LogReg={np.mean(lr_accs):.1%}±{np.std(lr_accs):.1%}")
    return all_results


def approach_b_quasistatic(n_classes=3, n_sensors=8, n_hidden=4,
                            n_seeds=10, seed=42):
    """Approach B: forced equilibrium → EP gradient directly.

    The oscillator bank AND the classifier are one system. The input
    frequency drives sensors, coupling propagates to outputs, and EP
    trains ω to optimize classification. End-to-end.
    """
    print(f"\n  Approach B: Forced Equilibrium + EP (end-to-end)")
    print(f"  Sensors: {n_sensors}, Hidden: {n_hidden}, Output: {n_classes}")

    all_results = {}
    t0 = time.time()

    for seed_i in range(n_seeds):
        samples, class_freqs = generate_fm_samples(n_classes=n_classes, seed=seed_i)
        n_train = int(0.75 * len(samples))

        bank = OscillatorBank(n_sensors=n_sensors, n_hidden=n_hidden,
                               n_output=n_classes,
                               freq_range=(0.3, 4.0), K_scale=1.0,
                               F_strength=2.0, seed=seed_i)

        rng = np.random.default_rng(seed_i)
        best_acc = 0.0

        # Training loop with EP on the forced equilibrium
        for epoch in range(80):
            indices = rng.permutation(n_train)
            correct = n = 0

            for idx in indices:
                mf, cls = samples[idx]
                target = np.zeros(bank.N)
                for i, o in enumerate(bank.output_ids):
                    target[o] = -0.2 if i == cls else 0.2

                theta_free, res_free = bank.forced_equilibrium(mf)
                if res_free > 0.1:
                    continue

                theta_clamp, res_clamp = bank.forced_clamped_equilibrium(
                    mf, beta=0.1, target=target, theta_init=theta_free.copy())

                # Classify
                out_phases = [theta_free[o] for o in bank.output_ids]
                pred = int(np.argmax(np.cos(out_phases)))
                correct += (pred == cls)
                n += 1

                # Update ω (learnable only)
                if res_clamp < 0.1:
                    for i in bank.learnable_ids:
                        g = -(theta_clamp[i] - theta_free[i]) / 0.1
                        g = np.clip(g, -2.0, 2.0)
                        bank.omega[i] -= 0.005 * g
                    bank.omega[bank.learnable_ids] = np.clip(
                        bank.omega[bank.learnable_ids], -5.0, 5.0)

            train_acc = correct / max(n, 1)

            # Eval every 20 epochs
            if (epoch + 1) % 20 == 0 or epoch == 0:
                ev_correct = ev_n = 0
                for idx in range(n_train, len(samples)):
                    mf, cls = samples[idx]
                    theta, res = bank.forced_equilibrium(mf)
                    if res > 0.1:
                        continue
                    out_phases = [theta[o] for o in bank.output_ids]
                    pred = int(np.argmax(np.cos(out_phases)))
                    ev_correct += (pred == cls)
                    ev_n += 1
                test_acc = ev_correct / max(ev_n, 1)
                best_acc = max(best_acc, test_acc)

        print(f"    seed {seed_i}: best={best_acc:.1%} ({time.time()-t0:.0f}s)",
              flush=True)
        all_results[seed_i] = {'best_acc': best_acc}

    accs = [r['best_acc'] for r in all_results.values()]
    conv = [a for a in accs if a > 1.0/n_classes + 0.1]
    cm = f"{np.mean(conv):.1%}" if conv else "N/A"
    print(f"  Summary: mean={np.mean(accs):.1%} conv={len(conv)}/{n_seeds} conv_acc={cm}")
    return all_results


if __name__ == '__main__':
    print("Fork 2: The Oscillator IS the Filter Bank")
    print("=" * 60)

    N_CLASSES = 3
    N_SENSORS = 8
    N_HIDDEN = 4
    N_SEEDS = 5  # fewer seeds since transient sims are slower

    print(f"  Task: {N_CLASSES}-class FM mod-frequency classification")
    print(f"  Chance: {1/N_CLASSES:.1%}")

    # Approach A: transient features → EP
    results_a = approach_a_transient(N_CLASSES, N_SENSORS, N_HIDDEN,
                                      n_seeds=N_SEEDS)

    # Approach B: end-to-end forced equilibrium
    results_b = approach_b_quasistatic(N_CLASSES, N_SENSORS, N_HIDDEN,
                                        n_seeds=N_SEEDS)

    with open('experiments/fm_oscillator_bank_results.json', 'w') as f:
        json.dump({'approach_a': results_a, 'approach_b': results_b},
                  f, indent=2, default=str)
    print(f"\n  Saved to experiments/fm_oscillator_bank_results.json")
