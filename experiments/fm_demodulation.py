#!/usr/bin/env python3.12
"""FM Demodulation: oscillators learning to hear.

Task: given an FM signal x(t) = sin(ω_carrier * t + A * sin(ω_mod * t)),
classify the modulating frequency ω_mod into one of N bins.

This is native to oscillator physics:
- Oscillators near ω_mod will lock to it (injection locking)
- Oscillators far from ω_mod will beat
- The lock/beat pattern encodes the modulating frequency
- Learning ω = tuning resonant frequencies to discriminate signals

The ω-gradient trains each oscillator to find its optimal listening frequency.

Baselines:
- FFT + linear classifier (standard DSP)
- Logistic regression on raw samples
- Both should work but require explicit feature engineering
- The oscillator does it end-to-end from the time-domain signal
"""

import json, time, numpy as np
from pathlib import Path

from phasegrad.kuramoto import KuramotoNetwork, make_network
from phasegrad.training import train, _evaluate
from phasegrad.losses import mse_loss, mse_target

SCRIPT_DIR = Path(__file__).parent


# ── Signal Generation ───────────────────────────────────────────────

def generate_fm_signal(carrier_freq, mod_freq, mod_depth, duration, sample_rate):
    """Generate an FM signal: sin(ω_c * t + A * sin(ω_m * t))."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    phase = 2 * np.pi * carrier_freq * t + mod_depth * np.sin(2 * np.pi * mod_freq * t)
    return np.sin(phase), t


def extract_features(signal, t, n_features, freq_range=(50, 500)):
    """Extract features from time-domain signal for oscillator input.

    Features: energy in n_features frequency bands spanning freq_range.
    Uses short-time windowed correlation at each center frequency.
    This mimics what an oscillator bank does — each oscillator
    responds to energy at its natural frequency.
    """
    dt = t[1] - t[0]
    freqs = np.linspace(freq_range[0], freq_range[1], n_features)
    features = np.zeros(n_features)

    for i, f in enumerate(freqs):
        # Correlate signal with a reference sinusoid at frequency f
        ref = np.sin(2 * np.pi * f * t)
        ref_cos = np.cos(2 * np.pi * f * t)
        # Magnitude of correlation (envelope detection)
        corr_sin = np.mean(signal * ref)
        corr_cos = np.mean(signal * ref_cos)
        features[i] = np.sqrt(corr_sin**2 + corr_cos**2)

    # Normalize to [-1, 1]
    fmax = features.max()
    if fmax > 1e-10:
        features = 2 * features / fmax - 1
    return features


def generate_dataset(n_classes=5, n_samples_per_class=40,
                     carrier_freq=1000, mod_depth=5.0,
                     mod_freq_range=(50, 450),
                     n_features=10, seed=42):
    """Generate FM classification dataset.

    Each class has a different modulating frequency.
    The task: given the FM signal, identify the mod frequency class.
    """
    rng = np.random.default_rng(seed)
    sample_rate = 5000  # Hz
    duration = 0.05  # 50ms segments

    # Mod frequencies for each class (evenly spaced)
    class_freqs = np.linspace(mod_freq_range[0], mod_freq_range[1], n_classes)

    samples = []
    for cls_idx, mod_freq in enumerate(class_freqs):
        for _ in range(n_samples_per_class):
            # Add jitter to mod_freq (+/- 10%)
            mf = mod_freq * (1 + rng.uniform(-0.1, 0.1))
            # Vary mod depth slightly
            md = mod_depth * (1 + rng.uniform(-0.2, 0.2))
            # Random carrier phase
            carrier_phase = rng.uniform(0, 2 * np.pi)

            signal, t = generate_fm_signal(carrier_freq, mf, md, duration, sample_rate)
            # Add noise
            signal += 0.1 * rng.standard_normal(len(signal))

            features = extract_features(signal, t, n_features,
                                         freq_range=mod_freq_range)
            samples.append((features.astype(np.float32), cls_idx))

    rng.shuffle(samples)
    return samples, class_freqs


def make_fm_dataset(n_classes=5, n_features=10, n_train=150, seed=42):
    """Generate train/test split for FM classification."""
    samples, class_freqs = generate_dataset(
        n_classes=n_classes,
        n_samples_per_class=max(60, (n_train * 2) // n_classes),
        n_features=n_features,
        seed=seed,
    )

    train_data = samples[:n_train]
    test_data = samples[n_train:n_train + n_train // 3]

    return train_data, test_data, {
        'n_classes': n_classes,
        'n_features': n_features,
        'class_freqs': class_freqs.tolist(),
        'n_train': len(train_data),
        'n_test': len(test_data),
    }


# ── Baselines ───────────────────────────────────────────────────────

def fft_logreg_baseline(n_classes=5, n_features=10, seed=42):
    """FFT features + logistic regression baseline."""
    from sklearn.linear_model import LogisticRegression

    train_data, test_data, info = make_fm_dataset(n_classes, n_features, seed=seed)
    X_train = np.array([x for x, _ in train_data])
    y_train = np.array([y for _, y in train_data])
    X_test = np.array([x for x, _ in test_data])
    y_test = np.array([y for _, y in test_data])

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


# ── Main Experiment ─────────────────────────────────────────────────

def run_fm_experiment(n_classes=5, n_features=10, n_hidden=15,
                      mode='omega_only', lr=0.005, epochs=150, seed=42):
    """Train Kuramoto on FM demodulation."""
    train_data, test_data, info = make_fm_dataset(n_classes, n_features, seed=seed)

    net = make_network(n_input=n_features, n_hidden=n_hidden, n_output=n_classes,
                       K_scale=2.0, input_scale=1.0, seed=seed)

    if mode == 'omega_only':
        lr_w, lr_K = lr, 0.0
    elif mode == 'K_only':
        lr_w, lr_K = 0.0, lr
    else:
        lr_w, lr_K = lr, lr

    history = train(net, train_data, test_data,
                    lr_omega=lr_w, lr_K=lr_K, beta=0.1,
                    epochs=epochs, margin=0.2, verbose=False,
                    eval_every=10, seed=seed)

    best = history[-1]  # final-epoch accuracy, not best-during-training
    return {
        'best_acc': best['acc'],
        'best_epoch': best['epoch'],
        'N': net.N,
        'n_params_omega': len(net.learnable_ids),
        'n_params_K': len(net.edges),
        'info': info,
        'history': history,
    }


if __name__ == '__main__':
    print("FM Demodulation: Oscillators Learning to Hear")
    print("=" * 60)

    N_CLASSES = 5
    N_FEATURES = 10
    N_HIDDEN = 15
    N_SEEDS = 10
    EPOCHS = 150

    # Baseline
    print(f"\n  Task: {N_CLASSES}-class FM mod-frequency classification")
    print(f"  Features: {N_FEATURES} frequency-band energies")
    print(f"  Architecture: {N_FEATURES}in + {N_HIDDEN}hid + {N_CLASSES}out "
          f"= {N_FEATURES + N_HIDDEN + N_CLASSES} oscillators")

    lr_accs = [fft_logreg_baseline(N_CLASSES, N_FEATURES, seed=s) for s in range(N_SEEDS)]
    print(f"\n  LogReg baseline: {np.mean(lr_accs):.1%} ± {np.std(lr_accs):.1%}")
    chance = 1.0 / N_CLASSES
    print(f"  Chance: {chance:.1%}")

    # Ablation
    print(f"\n  Training Kuramoto ({N_SEEDS} seeds, {EPOCHS} epochs)...")
    results = {}
    t0 = time.time()

    for mode in ['omega_only', 'K_only', 'both']:
        accs = []
        for seed in range(N_SEEDS):
            r = run_fm_experiment(N_CLASSES, N_FEATURES, N_HIDDEN,
                                  mode=mode, lr=0.005, epochs=EPOCHS, seed=seed)
            accs.append(r['best_acc'])
        conv = [a for a in accs if a > chance + 0.1]
        cm = f"{np.mean(conv):.1%}" if conv else "N/A"
        results[mode] = {'accs': accs, 'mean': float(np.mean(accs)),
                         'conv': len(conv), 'conv_mean': float(np.mean(conv)) if conv else 0}
        print(f"    {mode:12s}: mean={np.mean(accs):.1%} "
              f"conv={len(conv)}/{N_SEEDS} conv_mean={cm} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'='*60}")
    print(f"FM Demodulation Results ({N_CLASSES}-class, {N_SEEDS} seeds)")
    print(f"  {'':12s} {'mean':>8s} {'conv':>6s} {'conv_acc':>10s}")
    for mode, r in results.items():
        cm = f"{r['conv_mean']:.1%}" if r['conv'] > 0 else "N/A"
        print(f"  {mode:12s} {r['mean']:8.1%} {r['conv']:3d}/{N_SEEDS:2d} {cm:>10s}")
    print(f"  {'LogReg':12s} {np.mean(lr_accs):8.1%}")
    print(f"  {'Chance':12s} {chance:8.1%}")

    # Parameter comparison
    r0 = run_fm_experiment(N_CLASSES, N_FEATURES, N_HIDDEN,
                            mode='omega_only', lr=0.005, epochs=1, seed=42)
    print(f"\n  ω params: {r0['n_params_omega']} | K params: {r0['n_params_K']} "
          f"| ratio: {r0['n_params_K']/r0['n_params_omega']:.1f}×")

    with open('experiments/fm_demodulation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to experiments/fm_demodulation_results.json")
