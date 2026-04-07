#!/usr/bin/env python3.12
"""FM demodulation v2: raw features that DON'T already solve the problem.

Binary task: low mod-frequency vs high mod-frequency.
Features: signal autocorrelation at multiple lags.
Autocorrelation preserves temporal structure but doesn't directly
reveal the modulating frequency — the oscillator has to extract it.

The key: autocorrelation of an FM signal has periodic structure at
the modulating frequency, but it's mixed into the carrier modulation.
A linear model sees noise. An oscillator can lock to the periodicity.
"""
import json, time, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from phasegrad.kuramoto import make_network
from phasegrad.training import train


def generate_fm(carrier, mod_freq, mod_depth, n_samples, noise=0.1, rng=None):
    """Generate FM signal samples."""
    if rng is None:
        rng = np.random.default_rng()
    t = np.linspace(0, 0.1, n_samples)  # 100ms
    phase_offset = rng.uniform(0, 2 * np.pi)
    signal = np.sin(2*np.pi*carrier*t + mod_depth*np.sin(2*np.pi*mod_freq*t) + phase_offset)
    signal += noise * rng.standard_normal(n_samples)
    return signal, t


def autocorr_features(signal, n_lags=8):
    """Autocorrelation at geometrically-spaced lags.

    These features encode temporal periodicity without directly
    revealing frequency content. A linear model gets ambiguous
    information; the oscillator network can exploit the nonlinear
    phase dynamics to decode it.
    """
    n = len(signal)
    lags = np.unique(np.geomspace(1, n // 4, n_lags).astype(int))
    if len(lags) < n_lags:
        lags = np.linspace(1, n // 4, n_lags).astype(int)

    features = np.zeros(len(lags))
    norm = np.sum(signal ** 2)
    if norm < 1e-10:
        return features

    for i, lag in enumerate(lags):
        features[i] = np.sum(signal[:n-lag] * signal[lag:]) / norm

    # Also add zero-crossing rate and energy variance as extra features
    zc = np.sum(np.abs(np.diff(np.sign(signal)))) / (2 * len(signal))
    energy_var = np.std(signal[:n//2]**2) / (np.mean(signal**2) + 1e-10)

    features = np.concatenate([features, [zc, energy_var]])

    # Normalize to [-1, 1]
    fmax = np.max(np.abs(features))
    if fmax > 1e-10:
        features = features / fmax
    return features.astype(np.float32)


def make_binary_fm_dataset(n_train=200, n_test=80, carrier=500,
                            low_range=(30, 80), high_range=(150, 300),
                            mod_depth=3.0, n_features=10, seed=42):
    """Binary: low mod-freq (class 0) vs high mod-freq (class 1).

    The frequency ranges are chosen so autocorrelation features
    overlap — making it hard for a linear model.
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_train + n_test):
        cls = rng.integers(2)
        if cls == 0:
            mod_freq = rng.uniform(*low_range)
        else:
            mod_freq = rng.uniform(*high_range)

        # Vary carrier slightly
        c = carrier * (1 + rng.uniform(-0.05, 0.05))
        md = mod_depth * (1 + rng.uniform(-0.3, 0.3))

        signal, t = generate_fm(c, mod_freq, md, n_samples=500, noise=0.15, rng=rng)
        features = autocorr_features(signal, n_lags=n_features - 2)
        samples.append((features, cls))

    return samples[:n_train], samples[n_train:n_train+n_test]


def run_experiment(n_features=10, n_hidden=10, mode='omega_only',
                   lr=0.005, epochs=200, seed=42):
    """Run one training configuration."""
    train_data, test_data = make_binary_fm_dataset(
        n_features=n_features, seed=seed)

    net = make_network(n_input=n_features, n_hidden=n_hidden, n_output=2,
                       K_scale=2.0, input_scale=1.5, seed=seed)

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
    return best['acc'], net.N, len(net.learnable_ids), len(net.edges)


if __name__ == '__main__':
    print("FM Demodulation v2: Raw Autocorrelation Features")
    print("=" * 60)

    N_FEATURES = 10
    N_HIDDEN = 10
    N_SEEDS = 20
    EPOCHS = 200

    # LogReg baseline
    lr_accs = []
    for seed in range(N_SEEDS):
        tr, te = make_binary_fm_dataset(n_features=N_FEATURES, seed=seed)
        X_tr = np.array([x for x, _ in tr])
        y_tr = np.array([y for _, y in tr])
        X_te = np.array([x for x, _ in te])
        y_te = np.array([y for _, y in te])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_tr, y_tr)
        lr_accs.append(float(clf.score(X_te, y_te)))
    print(f"  LogReg: {np.mean(lr_accs):.1%} ± {np.std(lr_accs):.1%}")
    print(f"  Chance: 50.0%")
    print(f"  Architecture: {N_FEATURES}in + {N_HIDDEN}hid + 2out = {N_FEATURES+N_HIDDEN+2} osc")

    # Ablation across lr values too
    print(f"\n  Running ablation ({N_SEEDS} seeds, {EPOCHS} epochs)...")
    t0 = time.time()
    results = {}

    for mode in ['omega_only', 'K_only', 'both']:
        for lr in [0.001, 0.005, 0.01]:
            accs = []
            for seed in range(N_SEEDS):
                acc, N, nw, nk = run_experiment(
                    N_FEATURES, N_HIDDEN, mode=mode, lr=lr,
                    epochs=EPOCHS, seed=seed)
                accs.append(acc)
            conv = [a for a in accs if a > 0.6]
            key = f"{mode}_lr{lr}"
            results[key] = accs
            cm = f"{np.mean(conv):.1%}" if conv else "N/A"
            print(f"    {mode:12s} lr={lr}: mean={np.mean(accs):.1%} "
                  f"conv={len(conv)}/{N_SEEDS} conv_acc={cm} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # Summary: best config per mode
    print(f"\n{'='*60}")
    print(f"Best config per mode:")
    for mode in ['omega_only', 'K_only', 'both']:
        best_key = max([k for k in results if k.startswith(mode)],
                       key=lambda k: np.mean(results[k]))
        accs = results[best_key]
        conv = [a for a in accs if a > 0.6]
        cm = f"{np.mean(conv):.1%}" if conv else "N/A"
        print(f"  {best_key:25s}: mean={np.mean(accs):.1%} "
              f"conv={len(conv)}/{N_SEEDS} conv_acc={cm}")
    print(f"  {'LogReg':25s}: mean={np.mean(lr_accs):.1%}")

    _, _, nw, nk = run_experiment(N_FEATURES, N_HIDDEN, epochs=1, seed=42)
    print(f"\n  ω params: {nw} | K params: {nk} | ratio: {nk/nw:.1f}×")

    with open('experiments/fm_raw_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to experiments/fm_raw_results.json")
