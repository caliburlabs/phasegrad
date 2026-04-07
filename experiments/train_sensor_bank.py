#!/usr/bin/env python3.12
"""Train sensor oscillator frequencies: learning to hear (v2).

Fixed from v1:
- Use LogReg log-loss (smooth) instead of accuracy (piecewise constant)
- Harder task: 5 classes, 4 sensors (forces selectivity)
- Larger perturbation epsilon for FD to get nonzero gradients
"""
import json, time, numpy as np
from sklearn.linear_model import LogisticRegression

from phasegrad.forced import OscillatorBank


def generate_samples(n_classes=5, n_per_class=80,
                      mod_freq_range=(0.5, 4.0), seed=42):
    rng = np.random.default_rng(seed)
    class_freqs = np.linspace(mod_freq_range[0], mod_freq_range[1], n_classes)
    samples = []
    for cls, mf in enumerate(class_freqs):
        for _ in range(n_per_class):
            freq = mf * (1 + rng.uniform(-0.12, 0.12))
            samples.append((freq, cls))
    rng.shuffle(samples)
    return samples, class_freqs


def get_features(bank, samples):
    """Extract coherence features for all samples."""
    X, y = [], []
    for mf, cls in samples:
        feats = bank.simulate_transient(mf, duration=12.0, settle=6.0)
        if feats is None:
            vec = np.zeros(bank.n_sensors)
        else:
            vec = np.array([feats[i]['coherence'] for i in bank.sensor_ids])
        X.append(vec)
        y.append(cls)
    return np.array(X, dtype=np.float32), np.array(y)


def evaluate_logloss(bank, samples, train_frac=0.75):
    """Train LogReg, return negative log-loss (smooth) and accuracy."""
    X, y = get_features(bank, samples)
    n = int(len(X) * train_frac)
    X_tr, y_tr = X[:n], y[:n]
    X_te, y_te = X[n:], y[n:]

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)

    # Log-loss on test set (lower is better)
    probs = clf.predict_proba(X_te)
    logloss = -np.mean([np.log(probs[i, y_te[i]] + 1e-10) for i in range(len(y_te))])
    acc = float(clf.score(X_te, y_te))
    return logloss, acc


def make_bank(n_sensors, freq_range, seed, random_init=True):
    """Create oscillator bank with random or uniform sensor frequencies."""
    bank = OscillatorBank(n_sensors=n_sensors, n_hidden=0, n_output=0,
                           freq_range=freq_range, K_scale=0.3,
                           F_strength=2.0, seed=seed)
    if random_init:
        rng = np.random.default_rng(seed + 5000)
        for sid in bank.sensor_ids:
            bank.omega[sid] = rng.uniform(*freq_range)
    return bank


def train_sensors(n_classes=5, n_sensors=4, n_steps=20, lr=0.5,
                   eps=0.3, seed=42):
    """Train sensor ω by FD gradient on log-loss."""
    samples, class_freqs = generate_samples(n_classes=n_classes, seed=seed)
    freq_range = (0.3, 5.0)
    bank = make_bank(n_sensors, freq_range, seed, random_init=True)

    # Subsample for faster gradient
    grad_samples = samples[:min(80, len(samples))]

    history = []
    logloss, acc = evaluate_logloss(bank, samples)
    print(f"  Step  0: loss={logloss:.3f} acc={acc:.1%} "
          f"ω={np.sort(bank.omega[bank.sensor_ids]).round(2)}")
    history.append({'step': 0, 'logloss': logloss, 'acc': acc,
                    'omega': sorted(bank.omega[bank.sensor_ids].tolist())})

    for step in range(1, n_steps + 1):
        grad = np.zeros(n_sensors)
        for k, sid in enumerate(bank.sensor_ids):
            orig = bank.omega[sid]

            bank.omega[sid] = orig + eps
            loss_plus, _ = evaluate_logloss(bank, grad_samples)

            bank.omega[sid] = orig - eps
            loss_minus, _ = evaluate_logloss(bank, grad_samples)

            # Gradient of log-loss (we want to MINIMIZE loss)
            grad[k] = (loss_plus - loss_minus) / (2 * eps)
            bank.omega[sid] = orig

        # Gradient descent on log-loss
        for k, sid in enumerate(bank.sensor_ids):
            bank.omega[sid] -= lr * grad[k]
            bank.omega[sid] = np.clip(bank.omega[sid], 0.1, 6.0)

        logloss, acc = evaluate_logloss(bank, samples)
        print(f"  Step {step:2d}: loss={logloss:.3f} acc={acc:.1%} "
              f"ω={np.sort(bank.omega[bank.sensor_ids]).round(2)} "
              f"|∇|={np.linalg.norm(grad):.3f}", flush=True)
        history.append({'step': step, 'logloss': logloss, 'acc': acc,
                        'omega': sorted(bank.omega[bank.sensor_ids].tolist()),
                        'grad_norm': float(np.linalg.norm(grad))})

    return history, bank, class_freqs


if __name__ == '__main__':
    print("Train Sensor Bank v2: 5-class, 4 sensors, log-loss gradient")
    print("=" * 60)

    N_CLASSES = 5
    N_SENSORS = 4
    N_STEPS = 15
    N_SEEDS = 3

    samples, class_freqs = generate_samples(n_classes=N_CLASSES, seed=42)
    print(f"  Classes: {N_CLASSES}, Sensors: {N_SENSORS}")
    print(f"  Class freqs: {class_freqs.round(2)}")
    print(f"  Chance: {1/N_CLASSES:.1%}")

    # Uniform baseline
    bank_uni = make_bank(N_SENSORS, (0.3, 5.0), seed=42, random_init=False)
    ll_uni, acc_uni = evaluate_logloss(bank_uni, samples)
    print(f"\n  Uniform: loss={ll_uni:.3f} acc={acc_uni:.1%} "
          f"ω={np.sort(bank_uni.omega[bank_uni.sensor_ids]).round(2)}")

    # Random baselines (no training)
    random_accs = []
    for s in range(10):
        bank_r = make_bank(N_SENSORS, (0.3, 5.0), seed=s, random_init=True)
        _, acc_r = evaluate_logloss(bank_r, samples)
        random_accs.append(acc_r)
    print(f"  Random (10 seeds): {np.mean(random_accs):.1%} ± {np.std(random_accs):.1%}")

    # Trained
    all_results = []
    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed} ---")
        history, bank, cf = train_sensors(N_CLASSES, N_SENSORS,
                                           n_steps=N_STEPS, seed=seed)
        all_results.append({
            'seed': seed,
            'init_acc': history[0]['acc'],
            'final_acc': history[-1]['acc'],
            'init_loss': history[0]['logloss'],
            'final_loss': history[-1]['logloss'],
            'history': history,
        })

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Uniform:  {acc_uni:.1%}")
    print(f"  Random:   {np.mean(random_accs):.1%} ± {np.std(random_accs):.1%}")
    for r in all_results:
        print(f"  Trained (seed {r['seed']}): {r['init_acc']:.1%} → {r['final_acc']:.1%} "
              f"(loss {r['init_loss']:.3f} → {r['final_loss']:.3f})")
    print(f"  Chance:   {1/N_CLASSES:.1%}")

    with open('experiments/train_sensor_bank_results.json', 'w') as f:
        json.dump({'uniform_acc': acc_uni, 'random_mean': float(np.mean(random_accs)),
                   'results': all_results}, f, indent=2, default=str)
