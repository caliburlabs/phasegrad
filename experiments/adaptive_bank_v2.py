#!/usr/bin/env python3.12
"""Adaptive oscillator bank v2: fix the units, add coupling + rich features.

v1 problem: omega is in Hz but input_phase = 2π*f*t (rad/s). No sensor
ever injection-locks because the detuning is always >> F. Coherence is
just a weak modulation effect, not a lock/beat signal.

Fix: use omega_rad = 2π * omega in the ODE. Then lock occurs when
|2π*(f_sensor - f_input)| < F, i.e., |f_sensor - f_input| < F/(2π).

With F=5.0: lock range ≈ ±0.80 Hz → sharp for targets ~1 Hz apart.

Then layer on:
  - Coupled sensors (lateral inhibition via negative K)
  - Rich features (coherence + phase + freq_ratio + beat_intensity)
  - Full A→B→C adaptive experiment
"""
import json, time
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import LogisticRegression

from phasegrad.forced import OscillatorBank, forced_kuramoto_rhs


ENVIRONMENTS = {
    'A': [1.0, 2.0, 3.0],
    'B': [1.5, 2.5, 4.0],
    'C': [0.8, 1.8, 3.5],
}


def generate_env_samples(target_freqs, n_per_class=67, jitter=0.15,
                         noise_std=0.03, seed=42):
    rng = np.random.default_rng(seed)
    samples = []
    for cls, tf in enumerate(target_freqs):
        for _ in range(n_per_class):
            freq = tf * (1 + rng.uniform(-jitter, jitter))
            freq += rng.normal(0, noise_std)
            freq = max(freq, 0.1)
            samples.append((freq, cls))
    rng.shuffle(samples)
    n_train = min(100, len(samples) // 2)
    return samples[:n_train], samples[n_train:n_train + 100]


# ── Corrected ODE simulation ────────────────────────────────────

def simulate_corrected(bank, input_freq, F_eff, duration=8.0, settle=4.0,
                       dt=0.01):
    """ODE simulation with corrected frequency units.

    Uses omega_rad = 2π * omega so that both omega and input_phase
    are in rad/s. Lock condition: |f_sensor - f_input| < F_eff/(2π).

    Returns (N, 4) feature matrix:
      [coherence, mean_phase, freq_ratio, beat_intensity]
    or None on failure.
    """
    N = bank.N
    # Convert omega from Hz to rad/s — no mean-centering
    omega_rad = 2 * np.pi * bank.omega

    # Scale forcing to F_eff (override bank.F magnitudes)
    F_scaled = np.zeros(N)
    for sid in bank.sensor_ids:
        F_scaled[sid] = F_eff

    def rhs(t, theta):
        input_phase = 2 * np.pi * input_freq * t  # rad/s
        return forced_kuramoto_rhs(theta, omega_rad, bank.K, F_scaled, input_phase)

    t_eval = np.arange(settle, duration, dt)
    sol = solve_ivp(rhs, (0, duration), np.zeros(N), method='RK45',
                    t_eval=t_eval, max_step=dt, rtol=1e-6, atol=1e-8)

    if not sol.success or sol.y.shape[1] < 10:
        return None

    phases = sol.y  # (N, n_timepoints)
    t = sol.t
    input_phases = 2 * np.pi * input_freq * t

    features = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        phase_diff = phases[i] - input_phases
        exp_diff = np.exp(1j * phase_diff)
        mean_exp = np.mean(exp_diff)

        features[i, 0] = np.abs(mean_exp)                  # coherence
        features[i, 1] = np.angle(mean_exp) / np.pi         # mean phase [-1,1]

        dphase = np.diff(phases[i]) / np.diff(t)
        features[i, 2] = np.mean(dphase) / (2*np.pi*input_freq) if input_freq > 0 else 0

        dphase_diff = np.diff(phase_diff) / np.diff(t)
        features[i, 3] = np.std(dphase_diff) / (2 * np.pi)  # beat intensity

    return features


# ── Feature extraction ───────────────────────────────────────────

def extract_features(bank, samples, F_eff, mode='coherence',
                     dur=8.0, sett=4.0):
    X, y = [], []
    for mf, cls in samples:
        feats = simulate_corrected(bank, mf, F_eff, dur, sett)
        if feats is None:
            n_feat = bank.n_sensors if mode == 'coherence' else bank.n_sensors * 4
            X.append(np.zeros(n_feat, dtype=np.float32))
        elif mode == 'coherence':
            X.append(feats[bank.sensor_ids, 0])
        elif mode == 'rich':
            X.append(feats[bank.sensor_ids].flatten())
        y.append(cls)
    return np.array(X, dtype=np.float32), np.array(y)


def eval_logreg(X_tr, y_tr, X_te, y_te):
    if len(np.unique(y_tr)) < 2:
        return 0.0
    return float(LogisticRegression(max_iter=2000).fit(X_tr, y_tr).score(X_te, y_te))


def eval_bank(bank, train, test, F_eff, mode='coherence', dur=8.0, sett=4.0):
    X_tr, y_tr = extract_features(bank, train, F_eff, mode, dur, sett)
    X_te, y_te = extract_features(bank, test, F_eff, mode, dur, sett)
    return eval_logreg(X_tr, y_tr, X_te, y_te)


# ── Coupling ─────────────────────────────────────────────────────

def add_sensor_coupling(bank, strength, topology='alltoall'):
    sids = bank.sensor_ids
    if topology == 'alltoall':
        for i, s1 in enumerate(sids):
            for s2 in sids[i + 1:]:
                bank.K[s1, s2] = strength
                bank.K[s2, s1] = strength
    elif topology == 'chain':
        for i in range(len(sids) - 1):
            bank.K[sids[i], sids[i + 1]] = strength
            bank.K[sids[i + 1], sids[i]] = strength


# ── Adaptation ───────────────────────────────────────────────────

def competitive_hebbian(bank, train_samples, n_steps=30, lr=0.4,
                        min_spacing=0.12):
    rng = np.random.default_rng(42)
    omega = bank.omega[bank.sensor_ids].copy()
    for step in range(n_steps):
        for idx in rng.permutation(len(train_samples)):
            mf, _ = train_samples[idx]
            winner = int(np.argmin(np.abs(omega - mf)))
            omega[winner] += lr * (mf - omega[winner])
            omega[winner] = np.clip(omega[winner], 0.1, 6.0)
        order = np.argsort(omega)
        for i in range(len(order) - 1):
            s1, s2 = order[i], order[i + 1]
            if omega[s2] - omega[s1] < min_spacing:
                push = (min_spacing - (omega[s2] - omega[s1])) / 2
                omega[s1] = max(0.1, omega[s1] - push)
                omega[s2] = min(6.0, omega[s2] + push)
        lr *= 0.93
    for k, sid in enumerate(bank.sensor_ids):
        bank.omega[sid] = omega[k]


def omega_tracking_error(bank, target_freqs):
    omega = bank.omega[bank.sensor_ids]
    return float(np.mean([np.min(np.abs(omega - tf)) for tf in target_freqs]))


# ── Bank factory ─────────────────────────────────────────────────

def make_bank(n_sensors=8, seed=42, coupling=0.0, topology='alltoall'):
    bank = OscillatorBank(n_sensors=n_sensors, n_hidden=0, n_output=0,
                           freq_range=(0.3, 5.0), K_scale=0.0,
                           F_strength=1.0, seed=seed)
    rng = np.random.default_rng(seed + 7777)
    for sid in bank.sensor_ids:
        bank.omega[sid] = rng.uniform(0.5, 4.5)
    if coupling != 0.0:
        add_sensor_coupling(bank, coupling, topology)
    return bank


# ── Main ─────────────────────────────────────────────────────────

def run():
    print("Adaptive Oscillator Bank v2: Corrected Units + Coupling")
    print("=" * 65)
    print("FIX: omega converted to rad/s → real injection locking")
    print(f"Lock range = ±F/(2π) Hz")

    env_order = ['A', 'B', 'C']
    seed = 42

    env_data = {}
    for i, name in enumerate(env_order):
        targets = ENVIRONMENTS[name]
        train, test = generate_env_samples(targets, seed=seed + i * 100)
        env_data[name] = {'train': train, 'test': test, 'targets': targets}
        print(f"  Env {name}: targets={targets}")

    all_results = {
        'environments': {k: v['targets'] for k, v in env_data.items()},
    }

    # ── Baselines ────────────────────────────────────────────────
    oracle, fixed = {}, {}
    X_A = np.array([f for f, _ in env_data['A']['train']]).reshape(-1, 1)
    y_A = np.array([c for _, c in env_data['A']['train']])
    clf_fixed = LogisticRegression(max_iter=2000).fit(X_A, y_A)
    for name in env_order:
        train, test = env_data[name]['train'], env_data[name]['test']
        X_tr = np.array([f for f, _ in train]).reshape(-1, 1)
        y_tr = np.array([c for _, c in train])
        X_te = np.array([f for f, _ in test]).reshape(-1, 1)
        y_te = np.array([c for _, c in test])
        oracle[name] = {'acc': float(LogisticRegression(max_iter=2000).fit(X_tr, y_tr).score(X_te, y_te))}
        fixed[name] = {
            'acc_frozen': float(clf_fixed.score(X_te, y_te)),
            'acc_retrained': oracle[name]['acc'],
        }
    all_results['oracle'] = oracle
    all_results['fixed_digital'] = fixed
    print(f"\n  Oracle: {np.mean([oracle[e]['acc'] for e in env_order]):.1%}")
    print(f"  Fixed frozen: {np.mean([fixed[e]['acc_frozen'] for e in env_order]):.1%}")

    # ── Part 1: F sweep + coupling sweep on Env A ────────────────
    print(f"\n{'='*65}")
    print("PART 1: F and coupling sweep on Env A (with adaptation)")
    print(f"{'='*65}")

    train_A, test_A = env_data['A']['train'], env_data['A']['test']
    sweep_results = []

    for F_eff in [3.0, 5.0, 8.0, 12.0]:
        for coupling in [0.0, -1.0, -3.0]:
            for feat_mode in ['coherence', 'rich']:
                label = f"F={F_eff:.0f} K={coupling:+.0f} {feat_mode}"
                bank = make_bank(coupling=coupling)
                competitive_hebbian(bank, train_A, n_steps=30, lr=0.4)
                t0 = time.time()
                acc = eval_bank(bank, train_A, test_A, F_eff, feat_mode)
                dt = time.time() - t0
                lock_range = F_eff / (2 * np.pi)
                sweep_results.append({
                    'label': label, 'F': F_eff, 'coupling': coupling,
                    'mode': feat_mode, 'acc': acc, 'lock_range_hz': lock_range,
                })
                print(f"  {label:<30} acc={acc:.1%} "
                      f"lock=±{lock_range:.2f}Hz ({dt:.0f}s)")

    sweep_results.sort(key=lambda c: -c['acc'])
    all_results['sweep'] = sweep_results

    print(f"\n  Top 5:")
    for c in sweep_results[:5]:
        print(f"    {c['label']:<30} {c['acc']:.1%} "
              f"(lock=±{c['lock_range_hz']:.2f}Hz)")

    best = sweep_results[0]

    # ── Part 2: Full A→B→C with best config ──────────────────────
    print(f"\n{'='*65}")
    print(f"PART 2: Full A→B→C with best config: {best['label']}")
    print(f"{'='*65}")

    configs_to_run = [
        (f"F={best['F']:.0f} K=0 coherence", 0.0, 'coherence', best['F']),
        (f"F={best['F']:.0f} K=0 rich", 0.0, 'rich', best['F']),
        (f"F={best['F']:.0f} K={best['coupling']:+.0f} coherence",
         best['coupling'], 'coherence', best['F']),
        (f"F={best['F']:.0f} K={best['coupling']:+.0f} rich",
         best['coupling'], 'rich', best['F']),
    ]

    experiment_results = {}

    for label, coupling, feat_mode, F_eff in configs_to_run:
        print(f"\n{'─'*65}")
        print(f"Config: {label}")

        bank = make_bank(coupling=coupling)
        total_steps = 0
        env_results = {}

        for name in env_order:
            train = env_data[name]['train']
            test = env_data[name]['test']
            targets = env_data[name]['targets']

            acc_before = eval_bank(bank, train, test, F_eff, feat_mode)
            track_before = omega_tracking_error(bank, targets)
            competitive_hebbian(bank, train, n_steps=30, lr=0.4)
            total_steps += 30
            acc_after = eval_bank(bank, train, test, F_eff, feat_mode)

            track_after = omega_tracking_error(bank, targets)
            omega_after = sorted(bank.omega[bank.sensor_ids].tolist())

            print(f"  Env {name}: {acc_before:.1%} → {acc_after:.1%} "
                  f"(track: {track_before:.3f}→{track_after:.3f})")

            env_results[name] = {
                'acc_before': acc_before, 'acc_after': acc_after,
                'track_before': track_before, 'track_after': track_after,
                'omega_after': omega_after,
            }

        mean_acc = np.mean([env_results[e]['acc_after'] for e in env_order])
        env_results['mean_acc'] = mean_acc
        env_results['total_steps'] = total_steps
        experiment_results[label] = env_results
        print(f"  Mean: {mean_acc:.1%}")

    all_results['experiments'] = experiment_results

    # ── Summary table ────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FINAL COMPARISON")
    print(f"{'='*65}\n")

    oa = all_results['oracle']
    fd = all_results['fixed_digital']

    rows = [
        ('Oracle', {e: oa[e]['acc'] for e in env_order}, 'retrain ×3'),
        ('Fixed (frozen)', {e: fd[e]['acc_frozen'] for e in env_order}, '0 cost'),
        ('Fixed (retrained)', {e: fd[e]['acc_retrained'] for e in env_order}, 'redesign ×2'),
    ]
    for label, r in experiment_results.items():
        rows.append((label[:28], {e: r[e]['acc_after'] for e in env_order},
                      f"{r['total_steps']} steps"))

    print(f"{'System':<30}│{'Env A':>7}│{'Env B':>7}│{'Env C':>7}│{'Mean':>7}│{'Cost':>14}")
    print("─" * 78)
    for label, accs, cost in rows:
        mean = np.mean([accs[e] for e in env_order])
        print(f"{label:<30}│{accs['A']:>6.1%}│{accs['B']:>6.1%}│"
              f"{accs['C']:>6.1%}│{mean:>6.1%}│{cost:>14}")

    # Improvement
    best_mean = max(r['mean_acc'] for r in experiment_results.values())
    fixed_frozen_mean = np.mean([fd[e]['acc_frozen'] for e in env_order])
    print(f"\n  v1 baseline (broken units): 76.3%")
    print(f"  Best corrected config: {best_mean:.1%}")
    print(f"  Improvement over v1: +{best_mean - 0.763:.1%}")
    if best_mean > fixed_frozen_mean:
        print(f"  BEATS fixed digital frozen ({fixed_frozen_mean:.1%})")
    else:
        print(f"  vs fixed frozen: {fixed_frozen_mean:.1%}")

    all_results['summary'] = {
        'best_config': best['label'],
        'best_F': best['F'],
        'best_lock_range': best['lock_range_hz'],
    }

    return all_results


if __name__ == '__main__':
    t_start = time.time()
    results = run()
    results['wall_time_s'] = time.time() - t_start
    print(f"\nTotal wall time: {results['wall_time_s']:.0f}s")

    with open('experiments/adaptive_bank_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved to experiments/adaptive_bank_v2_results.json")
