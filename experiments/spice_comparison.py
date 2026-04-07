#!/usr/bin/env python3.12
"""SPICE vs Kuramoto comparison for coupled ring oscillator pairs.

Uses the existing SPICE infrastructure from kb/exploration/substrate/
to simulate real ring oscillator pairs in GPDK045 and compare the
phase relationship with the Kuramoto model prediction.

For each (VDD_A, VDD_B, coupling_strength) triple:
1. Run SPICE simulation → extract phase_A, phase_B, locked/beat
2. Run Kuramoto model with equivalent parameters → predict phase diff
3. Compare

This bridges the theory (Kuramoto) to hardware (CMOS ring oscillators).
"""
import json, sys, math, time, os
import numpy as np

# Add the substrate code to path
SUBSTRATE_DIR = os.path.expanduser('~/analog-gradients/kb/exploration/substrate')
sys.path.insert(0, SUBSTRATE_DIR)

try:
    from freq_discrimination import run_pair
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False
    print("WARNING: SPICE infrastructure not available. Run from CMC machine.")


def kuramoto_phase_diff(omega_a, omega_b, K):
    """Kuramoto model prediction for phase difference of a locked pair.

    For two coupled oscillators:
        dθ_a/dt = ω_a + K sin(θ_b - θ_a)
        dθ_b/dt = ω_b + K sin(θ_a - θ_b)

    At equilibrium in the rotating frame:
        Δθ = arcsin((ω_a - ω_b) / (2K))  if |ω_a - ω_b| < 2K (locked)
        otherwise: unlocked (beating)

    Returns (phase_diff_rad, locked).
    """
    delta_omega = omega_a - omega_b
    lock_range = 2 * K
    if abs(delta_omega) < lock_range:
        phase_diff = math.asin(delta_omega / lock_range)
        return phase_diff, True
    else:
        return None, False


def vdd_to_freq_ghz(vdd):
    """Approximate VDD → frequency mapping for 3-inverter ring oscillator.

    Based on SPICE characterization data from freq_discrimination experiments.
    Linear approximation: f ≈ 0.5 + 1.5 * (VDD - 1.4) / 0.8 GHz
    for VDD in [1.4, 2.2]V range.

    This is a rough model. The actual mapping will be extracted from SPICE.
    """
    return 0.5 + 1.5 * (vdd - 1.4) / 0.8


def coupling_strength_to_K(strength, coupling_type='capacitive'):
    """Map coupling strength [0,1] to effective Kuramoto K.

    For capacitive coupling, K depends on the capacitance ratio
    and the oscillation frequency. This is calibrated empirically.
    """
    # Rough calibration: at strength=0.85 (our standard), K ≈ 2.0 GHz
    # Scale roughly linearly with strength in the relevant range
    return 2.0 * (strength / 0.85)


def run_spice_comparison(n_pairs=20, coupling_strength=0.85, seed=42):
    """Compare SPICE and Kuramoto for multiple VDD pairs."""
    rng = np.random.default_rng(seed)
    VDD_MIN, VDD_MAX = 1.5, 2.1  # stay in the well-characterized range

    results = []

    for i in range(n_pairs):
        vdd_a = rng.uniform(VDD_MIN, VDD_MAX)
        vdd_b = rng.uniform(VDD_MIN, VDD_MAX)

        # SPICE simulation
        spice_result = run_pair(vdd_a, vdd_b, coupling_strength,
                                coupling_type='capacitive',
                                sim_time_ns=100.0, settle_ns=20.0)

        if spice_result is None:
            continue

        spice_phase_diff = spice_result['phase_diff']
        spice_locked = spice_result['locked']
        spice_freq_a = spice_result.get('freq_a_ghz', 0)
        spice_freq_b = spice_result.get('freq_b_ghz', 0)

        # Kuramoto prediction
        # Use SPICE-measured frequencies as the "true" natural frequencies
        # (This avoids the VDD→freq mapping uncertainty)
        if spice_freq_a > 0 and spice_freq_b > 0:
            omega_a = 2 * math.pi * spice_freq_a  # rad/ns
            omega_b = 2 * math.pi * spice_freq_b
            K_eff = coupling_strength_to_K(coupling_strength)
            K_rad = 2 * math.pi * K_eff  # convert to rad/ns

            kuramoto_pd, kuramoto_locked = kuramoto_phase_diff(omega_a, omega_b, K_rad)

            results.append({
                'vdd_a': vdd_a, 'vdd_b': vdd_b,
                'spice_phase_diff': spice_phase_diff,
                'spice_locked': spice_locked,
                'spice_freq_a': spice_freq_a,
                'spice_freq_b': spice_freq_b,
                'kuramoto_phase_diff': kuramoto_pd,
                'kuramoto_locked': kuramoto_locked,
                'freq_diff_ghz': abs(spice_freq_a - spice_freq_b),
            })

            status = 'LOCK' if spice_locked else 'BEAT'
            kpd = f"{kuramoto_pd:.3f}" if kuramoto_pd is not None else "N/A"
            print(f"  pair {i:2d}: VDD={vdd_a:.2f}/{vdd_b:.2f} "
                  f"SPICE={spice_phase_diff:.3f}rad {status} "
                  f"Kuramoto={kpd}rad", flush=True)

    if not results:
        print("No successful SPICE simulations.")
        return []

    # Analysis: correlation between SPICE and Kuramoto phase differences
    locked_results = [r for r in results if r['spice_locked'] and r['kuramoto_locked']]
    if len(locked_results) >= 5:
        spice_phases = np.array([r['spice_phase_diff'] for r in locked_results])
        kura_phases = np.array([r['kuramoto_phase_diff'] for r in locked_results])

        if np.std(spice_phases) > 1e-6 and np.std(kura_phases) > 1e-6:
            corr = float(np.corrcoef(spice_phases, kura_phases)[0, 1])
            rmse = float(np.sqrt(np.mean((spice_phases - kura_phases)**2)))
        else:
            corr = 0.0
            rmse = 0.0

        print(f"\n  Locked pairs: {len(locked_results)}/{len(results)}")
        print(f"  Phase correlation (SPICE vs Kuramoto): {corr:+.4f}")
        print(f"  Phase RMSE: {rmse:.4f} rad")

    # Lock/beat agreement
    lock_agree = sum(1 for r in results
                     if r['spice_locked'] == r['kuramoto_locked'])
    print(f"  Lock/beat agreement: {lock_agree}/{len(results)} "
          f"({lock_agree/len(results):.0%})")

    return results


if __name__ == '__main__':
    if not SPICE_AVAILABLE:
        print("SPICE tools not available. Exiting.")
        sys.exit(1)

    print("SPICE vs Kuramoto Comparison")
    print(f"  Using capacitive coupling at strength=0.85\n")

    results = run_spice_comparison(n_pairs=30, coupling_strength=0.85)

    out = os.path.join(SUBSTRATE_DIR, 'spice_vs_kuramoto.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")
