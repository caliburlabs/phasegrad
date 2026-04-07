"""Forced Kuramoto oscillator bank: the oscillator IS the filter.

Extends the Kuramoto model with an external driving signal:

    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i) + F_i sin(Ψ - θ_i)    (Eq. F1)

where Ψ is the instantaneous phase of the input signal and F_i is the
forcing strength on oscillator i.

When the input frequency is near ω_i, oscillator i injection-locks:
its phase tracks the input. When far, it beats. The lock/beat pattern
across the bank encodes the frequency content — this IS spectral
decomposition, done by physics.

Two operating modes:
1. Transient: integrate the ODE, extract steady-state features
2. Quasi-static: for slowly varying input, treat as equilibrium at
   each time step (compatible with EP gradient computation)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def forced_kuramoto_rhs(theta: np.ndarray, omega: np.ndarray,
                        K: np.ndarray, F: np.ndarray,
                        input_phase: float) -> np.ndarray:
    """RHS of forced Kuramoto ODE (Eq. F1).

    Args:
        theta: (N,) oscillator phases.
        omega: (N,) natural frequencies (mean-centered).
        K: (N, N) coupling matrix.
        F: (N,) forcing strength per oscillator (0 = unforced).
        input_phase: scalar phase of the external driving signal.
    """
    diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    coupling = np.sum(K * np.sin(diff), axis=1)
    forcing = F * np.sin(input_phase - theta)
    return omega + coupling + forcing


class OscillatorBank:
    """A bank of oscillators driven by an external signal.

    Architecture:
        - Sensor oscillators: directly driven by input (F > 0)
        - Hidden oscillators: coupled to sensors, not directly driven
        - Output oscillators: readout nodes for classification

    The bank performs spectral decomposition by injection locking:
    sensors near the input frequency lock, others beat.
    """

    def __init__(self, n_sensors: int, n_hidden: int, n_output: int,
                 freq_range: tuple[float, float] = (0.5, 5.0),
                 K_scale: float = 0.5, F_strength: float = 2.0,
                 seed: int = 42):
        """
        Args:
            n_sensors: oscillators directly driven by input signal.
            n_hidden: intermediate processing oscillators.
            n_output: readout oscillators (one per class).
            freq_range: (min, max) natural frequency range for sensors.
            K_scale: coupling strength.
            F_strength: forcing amplitude on sensor oscillators.
            seed: random seed.
        """
        rng = np.random.default_rng(seed)
        self.n_sensors = n_sensors
        self.n_hidden = n_hidden
        self.n_output = n_output
        N = n_sensors + n_hidden + n_output
        self.N = N

        self.sensor_ids = list(range(n_sensors))
        self.hidden_ids = list(range(n_sensors, n_sensors + n_hidden))
        self.output_ids = list(range(n_sensors + n_hidden, N))
        self.learnable_ids = self.hidden_ids + self.output_ids

        # Natural frequencies: sensors span freq_range, others random
        self.omega = np.zeros(N)
        for i, sid in enumerate(self.sensor_ids):
            self.omega[sid] = freq_range[0] + (freq_range[1] - freq_range[0]) * i / max(n_sensors - 1, 1)
        for i in self.learnable_ids:
            self.omega[i] = rng.uniform(*freq_range) * 0.5

        # Forcing: only sensors are driven
        self.F = np.zeros(N)
        for sid in self.sensor_ids:
            self.F[sid] = F_strength

        # Coupling: sensor→hidden, hidden→output, hidden↔hidden
        self.K = np.zeros((N, N))
        for s in self.sensor_ids:
            for h in self.hidden_ids:
                v = K_scale * rng.uniform(0.5, 1.5)
                self.K[s, h] = self.K[h, s] = v
        for h in self.hidden_ids:
            for o in self.output_ids:
                v = K_scale * rng.uniform(0.5, 1.5)
                self.K[h, o] = self.K[o, h] = v
        for i, h1 in enumerate(self.hidden_ids):
            for h2 in self.hidden_ids[i+1:]:
                v = K_scale * rng.uniform(0.3, 0.8)
                self.K[h1, h2] = self.K[h2, h1] = v

    def simulate_transient(self, input_freq: float, duration: float = 20.0,
                           settle: float = 10.0,
                           dt: float = 0.01) -> dict:
        """Run transient simulation with a sinusoidal input at input_freq.

        The input signal has phase Ψ(t) = 2π·input_freq·t.
        After settling, extract features from the oscillator bank.

        Returns dict with per-oscillator features:
            - amplitude: oscillation amplitude (locked ≈ constant, beating ≈ varying)
            - phase_coherence: how stable the phase is relative to input
            - freq_ratio: measured frequency / input frequency
        """
        N = self.N
        omega_c = self.omega - self.omega.mean()
        theta0 = np.zeros(N)

        def rhs(t, theta):
            input_phase = 2 * np.pi * input_freq * t
            return forced_kuramoto_rhs(theta, omega_c, self.K, self.F, input_phase)

        # Integrate
        t_span = (0, duration)
        t_eval = np.arange(settle, duration, dt)

        sol = solve_ivp(rhs, t_span, theta0, method='RK45',
                        t_eval=t_eval, max_step=dt*2,
                        rtol=1e-6, atol=1e-8)

        if not sol.success or sol.y.shape[1] < 10:
            return None

        # Extract features from the settled portion
        phases = sol.y  # (N, n_timepoints)
        t = sol.t

        # Input phase at each timepoint
        input_phases = 2 * np.pi * input_freq * t

        features = {}
        for i in range(N):
            # Phase difference from input
            phase_diff = phases[i] - input_phases
            # Phase coherence: |mean(exp(j·Δφ))| — 1 if locked, ~0 if beating
            coherence = float(np.abs(np.mean(np.exp(1j * phase_diff))))

            # Instantaneous frequency (from phase derivative)
            dphase = np.diff(phases[i]) / np.diff(t)
            inst_freq = float(np.mean(dphase)) / (2 * np.pi)
            freq_ratio = inst_freq / input_freq if input_freq > 0 else 0

            features[i] = {
                'coherence': coherence,
                'freq_ratio': freq_ratio,
                'locked': coherence > 0.7,
            }

        return features

    def extract_feature_vector(self, input_freq: float, **kwargs) -> np.ndarray | None:
        """Run transient sim and return a feature vector for classification.

        Feature vector: [coherence_0, coherence_1, ..., coherence_{N-1}]
        normalized to [-1, 1].
        """
        features = self.simulate_transient(input_freq, **kwargs)
        if features is None:
            return None

        # Use coherence of all oscillators as features
        vec = np.array([features[i]['coherence'] for i in range(self.N)])
        # Normalize to [-1, 1]
        vec = 2 * vec - 1
        return vec.astype(np.float32)

    def _omega_in_input_frame(self, input_freq: float) -> np.ndarray:
        """Compute detuning frequencies in the input's rotating frame.

        Each oscillator's detuning = ω_i - 2π·f_input.
        NO centering — the forcing term F·sin(Ψ - θ) provides the
        restoring force that allows a fixed point. Mean-centering
        would cancel the input frequency (uniform shift).
        """
        return self.omega - 2 * np.pi * input_freq

    def forced_equilibrium(self, input_freq: float,
                           theta_init: np.ndarray | None = None,
                           ) -> tuple[np.ndarray, float]:
        """Find steady-state of the forced system (quasi-static).

        In the rotating frame of the input signal, each oscillator
        has detuning ω_i - 2π·f_input. Sensors near the input frequency
        have small detuning and lock; distant ones beat.

        The detuning pattern is NOT re-centered — it carries the
        spectral information.
        """
        N = self.N
        omega_rot = self._omega_in_input_frame(input_freq)

        if theta_init is None:
            theta_init = np.zeros(N)

        input_phase = 0.0  # reference frame

        def reduced(phi):
            theta = np.zeros(N)
            theta[1:] = phi
            return forced_kuramoto_rhs(theta, omega_rot, self.K,
                                        self.F, input_phase)[1:]

        phi, info, ier, _ = fsolve(reduced, theta_init[1:], full_output=True)
        theta = np.zeros(N)
        theta[1:] = phi
        residual = float(np.max(np.abs(info['fvec']))) if 'fvec' in info else 1.0
        return theta, residual

    def forced_clamped_equilibrium(self, input_freq: float, beta: float,
                                    target: np.ndarray,
                                    theta_init: np.ndarray | None = None,
                                    ) -> tuple[np.ndarray, float]:
        """Clamped equilibrium of the forced system (for EP)."""
        N = self.N
        omega_rot = self._omega_in_input_frame(input_freq)

        if theta_init is None:
            theta_init = np.zeros(N)

        input_phase = 0.0
        output_mask = np.zeros(N)
        for o in self.output_ids:
            output_mask[o] = 1.0

        def reduced(phi):
            theta = np.zeros(N)
            theta[1:] = phi
            F_val = forced_kuramoto_rhs(theta, omega_rot, self.K,
                                         self.F, input_phase)
            F_val += beta * output_mask * (target - theta)
            return F_val[1:]

        phi, info, ier, _ = fsolve(reduced, theta_init[1:], full_output=True)
        theta = np.zeros(N)
        theta[1:] = phi
        residual = float(np.max(np.abs(info['fvec']))) if 'fvec' in info else 1.0
        return theta, residual
