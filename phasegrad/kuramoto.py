"""Kuramoto oscillator network simulation.

Implements the coupled oscillator dynamics:

    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)     (Eq. 1)

At equilibrium (dθ/dt = 0), the phases θ* encode the computation.
The equilibrium exists in the rotating frame where ω is mean-centered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import fsolve


def kuramoto_rhs(theta: np.ndarray, omega: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Right-hand side of the Kuramoto ODE (Eq. 1).

    Args:
        theta: (N,) phase vector.
        omega: (N,) natural frequencies (should be mean-centered).
        K: (N, N) symmetric coupling matrix.

    Returns:
        (N,) vector dθ/dt.
    """
    # sin(θ_j - θ_i) for all pairs: diff[i,j] = θ_j - θ_i
    diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    return omega + np.sum(K * np.sin(diff), axis=1)


def kuramoto_jacobian(theta: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Jacobian of the Kuramoto equilibrium equations (Eq. 5).

    J_ij = K_ij cos(θ_j - θ_i)           for i ≠ j
    J_ii = -Σ_j K_ij cos(θ_j - θ_i)

    This is a weighted graph Laplacian. Its inverse mediates gradient flow.
    """
    diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    cos_diff = np.cos(diff)
    J = K * cos_diff
    np.fill_diagonal(J, 0.0)
    np.fill_diagonal(J, -np.sum(J, axis=1))
    return J


@dataclass
class KuramotoNetwork:
    """A coupled Kuramoto oscillator network.

    Attributes:
        omega: (N,) natural frequencies. Input oscillators are set per-sample.
        K: (N, N) symmetric coupling matrix. K[i,j] > 0 means i and j are coupled.
        input_ids: indices of input oscillators (frequencies set by data).
        output_ids: indices of output oscillators (phases read for classification).
        input_scale: multiplier for input encoding into frequencies.
    """
    omega: np.ndarray
    K: np.ndarray
    input_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)
    input_scale: float = 1.5

    @property
    def N(self) -> int:
        return len(self.omega)

    @property
    def learnable_ids(self) -> list[int]:
        """Indices of oscillators with learnable frequencies."""
        inp = set(self.input_ids)
        return [i for i in range(self.N) if i not in inp]

    @property
    def edges(self) -> list[tuple[int, int]]:
        """List of (i, j) pairs with i < j where K[i,j] > 0."""
        pairs = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.K[i, j] > 0:
                    pairs.append((i, j))
        return pairs

    @property
    def output_mask(self) -> np.ndarray:
        """Binary mask: 1 at output nodes, 0 elsewhere."""
        m = np.zeros(self.N)
        for o in self.output_ids:
            m[o] = 1.0
        return m

    def set_input(self, x: np.ndarray) -> None:
        """Encode input vector x into input oscillator frequencies.

        Args:
            x: (n_inputs,) feature vector, expected in [-1, 1].
        """
        for k, idx in enumerate(self.input_ids):
            self.omega[idx] = x[k] * self.input_scale

    @property
    def omega_centered(self) -> np.ndarray:
        """Mean-centered frequencies for the rotating frame."""
        return self.omega - self.omega.mean()

    def equilibrium(self, theta_init: np.ndarray | None = None,
                    omega_c: np.ndarray | None = None,
                    ) -> tuple[np.ndarray, float]:
        """Find steady-state phases in the rotating frame.

        Subtracts mean(ω) to enter the rotating frame where a fixed point
        exists. Pins θ_0 = 0 to break the global phase symmetry. Solves
        the reduced (N-1)-dimensional system via Newton's method (fsolve).

        Args:
            theta_init: (N,) initial guess. None → zeros.
            omega_c: (N,) mean-centered frequencies. None → self.omega_centered.

        Returns:
            (theta_star, residual): equilibrium phases and max |RHS|.
        """
        N = self.N
        if omega_c is None:
            omega_c = self.omega_centered

        if theta_init is None:
            theta_init = np.zeros(N)

        def reduced(phi: np.ndarray) -> np.ndarray:
            theta = np.zeros(N)
            theta[1:] = phi
            return kuramoto_rhs(theta, omega_c, self.K)[1:]

        phi, info, ier, _ = fsolve(reduced, theta_init[1:], full_output=True)
        theta = np.zeros(N)
        theta[1:] = phi

        residual = float(np.max(np.abs(info['fvec']))) if 'fvec' in info else 1.0
        return theta, residual

    def clamped_equilibrium(self, beta: float, target: np.ndarray,
                            theta_init: np.ndarray | None = None,
                            omega_c: np.ndarray | None = None,
                            ) -> tuple[np.ndarray, float]:
        """Find equilibrium with output clamping.

        Adds β · mask · (target - θ) to the dynamics at output nodes,
        pulling them toward the target phases. This is the "nudged" phase
        of equilibrium propagation.

        Args:
            beta: clamping strength.
            target: (N,) target phases (only output entries matter).
            theta_init: initial guess.
            omega_c: mean-centered frequencies. None → self.omega_centered.

        Returns:
            (theta_clamped, residual).
        """
        N = self.N
        if omega_c is None:
            omega_c = self.omega_centered
        mask = self.output_mask

        if theta_init is None:
            theta_init = np.zeros(N)

        def reduced(phi: np.ndarray) -> np.ndarray:
            theta = np.zeros(N)
            theta[1:] = phi
            F = kuramoto_rhs(theta, omega_c, self.K)
            F += beta * mask * (target - theta)
            return F[1:]

        phi, info, ier, _ = fsolve(reduced, theta_init[1:], full_output=True)
        theta = np.zeros(N)
        theta[1:] = phi

        residual = float(np.max(np.abs(info['fvec']))) if 'fvec' in info else 1.0
        return theta, residual

    def classify(self, theta: np.ndarray) -> int:
        """Classify by which output oscillator has phase closest to 0.

        Returns the index (0-based within output_ids) of the winning class.
        """
        out_phases = np.array([theta[o] for o in self.output_ids])
        return int(np.argmax(np.cos(out_phases)))

    def clone(self) -> KuramotoNetwork:
        """Deep copy of the network."""
        return KuramotoNetwork(
            omega=self.omega.copy(),
            K=self.K.copy(),
            input_ids=list(self.input_ids),
            output_ids=list(self.output_ids),
            input_scale=self.input_scale,
        )


def make_network(n_input: int = 2, n_hidden: int = 5, n_output: int = 2,
                 K_scale: float = 2.0, input_scale: float = 1.5,
                 seed: int = 42) -> KuramotoNetwork:
    """Create a layered Kuramoto network.

    Architecture: input → hidden → output, with hidden↔hidden chain coupling.

    Args:
        n_input: number of input oscillators.
        n_hidden: number of hidden oscillators.
        n_output: number of output oscillators.
        K_scale: mean coupling strength (must exceed K_c ≈ 2.5 for sync).
        input_scale: multiplier for input frequency encoding.
        seed: random seed for reproducible initialization.
    """
    rng = np.random.default_rng(seed)
    N = n_input + n_hidden + n_output

    input_ids = list(range(n_input))
    hidden_ids = list(range(n_input, n_input + n_hidden))
    output_ids = list(range(n_input + n_hidden, N))

    omega = np.zeros(N)
    for i in hidden_ids + output_ids:
        omega[i] = rng.uniform(-0.3, 0.3)

    K = np.zeros((N, N))

    # Input → hidden (all-to-all)
    for i in input_ids:
        for h in hidden_ids:
            s = K_scale * rng.uniform(0.5, 1.5)
            K[i, h] = K[h, i] = s

    # Hidden → output (all-to-all)
    for h in hidden_ids:
        for o in output_ids:
            s = K_scale * rng.uniform(0.5, 1.5)
            K[h, o] = K[o, h] = s

    # Hidden ↔ hidden (chain)
    for k in range(len(hidden_ids) - 1):
        h1, h2 = hidden_ids[k], hidden_ids[k + 1]
        s = K_scale * rng.uniform(0.5, 1.0)
        K[h1, h2] = K[h2, h1] = s

    return KuramotoNetwork(
        omega=omega, K=K,
        input_ids=input_ids, output_ids=output_ids,
        input_scale=input_scale,
    )


def make_random_network(N: int, K_mean: float = 5.0, omega_spread: float = 0.3,
                        connectivity: float = 0.6, n_output: int = 2,
                        seed: int = 42) -> KuramotoNetwork:
    """Create a random (non-layered) network for theorem verification.

    Args:
        N: total oscillators.
        K_mean: mean coupling strength.
        omega_spread: standard deviation of natural frequencies.
        connectivity: probability of edge between any two nodes.
        n_output: number of output nodes (taken from end).
        seed: random seed.
    """
    rng = np.random.default_rng(seed)

    omega = omega_spread * rng.standard_normal(N)
    omega -= omega.mean()

    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < connectivity:
                s = K_mean * rng.uniform(0.5, 1.5)
                K[i, j] = K[j, i] = s

    output_ids = list(range(N - n_output, N))

    return KuramotoNetwork(
        omega=omega, K=K,
        input_ids=[], output_ids=output_ids,
    )
