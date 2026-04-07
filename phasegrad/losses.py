"""Loss functions for Kuramoto oscillator networks."""

from __future__ import annotations

import numpy as np


def mse_loss(theta: np.ndarray, target: np.ndarray,
             output_ids: list[int]) -> float:
    """Mean squared error on output phases.

    L = ½ Σ_{i ∈ output} (θ_i - target_i)²
    """
    return 0.5 * sum((theta[i] - target[i]) ** 2 for i in output_ids)


def mse_target(n_total: int, output_ids: list[int],
               class_idx: int, margin: float = 0.2) -> np.ndarray:
    """MSE target: correct class at -margin, wrong classes at +margin.

    The total separation between correct and wrong is 2*margin.
    This must be within the network's dynamic range:
    max achievable Δθ ≈ arcsin(input_range / K_coupling).

    Args:
        n_total: total number of oscillators.
        output_ids: indices of output oscillators.
        class_idx: which output corresponds to the correct class.
        margin: half the phase separation between correct and wrong.

    Returns:
        (N,) target vector (nonzero only at output nodes).
    """
    target = np.zeros(n_total)
    for i, o in enumerate(output_ids):
        target[o] = -margin if i == class_idx else margin
    return target
