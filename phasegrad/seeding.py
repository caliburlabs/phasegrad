"""Spectral seeding for Kuramoto oscillator networks.

Initializes natural frequencies from the spectral structure of the coupling
graph Laplacian. This breaks the topological symmetry between output nodes
in a way aligned with the coupling topology, eliminating the ~50% convergence
failure rate under random initialization.

The key formula (Eq. 10 in the paper):

    ω_k = α Σ_i (s_i / λ_i) [v_i]_k

where v_i, λ_i are eigenvectors/eigenvalues of the reduced graph Laplacian,
s_i = [v_i]_{out_1} - [v_i]_{out_2} is the signed output separation of mode i,
and α normalizes the maximum frequency to a target scale.

Each mode is weighted by:
  - s_i: how well it separates the output nodes (discrimination)
  - 1/λ_i: how strongly J⁻¹ amplifies it (inverse Laplacian amplification)
"""

from __future__ import annotations

import numpy as np

from phasegrad.kuramoto import KuramotoNetwork


def graph_laplacian_reduced(net: KuramotoNetwork) -> np.ndarray:
    """Reduced graph Laplacian of the coupling matrix K.

    L = D - K where D is the degree matrix. The reduced form removes the
    pinned node (index 0), yielding an (N-1) × (N-1) matrix.

    At θ = 0 (before training), J = -L, so the Laplacian structure
    determines the initial gradient propagation.

    Args:
        net: Kuramoto network with coupling matrix K.

    Returns:
        (N-1, N-1) reduced graph Laplacian.
    """
    K = net.K
    D = np.diag(K.sum(axis=1))
    L = D - K
    return L[1:, 1:]


def spectral_seed(net: KuramotoNetwork, scale: float = 0.3) -> None:
    """Initialize natural frequencies via spectral seeding (Eq. 10).

    Requires exactly 2 output nodes (binary classification).
    Sets ω for all learnable nodes; input node frequencies are zeroed
    (they will be overwritten per-sample during training).

    Args:
        net: Kuramoto network (modified in place).
        scale: maximum absolute frequency after initialization.

    Raises:
        ValueError: if the network does not have exactly 2 output nodes.
    """
    if len(net.output_ids) != 2:
        raise ValueError(
            f"Spectral seeding requires exactly 2 output nodes, "
            f"got {len(net.output_ids)}"
        )

    # Output node 0 is pinned (θ_0 = 0) — it cannot be an output node
    # because the reduced system removes it, and out_red = out - 1 = -1
    if 0 in net.output_ids:
        raise ValueError(
            "Output node 0 is the pinned node and cannot be used with "
            "spectral seeding. Re-index the network so output nodes have "
            "indices >= 1."
        )

    N = net.N
    L_red = graph_laplacian_reduced(net)
    evals, evecs = np.linalg.eigh(L_red)

    # Output node indices in the reduced system (shifted by -1 for pinned node)
    out_red = [o - 1 for o in net.output_ids]

    # Guard against near-zero eigenvalues (disconnected or nearly disconnected graph).
    # The smallest eigenvalue of a connected graph Laplacian is > 0 (Fiedler value).
    # If any eigenvalue is near zero, the graph is disconnected and spectral seeding
    # would produce infinite weights.
    min_eval = np.min(np.abs(evals))
    if min_eval < 1e-10:
        raise ValueError(
            f"Coupling graph appears disconnected (smallest Laplacian eigenvalue "
            f"= {min_eval:.2e}). Spectral seeding requires a connected graph."
        )

    # Weight = (output_separation × inverse_eigenvalue) per mode
    combo = np.zeros(N - 1)
    for i in range(len(evals)):
        sep = evecs[out_red[0], i] - evecs[out_red[1], i]  # signed
        weight = sep / evals[i]  # 1/λ amplification
        combo += weight * evecs[:, i]

    # Reconstruct full ω vector
    new_omega = np.zeros(N)
    new_omega[1:] = combo
    for inp in net.input_ids:
        new_omega[inp] = 0.0

    # Normalize to target scale
    max_abs = np.max(np.abs(new_omega))
    if max_abs > 1e-10:
        new_omega = new_omega / max_abs * scale

    net.omega = new_omega
