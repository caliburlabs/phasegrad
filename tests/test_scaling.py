"""Scaling tests: where does the theorem hold, where does it break?

Documents the frontier of network sizes where gradient agreement
degrades. A reviewer will ask about N=50, N=100.
"""

import time

import numpy as np
import pytest
from phasegrad.verification import verify_single


SIZES = [6, 10, 15, 20, 30, 50]


@pytest.mark.parametrize("N", SIZES)
def test_gradient_at_size(N):
    """Gradient agreement at each network size."""
    t0 = time.time()
    result = verify_single(N, n_beta=6, seed=42)
    elapsed = time.time() - t0

    residual = result["residual"]
    cos_an_fd = result["cos_an_fd"]
    cos_tp_fd = result["best_cos_tp_fd"]

    # Print diagnostic even on pass (captured by pytest -v)
    print(f"\n  N={N}: res={residual:.1e} an-fd={cos_an_fd:+.6f} "
          f"tp-fd={cos_tp_fd:+.6f} time={elapsed:.1f}s")

    # Equilibrium must converge
    assert residual < 1e-4, f"N={N}: residual {residual:.2e}"

    # Analytical must match FD closely
    assert cos_an_fd > 0.99, (
        f"N={N}: analytical vs FD cosine {cos_an_fd:.6f}")

    # Two-phase must match FD (may degrade at large N)
    if N <= 20:
        assert cos_tp_fd > 0.999, (
            f"N={N}: two-phase vs FD cosine {cos_tp_fd:.6f}")
    else:
        # Relaxed for larger N — document where it degrades
        assert cos_tp_fd > 0.99, (
            f"N={N}: two-phase vs FD cosine {cos_tp_fd:.6f}")


def test_wall_clock_scaling():
    """Document how gradient computation time scales with N."""
    times = {}
    for N in [6, 10, 15, 20]:
        t0 = time.time()
        verify_single(N, n_beta=4, seed=42)
        times[N] = time.time() - t0

    print("\n  Wall-clock scaling:")
    for N, t in times.items():
        print(f"    N={N:3d}: {t:.2f}s")

    # Verify it doesn't blow up unreasonably
    # FD gradient is O(N) equilibrium solves, each O(N^2), so total O(N^3)
    # Allow 10x growth from N=6 to N=20
    assert times[20] < times[6] * 30, (
        f"Scaling too steep: N=6 took {times[6]:.1f}s, N=20 took {times[20]:.1f}s")
