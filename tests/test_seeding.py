"""Tests for spectral seeding initialization."""

import numpy as np
import pytest

from phasegrad.kuramoto import make_network
from phasegrad.seeding import spectral_seed, graph_laplacian_reduced


class TestSpectralSeed:
    """Verify spectral seeding properties."""

    def test_basic_seeding(self):
        """Spectral seeding should set all learnable frequencies."""
        net = make_network(n_input=2, n_hidden=5, n_output=2, seed=42)
        spectral_seed(net, scale=0.3)

        # Input frequencies should be zero (overwritten per-sample)
        for i in net.input_ids:
            assert net.omega[i] == 0.0

        # Learnable frequencies should be nonzero
        learnable_omegas = [net.omega[i] for i in net.learnable_ids]
        assert any(abs(w) > 0.01 for w in learnable_omegas), (
            "Spectral seeding should produce nonzero frequencies")

    def test_scale_respected(self):
        """Maximum absolute frequency should match the requested scale."""
        net = make_network(n_input=2, n_hidden=5, n_output=2, seed=42)
        spectral_seed(net, scale=0.5)

        max_abs = max(abs(net.omega[i]) for i in net.learnable_ids)
        assert abs(max_abs - 0.5) < 1e-10, (
            f"Max |ω| should be 0.5, got {max_abs}")

    def test_output_separation(self):
        """Spectral seeding should give output nodes opposite-sign frequencies."""
        net = make_network(n_input=2, n_hidden=5, n_output=2, seed=42)
        spectral_seed(net)

        out_omegas = [net.omega[o] for o in net.output_ids]
        # Output nodes should have different signs (symmetry broken)
        assert out_omegas[0] * out_omegas[1] < 0, (
            f"Output frequencies should have opposite signs: {out_omegas}")

    def test_requires_two_outputs(self):
        """Spectral seeding should raise for non-binary classification."""
        net = make_network(n_input=2, n_hidden=5, n_output=3, seed=42)
        with pytest.raises(ValueError, match="exactly 2 output nodes"):
            spectral_seed(net)

    def test_laplacian_reduced_shape(self):
        """Reduced Laplacian should be (N-1) × (N-1)."""
        net = make_network(n_input=2, n_hidden=5, n_output=2, seed=42)
        L = graph_laplacian_reduced(net)
        assert L.shape == (net.N - 1, net.N - 1)

    def test_laplacian_reduced_symmetric(self):
        """Reduced Laplacian should be symmetric for symmetric K."""
        net = make_network(n_input=2, n_hidden=5, n_output=2, seed=42)
        L = graph_laplacian_reduced(net)
        assert np.allclose(L, L.T), "Reduced Laplacian should be symmetric"

    def test_seeding_deterministic(self):
        """Same network topology should produce same seeding."""
        net1 = make_network(n_input=2, n_hidden=5, n_output=2, seed=42)
        net2 = make_network(n_input=2, n_hidden=5, n_output=2, seed=42)
        spectral_seed(net1)
        spectral_seed(net2)
        np.testing.assert_array_equal(net1.omega, net2.omega)

    def test_different_seeds_different_seeding(self):
        """Different network topologies should produce different seedings."""
        net1 = make_network(n_input=2, n_hidden=5, n_output=2, seed=42)
        net2 = make_network(n_input=2, n_hidden=5, n_output=2, seed=99)
        spectral_seed(net1)
        spectral_seed(net2)
        assert not np.allclose(net1.omega, net2.omega), (
            "Different topologies should give different spectral seeds")

    def test_rejects_pinned_output_node(self):
        """Spectral seeding should reject output node 0 (pinned)."""
        from phasegrad.kuramoto import KuramotoNetwork
        net = KuramotoNetwork(
            omega=np.zeros(5),
            K=np.ones((5, 5)),
            output_ids=[0, 4],  # node 0 is pinned
        )
        with pytest.raises(ValueError, match="pinned node"):
            spectral_seed(net)

    def test_rejects_disconnected_graph(self):
        """Spectral seeding should reject disconnected coupling graphs."""
        from phasegrad.kuramoto import KuramotoNetwork
        # Two disconnected components: {0,1,2} and {3,4}
        K = np.zeros((5, 5))
        K[0, 1] = K[1, 0] = 1.0
        K[1, 2] = K[2, 1] = 1.0
        K[3, 4] = K[4, 3] = 1.0
        net = KuramotoNetwork(
            omega=np.zeros(5),
            K=K,
            output_ids=[2, 4],
        )
        with pytest.raises(ValueError, match="disconnected"):
            spectral_seed(net)
