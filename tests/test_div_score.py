import numpy as np
import pytest
from lotf.core import div_score

# Optional FAISS import with graceful fallback (same pattern as core.py)
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None


class TestDivScore:
    def test_div_score_basic(self):
        """Test basic functionality of div_score with known inputs."""
        # Simple 2D dataset
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

        # Query result: distances and ids for nearest neighbors
        dists = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        ids = np.array([0, 1, 2], dtype=np.int64)
        lam = 0.5  # Equal weight for similarity and diversity

        total_score, sim_term, div_term = div_score(dists, ids, X, lam)

        # Check that all return values are finite numbers
        assert np.isfinite(total_score)
        assert np.isfinite(sim_term)
        assert np.isfinite(div_term)

        # Similarity term should be average of distances
        expected_sim = np.mean(dists)
        assert np.isclose(sim_term, expected_sim)

        # Total score should combine similarity and diversity terms
        expected_total = (1.0 - lam) * sim_term + lam * div_term
        assert np.isclose(total_score, expected_total)

    def test_div_score_lambda_extremes(self):
        """Test div_score behavior at lambda extremes (0 and 1)."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)

        dists = np.array([1.0, 2.0], dtype=np.float32)
        ids = np.array([0, 1], dtype=np.int64)

        # Lambda = 0: pure similarity (no diversity)
        total_score_0, sim_term_0, div_term_0 = div_score(dists, ids, X, lam=0.0)
        assert np.isclose(total_score_0, sim_term_0)

        # Lambda = 1: pure diversity (no similarity)
        total_score_1, sim_term_1, div_term_1 = div_score(dists, ids, X, lam=1.0)
        assert np.isclose(total_score_1, div_term_1)

    def test_div_score_identical_points(self):
        """Test div_score when selected points are identical (zero diversity)."""
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],  # Identical to first point
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )

        dists = np.array([0.5, 0.6], dtype=np.float32)
        ids = np.array([0, 1], dtype=np.int64)  # Select two identical points
        lam = 0.5

        total_score, sim_term, div_term = div_score(dists, ids, X, lam)

        # Diversity term should be 0 (negative of minimum distance = 0)
        assert np.isclose(div_term, 0.0, atol=1e-6)

    def test_div_score_single_point(self):
        """Test div_score with only one neighbor (edge case)."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        dists = np.array([1.5], dtype=np.float32)
        ids = np.array([0], dtype=np.int64)
        lam = 0.3

        total_score, sim_term, div_term = div_score(dists, ids, X, lam)

        # With only one point, no pairwise distances exist
        # The diversity term should be 0.0
        assert np.isfinite(total_score)
        assert np.isclose(sim_term, 1.5)  # Average of single distance
        assert np.isclose(div_term, 0.0)  # No diversity information available

        # Total score should be (1-lam) * sim_term + lam * 0
        expected_total = (1.0 - lam) * sim_term + lam * 0.0
        assert np.isclose(total_score, expected_total)

    def test_div_score_input_conversion(self):
        """Test that div_score properly converts list/tuple inputs to numpy arrays."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        # Test with list inputs
        dists_list = [0.5, 1.0]
        ids_list = [0, 1]
        lam = 0.5

        total_score, sim_term, div_term = div_score(dists_list, ids_list, X, lam)

        # Should work without errors
        assert np.isfinite(total_score)
        assert np.isfinite(sim_term)
        assert np.isfinite(div_term)

        # Test with tuple inputs
        dists_tuple = (0.5, 1.0)
        ids_tuple = (0, 1)

        total_score_2, sim_term_2, div_term_2 = div_score(
            dists_tuple, ids_tuple, X, lam
        )

        # Results should be identical
        assert np.isclose(total_score, total_score_2)
        assert np.isclose(sim_term, sim_term_2)
        assert np.isclose(div_term, div_term_2)

    def test_div_score_with_invalid_distances(self):
        """Test div_score behavior when distances contain invalid values (NaN, inf)."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        # Distances with invalid values
        dists = np.array([0.5, np.inf, np.nan, 1.0], dtype=np.float32)
        ids = np.array([0, 1, 2, 0], dtype=np.int64)
        lam = 0.5

        total_score, sim_term, div_term = div_score(dists, ids, X, lam)

        # Should handle invalid values gracefully
        assert np.isfinite(total_score)
        assert np.isfinite(sim_term)
        assert np.isfinite(div_term)

    def test_div_score_dimension_mismatch_error(self):
        """Test that div_score raises assertion error for mismatched dimensions."""
        X = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        # Mismatched lengths
        dists = np.array([0.5, 1.0], dtype=np.float32)
        ids = np.array([0], dtype=np.int64)  # Different length
        lam = 0.5

        with pytest.raises(AssertionError):
            div_score(dists, ids, X, lam)

    def test_div_score_multidimensional_input_error(self):
        """Test that div_score raises assertion error for multi-dimensional input arrays."""
        X = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        # 2D input arrays (should be 1D)
        dists = np.array([[0.5, 1.0]], dtype=np.float32)  # 2D instead of 1D
        ids = np.array([0, 1], dtype=np.int64)
        lam = 0.5

        with pytest.raises(AssertionError):
            div_score(dists, ids, X, lam)

    def test_div_score_backends(self):
        """Test div_score with both FAISS and NumPy backends."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        dists = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        ids = np.array([0, 1, 2], dtype=np.int64)
        lam = 0.4

        # Test with FAISS backend (if available)
        faiss_results = None
        if HAS_FAISS:
            faiss_results = div_score(dists, ids, X, lam)
            assert all(np.isfinite(faiss_results))

        # Test with NumPy backend (force fallback)
        import lotf.core as core_module

        original_has_faiss = core_module.HAS_FAISS
        core_module.HAS_FAISS = False

        try:
            numpy_results = div_score(dists, ids, X, lam)
            assert all(np.isfinite(numpy_results))

            # If both backends available, results should be very close
            if HAS_FAISS and faiss_results is not None:
                np.testing.assert_allclose(faiss_results, numpy_results, rtol=1e-5)

        finally:
            core_module.HAS_FAISS = original_has_faiss
