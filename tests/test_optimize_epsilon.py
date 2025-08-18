import pytest
import numpy as np
from unittest.mock import Mock, patch

from lotf.core import optimize_epsilon

# Check if FAISS is available for integration tests
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class TestOptimizeEpsilon:
    """Test suite for the optimize_epsilon function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        N, D = 100, 10
        Nq = 20

        X = np.random.randn(N, D).astype(np.float32)
        Xq = np.random.randn(Nq, D).astype(np.float32)

        return X, Xq

    @pytest.fixture
    def mock_index(self, sample_data):
        """Create a mock Faiss index."""
        X, Xq = sample_data
        index = Mock()

        # Mock search results - return reasonable distances and indices
        candidate_k = 50
        dists = np.random.uniform(0.1, 2.0, (Xq.shape[0], candidate_k)).astype(
            np.float32
        )
        ids = np.random.randint(0, X.shape[0], (Xq.shape[0], candidate_k))

        index.search.return_value = (dists, ids)
        return index

    @pytest.fixture
    def mock_build_neighbor_lists(self):
        """Mock the build_neighbor_lists function."""
        with patch("lotf.core.build_neighbor_lists") as mock_func:
            # Return mock neighbor lists
            neighbor_lists = [np.array([0, 1, 2, 3, 4]) for _ in range(10)]
            neighbor_dists = [np.array([0.1, 0.2, 0.3, 0.4, 0.5]) for _ in range(10)]
            mock_func.return_value = (neighbor_lists, neighbor_dists)
            yield mock_func

    @pytest.fixture
    def mock_cutoff_table(self):
        """Mock the CutoffTable class."""
        with patch("lotf.core.CutoffTable") as mock_class:
            mock_instance = Mock()
            mock_instance.L = 25  # Mock cutoff table size

            # Mock filter method to return reasonable diversified results
            def mock_filter(dists, ids, final_k):
                return dists[:, :final_k], ids[:, :final_k]

            mock_instance.filter.side_effect = mock_filter
            mock_class.from_neighbor_lists.return_value = mock_instance
            yield mock_class

    @pytest.fixture
    def mock_div_score(self):
        """Mock the div_score function."""
        with patch("lotf.core.div_score") as mock_func:
            # Return mock scores: (combined, similarity, diversity)
            mock_func.return_value = (0.5, 0.3, 0.2)
            yield mock_func

    def test_basic_functionality(
        self,
        sample_data,
        mock_index,
        mock_build_neighbor_lists,
        mock_cutoff_table,
        mock_div_score,
    ):
        """Test basic functionality of optimize_epsilon."""
        X, Xq = sample_data

        result = optimize_epsilon(
            X=X,
            Xq=Xq,
            index=mock_index,
            candidate_k=50,
            final_k=10,
            lam=0.3,
            verbose=False,
            num_iter=2,  # Reduce iterations for faster testing
            coarse_divisions=3,
            fine_divisions=5,
        )

        # Check return value structure
        assert isinstance(result, dict)
        assert "epsilon" in result
        assert "div_score" in result
        assert "L" in result

        # Check that epsilon is reasonable
        assert 0.0 <= result["epsilon"] <= 10.0
        assert result["div_score"] >= 0.0
        assert result["L"] >= 0

    def test_epsilon_max_auto_computation(
        self,
        sample_data,
        mock_index,
        mock_build_neighbor_lists,
        mock_cutoff_table,
        mock_div_score,
    ):
        """Test automatic epsilon_max computation when not provided."""
        X, Xq = sample_data

        result = optimize_epsilon(
            X=X,
            Xq=Xq,
            index=mock_index,
            candidate_k=50,
            final_k=10,
            lam=0.3,
            epsilon_max=None,  # Should auto-compute
            verbose=False,
            num_iter=1,
            coarse_divisions=2,
        )

        # Should have computed epsilon_max and returned valid result
        assert "epsilon" in result
        assert result["epsilon"] >= 0.0  # Can be 0.0 in edge cases

    def test_epsilon_max_provided(
        self,
        sample_data,
        mock_index,
        mock_build_neighbor_lists,
        mock_cutoff_table,
        mock_div_score,
    ):
        """Test when epsilon_max is explicitly provided."""
        X, Xq = sample_data
        epsilon_max = 1.5

        result = optimize_epsilon(
            X=X,
            Xq=Xq,
            index=mock_index,
            candidate_k=50,
            final_k=10,
            lam=0.3,
            epsilon_max=epsilon_max,
            verbose=False,
            num_iter=1,
            coarse_divisions=2,
        )

        # Result epsilon should not exceed provided epsilon_max
        assert result["epsilon"] <= epsilon_max

    def test_verbose_output(
        self,
        sample_data,
        mock_index,
        mock_build_neighbor_lists,
        mock_cutoff_table,
        mock_div_score,
        capsys,
    ):
        """Test that verbose=True produces output."""
        X, Xq = sample_data

        optimize_epsilon(
            X=X,
            Xq=Xq,
            index=mock_index,
            candidate_k=50,
            final_k=10,
            lam=0.3,
            verbose=True,  # Should print progress
            num_iter=1,
            coarse_divisions=2,
        )

        captured = capsys.readouterr()
        assert "epsilon_max:" in captured.out
        assert "Epsilon optimization iteration" in captured.out

    def test_parameter_validation(
        self,
        sample_data,
        mock_index,
        mock_build_neighbor_lists,
        mock_cutoff_table,
        mock_div_score,
    ):
        """Test with different parameter combinations."""
        X, Xq = sample_data

        # Test with different lambda values
        for lam in [0.0, 0.5, 1.0]:
            result = optimize_epsilon(
                X=X,
                Xq=Xq,
                index=mock_index,
                candidate_k=50,
                final_k=10,
                lam=lam,
                verbose=False,
                num_iter=1,
                coarse_divisions=2,
            )
            assert "epsilon" in result

        # Test with different iteration counts
        for num_iter in [1, 3, 5]:
            result = optimize_epsilon(
                X=X,
                Xq=Xq,
                index=mock_index,
                candidate_k=50,
                final_k=10,
                lam=0.3,
                verbose=False,
                num_iter=num_iter,
                coarse_divisions=2,
                fine_divisions=3,
            )
            assert "epsilon" in result

    def test_coarse_to_fine_progression(
        self,
        sample_data,
        mock_index,
        mock_build_neighbor_lists,
        mock_cutoff_table,
        mock_div_score,
    ):
        """Test that the algorithm progresses from coarse to fine divisions."""
        X, Xq = sample_data

        # Reset call count
        mock_cutoff_table.from_neighbor_lists.reset_mock()

        # Track calls to verify coarse-to-fine behavior
        optimize_epsilon(
            X=X,
            Xq=Xq,
            index=mock_index,
            candidate_k=50,
            final_k=10,
            lam=0.3,
            verbose=False,
            num_iter=3,
            coarse_divisions=3,
            fine_divisions=7,
        )

        # Should have called CutoffTable.from_neighbor_lists multiple times
        # (2 coarse iterations × 3 divisions + 1 fine iteration × 7 divisions)
        expected_calls = 2 * 3 + 1 * 7
        assert mock_cutoff_table.from_neighbor_lists.call_count == expected_calls

    def test_edge_cases(
        self,
        sample_data,
        mock_index,
        mock_build_neighbor_lists,
        mock_cutoff_table,
        mock_div_score,
    ):
        """Test edge cases and boundary conditions including small datasets."""
        X, Xq = sample_data

        # Test with small dataset
        np.random.seed(42)
        X_small = np.random.randn(5, 3).astype(np.float32)
        Xq_small = np.random.randn(2, 3).astype(np.float32)
        result = optimize_epsilon(
            X=X_small,
            Xq=Xq_small,
            index=mock_index,
            candidate_k=3,
            final_k=2,
            lam=0.5,
            verbose=False,
            num_iter=1,
            coarse_divisions=2,
        )
        assert "epsilon" in result and "div_score" in result and "L" in result

        # Test lambda extremes and final_k variations
        for lam_val in [0.0, 1.0]:  # Pure similarity vs pure diversity
            result = optimize_epsilon(
                X=X,
                Xq=Xq,
                index=mock_index,
                candidate_k=50,
                final_k=10,
                lam=lam_val,
                verbose=False,
                num_iter=1,
                coarse_divisions=2,
            )
            assert result["epsilon"] >= 0.0

        # Test with final_k = 1
        result = optimize_epsilon(
            X=X,
            Xq=Xq,
            index=mock_index,
            candidate_k=50,
            final_k=1,
            lam=0.3,
            verbose=False,
            num_iter=1,
            coarse_divisions=2,
        )
        assert result["epsilon"] >= 0.0

    @pytest.mark.skipif(not HAS_FAISS, reason="FAISS not available")
    def test_regression_fixed_input_output(self):
        """Regression test: verify exact output for fixed input.

        This test uses real (not mocked) components and verifies that the
        optimize_epsilon function produces the exact same result for a fixed
        input. This serves as a regression test to catch any unintended changes
        in the optimization algorithm.
        """
        # Create small, fixed dataset (same as used to generate ground truth)

        # Fixed data arrays for deterministic testing
        X = np.array(
            [
                [-0.20470765, 0.47894335, -0.51943874, -0.55621296],
                [1.9657806, 1.3934058, 0.09290788, -0.84249002],
                [0.7690226, 1.2464347, 1.0071894, 0.31563495],
                [-2.0201528, -0.90729836, 0.7709464, 0.91606766],
                [1.6328945, -0.77888274, -0.05837129, 1.3072731],
                [-0.23359746, -1.2399721, -1.1275637, 0.39516553],
                [1.4587322, -0.46572975, 0.24196227, -1.9132802],
                [1.2629544, 0.33367434, 1.4940791, -0.20515826],
                [0.31306773, 0.04912588, 0.64137536, -0.40317695],
                [1.2286863, 1.4940791, -0.85506946, 0.96864003],
                [-0.21008466, -0.89546013, 0.3869025, -0.5107697],
                [0.05739296, 1.3893421, -1.2709924, -1.5457467],
                [0.65361965, -0.74216187, 2.2697546, -0.45435858],
                [-0.9623255, 0.19829972, 1.2167502, 1.4842205],
                [-0.23361993, 0.40165138, -0.16317122, -0.8779149],
                [0.9788203, -0.24937038, 1.2845794, -1.2845476],
                [0.0513636, 1.4542735, -0.19183928, -0.88762337],
                [0.19829972, 0.6347137, -0.8524994, 0.2824932],
                [-1.9465477, -0.40181404, -0.01516976, 0.6353431],
                [0.43441123, -0.59109324, 0.43804508, -0.48934233],
            ],
            dtype=np.float32,
        )

        Xq = np.array(
            [
                [0.0513161, -1.1577195, 0.81670696, 0.98906916],
                [1.010737, 1.8248752, -0.99751824, -0.13187276],
                [-0.1315776, 0.91241413, 0.18821068, 1.8291928],
                [-1.1577195, -1.6328945, 0.39516553, -0.84249002],
                [1.6328945, 0.65361965, -1.2709924, 0.31563495],
            ],
            dtype=np.float32,
        )

        # Create real FAISS index
        index = faiss.IndexFlatL2(4)  # 4-dimensional vectors
        index.add(X)

        # Run optimize_epsilon with exact same parameters as ground truth generation
        result = optimize_epsilon(
            X=X,
            Xq=Xq,
            index=index,
            candidate_k=8,
            final_k=3,
            lam=0.3,
            epsilon_max=None,  # Auto-compute
            verbose=False,  # Suppress output during test
            num_iter=2,
            batch_size=50,
            coarse_divisions=3,
            fine_divisions=5,
        )

        # Ground truth values (generated with fixed array data)
        expected_epsilon = 2.861268997192383
        expected_div_score = 0.8301931206385295
        expected_L = 3.0

        # Verify exact match (within floating point precision)
        assert abs(result["epsilon"] - expected_epsilon) < 1e-6, (
            f"Expected epsilon {expected_epsilon}, got {result['epsilon']}"
        )
        assert abs(result["div_score"] - expected_div_score) < 1e-6, (
            f"Expected div_score {expected_div_score}, got {result['div_score']}"
        )
        assert result["L"] == expected_L, f"Expected L {expected_L}, got {result['L']}"

        # Verify return value structure
        assert isinstance(result, dict)
        assert set(result.keys()) == {"epsilon", "div_score", "L"}

        # Verify reasonable value ranges
        assert result["epsilon"] >= 0.0
        assert result["div_score"] >= 0.0
        assert result["L"] >= 0
