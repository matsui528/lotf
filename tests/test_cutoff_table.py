import pytest
import numpy as np
import lotf

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None

# Skip all tests in this module if faiss is not available
pytestmark = pytest.mark.skipif(not HAS_FAISS, reason="faiss not available")


class TestCutoffTable:
    """Test suite for CutoffTable class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        N, D = 50, 3
        X = np.random.randn(N, D).astype(np.float32)

        # Create FAISS index
        index = faiss.IndexFlatL2(D)
        index.add(X)

        return X, index

    @pytest.fixture
    def small_data(self):
        """Create small test dataset."""
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        index = faiss.IndexFlatL2(3)
        index.add(X)

        return X, index

    def test_init_basic(self, sample_data):
        """Test basic CutoffTable initialization."""
        X, index = sample_data
        epsilon = 1.0

        ctable = lotf.CutoffTable(X=X, index=index, epsilon=epsilon, verbose=False)

        assert ctable.epsilon == epsilon
        assert ctable.N == X.shape[0]
        assert ctable.D == X.shape[1]
        assert ctable.L >= 0
        assert isinstance(ctable.flattened_neighbors, np.ndarray)
        assert isinstance(ctable.neighbor_offsets, np.ndarray)
        assert len(ctable.neighbor_offsets) == X.shape[0] + 1

    def test_init_validation(self, sample_data):
        """Test input validation in CutoffTable.__init__."""
        X, index = sample_data

        # Invalid epsilon
        with pytest.raises(AssertionError, match="epsilon must be a positive number"):
            lotf.CutoffTable(X=X, index=index, epsilon=0, verbose=False)

        with pytest.raises(AssertionError, match="epsilon must be a positive number"):
            lotf.CutoffTable(X=X, index=index, epsilon=-1.0, verbose=False)

        # Invalid batch_size
        with pytest.raises(
            AssertionError, match="batch_size must be a positive integer"
        ):
            lotf.CutoffTable(X=X, index=index, epsilon=1.0, batch_size=0, verbose=False)

        # Invalid X dimension
        X_1d = np.array([1, 2, 3], dtype=np.float32)
        with pytest.raises(AssertionError, match="X must be a 2D array"):
            lotf.CutoffTable(X=X_1d, index=index, epsilon=1.0, verbose=False)

        # Empty dataset
        X_empty = np.empty((0, 3), dtype=np.float32)
        index_empty = faiss.IndexFlatL2(3)
        with pytest.raises(AssertionError, match="Dataset cannot be empty"):
            lotf.CutoffTable(X=X_empty, index=index_empty, epsilon=1.0, verbose=False)

        # Mismatched dataset and index size
        X_small = X[:10]
        with pytest.raises(AssertionError, match="Dataset size mismatch"):
            lotf.CutoffTable(X=X_small, index=index, epsilon=1.0, verbose=False)

    def test_from_neighbor_lists(self):
        """Test CutoffTable.from_neighbor_lists class method."""
        neighbor_lists = [
            np.array([1, 2]),
            np.array([0, 3, 4]),
            np.array([0, 1]),
            np.array([1, 4]),
            np.array([3]),
        ]
        epsilon = 0.5
        N, D = 5, 3

        ctable = lotf.CutoffTable.from_neighbor_lists(
            neighbor_lists=neighbor_lists, epsilon=epsilon, N=N, D=D
        )

        assert ctable.epsilon == epsilon
        assert ctable.N == N
        assert ctable.D == D
        assert ctable.L == np.mean([len(ids) for ids in neighbor_lists])
        assert len(ctable.neighbor_offsets) == N + 1

    def test_filter_basic(self, small_data):
        """Test basic filtering functionality."""
        X, index = small_data
        epsilon = 1.5

        ctable = lotf.CutoffTable(X=X, index=index, epsilon=epsilon, verbose=False)

        # Create mock search results
        final_k = 2
        Nq = 2

        dists = np.array([[0.1, 0.5, 1.0, 2.0], [0.2, 0.8, 1.5, 3.0]], dtype=np.float32)

        ids = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)

        filtered_dists, filtered_ids = ctable.filter(
            dists=dists, ids=ids, final_k=final_k
        )

        # Expected values based on the filtering algorithm
        expected_ids = np.array([[0, 3], [1, 2]], dtype=np.int64)

        expected_dists = np.array([[0.1, 2.0], [0.2, 0.8]], dtype=np.float32)

        assert filtered_dists.shape == (Nq, final_k)
        assert filtered_ids.shape == (Nq, final_k)
        assert filtered_dists.dtype == np.float32
        assert filtered_ids.dtype == np.int64

        # Verify exact values match expected output
        np.testing.assert_array_equal(filtered_ids, expected_ids)
        np.testing.assert_allclose(filtered_dists, expected_dists, rtol=1e-5, atol=1e-5)

    def test_filter_validation(self, small_data):
        """Test input validation in CutoffTable.filter."""
        X, index = small_data
        ctable = lotf.CutoffTable(X=X, index=index, epsilon=1.0, verbose=False)

        dists = np.array([[0.1, 0.5], [0.2, 0.8]], dtype=np.float32)
        ids = np.array([[0, 1], [1, 2]], dtype=np.int64)

        # Invalid final_k
        with pytest.raises(AssertionError, match="final_k must be a positive integer"):
            ctable.filter(dists=dists, ids=ids, final_k=0)

        # Invalid array dimensions
        dists_1d = np.array([0.1, 0.5], dtype=np.float32)
        with pytest.raises(AssertionError, match="dists and ids must be 2D arrays"):
            ctable.filter(dists=dists_1d, ids=ids, final_k=1)

        # Shape mismatch
        ids_wrong = np.array([[0], [1]], dtype=np.int64)
        with pytest.raises(AssertionError, match="Shape mismatch"):
            ctable.filter(dists=dists, ids=ids_wrong, final_k=1)

        # final_k too large
        with pytest.raises(AssertionError, match="Cannot select .* results from only"):
            ctable.filter(dists=dists, ids=ids, final_k=5)

        # Empty query results
        dists_empty = np.empty((0, 2), dtype=np.float32)
        ids_empty = np.empty((0, 2), dtype=np.int64)
        with pytest.raises(AssertionError, match="Cannot filter empty query results"):
            ctable.filter(dists=dists_empty, ids=ids_empty, final_k=1)

    def test_filter_diversity(self, small_data):
        """Test that filtering actually increases diversity."""
        X, index = small_data
        epsilon = 0.8  # Small epsilon to create tight neighborhoods

        ctable = lotf.CutoffTable(X=X, index=index, epsilon=epsilon, verbose=False)

        # Query point close to [0,0,0]
        query = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        candidate_k = 4
        final_k = 2

        # Get candidate results
        candidate_dists, candidate_ids = index.search(query, candidate_k)

        # Apply diversity filtering
        dists_filtered, ids_filtered = ctable.filter(
            dists=candidate_dists, ids=candidate_ids, final_k=final_k
        )

        # Results should be properly shaped
        assert dists_filtered.shape == (1, final_k)
        assert ids_filtered.shape == (1, final_k)

        # All returned IDs should be valid
        assert np.all(ids_filtered >= 0)
        assert np.all(ids_filtered < X.shape[0])

    def test_different_epsilons(self, sample_data):
        """Test behavior with different epsilon values."""
        X, index = sample_data

        # Small epsilon - fewer neighbors
        ctable_small = lotf.CutoffTable(X=X, index=index, epsilon=0.5, verbose=False)

        # Large epsilon - more neighbors
        ctable_large = lotf.CutoffTable(X=X, index=index, epsilon=5.0, verbose=False)

        # Large epsilon should result in higher average neighbor count
        assert ctable_large.L >= ctable_small.L

    def test_batch_processing(self, sample_data):
        """Test that batch_size parameter works correctly."""
        X, index = sample_data
        epsilon = 1.0

        # Test with different batch sizes
        ctable_batch_10 = lotf.CutoffTable(
            X=X, index=index, epsilon=epsilon, batch_size=10, verbose=False
        )
        ctable_batch_100 = lotf.CutoffTable(
            X=X, index=index, epsilon=epsilon, batch_size=100, verbose=False
        )

        # Results should be identical regardless of batch size
        assert ctable_batch_10.L == ctable_batch_100.L
        np.testing.assert_array_equal(
            ctable_batch_10.flattened_neighbors, ctable_batch_100.flattened_neighbors
        )
        np.testing.assert_array_equal(
            ctable_batch_10.neighbor_offsets, ctable_batch_100.neighbor_offsets
        )

    def test_large_dataset(self):
        """Test with larger dataset to ensure scalability."""
        np.random.seed(42)
        N, D = 1000, 10
        X = np.random.randn(N, D).astype(np.float32)

        index = faiss.IndexFlatL2(D)
        index.add(X)

        ctable = lotf.CutoffTable(
            X=X, index=index, epsilon=2.0, batch_size=100, verbose=False
        )

        assert ctable.N == N
        assert ctable.D == D
        assert ctable.L > 0

        # Test filtering on subset of data
        query = X[:5]  # First 5 vectors as queries
        candidate_k = 50
        final_k = 10

        candidate_dists, candidate_ids = index.search(query, candidate_k)
        dists_filtered, ids_filtered = ctable.filter(
            dists=candidate_dists, ids=candidate_ids, final_k=final_k
        )

        # Expected values based on deterministic seed=42
        expected_ids = np.array(
            [
                [0, 174, 91, 601, 262, 649, 699, 111, 469, 933],
                [1, 574, 922, 585, 385, 316, 692, 124, 932, 852],
                [2, 690, 609, 475, 455, 157, 425, 147, 623, 79],
                [3, 415, 923, 842, 409, 708, 209, 211, 772, 112],
                [4, 852, 189, 505, 229, 513, 797, 205, 243, 207],
            ]
        )

        expected_dists = np.array(
            [
                [0.0, 2.61, 2.62, 2.91, 3.12, 4.07, 4.24, 4.28, 4.35, 4.86],
                [0.0, 3.43, 5.83, 6.29, 6.44, 6.65, 6.66, 6.75, 7.03, 7.19],
                [0.0, 2.42, 2.71, 3.26, 3.31, 3.73, 3.75, 3.85, 4.26, 4.33],
                [0.0, 4.73, 5.15, 5.68, 5.76, 6.44, 6.48, 7.37, 7.49, 7.53],
                [0.0, 4.70, 4.79, 4.94, 4.97, 5.26, 5.27, 5.28, 5.29, 5.52],
            ],
            dtype=np.float32,
        )

        assert dists_filtered.shape == (5, final_k)
        assert ids_filtered.shape == (5, final_k)

        # Verify exact values match expected output
        np.testing.assert_array_equal(ids_filtered, expected_ids)
        np.testing.assert_allclose(dists_filtered, expected_dists, rtol=1e-2, atol=1e-2)

    def test_eq_method(self, sample_data):
        """Test __eq__ method for CutoffTable equality comparison."""
        import pickle

        X, index = sample_data
        epsilon = 1.0

        # Create two identical tables
        ctable1 = lotf.CutoffTable(X=X, index=index, epsilon=epsilon, verbose=False)
        ctable2 = lotf.CutoffTable(X=X, index=index, epsilon=epsilon, verbose=False)

        # Test basic equality
        assert ctable1 == ctable2
        assert ctable1 == ctable1

        # Test with different epsilon
        ctable3 = lotf.CutoffTable(X=X, index=index, epsilon=2.0, verbose=False)
        assert ctable1 != ctable3

        # Test with different data size
        X_small = X[:25]
        index_small = faiss.IndexFlatL2(X.shape[1])
        index_small.add(X_small)
        ctable4 = lotf.CutoffTable(
            X=X_small, index=index_small, epsilon=epsilon, verbose=False
        )
        assert ctable1 != ctable4

        # Test comparison with non-CutoffTable objects
        assert ctable1 != "not a cutoff table"
        assert ctable1 != 42

        # Test floating point precision with epsilon
        epsilon_close = 1.0 + 1e-15  # Very small difference
        ctable_close = lotf.CutoffTable(
            X=X, index=index, epsilon=epsilon_close, verbose=False
        )
        assert ctable1 == ctable_close  # Should be equal due to np.isclose

        epsilon_different = 1.001  # Noticeable difference
        ctable_different = lotf.CutoffTable(
            X=X, index=index, epsilon=epsilon_different, verbose=False
        )
        assert ctable1 != ctable_different

        # Test pickle serialization equality
        serialized = pickle.dumps(ctable1)
        ctable_deserialized = pickle.loads(serialized)
        assert ctable1 == ctable_deserialized
        assert ctable1 is not ctable_deserialized

    def test_eq_l_values(self):
        """Test __eq__ method with various L value scenarios."""
        # Test with identical neighbor lists
        neighbor_lists = [
            np.array([1, 2]),
            np.array([0, 3]),
            np.array([0, 1]),
            np.array([1, 4]),
            np.array([3]),
        ]
        epsilon = 1.0
        N, D = 5, 3

        ctable1 = lotf.CutoffTable.from_neighbor_lists(
            neighbor_lists=neighbor_lists, epsilon=epsilon, N=N, D=D
        )
        ctable2 = lotf.CutoffTable.from_neighbor_lists(
            neighbor_lists=neighbor_lists, epsilon=epsilon, N=N, D=D
        )

        # Test basic L equality
        assert ctable1 == ctable2
        assert ctable1.L == ctable2.L

        # Test L floating point precision
        original_L = ctable2.L
        ctable2.L = original_L + 1e-15  # Tiny difference
        assert ctable1 == ctable2  # Should be equal due to np.isclose

        ctable2.L = original_L + 0.001  # Larger difference
        assert ctable1 != ctable2  # Should not be equal

        # Test with significantly different L
        ctable2.L = original_L + 10.0
        assert ctable1 != ctable2

        # Test different neighbor structures with different L values
        neighbor_lists_short = [
            np.array([1]),
            np.array([0]),
            np.array([]),
            np.array([]),
            np.array([]),
        ]
        neighbor_lists_long = [
            np.array([1, 2, 3]),
            np.array([0, 2, 3]),
            np.array([0, 1, 3]),
            np.array([0, 1, 2]),
            np.array([]),
        ]

        ctable_short = lotf.CutoffTable.from_neighbor_lists(
            neighbor_lists=neighbor_lists_short, epsilon=epsilon, N=N, D=D
        )
        ctable_long = lotf.CutoffTable.from_neighbor_lists(
            neighbor_lists=neighbor_lists_long, epsilon=epsilon, N=N, D=D
        )

        # Verify different L values and inequality
        assert abs(ctable_short.L - 0.4) < 1e-10
        assert abs(ctable_long.L - 2.4) < 1e-10
        assert ctable_short != ctable_long

    def test_repr_method(self, sample_data):
        """Test __repr__ method for CutoffTable string representation."""
        X, index = sample_data
        epsilon = 1.2345

        ctable = lotf.CutoffTable(X=X, index=index, epsilon=epsilon, verbose=False)
        repr_str = repr(ctable)

        # Check basic format and content
        assert "CutoffTable" in repr_str
        assert "epsilon=1.2345" in repr_str
        assert f"N={ctable.N}" in repr_str
        assert f"D={ctable.D}" in repr_str
        assert f"L={ctable.L:.2f}" in repr_str

        # Verify exact format
        expected = (
            f"CutoffTable(epsilon=1.2345, N={ctable.N}, D={ctable.D}, L={ctable.L:.2f})"
        )
        assert repr_str == expected

        # Test edge cases with from_neighbor_lists
        neighbor_lists_small = [np.array([1]), np.array([0])]
        ctable_small = lotf.CutoffTable.from_neighbor_lists(
            neighbor_lists=neighbor_lists_small, epsilon=1e-6, N=2, D=5
        )
        repr_small = repr(ctable_small)
        assert "epsilon=0.0000" in repr_small  # 4 decimal precision
        assert "N=2" in repr_small
        assert "D=5" in repr_small

        # Test with large values
        neighbor_lists_large = [
            np.array([1, 2, 3]),
            np.array([0, 2, 3]),
            np.array([0, 1, 3]),
            np.array([0, 1, 2]),
        ]
        ctable_large = lotf.CutoffTable.from_neighbor_lists(
            neighbor_lists=neighbor_lists_large, epsilon=999.5678, N=4, D=100
        )
        repr_large = repr(ctable_large)
        assert "epsilon=999.5678" in repr_large
        assert "N=4" in repr_large
        assert "D=100" in repr_large
