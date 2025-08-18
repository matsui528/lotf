import numpy as np
import pytest
from lotf.core import build_neighbor_lists

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None

# Skip all tests in this module if faiss is not available
pytestmark = pytest.mark.skipif(not HAS_FAISS, reason="faiss not available")


def test_build_neighbor_lists_basic():
    """Test basic functionality of build_neighbor_lists with exact values."""
    X = np.array(
        [
            [0.0, 0.0],  # Point 0
            [1.0, 0.0],  # Point 1: distance 1.0 from point 0
            [0.0, 1.0],  # Point 2: distance 1.0 from point 0
            [2.0, 0.0],  # Point 3: distance 2.0 from point 0, sqrt(2) from point 1
        ],
        dtype=np.float32,
    )

    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    epsilon = 1.5  # Should include points within distance 1.5
    neighbor_lists, neighbor_dists = build_neighbor_lists(
        X=X, index=index, epsilon=epsilon, with_dist=True, batch_size=10, verbose=False
    )

    # Point 0 should have neighbors [1, 2] (both at distance 1.0)
    assert len(neighbor_lists[0]) == 2
    assert set(neighbor_lists[0]) == {1, 2}
    np.testing.assert_array_almost_equal(neighbor_dists[0], [1.0, 1.0], decimal=5)

    # Point 1 should have neighbors [0, 3] (both at distance 1.0)
    assert len(neighbor_lists[1]) == 2
    assert neighbor_lists[1][0] == 0  # Closest first (or same distance)
    assert neighbor_lists[1][1] == 3
    np.testing.assert_array_almost_equal(neighbor_dists[1], [1.0, 1.0], decimal=5)

    # Point 2 should have neighbor [0] (at distance 1.0)
    assert len(neighbor_lists[2]) == 1
    assert neighbor_lists[2][0] == 0
    np.testing.assert_array_almost_equal(neighbor_dists[2], [1.0], decimal=5)

    # Point 3 should have neighbor [1] (at distance 1.0)
    assert len(neighbor_lists[3]) == 1
    assert neighbor_lists[3][0] == 1
    np.testing.assert_array_almost_equal(neighbor_dists[3], [1.0], decimal=5)


def test_build_neighbor_lists_without_dist():
    """Test build_neighbor_lists with with_dist=False and exact neighbor IDs."""
    X = np.array(
        [
            [0.0, 0.0],  # Point 0
            [1.0, 0.0],  # Point 1
            [0.0, 1.0],  # Point 2
        ],
        dtype=np.float32,
    )

    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    epsilon = 1.1  # Should include all neighbors at distance 1.0
    neighbor_lists, neighbor_dists = build_neighbor_lists(
        X=X, index=index, epsilon=epsilon, with_dist=False, batch_size=10, verbose=False
    )

    assert neighbor_dists is None

    # Point 0 should have neighbors [1, 2] (sorted by distance, both at 1.0)
    assert len(neighbor_lists[0]) == 2
    assert set(neighbor_lists[0]) == {1, 2}

    # Point 1 should have neighbor [0]
    assert len(neighbor_lists[1]) == 1
    assert neighbor_lists[1][0] == 0

    # Point 2 should have neighbor [0]
    assert len(neighbor_lists[2]) == 1
    assert neighbor_lists[2][0] == 0


def test_build_neighbor_lists_batch_vs_non_batch():
    """Test that batch and non-batch processing produce identical results."""
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    X = np.random.rand(10, 3).astype(np.float32)

    index1 = faiss.IndexFlatL2(X.shape[1])
    index1.add(X)
    index2 = faiss.IndexFlatL2(X.shape[1])
    index2.add(X)

    epsilon = 1.0

    # Non-batch mode (batch_size > N)
    neighbor_lists_non_batch, neighbor_dists_non_batch = build_neighbor_lists(
        X=X,
        index=index1,
        epsilon=epsilon,
        with_dist=True,
        batch_size=20,  # > N, so non-batch
        verbose=False,
    )

    # Batch mode (batch_size < N)
    neighbor_lists_batch, neighbor_dists_batch = build_neighbor_lists(
        X=X,
        index=index2,
        epsilon=epsilon,
        with_dist=True,
        batch_size=5,  # < N, so batch mode
        verbose=False,
    )

    # Results should be identical
    assert len(neighbor_lists_batch) == len(neighbor_lists_non_batch)
    assert len(neighbor_dists_batch) == len(neighbor_dists_non_batch)

    for i in range(len(X)):
        # Check neighbor IDs are identical
        np.testing.assert_array_equal(
            neighbor_lists_batch[i], neighbor_lists_non_batch[i]
        )
        # Check distances are identical
        np.testing.assert_array_almost_equal(
            neighbor_dists_batch[i], neighbor_dists_non_batch[i], decimal=5
        )
