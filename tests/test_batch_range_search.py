import numpy as np
import pytest
from lotf.core import batch_range_search

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None

# Skip all tests in this module if faiss is not available
pytestmark = pytest.mark.skipif(not HAS_FAISS, reason="faiss not available")


def test_batch_range_search_basic():
    """Test basic functionality of batch_range_search"""
    # Create test data
    Xb = np.array(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0], [1.5, 1.5]], dtype=np.float32
    )

    # Create FAISS index
    index = faiss.IndexFlatL2(2)
    index.add(Xb)

    # Test data for range search
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    thresh = 2.0
    bs = 1

    # Run batch range search
    lims_batch, dists_batch, ids_batch = batch_range_search(index, X, thresh, bs)

    # Run regular range search for comparison
    lims_regular, dists_regular, ids_regular = index.range_search(X, thresh)

    # Verify results are identical
    assert np.array_equal(lims_batch, lims_regular)
    assert np.array_equal(dists_batch, dists_regular)
    assert np.array_equal(ids_batch, ids_regular)


def test_batch_range_search_vs_regular():
    """Test that batch_range_search produces same results as regular range_search"""
    np.random.seed(42)

    # Create larger test dataset
    Xb = np.random.random((20, 5)).astype(np.float32)
    X = np.random.random((8, 5)).astype(np.float32)

    # Create FAISS index
    index = faiss.IndexFlatL2(5)
    index.add(Xb)

    thresh = 1.5
    bs = 3  # Small batch size to test batching logic

    # Run both methods
    lims_batch, dists_batch, ids_batch = batch_range_search(index, X, thresh, bs)
    lims_regular, dists_regular, ids_regular = index.range_search(X, thresh)

    # Verify results match
    assert np.array_equal(lims_batch, lims_regular)
    assert np.array_equal(dists_batch, dists_regular)
    assert np.array_equal(ids_batch, ids_regular)


# def test_batch_range_search_empty_input():
#     """Test batch_range_search with empty input"""
#     # Create empty test data
#     X = np.array([]).reshape(0, 2).astype(np.float32)
#     Xb = np.array([[1.0, 0.0]], dtype=np.float32)

#     index = faiss.IndexFlatL2(2)
#     index.add(Xb)

#     lims, dists, ids = batch_range_search(index, X, 1.0, 5)

#     # Should return arrays with appropriate shapes for empty input
#     assert len(lims) == 1 and lims[0] == 0
#     assert len(dists) == 0
#     assert len(ids) == 0


def test_batch_range_search_single_batch():
    """Test batch_range_search when batch size is larger than data"""
    Xb = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    X = np.array([[0.5, 0.5]], dtype=np.float32)

    index = faiss.IndexFlatL2(2)
    index.add(Xb)

    # Large batch size (larger than X)
    lims, dists, ids = batch_range_search(index, X, 2.0, 100)

    # Should work the same as regular range search
    lims_regular, dists_regular, ids_regular = index.range_search(X, 2.0)

    assert np.array_equal(lims, lims_regular)
    assert np.array_equal(dists, dists_regular)
    assert np.array_equal(ids, ids_regular)


def test_batch_range_search_different_batch_sizes():
    """Test batch_range_search with different batch sizes produces same results"""
    np.random.seed(123)

    Xb = np.random.random((15, 3)).astype(np.float32)
    X = np.random.random((6, 3)).astype(np.float32)

    index = faiss.IndexFlatL2(3)
    index.add(Xb)

    thresh = 1.0

    # Test with different batch sizes
    results = []
    for bs in [1, 2, 3, 6, 10]:
        lims, dists, ids = batch_range_search(index, X, thresh, bs)
        results.append((lims, dists, ids))

    # All results should be identical
    base_lims, base_dists, base_ids = results[0]
    for lims, dists, ids in results[1:]:
        assert np.array_equal(lims, base_lims)
        assert np.array_equal(dists, base_dists)
        assert np.array_equal(ids, base_ids)
