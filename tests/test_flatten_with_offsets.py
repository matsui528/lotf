import numpy as np
from lotf.core import flatten_with_offsets


def test_flatten_with_offsets_basic():
    """Test basic functionality of flatten_with_offsets"""
    list_of_list = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]

    flattened, offsets = flatten_with_offsets(list_of_list)

    expected_flattened = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    expected_offsets = np.array([0, 3, 7, 9])

    assert np.array_equal(flattened, expected_flattened)
    assert np.array_equal(offsets, expected_offsets)


# def test_flatten_with_offsets_empty_list():
#     """Test with an empty list"""
#     list_of_list = []

#     flattened, offsets = flatten_with_offsets(list_of_list)

#     assert len(flattened) == 0
#     assert np.array_equal(offsets, np.array([0]))


def test_flatten_with_offsets_single_list():
    """Test with a single list"""
    list_of_list = [[1, 2, 3, 4, 5]]

    flattened, offsets = flatten_with_offsets(list_of_list)

    expected_flattened = np.array([1, 2, 3, 4, 5])
    expected_offsets = np.array([0, 5])

    assert np.array_equal(flattened, expected_flattened)
    assert np.array_equal(offsets, expected_offsets)


def test_flatten_with_offsets_empty_sublists():
    """Test with empty sublists"""
    list_of_list = [[1, 2], [], [3, 4, 5], []]

    flattened, offsets = flatten_with_offsets(list_of_list)

    expected_flattened = np.array([1, 2, 3, 4, 5])
    expected_offsets = np.array([0, 2, 2, 5, 5])

    assert np.array_equal(flattened, expected_flattened)
    assert np.array_equal(offsets, expected_offsets)


def test_flatten_with_offsets_numpy_arrays():
    """Test with numpy arrays as input"""
    list_of_list = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]

    flattened, offsets = flatten_with_offsets(list_of_list)

    expected_flattened = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_offsets = np.array([0, 2, 5])

    assert np.array_equal(flattened, expected_flattened)
    assert np.array_equal(offsets, expected_offsets)
