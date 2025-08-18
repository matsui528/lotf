import numpy as np
from lotf.core import clean_invalid_values


class TestCleanInvalidValues:
    """Essential tests for the clean_invalid_values function."""

    def test_normal_values(self):
        """Test that normal finite values pass through unchanged."""
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = clean_invalid_values(vec)
        np.testing.assert_array_equal(result, vec)

    def test_mixed_invalid_values(self):
        """Test removal of all types of invalid values: NaN, inf, and extreme values."""
        vec = np.array([1.0, np.nan, 2.0, np.inf, 3.0, -np.inf, 1e15, 4.0, -1e12, 5.0])
        result = clean_invalid_values(vec)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(result, expected)

    def test_boundary_threshold_1e10(self):
        """Test the 1e10 threshold boundary for extreme values."""
        vec = np.array([1e10, -1e10, 9.99e9, -9.99e9])  # First two should be removed
        result = clean_invalid_values(vec)
        expected = np.array([9.99e9, -9.99e9])
        np.testing.assert_array_equal(result, expected)

    def test_all_invalid_array(self):
        """Test array where all values are invalid."""
        vec = np.array([np.nan, np.inf, -np.inf, 1e15])
        result = clean_invalid_values(vec)
        assert len(result) == 0
