"""Tests for version access functionality."""

import re


def test_version_import():
    """Test that __version__ can be imported and accessed."""
    import lotf

    # Should be able to access __version__
    assert hasattr(lotf, "__version__")

    # Version should be a string
    assert isinstance(lotf.__version__, str)

    # Version should not be empty
    assert len(lotf.__version__) > 0


def test_version_in_all():
    """Test that __version__ is included in __all__."""
    import lotf

    assert "__version__" in lotf.__all__


def test_version_format():
    """Test that version follows semantic versioning pattern."""
    import lotf

    # Should match PEP 440 version pattern (semantic version with optional dev/alpha/beta/rc)
    version_pattern = r"^\d+\.\d+\.\d+(?:\.(?:dev|a|b|rc)\d+|[-+]\w+)?$"
    assert re.match(version_pattern, lotf.__version__), (
        f"Version '{lotf.__version__}' does not match expected pattern"
    )


def test_version_reasonable():
    """Test that version is reasonable (not empty and follows basic pattern)."""
    import lotf

    # Version should not be empty and should be a reasonable semantic version
    assert lotf.__version__
    assert isinstance(lotf.__version__, str)

    # Should either be a proper semantic version or development version
    import re

    version_pattern = r"^\d+\.\d+\.\d+(?:\.(?:dev|a|b|rc)\d+|[-+]\w+)?$"
    assert re.match(version_pattern, lotf.__version__), (
        f"Version '{lotf.__version__}' does not follow expected pattern"
    )


def test_star_import_includes_version():
    """Test that version is available with star import."""
    # This test uses exec to simulate 'from lotf import *'
    namespace = {}
    exec("from lotf import *", namespace)

    assert "__version__" in namespace
    assert isinstance(namespace["__version__"], str)
