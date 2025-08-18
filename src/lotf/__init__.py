# Version management using importlib.metadata (Python 3.10+)
try:
    from importlib.metadata import version

    __version__ = version("lotf")
except Exception:
    # Fallback for development/uninstalled package
    __version__ = "0.0.1-dev"


from .core import (
    CutoffTable,
    div_score,
    optimize_epsilon,
    print_backend,
    clean_invalid_values,
    batch_range_search,
    flatten_with_offsets,
    build_neighbor_lists,
)

# Make all functions available at package level
__all__ = [
    "__version__",
    "CutoffTable",
    "div_score",
    "optimize_epsilon",
    "print_backend",
    "clean_invalid_values",
    "batch_range_search",
    "flatten_with_offsets",
    "build_neighbor_lists",
]
