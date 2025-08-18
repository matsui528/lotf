# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LotusFilter (lotf) is a Python library for diversity-aware approximate nearest neighbor search. It combines Faiss with cutoff tables to prevent similar results from dominating search outputs. The project consists of a Python API with C++ extensions for performance-critical operations.

## Architecture

### Core Components

- **`src/lotf/core.py`**: Main Python API containing the `CutoffTable` class and utility functions
- **`src/lotf_ext.cpp`**: C++ extension module using nanobind for high-performance filtering operations
- **`src/lotf/__init__.py`**: Package initialization with version management and exports

### Key Classes and Functions

- **`CutoffTable`**: Primary class for diversity filtering using precomputed neighbor lists
- **`div_score()`**: Computes diversity scores combining similarity and diversity terms
- **`optimize_epsilon()`**: Hyperparameter optimization for epsilon values using iterative grid search
- **`build_neighbor_lists()`**: Constructs neighbor lists from Faiss range search results
- **`batch_range_search()`**: Memory-efficient range search for large datasets

### C++ Extension

The C++ module (`lotf_ext.cpp`) implements performance-critical filtering algorithms and automatically detects boost containers for optimal performance. Uses `boost::unordered_flat_map/set` when available, falls back to `std::unordered_map/set`.

## Development Commands

### Installation
```bash
# Development installation with dependencies
make install_dev

# Regular installation
make install

# Uninstall
make uninstall
```

### Testing
```bash
# Run all tests
make test

# Or directly with pytest
python -m pytest tests

# Run specific test file
python -m pytest tests/test_cutoff_table.py
```

### Code Quality
```bash
# Lint code (check only)
make lint

# Format code (apply fixes)
make format

# Manual ruff commands
ruff check src/ tests/
ruff format src/ tests/
```

### Documentation
```bash
# Build documentation
make docs

# Serve documentation with auto-reload
make docs_serve

# Clean documentation build
make docs_clean
```

### Build System
```bash
# Clean build artifacts
make clean
```

## Build System Details

- **Build backend**: scikit-build-core with nanobind
- **Python bindings**: nanobind (modern alternative to pybind11)
- **C++ compilation**: CMake with automatic dependency detection
- **Package config**: pyproject.toml with all metadata and dependencies

### Dependencies

- **Core**: numpy (required)
- **Optional**: faiss-cpu (for accelerated distance computations)
- **Build**: nanobind, scikit-build-core
- **Dev/Docs**: sphinx, furo, myst-parser, pytest

## Key Implementation Notes

### Performance Optimizations

- The C++ extension automatically detects and uses boost flat containers when available
- Faiss integration provides accelerated pairwise distance computations in `div_score()`
- Batch processing in `batch_range_search()` handles large datasets efficiently
- Memory-efficient neighbor list flattening for C++ processing

### Error Handling

- `clean_invalid_values()` removes NaN, infinite, and extreme outlier values from Faiss outputs
- Comprehensive input validation in all public methods
- Graceful fallbacks when optional dependencies are unavailable

### Testing Strategy

Tests cover all major functionality with separate files for each component:
- `test_cutoff_table.py`: Core filtering functionality
- `test_div_score.py`: Diversity scoring
- `test_optimize_epsilon.py`: Parameter optimization
- `test_batch_range_search.py`: Large dataset handling
- Additional utility function tests

## Platform Support

- **Python**: 3.10+ (type hints use modern syntax)
- **Platforms**: Windows, macOS (10.14+), Linux
- **CI**: GitHub Actions with conda environments
- **Wheel building**: cibuildwheel for cross-platform distribution