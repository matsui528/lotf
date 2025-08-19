# Installation

## Prerequisites

- Python 3.10 or later
- [Faiss](https://github.com/facebookresearch/faiss) (CPU)

````{note}

This library is for "post-processing" for faiss. To install faiss, run the following using conda:

```bash
conda install -c pytorch faiss-cpu
```

If conda cannot be used, you can install the [unofficial pip version](https://github.com/kyamagu/faiss-wheels) with:

```bash
pip install faiss-cpu
```

Even if Faiss cannot be used, as long as an interface emulating Faiss is prepared, it is possible to use other approximate nearest neighbor search libraries.
````





## Install from PyPI

You can install the LousFilter via pip with the following command:


```bash
pip install lotf
```

````{note}
We provide pre-built wheels for Linux (x64/arm64), macOS (Intel/arm64), and Windows (x64). For other environments, installing via pip will build the library locally. In that case, as shown below, we recommend ensuring a C++17-capable toolchain and preparing Boost Unordered. Such cases apply to PyPy and older Linux distributions.
````


## Quick Example
If the installation is successful, you can run the following.

```python
import lotf
import faiss
import numpy as np

# Prepare data
Xb = np.random.rand(10000, 128).astype('float32')  # Database vectors
Xq = np.random.rand(5, 128).astype('float32')      # Query vectors

# Step 1: Build Faiss index
index = faiss.IndexFlatL2(Xb.shape[1])
index.add(Xb)

# Step 2: Build cutoff table for diversity filtering
epsilon = 15.0
ctable = lotf.CutoffTable(X=Xb, index=index, epsilon=epsilon)

# Step 3: Search candidates
candidate_k = 300
candidate_dists, candidate_ids = index.search(Xq, candidate_k)

# Step 4: Filter out candidates to obtain diverse results
final_k = 100
diverse_dists, diverse_ids = ctable.filter(
    dists=candidate_dists, 
    ids=candidate_ids, 
    final_k=final_k
)

print(f"Diverse results: {diverse_ids}")
```




## Install from Source (for developers)

### Prerequisites for Source Build
- C++17 or later
- [nanobind](https://github.com/wjakob/nanobind)
- [Boost Unordered](https://www.boost.org/doc/libs/latest/libs/unordered/index.html)
    ```bash
    # Ubuntu 24.04+
    sudo apt install libboost-dev
    
    # macOS (note tested)
    brew install boost
    
    # Windows (with vcpkg)
    vcpkg install boost:x64-windows
    ```

````{note}
**For most users**: No additional dependencies are needed. The PyPI wheels are pre-built with Boost Unordered.

**For developers building from source**: Installing Boost Unordered is recommended for better performance, but not required. If Boost is not available during source build, the package will automatically fall back to standard library containers. Performance will be slightly reduced but functionality remains identical.

You can check which backend is being used:

```bash
python -c "import lotf; lotf.print_backend()"
```

Expected output for PyPI installation:
```
div_score: Faiss  # or NumPy if Faiss not installed
filter: boost::unordered_flat_map/set  # Pre-built with Boost optimization
```

````




### Build and Install

```bash
git clone https://github.com/matsui528/lotf.git
cd lotf
make install_dev
```
