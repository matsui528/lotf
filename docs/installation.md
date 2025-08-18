# Installation

## Prerequisites

- Python 3.10 or later
- C++ 17 or later
- [Faiss](https://github.com/facebookresearch/faiss) (CPU)

````{note}

This library is for "post-processing" the approximate nearest neighbor search results using faiss.

To install faiss, run the following using conda:

```bash
conda install -c pytorch faiss-cpu
```

If conda cannot be used, you can install the [unofficial pip version](https://github.com/kyamagu/faiss-wheels) with:

```bash
pip install faiss-cpu
```

Even if Faiss cannot be used, as long as an interface emulating Faiss is prepared, it is possible to use other approximate nearest neighbor search libraries.
````


````{note}
**For most users**: No additional dependencies are needed. The PyPI wheels are pre-built with optimal performance libraries.

**For developers building from source**: Installing [Boost Unordered](https://www.boost.org/doc/libs/latest/libs/unordered/index.html) is recommended for better performance, but not required.

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


## Install from PyPI

You can install the LousFilter via pip with the following command:


```bash
pip install lotf
```


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

# Step 3: Search with diversity
candidate_k, final_k = 300, 100
candidate_dists, candidate_ids = index.search(Xq, candidate_k)
diverse_dists, diverse_ids = ctable.filter(
    dists=candidate_dists, 
    ids=candidate_ids, 
    final_k=final_k
)

print(f"Diverse results: {diverse_ids}")
```




## Install from Source (for developers)

### Prerequisites for Source Build

Optional but recommended for optimal performance:

```bash
# Ubuntu/Debian
sudo apt install libboost-dev

# macOS
brew install boost

# Windows (with vcpkg)
vcpkg install boost:x64-windows
```

### Build and Install

```bash
git clone https://github.com/matsui528/lotf.git
cd lotf
make install_dev
```

````{tip}
If Boost is not available during source build, the package will automatically fall back to standard library containers. Performance will be slightly reduced but functionality remains identical.
````