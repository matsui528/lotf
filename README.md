<div align="center">

![LotusFilter Logo](logo.png)

[![PyPI version](https://badge.fury.io/py/lotf.svg)](https://badge.fury.io/py/lotf)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://matsui528.github.io/lotf)
[![CI](https://github.com/matsui528/lotf/workflows/Tests/badge.svg)](https://github.com/matsui528/lotf/actions)

**LotusFilter**: Diversity-aware approximate nearest neighbor search combining Faiss with cutoff tables to prevent similar results from dominating search outputs.

</div>

## ✨ Features

- ⚡ **Lightweight**: Ultra-fast processing < 0.1ms per query
- 🔓 **Independence**: Post-processing module that doesn't require original data retention  
- 🔧 **Flexibility**: Works seamlessly with various Faiss indexes
- 📦 **Simplicity**: Single `CutoffTable` class - easy to integrate


## 🚀 Quick Start

### Installation

```bash
pip install lotf
```




### Simple Usage

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

## 📄 Citation

If you use LotusFilter in your research, please cite our CVPR 2025 paper:

```bibtex
@inproceedings{mtasui2025cvpr,
    author    = {Yusuke Matsui},
    title     = {LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025},
    pages     = {30430-30439}
}
```

## 📚 Documentation

- **Repository**: https://github.com/matsui528/lotf
- **Documentation**: https://matsui528.github.io/lotf
- **Paper**: https://www.arxiv.org/abs/2506.04790




## 👤 Author

**Yusuke Matsui**
- Email: matsui@hal.t.u-tokyo.ac.jp
- GitHub: [@matsui528](https://github.com/matsui528)
- Web: https://yusukematsui.me

