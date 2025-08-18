# LotusFilter Documentation

[LotusFilter (lotf)](https://github.com/matsui528/lotf) is a Python package implementing diversity-aware approximate nearest neighbor search, [presented at CVPR 2025](https://www.arxiv.org/abs/2506.04790). It combines faiss for fast similarity search with cutoff tables to filter results for diversity, preventing similar results from dominating the output.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
tutorial
api
```

## Citation
```
@inproceedings{mtasui2025cvpr,
    author    = {Yusuke Matsui},
    title     = {LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025},
    pages     = {30430-30439}
}
```

