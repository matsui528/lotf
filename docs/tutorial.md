# Tutorial

LotusFilter (`lotf`) is a library designed for efficient search result diversification. By precomputing the CutoffTable and filtering the results of the approximate nearest neighbor search, LotusFilter enables diverse searches quickly.
The features of LotusFilter are as follows:

- **Lightweight**: LotusFilter is extremely fast, taking less than 0.1 milliseconds per query vector in most cases.
- **Independence**: By precomputing and storing the CutoffTable, LotusFilter can be treated as a completely independent post-processing module. In other words, LotsuFilter does not need to retain the original data (the set of vectors) to achieve diverse search.
- **Flexibility**: Since we assume ANN search is achieved by faiss, LotusFilter functions as a complete post-processing step for various faiss indexes. 
- **Simplicity**: LotusFilter is implemented as a single CutoffTable class, making it very easy to use. 

For technical details, please refer to our [CVPR paper](https://www.arxiv.org/abs/2506.04790).


## Basic usage

Let's see how to use LotusFilter. First, consider a standard search with faiss.

```python
import numpy as np
import faiss

np.set_printoptions(suppress=True, precision=4)  # Just for better visualization

# Prepare database. 3 dim x 15 vectors
Xb = np.array([
    [0.12, 0.34, 0.56],
    [0.78, 0.90, 0.11],
    [0.21, 0.42, 0.67],
    [0.21, 0.43, 0.66],
    [0.21, 0.46, 0.65],
    [0.20, 0.43, 0.64],
    [0.67, 0.89, 0.12],
    [0.34, 0.56, 0.78],
    [0.55, 0.23, 0.88],
    [0.31, 0.77, 0.45],
    [0.62, 0.14, 0.39],
    [0.81, 0.52, 0.27],
    [0.09, 0.68, 0.73],
    [0.90, 0.11, 0.23],
    [0.45, 0.67, 0.89]
], dtype=np.float32)

# Query. 3 dim x 1 vector
Xq = np.array([[0.21, 0.43, 0.65]], dtype=np.float32)

# Build index
index = faiss.IndexFlatL2(Xb.shape[1])
index.add(Xb)

# Search
k = 5
dists, ids = index.search(Xq, k)
print(dists)  # [[0.0001 0.0002 0.0005 0.0009 0.0243]]
print(ids)    # [[3 5 2 4 0]]
```

In the above, we perform a nearest neighbor search with a query vector `Xq` against 15 three-dimensional vectors `Xb`. Here we retrieve the top `k=5` results.

However, `Xb[3], Xb[5], Xb[2], Xb[4]` are all very similar to the query, so the top four results end up being almost identical (with `dists = [[0.0001 0.0002 0.0005 0.0009 0.0243]]`). Thus, the search result is not diverse. Depending on the application, such non-diverse results may be undesirable.

By using the LotusFilter, we can make the results more diverse. Let's try it on the above example:

```python
import lotf
epsilon = 0.01
ctable = lotf.CutoffTable(X=Xb, index=index, epsilon=epsilon)
```

Here, we create a CutoffTable by giving the original data `Xb`, the faiss index `index`, and the distance threshold `epsilon`. Using this, we rerun the search with diversification. First, we take a larger candidate set by setting `candidate_k=10`.

```python
# Candidate search
candidate_k = 10
candidate_dists, candidate_ids = index.search(Xq, candidate_k)
```

We now have more candidates. Next, filter the candidates by setting the final result size to `final_k=5`.

```python
# Diversification
final_k = 5
diverse_dists, diverse_ids = ctable.filter(
    dists=candidate_dists,
    ids=candidate_ids,
    final_k=final_k
)

print(diverse_dists)  # [[0.0001 0.0243 0.0507 0.0833 0.1656]]
print(diverse_ids)    # [[ 3  0  7 12  9]]
```

Here, filtering is applied to the candidate results, producing diverse results. For example, `X[5]` has been excluded, and the results are more diverse.

The function `ctable.filter` runs extremely fast. For about $10^6$ vectors of 1000 dimensions, it executes in less than 0.1 [ms/query]. In most cases, the search itself takes longer, so the cost of this diversification step is practically zero.

Moreover, the only requirement for filtering is the precomputed CutoffTable. We don't need to maintain the original data `Xb`. Thus, LotusFilter functions as a completely independent post-processing module.

The parameter `epsilon` controls how diverse the results are. The selected vectors are guaranteed at least `epsilon` apart (in squared Euclidean distance). A larger `epsilon` yields more diverse results.

The complete code is shown below:

```python
import numpy as np
import faiss

np.set_printoptions(suppress=True, precision=4)  # Just for better visualization

# Prepare database. 3 dim x 15 vectors
Xb = np.array([
    [0.12, 0.34, 0.56],
    [0.78, 0.90, 0.11],
    [0.21, 0.42, 0.67],
    [0.21, 0.43, 0.66],
    [0.21, 0.46, 0.65],
    [0.20, 0.43, 0.64],
    [0.67, 0.89, 0.12],
    [0.34, 0.56, 0.78],
    [0.55, 0.23, 0.88],
    [0.31, 0.77, 0.45],
    [0.62, 0.14, 0.39],
    [0.81, 0.52, 0.27],
    [0.09, 0.68, 0.73],
    [0.90, 0.11, 0.23],
    [0.45, 0.67, 0.89]
], dtype=np.float32)

# Query. 3 dim x 1 vector
Xq = np.array([[0.21, 0.43, 0.65]], dtype=np.float32)

# Build index
index = faiss.IndexFlatL2(Xb.shape[1])
index.add(Xb)

# Search
k = 5
dists, ids = index.search(Xq, k)
print(dists)  # [[0.0001 0.0002 0.0005 0.0009 0.0243]]
print(ids)    # [[3 5 2 4 0]]

import lotf
epsilon = 0.01
ctable = lotf.CutoffTable(X=Xb, index=index, epsilon=epsilon)

# Candidate search
candidate_k = 10
candidate_dists, candidate_ids = index.search(Xq, candidate_k)

# Diversification
final_k = 5
diverse_dists, diverse_ids = ctable.filter(
    dists=candidate_dists,
    ids=candidate_ids,
    final_k=final_k
)

print(diverse_dists)  # [[0.0001 0.0243 0.0507 0.0833 0.1656]]
print(diverse_ids)    # [[ 3  0  7 12  9]]
```

## Parameter tuning

The parameters to be set for using LotusFilter are as follows:

- `candidate_k`
    - This corresponds to $S$ in the paper. It determines the number of candidates for the search.
    - The choice is non-trivial, but generally, a larger value improves accuracy at the cost of computation. A good rule is to set it to a few times `final_k`, depending on runtime requirements.
- `final_k`
    - This corresponds to $K$ in the paper. It specifies the final number of results.
    - The user must set this depending on the required output size.
- `lam`
    - This corresponds to $\lambda$ in the paper. It balances between search results and diversification. With `lam=0.0`, only search results are emphasized. With `lam=1.0`, only diversification is emphasized, ignoring the query. Intermediate values such as `lam=0.3` balance the two.
    - When `epsilon` is manually chosen, `lam` does not directly appear, so we don't need to decide.
    - When `epsilon` is optimized automatically, `lam` must be considered.
- `epsilon`
    - This corresponds to $\varepsilon$ in the paper. It controls the degree of diversification. With `epsilon=0.0`, results are returned as is. A larger `epsilon` yields more diverse results.
    - In the "Basic usage" above, `epsilon` was manually chosen. That works fine, since LotusFilter simply selects vectors separated by at least `epsilon`.
    - Since determining `epsilon` is non-trivial, it can be chosen automatically using `optimize_epsilon`.

Thus, the most important parameter is `epsilon`. We can optimize it preliminarily as follows.

```python
import numpy as np
import lotf
import faiss

# Prepare the data
Xb = np.random.rand(10000, 128).astype('float32')  # Database vectors
Xq = np.random.rand(100, 128).astype('float32')    # Query vectors
Xt = np.random.rand(50, 128).astype('float32')     # Train vectors

# Build faiss index
index = faiss.IndexFlatL2(Xb.shape[1])
index.add(Xb)

# Hyperparameter tuning
candidate_k = 500
final_k = 100
lam = 0.3
best_parm = lotf.optimize_epsilon(
    X=Xb,
    Xq=Xt,  # Use training data for pseudo-query
    index=index,
    candidate_k=candidate_k,
    final_k=final_k,
    lam=lam,
)
print(best_param)  # {'epsilon': np.float32(15.420803), 'div_score': np.float32(6.9202075), 'L': np.float64(24.2842)}

# Build cutoff table for diversity
ctable = lotf.CutoffTable(X=Xb, index=index, epsilon=best_param['epsilon'])

# Do search here
```

Here, training data `Xt` is given and used as pseudo-queries when calling `optimize_epsilon`. This function constructs CutoffTables with various `epsilon` values, performs searches, evaluates the objective `div_score`, and finds the optimal `epsilon`.

Note that `candidate_k` and `final_k` must be set beforehand for this optimization.

Since real queries are usually unknown at data construction time, it is recommended to use training data `Xt` drawn from the same distribution, or part of `Xb` (removing them from `Xb` beforehand improves accuracy).

This parameter tuning may take time; reducing the number of pseudo-queries helps speed it up.


## I/O

CutoffTable can be read and written with `pickle`. CutoffTable consists of just numpy arrays and does not contain the original data and the nearest neighbor search index. As a result, it is easier to manage and requires less memory.

```python
import pickle

# Write
ctable = lotf.CutoffTable(...)

with open('cutoff_table.pkl', 'wb') as f:
    pickle.dump(ctable, f)

# Read
with open('cutoff_table.pkl', 'rb') as f:
    ctable_ = pickle.load(f)

# Same
assert(ctable == ctable_)
```

## Larger, real-world dataset

For larger datasets, it is better to use fast search methods such as `faiss.IndexHNSWFlat`. An example is shown below. [todo]

```python
...
# Build faiss index
D = ...
hnsw_m = 32
index = faiss.IndexHNSWFlat(D, hnsw_m)
index.add(Xb)

# Build cutoff table for diversity
epsilon = 15.0
ctable = lotf.CutoffTable(X=Xb, index=index, epsilon=epsilon)
...
```




## Implementation details

* LotusFilter is implemented in C++17 using boost unordered internally, with the C++ code exposed to Python via nanobind.
* The filter function of LotusFilter's CutoffTable is not parallelized; it simply applies single-threaded processing per query.


