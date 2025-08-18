import time
import numpy as np
from .lotf_ext import search_cpp, get_unordered_container_type

# Optional faiss import with graceful fallback
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None


def print_backend() -> None:
    """Print information about available backends.

    Shows which backend implementations are being used:

    - div_score: Uses Faiss for pairwise distance computation if available, otherwise falls back to NumPy
    - filter: Uses boost::unordered_flat_map/set if available for faster filtering, otherwise uses std::unordered_map/set

    Example::

        lotf.print_backend()
        > div_score: Faiss
        > filter: boost::unordered_flat_map/set

    """
    print(f"div_score: {'Faiss' if HAS_FAISS else 'NumPy'}")
    print(f"filter: {get_unordered_container_type()}")


def div_score(
    dists: np.ndarray, ids: np.ndarray, X: np.ndarray, lam: float
) -> tuple[float, float, float]:
    """Compute diversity score for a search result (function :math:`f` from Eq 2 in the paper).

    Args:
        dists: 1D array of distances from query to its top-k neighbors (squared L2 distances from faiss search). Shape: (``topk``,)
        ids: 1D array of neighbor IDs (corresponding to the distances from faiss search). Shape: (``topk``,)
        X: 2D array containing the original dataset vectors. Shape: (``N``, ``D``)
        lam: weight parameter for the diversity term (0 = pure NN, 1 = pure diversity)

    Returns:
        tuple:
            - total_score: Combined diversity-aware score ``= (1-lam) * similarity_term + lam * diversity_term``
            - similarity_term: Average similarity score
            - diversity_term: Average diversity score (negative value)


    Example::

        import numpy as np
        X = np.random.rand(10, 3).astype(np.float32)
        q = np.random.rand(1, 3).astype(np.float32)
        topk = 3
        dists = np.sum((q - X)**2, axis=1)
        ids = np.argsort(dists)[:topk]
        dists = dists[ids]

        import lotf
        score, sim_term, div_term = lotf.div_score(dists, ids, X, lam=0.5)

    """
    if type(dists) in [list, tuple]:
        dists = np.array(dists)
    if type(ids) in [list, tuple]:
        ids = np.array(ids)
    assert dists.ndim == ids.ndim == 1
    assert dists.shape[0] == ids.shape[0]

    topk = dists.shape[0]

    # Similarity term: Average query-to-neighbor distance (remove invalid values that may appear in faiss output)
    similarity_score = np.sum(clean_invalid_values(dists.ravel())) / topk

    # Diversity term: Measure how diverse the selected neighbors are from each other
    diversity_score = 0.0
    Xs = X[ids]  # Extract the actual vectors of selected neighbors

    # Compute pairwise distances between selected neighbors (use faiss if available for speed)
    if HAS_FAISS:
        distmat = faiss.pairwise_distances(Xs, Xs)
    else:
        # NumPy fallback: compute pairwise squared L2 distances using the identity
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
        norms_sq = np.sum(Xs**2, axis=1, keepdims=True)
        distmat = norms_sq + norms_sq.T - 2 * np.dot(Xs, Xs.T)
        # Clamp to zero to handle numerical precision issues
        distmat = np.maximum(distmat, 0.0)

    # Extract upper triangular elements (excluding diagonal) to get unique pairwise distances
    triu_inds = np.triu_indices(distmat.shape[0], k=1)  # k=1 excludes diagonal elements

    # Diversity score is the negative minimum pairwise distance (closer neighbors = lower diversity)
    # Handle edge case: if only one point selected, no pairwise distances exist
    if len(triu_inds[0]) == 0:
        diversity_score = 0.0  # No diversity information available
    else:
        diversity_score = -np.min(distmat[triu_inds])

    # Combine similarity and diversity terms using lambda weighting
    combined_score = (1.0 - lam) * similarity_score + lam * diversity_score
    return combined_score, similarity_score, diversity_score


def clean_invalid_values(vec: np.ndarray) -> np.ndarray:
    """Remove invalid and extreme values from a numeric array.

    Faiss search results sometimes contain problematic values that can cause
    numerical issues in downstream computations. This function filters out:

    - NaN (Not a Number) values
    - Infinite values (+inf, -inf)
    - Extremely large outliers (absolute value > 1e10)

    Args:
        vec: 1D numpy array of numeric values to clean

    Returns:
        Cleaned array containing only valid, finite values within reasonable range
    """
    # Step 1: Remove NaN and infinite values using numpy's isfinite check
    valid_vec = vec[np.isfinite(vec)]

    # Step 2: Remove extreme outliers that could cause numerical instability
    # Threshold of 1e10 helps filter out unrealistic distance values
    valid_vec = valid_vec[np.abs(valid_vec) < 1e10]

    return valid_vec


def batch_range_search(
    index, X: np.ndarray, thresh: float, bs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform range search in batches to handle large datasets efficiently.

    This function processes large datasets by splitting them into smaller batches,
    performing range search on each batch, and then properly combining the results.
    This approach helps manage memory usage while maintaining identical results to
    a single large range search operation.

    Args:
        index: Faiss index object with range_search capability
        X: 2D numpy array of query vectors, shape (``N``, ``D``)
        thresh: maximum squared L2 distance threshold for neighbors
        bs: batch size (number of queries to process per batch)

    Returns:
        tuple: (lims, dists, ids) matching Faiss range_search format
            - lims: 1D array of cumulative neighbor counts, length N+1, starts with 0
            - dists: 1D array of all distances (flattened across all queries)
            - ids: 1D array of all neighbor IDs (flattened across all queries)

    Example::

        # These produce identical results:
        lims1, dists1, ids1 = index.range_search(x=X, thresh=0.5)
        lims2, dists2, ids2 = batch_range_search(index, X, thresh=0.5, bs=100)
        assert np.array_equal(lims1, lims2)
        assert np.array_equal(dists1, dists2)
        assert np.array_equal(ids1, ids2)

    Note:
        For very large datasets, this function prevents memory issues that could
        occur with direct range_search calls on the full dataset.
    """
    N = X.shape[0]

    # Initialize containers for batch results
    lims_all = []
    dists_all = []
    ids_all = []

    # Process data in batches
    for i in range(0, N, bs):
        j = min(i + bs, N)  # Handle final partial batch

        # Perform range search on current batch
        lims, dists, ids = index.range_search(x=X[i:j], thresh=thresh)

        # Handle cumulative limit offset for proper concatenation
        if i != 0:
            # Remove the leading zero from subsequent batches (FAISS always starts limits with 0)
            lims = lims[1:]
            # Add offset to align with previous batches' cumulative neighbor counts
            lims += lims_all[-1][-1]

        # Store batch results
        lims_all.append(lims)
        dists_all.append(dists)
        ids_all.append(ids)

    # Concatenate all batch results into final arrays
    lims_all = np.concatenate(lims_all, axis=0)
    dists_all = np.concatenate(dists_all, axis=0)
    ids_all = np.concatenate(ids_all, axis=0)

    return lims_all, dists_all, ids_all


def flatten_with_offsets(list_of_list: list) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a list of lists (or np arrays) into a single vector with offset information.

    Args:
        list_of_list: A list of lists (or np arrays), e.g., ``[[1, 2, 3], [4, 5, 6, 7], [8, 9]]``

    Returns:
        tuple:
            - flattened: A single vector, e.g., ``np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])``
            - offsets: Cumulative sum of the lengths of the lists/arrays, e.g., ``np.array([0, 3, 7, 9])``

    Example::

        flattened, offsets = flatten_with_offsets([[1, 2, 3], [4, 5, 6, 7], [8, 9]])
        print(flattened)  # Output: [1 2 3 4 5 6 7 8 9]
        print(offsets)    # Output: [0 3 7 9]

    """
    flattened = np.hstack(list_of_list)
    offsets = np.cumsum([0] + [len(lst) for lst in list_of_list])
    return flattened, offsets


def build_neighbor_lists(
    X: np.ndarray,
    index,
    epsilon: float,
    with_dist: bool,
    batch_size: int,
    verbose: bool,
) -> tuple[list, list | None]:
    """Build raw neighbor lists (:math:`\{\mathcal{L}_n\}_{n=1}^N` in the paper). This function is used inside :class:`CutoffTable`. End users don't need to call this usually.

    Args:
        X: 2D numpy array of dataset vectors
        index: Faiss index for range search. Should be created with ``X``.
        epsilon: distance threshold
        with_dist: whether to return distances
        batch_size: batch size for large datasets
        verbose: whether to print progress information

    Returns:
        tuple:
            - neighbor_lists: list of np.array, :math:`\{\mathcal{L}_n\}_{n=1}^N` in the paper
            - dist_lists: list of np.array or None, distances corresponding to ``neighbor_lists`` if ``with_dist=True``
    """
    N, _ = X.shape
    if verbose:
        print(f"Starting range search on dataset with shape {X.shape}")

    t0 = time.time()
    if N > batch_size:
        # If N is too large, run range search in a batch manner.
        if verbose:
            print("Running batch range search")
        lims, dists, ids = batch_range_search(
            index=index, X=X, thresh=epsilon, bs=batch_size
        )
    else:
        if verbose:
            print("Running standard range search")
        lims, dists, ids = index.range_search(x=X, thresh=epsilon)
    t_range = time.time() - t0

    if verbose:
        print(f"Range search completed in {t_range * 1000:.1f} ms")
        print("Constructing neighbor lists for cutoff table")

    neighbor_lists = []
    neighbor_lists_dists = []

    t0 = time.time()
    for n in range(N):
        target_ids = ids[
            lims[n] : lims[n + 1]
        ]  # ids of close items (d < eplison) for n

        # Sort target_ids by dists, e.g., if
        # target_ids =   [  2,   3,   5,  32,  67] and
        # target_dists = [0.3, 1.9, 0.2, 1.8, 2.2], then
        # target_ids =   [  5,   2,  32,  3,   67]
        target_dists = dists[lims[n] : lims[n + 1]]

        # (1) Sort by dist
        sorted_ids = np.argsort(target_dists)
        target_ids = target_ids[sorted_ids]
        if with_dist:
            target_dists = target_dists[sorted_ids]

        # (2) Delete the current n itself. e.g., if n = 3 and target_ids_sorted = [5, 2, 32, 3, 67],
        # then target_ids would be [5, 2, 32, 67]
        if with_dist:
            target_dists = target_dists[target_ids != n]
        target_ids = target_ids[target_ids != n]

        # (3) Save the result
        neighbor_lists.append(target_ids)
        if with_dist:
            neighbor_lists_dists.append(target_dists)

    t_construct = time.time() - t0
    if verbose:
        print(f"Neighbor list construction completed in {t_construct * 1000:.1f} ms")
        print(f"Total preprocessing time: {(t_range + t_construct) * 1000:.1f} ms")

    if with_dist:
        assert len(neighbor_lists) == len(neighbor_lists_dists)
        assert [len(a) for a in neighbor_lists] == [
            len(b) for b in neighbor_lists_dists
        ]
        return neighbor_lists, neighbor_lists_dists
    else:
        return neighbor_lists, None


class CutoffTable:
    """Cutoff table for diversity-aware filtering of search results. The wrapper class for :math:`\{\mathcal{L}_n\}_{n=1}^N` in the paper.

    Precomputes neighbor lists within epsilon distance for efficient filtering.
    Used to remove similar items from search results to increase diversity.

    Note:
        This class takes ``X`` and a faiss index as arguments, but they are only used for constructing the table and are not stored.
        Therefore, this class consumes only the memory required for the table after construction.

    """

    def __init__(
        self,
        X: np.ndarray,
        index,
        epsilon: float,
        batch_size: int = 1000,
        verbose: bool = True,
    ) -> None:
        """Initialize CutoffTable.

        Args:
            X: 2D numpy array of dataset vectors
            index: Faiss index for range search constructed with ``X``. E.g., ``faiss.IndexFlatL2`` or ``faiss.IndexHNSWFlat``
            epsilon: distance threshold
            batch_size: batch size for large datasets
            verbose: whether to print progress information
        """
        # Type declarations for all instance attributes
        self.epsilon: float
        self.N: int
        self.D: int
        self.L: float
        self.flattened_neighbors: np.ndarray
        self.neighbor_offsets: np.ndarray

        # Input validation
        assert isinstance(epsilon, (int, float)) and epsilon > 0, (
            f"epsilon must be a positive number, got {epsilon}"
        )
        assert isinstance(batch_size, int) and batch_size > 0, (
            f"batch_size must be a positive integer, got {batch_size}"
        )
        assert X.ndim == 2, f"X must be a 2D array, got shape {X.shape}"

        # Initialize basic attributes
        self.epsilon = epsilon
        self.N, self.D = X.shape

        # Verify index consistency
        assert self.N == index.ntotal, (
            f"Dataset size mismatch: X has {self.N} vectors but index has {index.ntotal} vectors"
        )
        assert self.N > 0, f"Dataset cannot be empty, got {self.N} vectors"
        assert hasattr(index, "range_search") and callable(
            getattr(index, "range_search")
        ), (
            "The provided index must have a callable 'range_search' method for cutoff table construction"
        )

        # Build neighbor lists using range search
        # neighbor_lists is a raw data structure for {\mathcal{L}_n}.
        # Example:
        #   If neighbor_lists = [[24, 12, 77], [21, 41, 33, 19], [23, 77]], it means
        #   L_1 = [24, 12, 77]
        #   L_2 = [21, 41, 33, 19]
        #   L_3 = [23, 77]
        neighbor_lists, _ = build_neighbor_lists(
            X=X,
            index=index,
            epsilon=epsilon,
            with_dist=False,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Compute average neighbor list length (L from the paper)
        self.L = np.mean([len(ids) for ids in neighbor_lists])
        if verbose:
            print(f"Average neighbor list length (L): {self.L:.2f}")

        # Flatten neighbor lists for efficient C++ processing
        # Example:
        #   If neighbor_lists = [[24, 12, 77], [21, 41, 33, 19], [23, 77]], then
        #   flattened_neighbors = np.array([24, 12, 77, 21, 41, 33, 19, 23, 77])
        #   neighbor_offsets = np.array([0, 3, 7, 9])
        self.flattened_neighbors, self.neighbor_offsets = flatten_with_offsets(
            neighbor_lists
        )

    @classmethod
    def from_neighbor_lists(
        cls,
        neighbor_lists: list,
        epsilon: float | None = None,
        N: int | None = None,
        D: int | None = None,
    ) -> "CutoffTable":
        """Create CutoffTable directly from precomputed neighbor lists.

        Useful for parameter optimization where neighbor lists are reused.
        Since this function is used inside optimize_epsilon, end users generally do not need to use this.

        Args:
            neighbor_lists: list of numpy arrays, precomputed neighbor lists for each data point
            epsilon: distance threshold used to create the neighbor lists
            N: number of data points
            D: dimensionality of data points

        Returns:
            CutoffTable: New instance created from the neighbor lists
        """
        instance = cls.__new__(cls)
        instance.epsilon = epsilon
        instance.N = N
        instance.D = D
        instance.L = np.mean([len(ids) for ids in neighbor_lists])
        instance.flattened_neighbors, instance.neighbor_offsets = flatten_with_offsets(
            neighbor_lists
        )
        return instance

    def __eq__(self, other) -> bool:
        """Check equality of two CutoffTable instances.

        Two CutoffTable instances are considered equal if they have:

        - Same epsilon value
        - Same dataset dimensions (``N``, ``D``)
        - Same average neighbor list length (``L``)
        - Same flattened neighbor lists and offsets

        Args:
            other: Another CutoffTable instance to compare with

        Returns:
            bool: True if the instances are equal, False otherwise
        """
        if not isinstance(other, CutoffTable):
            return False

        # Compare basic attributes
        if (
            not np.isclose(self.epsilon, other.epsilon)
            or self.N != other.N
            or self.D != other.D
            or not np.isclose(self.L, other.L)
        ):
            return False

        # Compare flattened neighbor arrays
        if not np.array_equal(self.flattened_neighbors, other.flattened_neighbors):
            return False

        # Compare neighbor offsets
        if not np.array_equal(self.neighbor_offsets, other.neighbor_offsets):
            return False

        return True

    def __repr__(self) -> str:
        """Return string representation of CutoffTable.

        Returns:
            str: Descriptive string with key attributes of the table
        """
        return (
            f"CutoffTable(epsilon={self.epsilon:.4f}, N={self.N}, D={self.D}, "
            f"L={self.L:.2f})"
        )

    def filter(
        self, dists: np.ndarray, ids: np.ndarray, final_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter search results to increase diversity (L3-L7 in Algorithm 2 in the paper).

        Takes candidate search results and removes similar items based on
        precomputed neighbor lists, selecting the most diverse ``final_k`` results.

        Args:
            dists: 2D array of candidate distances, shape (``Nq``, ``candidate_k``).
                Typically, this is the output of a search by faiss.
            ids: 2D array of candidate IDs, shape (``Nq``, ``candidate_k``).
                Typically, this is the output of a search by faiss.
            final_k: Number of diverse results to return

        Returns:
            tuple:
                - ``diverse_dists``: 2D array of diverse distances, shape (``Nq``, ``final_k``)
                - ``diverse_ids``: 2D array of diverse IDs, shape (``Nq``, ``final_k``)
        """
        # Input validation
        assert isinstance(final_k, int) and final_k > 0, (
            f"final_k must be a positive integer, got {final_k}"
        )
        assert dists.ndim == 2 and ids.ndim == 2, (
            f"dists and ids must be 2D arrays, got shapes {dists.shape}, {ids.shape}"
        )
        assert dists.shape == ids.shape, (
            f"Shape mismatch: distances shape {dists.shape} != ids shape {ids.shape}"
        )
        assert dists.shape[1] >= final_k, (
            f"Cannot select {final_k} results from only {dists.shape[1]} candidates"
        )
        assert dists.shape[0] > 0, "Cannot filter empty query results"

        Nq = dists.shape[0]

        # Allocate output arrays
        diverse_dists = np.empty((Nq, final_k), dtype=np.float32)
        diverse_ids = np.empty((Nq, final_k), dtype=np.int64)

        # Call C++ implementation for efficient filtering
        search_cpp(
            self.flattened_neighbors,
            self.neighbor_offsets,
            dists,
            ids,
            diverse_dists,
            diverse_ids,
        )
        return diverse_dists, diverse_ids


def optimize_epsilon(
    X: np.ndarray,
    Xq: np.ndarray,
    index,
    candidate_k: int,
    final_k: int,
    lam: float,
    epsilon_max: float | None = None,
    verbose: bool = True,
    num_iter: int = 5,
    batch_size: int = 1000,
    coarse_divisions: int = 10,
    fine_divisions: int = 100,
) -> dict:
    """Optimize the epsilon parameter for Cutoff Table.

    This function uses iterative grid search (bracketing) to find the optimal epsilon value that
    minimizes the combined diversity score (Sec. 6 and Sec. B). It performs a coarse-to-fine search,
    progressively narrowing the search range around the best found parameters.

    Args:
        X: Database vectors of shape (``N``, ``D``). This ``X`` can slightly differ from the ``X`` that the search actually runs on
        Xq: Query vectors of shape (``Nq``, ``D``). Since queries are usually unavailable during this hyperparameter search, one can use the training data or a subset of ``X``.
        index: Faiss index for similarity search
        candidate_k: Number of candidates to retrieve from Faiss
        final_k: Final number of diverse results to return
        lam: Balance parameter between similarity (``1-lam``) and diversity (``lam``)
        epsilon_max: Maximum epsilon to search. If None, computed automatically
        verbose: Whether to print progress information
        num_iter: Number of optimization iterations (coarse-to-fine refinement)
        batch_size: Batch size for neighbor list construction
        coarse_divisions: Number of grid points in non-final iterations
        fine_divisions: Number of grid points in the final iteration

    Returns:
        dict: Best parameters found with keys:
            - ``epsilon``: Optimal epsilon value
            - ``div_score``: Best diversity score achieved
            - ``L``: Number of cutoff table entries for optimal epsilon
    """
    # Step 1: Determine the search range for epsilon
    if epsilon_max is None:
        # Auto-compute epsilon_max as the mean of candidate_k-th distances
        dists, _ = index.search(Xq, candidate_k)
        valid_dists = clean_invalid_values(
            dists[:, -1]
        )  # Extract furthest candidate distances
        epsilon_max = np.mean(valid_dists)
    if verbose:
        print("epsilon_max:", epsilon_max)

    # Step 2: Pre-compute neighbor lists for the maximum epsilon
    # This contains all possible neighbors; we'll filter by smaller epsilons later
    neighbor_lists_max, neighbor_lists_max_dist = build_neighbor_lists(
        X=X,
        index=index,
        epsilon=epsilon_max,
        with_dist=True,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Step 3: Pre-compute candidate search results (constant across all epsilon values)
    candidate_dists, candidate_ids = index.search(Xq, candidate_k)

    # Step 4: Initialize optimization parameters
    a = 0.0  # Lower bound of search range. epsilon_left in the paper.
    b = epsilon_max  # Upper bound of search range. epsilon_right in the paper.
    r = b - a  # Current search range width. r in the paper.
    best_param = {
        "epsilon": epsilon_max,
        "div_score": float("inf"),
        "L": -1,
    }  # Track best parameters. epsilon_star and f_star in the paper.

    # Step 5: Iterative coarse-to-fine optimization
    for iteration in range(num_iter):
        if verbose:
            print(
                f"========= Epsilon optimization iteration {iteration + 1}/{num_iter}: range=[{a:.6f}, {b:.6f}] ========="
            )
        r /= 2  # Shrink the search range for next iteration

        # Use finer grid in the final iteration for better precision. W in the paper.
        num_divisions = (
            fine_divisions if iteration == num_iter - 1 else coarse_divisions
        )
        epsilons = np.linspace(a, b, num_divisions)  # \mathcal{E} in the paper.

        # Evaluate each epsilon candidate in the current range
        for n, epsilon in enumerate(epsilons):
            if verbose:
                print(f"=== Testing epsilon {n + 1}/{len(epsilons)}: {epsilon:.6f} ===")

            # Filter neighbor lists to only include neighbors within epsilon distance
            neighbor_lists_less_than_epsilon = [
                ids[dists < epsilon]
                for ids, dists in zip(neighbor_lists_max, neighbor_lists_max_dist)
            ]

            # Build cutoff table and apply diversity filtering
            ctable_less_than_epsilon = CutoffTable.from_neighbor_lists(
                neighbor_lists=neighbor_lists_less_than_epsilon
            )
            diverse_dists, diverse_ids = ctable_less_than_epsilon.filter(
                dists=candidate_dists, ids=candidate_ids, final_k=final_k
            )

            # Compute average diversity scores across all queries
            scores = np.array(
                [
                    div_score(diverse_dists[nq], diverse_ids[nq], X, lam)
                    for nq in range(Xq.shape[0])
                ]
            )
            combined_avg, similarity_avg, diversity_avg = np.mean(scores, axis=0)

            param = {
                "epsilon": epsilon,
                "div_score": combined_avg,
                "L": ctable_less_than_epsilon.L,
            }

            # Update best parameters if we found a better solution
            if param["div_score"] < best_param["div_score"]:
                if verbose:
                    print("Best parameters updated from:", best_param)
                    print("                        to:", param)
                    print(
                        f"Scores - Combined: {combined_avg:.6f}, Similarity: {similarity_avg:.6f}, Diversity: {diversity_avg:.6f}"
                    )
                best_param = param

        # Narrow the search range around the best epsilon found in this iteration
        a = max(best_param["epsilon"] - r, 0.0)  # Ensure lower bound >= 0.0
        b = min(
            best_param["epsilon"] + r, epsilon_max
        )  # Ensure upper bound <= epsilon_max

    return best_param
