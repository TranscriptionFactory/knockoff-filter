"""Optimization solvers for knockoff construction."""

from typing import List, Tuple, Optional
import warnings
import numpy as np
from scipy import linalg
from scipy.cluster.hierarchy import linkage, fcluster

from .utils import is_posdef, cov2cor


def create_solve_equi(Sigma: np.ndarray, **kwargs) -> np.ndarray:
    """
    Optimization for equi-correlated knockoffs.

    Computes the closed-form solution to the semidefinite programming problem:
        maximize    s
        subject to  0 <= s <= 1
                    2*Sigma - s*I >= 0

    The closed-form solution is s = min(2 * lambda_min(Sigma), 1).

    Parameters
    ----------
    Sigma : array-like of shape (p, p)
        Positive-definite covariance matrix.

    Returns
    -------
    np.ndarray of shape (p,)
        The solution s to the optimization problem.
    """
    Sigma = np.asarray(Sigma)

    # Check symmetry
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Covariance matrix must be symmetric")

    p = Sigma.shape[0]
    tol = 1e-10

    # Convert to correlation matrix
    G = cov2cor(Sigma)

    # Check positive-definiteness
    if not is_posdef(G):
        raise ValueError("The covariance matrix is not positive-definite: cannot solve SDP")

    # Compute smallest eigenvalue
    if p > 2:
        try:
            from scipy.sparse.linalg import eigsh
            # Use sparse eigenvalue solver for efficiency
            maxiter = 100000
            lambda_min = eigsh(
                G, k=1, which='SA',
                return_eigenvectors=False,
                maxiter=maxiter,
                tol=1e-8
            )[0]
        except Exception:
            # Fallback to dense solver
            warnings.warn(
                "Sparse eigenvalue solver did not converge. "
                "Using dense computation instead."
            )
            eigenvalues = linalg.eigvalsh(G)
            lambda_min = eigenvalues[0]  # Already sorted ascending
    else:
        eigenvalues = linalg.eigvalsh(G)
        lambda_min = eigenvalues[0]

    if lambda_min < 0:
        raise ValueError(
            "The covariance matrix is not positive-definite. "
            "Cannot create equi-correlated knockoffs."
        )

    # Closed-form solution
    s = np.ones(p) * min(2 * lambda_min, 1)

    # Compensate for numerical errors (feasibility)
    s_eps = 1e-8
    while s_eps <= 0.1:
        test_matrix = 2 * G - np.diag(s * (1 - s_eps))
        if is_posdef(test_matrix):
            break
        s_eps *= 10

    s = s * (1 - s_eps)

    # Scale back for original covariance matrix
    return s * np.diag(Sigma)


def create_solve_sdp(
    Sigma: np.ndarray,
    gaptol: float = 1e-6,
    maxit: int = 1000,
    verbose: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Full SDP optimization for knockoffs.

    Solves the semidefinite programming problem:
        maximize    sum(s)
        subject to  0 <= s <= 1
                    2*Sigma - diag(s) >= 0 (positive semidefinite)

    Parameters
    ----------
    Sigma : array-like of shape (p, p)
        Positive-definite covariance matrix.
    gaptol : float, default=1e-6
        Tolerance for duality gap.
    maxit : int, default=1000
        Maximum number of iterations for the solver.
    verbose : bool, default=False
        Whether to display progress.

    Returns
    -------
    np.ndarray of shape (p,)
        The solution s to the optimization problem.
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError(
            "cvxpy is required for SDP solving. "
            "Install with: pip install cvxpy"
        )

    Sigma = np.asarray(Sigma)

    # Check symmetry
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Covariance matrix must be symmetric")

    # Convert to correlation matrix
    G = cov2cor(Sigma)
    p = G.shape[0]

    # Check positive-definiteness
    if not is_posdef(G):
        warnings.warn(
            "The covariance matrix is not positive-definite: "
            "knockoffs may not have power."
        )

    if verbose:
        print("Solving SDP ... ", end="", flush=True)

    # Define the optimization problem using CVXPY
    s = cp.Variable(p)

    # Objective: maximize sum(s)
    objective = cp.Maximize(cp.sum(s))

    # Constraints
    constraints = [
        s >= 0,
        s <= 1,
        2 * G - cp.diag(s) >> 0  # Positive semidefinite constraint
    ]

    problem = cp.Problem(objective, constraints)

    # Solve using SCS solver
    try:
        problem.solve(
            solver=cp.SCS,
            max_iters=maxit,
            eps=gaptol,
            verbose=False
        )
    except Exception as e:
        warnings.warn(f"SDP solver failed with error: {e}")
        return np.zeros(p) * np.diag(Sigma)

    if verbose:
        print("done.")

    # Check solution status
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        warnings.warn(
            f"The SDP solver returned status '{problem.status}'. "
            "Knockoffs may lose power."
        )

    # Extract and clip solution
    s_val = s.value
    if s_val is None:
        warnings.warn("SDP solver returned None. Knockoffs will have no power.")
        return np.zeros(p) * np.diag(Sigma)

    s_val = np.clip(s_val, 0, 1)

    # Compensate for numerical errors (feasibility)
    if verbose:
        print("Verifying that the solution is correct ... ", end="", flush=True)

    s_eps = 1e-8
    while s_eps <= 0.1:
        test_matrix = 2 * G - np.diag(s_val * (1 - s_eps))
        if is_posdef(test_matrix, tol=1e-9):
            break
        s_eps *= 10

    s_val = s_val * (1 - s_eps)
    s_val = np.clip(s_val, 0, 1)

    if verbose:
        print("done.")

    # Verify solution
    if np.all(s_val == 0):
        warnings.warn(
            "In creation of SDP knockoffs, procedure failed. "
            "Knockoffs will have no power."
        )

    # Scale back for original covariance matrix
    return s_val * np.diag(Sigma)


def _merge_clusters(clusters: np.ndarray, max_size: int) -> np.ndarray:
    """
    Merge consecutive clusters while ensuring no cluster exceeds max_size.

    Parameters
    ----------
    clusters : array-like of shape (p,)
        Cluster assignments (1-indexed).
    max_size : int
        Maximum cluster size.

    Returns
    -------
    np.ndarray of shape (p,)
        New cluster assignments (1-indexed).
    """
    unique_clusters = np.unique(clusters)
    cluster_sizes = {k: np.sum(clusters == k) for k in unique_clusters}

    clusters_new = np.zeros_like(clusters)
    g = 1
    g_size = 0

    for k in sorted(unique_clusters):
        if g_size + cluster_sizes[k] > max_size:
            g += 1
            g_size = 0
        clusters_new[clusters == k] = g
        g_size += cluster_sizes[k]

    return clusters_new


def _divide_sdp(
    Sigma: np.ndarray,
    max_size: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Approximate a covariance matrix by a block diagonal matrix.

    Uses hierarchical clustering to divide variables into groups.

    Parameters
    ----------
    Sigma : array-like of shape (p, p)
        Covariance matrix.
    max_size : int
        Maximum block size.

    Returns
    -------
    clusters : np.ndarray of shape (p,)
        Cluster assignments (1-indexed).
    sub_sigmas : list of np.ndarray
        List of covariance sub-matrices for each cluster.
    """
    p = Sigma.shape[0]

    # Convert to dissimilarity matrix with small perturbation
    eps = np.random.randn(p, p) * 1e-6
    G = cov2cor(Sigma)
    dissimilarity = 1 - np.abs(G + eps)

    # Make symmetric and extract condensed form for linkage
    dissimilarity = (dissimilarity + dissimilarity.T) / 2
    np.fill_diagonal(dissimilarity, 0)

    # Convert to condensed distance matrix
    from scipy.spatial.distance import squareform
    distance = squareform(dissimilarity)

    # Hierarchical clustering with single linkage
    Z = linkage(distance, method='single')

    # Binary search for optimal number of clusters
    n_blocks_min = 1
    n_blocks_max = p

    for _ in range(100):
        n_blocks = (n_blocks_min + n_blocks_max + 1) // 2
        clusters = fcluster(Z, n_blocks, criterion='maxclust')
        max_cluster_size = max(np.bincount(clusters)[1:]) if np.max(clusters) > 0 else p

        if max_cluster_size <= max_size:
            n_blocks_max = n_blocks
        if max_cluster_size >= max_size:
            n_blocks_min = n_blocks
        if n_blocks_min == n_blocks_max:
            break

    # Merge small clusters
    clusters_new = _merge_clusters(clusters, max_size)
    while not np.array_equal(clusters_new, clusters):
        clusters = clusters_new
        clusters_new = _merge_clusters(clusters, max_size)
    clusters = clusters_new

    # Create covariance sub-matrices
    n_clusters = int(np.max(clusters))
    sub_sigmas = []
    for k in range(1, n_clusters + 1):
        indices = np.where(clusters == k)[0]
        sub_sigmas.append(Sigma[np.ix_(indices, indices)])

    return clusters, sub_sigmas


def create_solve_asdp(
    Sigma: np.ndarray,
    max_size: int = 500,
    gaptol: float = 1e-6,
    maxit: int = 1000,
    verbose: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Approximate SDP optimization for knockoffs.

    For high-dimensional problems (p > max_size), this function approximates
    the covariance matrix as block-diagonal and solves smaller SDPs.

    Parameters
    ----------
    Sigma : array-like of shape (p, p)
        Positive-definite covariance matrix.
    max_size : int, default=500
        Maximum size of each block.
    gaptol : float, default=1e-6
        Tolerance for duality gap.
    maxit : int, default=1000
        Maximum number of iterations.
    verbose : bool, default=False
        Whether to display progress.

    Returns
    -------
    np.ndarray of shape (p,)
        The solution s to the optimization problem.
    """
    Sigma = np.asarray(Sigma)

    # Check symmetry
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Covariance matrix must be symmetric")

    p = Sigma.shape[0]

    # If small enough, use full SDP
    if p <= max_size:
        return create_solve_sdp(Sigma, gaptol=gaptol, maxit=maxit, verbose=verbose)

    if verbose:
        print(f"Dividing the problem into subproblems of size <= {max_size} ... ", end="", flush=True)

    # Divide into blocks
    clusters, sub_sigmas = _divide_sdp(Sigma, max_size)
    n_blocks = int(np.max(clusters))

    if verbose:
        print("done.")
        print(f"Solving {n_blocks} smaller SDPs ... ")

    # Solve SDPs for each block
    s_list = []
    for k in range(n_blocks):
        s_k = create_solve_sdp(sub_sigmas[k], gaptol=gaptol, maxit=maxit, verbose=False)
        s_list.append(s_k)
        if verbose:
            print(f"  Block {k + 1}/{n_blocks} done")

    # Assemble solutions
    idx_count = np.zeros(n_blocks, dtype=int)
    s_asdp = np.zeros(p)

    for j in range(p):
        cluster_j = clusters[j] - 1  # Convert to 0-indexed
        s_asdp[j] = s_list[cluster_j][idx_count[cluster_j]]
        idx_count[cluster_j] += 1

    # Maximize shrinkage factor via binary search
    if verbose:
        print(f"Combining the solutions of the {n_blocks} smaller SDPs ... ", end="", flush=True)

    gamma_range = np.linspace(0, 1, 1000)

    def check_psd(gamma_idx: int) -> float:
        """Check if 2*Sigma - gamma*diag(s) is positive definite."""
        gamma = gamma_range[gamma_idx]
        G = 2 * Sigma - gamma * np.diag(s_asdp)
        try:
            from scipy.sparse.linalg import eigsh
            lambda_min = eigsh(
                G, k=1, which='SA',
                return_eigenvectors=False,
                maxiter=100000,
                tol=1e-12
            )[0]
        except Exception:
            lambda_min = np.min(linalg.eigvalsh(G))
        return lambda_min

    # Binary search for optimal gamma
    low, high = 0, len(gamma_range) - 1
    while low < high:
        mid = (low + high + 1) // 2
        if check_psd(mid) > 0:
            low = mid
        else:
            high = mid - 1

    gamma_opt = gamma_range[low]
    s_asdp_scaled = gamma_opt * s_asdp

    if verbose:
        print("done.")
        print("Verifying that the solution is correct ... ", end="", flush=True)

    # Verify solution
    if not is_posdef(2 * Sigma - np.diag(s_asdp_scaled)):
        warnings.warn(
            "In creation of approximate SDP knockoffs, procedure failed. "
            "Knockoffs will have no power."
        )
        s_asdp_scaled = np.zeros(p)

    if verbose:
        print("done.")

    return s_asdp_scaled
