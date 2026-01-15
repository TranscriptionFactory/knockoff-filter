"""Utility functions for knockoff filter."""

from typing import Tuple, Any, Callable, Optional
import numpy as np
from scipy import linalg


def diag_pre_multiply(d: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Efficient computation of diag(d) @ X.

    Equivalent to R's `%diag*%` operator.

    Parameters
    ----------
    d : array-like of shape (n,)
        Diagonal elements.
    X : array-like of shape (n, m)
        Matrix to multiply.

    Returns
    -------
    np.ndarray
        Result of diag(d) @ X.
    """
    return d[:, np.newaxis] * X


def diag_post_multiply(X: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Efficient computation of X @ diag(d).

    Equivalent to R's `%*diag%` operator.

    Parameters
    ----------
    X : array-like of shape (n, m)
        Matrix to multiply.
    d : array-like of shape (m,)
        Diagonal elements.

    Returns
    -------
    np.ndarray
        Result of X @ diag(d).
    """
    return X * d


def is_posdef(A: np.ndarray, tol: float = 1e-9, **kwargs) -> bool:
    """
    Efficient test for matrix positive-definiteness.

    Computes the smallest eigenvalue of a matrix A to verify whether
    A is positive-definite.

    Parameters
    ----------
    A : array-like of shape (n, n)
        Symmetric matrix to test.
    tol : float, default=1e-9
        Tolerance for eigenvalue positivity.

    Returns
    -------
    bool
        True if A is positive-definite.
    """
    A = np.asarray(A)
    p = A.shape[0]

    if p < 500:
        # Use dense eigenvalue solver for small matrices
        lambda_min = np.min(linalg.eigvalsh(A))
    else:
        # Use sparse eigenvalue solver for large matrices
        try:
            from scipy.sparse.linalg import eigsh
            # Request smallest algebraic eigenvalue
            lambda_min = eigsh(A, k=1, which='SA', return_eigenvectors=False)[0]
        except Exception:
            # Fallback to dense computation
            lambda_min = np.min(linalg.eigvalsh(A))

    return lambda_min > tol * 10


def canonical_svd(X: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduced SVD with canonical sign choice.

    Convention: the sign of each vector in U is chosen such that the
    coefficient with the largest absolute value is positive.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Matrix to decompose.

    Returns
    -------
    u : np.ndarray of shape (n, min(n, p))
        Left singular vectors.
    d : np.ndarray of shape (min(n, p),)
        Singular values.
    v : np.ndarray of shape (p, min(n, p))
        Right singular vectors.
    """
    try:
        U, d, Vt = linalg.svd(X, full_matrices=False)
    except Exception as e:
        raise RuntimeError(
            "SVD failed in the creation of fixed-design knockoffs. "
            f"Error: {e}"
        )

    # Canonical sign choice: largest |u| element positive
    for j in range(U.shape[1]):
        i = np.argmax(np.abs(U[:, j]))
        if U[i, j] < 0:
            U[:, j] = -U[:, j]
            Vt[j, :] = -Vt[j, :]

    return U, d, Vt.T


def normc(X: np.ndarray, center: bool = True, **kwargs) -> np.ndarray:
    """
    Scale the columns of a matrix to have unit norm.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Matrix to normalize.
    center : bool, default=True
        Whether to center columns before scaling.

    Returns
    -------
    np.ndarray of shape (n, p)
        Matrix with unit-norm columns.
    """
    X = np.asarray(X, dtype=np.float64)

    if center:
        X = X - X.mean(axis=0)

    norms = np.sqrt(np.sum(X ** 2, axis=0))
    # Avoid division by zero
    norms[norms == 0] = 1.0

    return X / norms


def cov2cor(Sigma: np.ndarray, **kwargs) -> np.ndarray:
    """
    Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    Sigma : array-like of shape (p, p)
        Covariance matrix.

    Returns
    -------
    np.ndarray of shape (p, p)
        Correlation matrix.
    """
    Sigma = np.asarray(Sigma)
    d = np.sqrt(np.diag(Sigma))
    # Avoid division by zero
    d[d == 0] = 1.0
    return Sigma / np.outer(d, d)


def rnorm_matrix(n: int, p: int, mean: float = 0.0, sd: float = 1.0, **kwargs) -> np.ndarray:
    """
    Generate a random matrix with normally distributed entries.

    Parameters
    ----------
    n : int
        Number of rows.
    p : int
        Number of columns.
    mean : float, default=0.0
        Mean of the distribution.
    sd : float, default=1.0
        Standard deviation of the distribution.

    Returns
    -------
    np.ndarray of shape (n, p)
        Random matrix.
    """
    return np.random.normal(mean, sd, size=(n, p))


def random_problem(
    n: int,
    p: int,
    k: Optional[int] = None,
    amplitude: float = 3.0,
    seed: Optional[int] = None,
    **kwargs
) -> dict:
    """
    Generate a random, sparse regression problem.

    Parameters
    ----------
    n : int
        Number of samples.
    p : int
        Number of features.
    k : int, optional
        Number of nonzero coefficients. Default is max(1, p // 5).
    amplitude : float, default=3.0
        Amplitude of nonzero coefficients.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - 'X': Feature matrix of shape (n, p)
        - 'y': Response vector of shape (n,)
        - 'beta': True coefficients of shape (p,)
        - 'nonzero': Indices of nonzero coefficients
    """
    if seed is not None:
        np.random.seed(seed)

    if k is None:
        k = max(1, p // 5)

    X = normc(rnorm_matrix(n, p))
    nonzero = np.random.choice(p, k, replace=False)
    beta = amplitude * np.isin(np.arange(p), nonzero).astype(float)
    y = X @ beta + np.random.randn(n)

    return {
        'X': X,
        'y': y,
        'beta': beta,
        'nonzero': nonzero
    }


def with_seed(seed: int, func: Callable[[], Any], **kwargs) -> Any:
    """
    Execute a function with a fixed random seed, then restore state.

    Parameters
    ----------
    seed : int
        Random seed to use.
    func : callable
        Function to execute.

    Returns
    -------
    Any
        Return value of func().
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return func()
    finally:
        np.random.set_state(state)
