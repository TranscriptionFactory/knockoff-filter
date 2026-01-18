"""Square-root lasso statistics for knockoff filter."""

import warnings
import numpy as np
from .base import swap_columns, correct_for_swap


def _sqrt_lasso_path(
    X: np.ndarray,
    y: np.ndarray,
    nlambda: int = 100,
    **kwargs
) -> np.ndarray:
    """
    Compute the square-root lasso regularization path.

    The square-root lasso solves:
        minimize ||y - X @ beta|| + lambda * ||beta||_1

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Response vector.
    nlambda : int, default=100
        Number of lambda values.

    Returns
    -------
    np.ndarray of shape (p,)
        Maximum lambda at which each variable enters the model.
    """
    n, p = X.shape

    # Standardize
    X = X - X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = X / std

    y = y - y.mean()

    # For sqrt-lasso, the optimal lambda is approximately sqrt(n) * quantile
    # We'll approximate by using regular lasso with scaled lambdas
    try:
        from sklearn.linear_model import lasso_path

        # Generate lambda sequence
        # For sqrt-lasso, lambda_max ~ max|X'y| / (n * ||y||)
        lambda_max = np.max(np.abs(X.T @ y)) / (np.sqrt(n) * np.linalg.norm(y))
        lambda_min = lambda_max / 1000
        k = np.arange(nlambda) / nlambda
        lambdas = lambda_max * (lambda_min / lambda_max) ** k

        # Use standard lasso as approximation
        # True sqrt-lasso would require custom solver
        alphas, coefs, _ = lasso_path(X, y, alphas=lambdas, max_iter=10000)

        # Find entry lambda for each variable
        lambda_entry = np.zeros(p)
        for j in range(p):
            nonzero_idx = np.where(np.abs(coefs[j, :]) > 0)[0]
            if len(nonzero_idx) > 0:
                lambda_entry[j] = alphas[nonzero_idx[0]] * np.sqrt(n)

        return lambda_entry

    except Exception as e:
        warnings.warn(f"Square-root lasso failed: {e}. Using regular lasso.")
        from sklearn.linear_model import lasso_path

        lambda_max = np.max(np.abs(X.T @ y)) / n
        lambda_min = lambda_max / 1000
        k = np.arange(nlambda) / nlambda
        lambdas = lambda_max * (lambda_min / lambda_max) ** k

        alphas, coefs, _ = lasso_path(X, y, alphas=lambdas, max_iter=10000)

        lambda_entry = np.zeros(p)
        for j in range(p):
            nonzero_idx = np.where(np.abs(coefs[j, :]) > 0)[0]
            if len(nonzero_idx) > 0:
                lambda_entry[j] = alphas[nonzero_idx[0]] * n

        return lambda_entry


def stat_sqrt_lasso(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Square-root lasso statistic.

    Computes W_j = Z_j - Z_{j+p} where Z is the maximum lambda at which
    each variable enters the square-root lasso model.

    The square-root lasso is a variant of lasso that is scale-invariant.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Numeric response vector.

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.

    Notes
    -----
    This implementation uses an approximation based on regular lasso.
    For a true square-root lasso, a specialized solver would be needed.
    """
    X = np.asarray(X, dtype=np.float64)
    X_k = np.asarray(X_k, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("stat_sqrt_lasso requires numeric response y")

    p = X.shape[1]

    # Randomly swap columns
    X_swap, Xk_swap, swap = swap_columns(X, X_k)

    # Concatenate
    X_combined = np.hstack([X_swap, Xk_swap])

    # Compute sqrt-lasso path
    Z = _sqrt_lasso_path(X_combined, y, **kwargs)

    # Compute difference statistic
    orig = np.arange(p)
    W = Z[orig] - Z[orig + p]

    # Correct for swapping
    return correct_for_swap(W, swap)
