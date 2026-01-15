"""Forward selection statistics for knockoff filter."""

import numpy as np
from .base import swap_columns, correct_for_swap, compute_signed_max_stat, standardize


def _forward_selection(
    X: np.ndarray,
    y: np.ndarray,
    omp: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Perform forward variable selection.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Response vector.
    omp : bool, default=False
        Whether to use orthogonal matching pursuit (OMP).

    Returns
    -------
    np.ndarray of shape (p,)
        Order in which variables were added to the model.
    """
    n, p = X.shape
    path = np.zeros(p, dtype=int)
    in_model = np.zeros(p, dtype=bool)
    residual = y.copy()

    if omp:
        Q = np.zeros((n, p))

    for step in range(p):
        # Find the best variable to add
        available_vars = np.where(~in_model)[0]

        if len(available_vars) == 0:
            break

        # Compute inner products with residual
        products = np.abs(X[:, available_vars].T @ residual)
        best_idx = np.argmax(products)
        best_var = available_vars[best_idx]

        path[step] = best_var
        in_model[best_var] = True

        # Update residual
        if step == p - 1:
            break

        x = X[:, best_var].copy()

        if omp:
            # Orthogonal matching pursuit: project onto span of all selected
            for j in range(step):
                x = x - (Q[:, j] @ x) * Q[:, j]
            norm_x = np.sqrt(np.sum(x ** 2))
            if norm_x > 1e-10:
                q = x / norm_x
                Q[:, step] = q
                residual = residual - (q @ y) * q
        else:
            # Standard forward selection
            norm_sq = np.sum(x ** 2)
            if norm_sq > 1e-10:
                residual = residual - (x @ residual) * x / norm_sq

    return path


def stat_forward_selection(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    omp: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Forward selection importance statistic.

    Computes W_j = max(Z_j, Z_{j+p}) * sign(Z_j - Z_{j+p}), where Z gives
    the reverse order in which variables enter the forward selection model.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Numeric response vector.
    omp : bool, default=False
        Whether to use orthogonal matching pursuit.

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.
    """
    X = np.asarray(X, dtype=np.float64)
    X_k = np.asarray(X_k, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(
            "stat_forward_selection requires numeric response y"
        )

    p = X.shape[1]

    # Standardize
    X = standardize(X)
    X_k = standardize(X_k)

    # Randomly swap columns
    X_swap, Xk_swap, swap = swap_columns(X, X_k)

    # Concatenate
    X_combined = np.hstack([X_swap, Xk_swap])

    # Run forward selection
    path = _forward_selection(X_combined, y, omp=omp)

    # Compute importance: reverse order of entry
    Z = np.zeros(2 * p)
    for i, var in enumerate(path):
        Z[var] = 2 * p + 1 - (i + 1)

    # Compute signed max statistic
    W = compute_signed_max_stat(Z, p)

    # Correct for swapping
    return correct_for_swap(W, swap)
