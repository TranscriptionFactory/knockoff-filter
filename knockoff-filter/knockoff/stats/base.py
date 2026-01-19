"""Base utilities for knockoff statistics."""

from typing import Tuple
import numpy as np


def swap_columns(
    X: np.ndarray,
    X_k: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly swap columns between X and X_k for symmetry.

    This is a key step in knockoff statistics: by randomly swapping
    columns, we ensure that the statistics are symmetric with respect
    to the original and knockoff variables.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.

    Returns
    -------
    X_swap : np.ndarray of shape (n, p)
        Swapped original variables.
    Xk_swap : np.ndarray of shape (n, p)
        Swapped knockoff variables.
    swap : np.ndarray of shape (p,)
        Binary indicator of which columns were swapped.
    """
    X = np.asarray(X)
    X_k = np.asarray(X_k)

    n, p = X.shape

    # Generate random swap indicators
    swap = np.random.binomial(1, 0.5, p)

    # Perform swaps using broadcasting (swap shape (p,) broadcasts with (n, p))
    swap_bool = swap.astype(bool)
    X_swap = np.where(swap_bool, X_k, X)
    Xk_swap = np.where(swap_bool, X, X_k)

    return X_swap, Xk_swap, swap


def correct_for_swap(W: np.ndarray, swap: np.ndarray, **kwargs) -> np.ndarray:
    """
    Correct statistics for column swapping.

    After computing statistics on swapped columns, we need to flip
    the sign for statistics corresponding to swapped columns.

    Parameters
    ----------
    W : array-like of shape (p,)
        Uncorrected statistics.
    swap : array-like of shape (p,)
        Binary indicator of which columns were swapped.

    Returns
    -------
    np.ndarray of shape (p,)
        Corrected statistics.
    """
    return W * (1 - 2 * swap)


def compute_difference_stat(
    Z: np.ndarray,
    p: int,
    **kwargs
) -> np.ndarray:
    """
    Compute difference statistic W_j = Z_j - Z_{j+p}.

    Parameters
    ----------
    Z : array-like of shape (2*p,)
        Importance scores for original and knockoff variables.
    p : int
        Number of original variables.

    Returns
    -------
    np.ndarray of shape (p,)
        Difference statistics.
    """
    orig = np.arange(p)
    return Z[orig] - Z[orig + p]


def compute_signed_max_stat(
    Z: np.ndarray,
    p: int,
    **kwargs
) -> np.ndarray:
    """
    Compute signed maximum statistic.

    W_j = max(Z_j, Z_{j+p}) * sign(Z_j - Z_{j+p})

    Parameters
    ----------
    Z : array-like of shape (2*p,)
        Importance scores for original and knockoff variables.
    p : int
        Number of original variables.

    Returns
    -------
    np.ndarray of shape (p,)
        Signed maximum statistics.
    """
    orig = np.arange(p)
    Z_orig = Z[orig]
    Z_knock = Z[orig + p]

    W = np.maximum(Z_orig, Z_knock) * np.sign(Z_orig - Z_knock)
    return W


def standardize(X: np.ndarray, **kwargs) -> np.ndarray:
    """
    Standardize columns to have zero mean and unit variance.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Matrix to standardize.

    Returns
    -------
    np.ndarray of shape (n, p)
        Standardized matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return X / std
