"""Lasso-based statistics for knockoff filter (Binary response)."""

import numpy as np
from .glmnet import stat_glmnet_lambdadiff, stat_glmnet_lambdasmax, stat_glmnet_coefdiff


def stat_lasso_lambdadiff_bin(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Logistic lasso lambda difference statistic for binary classification.

    Computes W_j = Z_j - Z_{j+p} where Z is the maximum lambda at which
    each variable enters the penalized logistic regression model.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Binary response vector (0/1 or factor with two levels).

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.

    Notes
    -----
    This is a wrapper around stat_glmnet_lambdadiff with family='binomial'.
    """
    return stat_glmnet_lambdadiff(X, X_k, y, family='binomial', **kwargs)


def stat_lasso_lambdasmax_bin(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Logistic lasso signed maximum lambda statistic.

    Computes W_j = max(Z_j, Z_{j+p}) * sign(Z_j - Z_{j+p}).

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Binary response vector.

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.

    Notes
    -----
    This is a wrapper around stat_glmnet_lambdasmax with family='binomial'.
    """
    return stat_glmnet_lambdasmax(X, X_k, y, family='binomial', **kwargs)


def stat_lasso_coefdiff_bin(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    cores: int = 2,
    **kwargs
) -> np.ndarray:
    """
    Logistic lasso coefficient difference statistic with cross-validation.

    Computes W_j = |Z_j| - |Z_{j+p}| where Z are coefficients at
    CV-selected lambda.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Binary response vector.
    cores : int, default=2
        Number of CPU cores for parallel CV.

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.

    Notes
    -----
    This is a wrapper around stat_glmnet_coefdiff with family='binomial'.
    """
    return stat_glmnet_coefdiff(X, X_k, y, family='binomial', cores=cores, **kwargs)
