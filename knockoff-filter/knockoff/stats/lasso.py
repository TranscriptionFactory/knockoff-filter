"""Lasso-based statistics for knockoff filter (Gaussian response)."""

import numpy as np
from .glmnet import stat_glmnet_lambdadiff, stat_glmnet_lambdasmax, stat_glmnet_coefdiff


def stat_lasso_lambdadiff(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Lasso lambda difference statistic for linear regression.

    Computes W_j = Z_j - Z_{j+p} where Z is the maximum lambda at which
    each variable enters the lasso model.

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
    This is a wrapper around stat_glmnet_lambdadiff with family='gaussian'.
    """
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(
            "stat_lasso_lambdadiff requires numeric response y"
        )
    y = y.ravel()

    return stat_glmnet_lambdadiff(X, X_k, y, family='gaussian', **kwargs)


def stat_lasso_lambdasmax(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Lasso signed maximum lambda statistic for linear regression.

    Computes W_j = max(Z_j, Z_{j+p}) * sign(Z_j - Z_{j+p}).

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
    This is a wrapper around stat_glmnet_lambdasmax with family='gaussian'.
    """
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(
            "stat_lasso_lambdasmax requires numeric response y"
        )
    y = y.ravel()

    return stat_glmnet_lambdasmax(X, X_k, y, family='gaussian', **kwargs)


def stat_lasso_coefdiff(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    cores: int = 2,
    **kwargs
) -> np.ndarray:
    """
    Lasso coefficient difference statistic with cross-validation.

    Computes W_j = |Z_j| - |Z_{j+p}| where Z are coefficients at
    CV-selected lambda.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Numeric response vector.
    cores : int, default=2
        Number of CPU cores for parallel CV.

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.

    Notes
    -----
    This is a wrapper around stat_glmnet_coefdiff with family='gaussian'.
    """
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(
            "stat_lasso_coefdiff requires numeric response y"
        )
    y = y.ravel()

    return stat_glmnet_coefdiff(X, X_k, y, family='gaussian', cores=cores, **kwargs)
