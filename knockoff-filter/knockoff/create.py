"""Knockoff construction methods."""

from dataclasses import dataclass
from typing import Optional, Union, Callable
import warnings
import numpy as np
from scipy import linalg

from .utils import (
    normc, canonical_svd, rnorm_matrix, with_seed,
    is_posdef, diag_pre_multiply, diag_post_multiply
)
from .solve import create_solve_equi, create_solve_sdp, create_solve_asdp


@dataclass
class KnockoffVariables:
    """
    Container for knockoff variables.

    Attributes
    ----------
    X : np.ndarray
        Original variables (possibly augmented or transformed).
    Xk : np.ndarray
        Knockoff variables.
    y : np.ndarray, optional
        Response variables (possibly augmented).
    """
    X: np.ndarray
    Xk: np.ndarray
    y: Optional[np.ndarray] = None


def _decompose(X: np.ndarray, randomize: bool = False) -> dict:
    """
    Compute the SVD of X and construct orthogonal matrix U_perp.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Data matrix where n >= 2*p.
    randomize : bool, default=False
        Whether to randomize the orthogonal complement.

    Returns
    -------
    dict
        Dictionary with keys 'u', 'd', 'v', 'u_perp'.
    """
    n, p = X.shape
    if n < 2 * p:
        raise ValueError(f"Matrix must have n >= 2*p. Got n={n}, p={p}")

    u, d, v = canonical_svd(X)

    # Construct U_perp: orthogonal to columns of u
    Q, _ = linalg.qr(np.hstack([u, np.zeros((n, p))]))
    u_perp = Q[:, p:2*p]

    if randomize:
        # Random orthogonal matrix
        Q_rand, _ = linalg.qr(rnorm_matrix(p, p))
        u_perp = u_perp @ Q_rand

    return {'u': u, 'd': d, 'v': v, 'u_perp': u_perp}


def _create_equicorrelated(X: np.ndarray, randomize: bool = False) -> np.ndarray:
    """
    Create equicorrelated fixed-X knockoffs.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Normalized data matrix.
    randomize : bool, default=False
        Whether to randomize knockoff construction.

    Returns
    -------
    np.ndarray of shape (n, p)
        Knockoff matrix.
    """
    # Compute SVD and U_perp
    svd_result = _decompose(X, randomize)
    u = svd_result['u']
    d = svd_result['d']
    v = svd_result['v']
    u_perp = svd_result['u_perp']

    # Check rank
    if np.any(d <= 1e-5 * np.max(d)):
        raise ValueError(
            "Data matrix is rank deficient. "
            "Equicorrelated knockoffs will have no power."
        )

    # Set s = min(2 * smallest eigenvalue of X'X, 1)
    lambda_min = np.min(d) ** 2
    s = min(2 * lambda_min, 1)

    # Construct the knockoff according to Equation 1.4
    s_diff = np.maximum(0, 2 * s - (s / d) ** 2)

    # X_ko = u @ diag(d - s/d) @ v.T + u_perp @ diag(sqrt(s_diff)) @ v.T
    X_ko = (diag_post_multiply(u, d - s / d) @ v.T +
            diag_post_multiply(u_perp, np.sqrt(s_diff)) @ v.T)

    return X_ko


def _create_sdp(X: np.ndarray, randomize: bool = False) -> np.ndarray:
    """
    Create SDP fixed-X knockoffs.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Normalized data matrix.
    randomize : bool, default=False
        Whether to randomize knockoff construction.

    Returns
    -------
    np.ndarray of shape (n, p)
        Knockoff matrix.
    """
    # Compute SVD and U_perp
    svd_result = _decompose(X, randomize)
    u = svd_result['u']
    d = svd_result['d']
    v = svd_result['v']
    u_perp = svd_result['u_perp']

    # Check for rank deficiency
    tol = 1e-5
    d_inv = 1.0 / d
    d_zeros = d <= tol * np.max(d)
    if np.any(d_zeros):
        warnings.warn(
            "Data matrix is rank deficient. "
            "Model is not identifiable, but proceeding with SDP knockoffs"
        )
        d_inv[d_zeros] = 0

    # Compute the Gram matrix and its (pseudo)inverse
    G = diag_post_multiply(v, d ** 2) @ v.T
    G_inv = diag_post_multiply(v, d_inv ** 2) @ v.T

    # Optimize the parameter s using SDP
    s = create_solve_sdp(G)
    s[s <= tol] = 0

    # Construct the knockoff according to Equation 1.4
    # C = 2*diag(s) - s %diag*% G_inv %*diag% s
    C = 2 * np.diag(s) - diag_post_multiply(diag_pre_multiply(s, G_inv), s)
    C_svd = canonical_svd(C)

    # X_ko = X - X @ G_inv @ diag(s) + u_perp @ diag(sqrt(max(0, C.d))) @ C.v.T
    X_ko = (X - diag_post_multiply(X @ G_inv, s) +
            diag_post_multiply(u_perp, np.sqrt(np.maximum(0, C_svd[1]))) @ C_svd[2].T)

    return X_ko


def create_fixed(
    X: np.ndarray,
    method: str = 'sdp',
    sigma: Optional[float] = None,
    y: Optional[np.ndarray] = None,
    randomize: bool = False,
    **kwargs
) -> KnockoffVariables:
    """
    Create fixed-X knockoff variables.

    Fixed-X knockoffs assume a homoscedastic linear regression model for Y|X.
    They guarantee FDR control when used with statistics satisfying the
    "sufficiency" property.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Normalized matrix of original variables (n > p).
    method : {'sdp', 'equi'}, default='sdp'
        Method for minimizing correlation between original and knockoffs.
    sigma : float, optional
        Noise level for data augmentation when p <= n < 2p.
    y : array-like of shape (n,), optional
        Response variables. Required if n < 2*p and sigma not provided.
    randomize : bool, default=False
        Whether to construct knockoffs randomly or deterministically.

    Returns
    -------
    KnockoffVariables
        Object containing X, Xk, and y (possibly augmented).

    References
    ----------
    Barber and Candes,
    Controlling the false discovery rate via knockoffs.
    Ann. Statist. 43 (2015), no. 5, 2055--2085.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape

    if method not in ['sdp', 'equi']:
        raise ValueError(f"method must be 'sdp' or 'equi', got '{method}'")

    if n <= p:
        raise ValueError(f"Input X must have dimensions n > p. Got n={n}, p={p}")

    y_out = y

    if n < 2 * p:
        warnings.warn(
            f"Input X has dimensions p < n < 2p (n={n}, p={p}). "
            "Augmenting the model with extra rows."
        )

        # SVD to get orthogonal complement
        U, _, _ = linalg.svd(X, full_matrices=True)
        u2 = U[:, p:n]

        # Augment X with zero rows
        X = np.vstack([X, np.zeros((2 * p - n, p))])

        if sigma is None:
            if y is None:
                raise ValueError(
                    "Either 'sigma' or 'y' must be provided to augment "
                    "the data with extra rows when p <= n < 2p."
                )
            else:
                y = np.asarray(y, dtype=np.float64)
                sigma = np.sqrt(np.mean((u2.T @ y) ** 2))

        # Augment y
        if randomize:
            y_extra = np.random.normal(0, sigma, size=2 * p - n)
        else:
            y_extra = with_seed(0, lambda: np.random.normal(0, sigma, size=2 * p - n))

        if y is not None:
            y_out = np.concatenate([y, y_extra])
        else:
            y_out = y_extra

    # Normalize X columns to unit norm
    X = normc(X, center=False)

    # Create knockoffs
    if method == 'equi':
        Xk = _create_equicorrelated(X, randomize)
    else:
        Xk = _create_sdp(X, randomize)

    return KnockoffVariables(X=X, Xk=Xk, y=y_out)


def create_gaussian(
    X: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    method: str = 'asdp',
    diag_s: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """
    Sample multivariate Gaussian model-X knockoff variables.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Matrix of original variables.
    mu : array-like of shape (p,)
        Mean parameter of the Gaussian model for X.
    Sigma : array-like of shape (p, p)
        Covariance matrix for the Gaussian model of X.
    method : {'asdp', 'sdp', 'equi'}, default='asdp'
        Method for minimizing correlation between original and knockoffs.
    diag_s : array-like of shape (p,), optional
        Pre-computed covariances between original variables and knockoffs.

    Returns
    -------
    np.ndarray of shape (n, p)
        Matrix of knockoff variables.

    References
    ----------
    Candes et al., Panning for Gold: Model-free Knockoffs for
    High-dimensional Controlled Variable Selection, arXiv:1610.02351 (2016).
    """
    X = np.asarray(X, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    n, p = X.shape

    if method not in ['asdp', 'sdp', 'equi']:
        raise ValueError(f"method must be 'asdp', 'sdp', or 'equi', got '{method}'")

    # Do not use ASDP unless p > 500
    if p <= 500 and method == 'asdp':
        method = 'sdp'

    # Compute diag_s if not provided
    if diag_s is None:
        if method == 'equi':
            diag_s = create_solve_equi(Sigma)
        elif method == 'sdp':
            diag_s = create_solve_sdp(Sigma)
        else:  # asdp
            diag_s = create_solve_asdp(Sigma)

    diag_s = np.asarray(diag_s)

    # Ensure diag_s is a 1D array
    if diag_s.ndim == 2:
        diag_s = np.diag(diag_s)

    # If diag_s is zero, we can only generate trivial knockoffs
    if np.all(diag_s == 0):
        warnings.warn(
            "The conditional knockoff covariance matrix is not positive definite. "
            "Knockoffs will have no power."
        )
        return X.copy()

    # Compute knockoff distribution parameters
    # SigmaInv_s = Sigma^{-1} @ diag(s)
    diag_s_matrix = np.diag(diag_s)
    SigmaInv_s = linalg.solve(Sigma, diag_s_matrix)

    # mu_k = X - (X - mu) @ SigmaInv_s
    mu_k = X - (X - mu) @ SigmaInv_s

    # Sigma_k = 2*diag(s) - diag(s) @ SigmaInv_s
    Sigma_k = 2 * diag_s_matrix - diag_s_matrix @ SigmaInv_s

    # Ensure Sigma_k is positive definite (may need small adjustment)
    try:
        L = linalg.cholesky(Sigma_k, lower=True)
    except linalg.LinAlgError:
        # Add small regularization
        eps = 1e-10
        while eps < 1:
            try:
                L = linalg.cholesky(Sigma_k + eps * np.eye(p), lower=True)
                break
            except linalg.LinAlgError:
                eps *= 10
        else:
            warnings.warn(
                "Could not compute Cholesky decomposition of knockoff covariance. "
                "Knockoffs will have no power."
            )
            return X.copy()

    # Sample knockoffs: X_k = mu_k + randn(n, p) @ L.T
    X_k = mu_k + np.random.randn(n, p) @ L.T

    return X_k


def create_second_order(
    X: np.ndarray,
    method: str = 'asdp',
    shrink: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Sample second-order multivariate Gaussian knockoff variables.

    First fits a multivariate Gaussian distribution to X, then generates
    Gaussian knockoffs according to the estimated model.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Matrix of original variables.
    method : {'asdp', 'sdp', 'equi'}, default='asdp'
        Method for minimizing correlation between original and knockoffs.
    shrink : bool, default=False
        Whether to shrink the estimated covariance matrix.

    Returns
    -------
    np.ndarray of shape (n, p)
        Matrix of knockoff variables.

    Notes
    -----
    If shrink=True, uses Ledoit-Wolf shrinkage for covariance estimation.
    Even if shrink=False, shrinkage will be applied if the estimated
    covariance matrix is not positive-definite.

    References
    ----------
    Candes et al., Panning for Gold: Model-free Knockoffs for
    High-dimensional Controlled Variable Selection, arXiv:1610.02351 (2016).
    """
    X = np.asarray(X, dtype=np.float64)

    if method not in ['asdp', 'sdp', 'equi']:
        raise ValueError(f"method must be 'asdp', 'sdp', or 'equi', got '{method}'")

    # Estimate the mean vector
    mu = np.mean(X, axis=0)

    # Estimate the covariance matrix
    if not shrink:
        Sigma = np.cov(X, rowvar=False)
        # Ensure it's 2D for single feature case
        if Sigma.ndim == 0:
            Sigma = np.array([[Sigma]])

        # Verify positive-definiteness
        if not is_posdef(Sigma):
            shrink = True

    if shrink:
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(X)
            Sigma = lw.covariance_
        except ImportError:
            warnings.warn(
                "sklearn is not installed. Using manual shrinkage."
            )
            # Manual shrinkage: λ * I + (1 - λ) * S
            S = np.cov(X, rowvar=False)
            if S.ndim == 0:
                S = np.array([[S]])
            n, p = X.shape
            trace_S = np.trace(S)
            # Simple shrinkage towards identity
            shrinkage = min(1.0, max(0.0, (p / n)))
            Sigma = (1 - shrinkage) * S + shrinkage * (trace_S / p) * np.eye(p)

    # Sample the Gaussian knockoffs
    return create_gaussian(X, mu, Sigma, method=method)
