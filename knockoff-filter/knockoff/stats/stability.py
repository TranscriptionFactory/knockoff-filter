"""Stability selection statistics for knockoff filter."""

import warnings
import numpy as np
from .base import swap_columns, correct_for_swap


def _stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.5,
    threshold: float = 0.6,
    **kwargs
) -> np.ndarray:
    """
    Perform stability selection.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Response vector.
    n_bootstrap : int, default=100
        Number of bootstrap iterations.
    sample_fraction : float, default=0.5
        Fraction of samples to use in each iteration.
    threshold : float, default=0.6
        Regularization strength threshold.

    Returns
    -------
    np.ndarray of shape (p,)
        Selection probability for each variable.
    """
    try:
        from sklearn.linear_model import LassoCV, LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("scikit-learn is required for stability selection")

    n, p = X.shape
    selection_counts = np.zeros(p)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Determine if classification or regression
    is_classification = False
    if not np.issubdtype(y.dtype, np.number):
        is_classification = True
    else:
        unique_vals = np.unique(y)
        if len(unique_vals) <= 10 and np.all(unique_vals == unique_vals.astype(int)):
            is_classification = True

    subsample_size = int(n * sample_fraction)

    for _ in range(n_bootstrap):
        # Subsample
        indices = np.random.choice(n, size=subsample_size, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]

        try:
            if is_classification:
                model = LogisticRegressionCV(
                    penalty='l1',
                    solver='saga',
                    cv=3,
                    max_iter=500,
                    n_jobs=1
                )
            else:
                model = LassoCV(cv=3, n_jobs=1, max_iter=10000)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_sub, y_sub)

            # Get selected variables
            if is_classification:
                coefs = model.coef_.ravel()
            else:
                coefs = model.coef_

            selected = np.abs(coefs) > 1e-10
            selection_counts += selected.astype(float)

        except Exception:
            # Skip this iteration if fitting fails
            continue

    # Compute selection probability
    selection_prob = selection_counts / n_bootstrap

    return selection_prob


def stat_stability_selection(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.5,
    **kwargs
) -> np.ndarray:
    """
    Stability selection importance statistic.

    Computes W_j = |Z_j| - |Z_{j+p}| where Z is the selection probability
    from stability selection.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Response vector.
    n_bootstrap : int, default=100
        Number of bootstrap iterations.
    sample_fraction : float, default=0.5
        Fraction of samples per iteration.

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.

    Notes
    -----
    Stability selection measures the probability that each variable
    is selected across multiple subsamples. Variables that are
    consistently selected are more likely to be truly important.
    """
    X = np.asarray(X, dtype=np.float64)
    X_k = np.asarray(X_k, dtype=np.float64)
    y = np.asarray(y)

    p = X.shape[1]

    # Randomly swap columns
    X_swap, Xk_swap, swap = swap_columns(X, X_k)

    # Concatenate
    X_combined = np.hstack([X_swap, Xk_swap])

    # Compute stability selection
    Z = _stability_selection(
        X_combined, y,
        n_bootstrap=n_bootstrap,
        sample_fraction=sample_fraction,
        **kwargs
    )

    # Compute difference statistic
    orig = np.arange(p)
    W = np.abs(Z[orig]) - np.abs(Z[orig + p])

    # Correct for swapping
    return correct_for_swap(W, swap)
