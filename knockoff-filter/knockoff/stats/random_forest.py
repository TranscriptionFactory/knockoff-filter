"""Random forest statistics for knockoff filter."""

import numpy as np
from .base import swap_columns, correct_for_swap


def stat_random_forest(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    n_jobs: int = -1,
    **kwargs
) -> np.ndarray:
    """
    Random forest importance difference statistic.

    Computes W_j = |Z_j| - |Z_{j+p}| where Z is the feature importance
    from a random forest model.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Response vector (numeric for regression, categorical for classification).
    n_estimators : int, default=100
        Number of trees in the forest.
    n_jobs : int, default=-1
        Number of parallel jobs.

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    except ImportError:
        raise ImportError("scikit-learn is required for random forest statistics")

    X = np.asarray(X, dtype=np.float64)
    X_k = np.asarray(X_k, dtype=np.float64)
    y = np.asarray(y)

    p = X.shape[1]

    # Randomly swap columns
    X_swap, Xk_swap, swap = swap_columns(X, X_k)

    # Concatenate
    X_combined = np.hstack([X_swap, Xk_swap])

    # Determine if regression or classification
    if np.issubdtype(y.dtype, np.number):
        unique_vals = np.unique(y)
        if len(unique_vals) <= 10 and np.all(unique_vals == unique_vals.astype(int)):
            # Likely classification
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                **kwargs
            )
        else:
            # Regression
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                **kwargs
            )
    else:
        # Categorical response - classification
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            **kwargs
        )

    # Fit model
    model.fit(X_combined, y)

    # Get feature importances
    Z = model.feature_importances_

    # Compute difference statistic
    orig = np.arange(p)
    W = np.abs(Z[orig]) - np.abs(Z[orig + p])

    # Correct for swapping
    return correct_for_swap(W, swap)
