"""GLMNet-based statistics for knockoff filter."""

from typing import Optional
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import swap_columns, correct_for_swap, compute_difference_stat, compute_signed_max_stat


def _lasso_max_lambda_glmnet(
    X: np.ndarray,
    y: np.ndarray,
    nlambda: int = 500,
    intercept: bool = True,
    standardize: bool = True,
    family: str = 'gaussian',
    **kwargs
) -> np.ndarray:
    """
    Compute the maximum lambda at which each variable enters the lasso model.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Response vector.
    nlambda : int, default=500
        Number of lambda values.
    intercept : bool, default=True
        Whether to fit an intercept.
    standardize : bool, default=True
        Whether to standardize features.
    family : str, default='gaussian'
        Response family.

    Returns
    -------
    np.ndarray of shape (p,)
        Maximum lambda values for each variable.
    """
    try:
        from sklearn.linear_model import lasso_path, LogisticRegression
    except ImportError:
        raise ImportError("scikit-learn is required for lasso statistics")

    n, p = X.shape

    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if family == 'gaussian':
        # Generate lambda sequence
        lambda_max = np.max(np.abs(X.T @ y)) / n
        lambda_min = lambda_max / 2000
        k = np.arange(nlambda) / nlambda
        lambdas = lambda_max * (lambda_min / lambda_max) ** k

        # Compute lasso path
        try:
            alphas, coefs, _ = lasso_path(X, y, alphas=lambdas, fit_intercept=intercept)
        except Exception:
            # Fallback: let sklearn choose alphas
            alphas, coefs, _ = lasso_path(X, y, n_alphas=nlambda, fit_intercept=intercept)

        # coefs has shape (p, n_alphas)
        # Find first nonzero entry for each variable
        lambda_entry = np.zeros(p)
        for j in range(p):
            nonzero_idx = np.where(np.abs(coefs[j, :]) > 0)[0]
            if len(nonzero_idx) > 0:
                # alphas are sorted in decreasing order by sklearn
                lambda_entry[j] = alphas[nonzero_idx[0]] * n

        return lambda_entry

    elif family == 'binomial':
        from sklearn.linear_model import LogisticRegressionCV

        # Use logistic regression with l1 penalty
        # Get regularization path
        try:
            # Generate C values (inverse of lambda)
            lambda_max = np.max(np.abs(X.T @ (y - y.mean()))) / n
            lambda_min = lambda_max / 2000
            k = np.arange(nlambda) / nlambda
            lambdas = lambda_max * (lambda_min / lambda_max) ** k
            Cs = 1 / (lambdas + 1e-10)

            lambda_entry = np.zeros(p)
            # Fit for each lambda
            for i, C in enumerate(Cs[:min(100, nlambda)]):  # Limit iterations
                clf = LogisticRegression(
                    penalty='l1', C=C, solver='saga',
                    fit_intercept=intercept, max_iter=1000
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(X, y)

                # Check which variables have entered
                coefs = clf.coef_.ravel()
                for j in range(p):
                    if np.abs(coefs[j]) > 0 and lambda_entry[j] == 0:
                        lambda_entry[j] = lambdas[i] * n

            return lambda_entry

        except Exception as e:
            warnings.warn(f"Logistic regression path failed: {e}")
            return np.zeros(p)

    else:
        raise ValueError(f"Unsupported family: {family}")


def _cv_coeffs_glmnet(
    X: np.ndarray,
    y: np.ndarray,
    nlambda: int = 500,
    intercept: bool = True,
    family: str = 'gaussian',
    n_jobs: int = -1,
    **kwargs
) -> np.ndarray:
    """
    Compute coefficients at CV-selected lambda.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        Response vector.
    nlambda : int, default=500
        Number of lambda values.
    intercept : bool, default=True
        Whether to fit an intercept.
    family : str, default='gaussian'
        Response family.
    n_jobs : int, default=-1
        Number of parallel jobs.

    Returns
    -------
    np.ndarray
        Coefficients at optimal lambda.
    """
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n, p = X.shape

    if family == 'gaussian':
        from sklearn.linear_model import LassoCV

        # Generate lambda sequence
        lambda_max = np.max(np.abs(X.T @ y)) / n
        lambda_min = lambda_max / 2000
        k = np.arange(nlambda) / nlambda
        alphas = lambda_max * (lambda_min / lambda_max) ** k

        # Fit LassoCV
        cv = LassoCV(alphas=alphas, fit_intercept=intercept, n_jobs=n_jobs, cv=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv.fit(X, y)

        # Return coefficients with intercept
        if intercept:
            return np.concatenate([[cv.intercept_], cv.coef_])
        else:
            return np.concatenate([[0], cv.coef_])

    elif family == 'binomial':
        from sklearn.linear_model import LogisticRegressionCV

        # Fit LogisticRegressionCV
        cv = LogisticRegressionCV(
            penalty='l1', solver='saga',
            fit_intercept=intercept,
            n_jobs=n_jobs, cv=10, max_iter=1000
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv.fit(X, y)

        # Return coefficients with intercept
        if intercept:
            return np.concatenate([cv.intercept_, cv.coef_.ravel()])
        else:
            return np.concatenate([[0], cv.coef_.ravel()])

    elif family == 'poisson':
        # Poisson regression with CV
        try:
            from sklearn.linear_model import PoissonRegressor
            from sklearn.model_selection import GridSearchCV

            # Simple grid search for alpha
            alphas = np.logspace(-4, 2, 20)
            best_score = -np.inf
            best_coef = np.zeros(p)

            for alpha in alphas:
                model = PoissonRegressor(alpha=alpha, fit_intercept=intercept, max_iter=1000)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X, y)
                    score = model.score(X, y)
                    if score > best_score:
                        best_score = score
                        if intercept:
                            best_coef = np.concatenate([[model.intercept_], model.coef_])
                        else:
                            best_coef = np.concatenate([[0], model.coef_])

            return best_coef

        except Exception as e:
            warnings.warn(f"Poisson regression failed: {e}")
            return np.zeros(p + 1)

    elif family == 'multinomial':
        from sklearn.linear_model import LogisticRegressionCV

        cv = LogisticRegressionCV(
            penalty='l1', solver='saga',
            multi_class='multinomial',
            fit_intercept=intercept,
            n_jobs=n_jobs, cv=10, max_iter=1000
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv.fit(X, y)

        # For multinomial, sum absolute coefficients across classes
        coefs_sum = np.sum(np.abs(cv.coef_), axis=0)
        if intercept:
            return np.concatenate([[0], coefs_sum])
        else:
            return np.concatenate([[0], coefs_sum])

    else:
        raise ValueError(f"Unsupported family: {family}")


def stat_glmnet_lambdadiff(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    family: str = 'gaussian',
    **kwargs
) -> np.ndarray:
    """
    GLM lambda difference statistic.

    Computes W_j = Z_j - Z_{j+p} where Z is the maximum lambda at which
    each variable enters the model.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Response vector.
    family : str, default='gaussian'
        Response family ('gaussian', 'binomial', 'poisson', 'multinomial').

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.
    """
    X = np.asarray(X, dtype=np.float64)
    X_k = np.asarray(X_k, dtype=np.float64)
    y = np.asarray(y)
    p = X.shape[1]

    # Randomly swap columns
    X_swap, Xk_swap, swap = swap_columns(X, X_k)

    # Concatenate and compute statistics
    X_combined = np.hstack([X_swap, Xk_swap])
    Z = _lasso_max_lambda_glmnet(X_combined, y, family=family, **kwargs)

    # Compute difference statistic
    W = compute_difference_stat(Z, p)

    # Correct for swapping
    return correct_for_swap(W, swap)


def stat_glmnet_lambdasmax(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    family: str = 'gaussian',
    **kwargs
) -> np.ndarray:
    """
    GLM signed maximum lambda statistic.

    Computes W_j = max(Z_j, Z_{j+p}) * sign(Z_j - Z_{j+p}).

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Response vector.
    family : str, default='gaussian'
        Response family.

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.
    """
    X = np.asarray(X, dtype=np.float64)
    X_k = np.asarray(X_k, dtype=np.float64)
    y = np.asarray(y)
    p = X.shape[1]

    # Randomly swap columns
    X_swap, Xk_swap, swap = swap_columns(X, X_k)

    # Concatenate and compute statistics
    X_combined = np.hstack([X_swap, Xk_swap])
    Z = _lasso_max_lambda_glmnet(X_combined, y, family=family, **kwargs)

    # Compute signed max statistic
    W = compute_signed_max_stat(Z, p)

    # Correct for swapping
    return correct_for_swap(W, swap)


def stat_glmnet_coefdiff(
    X: np.ndarray,
    X_k: np.ndarray,
    y: np.ndarray,
    family: str = 'gaussian',
    cores: int = 2,
    **kwargs
) -> np.ndarray:
    """
    GLM coefficient difference statistic with cross-validation.

    Computes W_j = |Z_j| - |Z_{j+p}| where Z are coefficients at
    CV-selected lambda.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Original variables.
    X_k : array-like of shape (n, p)
        Knockoff variables.
    y : array-like of shape (n,)
        Response vector.
    family : str, default='gaussian'
        Response family.
    cores : int, default=2
        Number of CPU cores for parallel CV.

    Returns
    -------
    np.ndarray of shape (p,)
        Statistics W.
    """
    X = np.asarray(X, dtype=np.float64)
    X_k = np.asarray(X_k, dtype=np.float64)
    y = np.asarray(y)
    p = X.shape[1]

    # Randomly swap columns
    X_swap, Xk_swap, swap = swap_columns(X, X_k)

    # Concatenate and compute statistics
    X_combined = np.hstack([X_swap, Xk_swap])
    glmnet_coefs = _cv_coeffs_glmnet(
        X_combined, y, family=family, n_jobs=cores, **kwargs
    )

    # Extract coefficients (skip intercept)
    if family == 'multinomial':
        # Already handled in _cv_coeffs_glmnet
        Z = np.abs(glmnet_coefs[1:2*p+1])
    elif family == 'cox':
        Z = glmnet_coefs[:2*p]
    else:
        Z = glmnet_coefs[1:2*p+1]

    # Compute absolute difference statistic
    orig = np.arange(p)
    W = np.abs(Z[orig]) - np.abs(Z[orig + p])

    # Correct for swapping
    return correct_for_swap(W, swap)
