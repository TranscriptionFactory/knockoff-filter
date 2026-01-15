"""
Knockoff Filter for Controlled Variable Selection.

This package implements the Knockoff Filter, a powerful and versatile tool
for controlled variable selection with FDR (False Discovery Rate) control.

The procedure is based on the construction of artificial 'knockoff copies'
of the variables present in the given statistical model. Then, it selects
those variables that are clearly better than their corresponding knockoffs,
based on some measure of variable importance.

References
----------
Candes et al., Panning for Gold: Model-free Knockoffs for High-dimensional
Controlled Variable Selection, arXiv:1610.02351 (2016).

Barber and Candes, Controlling the false discovery rate via knockoffs.
Ann. Statist. 43 (2015), no. 5, 2055--2085.

Examples
--------
>>> import numpy as np
>>> from knockoff import knockoff_filter, create_second_order
>>> n, p = 100, 50
>>> X = np.random.randn(n, p)
>>> beta = np.zeros(p)
>>> beta[:5] = 3.0
>>> y = X @ beta + np.random.randn(n)
>>> result = knockoff_filter(X, y, fdr=0.1)
>>> print(result.selected)
"""

__version__ = "0.1.0"

# Main filter
from .filter import (
    knockoff_filter,
    knockoff_threshold,
    KnockoffResult,
)

# Knockoff construction methods
from .create import (
    create_fixed,
    create_gaussian,
    create_second_order,
    KnockoffVariables,
)

# SDP solvers
from .solve import (
    create_solve_equi,
    create_solve_sdp,
    create_solve_asdp,
)

# Utility functions
from .utils import (
    normc,
    canonical_svd,
    is_posdef,
    cov2cor,
    random_problem,
    rnorm_matrix,
    with_seed,
    diag_pre_multiply,
    diag_post_multiply,
)

# Statistics (import submodule)
from . import stats

# Import commonly used statistics for convenience
from .stats import (
    stat_glmnet_coefdiff,
    stat_glmnet_lambdadiff,
    stat_glmnet_lambdasmax,
    stat_lasso_coefdiff,
    stat_lasso_lambdadiff,
    stat_lasso_lambdasmax,
    stat_lasso_coefdiff_bin,
    stat_lasso_lambdadiff_bin,
    stat_lasso_lambdasmax_bin,
    stat_forward_selection,
    stat_random_forest,
    stat_sqrt_lasso,
    stat_stability_selection,
)

__all__ = [
    # Version
    "__version__",
    # Main filter
    "knockoff_filter",
    "knockoff_threshold",
    "KnockoffResult",
    # Knockoff construction
    "create_fixed",
    "create_gaussian",
    "create_second_order",
    "KnockoffVariables",
    # SDP solvers
    "create_solve_equi",
    "create_solve_sdp",
    "create_solve_asdp",
    # Utility functions
    "normc",
    "canonical_svd",
    "is_posdef",
    "cov2cor",
    "random_problem",
    "rnorm_matrix",
    "with_seed",
    "diag_pre_multiply",
    "diag_post_multiply",
    # Statistics submodule
    "stats",
    # Common statistics
    "stat_glmnet_coefdiff",
    "stat_glmnet_lambdadiff",
    "stat_glmnet_lambdasmax",
    "stat_lasso_coefdiff",
    "stat_lasso_lambdadiff",
    "stat_lasso_lambdasmax",
    "stat_lasso_coefdiff_bin",
    "stat_lasso_lambdadiff_bin",
    "stat_lasso_lambdasmax_bin",
    "stat_forward_selection",
    "stat_random_forest",
    "stat_sqrt_lasso",
    "stat_stability_selection",
]
