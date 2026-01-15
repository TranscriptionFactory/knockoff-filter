"""Statistics for knockoff filter.

This module provides various importance statistics for the knockoff filter,
including lasso-based, random forest, forward selection, and stability
selection methods.
"""

# GLMNet statistics (generic GLM)
from .glmnet import (
    stat_glmnet_lambdadiff,
    stat_glmnet_lambdasmax,
    stat_glmnet_coefdiff,
)

# Lasso statistics (Gaussian response)
from .lasso import (
    stat_lasso_lambdadiff,
    stat_lasso_lambdasmax,
    stat_lasso_coefdiff,
)

# Lasso statistics (Binary response)
from .lasso_bin import (
    stat_lasso_lambdadiff_bin,
    stat_lasso_lambdasmax_bin,
    stat_lasso_coefdiff_bin,
)

# Forward selection
from .forward import stat_forward_selection

# Random forest
from .random_forest import stat_random_forest

# Square-root lasso
from .sqrt_lasso import stat_sqrt_lasso

# Stability selection
from .stability import stat_stability_selection

# Base utilities
from .base import (
    swap_columns,
    correct_for_swap,
    compute_difference_stat,
    compute_signed_max_stat,
    standardize,
)

__all__ = [
    # GLMNet
    "stat_glmnet_lambdadiff",
    "stat_glmnet_lambdasmax",
    "stat_glmnet_coefdiff",
    # Lasso (Gaussian)
    "stat_lasso_lambdadiff",
    "stat_lasso_lambdasmax",
    "stat_lasso_coefdiff",
    # Lasso (Binary)
    "stat_lasso_lambdadiff_bin",
    "stat_lasso_lambdasmax_bin",
    "stat_lasso_coefdiff_bin",
    # Other statistics
    "stat_forward_selection",
    "stat_random_forest",
    "stat_sqrt_lasso",
    "stat_stability_selection",
    # Base utilities
    "swap_columns",
    "correct_for_swap",
    "compute_difference_stat",
    "compute_signed_max_stat",
    "standardize",
]
