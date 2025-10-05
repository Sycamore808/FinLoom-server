"""
超参数调优子模块
"""

from module_07_optimization.hyperparameter_tuning.bayesian_optimizer import (
    BayesianOptimizer,
)
from module_07_optimization.hyperparameter_tuning.grid_search import GridSearchOptimizer
from module_07_optimization.hyperparameter_tuning.random_search import (
    RandomSearchOptimizer,
)

# Optuna 是可选依赖
try:
    from module_07_optimization.hyperparameter_tuning.optuna_optimizer import (
        OptunaOptimizer,
    )

    __all__ = [
        "BayesianOptimizer",
        "GridSearchOptimizer",
        "OptunaOptimizer",
        "RandomSearchOptimizer",
    ]
except ImportError:
    __all__ = [
        "BayesianOptimizer",
        "GridSearchOptimizer",
        "RandomSearchOptimizer",
    ]
