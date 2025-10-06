"""
Module 07 - 优化模块
提供超参数优化、策略优化、多目标优化和资源优化功能
"""

# 基础优化器
from module_07_optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationStatus,
    Parameter,
    Trial,
)

# 超参数调优
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

    OPTUNA_AVAILABLE = True
except ImportError:
    OptunaOptimizer = None
    OPTUNA_AVAILABLE = False

# 多目标优化
# 数据库管理
from module_07_optimization.database_manager import (
    OptimizationDatabaseManager,
    get_optimization_database_manager,
)
from module_07_optimization.multi_objective_opt.nsga_optimizer import (
    NSGAIndividual,
    NSGAOptimizer,
)
from module_07_optimization.multi_objective_opt.objective_functions import (
    PortfolioObjectives,
    create_portfolio_objectives,
    params_to_weights,
    weights_to_params,
)
from module_07_optimization.multi_objective_opt.pareto_frontier import ParetoFrontier

# 优化管理器
from module_07_optimization.optimization_manager import (
    OptimizationManager,
    get_optimization_manager,
)

# 资源优化
from module_07_optimization.resource_optimization.compute_optimizer import (
    ComputeOptimizer,
    ComputeResource,
    ComputeTask,
)
from module_07_optimization.resource_optimization.cost_optimizer import (
    CostComponent,
    CostOptimizer,
)
from module_07_optimization.resource_optimization.memory_optimizer import (
    MemoryOptimizer,
    MemoryProfile,
)

# 策略优化
from module_07_optimization.strategy_optimization.parameter_space import (
    STRATEGY_SPACES,
    create_bollinger_bands_space,
    create_ma_crossover_space,
    create_macd_strategy_space,
    create_mean_reversion_space,
    create_momentum_strategy_space,
    create_rsi_strategy_space,
    get_strategy_space,
)
from module_07_optimization.strategy_optimization.performance_evaluator import (
    PerformanceEvaluator,
    PerformanceMetrics,
)
from module_07_optimization.strategy_optimization.strategy_optimizer import (
    StrategyOptimizer,
    StrategyPerformance,
)

__all__ = [
    # 基础类
    "BaseOptimizer",
    "Parameter",
    "Trial",
    "OptimizationResult",
    "OptimizationStatus",
    # 超参数优化
    "BayesianOptimizer",
    "GridSearchOptimizer",
    "OptunaOptimizer",
    "RandomSearchOptimizer",
    # 多目标优化
    "NSGAOptimizer",
    "NSGAIndividual",
    "ParetoFrontier",
    "PortfolioObjectives",
    "create_portfolio_objectives",
    "weights_to_params",
    "params_to_weights",
    # 策略优化
    "StrategyOptimizer",
    "StrategyPerformance",
    "PerformanceEvaluator",
    "PerformanceMetrics",
    "get_strategy_space",
    "create_ma_crossover_space",
    "create_rsi_strategy_space",
    "create_bollinger_bands_space",
    "create_macd_strategy_space",
    "create_mean_reversion_space",
    "create_momentum_strategy_space",
    "STRATEGY_SPACES",
    # 资源优化
    "ComputeOptimizer",
    "ComputeResource",
    "ComputeTask",
    "CostOptimizer",
    "CostComponent",
    "MemoryOptimizer",
    "MemoryProfile",
    # 数据库
    "OptimizationDatabaseManager",
    "get_optimization_database_manager",
    # 管理器
    "OptimizationManager",
    "get_optimization_manager",
]

__version__ = "1.0.0"
