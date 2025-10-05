"""
Module 05 - 风险管理模块

提供全面的风险管理功能，包括：
- 风险分析：投资组合风险评估、VaR/CVaR计算、风险敞口分析
- 仓位管理：凯利准则、风险平价、动态仓位调整
- 止损策略：自适应止损、追踪止损、多种止损方法
- 投资组合优化：均值方差优化、风险预算、Black-Litterman
- 压力测试：蒙特卡洛模拟、历史场景、因子冲击分析
- 数据库管理：风险数据持久化和查询
"""

# ==================== 风险分析 ====================
# ==================== 数据库管理 ====================
from .database_manager import RiskDatabaseManager, get_risk_database_manager

# ==================== 投资组合优化 ====================
from .portfolio_optimization import (
    EfficientFrontier,
    # 均值方差优化
    MeanVarianceOptimizer,
    MVOConfig,
    OptimizationObjective,
    OptimizationResult,
    RiskBudget,
    RiskBudgetConfig,
    # 风险预算
    RiskBudgetingOptimizer,
    RiskBudgetingResult,
    optimize_mean_variance,
)

# ==================== 仓位管理 ====================
from .position_sizing import (
    # 凯利准则
    KellyCriterion,
    KellyResult,
    # 风险平价
    RiskParity,
    RiskParityConfig,
    RiskParityResult,
    calculate_kelly_position,
    calculate_risk_parity_weights,
)
from .risk_analysis import (
    ExposureConfig,
    ExposureResult,
    # 风险分析器
    PortfolioRiskAnalyzer,
    RiskConfig,
    # 风险敞口分析器
    RiskExposureAnalyzer,
    # VaR计算器
    VaRCalculator,
    VaRConfig,
    analyze_exposure,
    calculate_portfolio_var,
)

# ==================== 止损策略 ====================
from .stop_loss_strategies import (
    # 自适应止损
    AdaptiveStopLoss,
    StopLossConfig,
    # 基础止损管理
    StopLossManager,
    StopLossOrder,
    StopLossResult,
    StopLossType,
    # 追踪止损
    TrailingStop,
    TrailingStopConfig,
    TrailingStopState,
    TrailingStopUpdate,
    calculate_adaptive_stop,
    create_trailing_stop,
)

# ==================== 压力测试 ====================
from .stress_testing import (
    DistributionType,
    MonteCarloResult,
    # 蒙特卡洛模拟
    MonteCarloSimulator,
    ScenarioConfig,
    # 场景生成
    ScenarioGenerator,
    ScenarioSet,
    ScenarioType,
    SimulationConfig,
    SimulationPath,
    StressScenario,
    generate_stress_scenarios,
    run_monte_carlo_simulation,
)

__all__ = [
    # ===== 风险分析 =====
    "PortfolioRiskAnalyzer",
    "RiskConfig",
    "calculate_portfolio_var",
    "VaRCalculator",
    "VaRConfig",
    "RiskExposureAnalyzer",
    "ExposureConfig",
    "ExposureResult",
    "analyze_exposure",
    # ===== 仓位管理 =====
    "KellyCriterion",
    "KellyResult",
    "calculate_kelly_position",
    "RiskParity",
    "RiskParityConfig",
    "RiskParityResult",
    "calculate_risk_parity_weights",
    # ===== 止损策略 =====
    "AdaptiveStopLoss",
    "StopLossConfig",
    "StopLossOrder",
    "StopLossResult",
    "StopLossType",
    "calculate_adaptive_stop",
    "StopLossManager",
    "TrailingStop",
    "TrailingStopConfig",
    "TrailingStopState",
    "TrailingStopUpdate",
    "create_trailing_stop",
    # ===== 投资组合优化 =====
    "MeanVarianceOptimizer",
    "MVOConfig",
    "OptimizationObjective",
    "OptimizationResult",
    "EfficientFrontier",
    "optimize_mean_variance",
    "RiskBudgetingOptimizer",
    "RiskBudget",
    "RiskBudgetConfig",
    "RiskBudgetingResult",
    # ===== 压力测试 =====
    "MonteCarloSimulator",
    "SimulationConfig",
    "SimulationPath",
    "MonteCarloResult",
    "DistributionType",
    "run_monte_carlo_simulation",
    "ScenarioGenerator",
    "ScenarioConfig",
    "StressScenario",
    "ScenarioSet",
    "ScenarioType",
    "generate_stress_scenarios",
    # ===== 数据库 =====
    "RiskDatabaseManager",
    "get_risk_database_manager",
]

__version__ = "1.0.0"
__author__ = "FinLoom Team"
