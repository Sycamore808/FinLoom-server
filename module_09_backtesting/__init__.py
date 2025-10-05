"""
Module 09 - 回测模块
提供完整的回测引擎、性能分析、交易模拟和报告生成功能
"""

# 回测引擎
from .backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    create_backtest_engine,
)

# 数据库管理
from .database_manager import BacktestDatabaseManager, get_backtest_database_manager

# 市场冲击模型
from .market_impact_model import (
    ImpactEstimate,
    ImpactParameters,
    LinearImpactModel,
    MarketImpactModel,
    SquareRootImpactModel,
)

# 性能分析
from .performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceReport,
    analyze_performance,
    compare_strategies,
)

# 报告生成
from .report_generator import BacktestReportGenerator, ReportConfig, ReportSection

# 风险归因
from .risk_attribution import RiskAttributionReport, RiskAttributor

# 交易模拟
from .transaction_simulator import (
    SimulatedOrderBook,
    TransactionResult,
    TransactionSimulator,
)

# 验证工具
from .validation_tools import BacktestValidator, ValidationReport

# Walk-forward分析
from .walk_forward_analyzer import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
)

__all__ = [
    # 回测引擎
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "create_backtest_engine",
    # 性能分析
    "PerformanceAnalyzer",
    "PerformanceReport",
    "analyze_performance",
    "compare_strategies",
    # 交易模拟
    "TransactionSimulator",
    "TransactionResult",
    "SimulatedOrderBook",
    # 市场冲击模型
    "MarketImpactModel",
    "LinearImpactModel",
    "SquareRootImpactModel",
    "ImpactParameters",
    "ImpactEstimate",
    # 报告生成
    "BacktestReportGenerator",
    "ReportConfig",
    "ReportSection",
    # 验证工具
    "BacktestValidator",
    "ValidationReport",
    # 风险归因
    "RiskAttributor",
    "RiskAttributionReport",
    # Walk-forward分析
    "WalkForwardAnalyzer",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardWindow",
    # 数据库管理
    "BacktestDatabaseManager",
    "get_backtest_database_manager",
]

__version__ = "1.0.0"
__author__ = "FinLoom Team"
__description__ = "回测验证模块 - 提供专业的策略回测和性能分析功能"
