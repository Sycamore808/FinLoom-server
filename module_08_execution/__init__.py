"""
执行模块初始化文件
Module 08 - 交易执行引擎
"""

# 信号生成和过滤

# 执行接口（简化）
# 数据库管理
from .database_manager import ExecutionDatabaseManager, get_execution_database_manager

# 执行算法
from .execution_algorithms import (
    AdaptiveAlgorithm,
    ExecutionAlgorithm,
    ExecutionPlan,
    ExecutionSlice,
    ImplementationShortfallAlgorithm,
    TWAPAlgorithm,
    VWAPAlgorithm,
    analyze_execution_quality,
    create_execution_algorithm,
)
from .execution_interface import (
    ExecutionDestination,
    ExecutionInterface,
    ExecutionRequest,
    ExecutionResult,
    get_execution_interface,
)

# 执行监控
from .execution_monitor import (
    ExecutionAlert,
    ExecutionMetrics,
    ExecutionMonitor,
    ExecutionPerformance,
    create_execution_monitor,
)

# 市场冲击模型
from .market_impact_model import (
    AlmgrenChrissModel,
    ImpactEstimate,
    LinearPropagatorModel,
    MachineLearningImpactModel,
    MarketImpactModel,
    MarketMicrostructure,
    analyze_market_microstructure,
    create_impact_model,
)

# 订单管理
from .order_manager import (
    Order,
    OrderManager,
    OrderStatus,
    OrderType,
    OrderUpdate,
    get_order_manager,
)

# 订单路由
from .order_router import (
    ExecutionStrategy,
    ExecutionVenue,
    OrderRouter,
    RouterConfig,
    RoutingDecision,
    get_order_router,
    route_order_quick,
)
from .signal_filter import (
    FilterConfig,
    FilterResult,
    SignalFilter,
    create_signal_filter,
    quick_filter_signals,
)
from .signal_generator import (
    EnhancedSignal,
    SignalGenerator,
    SignalPriority,
    SignalType,
    generate_trading_signals,
)

# 交易日志
from .transaction_logger import (
    TransactionLog,
    TransactionLogger,
    close_transaction_logger,
    get_transaction_logger,
)

__all__ = [
    # 信号生成和过滤
    "SignalGenerator",
    "EnhancedSignal",
    "SignalType",
    "SignalPriority",
    "generate_trading_signals",
    "SignalFilter",
    "FilterConfig",
    "FilterResult",
    "create_signal_filter",
    "quick_filter_signals",
    # 订单管理
    "OrderManager",
    "Order",
    "OrderStatus",
    "OrderType",
    "OrderUpdate",
    "get_order_manager",
    # 执行接口（简化）
    "ExecutionInterface",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionDestination",
    "get_execution_interface",
    # 订单路由
    "OrderRouter",
    "RoutingDecision",
    "RouterConfig",
    "ExecutionVenue",
    "ExecutionStrategy",
    "get_order_router",
    "route_order_quick",
    # 执行算法
    "ExecutionAlgorithm",
    "TWAPAlgorithm",
    "VWAPAlgorithm",
    "ImplementationShortfallAlgorithm",
    "AdaptiveAlgorithm",
    "ExecutionPlan",
    "ExecutionSlice",
    "create_execution_algorithm",
    "analyze_execution_quality",
    # 执行监控
    "ExecutionMonitor",
    "ExecutionMetrics",
    "ExecutionAlert",
    "ExecutionPerformance",
    "create_execution_monitor",
    # 市场冲击模型
    "MarketImpactModel",
    "AlmgrenChrissModel",
    "LinearPropagatorModel",
    "MachineLearningImpactModel",
    "ImpactEstimate",
    "MarketMicrostructure",
    "create_impact_model",
    "analyze_market_microstructure",
    # 交易日志
    "TransactionLogger",
    "TransactionLog",
    "get_transaction_logger",
    "close_transaction_logger",
    # 数据库管理
    "ExecutionDatabaseManager",
    "get_execution_database_manager",
]

__version__ = "1.0.0"
__author__ = "FinLoom Team"
__description__ = (
    "Trading execution engine for automated order management and execution"
)
