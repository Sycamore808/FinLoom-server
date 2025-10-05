"""
通用组件模块初始化文件
"""

from .constants import (
    TRADING_DAYS_PER_YEAR,
    MINUTES_PER_TRADING_DAY,
    DEFAULT_TIMEZONE,
    MAX_SYMBOLS_PER_REQUEST,
    DEFAULT_LOOKBACK_DAYS,
    DATA_QUALITY_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MIN_POSITION_SIZE,
    MAX_POSITION_PCT,
    DEFAULT_SLIPPAGE_BPS,
    LOG_LEVEL,
    MAX_RETRY_ATTEMPTS,
    TIMEOUT_SECONDS
)

from .data_structures import MarketData, Signal, Position
from .exceptions import QuantSystemError, DataError, ModelError, ExecutionError
from .logging_system import setup_logger
from .communication_protocol import (
    InterModuleMessage,
    MessageQueue,
    ModuleCommunicator,
    create_message_queue,
    create_module_communicator,
    send_data_message
)

__all__ = [
    # 常量
    "TRADING_DAYS_PER_YEAR",
    "MINUTES_PER_TRADING_DAY",
    "DEFAULT_TIMEZONE",
    "MAX_SYMBOLS_PER_REQUEST",
    "DEFAULT_LOOKBACK_DAYS",
    "DATA_QUALITY_THRESHOLD",
    "DEFAULT_BATCH_SIZE",
    "MAX_EPOCHS",
    "EARLY_STOPPING_PATIENCE",
    "MIN_POSITION_SIZE",
    "MAX_POSITION_PCT",
    "DEFAULT_SLIPPAGE_BPS",
    "LOG_LEVEL",
    "MAX_RETRY_ATTEMPTS",
    "TIMEOUT_SECONDS",
    
    # 数据结构
    "MarketData",
    "Signal",
    "Position",
    
    # 异常
    "QuantSystemError",
    "DataError",
    "ModelError",
    "ExecutionError",
    
    # 日志
    "setup_logger",
    
    # 通信协议
    "InterModuleMessage",
    "MessageQueue",
    "ModuleCommunicator",
    "create_message_queue",
    "create_module_communicator",
    "send_data_message"
]