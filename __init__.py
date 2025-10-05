"""
FinLoom 量化投资引擎
模块初始化文件
"""

__version__ = "1.0.0"
__author__ = "FinLoom Team"

# 通用组件
from common.data_structures import MarketData, Signal, Position
from common.exceptions import QuantSystemError, DataError, ModelError, ExecutionError
from common.logging_system import setup_logger

__all__ = [
    # 通用组件
    "MarketData",
    "Signal", 
    "Position",
    "QuantSystemError",
    "DataError",
    "ModelError",
    "ExecutionError",
    "setup_logger"
]