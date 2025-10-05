"""
止损策略模块初始化文件
"""

from .adaptive_stop_loss import (
    AdaptiveStopLoss,
    StopLossOrder,
    StopLossType,
    calculate_adaptive_stop,
)
from .adaptive_stop_loss import (
    StopLossConfig as AdaptiveStopLossConfig,
)
from .adaptive_stop_loss import (
    StopLossResult as AdaptiveStopLossResult,
)
from .stop_loss_manager import StopLossConfig, StopLossManager, StopLossResult
from .trailing_stop import (
    TrailingStop,
    TrailingStopConfig,
    TrailingStopState,
    TrailingStopUpdate,
    create_trailing_stop,
)

__all__ = [
    # 基础止损管理器（主要使用）
    "StopLossManager",
    "StopLossConfig",
    "StopLossResult",
    # 自适应止损
    "AdaptiveStopLoss",
    "AdaptiveStopLossConfig",
    "StopLossOrder",
    "AdaptiveStopLossResult",
    "StopLossType",
    "calculate_adaptive_stop",
    # 追踪止损
    "TrailingStop",
    "TrailingStopConfig",
    "TrailingStopState",
    "TrailingStopUpdate",
    "create_trailing_stop",
]
