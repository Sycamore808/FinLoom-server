"""
止损管理器
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("stop_loss_manager")


@dataclass
class StopLossConfig:
    """止损配置"""

    method: str = "atr"  # 'fixed', 'percent', 'atr', 'time'
    atr_multiplier: float = 2.0  # ATR倍数
    max_loss_percent: float = 0.05  # 最大亏损百分比
    trailing_stop: bool = True  # 启用移动止损
    time_stop_days: int = 30  # 时间止损天数


@dataclass
class StopLossResult:
    """止损结果"""

    stop_price: float
    max_loss: float
    max_loss_percent: float
    stop_type: str
    trigger_condition: str


class StopLossManager:
    """智能止损管理器"""

    def __init__(self, config: Optional[StopLossConfig] = None):
        """初始化止损管理器

        Args:
            config: 止损配置
        """
        self.config = config or StopLossConfig()
        logger.info(f"Initialized StopLossManager with method={self.config.method}")

    def calculate_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        atr: Optional[float] = None,
        position_type: str = "long",
    ) -> StopLossResult:
        """计算止损价格

        Args:
            entry_price: 入场价格
            current_price: 当前价格
            atr: ATR值（可选）
            position_type: 持仓类型 ('long' or 'short')

        Returns:
            止损结果
        """
        try:
            if self.config.method == "fixed":
                stop_price = self._fixed_stop_loss(entry_price, position_type)
            elif self.config.method == "percent":
                stop_price = self._percent_stop_loss(entry_price, position_type)
            elif self.config.method == "atr":
                if atr is None:
                    logger.warning("ATR not provided, falling back to percent stop")
                    stop_price = self._percent_stop_loss(entry_price, position_type)
                else:
                    stop_price = self._atr_based_stop(entry_price, atr, position_type)
            else:
                stop_price = self._percent_stop_loss(entry_price, position_type)

            # 如果启用移动止损，检查是否需要调整
            if self.config.trailing_stop:
                stop_price = self._update_trailing_stop(
                    entry_price, current_price, stop_price, position_type
                )

            # 计算最大损失
            if position_type == "long":
                max_loss = entry_price - stop_price
                max_loss_percent = max_loss / entry_price
            else:
                max_loss = stop_price - entry_price
                max_loss_percent = max_loss / entry_price

            result = StopLossResult(
                stop_price=stop_price,
                max_loss=abs(max_loss),
                max_loss_percent=abs(max_loss_percent),
                stop_type=self.config.method,
                trigger_condition=f"Price {'<=' if position_type == 'long' else '>='} {stop_price:.2f}",
            )

            logger.info(
                f"Stop loss calculated: {stop_price:.2f} (max loss: {max_loss_percent:.2%})"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to calculate stop loss: {e}")
            raise QuantSystemError(f"Stop loss calculation failed: {e}")

    def _fixed_stop_loss(self, entry_price: float, position_type: str) -> float:
        """固定金额止损"""
        fixed_amount = entry_price * self.config.max_loss_percent

        if position_type == "long":
            return entry_price - fixed_amount
        else:
            return entry_price + fixed_amount

    def _percent_stop_loss(self, entry_price: float, position_type: str) -> float:
        """百分比止损"""
        if position_type == "long":
            return entry_price * (1 - self.config.max_loss_percent)
        else:
            return entry_price * (1 + self.config.max_loss_percent)

    def _atr_based_stop(
        self, entry_price: float, atr: float, position_type: str
    ) -> float:
        """基于ATR的止损"""
        stop_distance = atr * self.config.atr_multiplier

        if position_type == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def _update_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float,
        position_type: str,
    ) -> float:
        """更新移动止损

        Args:
            entry_price: 入场价格
            current_price: 当前价格
            current_stop: 当前止损价
            position_type: 持仓类型

        Returns:
            新的止损价
        """
        try:
            if position_type == "long":
                # 对于多头，只向上移动止损
                if current_price > entry_price:
                    profit = current_price - entry_price
                    new_stop = entry_price + profit * (1 - self.config.max_loss_percent)
                    return max(current_stop, new_stop)
            else:
                # 对于空头，只向下移动止损
                if current_price < entry_price:
                    profit = entry_price - current_price
                    new_stop = entry_price - profit * (1 - self.config.max_loss_percent)
                    return min(current_stop, new_stop)

            return current_stop

        except Exception as e:
            logger.error(f"Failed to update trailing stop: {e}")
            return current_stop

    def check_stop_triggered(
        self, current_price: float, stop_price: float, position_type: str = "long"
    ) -> bool:
        """检查是否触发止损

        Args:
            current_price: 当前价格
            stop_price: 止损价格
            position_type: 持仓类型

        Returns:
            是否触发止损
        """
        try:
            if position_type == "long":
                triggered = current_price <= stop_price
            else:
                triggered = current_price >= stop_price

            if triggered:
                logger.warning(
                    f"Stop loss triggered: current={current_price:.2f}, stop={stop_price:.2f}"
                )

            return triggered

        except Exception as e:
            logger.error(f"Failed to check stop trigger: {e}")
            return False

    def calculate_position_stop_loss(
        self,
        positions: Dict[str, Dict],
        current_prices: Dict[str, float],
        atr_values: Optional[Dict[str, float]] = None,
    ) -> Dict[str, StopLossResult]:
        """计算所有持仓的止损

        Args:
            positions: 持仓字典 {symbol: {'cost': float, 'shares': int}}
            current_prices: 当前价格字典
            atr_values: ATR值字典（可选）

        Returns:
            止损结果字典
        """
        try:
            stop_losses = {}

            for symbol, position in positions.items():
                if symbol not in current_prices:
                    logger.warning(f"No current price for {symbol}")
                    continue

                entry_price = position["cost"]
                current_price = current_prices[symbol]
                atr = atr_values.get(symbol) if atr_values else None

                stop_loss = self.calculate_stop_loss(
                    entry_price=entry_price,
                    current_price=current_price,
                    atr=atr,
                    position_type="long",
                )

                stop_losses[symbol] = stop_loss

            logger.info(f"Calculated stop losses for {len(stop_losses)} positions")
            return stop_losses

        except Exception as e:
            logger.error(f"Failed to calculate position stop losses: {e}")
            raise QuantSystemError(f"Position stop loss calculation failed: {e}")
