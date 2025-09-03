"""
追踪止损策略模块
实现动态追踪止损机制
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.constants import MIN_POSITION_SIZE
from common.exceptions import ModelError
from common.logging_system import setup_logger

logger = setup_logger("trailing_stop")


class TrailingMethod(Enum):
    """追踪方法枚举"""

    FIXED_PERCENTAGE = "fixed_percentage"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    HIGH_WATER_MARK = "high_water_mark"
    CHANDELIER_EXIT = "chandelier_exit"
    PARABOLIC_SAR = "parabolic_sar"


@dataclass
class TrailingStopConfig:
    """追踪止损配置"""

    method: TrailingMethod = TrailingMethod.FIXED_PERCENTAGE
    trailing_percentage: float = 0.05  # 5%追踪距离
    atr_multiplier: float = 2.5
    volatility_lookback: int = 20
    sar_acceleration: float = 0.02
    sar_max_acceleration: float = 0.20
    use_intraday_high: bool = True
    update_frequency: str = "tick"  # tick, minute, daily
    min_profit_to_activate: float = 0.01  # 1%最小盈利激活
    step_trailing: bool = False  # 阶梯式追踪
    step_size: float = 0.005  # 0.5%阶梯


@dataclass
class TrailingStopState:
    """追踪止损状态"""

    symbol: str
    entry_price: float
    current_stop: float
    highest_price: float
    lowest_price: float
    last_update_time: datetime
    activation_price: float
    is_activated: bool
    trail_amount: float
    updates_count: int
    profit_locked: float


@dataclass
class TrailingStopUpdate:
    """追踪止损更新"""

    symbol: str
    old_stop: float
    new_stop: float
    reason: str
    highest_price: float
    current_price: float
    distance_pct: float
    profit_locked_pct: float
    timestamp: datetime


class TrailingStop:
    """追踪止损策略类"""

    def __init__(self, config: Optional[TrailingStopConfig] = None):
        """初始化追踪止损

        Args:
            config: 追踪止损配置
        """
        self.config = config or TrailingStopConfig()
        self.active_stops: Dict[str, TrailingStopState] = {}
        self.stop_history: List[TrailingStopUpdate] = []
        self.performance_metrics: Dict[str, float] = {}

    def initialize_trailing_stop(
        self, symbol: str, entry_price: float, initial_stop: float, current_price: float
    ) -> TrailingStopState:
        """初始化追踪止损

        Args:
            symbol: 标的代码
            entry_price: 入场价格
            initial_stop: 初始止损价
            current_price: 当前价格

        Returns:
            追踪止损状态
        """
        logger.info(f"Initializing trailing stop for {symbol}")

        # 计算激活价格
        activation_price = entry_price * (1 + self.config.min_profit_to_activate)

        # 检查是否已激活
        is_activated = current_price >= activation_price

        # 计算追踪距离
        if self.config.method == TrailingMethod.FIXED_PERCENTAGE:
            trail_amount = entry_price * self.config.trailing_percentage
        else:
            trail_amount = self._calculate_dynamic_trail_amount(
                symbol, current_price, entry_price
            )

        state = TrailingStopState(
            symbol=symbol,
            entry_price=entry_price,
            current_stop=initial_stop,
            highest_price=current_price,
            lowest_price=current_price,
            last_update_time=datetime.now(),
            activation_price=activation_price,
            is_activated=is_activated,
            trail_amount=trail_amount,
            updates_count=0,
            profit_locked=max(0, initial_stop - entry_price),
        )

        self.active_stops[symbol] = state

        logger.info(
            f"Trailing stop initialized: Stop={initial_stop:.2f}, Trail={trail_amount:.2f}"
        )

        return state

    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        high_price: Optional[float] = None,
        market_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, bool, Optional[TrailingStopUpdate]]:
        """更新追踪止损

        Args:
            symbol: 标的代码
            current_price: 当前价格
            high_price: 最高价（可选）
            market_data: 市场数据（可选）

        Returns:
            (新止损价, 是否触发, 更新记录)
        """
        if symbol not in self.active_stops:
            raise ValueError(f"No active trailing stop for {symbol}")

        state = self.active_stops[symbol]
        old_stop = state.current_stop

        # 更新最高价
        if high_price and self.config.use_intraday_high:
            state.highest_price = max(state.highest_price, high_price)
        else:
            state.highest_price = max(state.highest_price, current_price)

        # 检查是否激活
        if not state.is_activated:
            if current_price >= state.activation_price:
                state.is_activated = True
                logger.info(
                    f"Trailing stop activated for {symbol} at {current_price:.2f}"
                )
            else:
                # 未激活时保持初始止损
                return state.current_stop, current_price <= state.current_stop, None

        # 计算新的止损价
        new_stop = self._calculate_new_stop(state, current_price, market_data)

        # 止损只能上移（多头）或下移（空头）
        if new_stop > state.current_stop:  # 假设多头
            # 检查是否使用阶梯式追踪
            if self.config.step_trailing:
                new_stop = self._apply_step_trailing(state.current_stop, new_stop)

            state.current_stop = new_stop
            state.updates_count += 1
            state.profit_locked = new_stop - state.entry_price
            state.last_update_time = datetime.now()

            # 创建更新记录
            update = TrailingStopUpdate(
                symbol=symbol,
                old_stop=old_stop,
                new_stop=new_stop,
                reason=f"Trailing {self.config.method.value}",
                highest_price=state.highest_price,
                current_price=current_price,
                distance_pct=(state.highest_price - new_stop) / state.highest_price,
                profit_locked_pct=state.profit_locked / state.entry_price,
                timestamp=datetime.now(),
            )

            self.stop_history.append(update)

            logger.info(
                f"Trailing stop updated for {symbol}: {old_stop:.2f} -> {new_stop:.2f}"
            )

            # 检查是否触发
            triggered = current_price <= new_stop

            return new_stop, triggered, update

        # 止损未更新
        triggered = current_price <= state.current_stop
        return state.current_stop, triggered, None

    def calculate_atr_trailing_stop(
        self,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """计算ATR追踪止损

        Args:
            high_prices: 最高价序列
            low_prices: 最低价序列
            close_prices: 收盘价序列
            period: ATR周期

        Returns:
            ATR追踪止损序列
        """
        # 计算真实波幅
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift(1))
        tr3 = abs(low_prices - close_prices.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算ATR
        atr = tr.rolling(window=period).mean()

        # 计算追踪止损
        highest_high = high_prices.rolling(window=period).max()
        trailing_stop = highest_high - self.config.atr_multiplier * atr

        return trailing_stop

    def calculate_chandelier_exit(
        self,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        period: int = 22,
        multiplier: float = 3.0,
    ) -> Tuple[pd.Series, pd.Series]:
        """计算Chandelier Exit

        Args:
            high_prices: 最高价序列
            low_prices: 最低价序列
            close_prices: 收盘价序列
            period: 周期
            multiplier: ATR倍数

        Returns:
            (多头止损, 空头止损)
        """
        # 计算ATR
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift(1))
        tr3 = abs(low_prices - close_prices.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # 多头止损：最高价 - ATR * multiplier
        highest = high_prices.rolling(window=period).max()
        long_stop = highest - multiplier * atr

        # 空头止损：最低价 + ATR * multiplier
        lowest = low_prices.rolling(window=period).min()
        short_stop = lowest + multiplier * atr

        return long_stop, short_stop

    def calculate_parabolic_sar(
        self,
        high_prices: pd.Series,
        low_prices: pd.Series,
        initial_af: float = 0.02,
        max_af: float = 0.20,
    ) -> pd.Series:
        """计算抛物线SAR

        Args:
            high_prices: 最高价序列
            low_prices: 最低价序列
            initial_af: 初始加速因子
            max_af: 最大加速因子

        Returns:
            SAR序列
        """
        n = len(high_prices)
        sar = pd.Series(index=high_prices.index)
        trend = 1  # 1为上升趋势，-1为下降趋势
        af = initial_af
        ep = high_prices.iloc[0] if trend == 1 else low_prices.iloc[0]
        sar.iloc[0] = low_prices.iloc[0] if trend == 1 else high_prices.iloc[0]

        for i in range(1, n):
            # 更新SAR
            sar.iloc[i] = sar.iloc[i - 1] + af * (ep - sar.iloc[i - 1])

            # 检查趋势反转
            if trend == 1:
                if low_prices.iloc[i] <= sar.iloc[i]:
                    # 反转为下降趋势
                    trend = -1
                    sar.iloc[i] = ep
                    ep = low_prices.iloc[i]
                    af = initial_af
                else:
                    # 继续上升趋势
                    if high_prices.iloc[i] > ep:
                        ep = high_prices.iloc[i]
                        af = min(af + initial_af, max_af)
                    # 确保SAR不超过最近两期的最低价
                    sar.iloc[i] = min(sar.iloc[i], low_prices.iloc[i - 1 : i + 1].min())
            else:
                if high_prices.iloc[i] >= sar.iloc[i]:
                    # 反转为上升趋势
                    trend = 1
                    sar.iloc[i] = ep
                    ep = high_prices.iloc[i]
                    af = initial_af
                else:
                    # 继续下降趋势
                    if low_prices.iloc[i] < ep:
                        ep = low_prices.iloc[i]
                        af = min(af + initial_af, max_af)
                    # 确保SAR不低于最近两期的最高价
                    sar.iloc[i] = max(
                        sar.iloc[i], high_prices.iloc[i - 1 : i + 1].max()
                    )

        return sar

    def optimize_trailing_parameters(
        self,
        historical_data: pd.DataFrame,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
    ) -> Dict[str, float]:
        """优化追踪参数

        Args:
            historical_data: 历史数据
            entry_signals: 入场信号
            exit_signals: 出场信号

        Returns:
            最优参数字典
        """
        logger.info("Optimizing trailing stop parameters...")

        best_params = {}
        best_performance = -float("inf")

        # 参数搜索范围
        trail_percentages = np.arange(0.02, 0.10, 0.01)
        atr_multipliers = np.arange(1.5, 4.0, 0.5)
        activation_thresholds = np.arange(0.005, 0.03, 0.005)

        for trail_pct in trail_percentages:
            for atr_mult in atr_multipliers:
                for activation in activation_thresholds:
                    # 回测这组参数
                    performance = self._backtest_trailing_params(
                        historical_data,
                        entry_signals,
                        exit_signals,
                        trail_pct,
                        atr_mult,
                        activation,
                    )

                    if performance > best_performance:
                        best_performance = performance
                        best_params = {
                            "trailing_percentage": trail_pct,
                            "atr_multiplier": atr_mult,
                            "min_profit_to_activate": activation,
                        }

        logger.info(f"Optimal parameters found: {best_params}")

        return best_params

    def get_stop_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """获取止损统计

        Args:
            symbol: 标的代码（可选）

        Returns:
            统计信息字典
        """
        if symbol:
            updates = [u for u in self.stop_history if u.symbol == symbol]
        else:
            updates = self.stop_history

        if not updates:
            return {}

        stats = {
            "total_updates": len(updates),
            "avg_profit_locked": np.mean([u.profit_locked_pct for u in updates]),
            "max_profit_locked": max([u.profit_locked_pct for u in updates]),
            "avg_distance": np.mean([u.distance_pct for u in updates]),
            "update_frequency": len(updates) / len(set([u.symbol for u in updates])),
        }

        return stats

    def _calculate_dynamic_trail_amount(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        market_data: Optional[pd.DataFrame] = None,
    ) -> float:
        """计算动态追踪距离

        Args:
            symbol: 标的代码
            current_price: 当前价格
            entry_price: 入场价格
            market_data: 市场数据

        Returns:
            追踪距离
        """
        if self.config.method == TrailingMethod.FIXED_PERCENTAGE:
            return current_price * self.config.trailing_percentage

        elif self.config.method == TrailingMethod.VOLATILITY_BASED:
            if market_data is not None and "close" in market_data.columns:
                volatility = market_data["close"].pct_change().std()
                return current_price * (2 * volatility)
            else:
                return current_price * self.config.trailing_percentage

        elif self.config.method == TrailingMethod.ATR_BASED:
            if market_data is not None and all(
                col in market_data.columns for col in ["high", "low", "close"]
            ):
                # 计算ATR
                tr = self._calculate_true_range(market_data)
                atr = tr.rolling(window=14).mean().iloc[-1]
                return atr * self.config.atr_multiplier
            else:
                return current_price * self.config.trailing_percentage

        else:
            return current_price * self.config.trailing_percentage

    def _calculate_new_stop(
        self,
        state: TrailingStopState,
        current_price: float,
        market_data: Optional[pd.DataFrame],
    ) -> float:
        """计算新的止损价

        Args:
            state: 当前状态
            current_price: 当前价格
            market_data: 市场数据

        Returns:
            新止损价
        """
        if self.config.method == TrailingMethod.FIXED_PERCENTAGE:
            return state.highest_price * (1 - self.config.trailing_percentage)

        elif self.config.method == TrailingMethod.HIGH_WATER_MARK:
            # 基于高水位标记
            return state.highest_price - state.trail_amount

        elif self.config.method == TrailingMethod.VOLATILITY_BASED:
            if market_data is not None:
                volatility = self._calculate_volatility(market_data)
                return state.highest_price * (1 - 2 * volatility)
            else:
                return state.highest_price * (1 - self.config.trailing_percentage)

        elif self.config.method == TrailingMethod.ATR_BASED:
            if market_data is not None:
                atr = self._calculate_atr(market_data)
                return state.highest_price - self.config.atr_multiplier * atr
            else:
                return state.highest_price * (1 - self.config.trailing_percentage)

        elif self.config.method == TrailingMethod.CHANDELIER_EXIT:
            if market_data is not None:
                return self._calculate_chandelier_stop(market_data, state.highest_price)
            else:
                return state.highest_price * (1 - self.config.trailing_percentage)

        elif self.config.method == TrailingMethod.PARABOLIC_SAR:
            if market_data is not None:
                return self._calculate_sar_stop(market_data, current_price)
            else:
                return state.highest_price * (1 - self.config.trailing_percentage)

        else:
            return state.highest_price * (1 - self.config.trailing_percentage)

    def _apply_step_trailing(self, current_stop: float, new_stop: float) -> float:
        """应用阶梯式追踪

        Args:
            current_stop: 当前止损
            new_stop: 新止损

        Returns:
            调整后的止损
        """
        # 计算阶梯数
        diff = new_stop - current_stop
        steps = int(diff / (current_stop * self.config.step_size))

        if steps > 0:
            return current_stop * (1 + steps * self.config.step_size)
        else:
            return current_stop

    def _calculate_true_range(self, market_data: pd.DataFrame) -> pd.Series:
        """计算真实波幅

        Args:
            market_data: 市场数据

        Returns:
            真实波幅序列
        """
        high = market_data["high"]
        low = market_data["low"]
        close = market_data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """计算ATR

        Args:
            market_data: 市场数据
            period: 周期

        Returns:
            ATR值
        """
        tr = self._calculate_true_range(market_data)
        return tr.rolling(window=period).mean().iloc[-1]

    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """计算波动率

        Args:
            market_data: 市场数据

        Returns:
            波动率
        """
        returns = market_data["close"].pct_change()
        return returns.rolling(window=self.config.volatility_lookback).std().iloc[-1]

    def _calculate_chandelier_stop(
        self, market_data: pd.DataFrame, highest_price: float
    ) -> float:
        """计算Chandelier止损

        Args:
            market_data: 市场数据
            highest_price: 最高价

        Returns:
            止损价
        """
        atr = self._calculate_atr(market_data)
        return highest_price - self.config.atr_multiplier * atr

    def _calculate_sar_stop(
        self, market_data: pd.DataFrame, current_price: float
    ) -> float:
        """计算SAR止损

        Args:
            market_data: 市场数据
            current_price: 当前价格

        Returns:
            止损价
        """
        if len(market_data) < 2:
            return current_price * (1 - self.config.trailing_percentage)

        sar = self.calculate_parabolic_sar(
            market_data["high"],
            market_data["low"],
            self.config.sar_acceleration,
            self.config.sar_max_acceleration,
        )

        return sar.iloc[-1]

    def _backtest_trailing_params(
        self,
        historical_data: pd.DataFrame,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
        trail_pct: float,
        atr_mult: float,
        activation: float,
    ) -> float:
        """回测追踪参数

        Args:
            historical_data: 历史数据
            entry_signals: 入场信号
            exit_signals: 出场信号
            trail_pct: 追踪百分比
            atr_mult: ATR倍数
            activation: 激活阈值

        Returns:
            性能分数
        """
        # 简化的回测逻辑
        total_return = 0.0
        n_trades = 0

        in_position = False
        entry_price = 0.0
        stop_price = 0.0

        for i in range(len(historical_data)):
            if not in_position and entry_signals.iloc[i]:
                # 入场
                in_position = True
                entry_price = historical_data["close"].iloc[i]
                stop_price = entry_price * (1 - trail_pct)
                n_trades += 1

            elif in_position:
                current_price = historical_data["close"].iloc[i]
                high_price = historical_data["high"].iloc[i]

                # 更新追踪止损
                if current_price > entry_price * (1 + activation):
                    new_stop = high_price * (1 - trail_pct)
                    stop_price = max(stop_price, new_stop)

                # 检查出场
                if current_price <= stop_price or exit_signals.iloc[i]:
                    # 出场
                    in_position = False
                    total_return += (current_price - entry_price) / entry_price

        # 计算性能分数（简化版）
        if n_trades > 0:
            avg_return = total_return / n_trades
            return avg_return
        else:
            return 0.0


# 模块级别函数
def create_trailing_stop(
    symbol: str,
    entry_price: float,
    initial_stop: float,
    config: Optional[TrailingStopConfig] = None,
) -> TrailingStop:
    """创建追踪止损的便捷函数

    Args:
        symbol: 标的代码
        entry_price: 入场价格
        initial_stop: 初始止损
        config: 配置

    Returns:
        追踪止损实例
    """
    trailing = TrailingStop(config)
    trailing.initialize_trailing_stop(symbol, entry_price, initial_stop, entry_price)
    return trailing
