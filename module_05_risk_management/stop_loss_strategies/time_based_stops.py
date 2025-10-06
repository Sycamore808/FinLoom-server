"""
时间止损策略模块
基于持仓时间的止损策略
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.exceptions import ModelError
from common.logging_system import setup_logger

logger = setup_logger("time_based_stops")


class TimeStopType(Enum):
    """时间止损类型"""

    FIXED_DURATION = "fixed_duration"
    PERFORMANCE_BASED = "performance_based"
    DECAY_FUNCTION = "decay_function"
    MARKET_HOURS = "market_hours"
    CALENDAR_BASED = "calendar_based"
    ADAPTIVE = "adaptive"


@dataclass
class TimeStopConfig:
    """时间止损配置"""

    stop_type: TimeStopType = TimeStopType.FIXED_DURATION
    max_holding_days: int = 30
    min_holding_days: int = 1
    decay_rate: float = 0.95  # 每日衰减率
    performance_threshold: float = -0.05  # -5%触发时间止损
    market_hours_limit: int = 6  # 交易时间限制（小时）
    use_trading_days_only: bool = True
    weekend_counting: bool = False
    holiday_calendar: Optional[List[datetime]] = None
    adaptive_factor: float = 1.0


@dataclass
class TimeStopState:
    """时间止损状态"""

    symbol: str
    entry_time: datetime
    current_time: datetime
    holding_duration: timedelta
    trading_days_held: int
    remaining_time: timedelta
    time_decay_factor: float
    should_exit: bool
    exit_reason: str


@dataclass
class TimeStopAnalysis:
    """时间止损分析"""

    optimal_holding_period: int
    average_holding_period: float
    time_decay_curve: pd.Series
    exit_time_distribution: pd.Series
    performance_by_duration: pd.DataFrame


class TimeBasedStops:
    """时间止损策略类"""

    def __init__(self, config: Optional[TimeStopConfig] = None):
        """初始化时间止损

        Args:
            config: 时间止损配置
        """
        self.config = config or TimeStopConfig()
        self.active_positions: Dict[str, TimeStopState] = {}
        self.exit_history: List[TimeStopState] = []
        self.performance_stats: Dict[str, Any] = {}

    def check_time_stop(
        self,
        symbol: str,
        entry_time: datetime,
        current_time: datetime,
        current_performance: float = 0.0,
    ) -> TimeStopState:
        """检查时间止损

        Args:
            symbol: 标的代码
            entry_time: 入场时间
            current_time: 当前时间
            current_performance: 当前表现

        Returns:
            时间止损状态
        """
        logger.info(f"Checking time stop for {symbol}")

        # 计算持仓时间
        holding_duration = current_time - entry_time
        trading_days = self._calculate_trading_days(entry_time, current_time)

        # 检查是否应该退出
        should_exit, exit_reason = self._evaluate_exit_conditions(
            holding_duration, trading_days, current_performance
        )

        # 计算时间衰减因子
        decay_factor = self._calculate_decay_factor(trading_days)

        # 计算剩余时间
        remaining_time = self._calculate_remaining_time(trading_days)

        state = TimeStopState(
            symbol=symbol,
            entry_time=entry_time,
            current_time=current_time,
            holding_duration=holding_duration,
            trading_days_held=trading_days,
            remaining_time=remaining_time,
            time_decay_factor=decay_factor,
            should_exit=should_exit,
            exit_reason=exit_reason,
        )

        # 更新活跃持仓
        self.active_positions[symbol] = state

        if should_exit:
            logger.warning(f"Time stop triggered for {symbol}: {exit_reason}")
            self.exit_history.append(state)

        return state

    def calculate_optimal_holding_period(self, historical_trades: pd.DataFrame) -> int:
        """计算最优持仓期

        Args:
            historical_trades: 历史交易数据

        Returns:
            最优持仓天数
        """
        logger.info("Calculating optimal holding period...")

        # 按持仓期分组
        trades_by_duration = historical_trades.groupby("holding_days")

        # 计算每个持仓期的平均收益
        avg_returns = trades_by_duration["return"].mean()

        # 计算风险调整收益
        sharpe_by_duration = trades_by_duration.apply(
            lambda x: x["return"].mean() / x["return"].std()
            if x["return"].std() > 0
            else 0
        )

        # 找出最优持仓期
        optimal_period = sharpe_by_duration.idxmax()

        logger.info(f"Optimal holding period: {optimal_period} days")

        return int(optimal_period)

    def apply_time_decay_adjustment(
        self, position_size: float, holding_days: int
    ) -> float:
        """应用时间衰减调整

        Args:
            position_size: 原始仓位大小
            holding_days: 持仓天数

        Returns:
            调整后的仓位大小
        """
        decay_factor = self._calculate_decay_factor(holding_days)
        adjusted_size = position_size * decay_factor

        return adjusted_size

    def generate_time_exit_signals(
        self, positions: Dict[str, Dict[str, Any]], current_time: datetime
    ) -> List[str]:
        """生成时间退出信号

        Args:
            positions: 持仓字典
            current_time: 当前时间

        Returns:
            需要退出的标的列表
        """
        exit_signals = []

        for symbol, position in positions.items():
            entry_time = position["entry_time"]
            current_performance = position.get("return", 0.0)

            state = self.check_time_stop(
                symbol, entry_time, current_time, current_performance
            )

            if state.should_exit:
                exit_signals.append(symbol)

        return exit_signals

    def analyze_time_patterns(self, trade_history: pd.DataFrame) -> TimeStopAnalysis:
        """分析时间模式

        Args:
            trade_history: 交易历史

        Returns:
            时间止损分析结果
        """
        logger.info("Analyzing time patterns...")

        # 计算最优持仓期
        optimal_period = self.calculate_optimal_holding_period(trade_history)

        # 平均持仓期
        avg_holding = trade_history["holding_days"].mean()

        # 时间衰减曲线
        max_days = trade_history["holding_days"].max()
        decay_curve = pd.Series(
            [self._calculate_decay_factor(d) for d in range(max_days + 1)],
            index=range(max_days + 1),
        )

        # 退出时间分布
        exit_distribution = trade_history["holding_days"].value_counts().sort_index()

        # 按持仓期的表现
        performance_by_duration = trade_history.groupby("holding_days").agg(
            {"return": ["mean", "std", "count"], "max_drawdown": "mean"}
        )

        return TimeStopAnalysis(
            optimal_holding_period=optimal_period,
            average_holding_period=avg_holding,
            time_decay_curve=decay_curve,
            exit_time_distribution=exit_distribution,
            performance_by_duration=performance_by_duration,
        )

    def _evaluate_exit_conditions(
        self, holding_duration: timedelta, trading_days: int, performance: float
    ) -> Tuple[bool, str]:
        """评估退出条件

        Args:
            holding_duration: 持仓时长
            trading_days: 交易日数
            performance: 当前表现

        Returns:
            (是否退出, 退出原因)
        """
        if self.config.stop_type == TimeStopType.FIXED_DURATION:
            if trading_days >= self.config.max_holding_days:
                return True, f"Maximum holding period reached ({trading_days} days)"

        elif self.config.stop_type == TimeStopType.PERFORMANCE_BASED:
            if trading_days >= self.config.max_holding_days:
                if performance < self.config.performance_threshold:
                    return True, f"Poor performance after {trading_days} days"

        elif self.config.stop_type == TimeStopType.DECAY_FUNCTION:
            decay = self._calculate_decay_factor(trading_days)
            if decay < 0.1:  # 衰减到10%以下
                return True, f"Position decayed below threshold"

        elif self.config.stop_type == TimeStopType.MARKET_HOURS:
            hours = holding_duration.total_seconds() / 3600
            if hours >= self.config.market_hours_limit:
                return True, f"Market hours limit reached ({hours:.1f} hours)"

        elif self.config.stop_type == TimeStopType.ADAPTIVE:
            adaptive_limit = self.config.max_holding_days * self.config.adaptive_factor
            if trading_days >= adaptive_limit:
                return True, f"Adaptive time limit reached ({trading_days} days)"

        return False, ""

    def _calculate_trading_days(self, start_time: datetime, end_time: datetime) -> int:
        """计算交易日数

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            交易日数
        """
        if not self.config.use_trading_days_only:
            return (end_time - start_time).days

        # 计算交易日
        trading_days = 0
        current = start_time.date()
        end_date = end_time.date()

        while current <= end_date:
            # 检查是否是交易日
            if self._is_trading_day(current):
                trading_days += 1
            current += timedelta(days=1)

        return trading_days

    def _is_trading_day(self, date: datetime) -> bool:
        """判断是否是交易日

        Args:
            date: 日期

        Returns:
            是否是交易日
        """
        # 周末检查
        if not self.config.weekend_counting:
            if date.weekday() >= 5:  # 周六或周日
                return False

        # 假期检查
        if self.config.holiday_calendar:
            if date in self.config.holiday_calendar:
                return False

        return True

    def _calculate_decay_factor(self, holding_days: int) -> float:
        """计算衰减因子

        Args:
            holding_days: 持仓天数

        Returns:
            衰减因子
        """
        if self.config.stop_type == TimeStopType.DECAY_FUNCTION:
            return self.config.decay_rate**holding_days
        else:
            # 线性衰减
            remaining_ratio = max(0, 1 - holding_days / self.config.max_holding_days)
            return remaining_ratio

    def _calculate_remaining_time(self, trading_days_held: int) -> timedelta:
        """计算剩余时间

        Args:
            trading_days_held: 已持有交易日数

        Returns:
            剩余时间
        """
        remaining_days = max(0, self.config.max_holding_days - trading_days_held)
        return timedelta(days=remaining_days)


# 模块级别函数
def check_time_based_stop(
    symbol: str,
    entry_time: datetime,
    current_time: datetime,
    config: Optional[TimeStopConfig] = None,
) -> bool:
    """检查时间止损的便捷函数

    Args:
        symbol: 标的代码
        entry_time: 入场时间
        current_time: 当前时间
        config: 配置

    Returns:
        是否应该止损
    """
    time_stops = TimeBasedStops(config)
    state = time_stops.check_time_stop(symbol, entry_time, current_time)
    return state.should_exit
