"""
策略性能评估器模块
提供全面的策略性能评估指标
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from common.logging_system import setup_logger

logger = setup_logger("performance_evaluator")


@dataclass
class PerformanceMetrics:
    """性能评估指标"""

    # 收益指标
    total_return: float
    annual_return: float
    cumulative_return: float

    # 风险指标
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int

    # 风险调整收益指标
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # 交易统计
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # 盈亏统计
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_return: float

    # 其他指标
    max_consecutive_wins: int
    max_consecutive_losses: int
    recovery_factor: float

    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return self.__dict__.copy()


class PerformanceEvaluator:
    """策略性能评估器"""

    def __init__(self, annual_trading_days: int = 252):
        """初始化性能评估器

        Args:
            annual_trading_days: 年交易日数
        """
        self.annual_trading_days = annual_trading_days

    def evaluate(
        self,
        returns: pd.Series,
        trades: Optional[List[Dict]] = None,
        risk_free_rate: float = 0.03,
    ) -> PerformanceMetrics:
        """评估策略性能

        Args:
            returns: 收益率序列
            trades: 交易记录列表（可选）
            risk_free_rate: 无风险利率

        Returns:
            性能指标
        """
        # 收益指标
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        annual_return = (1 + total_return) ** (self.annual_trading_days / n_periods) - 1
        cumulative_return = total_return

        # 风险指标
        volatility = returns.std() * np.sqrt(self.annual_trading_days)
        max_dd, max_dd_duration = self._calculate_drawdown(returns)

        # 风险调整收益指标
        sharpe_ratio = self._calculate_sharpe(returns, risk_free_rate)
        sortino_ratio = self._calculate_sortino(returns, risk_free_rate)
        calmar_ratio = annual_return / max_dd if max_dd > 0 else 0
        omega_ratio = self._calculate_omega(returns, risk_free_rate)

        # 交易统计
        if trades:
            trade_stats = self._analyze_trades(trades)
        else:
            trade_stats = self._analyze_returns_as_trades(returns)

        # 其他指标
        max_consec_wins, max_consec_losses = self._calculate_consecutive_streaks(
            returns
        )
        recovery_factor = total_return / max_dd if max_dd > 0 else 0

        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            cumulative_return=cumulative_return,
            volatility=volatility,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            **trade_stats,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            recovery_factor=recovery_factor,
        )

    def _calculate_drawdown(self, returns: pd.Series) -> tuple:
        """计算最大回撤

        Args:
            returns: 收益率序列

        Returns:
            (最大回撤, 最大回撤持续期)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_dd = abs(drawdown.min())

        # 计算最大回撤持续期
        dd_duration = 0
        current_dd_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                dd_duration = max(dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0

        return max_dd, dd_duration

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float) -> float:
        """计算夏普比率

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率

        Returns:
            夏普比率
        """
        if returns.std() == 0:
            return 0.0

        excess_returns = returns.mean() - risk_free_rate / self.annual_trading_days
        return np.sqrt(self.annual_trading_days) * excess_returns / returns.std()

    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float) -> float:
        """计算索提诺比率

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率

        Returns:
            索提诺比率
        """
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        excess_returns = returns.mean() - risk_free_rate / self.annual_trading_days
        downside_std = downside_returns.std()

        return np.sqrt(self.annual_trading_days) * excess_returns / downside_std

    def _calculate_omega(self, returns: pd.Series, threshold: float) -> float:
        """计算Omega比率

        Args:
            returns: 收益率序列
            threshold: 阈值收益率

        Returns:
            Omega比率
        """
        threshold_daily = threshold / self.annual_trading_days
        gains = returns[returns > threshold_daily].sum()
        losses = abs(returns[returns < threshold_daily].sum())

        if losses == 0:
            return float("inf") if gains > 0 else 0.0

        return gains / losses

    def _analyze_trades(self, trades: List[Dict]) -> Dict:
        """分析交易记录

        Args:
            trades: 交易记录列表

        Returns:
            交易统计字典
        """
        if not trades:
            return self._empty_trade_stats()

        total_trades = len(trades)
        trade_returns = [t.get("return", 0) for t in trades]

        winning_trades = sum(1 for r in trade_returns if r > 0)
        losing_trades = sum(1 for r in trade_returns if r < 0)
        win_rate = winning_trades / max(1, total_trades)

        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": min(profit_factor, 100),  # 限制最大值
            "avg_trade_return": avg_trade_return,
        }

    def _analyze_returns_as_trades(self, returns: pd.Series) -> Dict:
        """将收益率序列视为交易进行分析

        Args:
            returns: 收益率序列

        Returns:
            交易统计字典
        """
        non_zero_returns = returns[returns != 0]

        if len(non_zero_returns) == 0:
            return self._empty_trade_stats()

        total_trades = len(non_zero_returns)
        winning_trades = sum(non_zero_returns > 0)
        losing_trades = sum(non_zero_returns < 0)
        win_rate = winning_trades / total_trades

        wins = non_zero_returns[non_zero_returns > 0]
        losses = non_zero_returns[non_zero_returns < 0]

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        avg_trade_return = non_zero_returns.mean()

        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": min(profit_factor, 100),
            "avg_trade_return": avg_trade_return,
        }

    def _empty_trade_stats(self) -> Dict:
        """返回空的交易统计"""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_trade_return": 0.0,
        }

    def _calculate_consecutive_streaks(self, returns: pd.Series) -> tuple:
        """计算连续盈亏次数

        Args:
            returns: 收益率序列

        Returns:
            (最大连续盈利, 最大连续亏损)
        """
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for ret in returns:
            if ret > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif ret < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return max_wins, max_losses
