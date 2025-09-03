"""
投资组合监控器模块
实时监控投资组合状态和性能
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.constants import TRADING_DAYS_PER_YEAR
from common.data_structures import Position, Signal
from common.exceptions import ModelError
from common.logging_system import setup_logger

logger = setup_logger("portfolio_monitor")


class MonitoringFrequency(Enum):
    """监控频率枚举"""

    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    FIVE_MINUTES = "5min"
    HOURLY = "hourly"
    DAILY = "daily"


class PortfolioStatus(Enum):
    """投资组合状态枚举"""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    HALTED = "halted"


@dataclass
class PortfolioMetrics:
    """投资组合指标"""

    timestamp: datetime
    total_value: float
    cash_balance: float
    positions_value: float
    daily_pnl: float
    daily_return: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    var_95: float
    leverage: float
    exposure: Dict[str, float]
    concentration: Dict[str, float]
    turnover: float


@dataclass
class PositionMetrics:
    """持仓指标"""

    symbol: str
    quantity: int
    market_value: float
    cost_basis: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float
    contribution_to_return: float
    contribution_to_risk: float
    days_held: int
    beta: float
    correlation_to_portfolio: float


@dataclass
class MonitoringConfig:
    """监控配置"""

    frequency: MonitoringFrequency = MonitoringFrequency.MINUTE
    metrics_window: int = 252  # 指标计算窗口
    history_size: int = 10000  # 历史数据保存大小
    enable_real_time: bool = True
    enable_alerts: bool = True
    save_snapshots: bool = True
    snapshot_interval: int = 300  # 快照间隔（秒）
    calculate_attribution: bool = True
    track_transactions: bool = True
    benchmark_symbol: Optional[str] = None


@dataclass
class PortfolioSnapshot:
    """投资组合快照"""

    timestamp: datetime
    metrics: PortfolioMetrics
    positions: List[PositionMetrics]
    pending_orders: List[Dict[str, Any]]
    active_signals: List[Signal]
    risk_limits: Dict[str, float]
    status: PortfolioStatus
    alerts: List[str]


class PortfolioMonitor:
    """投资组合监控器类"""

    def __init__(self, config: Optional[MonitoringConfig] = None):
        """初始化投资组合监控器

        Args:
            config: 监控配置
        """
        self.config = config or MonitoringConfig()
        self.current_metrics: Optional[PortfolioMetrics] = None
        self.position_metrics: Dict[str, PositionMetrics] = {}
        self.metrics_history: deque = deque(maxlen=self.config.history_size)
        self.snapshots: List[PortfolioSnapshot] = []
        self.status = PortfolioStatus.NORMAL
        self.monitoring_active = False
        self.last_snapshot_time = datetime.now()
        self.callbacks: Dict[str, List[Callable]] = {
            "on_update": [],
            "on_alert": [],
            "on_snapshot": [],
        }

    async def start_monitoring(
        self, portfolio_getter: Callable, market_data_getter: Callable
    ) -> None:
        """启动监控

        Args:
            portfolio_getter: 获取投资组合数据的函数
            market_data_getter: 获取市场数据的函数
        """
        logger.info("Starting portfolio monitoring")
        self.monitoring_active = True

        while self.monitoring_active:
            try:
                # 获取最新数据
                portfolio_data = await portfolio_getter()
                market_data = await market_data_getter()

                # 更新指标
                self._update_metrics(portfolio_data, market_data)

                # 检查快照
                if self._should_take_snapshot():
                    snapshot = self._create_snapshot(portfolio_data)
                    self.snapshots.append(snapshot)
                    await self._trigger_callbacks("on_snapshot", snapshot)

                # 触发更新回调
                await self._trigger_callbacks("on_update", self.current_metrics)

                # 等待下一个周期
                await asyncio.sleep(self._get_monitoring_interval())

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)

    def stop_monitoring(self) -> None:
        """停止监控"""
        logger.info("Stopping portfolio monitoring")
        self.monitoring_active = False

    def update_portfolio_metrics(
        self,
        positions: List[Position],
        cash_balance: float,
        market_prices: Dict[str, float],
    ) -> PortfolioMetrics:
        """更新投资组合指标

        Args:
            positions: 持仓列表
            cash_balance: 现金余额
            market_prices: 市场价格

        Returns:
            投资组合指标
        """
        logger.debug("Updating portfolio metrics")

        # 计算总价值
        positions_value = sum(
            pos.quantity * market_prices.get(pos.symbol, pos.current_price)
            for pos in positions
        )
        total_value = cash_balance + positions_value

        # 计算PnL
        unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        realized_pnl = sum(pos.realized_pnl for pos in positions)
        total_pnl = unrealized_pnl + realized_pnl

        # 计算收益率
        if len(self.metrics_history) > 0:
            prev_value = self.metrics_history[-1].total_value
            daily_pnl = total_value - prev_value
            daily_return = daily_pnl / prev_value if prev_value > 0 else 0
        else:
            daily_pnl = 0
            daily_return = 0

        # 计算风险指标
        returns = self._calculate_returns_series()
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown, current_drawdown = self._calculate_drawdowns(returns)
        var_95 = self._calculate_var(returns, 0.95)

        # 计算暴露和集中度
        exposure = self._calculate_exposure(positions, total_value)
        concentration = self._calculate_concentration(positions, total_value)

        # 计算换手率
        turnover = self._calculate_turnover(positions)

        # 计算胜率
        win_rate = self._calculate_win_rate(positions)

        # 计算杠杆
        leverage = positions_value / total_value if total_value > 0 else 0

        metrics = PortfolioMetrics(
            timestamp=datetime.now(),
            total_value=total_value,
            cash_balance=cash_balance,
            positions_value=positions_value,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_pnl=total_pnl,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            var_95=var_95,
            leverage=leverage,
            exposure=exposure,
            concentration=concentration,
            turnover=turnover,
        )

        self.current_metrics = metrics
        self.metrics_history.append(metrics)

        return metrics

    def update_position_metrics(
        self, position: Position, market_price: float, portfolio_value: float
    ) -> PositionMetrics:
        """更新持仓指标

        Args:
            position: 持仓对象
            market_price: 市场价格
            portfolio_value: 组合总价值

        Returns:
            持仓指标
        """
        market_value = position.quantity * market_price
        cost_basis = position.quantity * position.avg_cost
        unrealized_pnl = market_value - cost_basis
        unrealized_pnl_pct = unrealized_pnl / cost_basis if cost_basis > 0 else 0
        weight = market_value / portfolio_value if portfolio_value > 0 else 0

        # 计算贡献度（简化计算）
        contribution_to_return = weight * unrealized_pnl_pct
        contribution_to_risk = weight * abs(unrealized_pnl_pct)

        # 计算持有天数
        days_held = (datetime.now() - position.open_time).days

        # 计算Beta和相关性（需要历史数据）
        beta = self._calculate_position_beta(position.symbol)
        correlation = self._calculate_position_correlation(position.symbol)

        metrics = PositionMetrics(
            symbol=position.symbol,
            quantity=position.quantity,
            market_value=market_value,
            cost_basis=cost_basis,
            current_price=market_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            weight=weight,
            contribution_to_return=contribution_to_return,
            contribution_to_risk=contribution_to_risk,
            days_held=days_held,
            beta=beta,
            correlation_to_portfolio=correlation,
        )

        self.position_metrics[position.symbol] = metrics

        return metrics

    def calculate_performance_attribution(
        self, positions: List[Position], benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """计算业绩归因

        Args:
            positions: 持仓列表
            benchmark_returns: 基准收益率

        Returns:
            归因分析结果
        """
        attribution = {
            "alpha": 0.0,
            "beta_contribution": 0.0,
            "selection_effect": 0.0,
            "allocation_effect": 0.0,
            "interaction_effect": 0.0,
            "total_effect": 0.0,
        }

        if not positions:
            return attribution

        # 计算各部分贡献（简化版本）
        for position in positions:
            if position.symbol in self.position_metrics:
                pm = self.position_metrics[position.symbol]

                # 选择效应
                attribution["selection_effect"] += pm.contribution_to_return

                # 配置效应
                attribution["allocation_effect"] += pm.weight * 0.01  # 简化计算

        # 总效应
        attribution["total_effect"] = (
            attribution["selection_effect"] + attribution["allocation_effect"]
        )

        # Alpha（超额收益）
        if benchmark_returns is not None:
            portfolio_return = (
                self.current_metrics.daily_return if self.current_metrics else 0
            )
            benchmark_return = (
                benchmark_returns.iloc[-1] if len(benchmark_returns) > 0 else 0
            )
            attribution["alpha"] = portfolio_return - benchmark_return

        return attribution

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常

        Returns:
            异常列表
        """
        anomalies = []

        if not self.current_metrics:
            return anomalies

        # 检测异常大的回撤
        if self.current_metrics.current_drawdown < -0.10:
            anomalies.append(
                {
                    "type": "large_drawdown",
                    "severity": "high",
                    "value": self.current_metrics.current_drawdown,
                    "message": f"Large drawdown detected: {self.current_metrics.current_drawdown:.2%}",
                }
            )

        # 检测异常的换手率
        if self.current_metrics.turnover > 2.0:
            anomalies.append(
                {
                    "type": "high_turnover",
                    "severity": "medium",
                    "value": self.current_metrics.turnover,
                    "message": f"High turnover detected: {self.current_metrics.turnover:.2f}",
                }
            )

        # 检测集中度风险
        if self.current_metrics.concentration:
            max_concentration = max(self.current_metrics.concentration.values())
            if max_concentration > 0.30:
                anomalies.append(
                    {
                        "type": "concentration_risk",
                        "severity": "medium",
                        "value": max_concentration,
                        "message": f"High concentration detected: {max_concentration:.2%}",
                    }
                )

        # 检测异常的PnL
        if len(self.metrics_history) > 10:
            recent_returns = [m.daily_return for m in list(self.metrics_history)[-10:]]
            avg_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)

            if std_return > 0:
                z_score = (self.current_metrics.daily_return - avg_return) / std_return
                if abs(z_score) > 3:
                    anomalies.append(
                        {
                            "type": "abnormal_return",
                            "severity": "high",
                            "value": self.current_metrics.daily_return,
                            "z_score": z_score,
                            "message": f"Abnormal return detected: {self.current_metrics.daily_return:.2%} (z-score: {z_score:.2f})",
                        }
                    )

        return anomalies

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要

        Returns:
            监控摘要
        """
        if not self.current_metrics:
            return {}

        summary = {
            "timestamp": self.current_metrics.timestamp,
            "status": self.status.value,
            "portfolio_value": self.current_metrics.total_value,
            "daily_pnl": self.current_metrics.daily_pnl,
            "daily_return": self.current_metrics.daily_return,
            "total_pnl": self.current_metrics.total_pnl,
            "sharpe_ratio": self.current_metrics.sharpe_ratio,
            "current_drawdown": self.current_metrics.current_drawdown,
            "var_95": self.current_metrics.var_95,
            "n_positions": len(self.position_metrics),
            "top_positions": self._get_top_positions(5),
            "recent_alerts": self._get_recent_alerts(10),
            "anomalies": self.detect_anomalies(),
        }

        return summary

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """注册回调函数

        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"Registered callback for {event_type}")

    def _update_metrics(
        self, portfolio_data: Dict[str, Any], market_data: Dict[str, Any]
    ) -> None:
        """更新指标

        Args:
            portfolio_data: 投资组合数据
            market_data: 市场数据
        """
        positions = portfolio_data.get("positions", [])
        cash_balance = portfolio_data.get("cash_balance", 0)
        market_prices = market_data.get("prices", {})

        # 更新投资组合指标
        self.update_portfolio_metrics(positions, cash_balance, market_prices)

        # 更新各持仓指标
        for position in positions:
            if position.symbol in market_prices:
                self.update_position_metrics(
                    position,
                    market_prices[position.symbol],
                    self.current_metrics.total_value,
                )

    def _should_take_snapshot(self) -> bool:
        """判断是否应该创建快照

        Returns:
            是否创建快照
        """
        if not self.config.save_snapshots:
            return False

        time_since_last = (datetime.now() - self.last_snapshot_time).seconds
        return time_since_last >= self.config.snapshot_interval

    def _create_snapshot(self, portfolio_data: Dict[str, Any]) -> PortfolioSnapshot:
        """创建投资组合快照

        Args:
            portfolio_data: 投资组合数据

        Returns:
            投资组合快照
        """
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            metrics=self.current_metrics,
            positions=list(self.position_metrics.values()),
            pending_orders=portfolio_data.get("pending_orders", []),
            active_signals=portfolio_data.get("active_signals", []),
            risk_limits=portfolio_data.get("risk_limits", {}),
            status=self.status,
            alerts=[],
        )

        self.last_snapshot_time = datetime.now()

        return snapshot

    async def _trigger_callbacks(self, event_type: str, data: Any) -> None:
        """触发回调函数

        Args:
            event_type: 事件类型
            data: 事件数据
        """
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")

    def _get_monitoring_interval(self) -> float:
        """获取监控间隔

        Returns:
            间隔秒数
        """
        intervals = {
            MonitoringFrequency.TICK: 0.1,
            MonitoringFrequency.SECOND: 1,
            MonitoringFrequency.MINUTE: 60,
            MonitoringFrequency.FIVE_MINUTES: 300,
            MonitoringFrequency.HOURLY: 3600,
            MonitoringFrequency.DAILY: 86400,
        }

        return intervals.get(self.config.frequency, 60)

    def _calculate_returns_series(self) -> pd.Series:
        """计算收益率序列

        Returns:
            收益率序列
        """
        if len(self.metrics_history) < 2:
            return pd.Series()

        returns = []
        history_list = list(self.metrics_history)

        for i in range(1, len(history_list)):
            prev_value = history_list[i - 1].total_value
            curr_value = history_list[i].total_value

            if prev_value > 0:
                ret = (curr_value - prev_value) / prev_value
                returns.append(ret)

        return pd.Series(returns)

    def _calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """计算夏普比率

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率

        Returns:
            夏普比率
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / TRADING_DAYS_PER_YEAR

        if excess_returns.std() > 0:
            return (
                excess_returns.mean()
                / excess_returns.std()
                * np.sqrt(TRADING_DAYS_PER_YEAR)
            )
        else:
            return 0.0

    def _calculate_drawdowns(self, returns: pd.Series) -> Tuple[float, float]:
        """计算回撤

        Args:
            returns: 收益率序列

        Returns:
            (最大回撤, 当前回撤)
        """
        if len(returns) == 0:
            return 0.0, 0.0

        # 计算累计收益
        cum_returns = (1 + returns).cumprod()

        # 计算回撤
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max

        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0

        return max_drawdown, current_drawdown

    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """计算VaR

        Args:
            returns: 收益率序列
            confidence_level: 置信水平

        Returns:
            VaR值
        """
        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, (1 - confidence_level) * 100)

    def _calculate_exposure(
        self, positions: List[Position], total_value: float
    ) -> Dict[str, float]:
        """计算暴露度

        Args:
            positions: 持仓列表
            total_value: 总价值

        Returns:
            暴露度字典
        """
        exposure = {"long": 0.0, "short": 0.0, "net": 0.0, "gross": 0.0}

        for position in positions:
            value = position.quantity * position.current_price
            if position.quantity > 0:
                exposure["long"] += value
            else:
                exposure["short"] += abs(value)

        exposure["gross"] = exposure["long"] + exposure["short"]
        exposure["net"] = exposure["long"] - exposure["short"]

        # 转换为百分比
        if total_value > 0:
            for key in exposure:
                exposure[key] /= total_value

        return exposure

    def _calculate_concentration(
        self, positions: List[Position], total_value: float
    ) -> Dict[str, float]:
        """计算集中度

        Args:
            positions: 持仓列表
            total_value: 总价值

        Returns:
            集中度字典
        """
        concentration = {}

        for position in positions:
            value = abs(position.quantity * position.current_price)
            weight = value / total_value if total_value > 0 else 0
            concentration[position.symbol] = weight

        return concentration

    def _calculate_turnover(self, positions: List[Position]) -> float:
        """计算换手率

        Args:
            positions: 持仓列表

        Returns:
            换手率
        """
        # 简化计算：基于持仓天数
        if not positions:
            return 0.0

        avg_holding_days = np.mean(
            [(datetime.now() - pos.open_time).days for pos in positions]
        )

        if avg_holding_days > 0:
            return TRADING_DAYS_PER_YEAR / avg_holding_days
        else:
            return 0.0

    def _calculate_win_rate(self, positions: List[Position]) -> float:
        """计算胜率

        Args:
            positions: 持仓列表

        Returns:
            胜率
        """
        if not positions:
            return 0.0

        winning_positions = sum(1 for pos in positions if pos.unrealized_pnl > 0)

        return winning_positions / len(positions)

    def _calculate_position_beta(self, symbol: str) -> float:
        """计算持仓Beta

        Args:
            symbol: 标的代码

        Returns:
            Beta值
        """
        # 简化：返回默认值
        return 1.0

    def _calculate_position_correlation(self, symbol: str) -> float:
        """计算持仓与组合的相关性

        Args:
            symbol: 标的代码

        Returns:
            相关系数
        """
        # 简化：返回默认值
        return 0.5

    def _get_top_positions(self, n: int) -> List[Dict[str, Any]]:
        """获取前N大持仓

        Args:
            n: 数量

        Returns:
            持仓列表
        """
        sorted_positions = sorted(
            self.position_metrics.values(), key=lambda x: x.market_value, reverse=True
        )

        return [
            {
                "symbol": p.symbol,
                "weight": p.weight,
                "pnl": p.unrealized_pnl,
                "return": p.unrealized_pnl_pct,
            }
            for p in sorted_positions[:n]
        ]

    def _get_recent_alerts(self, n: int) -> List[str]:
        """获取最近的预警

        Args:
            n: 数量

        Returns:
            预警列表
        """
        # 从快照中提取预警
        alerts = []

        for snapshot in reversed(self.snapshots[-n:]):
            alerts.extend(snapshot.alerts)

        return alerts[-n:]


# 模块级别函数
def create_portfolio_monitor(
    config: Optional[MonitoringConfig] = None,
) -> PortfolioMonitor:
    """创建投资组合监控器的便捷函数

    Args:
        config: 监控配置

    Returns:
        投资组合监控器实例
    """
    return PortfolioMonitor(config)
