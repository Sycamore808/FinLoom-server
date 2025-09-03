"""
回测引擎模块
提供高保真的历史回测功能，考虑交易成本、滑点、市场冲击等真实因素
"""

import heapq
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from common.constants import (
    DEFAULT_SLIPPAGE_BPS,
    MIN_POSITION_SIZE,
    TRADING_DAYS_PER_YEAR,
)
from common.data_structures import MarketData, Position, Signal
from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("backtest_engine")


class OrderType(Enum):
    """订单类型枚举"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """订单状态枚举"""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """订单数据类"""

    order_id: str
    timestamp: datetime
    symbol: str
    order_type: OrderType
    side: str  # 'BUY' or 'SELL'
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"  # 'DAY', 'GTC', 'IOC', 'FOK'
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_filled(self) -> bool:
        """判断订单是否完全成交"""
        return self.status == OrderStatus.FILLED

    def is_active(self) -> bool:
        """判断订单是否仍然有效"""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL_FILLED,
        ]


@dataclass
class Trade:
    """成交记录数据类"""

    trade_id: str
    order_id: str
    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    slippage: float
    market_impact: float

    @property
    def total_cost(self) -> float:
        """计算总成本"""
        base_cost = self.quantity * self.price
        if self.side == "BUY":
            return base_cost + self.commission + self.slippage + self.market_impact
        else:
            return base_cost - self.commission - self.slippage - self.market_impact


@dataclass
class BacktestConfig:
    """回测配置数据类"""

    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float = 0.001  # 0.1%
    slippage_model: str = "fixed"  # 'fixed', 'linear', 'square_root'
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    market_impact_model: str = "linear"  # 'none', 'linear', 'almgren_chriss'
    impact_coefficient: float = 0.1
    min_commission: float = 1.0
    allow_short: bool = False
    use_adjusted_close: bool = True
    rebalance_frequency: str = "daily"  # 'daily', 'weekly', 'monthly'
    execution_delay: int = 0  # bars延迟
    random_seed: Optional[int] = None

    def validate(self) -> bool:
        """验证配置有效性"""
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.commission_rate < 0:
            raise ValueError("Commission rate cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("Slippage cannot be negative")
        return True


@dataclass
class BacktestResult:
    """回测结果数据类"""

    config: BacktestConfig
    equity_curve: pd.DataFrame
    trades: List[Trade]
    positions: Dict[str, Position]
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    execution_metrics: Dict[str, float]
    daily_returns: pd.Series
    monthly_returns: pd.Series
    drawdown_series: pd.Series

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "config": self.config.__dict__,
            "performance_metrics": self.performance_metrics,
            "risk_metrics": self.risk_metrics,
            "execution_metrics": self.execution_metrics,
            "trade_count": len(self.trades),
            "final_equity": self.equity_curve["total_equity"].iloc[-1],
            "total_return": self.performance_metrics.get("total_return", 0),
        }


class BacktestEngine:
    """回测引擎类"""

    def __init__(self, config: BacktestConfig):
        """初始化回测引擎

        Args:
            config: 回测配置
        """
        config.validate()
        self.config = config

        # 账户状态
        self.current_cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Order] = []
        self.order_history: List[Order] = []
        self.trades: List[Trade] = []

        # 市场数据
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_bar_index = 0
        self.current_timestamp: Optional[datetime] = None

        # 性能追踪
        self.equity_curve: List[Dict[str, Any]] = []
        self.daily_pnl: List[float] = []

        # 策略接口
        self.strategy_func: Optional[Callable] = None

        # 事件队列
        self.event_queue: List[Tuple[datetime, str, Any]] = []

        # 随机数生成器（用于模拟）
        if config.random_seed:
            np.random.seed(config.random_seed)

    def load_market_data(
        self, symbols: List[str], data_source: Union[str, Dict[str, pd.DataFrame]]
    ) -> None:
        """加载市场数据

        Args:
            symbols: 标的列表
            data_source: 数据源，可以是文件路径或数据字典
        """
        if isinstance(data_source, str):
            # 从文件加载
            for symbol in symbols:
                file_path = os.path.join(data_source, f"{symbol}.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, parse_dates=["timestamp"])
                    df = df.set_index("timestamp")
                    self.market_data[symbol] = df
                else:
                    logger.warning(f"Data file not found for {symbol}")
        else:
            # 直接使用提供的数据
            self.market_data = data_source

        # 验证数据时间范围
        for symbol, df in self.market_data.items():
            if df.index[0] > self.config.start_date:
                logger.warning(f"Data for {symbol} starts after backtest start date")
            if df.index[-1] < self.config.end_date:
                logger.warning(f"Data for {symbol} ends before backtest end date")

    def set_strategy(self, strategy_func: Callable) -> None:
        """设置策略函数

        Args:
            strategy_func: 策略函数，接收市场数据返回信号列表
        """
        self.strategy_func = strategy_func

    def run(self) -> BacktestResult:
        """运行回测

        Returns:
            回测结果对象
        """
        if not self.strategy_func:
            raise QuantSystemError("Strategy function not set")

        if not self.market_data:
            raise QuantSystemError("Market data not loaded")

        logger.info(
            f"Starting backtest from {self.config.start_date} to {self.config.end_date}"
        )

        # 获取交易日历
        trading_calendar = self._get_trading_calendar()

        # 主回测循环
        for timestamp in trading_calendar:
            self.current_timestamp = timestamp

            # 更新市场数据
            current_data = self._get_current_market_data()

            # 处理挂单
            self._process_pending_orders(current_data)

            # 生成策略信号
            signals = self.strategy_func(
                current_data, self.positions, self.current_cash
            )

            # 执行信号
            for signal in signals:
                self._execute_signal(signal, current_data)

            # 更新持仓市值
            self._update_positions(current_data)

            # 记录权益曲线
            self._record_equity()

            # 处理公司行动（分红、拆股等）
            self._process_corporate_actions(timestamp)

            self.current_bar_index += 1

        # 生成回测结果
        result = self._generate_result()

        logger.info("Backtest completed")
        return result

    def _get_trading_calendar(self) -> List[datetime]:
        """获取交易日历

        Returns:
            交易日期列表
        """
        # 使用第一个标的的时间索引作为基准
        first_symbol = list(self.market_data.keys())[0]
        df = self.market_data[first_symbol]

        # 筛选时间范围
        mask = (df.index >= self.config.start_date) & (df.index <= self.config.end_date)
        trading_days = df.index[mask].tolist()

        return trading_days

    def _get_current_market_data(self) -> Dict[str, MarketData]:
        """获取当前市场数据

        Returns:
            当前各标的的市场数据
        """
        current_data = {}

        for symbol, df in self.market_data.items():
            if self.current_timestamp in df.index:
                row = df.loc[self.current_timestamp]
                current_data[symbol] = MarketData(
                    symbol=symbol,
                    timestamp=self.current_timestamp,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    vwap=row.get("vwap", row["close"]),
                    bid=row.get("bid", row["close"] - 0.01),
                    ask=row.get("ask", row["close"] + 0.01),
                )

        return current_data

    def _execute_signal(
        self, signal: Signal, current_data: Dict[str, MarketData]
    ) -> Optional[Order]:
        """执行交易信号

        Args:
            signal: 交易信号
            current_data: 当前市场数据

        Returns:
            生成的订单对象
        """
        # 验证信号
        if signal.symbol not in current_data:
            logger.warning(f"No market data for {signal.symbol}")
            return None

        # 风险检查
        if not self._pass_risk_check(signal):
            logger.warning(f"Signal failed risk check: {signal}")
            return None

        # 创建订单
        order = Order(
            order_id=f"ORD_{self.current_timestamp}_{signal.signal_id}",
            timestamp=self.current_timestamp,
            symbol=signal.symbol,
            order_type=OrderType.MARKET,  # 默认市价单
            side=signal.action,
            quantity=signal.quantity,
            metadata=signal.metadata,
        )

        # 立即执行市价单
        if order.order_type == OrderType.MARKET:
            self._fill_order(order, current_data[signal.symbol])
        else:
            # 添加到挂单列表
            self.pending_orders.append(order)

        self.order_history.append(order)
        return order

    def _fill_order(self, order: Order, market_data: MarketData) -> Trade:
        """成交订单

        Args:
            order: 订单对象
            market_data: 市场数据

        Returns:
            成交记录
        """
        # 计算成交价格（考虑滑点）
        base_price = market_data.close
        slippage = self._calculate_slippage(order, market_data)

        if order.side == "BUY":
            fill_price = base_price * (1 + slippage)
        else:
            fill_price = base_price * (1 - slippage)

        # 计算佣金
        commission = self._calculate_commission(order.quantity, fill_price)

        # 计算市场冲击
        market_impact = self._calculate_market_impact(order, market_data)

        # 创建成交记录
        trade = Trade(
            trade_id=f"TRD_{self.current_timestamp}_{order.order_id}",
            order_id=order.order_id,
            timestamp=self.current_timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage * base_price,
            market_impact=market_impact,
        )

        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.commission = commission
        order.slippage = slippage * base_price

        # 更新账户
        self._update_account(trade)

        # 记录成交
        self.trades.append(trade)

        logger.debug(
            f"Order filled: {order.symbol} {order.side} {order.quantity} @ {fill_price:.2f}"
        )

        return trade

    def _update_account(self, trade: Trade) -> None:
        """更新账户状态

        Args:
            trade: 成交记录
        """
        total_cost = trade.total_cost

        if trade.side == "BUY":
            # 买入
            self.current_cash -= total_cost

            if trade.symbol in self.positions:
                # 更新现有持仓
                position = self.positions[trade.symbol]
                new_quantity = position.quantity + trade.quantity
                new_cost = (
                    position.avg_cost * position.quantity + total_cost
                ) / new_quantity
                position.quantity = new_quantity
                position.avg_cost = new_cost
            else:
                # 创建新持仓
                self.positions[trade.symbol] = Position(
                    position_id=f"POS_{trade.symbol}_{self.current_timestamp}",
                    symbol=trade.symbol,
                    quantity=trade.quantity,
                    avg_cost=trade.price,
                    current_price=trade.price,
                    market_value=trade.quantity * trade.price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    open_time=self.current_timestamp,
                    last_update=self.current_timestamp,
                )
        else:
            # 卖出
            self.current_cash += total_cost

            if trade.symbol in self.positions:
                position = self.positions[trade.symbol]

                # 计算已实现盈亏
                realized_pnl = (trade.price - position.avg_cost) * trade.quantity
                position.realized_pnl += realized_pnl

                # 更新持仓数量
                position.quantity -= trade.quantity

                # 如果全部卖出，删除持仓
                if position.quantity == 0:
                    del self.positions[trade.symbol]

    def _update_positions(self, current_data: Dict[str, MarketData]) -> None:
        """更新持仓市值

        Args:
            current_data: 当前市场数据
        """
        for symbol, position in self.positions.items():
            if symbol in current_data:
                current_price = current_data[symbol].close
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (
                    current_price - position.avg_cost
                ) * position.quantity
                position.last_update = self.current_timestamp

    def _process_pending_orders(self, current_data: Dict[str, MarketData]) -> None:
        """处理挂单

        Args:
            current_data: 当前市场数据
        """
        filled_orders = []

        for order in self.pending_orders:
            if order.symbol not in current_data:
                continue

            market_data = current_data[order.symbol]

            # 检查限价单
            if order.order_type == OrderType.LIMIT:
                if order.side == "BUY" and market_data.ask <= order.limit_price:
                    self._fill_order(order, market_data)
                    filled_orders.append(order)
                elif order.side == "SELL" and market_data.bid >= order.limit_price:
                    self._fill_order(order, market_data)
                    filled_orders.append(order)

            # 检查止损单
            elif order.order_type == OrderType.STOP:
                if order.side == "BUY" and market_data.close >= order.stop_price:
                    self._fill_order(order, market_data)
                    filled_orders.append(order)
                elif order.side == "SELL" and market_data.close <= order.stop_price:
                    self._fill_order(order, market_data)
                    filled_orders.append(order)

        # 移除已成交的订单
        for order in filled_orders:
            self.pending_orders.remove(order)

    def _calculate_slippage(self, order: Order, market_data: MarketData) -> float:
        """计算滑点

        Args:
            order: 订单
            market_data: 市场数据

        Returns:
            滑点比例
        """
        base_slippage = self.config.slippage_bps / 10000

        if self.config.slippage_model == "fixed":
            return base_slippage

        elif self.config.slippage_model == "linear":
            # 线性滑点模型：与订单大小成正比
            volume_ratio = (
                order.quantity / market_data.volume if market_data.volume > 0 else 0.1
            )
            return base_slippage * (1 + volume_ratio)

        elif self.config.slippage_model == "square_root":
            # 平方根滑点模型
            volume_ratio = (
                order.quantity / market_data.volume if market_data.volume > 0 else 0.1
            )
            return base_slippage * np.sqrt(1 + volume_ratio)

        return base_slippage

    def _calculate_commission(self, quantity: int, price: float) -> float:
        """计算佣金

        Args:
            quantity: 数量
            price: 价格

        Returns:
            佣金金额
        """
        commission = quantity * price * self.config.commission_rate
        return max(commission, self.config.min_commission)

    def _calculate_market_impact(self, order: Order, market_data: MarketData) -> float:
        """计算市场冲击

        Args:
            order: 订单
            market_data: 市场数据

        Returns:
            市场冲击成本
        """
        if self.config.market_impact_model == "none":
            return 0.0

        elif self.config.market_impact_model == "linear":
            # 线性市场冲击模型
            if market_data.volume > 0:
                participation_rate = order.quantity / market_data.volume
                impact = (
                    self.config.impact_coefficient
                    * participation_rate
                    * market_data.close
                )
                return impact
            return 0.0

        elif self.config.market_impact_model == "almgren_chriss":
            # Almgren-Chriss模型（简化版）
            if market_data.volume > 0:
                daily_volume = market_data.volume
                daily_volatility = 0.02  # 假设2%日波动率

                # 永久冲击
                permanent_impact = self.config.impact_coefficient * (
                    order.quantity / daily_volume
                )

                # 临时冲击
                temporary_impact = 0.1 * np.sqrt(order.quantity / daily_volume)

                total_impact = (permanent_impact + temporary_impact) * market_data.close
                return total_impact
            return 0.0

        return 0.0

    def _pass_risk_check(self, signal: Signal) -> bool:
        """风险检查

        Args:
            signal: 交易信号

        Returns:
            是否通过风险检查
        """
        # 检查资金是否充足
        if signal.action == "BUY":
            required_capital = signal.quantity * signal.price
            if required_capital > self.current_cash:
                logger.warning(
                    f"Insufficient cash for buy order: {required_capital} > {self.current_cash}"
                )
                return False

        # 检查是否允许做空
        elif signal.action == "SELL":
            if signal.symbol not in self.positions and not self.config.allow_short:
                logger.warning(f"Short selling not allowed for {signal.symbol}")
                return False

            if signal.symbol in self.positions:
                if signal.quantity > self.positions[signal.symbol].quantity:
                    if not self.config.allow_short:
                        logger.warning(
                            f"Cannot sell more than current position for {signal.symbol}"
                        )
                        return False

        # 检查最小交易量
        if signal.quantity < MIN_POSITION_SIZE:
            logger.warning(
                f"Order quantity below minimum: {signal.quantity} < {MIN_POSITION_SIZE}"
            )
            return False

        return True

    def _record_equity(self) -> None:
        """记录权益曲线"""
        # 计算总市值
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_equity = self.current_cash + total_market_value

        # 记录数据点
        equity_point = {
            "timestamp": self.current_timestamp,
            "cash": self.current_cash,
            "market_value": total_market_value,
            "total_equity": total_equity,
            "position_count": len(self.positions),
            "pending_orders": len(self.pending_orders),
        }

        self.equity_curve.append(equity_point)

        # 计算日收益
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]["total_equity"]
            daily_pnl = total_equity - prev_equity
            self.daily_pnl.append(daily_pnl)

    def _process_corporate_actions(self, timestamp: datetime) -> None:
        """处理公司行动（分红、拆股等）

        Args:
            timestamp: 当前时间戳
        """
        # 这里可以添加分红、拆股等公司行动的处理逻辑
        # 实际实现需要额外的公司行动数据
        pass

    def _generate_result(self) -> BacktestResult:
        """生成回测结果

        Returns:
            回测结果对象
        """
        # 转换权益曲线为DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index("timestamp", inplace=True)

        # 计算日收益率
        equity_df["daily_return"] = equity_df["total_equity"].pct_change()
        daily_returns = equity_df["daily_return"].dropna()

        # 计算月收益率
        monthly_returns = daily_returns.resample("M").apply(
            lambda x: (1 + x).prod() - 1
        )

        # 计算回撤序列
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown_series = (cumulative - running_max) / running_max

        # 计算性能指标
        performance_metrics = self._calculate_performance_metrics(
            equity_df, daily_returns
        )

        # 计算风险指标
        risk_metrics = self._calculate_risk_metrics(daily_returns, drawdown_series)

        # 计算执行指标
        execution_metrics = self._calculate_execution_metrics()

        return BacktestResult(
            config=self.config,
            equity_curve=equity_df,
            trades=self.trades,
            positions=self.positions,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            execution_metrics=execution_metrics,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            drawdown_series=drawdown_series,
        )

    def _calculate_performance_metrics(
        self, equity_df: pd.DataFrame, daily_returns: pd.Series
    ) -> Dict[str, float]:
        """计算性能指标

        Args:
            equity_df: 权益曲线DataFrame
            daily_returns: 日收益率序列

        Returns:
            性能指标字典
        """
        initial_equity = self.config.initial_capital
        final_equity = equity_df["total_equity"].iloc[-1]

        # 基本收益指标
        total_return = (final_equity - initial_equity) / initial_equity

        # 年化收益
        days = len(daily_returns)
        years = days / TRADING_DAYS_PER_YEAR
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 夏普比率
        risk_free_rate = 0.02  # 假设2%无风险利率
        excess_returns = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
        sharpe_ratio = (
            np.sqrt(TRADING_DAYS_PER_YEAR)
            * excess_returns.mean()
            / excess_returns.std()
            if excess_returns.std() > 0
            else 0
        )

        # 索提诺比率
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = (
            np.sqrt(TRADING_DAYS_PER_YEAR) * excess_returns.mean() / downside_std
            if downside_std > 0
            else 0
        )

        # 胜率
        winning_trades = [
            t
            for t in self.trades
            if t.side == "SELL"
            and self.positions.get(
                t.symbol,
                Position(
                    position_id="",
                    symbol="",
                    quantity=0,
                    avg_cost=0,
                    current_price=0,
                    market_value=0,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    open_time=datetime.now(),
                    last_update=datetime.now(),
                ),
            ).realized_pnl
            > 0
        ]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

        # 盈亏比
        profits = [
            t.price
            - self.positions.get(
                t.symbol,
                Position(
                    position_id="",
                    symbol="",
                    quantity=0,
                    avg_cost=t.price,
                    current_price=0,
                    market_value=0,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    open_time=datetime.now(),
                    last_update=datetime.now(),
                ),
            ).avg_cost
            for t in self.trades
            if t.side == "SELL"
        ]
        avg_profit = (
            np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
        )
        avg_loss = (
            abs(np.mean([p for p in profits if p < 0]))
            if any(p < 0 for p in profits)
            else 1
        )
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "total_trades": len(self.trades),
            "trading_days": days,
            "final_equity": final_equity,
        }

    def _calculate_risk_metrics(
        self, daily_returns: pd.Series, drawdown_series: pd.Series
    ) -> Dict[str, float]:
        """计算风险指标

        Args:
            daily_returns: 日收益率序列
            drawdown_series: 回撤序列

        Returns:
            风险指标字典
        """
        # 波动率
        annual_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # 最大回撤
        max_drawdown = drawdown_series.min()

        # 最大回撤持续时间
        drawdown_duration = 0
        current_duration = 0
        for dd in drawdown_series:
            if dd < 0:
                current_duration += 1
                drawdown_duration = max(drawdown_duration, current_duration)
            else:
                current_duration = 0

        # VaR (95%)
        var_95 = daily_returns.quantile(0.05)

        # CVaR (95%)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()

        # 下行偏差
        downside_deviation = (
            daily_returns[daily_returns < 0].std()
            if len(daily_returns[daily_returns < 0]) > 0
            else 0
        )

        # 卡尔玛比率
        calmar_ratio = (
            daily_returns.mean() * TRADING_DAYS_PER_YEAR / abs(max_drawdown)
            if max_drawdown != 0
            else 0
        )

        # 偏度和峰度
        skewness = daily_returns.skew()
        kurtosis = daily_returns.kurtosis()

        return {
            "annual_volatility": annual_volatility,
            "max_drawdown": abs(max_drawdown),
            "drawdown_duration_days": drawdown_duration,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "downside_deviation": downside_deviation,
            "calmar_ratio": calmar_ratio,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    def _calculate_execution_metrics(self) -> Dict[str, float]:
        """计算执行指标

        Returns:
            执行指标字典
        """
        if not self.trades:
            return {
                "avg_slippage": 0,
                "total_slippage": 0,
                "avg_commission": 0,
                "total_commission": 0,
                "avg_market_impact": 0,
                "total_market_impact": 0,
                "avg_trade_size": 0,
            }

        slippages = [t.slippage for t in self.trades]
        commissions = [t.commission for t in self.trades]
        impacts = [t.market_impact for t in self.trades]
        trade_sizes = [t.quantity * t.price for t in self.trades]

        return {
            "avg_slippage": np.mean(slippages),
            "total_slippage": np.sum(slippages),
            "avg_commission": np.mean(commissions),
            "total_commission": np.sum(commissions),
            "avg_market_impact": np.mean(impacts),
            "total_market_impact": np.sum(impacts),
            "avg_trade_size": np.mean(trade_sizes),
            "total_execution_cost": np.sum(slippages)
            + np.sum(commissions)
            + np.sum(impacts),
        }


# 模块级别函数
def create_backtest_engine(config_dict: Dict[str, Any]) -> BacktestEngine:
    """创建回测引擎的便捷函数

    Args:
        config_dict: 配置字典

    Returns:
        回测引擎实例
    """
    config = BacktestConfig(**config_dict)
    return BacktestEngine(config)


def run_simple_backtest(
    strategy_func: Callable,
    market_data: Dict[str, pd.DataFrame],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 1000000,
) -> BacktestResult:
    """运行简单回测的便捷函数

    Args:
        strategy_func: 策略函数
        market_data: 市场数据
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金

    Returns:
        回测结果
    """
    config = BacktestConfig(
        start_date=start_date, end_date=end_date, initial_capital=initial_capital
    )

    engine = BacktestEngine(config)
    engine.load_market_data(list(market_data.keys()), market_data)
    engine.set_strategy(strategy_func)

    return engine.run()
