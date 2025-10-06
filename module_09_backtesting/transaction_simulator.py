"""
交易模拟器模块
提供真实交易环境的高保真模拟，包括订单簿模拟、部分成交、价格滑动等
"""

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from common.constants import DEFAULT_SLIPPAGE_BPS
from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("transaction_simulator")


class OrderBookSide(Enum):
    """订单簿方向枚举"""

    BID = "BID"
    ASK = "ASK"


@dataclass
class OrderBookLevel:
    """订单簿价格层级"""

    price: float
    quantity: int
    order_count: int
    timestamp: datetime

    def __lt__(self, other):
        """用于堆排序"""
        return self.price < other.price


@dataclass
class SimulatedOrderBook:
    """模拟订单簿数据结构"""

    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]  # 买单队列（价格降序）
    asks: List[OrderBookLevel]  # 卖单队列（价格升序）
    last_price: float
    last_volume: int
    total_bid_volume: int
    total_ask_volume: int

    @property
    def best_bid(self) -> Optional[float]:
        """获取最佳买价"""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """获取最佳卖价"""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> float:
        """获取买卖价差"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return 0.0

    @property
    def mid_price(self) -> float:
        """获取中间价"""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        elif self.last_price:
            return self.last_price
        return 0.0


@dataclass
class TransactionResult:
    """交易结果数据结构"""

    order_id: str
    symbol: str
    side: str
    requested_quantity: int
    filled_quantity: int
    average_price: float
    execution_prices: List[Tuple[float, int]]  # (价格, 数量)对列表
    slippage: float
    market_impact: float
    transaction_cost: float
    timestamp: datetime
    execution_time_ms: float
    partial_fill: bool

    @property
    def fill_rate(self) -> float:
        """计算成交率"""
        if self.requested_quantity == 0:
            return 0.0
        return self.filled_quantity / self.requested_quantity


class TransactionSimulator:
    """交易模拟器类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化交易模拟器

        Args:
            config: 配置参数字典
        """
        self.config = config or {}

        # 模拟参数
        self.latency_ms = self.config.get("latency_ms", 10)
        self.tick_size = self.config.get("tick_size", 0.01)
        self.lot_size = self.config.get("lot_size", 100)
        self.max_participation_rate = self.config.get("max_participation_rate", 0.1)
        self.permanent_impact_coefficient = self.config.get("permanent_impact", 0.1)
        self.temporary_impact_coefficient = self.config.get("temporary_impact", 0.05)

        # 订单簿状态
        self.order_books: Dict[str, SimulatedOrderBook] = {}

        # 历史数据
        self.market_data: Optional[pd.DataFrame] = None
        self.volume_profiles: Dict[str, pd.Series] = {}
        self.liquidity_profiles: Dict[str, pd.DataFrame] = {}

        # 执行统计
        self.execution_stats: List[TransactionResult] = []

    def initialize_from_historical_data(
        self, market_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None
    ) -> None:
        """从历史数据初始化模拟器

        Args:
            market_data: 历史市场数据
            volume_data: 历史成交量数据
        """
        self.market_data = market_data

        # 分析成交量模式
        if volume_data is not None:
            for symbol in volume_data.columns:
                if symbol in volume_data.columns:
                    self.volume_profiles[symbol] = volume_data[symbol]

        # 估算流动性分布
        self._estimate_liquidity_profiles()

        logger.info(f"Initialized simulator with {len(market_data)} data points")

    def simulate_order_book(
        self,
        symbol: str,
        timestamp: datetime,
        mid_price: float,
        volume: int,
        volatility: float = 0.02,
    ) -> SimulatedOrderBook:
        """模拟订单簿

        Args:
            symbol: 标的代码
            timestamp: 时间戳
            mid_price: 中间价
            volume: 成交量
            volatility: 波动率

        Returns:
            模拟的订单簿
        """
        # 生成订单簿深度
        n_levels = 10

        # 价格间隔（基于波动率）
        price_step = self.tick_size * max(
            1, int(volatility * mid_price / self.tick_size)
        )

        # 生成买单簿
        bids = []
        cumulative_volume = 0
        for i in range(n_levels):
            price = mid_price - (i + 1) * price_step
            price = round(price / self.tick_size) * self.tick_size

            # 数量随价格距离递增（流动性分布）
            base_quantity = int(volume * 0.1 * (1 + i * 0.5))
            quantity = round(base_quantity / self.lot_size) * self.lot_size

            # 添加随机性
            quantity = int(quantity * (0.8 + np.random.random() * 0.4))
            quantity = max(self.lot_size, quantity)

            cumulative_volume += quantity
            order_count = np.random.randint(1, 10)

            bids.append(
                OrderBookLevel(
                    price=price,
                    quantity=quantity,
                    order_count=order_count,
                    timestamp=timestamp,
                )
            )

        total_bid_volume = cumulative_volume

        # 生成卖单簿
        asks = []
        cumulative_volume = 0
        for i in range(n_levels):
            price = mid_price + (i + 1) * price_step
            price = round(price / self.tick_size) * self.tick_size

            # 数量随价格距离递增
            base_quantity = int(volume * 0.1 * (1 + i * 0.5))
            quantity = round(base_quantity / self.lot_size) * self.lot_size

            # 添加随机性
            quantity = int(quantity * (0.8 + np.random.random() * 0.4))
            quantity = max(self.lot_size, quantity)

            cumulative_volume += quantity
            order_count = np.random.randint(1, 10)

            asks.append(
                OrderBookLevel(
                    price=price,
                    quantity=quantity,
                    order_count=order_count,
                    timestamp=timestamp,
                )
            )

        total_ask_volume = cumulative_volume

        # 创建订单簿
        order_book = SimulatedOrderBook(
            symbol=symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            last_price=mid_price,
            last_volume=volume,
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume,
        )

        # 缓存订单簿
        self.order_books[symbol] = order_book

        return order_book

    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> TransactionResult:
        """执行订单

        Args:
            symbol: 标的代码
            side: 买卖方向 ('BUY' or 'SELL')
            quantity: 订单数量
            order_type: 订单类型 ('MARKET' or 'LIMIT')
            limit_price: 限价（限价单使用）
            timestamp: 执行时间戳

        Returns:
            交易结果
        """
        if timestamp is None:
            timestamp = datetime.now()

        # 获取或生成订单簿
        if symbol not in self.order_books:
            # 从市场数据生成订单簿
            if self.market_data is not None and symbol in self.market_data.columns:
                price = self.market_data[symbol].iloc[-1]
                volume = self.volume_profiles.get(symbol, pd.Series([10000])).iloc[-1]
                self.simulate_order_book(symbol, timestamp, price, int(volume))
            else:
                raise QuantSystemError(f"No order book available for {symbol}")

        order_book = self.order_books[symbol]

        # 根据订单类型执行
        if order_type == "MARKET":
            result = self._execute_market_order(order_book, side, quantity, timestamp)
        elif order_type == "LIMIT":
            if limit_price is None:
                raise ValueError("Limit price required for limit order")
            result = self._execute_limit_order(
                order_book, side, quantity, limit_price, timestamp
            )
        else:
            raise ValueError(f"Unknown order type: {order_type}")

        # 更新订单簿（市场冲击）
        self._update_order_book_after_execution(order_book, result)

        # 记录执行统计
        self.execution_stats.append(result)

        return result

    def _execute_market_order(
        self,
        order_book: SimulatedOrderBook,
        side: str,
        quantity: int,
        timestamp: datetime,
    ) -> TransactionResult:
        """执行市价单

        Args:
            order_book: 订单簿
            side: 买卖方向
            quantity: 数量
            timestamp: 时间戳

        Returns:
            交易结果
        """
        order_id = (
            f"MKT_{timestamp.strftime('%Y%m%d%H%M%S')}_{np.random.randint(10000)}"
        )

        # 选择正确的订单簿侧
        if side == "BUY":
            levels = order_book.asks  # 买单吃卖单
            base_price = order_book.best_ask or order_book.last_price
        else:
            levels = order_book.bids  # 卖单吃买单
            base_price = order_book.best_bid or order_book.last_price

        # 执行扫单
        filled_quantity = 0
        execution_prices = []
        total_cost = 0.0

        for level in levels:
            if filled_quantity >= quantity:
                break

            # 计算本层可成交数量
            available_quantity = min(level.quantity, quantity - filled_quantity)

            # 应用参与率限制
            max_allowed = int(level.quantity * self.max_participation_rate)
            fill_quantity = min(available_quantity, max_allowed)

            if fill_quantity > 0:
                execution_prices.append((level.price, fill_quantity))
                total_cost += level.price * fill_quantity
                filled_quantity += fill_quantity

        # 计算平均价格
        if filled_quantity > 0:
            average_price = total_cost / filled_quantity
        else:
            average_price = base_price

        # 计算滑点
        if base_price > 0:
            if side == "BUY":
                slippage = (average_price - base_price) / base_price
            else:
                slippage = (base_price - average_price) / base_price
        else:
            slippage = 0.0

        # 计算市场冲击
        market_impact = self._calculate_market_impact(
            quantity,
            filled_quantity,
            order_book.total_bid_volume + order_book.total_ask_volume,
        )

        # 计算交易成本
        transaction_cost = filled_quantity * average_price * 0.001  # 0.1%手续费

        # 模拟执行延迟
        execution_time_ms = self.latency_ms + np.random.exponential(5)

        return TransactionResult(
            order_id=order_id,
            symbol=order_book.symbol,
            side=side,
            requested_quantity=quantity,
            filled_quantity=filled_quantity,
            average_price=average_price,
            execution_prices=execution_prices,
            slippage=slippage,
            market_impact=market_impact,
            transaction_cost=transaction_cost,
            timestamp=timestamp,
            execution_time_ms=execution_time_ms,
            partial_fill=(filled_quantity < quantity),
        )

    def _execute_limit_order(
        self,
        order_book: SimulatedOrderBook,
        side: str,
        quantity: int,
        limit_price: float,
        timestamp: datetime,
    ) -> TransactionResult:
        """执行限价单

        Args:
            order_book: 订单簿
            side: 买卖方向
            quantity: 数量
            limit_price: 限价
            timestamp: 时间戳

        Returns:
            交易结果
        """
        order_id = (
            f"LMT_{timestamp.strftime('%Y%m%d%H%M%S')}_{np.random.randint(10000)}"
        )

        # 选择正确的订单簿侧
        if side == "BUY":
            levels = order_book.asks
            # 买单只能成交价格<=限价的卖单
            eligible_levels = [l for l in levels if l.price <= limit_price]
        else:
            levels = order_book.bids
            # 卖单只能成交价格>=限价的买单
            eligible_levels = [l for l in levels if l.price >= limit_price]

        # 执行成交
        filled_quantity = 0
        execution_prices = []
        total_cost = 0.0

        for level in eligible_levels:
            if filled_quantity >= quantity:
                break

            # 计算本层可成交数量
            available_quantity = min(level.quantity, quantity - filled_quantity)

            # 应用参与率限制
            max_allowed = int(level.quantity * self.max_participation_rate)
            fill_quantity = min(available_quantity, max_allowed)

            if fill_quantity > 0:
                execution_prices.append((level.price, fill_quantity))
                total_cost += level.price * fill_quantity
                filled_quantity += fill_quantity

        # 计算平均价格
        if filled_quantity > 0:
            average_price = total_cost / filled_quantity
        else:
            average_price = limit_price

        # 计算滑点（相对于限价）
        if limit_price > 0:
            if side == "BUY":
                slippage = (average_price - limit_price) / limit_price
            else:
                slippage = (limit_price - average_price) / limit_price
        else:
            slippage = 0.0

        # 限价单通常滑点为负（价格改善）
        slippage = min(0, slippage)

        # 计算市场冲击
        market_impact = self._calculate_market_impact(
            quantity,
            filled_quantity,
            order_book.total_bid_volume + order_book.total_ask_volume,
        )

        # 计算交易成本
        transaction_cost = filled_quantity * average_price * 0.001

        # 限价单执行时间通常更长
        execution_time_ms = self.latency_ms + np.random.exponential(50)

        return TransactionResult(
            order_id=order_id,
            symbol=order_book.symbol,
            side=side,
            requested_quantity=quantity,
            filled_quantity=filled_quantity,
            average_price=average_price,
            execution_prices=execution_prices,
            slippage=slippage,
            market_impact=market_impact,
            transaction_cost=transaction_cost,
            timestamp=timestamp,
            execution_time_ms=execution_time_ms,
            partial_fill=(filled_quantity < quantity),
        )

    def _calculate_market_impact(
        self, requested_quantity: int, filled_quantity: int, total_liquidity: int
    ) -> float:
        """计算市场冲击

        Args:
            requested_quantity: 请求数量
            filled_quantity: 成交数量
            total_liquidity: 总流动性

        Returns:
            市场冲击（基点）
        """
        if total_liquidity == 0:
            return 0.0

        # 参与率
        participation_rate = filled_quantity / total_liquidity

        # 永久冲击（线性）
        permanent_impact = self.permanent_impact_coefficient * participation_rate

        # 临时冲击（平方根）
        temporary_impact = self.temporary_impact_coefficient * np.sqrt(
            participation_rate
        )

        # 总冲击（基点）
        total_impact = (permanent_impact + temporary_impact) * 10000

        return float(total_impact)

    def _update_order_book_after_execution(
        self, order_book: SimulatedOrderBook, result: TransactionResult
    ) -> None:
        """更新订单簿（反映市场冲击）

        Args:
            order_book: 订单簿
            result: 交易结果
        """
        # 更新成交价格
        if result.filled_quantity > 0:
            order_book.last_price = result.average_price
            order_book.last_volume = result.filled_quantity

        # 移除已成交的流动性
        if result.side == "BUY":
            # 从卖单簿移除
            remaining_to_remove = result.filled_quantity
            new_asks = []

            for level in order_book.asks:
                if remaining_to_remove <= 0:
                    new_asks.append(level)
                elif level.quantity > remaining_to_remove:
                    level.quantity -= remaining_to_remove
                    new_asks.append(level)
                    remaining_to_remove = 0
                else:
                    remaining_to_remove -= level.quantity

            order_book.asks = new_asks

        else:
            # 从买单簿移除
            remaining_to_remove = result.filled_quantity
            new_bids = []

            for level in order_book.bids:
                if remaining_to_remove <= 0:
                    new_bids.append(level)
                elif level.quantity > remaining_to_remove:
                    level.quantity -= remaining_to_remove
                    new_bids.append(level)
                    remaining_to_remove = 0
                else:
                    remaining_to_remove -= level.quantity

            order_book.bids = new_bids

        # 应用永久市场冲击（移动价格）
        impact_bps = result.market_impact
        price_change = order_book.last_price * impact_bps / 10000

        if result.side == "BUY":
            # 买单推高价格
            for level in order_book.asks:
                level.price += price_change * 0.5  # 部分永久冲击
            for level in order_book.bids:
                level.price += price_change * 0.5
        else:
            # 卖单压低价格
            for level in order_book.asks:
                level.price -= price_change * 0.5
            for level in order_book.bids:
                level.price -= price_change * 0.5

    def _estimate_liquidity_profiles(self) -> None:
        """估算流动性分布"""
        if self.market_data is None:
            return

        # 如果market_data是DataFrame,只使用close价格列
        if isinstance(self.market_data, pd.DataFrame):
            if "close" in self.market_data.columns:
                close_prices = self.market_data["close"]
                returns = close_prices.pct_change()
                volatility = returns.std()
                avg_price = close_prices.mean()

                # 简化：使用单一symbol处理
                symbol = "default"
                self.liquidity_profiles[symbol] = pd.DataFrame(
                    {"volatility": [volatility], "avg_price": [avg_price]}
                )
            return

        # 原有逻辑（如果需要）
        for symbol in self.market_data.columns:
            if symbol == "timestamp":
                continue

            # 计算日内波动率
            if "close" in self.market_data.columns:
                returns = self.market_data["close"].pct_change()
            else:
                returns = self.market_data[symbol].pct_change()
            volatility = returns.std()

            # 估算流动性（简化模型）
            avg_price = (
                self.market_data[symbol].mean()
                if symbol in self.market_data.columns
                else self.market_data["close"].mean()
            )

            # 生成流动性分布（假设）
            liquidity_levels = []
            for distance_bps in range(0, 100, 10):
                distance = distance_bps / 10000

                # 流动性随价格距离指数衰减
                liquidity = 10000 * np.exp(-distance * 100)
                liquidity_levels.append(
                    {
                        "distance_bps": distance_bps,
                        "bid_liquidity": liquidity,
                        "ask_liquidity": liquidity,
                    }
                )

            self.liquidity_profiles[symbol] = pd.DataFrame(liquidity_levels)

    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计

        Returns:
            执行统计字典
        """
        if not self.execution_stats:
            return {
                "total_orders": 0,
                "avg_fill_rate": 0.0,
                "avg_slippage_bps": 0.0,
                "avg_market_impact_bps": 0.0,
                "total_transaction_cost": 0.0,
            }

        fill_rates = [r.fill_rate for r in self.execution_stats]
        slippages = [r.slippage * 10000 for r in self.execution_stats]
        impacts = [r.market_impact for r in self.execution_stats]
        costs = [r.transaction_cost for r in self.execution_stats]

        return {
            "total_orders": len(self.execution_stats),
            "avg_fill_rate": float(np.mean(fill_rates)),
            "avg_slippage_bps": float(np.mean(slippages)),
            "avg_market_impact_bps": float(np.mean(impacts)),
            "total_transaction_cost": float(np.sum(costs)),
            "partial_fill_rate": float(
                np.mean([r.partial_fill for r in self.execution_stats])
            ),
            "avg_execution_time_ms": float(
                np.mean([r.execution_time_ms for r in self.execution_stats])
            ),
        }

    def reset(self) -> None:
        """重置模拟器状态"""
        self.order_books.clear()
        self.execution_stats.clear()
        logger.info("Transaction simulator reset")


# 模块级别函数
def create_transaction_simulator(
    market_data: Optional[pd.DataFrame] = None, config: Optional[Dict[str, Any]] = None
) -> TransactionSimulator:
    """创建交易模拟器的便捷函数

    Args:
        market_data: 市场数据
        config: 配置参数

    Returns:
        交易模拟器实例
    """
    simulator = TransactionSimulator(config)

    if market_data is not None:
        simulator.initialize_from_historical_data(market_data)

    return simulator


def simulate_order_execution(
    symbol: str,
    side: str,
    quantity: int,
    mid_price: float,
    volume: int = 100000,
    order_type: str = "MARKET",
) -> TransactionResult:
    """模拟单个订单执行的便捷函数

    Args:
        symbol: 标的代码
        side: 买卖方向
        quantity: 数量
        mid_price: 中间价
        volume: 市场成交量
        order_type: 订单类型

    Returns:
        交易结果
    """
    simulator = TransactionSimulator()

    # 生成订单簿
    timestamp = datetime.now()
    simulator.simulate_order_book(symbol, timestamp, mid_price, volume)

    # 执行订单
    result = simulator.execute_order(
        symbol, side, quantity, order_type, timestamp=timestamp
    )

    return result
