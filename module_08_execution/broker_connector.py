"""
券商连接器模块
负责与券商API的标准化接口
"""

import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from common.exceptions import ExecutionError
from common.logging_system import setup_logger
from module_08_execution.order_manager import Order, OrderStatus, OrderUpdate

logger = setup_logger("broker_connector")


class BrokerType(Enum):
    """券商类型枚举"""

    INTERACTIVE_BROKERS = "IB"
    BINANCE = "BINANCE"
    ALPACA = "ALPACA"
    MOCK = "MOCK"


@dataclass
class BrokerConfig:
    """券商配置"""

    broker_type: BrokerType
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    account_id: Optional[str] = None
    base_url: Optional[str] = None
    paper_trading: bool = True
    max_retry_attempts: int = 3
    connection_timeout: int = 30
    heartbeat_interval: int = 10


@dataclass
class AccountInfo:
    """账户信息"""

    account_id: str
    broker: BrokerType
    balance: float
    buying_power: float
    positions: Dict[str, float]
    pending_orders: int
    daily_trades_count: int
    daily_trades_limit: int
    is_pattern_day_trader: bool
    last_update: datetime


@dataclass
class Position:
    """持仓信息"""

    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    last_update: datetime


class BrokerConnector(ABC):
    """券商连接器基类"""

    def __init__(self, config: BrokerConfig):
        """初始化券商连接器

        Args:
            config: 券商配置
        """
        self.config = config
        self.is_connected = False
        self.connection_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.order_callbacks: List[Callable[[OrderUpdate], None]] = []
        self.account_callbacks: List[Callable[[AccountInfo], None]] = []
        self.error_callbacks: List[Callable[[str, Exception], None]] = []
        self.last_heartbeat: Optional[datetime] = None

    @abstractmethod
    def connect(self) -> bool:
        """连接到券商

        Returns:
            是否成功连接
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接

        Returns:
            是否成功断开
        """
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """提交订单

        Args:
            order: 订单对象

        Returns:
            券商订单ID
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """取消订单

        Args:
            order_id: 订单ID

        Returns:
            是否成功取消
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """获取订单状态

        Args:
            order_id: 订单ID

        Returns:
            订单状态
        """
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """获取账户信息

        Returns:
            账户信息对象
        """
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """获取持仓信息

        Returns:
            标的代码到持仓的映射
        """
        pass

    @abstractmethod
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """获取市场数据

        Args:
            symbol: 标的代码

        Returns:
            市场数据字典
        """
        pass

    def register_order_callback(self, callback: Callable[[OrderUpdate], None]) -> None:
        """注册订单回调

        Args:
            callback: 回调函数
        """
        self.order_callbacks.append(callback)

    def register_account_callback(
        self, callback: Callable[[AccountInfo], None]
    ) -> None:
        """注册账户回调

        Args:
            callback: 回调函数
        """
        self.account_callbacks.append(callback)

    def register_error_callback(
        self, callback: Callable[[str, Exception], None]
    ) -> None:
        """注册错误回调

        Args:
            callback: 回调函数
        """
        self.error_callbacks.append(callback)

    def _trigger_order_callbacks(self, update: OrderUpdate) -> None:
        """触发订单回调

        Args:
            update: 订单更新
        """
        for callback in self.order_callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Order callback error: {e}")

    def _trigger_account_callbacks(self, info: AccountInfo) -> None:
        """触发账户回调

        Args:
            info: 账户信息
        """
        for callback in self.account_callbacks:
            try:
                callback(info)
            except Exception as e:
                logger.error(f"Account callback error: {e}")

    def _trigger_error_callbacks(self, context: str, error: Exception) -> None:
        """触发错误回调

        Args:
            context: 错误上下文
            error: 异常对象
        """
        for callback in self.error_callbacks:
            try:
                callback(context, error)
            except Exception as e:
                logger.error(f"Error callback error: {e}")

    def _start_heartbeat(self) -> None:
        """启动心跳线程"""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return

        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        """心跳循环"""
        import time

        while self.is_connected:
            try:
                # 发送心跳
                self._send_heartbeat()
                self.last_heartbeat = datetime.now()

                # 等待下一次心跳
                time.sleep(self.config.heartbeat_interval)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                self._trigger_error_callbacks("heartbeat", e)

    @abstractmethod
    def _send_heartbeat(self) -> None:
        """发送心跳（子类实现）"""
        pass

    def is_healthy(self) -> bool:
        """检查连接健康状态

        Returns:
            是否健康
        """
        if not self.is_connected:
            return False

        if self.last_heartbeat:
            time_since_heartbeat = (
                datetime.now() - self.last_heartbeat
            ).total_seconds()
            if time_since_heartbeat > self.config.heartbeat_interval * 3:
                logger.warning(f"No heartbeat for {time_since_heartbeat} seconds")
                return False

        return True


class UnifiedBrokerInterface:
    """统一券商接口"""

    def __init__(self):
        """初始化统一接口"""
        self.brokers: Dict[BrokerType, BrokerConnector] = {}
        self.active_broker: Optional[BrokerConnector] = None
        self.order_routing: Dict[str, BrokerType] = {}  # symbol -> broker映射

    def add_broker(self, broker_type: BrokerType, config: BrokerConfig) -> None:
        """添加券商连接

        Args:
            broker_type: 券商类型
            config: 券商配置

        Note:
            当前版本不涉及真实券商连接，此方法预留用于未来扩展。
            实际交易执行请使用 ExecutionInterface。
        """
        raise NotImplementedError(
            "Broker connection not implemented. "
            "Use ExecutionInterface for order execution."
        )

    def connect_all(self) -> Dict[BrokerType, bool]:
        """连接所有券商

        Returns:
            券商类型到连接状态的映射
        """
        results = {}

        for broker_type, connector in self.brokers.items():
            try:
                success = connector.connect()
                results[broker_type] = success

                if success:
                    logger.info(f"Connected to {broker_type}")
                else:
                    logger.error(f"Failed to connect to {broker_type}")

            except Exception as e:
                logger.error(f"Error connecting to {broker_type}: {e}")
                results[broker_type] = False

        return results

    def disconnect_all(self) -> None:
        """断开所有券商连接"""
        for broker_type, connector in self.brokers.items():
            try:
                connector.disconnect()
                logger.info(f"Disconnected from {broker_type}")
            except Exception as e:
                logger.error(f"Error disconnecting from {broker_type}: {e}")

    def route_order(
        self, order: Order, preferred_broker: Optional[BrokerType] = None
    ) -> str:
        """路由订单到合适的券商

        Args:
            order: 订单对象
            preferred_broker: 首选券商

        Returns:
            券商订单ID
        """
        # 选择券商
        if preferred_broker and preferred_broker in self.brokers:
            broker = self.brokers[preferred_broker]
        elif order.symbol in self.order_routing:
            broker = self.brokers[self.order_routing[order.symbol]]
        else:
            broker = self.active_broker

        if broker is None:
            raise ExecutionError("No active broker available")

        # 提交订单
        broker_order_id = broker.submit_order(order)

        # 记录路由信息
        self.order_routing[order.symbol] = broker.config.broker_type

        logger.info(f"Routed order {order.order_id} to {broker.config.broker_type}")

        return broker_order_id

    def get_aggregated_positions(self) -> Dict[str, Position]:
        """获取聚合持仓

        Returns:
            所有券商的聚合持仓
        """
        aggregated = {}

        for broker in self.brokers.values():
            if broker.is_connected:
                positions = broker.get_positions()

                for symbol, position in positions.items():
                    if symbol in aggregated:
                        # 聚合同一标的的持仓
                        existing = aggregated[symbol]

                        # 计算加权平均成本
                        total_quantity = existing.quantity + position.quantity
                        if total_quantity > 0:
                            avg_cost = (
                                existing.avg_cost * existing.quantity
                                + position.avg_cost * position.quantity
                            ) / total_quantity
                        else:
                            avg_cost = 0

                        # 更新聚合持仓
                        aggregated[symbol] = Position(
                            symbol=symbol,
                            quantity=total_quantity,
                            avg_cost=avg_cost,
                            current_price=position.current_price,
                            market_value=existing.market_value + position.market_value,
                            unrealized_pnl=existing.unrealized_pnl
                            + position.unrealized_pnl,
                            realized_pnl=existing.realized_pnl + position.realized_pnl,
                            last_update=max(existing.last_update, position.last_update),
                        )
                    else:
                        aggregated[symbol] = position

        return aggregated

    def get_total_buying_power(self) -> float:
        """获取总购买力

        Returns:
            所有券商的总购买力
        """
        total = 0.0

        for broker in self.brokers.values():
            if broker.is_connected:
                account_info = broker.get_account_info()
                total += account_info.buying_power

        return total

    def select_best_broker_for_order(
        self, symbol: str, order_type: str, quantity: int
    ) -> BrokerType:
        """为订单选择最佳券商

        Args:
            symbol: 标的代码
            order_type: 订单类型
            quantity: 数量

        Returns:
            最佳券商类型
        """
        best_broker = None
        best_score = -1

        for broker_type, broker in self.brokers.items():
            if not broker.is_connected:
                continue

            # 计算评分
            score = 0

            # 检查账户信息
            account_info = broker.get_account_info()

            # 购买力评分
            if account_info.buying_power > quantity * 100:  # 假设价格100
                score += 30

            # 日内交易限制评分
            if account_info.daily_trades_count < account_info.daily_trades_limit * 0.8:
                score += 20

            # 手续费评分（需要从配置获取）
            if broker.config.paper_trading:
                score += 10  # 模拟交易优先

            # 历史执行质量评分（需要从历史数据获取）
            score += 40  # 默认分数

            if score > best_score:
                best_score = score
                best_broker = broker_type

        return best_broker if best_broker else BrokerType.MOCK


# 全局券商管理器
_broker_manager: Optional[UnifiedBrokerInterface] = None


def get_broker_manager() -> UnifiedBrokerInterface:
    """获取全局券商管理器

    Returns:
        券商管理器实例
    """
    global _broker_manager
    if _broker_manager is None:
        _broker_manager = UnifiedBrokerInterface()
    return _broker_manager


def initialize_brokers(configs: List[BrokerConfig]) -> None:
    """初始化券商连接

    Args:
        configs: 券商配置列表
    """
    manager = get_broker_manager()

    for config in configs:
        manager.add_broker(config.broker_type, config)

    # 连接所有券商
    results = manager.connect_all()

    # 检查连接结果
    successful = sum(1 for success in results.values() if success)
    logger.info(f"Connected to {successful}/{len(configs)} brokers")

    if successful == 0:
        logger.warning("No brokers connected successfully, using mock broker")

        # 添加模拟券商作为后备
        mock_config = BrokerConfig(broker_type=BrokerType.MOCK, paper_trading=True)
        manager.add_broker(BrokerType.MOCK, mock_config)
        manager.brokers[BrokerType.MOCK].connect()
