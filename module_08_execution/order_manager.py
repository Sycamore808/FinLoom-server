"""
订单管理器模块
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from common.data_structures import Signal
from common.exceptions import ExecutionError
from common.logging_system import setup_logger

logger = setup_logger("order_manager")


@dataclass
class OrderUpdate:
    """订单更新数据结构"""

    order_id: str
    status: "OrderStatus"
    filled_quantity: int
    filled_price: Optional[float]
    timestamp: datetime
    message: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class OrderStatus(Enum):
    """订单状态枚举"""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderType(Enum):
    """订单类型枚举"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class Order:
    """订单数据结构"""

    order_id: str
    signal_id: str
    symbol: str
    order_type: OrderType
    side: str  # BUY or SELL
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    created_at: datetime = None
    updated_at: datetime = None
    submitted_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class OrderManager:
    """订单管理器类"""

    def __init__(self):
        """初始化订单管理器"""
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []

    def create_order_from_signal(self, signal: Signal) -> Order:
        """从信号创建订单

        Args:
            signal: 交易信号

        Returns:
            创建的订单
        """
        try:
            order_id = str(uuid.uuid4())

            # 确定订单类型和价格
            order_type = OrderType.MARKET
            price = signal.price

            # 根据信号类型确定买卖方向
            # EnhancedSignal使用signal_type枚举，普通Signal使用action字符串
            if hasattr(signal, "signal_type"):
                # EnhancedSignal
                side = signal.signal_type.value  # SignalType.BUY.value = "BUY"
            elif hasattr(signal, "action"):
                # 普通Signal
                side = "BUY" if signal.action == "BUY" else "SELL"
            else:
                raise ValueError(
                    "Signal must have either 'signal_type' or 'action' attribute"
                )

            order = Order(
                order_id=order_id,
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                order_type=order_type,
                side=side,
                quantity=signal.quantity,
                price=price,
                metadata={
                    "strategy_name": signal.strategy_name,
                    "confidence": signal.confidence,
                    "signal_metadata": signal.metadata,
                },
            )

            self.orders[order_id] = order
            logger.info(f"Created order {order_id} from signal {signal.signal_id}")
            return order

        except Exception as e:
            logger.error(f"Failed to create order from signal: {e}")
            raise ExecutionError(f"Order creation failed: {e}")

    def submit_order(self, order: Order) -> bool:
        """提交订单

        Args:
            order: 订单对象

        Returns:
            是否提交成功
        """
        try:
            if order.order_id not in self.orders:
                raise ExecutionError(f"Order {order.order_id} not found")

            # 更新订单状态
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now()
            order.submitted_time = datetime.now()

            logger.info(f"Order {order.order_id} submitted for {order.symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to submit order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            return False

    def fill_order(
        self, order_id: str, filled_quantity: int, filled_price: float
    ) -> bool:
        """填充订单

        Args:
            order_id: 订单ID
            filled_quantity: 填充数量
            filled_price: 填充价格

        Returns:
            是否填充成功
        """
        try:
            if order_id not in self.orders:
                raise ExecutionError(f"Order {order_id} not found")

            order = self.orders[order_id]

            # 更新填充信息
            order.filled_quantity += filled_quantity
            order.filled_price = filled_price
            order.updated_at = datetime.now()

            # 更新订单状态
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                order.filled_time = datetime.now()
                logger.info(f"Order {order_id} fully filled")
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
                logger.info(
                    f"Order {order_id} partially filled: {order.filled_quantity}/{order.quantity}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to fill order {order_id}: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """取消订单

        Args:
            order_id: 订单ID

        Returns:
            是否取消成功
        """
        try:
            if order_id not in self.orders:
                raise ExecutionError(f"Order {order_id} not found")

            order = self.orders[order_id]

            # 检查订单状态
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                logger.warning(
                    f"Cannot cancel order {order_id} with status {order.status}"
                )
                return False

            # 更新订单状态
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()

            logger.info(f"Order {order_id} cancelled")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单

        Args:
            order_id: 订单ID

        Returns:
            订单对象
        """
        return self.orders.get(order_id)

    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """获取指定股票的订单

        Args:
            symbol: 股票代码

        Returns:
            订单列表
        """
        return [order for order in self.orders.values() if order.symbol == symbol]

    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """获取指定状态的订单

        Args:
            status: 订单状态

        Returns:
            订单列表
        """
        return [order for order in self.orders.values() if order.status == status]

    def get_pending_orders(self) -> List[Order]:
        """获取待处理订单

        Returns:
            待处理订单列表
        """
        return self.get_orders_by_status(OrderStatus.PENDING)

    def get_active_orders(self) -> List[Order]:
        """获取活跃订单（已提交但未完成）

        Returns:
            活跃订单列表
        """
        active_statuses = [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        return [
            order for order in self.orders.values() if order.status in active_statuses
        ]

    def archive_order(self, order_id: str) -> bool:
        """归档订单到历史记录

        Args:
            order_id: 订单ID

        Returns:
            是否归档成功
        """
        try:
            if order_id not in self.orders:
                raise ExecutionError(f"Order {order_id} not found")

            order = self.orders[order_id]

            # 添加到历史记录
            self.order_history.append(order)

            # 从活跃订单中移除
            del self.orders[order_id]

            logger.info(f"Order {order_id} archived")
            return True

        except Exception as e:
            logger.error(f"Failed to archive order {order_id}: {e}")
            return False

    def get_order_statistics(self) -> Dict[str, Any]:
        """获取订单统计信息

        Returns:
            统计信息字典
        """
        total_orders = len(self.orders) + len(self.order_history)
        active_orders = len(self.orders)
        filled_orders = len(
            [o for o in self.order_history if o.status == OrderStatus.FILLED]
        )
        cancelled_orders = len(
            [o for o in self.order_history if o.status == OrderStatus.CANCELLED]
        )

        return {
            "total_orders": total_orders,
            "active_orders": active_orders,
            "filled_orders": filled_orders,
            "cancelled_orders": cancelled_orders,
            "fill_rate": filled_orders / total_orders if total_orders > 0 else 0.0,
        }

    def process_signal(self, signal: Signal) -> Optional[Order]:
        """处理交易信号并创建订单

        Args:
            signal: 交易信号

        Returns:
            创建的订单
        """
        try:
            # 创建订单
            order = self.create_order_from_signal(signal)

            # 提交订单
            if self.submit_order(order):
                return order
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to process signal {signal.signal_id}: {e}")
            return None

    def simulate_market_execution(self, order: Order) -> bool:
        """模拟市场执行

        Args:
            order: 订单对象

        Returns:
            是否执行成功
        """
        try:
            if order.status != OrderStatus.SUBMITTED:
                return False

            # 模拟执行延迟
            import time

            time.sleep(0.1)  # 模拟网络延迟

            # 模拟完全填充
            return self.fill_order(order.order_id, order.quantity, order.price or 0.0)

        except Exception as e:
            logger.error(
                f"Failed to simulate execution for order {order.order_id}: {e}"
            )
            return False


# 全局订单管理器实例
_global_order_manager: Optional[OrderManager] = None


def get_order_manager() -> OrderManager:
    """获取全局订单管理器实例

    Returns:
        订单管理器实例
    """
    global _global_order_manager
    if _global_order_manager is None:
        _global_order_manager = OrderManager()
    return _global_order_manager
