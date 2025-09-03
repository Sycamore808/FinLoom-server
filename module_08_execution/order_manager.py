"""
订单管理器模块
负责订单生命周期管理、状态跟踪和执行控制
"""

import queue
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from common.constants import MAX_RETRY_ATTEMPTS, TIMEOUT_SECONDS
from common.exceptions import ExecutionError
from common.logging_system import setup_logger

logger = setup_logger("order_manager")


class OrderStatus(Enum):
    """订单状态枚举"""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    ERROR = "ERROR"


class OrderType(Enum):
    """订单类型枚举"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"


@dataclass
class Order:
    """订单数据结构"""

    order_id: str
    symbol: str
    side: str  # BUY or SELL
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    time_in_force: str = "DAY"
    status: OrderStatus = OrderStatus.PENDING
    created_time: datetime = field(default_factory=datetime.now)
    submitted_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    broker_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = 0

    def is_active(self) -> bool:
        """检查订单是否活跃

        Returns:
            是否活跃
        """
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
        ]

    def is_complete(self) -> bool:
        """检查订单是否完成

        Returns:
            是否完成
        """
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.ERROR,
        ]

    def fill_rate(self) -> float:
        """计算成交率

        Returns:
            成交率（0-1）
        """
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity


@dataclass
class OrderUpdate:
    """订单更新事件"""

    order_id: str
    timestamp: datetime
    status: OrderStatus
    filled_quantity: int = 0
    fill_price: float = 0.0
    commission: float = 0.0
    message: Optional[str] = None


@dataclass
class ExecutionReport:
    """执行报告"""

    order_id: str
    symbol: str
    side: str
    total_quantity: int
    filled_quantity: int
    avg_price: float
    total_commission: float
    slippage: float
    execution_time_seconds: float
    status: OrderStatus
    fill_details: List[Dict[str, Any]]


class OrderManager:
    """订单管理器类"""

    def __init__(self, config: Dict[str, Any]):
        """初始化订单管理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.orders: Dict[str, Order] = {}
        self.order_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}
        self.order_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

        # 订单限制
        self.max_open_orders = config.get("max_open_orders", 100)
        self.max_order_value = config.get("max_order_value", 1000000)
        self.daily_order_limit = config.get("daily_order_limit", 1000)
        self.daily_order_count = 0

        # 回调函数
        self.order_callbacks: Dict[str, List[Callable]] = {
            "on_submitted": [],
            "on_filled": [],
            "on_cancelled": [],
            "on_rejected": [],
            "on_error": [],
        }

        # 执行线程
        self.executor_thread: Optional[threading.Thread] = None
        self.is_running = False

    def start(self) -> None:
        """启动订单管理器"""
        if self.is_running:
            logger.warning("Order manager already running")
            return

        self.is_running = True
        self.executor_thread = threading.Thread(
            target=self._execution_loop, daemon=True
        )
        self.executor_thread.start()
        logger.info("Order manager started")

    def stop(self) -> None:
        """停止订单管理器"""
        self.is_running = False
        if self.executor_thread:
            self.executor_thread.join(timeout=5)
        logger.info("Order manager stopped")

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: OrderType,
        quantity: int,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "DAY",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Order:
        """创建订单

        Args:
            symbol: 标的代码
            side: 买卖方向
            order_type: 订单类型
            quantity: 数量
            price: 价格
            stop_price: 止损价
            time_in_force: 有效期
            metadata: 元数据

        Returns:
            创建的订单对象
        """
        # 生成订单ID
        order_id = (
            f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        # 创建订单对象
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            limit_price=price if order_type == OrderType.LIMIT else None,
            time_in_force=time_in_force,
            metadata=metadata or {},
        )

        # 验证订单
        if not self._validate_order(order):
            raise ExecutionError(f"Order validation failed: {order_id}")

        # 存储订单
        self.orders[order_id] = order
        self.active_orders[order_id] = order

        logger.info(f"Order created: {order_id} - {symbol} {side} {quantity}")
        return order

    def submit_order(self, order: Order, priority: int = 5) -> bool:
        """提交订单

        Args:
            order: 订单对象
            priority: 优先级（1-10，1最高）

        Returns:
            是否成功提交
        """
        with self.order_locks[order.order_id]:
            if order.status != OrderStatus.PENDING:
                logger.warning(f"Cannot submit order in status {order.status}")
                return False

            # 检查每日限制
            if self.daily_order_count >= self.daily_order_limit:
                logger.error(f"Daily order limit reached: {self.daily_order_limit}")
                order.status = OrderStatus.REJECTED
                order.error_message = "Daily order limit exceeded"
                return False

            # 加入执行队列
            self.order_queue.put((priority, datetime.now(), order))
            order.status = OrderStatus.SUBMITTED
            order.submitted_time = datetime.now()
            self.daily_order_count += 1

            # 触发回调
            self._trigger_callbacks("on_submitted", order)

            logger.info(f"Order submitted: {order.order_id}")
            return True

    def cancel_order(self, order_id: str, reason: str = "") -> bool:
        """取消订单

        Args:
            order_id: 订单ID
            reason: 取消原因

        Returns:
            是否成功取消
        """
        if order_id not in self.orders:
            logger.error(f"Order not found: {order_id}")
            return False

        order = self.orders[order_id]

        with self.order_locks[order_id]:
            if not order.is_active():
                logger.warning(f"Cannot cancel inactive order: {order_id}")
                return False

            order.status = OrderStatus.CANCELLED
            order.error_message = reason

            # 移动到完成订单
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            self.completed_orders[order_id] = order

            # 触发回调
            self._trigger_callbacks("on_cancelled", order)

            logger.info(f"Order cancelled: {order_id} - {reason}")
            return True

    def update_order_status(self, update: OrderUpdate) -> None:
        """更新订单状态

        Args:
            update: 订单更新事件
        """
        if update.order_id not in self.orders:
            logger.error(f"Order not found for update: {update.order_id}")
            return

        order = self.orders[update.order_id]

        with self.order_locks[update.order_id]:
            # 更新状态
            old_status = order.status
            order.status = update.status

            # 更新成交信息
            if update.filled_quantity > 0:
                prev_filled = order.filled_quantity
                order.filled_quantity += update.filled_quantity

                # 计算加权平均成交价
                if order.filled_quantity > 0:
                    order.avg_fill_price = (
                        prev_filled * order.avg_fill_price
                        + update.filled_quantity * update.fill_price
                    ) / order.filled_quantity

                order.commission += update.commission

            # 处理状态转换
            if order.status == OrderStatus.FILLED:
                order.filled_time = update.timestamp
                self._handle_order_filled(order)
            elif order.status == OrderStatus.REJECTED:
                order.error_message = update.message
                self._handle_order_rejected(order)
            elif order.status == OrderStatus.ERROR:
                order.error_message = update.message
                self._handle_order_error(order)

            logger.info(
                f"Order status updated: {update.order_id} {old_status} -> {update.status}"
            )

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取活跃订单

        Args:
            symbol: 标的代码（可选）

        Returns:
            活跃订单列表
        """
        orders = list(self.active_orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """获取订单状态

        Args:
            order_id: 订单ID

        Returns:
            订单状态
        """
        if order_id in self.orders:
            return self.orders[order_id].status
        return None

    def generate_execution_report(self, order_id: str) -> Optional[ExecutionReport]:
        """生成执行报告

        Args:
            order_id: 订单ID

        Returns:
            执行报告
        """
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]

        # 计算滑点
        if order.price and order.avg_fill_price:
            if order.side == "BUY":
                slippage = order.avg_fill_price - order.price
            else:
                slippage = order.price - order.avg_fill_price
        else:
            slippage = 0.0

        # 计算执行时间
        if order.submitted_time and order.filled_time:
            execution_time = (order.filled_time - order.submitted_time).total_seconds()
        else:
            execution_time = 0.0

        return ExecutionReport(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            total_quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            avg_price=order.avg_fill_price,
            total_commission=order.commission,
            slippage=slippage,
            execution_time_seconds=execution_time,
            status=order.status,
            fill_details=order.metadata.get("fill_details", []),
        )

    def register_callback(self, event: str, callback: Callable[[Order], None]) -> None:
        """注册回调函数

        Args:
            event: 事件类型
            callback: 回调函数
        """
        if event in self.order_callbacks:
            self.order_callbacks[event].append(callback)

    def _execution_loop(self) -> None:
        """执行循环"""
        while self.is_running:
            try:
                # 从队列获取订单（超时1秒）
                priority, timestamp, order = self.order_queue.get(timeout=1)

                # 执行订单
                self._execute_order(order)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")

    def _execute_order(self, order: Order) -> None:
        """执行订单（内部方法）

        Args:
            order: 订单对象
        """
        try:
            # 这里应该调用实际的券商接口
            # 现在使用模拟执行
            logger.info(f"Executing order: {order.order_id}")

            # 模拟执行延迟
            import time

            time.sleep(0.1)

            # 模拟成交
            update = OrderUpdate(
                order_id=order.order_id,
                timestamp=datetime.now(),
                status=OrderStatus.FILLED,
                filled_quantity=order.quantity,
                fill_price=order.price or 100.0,  # 模拟成交价
                commission=order.quantity * 0.001,  # 模拟佣金
            )

            self.update_order_status(update)

        except Exception as e:
            logger.error(f"Order execution failed: {order.order_id} - {e}")

            # 更新为错误状态
            update = OrderUpdate(
                order_id=order.order_id,
                timestamp=datetime.now(),
                status=OrderStatus.ERROR,
                message=str(e),
            )
            self.update_order_status(update)

    def _validate_order(self, order: Order) -> bool:
        """验证订单

        Args:
            order: 订单对象

        Returns:
            是否有效
        """
        # 检查数量
        if order.quantity <= 0:
            logger.error(f"Invalid quantity: {order.quantity}")
            return False

        # 检查订单数限制
        if len(self.active_orders) >= self.max_open_orders:
            logger.error(f"Max open orders reached: {self.max_open_orders}")
            return False

        # 检查订单价值
        if order.price:
            order_value = order.quantity * order.price
            if order_value > self.max_order_value:
                logger.error(
                    f"Order value {order_value} exceeds limit {self.max_order_value}"
                )
                return False

        # 检查订单类型特定要求
        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            logger.error("Limit order requires limit price")
            return False

        if (
            order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]
            and order.stop_price is None
        ):
            logger.error("Stop order requires stop price")
            return False

        return True

    def _handle_order_filled(self, order: Order) -> None:
        """处理订单成交

        Args:
            order: 订单对象
        """
        # 移动到完成订单
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
        self.completed_orders[order.order_id] = order

        # 触发回调
        self._trigger_callbacks("on_filled", order)

        logger.info(
            f"Order filled: {order.order_id} - {order.filled_quantity} @ {order.avg_fill_price}"
        )

    def _handle_order_rejected(self, order: Order) -> None:
        """处理订单拒绝

        Args:
            order: 订单对象
        """
        # 移动到完成订单
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
        self.completed_orders[order.order_id] = order

        # 触发回调
        self._trigger_callbacks("on_rejected", order)

        logger.warning(f"Order rejected: {order.order_id} - {order.error_message}")

    def _handle_order_error(self, order: Order) -> None:
        """处理订单错误

        Args:
            order: 订单对象
        """
        # 检查重试次数
        if order.retry_count < MAX_RETRY_ATTEMPTS:
            order.retry_count += 1
            order.status = OrderStatus.PENDING
            self.submit_order(order, priority=1)  # 高优先级重试
            logger.info(
                f"Retrying order: {order.order_id} (attempt {order.retry_count})"
            )
        else:
            # 移动到完成订单
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
            self.completed_orders[order.order_id] = order

            # 触发回调
            self._trigger_callbacks("on_error", order)

            logger.error(
                f"Order failed after {order.retry_count} retries: {order.order_id}"
            )

    def _trigger_callbacks(self, event: str, order: Order) -> None:
        """触发回调函数

        Args:
            event: 事件类型
            order: 订单对象
        """
        if event in self.order_callbacks:
            for callback in self.order_callbacks[event]:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Callback error for {event}: {e}")


# 模块级别函数
def create_order_manager(config: Dict[str, Any]) -> OrderManager:
    """创建订单管理器实例

    Args:
        config: 配置字典

    Returns:
        订单管理器实例
    """
    manager = OrderManager(config)
    manager.start()
    return manager
