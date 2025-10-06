"""
执行接口模块
提供简洁的订单提交和跟踪接口，不涉及具体券商连接
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from common.exceptions import ExecutionError
from common.logging_system import setup_logger
from module_08_execution.database_manager import get_execution_database_manager
from module_08_execution.order_manager import Order, OrderStatus, OrderUpdate

logger = setup_logger("execution_interface")


class ExecutionDestination(Enum):
    """执行目的地枚举"""

    EXCHANGE = "EXCHANGE"  # 交易所直接执行
    BROKER = "BROKER"  # 券商执行
    MANUAL = "MANUAL"  # 人工执行
    PENDING = "PENDING"  # 待执行


@dataclass
class ExecutionRequest:
    """执行请求数据结构"""

    order_id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    order_type: str  # MARKET, LIMIT, etc.
    price: Optional[float] = None
    stop_price: Optional[float] = None
    destination: ExecutionDestination = ExecutionDestination.PENDING
    notes: Optional[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ExecutionResult:
    """执行结果数据结构"""

    order_id: str
    status: OrderStatus
    executed_quantity: int
    executed_price: Optional[float]
    remaining_quantity: int
    execution_time: datetime
    commission: float = 0.0
    slippage_bps: float = 0.0
    message: Optional[str] = None

    def __post_init__(self):
        if self.execution_time is None:
            self.execution_time = datetime.now()


class ExecutionInterface:
    """
    执行接口类

    提供简洁的订单提交和跟踪接口，不涉及具体券商连接。
    主要用于记录需要执行的订单信息，实际执行由外部系统完成。
    """

    def __init__(self):
        """初始化执行接口"""
        self.pending_requests: Dict[str, ExecutionRequest] = {}
        self.execution_results: Dict[str, ExecutionResult] = {}
        self.db_manager = get_execution_database_manager()
        logger.info("ExecutionInterface initialized")

    def submit_execution_request(
        self,
        order: Order,
        destination: ExecutionDestination = ExecutionDestination.PENDING,
        notes: Optional[str] = None,
    ) -> ExecutionRequest:
        """
        提交执行请求

        Args:
            order: 订单对象
            destination: 执行目的地
            notes: 备注信息

        Returns:
            ExecutionRequest: 执行请求对象
        """
        request = ExecutionRequest(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type.value,
            price=order.price,
            stop_price=order.stop_price,
            destination=destination,
            notes=notes,
        )

        self.pending_requests[order.order_id] = request

        # 保存到数据库
        self.db_manager.save_order(
            order_id=order.order_id,
            signal_id=order.signal_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type.value,
            quantity=order.quantity,
            price=order.price,
            status=OrderStatus.PENDING.value,
            strategy_name=getattr(order, "strategy_name", None),
            timestamp=order.created_at,
            filled_quantity=0,
            stop_price=order.stop_price,
            updated_at=datetime.now(),
            metadata=str({"destination": destination.value, "notes": notes}),
        )

        logger.info(
            f"Submitted execution request for order {order.order_id} to {destination.value}"
        )
        return request

    def update_execution_status(
        self,
        order_id: str,
        status: OrderStatus,
        executed_quantity: int = 0,
        executed_price: Optional[float] = None,
        commission: float = 0.0,
        message: Optional[str] = None,
    ) -> ExecutionResult:
        """
        更新执行状态（外部系统调用）

        Args:
            order_id: 订单ID
            status: 订单状态
            executed_quantity: 已执行数量
            executed_price: 执行价格
            commission: 佣金
            message: 消息说明

        Returns:
            ExecutionResult: 执行结果对象
        """
        if order_id not in self.pending_requests:
            raise ExecutionError(f"Execution request {order_id} not found")

        request = self.pending_requests[order_id]
        remaining = request.quantity - executed_quantity

        # 计算滑点
        slippage_bps = 0.0
        if executed_price and request.price:
            slippage_bps = abs(executed_price - request.price) / request.price * 10000

        result = ExecutionResult(
            order_id=order_id,
            status=status,
            executed_quantity=executed_quantity,
            executed_price=executed_price,
            remaining_quantity=remaining,
            execution_time=datetime.now(),
            commission=commission,
            slippage_bps=slippage_bps,
            message=message,
        )

        self.execution_results[order_id] = result

        # 如果完全成交或取消，从待执行列表移除
        if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self.pending_requests.pop(order_id, None)

        # 更新数据库
        # 首先获取现有订单信息
        existing_order = self.db_manager.get_order(order_id)
        if existing_order:
            # 更新订单信息
            self.db_manager.save_order(
                order_id=order_id,
                signal_id=existing_order["signal_id"],
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                quantity=request.quantity,
                price=request.price,
                status=status.value,
                strategy_name=existing_order.get("strategy_name"),
                timestamp=existing_order.get("created_at"),
                filled_quantity=executed_quantity,
                filled_price=executed_price,
                stop_price=request.stop_price,
                updated_at=datetime.now(),
                filled_time=datetime.now()
                if status == OrderStatus.FILLED
                else existing_order.get("filled_time"),
                metadata=existing_order.get("metadata"),
            )

        # 如果有成交，保存成交记录
        if executed_quantity > 0 and executed_price:
            self.db_manager.save_trade(
                order_id=order_id,
                symbol=request.symbol,
                side=request.side,
                quantity=executed_quantity,
                price=executed_price,
                commission=commission,
                slippage_bps=slippage_bps,
                timestamp=datetime.now(),
            )

        logger.info(f"Updated execution status for order {order_id}: {status.value}")
        return result

    def get_execution_request(self, order_id: str) -> Optional[ExecutionRequest]:
        """
        获取执行请求

        Args:
            order_id: 订单ID

        Returns:
            ExecutionRequest或None
        """
        return self.pending_requests.get(order_id)

    def get_execution_result(self, order_id: str) -> Optional[ExecutionResult]:
        """
        获取执行结果

        Args:
            order_id: 订单ID

        Returns:
            ExecutionResult或None
        """
        return self.execution_results.get(order_id)

    def get_all_pending_requests(self) -> List[ExecutionRequest]:
        """
        获取所有待执行请求

        Returns:
            执行请求列表
        """
        return list(self.pending_requests.values())

    def get_pending_requests_by_symbol(self, symbol: str) -> List[ExecutionRequest]:
        """
        按股票代码获取待执行请求

        Args:
            symbol: 股票代码

        Returns:
            执行请求列表
        """
        return [req for req in self.pending_requests.values() if req.symbol == symbol]

    def get_pending_requests_by_destination(
        self, destination: ExecutionDestination
    ) -> List[ExecutionRequest]:
        """
        按执行目的地获取待执行请求

        Args:
            destination: 执行目的地

        Returns:
            执行请求列表
        """
        return [
            req
            for req in self.pending_requests.values()
            if req.destination == destination
        ]

    def cancel_execution_request(
        self, order_id: str, reason: Optional[str] = None
    ) -> bool:
        """
        取消执行请求

        Args:
            order_id: 订单ID
            reason: 取消原因

        Returns:
            是否成功取消
        """
        if order_id not in self.pending_requests:
            logger.warning(f"Execution request {order_id} not found for cancellation")
            return False

        # 更新状态为取消
        self.update_execution_status(
            order_id=order_id,
            status=OrderStatus.CANCELLED,
            message=reason or "Execution request cancelled",
        )

        logger.info(f"Cancelled execution request {order_id}")
        return True

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        获取执行摘要统计

        Returns:
            包含统计信息的字典
        """
        pending_count = len(self.pending_requests)
        completed_count = len(
            [
                r
                for r in self.execution_results.values()
                if r.status == OrderStatus.FILLED
            ]
        )
        cancelled_count = len(
            [
                r
                for r in self.execution_results.values()
                if r.status == OrderStatus.CANCELLED
            ]
        )

        total_executed_quantity = sum(
            r.executed_quantity for r in self.execution_results.values()
        )

        avg_slippage = 0.0
        if self.execution_results:
            avg_slippage = sum(
                r.slippage_bps for r in self.execution_results.values()
            ) / len(self.execution_results)

        total_commission = sum(r.commission for r in self.execution_results.values())

        return {
            "pending_requests": pending_count,
            "completed_executions": completed_count,
            "cancelled_executions": cancelled_count,
            "total_executed_quantity": total_executed_quantity,
            "avg_slippage_bps": avg_slippage,
            "total_commission": total_commission,
            "fill_rate": completed_count / (completed_count + cancelled_count)
            if (completed_count + cancelled_count) > 0
            else 0.0,
        }


# 全局执行接口实例
_global_execution_interface: Optional[ExecutionInterface] = None


def get_execution_interface() -> ExecutionInterface:
    """
    获取全局执行接口实例

    Returns:
        ExecutionInterface实例
    """
    global _global_execution_interface
    if _global_execution_interface is None:
        _global_execution_interface = ExecutionInterface()
    return _global_execution_interface
