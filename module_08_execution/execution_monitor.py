"""
执行监控器模块
负责实时监控交易执行质量和性能
"""

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from common.exceptions import ExecutionError
from common.logging_system import setup_logger
from module_08_execution.order_manager import Order, OrderStatus

logger = setup_logger("execution_monitor")


@dataclass
class ExecutionMetrics:
    """执行指标"""

    timestamp: datetime
    order_count: int
    fill_rate: float
    avg_slippage_bps: float
    avg_execution_time: float
    total_commission: float
    total_market_impact_bps: float
    rejection_rate: float
    cancellation_rate: float
    partial_fill_rate: float


@dataclass
class ExecutionAlert:
    """执行预警"""

    alert_id: str
    timestamp: datetime
    severity: str  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    alert_type: str
    message: str
    affected_orders: List[str]
    metrics: Dict[str, Any]
    action_required: bool


@dataclass
class ExecutionPerformance:
    """执行性能统计"""

    period: str  # 'DAILY', 'WEEKLY', 'MONTHLY'
    start_time: datetime
    end_time: datetime
    total_orders: int
    successful_orders: int
    failed_orders: int
    total_volume: int
    total_value: float
    avg_fill_rate: float
    avg_slippage_bps: float
    total_commission: float
    best_execution: Dict[str, Any]
    worst_execution: Dict[str, Any]
    by_symbol: Dict[str, Dict[str, Any]]
    by_algorithm: Dict[str, Dict[str, Any]]


class ExecutionMonitor:
    """执行监控器类"""

    def __init__(self, config: Dict[str, Any]):
        """初始化执行监控器

        Args:
            config: 配置字典
        """
        self.config = config

        # 监控数据存储
        self.order_history: deque = deque(maxlen=10000)
        self.metrics_history: deque = deque(maxlen=1440)  # 24小时，每分钟一个点
        self.alerts: List[ExecutionAlert] = []

        # 实时统计
        self.current_metrics = defaultdict(float)
        self.symbol_metrics = defaultdict(lambda: defaultdict(float))
        self.algorithm_metrics = defaultdict(lambda: defaultdict(float))

        # 阈值设置
        self.thresholds = {
            "max_slippage_bps": config.get("max_slippage_bps", 10),
            "min_fill_rate": config.get("min_fill_rate", 0.95),
            "max_rejection_rate": config.get("max_rejection_rate", 0.05),
            "max_execution_time": config.get("max_execution_time", 60),
            "max_market_impact_bps": config.get("max_market_impact_bps", 20),
        }

        # 回调函数
        self.alert_callbacks: List[Callable[[ExecutionAlert], None]] = []
        self.metrics_callbacks: List[Callable[[ExecutionMetrics], None]] = []

        # 监控线程
        self.monitor_thread: Optional[threading.Thread] = None
        self.is_running = False

        # 监控间隔
        self.monitor_interval = config.get("monitor_interval", 60)  # 秒

    def start(self) -> None:
        """启动监控"""
        if self.is_running:
            logger.warning("Execution monitor already running")
            return

        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Execution monitor started")

    def stop(self) -> None:
        """停止监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Execution monitor stopped")

    def track_order(self, order: Order, execution_report: Dict[str, Any]) -> None:
        """跟踪订单执行

        Args:
            order: 订单对象
            execution_report: 执行报告
        """
        # 记录订单历史
        order_record = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "status": order.status,
            "submitted_time": order.submitted_time,
            "filled_time": order.filled_time,
            "execution_report": execution_report,
            "timestamp": datetime.now(),
        }

        self.order_history.append(order_record)

        # 更新实时统计
        self._update_realtime_metrics(order_record)

        # 检查执行质量
        alerts = self._check_execution_quality(order_record)
        for alert in alerts:
            self._trigger_alert(alert)

    def get_current_metrics(self) -> ExecutionMetrics:
        """获取当前执行指标

        Returns:
            执行指标对象
        """
        # 计算最近一段时间的指标
        recent_orders = [
            o
            for o in self.order_history
            if o["timestamp"] > datetime.now() - timedelta(minutes=60)
        ]

        if not recent_orders:
            return ExecutionMetrics(
                timestamp=datetime.now(),
                order_count=0,
                fill_rate=1.0,
                avg_slippage_bps=0.0,
                avg_execution_time=0.0,
                total_commission=0.0,
                total_market_impact_bps=0.0,
                rejection_rate=0.0,
                cancellation_rate=0.0,
                partial_fill_rate=0.0,
            )

        # 计算各项指标
        order_count = len(recent_orders)

        filled_orders = [o for o in recent_orders if o["status"] == OrderStatus.FILLED]
        fill_rate = len(filled_orders) / order_count if order_count > 0 else 0

        rejected_orders = [
            o for o in recent_orders if o["status"] == OrderStatus.REJECTED
        ]
        rejection_rate = len(rejected_orders) / order_count if order_count > 0 else 0

        cancelled_orders = [
            o for o in recent_orders if o["status"] == OrderStatus.CANCELLED
        ]
        cancellation_rate = (
            len(cancelled_orders) / order_count if order_count > 0 else 0
        )

        partial_orders = [
            o for o in recent_orders if o["status"] == OrderStatus.PARTIALLY_FILLED
        ]
        partial_fill_rate = len(partial_orders) / order_count if order_count > 0 else 0

        # 计算滑点
        slippages = []
        for order in filled_orders:
            if "execution_report" in order and order["execution_report"]:
                slippage = order["execution_report"].get("slippage_bps", 0)
                slippages.append(slippage)
        avg_slippage_bps = np.mean(slippages) if slippages else 0

        # 计算执行时间
        execution_times = []
        for order in filled_orders:
            if order["submitted_time"] and order["filled_time"]:
                exec_time = (
                    order["filled_time"] - order["submitted_time"]
                ).total_seconds()
                execution_times.append(exec_time)
        avg_execution_time = np.mean(execution_times) if execution_times else 0

        # 计算总佣金
        total_commission = sum(
            order.get("execution_report", {}).get("commission", 0)
            for order in recent_orders
        )

        # 计算市场冲击
        impacts = []
        for order in filled_orders:
            if "execution_report" in order and order["execution_report"]:
                impact = order["execution_report"].get("market_impact_bps", 0)
                impacts.append(impact)
        total_market_impact_bps = np.mean(impacts) if impacts else 0

        return ExecutionMetrics(
            timestamp=datetime.now(),
            order_count=order_count,
            fill_rate=fill_rate,
            avg_slippage_bps=avg_slippage_bps,
            avg_execution_time=avg_execution_time,
            total_commission=total_commission,
            total_market_impact_bps=total_market_impact_bps,
            rejection_rate=rejection_rate,
            cancellation_rate=cancellation_rate,
            partial_fill_rate=partial_fill_rate,
        )

    def get_performance_report(
        self,
        period: str = "DAILY",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> ExecutionPerformance:
        """获取执行性能报告

        Args:
            period: 统计周期
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            执行性能对象
        """
        # 确定时间范围
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            if period == "DAILY":
                start_date = end_date - timedelta(days=1)
            elif period == "WEEKLY":
                start_date = end_date - timedelta(weeks=1)
            else:  # MONTHLY
                start_date = end_date - timedelta(days=30)

        # 筛选订单
        period_orders = [
            o for o in self.order_history if start_date <= o["timestamp"] <= end_date
        ]

        if not period_orders:
            return ExecutionPerformance(
                period=period,
                start_time=start_date,
                end_time=end_date,
                total_orders=0,
                successful_orders=0,
                failed_orders=0,
                total_volume=0,
                total_value=0.0,
                avg_fill_rate=0.0,
                avg_slippage_bps=0.0,
                total_commission=0.0,
                best_execution={},
                worst_execution={},
                by_symbol={},
                by_algorithm={},
            )

        # 统计基本指标
        total_orders = len(period_orders)
        successful_orders = len(
            [o for o in period_orders if o["status"] == OrderStatus.FILLED]
        )
        failed_orders = len(
            [
                o
                for o in period_orders
                if o["status"] in [OrderStatus.REJECTED, OrderStatus.ERROR]
            ]
        )

        # 计算总量和总值
        total_volume = sum(o["filled_quantity"] for o in period_orders)
        total_value = sum(
            o["filled_quantity"] * o.get("execution_report", {}).get("avg_price", 0)
            for o in period_orders
        )

        # 计算平均成交率
        fill_rates = []
        for order in period_orders:
            if order["quantity"] > 0:
                fill_rate = order["filled_quantity"] / order["quantity"]
                fill_rates.append(fill_rate)
        avg_fill_rate = np.mean(fill_rates) if fill_rates else 0

        # 计算平均滑点
        slippages = []
        for order in period_orders:
            if "execution_report" in order and order["execution_report"]:
                slippage = order["execution_report"].get("slippage_bps", 0)
                slippages.append(slippage)
        avg_slippage_bps = np.mean(slippages) if slippages else 0

        # 计算总佣金
        total_commission = sum(
            o.get("execution_report", {}).get("commission", 0) for o in period_orders
        )

        # 找出最佳和最差执行
        best_execution = self._find_best_execution(period_orders)
        worst_execution = self._find_worst_execution(period_orders)

        # 按标的统计
        by_symbol = self._aggregate_by_symbol(period_orders)

        # 按算法统计
        by_algorithm = self._aggregate_by_algorithm(period_orders)

        return ExecutionPerformance(
            period=period,
            start_time=start_date,
            end_time=end_date,
            total_orders=total_orders,
            successful_orders=successful_orders,
            failed_orders=failed_orders,
            total_volume=total_volume,
            total_value=total_value,
            avg_fill_rate=avg_fill_rate,
            avg_slippage_bps=avg_slippage_bps,
            total_commission=total_commission,
            best_execution=best_execution,
            worst_execution=worst_execution,
            by_symbol=by_symbol,
            by_algorithm=by_algorithm,
        )

    def register_alert_callback(
        self, callback: Callable[[ExecutionAlert], None]
    ) -> None:
        """注册预警回调

        Args:
            callback: 回调函数
        """
        self.alert_callbacks.append(callback)

    def register_metrics_callback(
        self, callback: Callable[[ExecutionMetrics], None]
    ) -> None:
        """注册指标回调

        Args:
            callback: 回调函数
        """
        self.metrics_callbacks.append(callback)

    def _monitor_loop(self) -> None:
        """监控循环"""
        import time

        while self.is_running:
            try:
                # 获取当前指标
                metrics = self.get_current_metrics()

                # 保存历史
                self.metrics_history.append(metrics)

                # 触发回调
                for callback in self.metrics_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Metrics callback error: {e}")

                # 检查阈值
                self._check_thresholds(metrics)

                # 等待下一个周期
                time.sleep(self.monitor_interval)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

    def _update_realtime_metrics(self, order_record: Dict[str, Any]) -> None:
        """更新实时指标

        Args:
            order_record: 订单记录
        """
        symbol = order_record["symbol"]

        # 更新标的指标
        self.symbol_metrics[symbol]["order_count"] += 1
        if order_record["status"] == OrderStatus.FILLED:
            self.symbol_metrics[symbol]["filled_count"] += 1

        # 更新算法指标（如果有）
        if "algorithm" in order_record.get("execution_report", {}):
            algo = order_record["execution_report"]["algorithm"]
            self.algorithm_metrics[algo]["order_count"] += 1
            if order_record["status"] == OrderStatus.FILLED:
                self.algorithm_metrics[algo]["filled_count"] += 1

    def _check_execution_quality(
        self, order_record: Dict[str, Any]
    ) -> List[ExecutionAlert]:
        """检查执行质量

        Args:
            order_record: 订单记录

        Returns:
            预警列表
        """
        alerts = []

        if (
            "execution_report" not in order_record
            or not order_record["execution_report"]
        ):
            return alerts

        report = order_record["execution_report"]

        # 检查滑点
        slippage = report.get("slippage_bps", 0)
        if abs(slippage) > self.thresholds["max_slippage_bps"]:
            alerts.append(
                ExecutionAlert(
                    alert_id=f"SLIP_{order_record['order_id']}",
                    timestamp=datetime.now(),
                    severity="WARNING",
                    alert_type="HIGH_SLIPPAGE",
                    message=f"High slippage detected: {slippage:.2f} bps",
                    affected_orders=[order_record["order_id"]],
                    metrics={"slippage_bps": slippage},
                    action_required=False,
                )
            )

        # 检查执行时间
        if order_record["submitted_time"] and order_record["filled_time"]:
            exec_time = (
                order_record["filled_time"] - order_record["submitted_time"]
            ).total_seconds()
            if exec_time > self.thresholds["max_execution_time"]:
                alerts.append(
                    ExecutionAlert(
                        alert_id=f"TIME_{order_record['order_id']}",
                        timestamp=datetime.now(),
                        severity="WARNING",
                        alert_type="SLOW_EXECUTION",
                        message=f"Slow execution: {exec_time:.1f} seconds",
                        affected_orders=[order_record["order_id"]],
                        metrics={"execution_time": exec_time},
                        action_required=False,
                    )
                )

        # 检查市场冲击
        impact = report.get("market_impact_bps", 0)
        if abs(impact) > self.thresholds["max_market_impact_bps"]:
            alerts.append(
                ExecutionAlert(
                    alert_id=f"IMPACT_{order_record['order_id']}",
                    timestamp=datetime.now(),
                    severity="WARNING",
                    alert_type="HIGH_MARKET_IMPACT",
                    message=f"High market impact: {impact:.2f} bps",
                    affected_orders=[order_record["order_id"]],
                    metrics={"market_impact_bps": impact},
                    action_required=True,
                )
            )

        return alerts

    def _check_thresholds(self, metrics: ExecutionMetrics) -> None:
        """检查阈值

        Args:
            metrics: 执行指标
        """
        # 检查成交率
        if metrics.fill_rate < self.thresholds["min_fill_rate"]:
            alert = ExecutionAlert(
                alert_id=f"FILL_RATE_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                severity="ERROR",
                alert_type="LOW_FILL_RATE",
                message=f"Low fill rate: {metrics.fill_rate:.2%}",
                affected_orders=[],
                metrics={"fill_rate": metrics.fill_rate},
                action_required=True,
            )
            self._trigger_alert(alert)

        # 检查拒绝率
        if metrics.rejection_rate > self.thresholds["max_rejection_rate"]:
            alert = ExecutionAlert(
                alert_id=f"REJECT_RATE_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                severity="CRITICAL",
                alert_type="HIGH_REJECTION_RATE",
                message=f"High rejection rate: {metrics.rejection_rate:.2%}",
                affected_orders=[],
                metrics={"rejection_rate": metrics.rejection_rate},
                action_required=True,
            )
            self._trigger_alert(alert)

    def _trigger_alert(self, alert: ExecutionAlert) -> None:
        """触发预警

        Args:
            alert: 预警对象
        """
        # 保存预警
        self.alerts.append(alert)

        # 触发回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        # 记录日志
        if alert.severity == "CRITICAL":
            logger.critical(f"[{alert.alert_type}] {alert.message}")
        elif alert.severity == "ERROR":
            logger.error(f"[{alert.alert_type}] {alert.message}")
        elif alert.severity == "WARNING":
            logger.warning(f"[{alert.alert_type}] {alert.message}")
        else:
            logger.info(f"[{alert.alert_type}] {alert.message}")

    def _find_best_execution(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """找出最佳执行

        Args:
            orders: 订单列表

        Returns:
            最佳执行信息
        """
        best_order = None
        best_score = float("-inf")

        for order in orders:
            if order["status"] != OrderStatus.FILLED:
                continue

            report = order.get("execution_report", {})

            # 计算得分（低滑点、快执行、低成本）
            slippage = report.get("slippage_bps", 0)
            exec_time = report.get("execution_time", float("inf"))

            score = -abs(slippage) - exec_time / 60  # 简化评分

            if score > best_score:
                best_score = score
                best_order = order

        if best_order:
            return {
                "order_id": best_order["order_id"],
                "symbol": best_order["symbol"],
                "slippage_bps": best_order.get("execution_report", {}).get(
                    "slippage_bps", 0
                ),
                "execution_time": best_order.get("execution_report", {}).get(
                    "execution_time", 0
                ),
                "score": best_score,
            }

        return {}

    def _find_worst_execution(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """找出最差执行

        Args:
            orders: 订单列表

        Returns:
            最差执行信息
        """
        worst_order = None
        worst_score = float("inf")

        for order in orders:
            report = order.get("execution_report", {})

            # 计算得分
            slippage = report.get("slippage_bps", 0)
            exec_time = report.get("execution_time", 0)

            score = -abs(slippage) - exec_time / 60

            if score < worst_score:
                worst_score = score
                worst_order = order

        if worst_order:
            return {
                "order_id": worst_order["order_id"],
                "symbol": worst_order["symbol"],
                "slippage_bps": worst_order.get("execution_report", {}).get(
                    "slippage_bps", 0
                ),
                "execution_time": worst_order.get("execution_report", {}).get(
                    "execution_time", 0
                ),
                "score": worst_score,
            }

        return {}

    def _aggregate_by_symbol(
        self, orders: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """按标的聚合统计

        Args:
            orders: 订单列表

        Returns:
            按标的的统计字典
        """
        by_symbol = defaultdict(
            lambda: {
                "order_count": 0,
                "filled_count": 0,
                "total_volume": 0,
                "avg_slippage_bps": 0,
                "total_commission": 0,
            }
        )

        for order in orders:
            symbol = order["symbol"]
            by_symbol[symbol]["order_count"] += 1

            if order["status"] == OrderStatus.FILLED:
                by_symbol[symbol]["filled_count"] += 1
                by_symbol[symbol]["total_volume"] += order["filled_quantity"]

                report = order.get("execution_report", {})
                by_symbol[symbol]["total_commission"] += report.get("commission", 0)

        # 计算平均值
        for symbol, stats in by_symbol.items():
            if stats["filled_count"] > 0:
                stats["fill_rate"] = stats["filled_count"] / stats["order_count"]
            else:
                stats["fill_rate"] = 0

        return dict(by_symbol)

    def _aggregate_by_algorithm(
        self, orders: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """按算法聚合统计

        Args:
            orders: 订单列表

        Returns:
            按算法的统计字典
        """
        by_algo = defaultdict(
            lambda: {
                "order_count": 0,
                "filled_count": 0,
                "total_volume": 0,
                "avg_slippage_bps": 0,
                "avg_execution_time": 0,
            }
        )

        for order in orders:
            report = order.get("execution_report", {})
            algo = report.get("algorithm", "UNKNOWN")

            by_algo[algo]["order_count"] += 1

            if order["status"] == OrderStatus.FILLED:
                by_algo[algo]["filled_count"] += 1
                by_algo[algo]["total_volume"] += order["filled_quantity"]

        # 计算平均值
        for algo, stats in by_algo.items():
            if stats["filled_count"] > 0:
                stats["fill_rate"] = stats["filled_count"] / stats["order_count"]
            else:
                stats["fill_rate"] = 0

        return dict(by_algo)


# 模块级别函数
def create_execution_monitor(config: Dict[str, Any]) -> ExecutionMonitor:
    """创建执行监控器实例

    Args:
        config: 配置字典

    Returns:
        执行监控器实例
    """
    monitor = ExecutionMonitor(config)
    monitor.start()
    return monitor
