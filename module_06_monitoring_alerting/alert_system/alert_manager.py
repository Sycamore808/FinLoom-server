"""
预警管理器模块
管理和协调系统预警
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from queue import PriorityQueue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.exceptions import ModelError
from common.logging_system import setup_logger

logger = setup_logger("alert_manager")


class AlertSeverity(Enum):
    """预警严重级别枚举"""

    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class AlertCategory(Enum):
    """预警类别枚举"""

    PORTFOLIO = "portfolio"
    RISK = "risk"
    MARKET = "market"
    SYSTEM = "system"
    EXECUTION = "execution"
    DATA = "data"
    COMPLIANCE = "compliance"


class AlertStatus(Enum):
    """预警状态枚举"""

    PENDING = "pending"
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


@dataclass
class AlertRule:
    """预警规则"""

    rule_id: str
    name: str
    description: str
    category: AlertCategory
    severity: AlertSeverity
    condition: str  # 条件表达式
    threshold: float
    comparison: str  # >, <, ==, !=, >=, <=
    metric: str  # 监控指标名称
    cooldown_seconds: int = 300  # 冷却时间
    auto_resolve: bool = False
    escalation_time: int = 600  # 升级时间（秒）
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """预警对象"""

    alert_id: str
    rule_id: str
    timestamp: datetime
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    metric_value: float
    threshold_value: float
    status: AlertStatus
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalated: bool = False
    escalation_level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """预警配置"""

    max_alerts_per_rule: int = 10  # 每个规则最大预警数
    alert_retention_days: int = 30  # 预警保留天数
    enable_auto_escalation: bool = True
    escalation_levels: List[int] = field(default_factory=lambda: [600, 1800, 3600])
    enable_alert_suppression: bool = True
    suppression_window: int = 300  # 抑制窗口（秒）
    enable_alert_aggregation: bool = True
    aggregation_window: int = 60  # 聚合窗口（秒）


@dataclass
class AlertStatistics:
    """预警统计"""

    total_alerts: int
    alerts_by_severity: Dict[AlertSeverity, int]
    alerts_by_category: Dict[AlertCategory, int]
    alerts_by_status: Dict[AlertStatus, int]
    average_resolution_time: float
    escalation_rate: float
    false_positive_rate: float
    top_triggered_rules: List[Tuple[str, int]]


class AlertManager:
    """预警管理器类"""

    def __init__(self, config: Optional[AlertConfig] = None):
        """初始化预警管理器

        Args:
            config: 预警配置
        """
        self.config = config or AlertConfig()
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_queue: PriorityQueue = PriorityQueue()
        self.rule_cooldowns: Dict[str, datetime] = {}
        self.suppressed_alerts: Dict[str, List[Alert]] = defaultdict(list)
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = {
            severity: [] for severity in AlertSeverity
        }
        self.monitoring_active = False

    def add_rule(self, rule: AlertRule) -> None:
        """添加预警规则

        Args:
            rule: 预警规则
        """
        self.rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_id: str) -> None:
        """移除预警规则

        Args:
            rule_id: 规则ID
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")

    def enable_rule(self, rule_id: str) -> None:
        """启用预警规则

        Args:
            rule_id: 规则ID
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            logger.info(f"Enabled alert rule: {rule_id}")

    def disable_rule(self, rule_id: str) -> None:
        """禁用预警规则

        Args:
            rule_id: 规则ID
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            logger.info(f"Disabled alert rule: {rule_id}")

    def check_rules(self, metrics: Dict[str, float]) -> List[Alert]:
        """检查预警规则

        Args:
            metrics: 指标字典

        Returns:
            触发的预警列表
        """
        triggered_alerts = []

        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue

            # 检查冷却时间
            if self._is_in_cooldown(rule_id):
                continue

            # 检查条件
            if self._evaluate_rule(rule, metrics):
                alert = self._create_alert(rule, metrics.get(rule.metric, 0))

                # 检查是否应该抑制
                if self.config.enable_alert_suppression:
                    if self._should_suppress_alert(alert):
                        self.suppressed_alerts[rule_id].append(alert)
                        continue

                triggered_alerts.append(alert)
                self._update_cooldown(rule_id, rule.cooldown_seconds)

        return triggered_alerts

    def trigger_alert(
        self, rule_id: str, metric_value: float, message: Optional[str] = None
    ) -> Alert:
        """手动触发预警

        Args:
            rule_id: 规则ID
            metric_value: 指标值
            message: 自定义消息

        Returns:
            预警对象
        """
        if rule_id not in self.rules:
            raise ValueError(f"Rule {rule_id} not found")

        rule = self.rules[rule_id]
        alert = self._create_alert(rule, metric_value, message)

        self._process_alert(alert)

        return alert

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """确认预警

        Args:
            alert_id: 预警ID
            acknowledged_by: 确认人

        Returns:
            是否成功确认
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True

        return False

    def resolve_alert(
        self, alert_id: str, resolution_notes: Optional[str] = None
    ) -> bool:
        """解决预警

        Args:
            alert_id: 预警ID
            resolution_notes: 解决说明

        Returns:
            是否成功解决
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()

            if resolution_notes:
                alert.metadata["resolution_notes"] = resolution_notes

            # 移到历史记录
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]

            logger.info(f"Alert {alert_id} resolved")
            return True

        return False

    def escalate_alert(self, alert_id: str) -> bool:
        """升级预警

        Args:
            alert_id: 预警ID

        Returns:
            是否成功升级
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.escalated = True
            alert.escalation_level += 1
            alert.status = AlertStatus.ESCALATED

            # 触发升级处理
            self._handle_escalation(alert)

            logger.warning(
                f"Alert {alert_id} escalated to level {alert.escalation_level}"
            )
            return True

        return False

    async def start_monitoring(self) -> None:
        """启动预警监控"""
        logger.info("Starting alert monitoring")
        self.monitoring_active = True

        while self.monitoring_active:
            try:
                # 检查预警升级
                self._check_escalations()

                # 处理预警队列
                await self._process_alert_queue()

                # 清理过期预警
                self._cleanup_old_alerts()

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                await asyncio.sleep(5)

    def stop_monitoring(self) -> None:
        """停止预警监控"""
        logger.info("Stopping alert monitoring")
        self.monitoring_active = False

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        category: Optional[AlertCategory] = None,
    ) -> List[Alert]:
        """获取活跃预警

        Args:
            severity: 严重级别筛选
            category: 类别筛选

        Returns:
            预警列表
        """
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if category:
            alerts = [a for a in alerts if a.category == category]

        return alerts

    def get_alert_statistics(self) -> AlertStatistics:
        """获取预警统计

        Returns:
            预警统计信息
        """
        all_alerts = list(self.active_alerts.values()) + self.alert_history

        if not all_alerts:
            return AlertStatistics(
                total_alerts=0,
                alerts_by_severity={},
                alerts_by_category={},
                alerts_by_status={},
                average_resolution_time=0,
                escalation_rate=0,
                false_positive_rate=0,
                top_triggered_rules=[],
            )

        # 按严重级别统计
        alerts_by_severity = defaultdict(int)
        for alert in all_alerts:
            alerts_by_severity[alert.severity] += 1

        # 按类别统计
        alerts_by_category = defaultdict(int)
        for alert in all_alerts:
            alerts_by_category[alert.category] += 1

        # 按状态统计
        alerts_by_status = defaultdict(int)
        for alert in all_alerts:
            alerts_by_status[alert.status] += 1

        # 计算平均解决时间
        resolved_alerts = [a for a in all_alerts if a.status == AlertStatus.RESOLVED]
        if resolved_alerts:
            resolution_times = [
                (a.resolved_at - a.timestamp).total_seconds()
                for a in resolved_alerts
                if a.resolved_at
            ]
            avg_resolution_time = np.mean(resolution_times) if resolution_times else 0
        else:
            avg_resolution_time = 0

        # 计算升级率
        escalated_count = sum(1 for a in all_alerts if a.escalated)
        escalation_rate = escalated_count / len(all_alerts) if all_alerts else 0

        # 统计触发最多的规则
        rule_counts = defaultdict(int)
        for alert in all_alerts:
            rule_counts[alert.rule_id] += 1
        top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return AlertStatistics(
            total_alerts=len(all_alerts),
            alerts_by_severity=dict(alerts_by_severity),
            alerts_by_category=dict(alerts_by_category),
            alerts_by_status=dict(alerts_by_status),
            average_resolution_time=avg_resolution_time,
            escalation_rate=escalation_rate,
            false_positive_rate=0,  # 需要额外跟踪
            top_triggered_rules=top_rules,
        )

    def register_handler(self, severity: AlertSeverity, handler: Callable) -> None:
        """注册预警处理器

        Args:
            severity: 严重级别
            handler: 处理函数
        """
        self.alert_handlers[severity].append(handler)
        logger.info(f"Registered handler for {severity.name} alerts")

    def _evaluate_rule(self, rule: AlertRule, metrics: Dict[str, float]) -> bool:
        """评估规则条件

        Args:
            rule: 预警规则
            metrics: 指标字典

        Returns:
            是否触发
        """
        if rule.metric not in metrics:
            return False

        metric_value = metrics[rule.metric]
        threshold = rule.threshold

        comparisons = {
            ">": metric_value > threshold,
            "<": metric_value < threshold,
            ">=": metric_value >= threshold,
            "<=": metric_value <= threshold,
            "==": metric_value == threshold,
            "!=": metric_value != threshold,
        }

        return comparisons.get(rule.comparison, False)

    def _create_alert(
        self, rule: AlertRule, metric_value: float, custom_message: Optional[str] = None
    ) -> Alert:
        """创建预警对象

        Args:
            rule: 预警规则
            metric_value: 指标值
            custom_message: 自定义消息

        Returns:
            预警对象
        """
        alert_id = f"{rule.rule_id}_{datetime.now().timestamp()}"

        if custom_message:
            message = custom_message
        else:
            message = (
                f"{rule.description}: {rule.metric} = {metric_value:.4f} "
                f"{rule.comparison} {rule.threshold:.4f}"
            )

        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            timestamp=datetime.now(),
            severity=rule.severity,
            category=rule.category,
            title=rule.name,
            message=message,
            metric_value=metric_value,
            threshold_value=rule.threshold,
            status=AlertStatus.TRIGGERED,
            metadata=rule.metadata.copy(),
        )

        return alert

    def _process_alert(self, alert: Alert) -> None:
        """处理预警

        Args:
            alert: 预警对象
        """
        # 添加到活跃预警
        self.active_alerts[alert.alert_id] = alert

        # 添加到队列
        priority = -alert.severity.value  # 负数使高严重级别优先
        self.alert_queue.put((priority, alert.timestamp, alert))

        # 触发处理器
        for handler in self.alert_handlers.get(alert.severity, []):
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

        logger.warning(f"Alert triggered: {alert.title} ({alert.severity.name})")

    def _is_in_cooldown(self, rule_id: str) -> bool:
        """检查是否在冷却期

        Args:
            rule_id: 规则ID

        Returns:
            是否在冷却期
        """
        if rule_id in self.rule_cooldowns:
            cooldown_end = self.rule_cooldowns[rule_id]
            return datetime.now() < cooldown_end

        return False

    def _update_cooldown(self, rule_id: str, cooldown_seconds: int) -> None:
        """更新冷却时间

        Args:
            rule_id: 规则ID
            cooldown_seconds: 冷却秒数
        """
        self.rule_cooldowns[rule_id] = datetime.now() + timedelta(
            seconds=cooldown_seconds
        )

    def _should_suppress_alert(self, alert: Alert) -> bool:
        """判断是否应该抑制预警

        Args:
            alert: 预警对象

        Returns:
            是否抑制
        """
        # 检查最近是否有相同规则的预警
        recent_alerts = self.suppressed_alerts.get(alert.rule_id, [])

        # 清理过期的抑制预警
        cutoff_time = datetime.now() - timedelta(seconds=self.config.suppression_window)
        recent_alerts = [a for a in recent_alerts if a.timestamp > cutoff_time]
        self.suppressed_alerts[alert.rule_id] = recent_alerts

        # 如果有最近的预警，则抑制
        return len(recent_alerts) > 0

    def _check_escalations(self) -> None:
        """检查需要升级的预警"""
        if not self.config.enable_auto_escalation:
            return

        for alert_id, alert in self.active_alerts.items():
            if alert.status == AlertStatus.TRIGGERED and not alert.escalated:
                # 检查是否超过升级时间
                time_since_trigger = (datetime.now() - alert.timestamp).total_seconds()

                if alert.escalation_level < len(self.config.escalation_levels):
                    escalation_time = self.config.escalation_levels[
                        alert.escalation_level
                    ]

                    if time_since_trigger > escalation_time:
                        self.escalate_alert(alert_id)

    async def _process_alert_queue(self) -> None:
        """处理预警队列"""
        while not self.alert_queue.empty():
            try:
                _, _, alert = self.alert_queue.get_nowait()

                # 异步处理预警
                await self._async_process_alert(alert)

            except Exception as e:
                logger.error(f"Error processing alert queue: {e}")

    async def _async_process_alert(self, alert: Alert) -> None:
        """异步处理预警

        Args:
            alert: 预警对象
        """
        # 这里可以添加异步处理逻辑
        # 例如发送通知、写入数据库等
        pass

    def _cleanup_old_alerts(self) -> None:
        """清理过期预警"""
        cutoff_date = datetime.now() - timedelta(days=self.config.alert_retention_days)

        # 清理历史预警
        self.alert_history = [
            a for a in self.alert_history if a.timestamp > cutoff_date
        ]

        # 清理抑制的预警
        for rule_id in list(self.suppressed_alerts.keys()):
            self.suppressed_alerts[rule_id] = [
                a for a in self.suppressed_alerts[rule_id] if a.timestamp > cutoff_date
            ]

    def _handle_escalation(self, alert: Alert) -> None:
        """处理预警升级

        Args:
            alert: 预警对象
        """
        # 这里可以添加升级处理逻辑
        # 例如发送给更高级别的人员、触发紧急响应等
        pass


# 模块级别函数
def create_alert_manager(config: Optional[AlertConfig] = None) -> AlertManager:
    """创建预警管理器的便捷函数

    Args:
        config: 预警配置

    Returns:
        预警管理器实例
    """
    return AlertManager(config)
