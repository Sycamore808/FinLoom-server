"""
通知管理器模块
管理和协调各种通知渠道
"""

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from queue import PriorityQueue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from common.exceptions import ModelError
from common.logging_system import setup_logger

logger = setup_logger("notification_manager")


class NotificationChannel(Enum):
    """通知渠道枚举"""

    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    PUSH = "push"


class NotificationPriority(Enum):
    """通知优先级枚举"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class NotificationType(Enum):
    """通知类型枚举"""

    ALERT = "alert"
    REPORT = "report"
    TRADE = "trade"
    SYSTEM = "system"
    MARKET = "market"
    RISK = "risk"
    PERFORMANCE = "performance"


@dataclass
class NotificationConfig:
    """通知配置"""

    enabled_channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.EMAIL]
    )
    rate_limits: Dict[NotificationChannel, int] = field(
        default_factory=dict
    )  # 每分钟最大数量
    retry_attempts: int = 3
    retry_delay: int = 60  # 秒
    queue_size: int = 1000
    batch_size: int = 10
    batch_interval: int = 30  # 秒
    enable_aggregation: bool = True
    aggregation_window: int = 300  # 秒
    quiet_hours: Optional[Tuple[int, int]] = None  # (开始小时, 结束小时)
    channel_configs: Dict[NotificationChannel, Dict[str, Any]] = field(
        default_factory=dict
    )


@dataclass
class Notification:
    """通知对象"""

    notification_id: str
    timestamp: datetime
    type: NotificationType
    priority: NotificationPriority
    channel: NotificationChannel
    recipient: str
    subject: str
    message: str
    data: Optional[Dict[str, Any]] = None
    attachments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    sent_at: Optional[datetime] = None
    delivered: bool = False
    error_message: Optional[str] = None


@dataclass
class NotificationTemplate:
    """通知模板"""

    template_id: str
    name: str
    type: NotificationType
    subject_template: str
    body_template: str
    html_template: Optional[str] = None
    variables: List[str] = field(default_factory=list)
    default_values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationStats:
    """通知统计"""

    total_sent: int
    total_delivered: int
    total_failed: int
    by_channel: Dict[NotificationChannel, int]
    by_type: Dict[NotificationType, int]
    by_priority: Dict[NotificationPriority, int]
    average_delivery_time: float
    success_rate: float


class NotificationManager:
    """通知管理器类"""

    def __init__(self, config: Optional[NotificationConfig] = None):
        """初始化通知管理器

        Args:
            config: 通知配置
        """
        self.config = config or NotificationConfig()
        self.notification_queue: PriorityQueue = PriorityQueue(
            maxsize=self.config.queue_size
        )
        self.pending_notifications: Dict[str, Notification] = {}
        self.sent_notifications: List[Notification] = []
        self.templates: Dict[str, NotificationTemplate] = {}
        self.channel_handlers: Dict[NotificationChannel, Callable] = {}
        self.rate_limiters: Dict[NotificationChannel, List[datetime]] = defaultdict(
            list
        )
        self.aggregation_buffer: Dict[str, List[Notification]] = defaultdict(list)
        self.processing_active = False
        self._initialize_handlers()
        self._load_templates()

    def send_notification(
        self,
        type: NotificationType,
        priority: NotificationPriority,
        channel: NotificationChannel,
        recipient: str,
        subject: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[str]] = None,
    ) -> str:
        """发送通知

        Args:
            type: 通知类型
            priority: 优先级
            channel: 通知渠道
            recipient: 接收者
            subject: 主题
            message: 消息内容
            data: 附加数据
            attachments: 附件列表

        Returns:
            通知ID
        """
        # 检查是否在安静时间
        if (
            self._is_quiet_hours()
            and priority.value < NotificationPriority.URGENT.value
        ):
            logger.info(f"Notification delayed due to quiet hours: {subject}")
            # 延迟到安静时间结束
            return self._schedule_after_quiet_hours(
                type, priority, channel, recipient, subject, message, data, attachments
            )

        notification = Notification(
            notification_id=f"{type.value}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            type=type,
            priority=priority,
            channel=channel,
            recipient=recipient,
            subject=subject,
            message=message,
            data=data or {},
            attachments=attachments or [],
        )

        # 检查是否需要聚合
        if (
            self.config.enable_aggregation
            and priority.value <= NotificationPriority.NORMAL.value
        ):
            self._add_to_aggregation_buffer(notification)
        else:
            self._queue_notification(notification)

        return notification.notification_id

    def send_templated_notification(
        self,
        template_id: str,
        priority: NotificationPriority,
        channel: NotificationChannel,
        recipient: str,
        variables: Dict[str, Any],
        attachments: Optional[List[str]] = None,
    ) -> str:
        """使用模板发送通知

        Args:
            template_id: 模板ID
            priority: 优先级
            channel: 通知渠道
            recipient: 接收者
            variables: 模板变量
            attachments: 附件列表

        Returns:
            通知ID
        """
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")

        template = self.templates[template_id]

        # 合并默认值和提供的变量
        vars = {**template.default_values, **variables}

        # 渲染模板
        subject = self._render_template(template.subject_template, vars)
        message = self._render_template(template.body_template, vars)

        return self.send_notification(
            type=template.type,
            priority=priority,
            channel=channel,
            recipient=recipient,
            subject=subject,
            message=message,
            data=vars,
            attachments=attachments,
        )

    def broadcast_notification(
        self,
        type: NotificationType,
        priority: NotificationPriority,
        channels: List[NotificationChannel],
        recipients: List[str],
        subject: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """广播通知到多个渠道和接收者

        Args:
            type: 通知类型
            priority: 优先级
            channels: 通知渠道列表
            recipients: 接收者列表
            subject: 主题
            message: 消息内容
            data: 附加数据

        Returns:
            通知ID列表
        """
        notification_ids = []

        for channel in channels:
            if channel not in self.config.enabled_channels:
                continue

            for recipient in recipients:
                notification_id = self.send_notification(
                    type=type,
                    priority=priority,
                    channel=channel,
                    recipient=recipient,
                    subject=subject,
                    message=message,
                    data=data,
                )
                notification_ids.append(notification_id)

        return notification_ids

    async def start_processing(self) -> None:
        """启动通知处理"""
        logger.info("Starting notification processing")
        self.processing_active = True

        # 启动多个协程
        await asyncio.gather(
            self._process_queue(),
            self._process_aggregation_buffer(),
            self._retry_failed_notifications(),
        )

    def stop_processing(self) -> None:
        """停止通知处理"""
        logger.info("Stopping notification processing")
        self.processing_active = False

    def get_notification_status(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """获取通知状态

        Args:
            notification_id: 通知ID

        Returns:
            状态信息
        """
        # 检查待发送
        if notification_id in self.pending_notifications:
            notification = self.pending_notifications[notification_id]
            return {"status": "pending", "notification": notification}

        # 检查已发送
        for notification in self.sent_notifications:
            if notification.notification_id == notification_id:
                return {
                    "status": "delivered" if notification.delivered else "failed",
                    "notification": notification,
                    "sent_at": notification.sent_at,
                    "error": notification.error_message,
                }

        return None

    def get_statistics(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> NotificationStats:
        """获取通知统计

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            统计信息
        """
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()

        # 筛选时间范围内的通知
        notifications = [
            n for n in self.sent_notifications if start_time <= n.timestamp <= end_time
        ]

        if not notifications:
            return NotificationStats(
                total_sent=0,
                total_delivered=0,
                total_failed=0,
                by_channel={},
                by_type={},
                by_priority={},
                average_delivery_time=0,
                success_rate=0,
            )

        # 计算统计
        total_sent = len(notifications)
        total_delivered = sum(1 for n in notifications if n.delivered)
        total_failed = total_sent - total_delivered

        # 按渠道统计
        by_channel = defaultdict(int)
        for n in notifications:
            by_channel[n.channel] += 1

        # 按类型统计
        by_type = defaultdict(int)
        for n in notifications:
            by_type[n.type] += 1

        # 按优先级统计
        by_priority = defaultdict(int)
        for n in notifications:
            by_priority[n.priority] += 1

        # 计算平均发送时间
        delivery_times = []
        for n in notifications:
            if n.delivered and n.sent_at:
                delivery_time = (n.sent_at - n.timestamp).total_seconds()
                delivery_times.append(delivery_time)

        avg_delivery_time = (
            sum(delivery_times) / len(delivery_times) if delivery_times else 0
        )

        # 计算成功率
        success_rate = total_delivered / total_sent if total_sent > 0 else 0

        return NotificationStats(
            total_sent=total_sent,
            total_delivered=total_delivered,
            total_failed=total_failed,
            by_channel=dict(by_channel),
            by_type=dict(by_type),
            by_priority=dict(by_priority),
            average_delivery_time=avg_delivery_time,
            success_rate=success_rate,
        )

    def add_template(self, template: NotificationTemplate) -> None:
        """添加通知模板

        Args:
            template: 通知模板
        """
        self.templates[template.template_id] = template
        logger.info(f"Added notification template: {template.name}")

    def register_channel_handler(
        self, channel: NotificationChannel, handler: Callable
    ) -> None:
        """注册渠道处理器

        Args:
            channel: 通知渠道
            handler: 处理函数
        """
        self.channel_handlers[channel] = handler
        logger.info(f"Registered handler for channel: {channel.value}")

    def _initialize_handlers(self) -> None:
        """初始化处理器"""
        # 默认处理器（应该被实际的处理器替换）
        self.channel_handlers = {
            NotificationChannel.EMAIL: self._handle_email,
            NotificationChannel.SMS: self._handle_sms,
            NotificationChannel.WEBHOOK: self._handle_webhook,
            NotificationChannel.TELEGRAM: self._handle_telegram,
            NotificationChannel.SLACK: self._handle_slack,
        }

    def _load_templates(self) -> None:
        """加载通知模板"""
        # 加载默认模板
        self.templates["alert_critical"] = NotificationTemplate(
            template_id="alert_critical",
            name="Critical Alert",
            type=NotificationType.ALERT,
            subject_template="[CRITICAL] {alert_title}",
            body_template="Critical alert triggered at {timestamp}:\n\n{alert_message}\n\nAffected: {affected_items}\nRecommended Action: {recommended_action}",
            variables=[
                "alert_title",
                "alert_message",
                "affected_items",
                "recommended_action",
                "timestamp",
            ],
        )

        self.templates["daily_report"] = NotificationTemplate(
            template_id="daily_report",
            name="Daily Report",
            type=NotificationType.REPORT,
            subject_template="Daily Report - {date}",
            body_template="Daily Performance Summary for {date}:\n\nTotal P&L: {total_pnl}\nReturn: {daily_return}%\nPositions: {num_positions}\n\nTop Gainers:\n{top_gainers}\n\nTop Losers:\n{top_losers}",
            variables=[
                "date",
                "total_pnl",
                "daily_return",
                "num_positions",
                "top_gainers",
                "top_losers",
            ],
        )

        self.templates["trade_execution"] = NotificationTemplate(
            template_id="trade_execution",
            name="Trade Execution",
            type=NotificationType.TRADE,
            subject_template="Trade Executed: {symbol} {action}",
            body_template="Trade executed successfully:\n\nSymbol: {symbol}\nAction: {action}\nQuantity: {quantity}\nPrice: {price}\nTotal Value: {total_value}\n\nOrder ID: {order_id}\nExecuted at: {timestamp}",
            variables=[
                "symbol",
                "action",
                "quantity",
                "price",
                "total_value",
                "order_id",
                "timestamp",
            ],
        )

    def _queue_notification(self, notification: Notification) -> None:
        """将通知加入队列

        Args:
            notification: 通知对象
        """
        # 检查速率限制
        if not self._check_rate_limit(notification.channel):
            logger.warning(
                f"Rate limit exceeded for channel {notification.channel.value}"
            )
            notification.metadata["rate_limited"] = True

        # 加入优先级队列
        priority = -notification.priority.value  # 负数使高优先级先处理
        self.notification_queue.put((priority, notification.timestamp, notification))
        self.pending_notifications[notification.notification_id] = notification

    def _add_to_aggregation_buffer(self, notification: Notification) -> None:
        """添加到聚合缓冲区

        Args:
            notification: 通知对象
        """
        key = f"{notification.type.value}_{notification.channel.value}_{notification.recipient}"
        self.aggregation_buffer[key].append(notification)

    async def _process_queue(self) -> None:
        """处理通知队列"""
        while self.processing_active:
            try:
                if not self.notification_queue.empty():
                    _, _, notification = self.notification_queue.get_nowait()

                    # 发送通知
                    success = await self._send_notification(notification)

                    if success:
                        notification.delivered = True
                        notification.sent_at = datetime.now()
                        self.sent_notifications.append(notification)
                        del self.pending_notifications[notification.notification_id]
                    else:
                        # 重试逻辑
                        notification.retry_count += 1
                        if notification.retry_count < self.config.retry_attempts:
                            await asyncio.sleep(self.config.retry_delay)
                            self._queue_notification(notification)
                        else:
                            notification.delivered = False
                            self.sent_notifications.append(notification)
                            del self.pending_notifications[notification.notification_id]

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error processing notification queue: {e}")
                await asyncio.sleep(1)

    async def _process_aggregation_buffer(self) -> None:
        """处理聚合缓冲区"""
        while self.processing_active:
            try:
                current_time = datetime.now()

                for key, notifications in list(self.aggregation_buffer.items()):
                    if not notifications:
                        continue

                    # 检查是否达到聚合窗口
                    oldest = min(n.timestamp for n in notifications)
                    if (
                        current_time - oldest
                    ).total_seconds() >= self.config.aggregation_window:
                        # 创建聚合通知
                        aggregated = self._create_aggregated_notification(notifications)
                        self._queue_notification(aggregated)

                        # 清空缓冲区
                        self.aggregation_buffer[key] = []

                await asyncio.sleep(self.config.batch_interval)

            except Exception as e:
                logger.error(f"Error processing aggregation buffer: {e}")
                await asyncio.sleep(5)

    async def _retry_failed_notifications(self) -> None:
        """重试失败的通知"""
        while self.processing_active:
            try:
                # 查找需要重试的通知
                for notification_id, notification in list(
                    self.pending_notifications.items()
                ):
                    if (
                        notification.retry_count > 0
                        and notification.retry_count < self.config.retry_attempts
                    ):
                        time_since_last_try = (
                            datetime.now() - notification.timestamp
                        ).total_seconds()

                        if (
                            time_since_last_try
                            >= self.config.retry_delay * notification.retry_count
                        ):
                            success = await self._send_notification(notification)

                            if success:
                                notification.delivered = True
                                notification.sent_at = datetime.now()
                                self.sent_notifications.append(notification)
                                del self.pending_notifications[notification_id]
                            else:
                                notification.retry_count += 1

                await asyncio.sleep(30)  # 检查间隔

            except Exception as e:
                logger.error(f"Error in retry logic: {e}")
                await asyncio.sleep(60)

    async def _send_notification(self, notification: Notification) -> bool:
        """发送单个通知

        Args:
            notification: 通知对象

        Returns:
            是否成功
        """
        try:
            handler = self.channel_handlers.get(notification.channel)

            if not handler:
                logger.error(f"No handler for channel {notification.channel.value}")
                notification.error_message = (
                    f"No handler for channel {notification.channel.value}"
                )
                return False

            # 调用处理器
            if asyncio.iscoroutinefunction(handler):
                success = await handler(notification)
            else:
                success = handler(notification)

            # 更新速率限制
            if success:
                self._update_rate_limit(notification.channel)

            return success

        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            notification.error_message = str(e)
            return False

    def _create_aggregated_notification(
        self, notifications: List[Notification]
    ) -> Notification:
        """创建聚合通知

        Args:
            notifications: 通知列表

        Returns:
            聚合后的通知
        """
        if not notifications:
            raise ValueError("No notifications to aggregate")

        # 使用第一个通知作为基础
        first = notifications[0]

        # 构建聚合消息
        subject = f"[Aggregated] {len(notifications)} {first.type.value} notifications"

        messages = []
        for n in notifications:
            messages.append(f"- {n.subject}: {n.message}")

        message = f"Aggregated {len(notifications)} notifications:\n\n" + "\n".join(
            messages
        )

        # 合并数据
        aggregated_data = {}
        for n in notifications:
            if n.data:
                aggregated_data.update(n.data)

        return Notification(
            notification_id=f"aggregated_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            type=first.type,
            priority=max(n.priority for n in notifications),
            channel=first.channel,
            recipient=first.recipient,
            subject=subject,
            message=message,
            data=aggregated_data,
            metadata={"aggregated_count": len(notifications)},
        )

    def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """检查速率限制

        Args:
            channel: 通知渠道

        Returns:
            是否允许发送
        """
        if channel not in self.config.rate_limits:
            return True

        limit = self.config.rate_limits[channel]
        current_time = datetime.now()

        # 清理过期的时间戳
        cutoff_time = current_time - timedelta(minutes=1)
        self.rate_limiters[channel] = [
            t for t in self.rate_limiters[channel] if t > cutoff_time
        ]

        # 检查是否超过限制
        if len(self.rate_limiters[channel]) >= limit:
            return False

        return True

    def _update_rate_limit(self, channel: NotificationChannel) -> None:
        """更新速率限制记录

        Args:
            channel: 通知渠道
        """
        self.rate_limiters[channel].append(datetime.now())

    def _is_quiet_hours(self) -> bool:
        """检查是否在安静时间

        Returns:
            是否在安静时间
        """
        if not self.config.quiet_hours:
            return False

        current_hour = datetime.now().hour
        start_hour, end_hour = self.config.quiet_hours

        if start_hour <= end_hour:
            return start_hour <= current_hour < end_hour
        else:  # 跨午夜
            return current_hour >= start_hour or current_hour < end_hour

    def _schedule_after_quiet_hours(
        self,
        type: NotificationType,
        priority: NotificationPriority,
        channel: NotificationChannel,
        recipient: str,
        subject: str,
        message: str,
        data: Optional[Dict[str, Any]],
        attachments: Optional[List[str]],
    ) -> str:
        """安排在安静时间后发送

        Args:
            各通知参数

        Returns:
            通知ID
        """
        # 计算安静时间结束
        _, end_hour = self.config.quiet_hours
        now = datetime.now()

        if now.hour < end_hour:
            scheduled_time = now.replace(hour=end_hour, minute=0, second=0)
        else:
            # 第二天
            scheduled_time = (now + timedelta(days=1)).replace(
                hour=end_hour, minute=0, second=0
            )

        notification = Notification(
            notification_id=f"{type.value}_{datetime.now().timestamp()}",
            timestamp=scheduled_time,  # 设置为计划时间
            type=type,
            priority=priority,
            channel=channel,
            recipient=recipient,
            subject=subject,
            message=message,
            data=data or {},
            attachments=attachments or [],
            metadata={"scheduled": True, "original_time": now},
        )

        self._queue_notification(notification)

        return notification.notification_id

    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """渲染模板

        Args:
            template: 模板字符串
            variables: 变量字典

        Returns:
            渲染后的字符串
        """
        # 简单的字符串格式化
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return template

    # 渠道处理器（应该被实际实现替换）
    def _handle_email(self, notification: Notification) -> bool:
        """处理邮件通知"""
        logger.info(
            f"Sending email to {notification.recipient}: {notification.subject}"
        )
        # 实际实现应该调用邮件服务
        return True

    def _handle_sms(self, notification: Notification) -> bool:
        """处理短信通知"""
        logger.info(
            f"Sending SMS to {notification.recipient}: {notification.message[:50]}"
        )
        # 实际实现应该调用短信服务
        return True

    def _handle_webhook(self, notification: Notification) -> bool:
        """处理Webhook通知"""
        logger.info(f"Sending webhook to {notification.recipient}")
        # 实际实现应该发送HTTP请求
        return True

    def _handle_telegram(self, notification: Notification) -> bool:
        """处理Telegram通知"""
        logger.info(f"Sending Telegram message to {notification.recipient}")
        # 实际实现应该调用Telegram Bot API
        return True

    def _handle_slack(self, notification: Notification) -> bool:
        """处理Slack通知"""
        logger.info(f"Sending Slack message to {notification.recipient}")
        # 实际实现应该调用Slack API
        return True


# 模块级别函数
def create_notification_manager(
    config: Optional[NotificationConfig] = None,
) -> NotificationManager:
    """创建通知管理器的便捷函数

    Args:
        config: 通知配置

    Returns:
        通知管理器实例
    """
    return NotificationManager(config)
