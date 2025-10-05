"""
通知服务子模块
提供多渠道通知管理功能
"""

from .email_notifier import (
    EmailConfig,
    EmailMessage,
    EmailNotifier,
    create_email_notifier,
)
from .notification_manager import (
    Notification,
    NotificationChannel,
    NotificationConfig,
    NotificationManager,
    NotificationPriority,
    NotificationStats,
    NotificationTemplate,
    NotificationType,
    create_notification_manager,
)
from .webhook_notifier import (
    WebhookConfig,
    WebhookFormat,
    WebhookMessage,
    WebhookMethod,
    WebhookNotifier,
    create_discord_webhook,
    create_slack_webhook,
    create_teams_webhook,
    create_webhook_notifier,
)

__all__ = [
    # 通知管理器
    "NotificationManager",
    "Notification",
    "NotificationConfig",
    "NotificationStats",
    "NotificationChannel",
    "NotificationPriority",
    "NotificationType",
    "NotificationTemplate",
    "create_notification_manager",
    # 邮件通知器
    "EmailNotifier",
    "EmailConfig",
    "EmailMessage",
    "create_email_notifier",
    # Webhook通知器
    "WebhookNotifier",
    "WebhookConfig",
    "WebhookMessage",
    "WebhookMethod",
    "WebhookFormat",
    "create_webhook_notifier",
    "create_slack_webhook",
    "create_discord_webhook",
    "create_teams_webhook",
]
