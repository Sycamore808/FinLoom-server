"""
Webhook通知器模块
通过HTTP Webhook发送通知
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

from common.logging_system import setup_logger

logger = setup_logger("webhook_notifier")


class WebhookMethod(Enum):
    """HTTP方法枚举"""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"


class WebhookFormat(Enum):
    """Webhook数据格式枚举"""

    JSON = "json"
    FORM = "form"
    XML = "xml"


@dataclass
class WebhookConfig:
    """Webhook配置"""

    url: str
    method: WebhookMethod = WebhookMethod.POST
    format: WebhookFormat = WebhookFormat.JSON
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 5
    verify_ssl: bool = True
    auth_token: Optional[str] = None


@dataclass
class WebhookMessage:
    """Webhook消息"""

    event_type: str
    payload: Dict[str, Any]
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class WebhookNotifier:
    """Webhook通知器类"""

    def __init__(self, config: WebhookConfig):
        """初始化Webhook通知器

        Args:
            config: Webhook配置
        """
        self.config = config
        self._validate_config()
        self._prepare_headers()

    def _validate_config(self) -> None:
        """验证配置"""
        if not self.config.url:
            raise ValueError("Webhook URL is required")
        if not self.config.url.startswith(("http://", "https://")):
            raise ValueError("Invalid webhook URL format")

    def _prepare_headers(self) -> None:
        """准备HTTP头"""
        # 设置默认头
        if self.config.format == WebhookFormat.JSON:
            self.config.headers.setdefault("Content-Type", "application/json")
        elif self.config.format == WebhookFormat.FORM:
            self.config.headers.setdefault(
                "Content-Type", "application/x-www-form-urlencoded"
            )
        elif self.config.format == WebhookFormat.XML:
            self.config.headers.setdefault("Content-Type", "application/xml")

        # 添加认证token
        if self.config.auth_token:
            self.config.headers.setdefault(
                "Authorization", f"Bearer {self.config.auth_token}"
            )

    def send_webhook(self, message: WebhookMessage) -> bool:
        """发送Webhook

        Args:
            message: Webhook消息

        Returns:
            是否发送成功
        """
        # 准备消息数据
        if message.timestamp is None:
            message.timestamp = datetime.now()

        payload = self._prepare_payload(message)

        # 尝试发送
        for attempt in range(self.config.retry_count):
            try:
                response = self._send_request(payload)

                if response.status_code in (200, 201, 202, 204):
                    logger.info(
                        f"Webhook sent successfully to {self.config.url} "
                        f"(status: {response.status_code})"
                    )
                    return True
                else:
                    logger.warning(
                        f"Webhook request failed with status {response.status_code}: "
                        f"{response.text}"
                    )

                    if attempt < self.config.retry_count - 1:
                        import time

                        time.sleep(self.config.retry_delay)
                        continue
                    else:
                        return False

            except requests.exceptions.Timeout:
                logger.error(
                    f"Webhook request timeout (attempt {attempt + 1}/{self.config.retry_count})"
                )
                if attempt < self.config.retry_count - 1:
                    import time

                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    return False

            except requests.exceptions.RequestException as e:
                logger.error(f"Webhook request failed: {e}")
                if attempt < self.config.retry_count - 1:
                    import time

                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    return False

        return False

    def send_alert_webhook(
        self,
        alert_type: str,
        alert_message: str,
        severity: str = "medium",
        alert_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """发送告警Webhook

        Args:
            alert_type: 告警类型
            alert_message: 告警消息
            severity: 严重程度
            alert_data: 告警数据

        Returns:
            是否发送成功
        """
        payload = {
            "alert_type": alert_type,
            "message": alert_message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
        }

        if alert_data:
            payload["data"] = alert_data

        message = WebhookMessage(
            event_type="alert",
            payload=payload,
            metadata={"source": "finloom_monitoring"},
        )

        return self.send_webhook(message)

    def send_report_webhook(
        self,
        report_type: str,
        report_path: str,
        report_summary: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """发送报告Webhook

        Args:
            report_type: 报告类型
            report_path: 报告路径
            report_summary: 报告摘要

        Returns:
            是否发送成功
        """
        payload = {
            "report_type": report_type,
            "report_path": report_path,
            "generated_at": datetime.now().isoformat(),
        }

        if report_summary:
            payload["summary"] = report_summary

        message = WebhookMessage(
            event_type="report",
            payload=payload,
            metadata={"source": "finloom_reporting"},
        )

        return self.send_webhook(message)

    def send_performance_webhook(
        self, metrics: Dict[str, Any], period: str = "daily"
    ) -> bool:
        """发送性能指标Webhook

        Args:
            metrics: 性能指标
            period: 统计周期

        Returns:
            是否发送成功
        """
        payload = {
            "period": period,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        message = WebhookMessage(
            event_type="performance_update",
            payload=payload,
            metadata={"source": "finloom_monitoring"},
        )

        return self.send_webhook(message)

    def send_system_status_webhook(self, status: Dict[str, Any]) -> bool:
        """发送系统状态Webhook

        Args:
            status: 系统状态

        Returns:
            是否发送成功
        """
        payload = {"status": status, "timestamp": datetime.now().isoformat()}

        message = WebhookMessage(
            event_type="system_status",
            payload=payload,
            metadata={"source": "finloom_monitoring"},
        )

        return self.send_webhook(message)

    def _prepare_payload(self, message: WebhookMessage) -> Dict[str, Any]:
        """准备发送的payload

        Args:
            message: Webhook消息

        Returns:
            准备好的payload
        """
        payload = {
            "event": message.event_type,
            "timestamp": (
                message.timestamp.isoformat()
                if message.timestamp
                else datetime.now().isoformat()
            ),
            "data": message.payload,
        }

        if message.metadata:
            payload["metadata"] = message.metadata

        return payload

    def _send_request(self, payload: Dict[str, Any]) -> requests.Response:
        """发送HTTP请求

        Args:
            payload: 请求数据

        Returns:
            HTTP响应

        Raises:
            requests.exceptions.RequestException: 请求失败
        """
        # 准备请求数据
        if self.config.format == WebhookFormat.JSON:
            data = json.dumps(payload)
        elif self.config.format == WebhookFormat.FORM:
            data = payload
        else:
            # XML格式需要转换
            data = self._dict_to_xml(payload)

        # 发送请求
        if self.config.method == WebhookMethod.GET:
            response = requests.get(
                self.config.url,
                params=payload,
                headers=self.config.headers,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
        elif self.config.method == WebhookMethod.POST:
            response = requests.post(
                self.config.url,
                data=data,
                headers=self.config.headers,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
        elif self.config.method == WebhookMethod.PUT:
            response = requests.put(
                self.config.url,
                data=data,
                headers=self.config.headers,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
        elif self.config.method == WebhookMethod.PATCH:
            response = requests.patch(
                self.config.url,
                data=data,
                headers=self.config.headers,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
        else:
            raise ValueError(f"Unsupported HTTP method: {self.config.method}")

        return response

    def _dict_to_xml(self, data: Dict[str, Any], root: str = "webhook") -> str:
        """将字典转换为XML

        Args:
            data: 数据字典
            root: 根元素名称

        Returns:
            XML字符串
        """
        xml_parts = [f"<?xml version='1.0' encoding='UTF-8'?>", f"<{root}>"]

        for key, value in data.items():
            if isinstance(value, dict):
                xml_parts.append(self._dict_to_xml(value, key))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        xml_parts.append(self._dict_to_xml(item, key))
                    else:
                        xml_parts.append(f"<{key}>{item}</{key}>")
            else:
                xml_parts.append(f"<{key}>{value}</{key}>")

        xml_parts.append(f"</{root}>")

        return "".join(xml_parts)

    def test_connection(self) -> bool:
        """测试Webhook连接

        Returns:
            连接是否正常
        """
        test_message = WebhookMessage(
            event_type="connection_test",
            payload={"message": "This is a test message from FinLoom"},
            timestamp=datetime.now(),
        )

        try:
            result = self.send_webhook(test_message)
            if result:
                logger.info("Webhook connection test successful")
            else:
                logger.warning("Webhook connection test failed")
            return result
        except Exception as e:
            logger.error(f"Webhook connection test error: {e}")
            return False


def create_webhook_notifier(config: WebhookConfig) -> WebhookNotifier:
    """创建Webhook通知器的便捷函数

    Args:
        config: Webhook配置

    Returns:
        Webhook通知器实例
    """
    return WebhookNotifier(config)


# 预定义的常用Webhook配置
def create_slack_webhook(webhook_url: str) -> WebhookNotifier:
    """创建Slack Webhook通知器

    Args:
        webhook_url: Slack Webhook URL

    Returns:
        配置好的Webhook通知器
    """
    config = WebhookConfig(
        url=webhook_url, method=WebhookMethod.POST, format=WebhookFormat.JSON
    )
    return WebhookNotifier(config)


def create_discord_webhook(webhook_url: str) -> WebhookNotifier:
    """创建Discord Webhook通知器

    Args:
        webhook_url: Discord Webhook URL

    Returns:
        配置好的Webhook通知器
    """
    config = WebhookConfig(
        url=webhook_url, method=WebhookMethod.POST, format=WebhookFormat.JSON
    )
    return WebhookNotifier(config)


def create_teams_webhook(webhook_url: str) -> WebhookNotifier:
    """创建Microsoft Teams Webhook通知器

    Args:
        webhook_url: Teams Webhook URL

    Returns:
        配置好的Webhook通知器
    """
    config = WebhookConfig(
        url=webhook_url, method=WebhookMethod.POST, format=WebhookFormat.JSON
    )
    return WebhookNotifier(config)
