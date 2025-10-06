"""
邮件通知器模块
通过SMTP发送邮件通知
"""

import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from common.logging_system import setup_logger

logger = setup_logger("email_notifier")


@dataclass
class EmailConfig:
    """邮件配置"""

    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    from_address: str
    use_tls: bool = True
    use_ssl: bool = False
    timeout: int = 30


@dataclass
class EmailMessage:
    """邮件消息"""

    to_addresses: List[str]
    subject: str
    body: str
    html_body: Optional[str] = None
    cc_addresses: Optional[List[str]] = None
    bcc_addresses: Optional[List[str]] = None
    attachments: Optional[List[Dict[str, Any]]] = None


class EmailNotifier:
    """邮件通知器类"""

    def __init__(self, config: EmailConfig):
        """初始化邮件通知器

        Args:
            config: 邮件配置
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """验证配置"""
        if not self.config.smtp_host:
            raise ValueError("SMTP host is required")
        if not self.config.smtp_user:
            raise ValueError("SMTP user is required")
        if not self.config.smtp_password:
            raise ValueError("SMTP password is required")
        if not self.config.from_address:
            raise ValueError("From address is required")

    def send_email(self, message: EmailMessage) -> bool:
        """发送邮件

        Args:
            message: 邮件消息

        Returns:
            是否发送成功
        """
        try:
            # 创建邮件消息
            msg = self._create_message(message)

            # 连接SMTP服务器并发送
            if self.config.use_ssl:
                with smtplib.SMTP_SSL(
                    self.config.smtp_host,
                    self.config.smtp_port,
                    timeout=self.config.timeout,
                ) as server:
                    self._send_via_server(server, msg, message.to_addresses)
            else:
                with smtplib.SMTP(
                    self.config.smtp_host,
                    self.config.smtp_port,
                    timeout=self.config.timeout,
                ) as server:
                    if self.config.use_tls:
                        server.starttls()
                    self._send_via_server(server, msg, message.to_addresses)

            logger.info(f"Email sent successfully to {', '.join(message.to_addresses)}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_alert_email(
        self,
        to_addresses: List[str],
        alert_type: str,
        alert_message: str,
        alert_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """发送告警邮件

        Args:
            to_addresses: 收件人地址列表
            alert_type: 告警类型
            alert_message: 告警消息
            alert_data: 告警数据

        Returns:
            是否发送成功
        """
        subject = f"[FinLoom Alert] {alert_type}"

        # 生成邮件正文
        body = self._generate_alert_body(alert_type, alert_message, alert_data)

        # 生成HTML正文
        html_body = self._generate_alert_html(alert_type, alert_message, alert_data)

        message = EmailMessage(
            to_addresses=to_addresses, subject=subject, body=body, html_body=html_body
        )

        return self.send_email(message)

    def send_report_email(
        self,
        to_addresses: List[str],
        report_type: str,
        report_path: str,
        report_summary: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """发送报告邮件

        Args:
            to_addresses: 收件人地址列表
            report_type: 报告类型
            report_path: 报告文件路径
            report_summary: 报告摘要

        Returns:
            是否发送成功
        """
        subject = f"[FinLoom Report] {report_type.upper()} Report"

        # 生成邮件正文
        body = self._generate_report_body(report_type, report_path, report_summary)

        # 生成HTML正文
        html_body = self._generate_report_html(report_type, report_path, report_summary)

        message = EmailMessage(
            to_addresses=to_addresses, subject=subject, body=body, html_body=html_body
        )

        return self.send_email(message)

    def _create_message(self, email_message: EmailMessage) -> MIMEMultipart:
        """创建邮件消息对象

        Args:
            email_message: 邮件消息

        Returns:
            MIME消息对象
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = email_message.subject
        msg["From"] = self.config.from_address
        msg["To"] = ", ".join(email_message.to_addresses)

        if email_message.cc_addresses:
            msg["Cc"] = ", ".join(email_message.cc_addresses)

        # 添加纯文本部分
        text_part = MIMEText(email_message.body, "plain", "utf-8")
        msg.attach(text_part)

        # 添加HTML部分
        if email_message.html_body:
            html_part = MIMEText(email_message.html_body, "html", "utf-8")
            msg.attach(html_part)

        return msg

    def _send_via_server(
        self, server: smtplib.SMTP, msg: MIMEMultipart, to_addresses: List[str]
    ) -> None:
        """通过SMTP服务器发送邮件

        Args:
            server: SMTP服务器对象
            msg: 邮件消息
            to_addresses: 收件人地址列表
        """
        server.login(self.config.smtp_user, self.config.smtp_password)
        server.send_message(msg)

    def _generate_alert_body(
        self, alert_type: str, alert_message: str, alert_data: Optional[Dict[str, Any]]
    ) -> str:
        """生成告警邮件正文

        Args:
            alert_type: 告警类型
            alert_message: 告警消息
            alert_data: 告警数据

        Returns:
            邮件正文
        """
        lines = [
            f"FinLoom Alert Notification",
            f"",
            f"Alert Type: {alert_type}",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"Message:",
            f"{alert_message}",
        ]

        if alert_data:
            lines.append("")
            lines.append("Alert Data:")
            for key, value in alert_data.items():
                lines.append(f"  {key}: {value}")

        lines.extend(
            [
                "",
                "---",
                "This is an automated message from FinLoom Monitoring System.",
                "Please do not reply to this email.",
            ]
        )

        return "\n".join(lines)

    def _generate_alert_html(
        self, alert_type: str, alert_message: str, alert_data: Optional[Dict[str, Any]]
    ) -> str:
        """生成告警邮件HTML正文

        Args:
            alert_type: 告警类型
            alert_message: 告警消息
            alert_data: 告警数据

        Returns:
            HTML正文
        """
        # 根据告警类型设置颜色
        color_map = {
            "high": "#dc3545",
            "medium": "#ffc107",
            "low": "#17a2b8",
            "critical": "#721c24",
            "warning": "#856404",
            "info": "#004085",
        }

        severity = "medium"
        for level in color_map.keys():
            if level in alert_type.lower():
                severity = level
                break

        color = color_map.get(severity, "#17a2b8")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 30px; border-radius: 8px; max-width: 600px; margin: 0 auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .header {{ background-color: {color}; color: white; padding: 20px; border-radius: 8px 8px 0 0; margin: -30px -30px 20px -30px; }}
                .alert-type {{ font-size: 24px; font-weight: bold; }}
                .timestamp {{ font-size: 14px; opacity: 0.9; margin-top: 5px; }}
                .message {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid {color}; margin: 20px 0; }}
                .data-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .data-table th, .data-table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                .data-table th {{ background-color: #f8f9fa; font-weight: bold; }}
                .footer {{ text-align: center; color: #6c757d; font-size: 12px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="alert-type">🚨 {alert_type}</div>
                    <div class="timestamp">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                </div>
                
                <div class="message">
                    <strong>Alert Message:</strong><br/>
                    {alert_message}
                </div>
        """

        if alert_data:
            html += """
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for key, value in alert_data.items():
                html += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
                """

            html += """
                    </tbody>
                </table>
            """

        html += """
                <div class="footer">
                    This is an automated message from FinLoom Monitoring System.<br/>
                    Please do not reply to this email.
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _generate_report_body(
        self,
        report_type: str,
        report_path: str,
        report_summary: Optional[Dict[str, Any]],
    ) -> str:
        """生成报告邮件正文

        Args:
            report_type: 报告类型
            report_path: 报告文件路径
            report_summary: 报告摘要

        Returns:
            邮件正文
        """
        lines = [
            f"FinLoom Report",
            f"",
            f"Report Type: {report_type.upper()}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Report Location: {report_path}",
            f"",
        ]

        if report_summary:
            lines.append("Report Summary:")
            for key, value in report_summary.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        lines.extend(
            [
                "Please check the attached report or access it at the specified location.",
                "",
                "---",
                "This is an automated message from FinLoom Reporting System.",
            ]
        )

        return "\n".join(lines)

    def _generate_report_html(
        self,
        report_type: str,
        report_path: str,
        report_summary: Optional[Dict[str, Any]],
    ) -> str:
        """生成报告邮件HTML正文

        Args:
            report_type: 报告类型
            report_path: 报告文件路径
            report_summary: 报告摘要

        Returns:
            HTML正文
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 30px; border-radius: 8px; max-width: 600px; margin: 0 auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .header {{ background-color: #007bff; color: white; padding: 20px; border-radius: 8px 8px 0 0; margin: -30px -30px 20px -30px; }}
                .title {{ font-size: 24px; font-weight: bold; }}
                .info {{ margin: 20px 0; }}
                .info-item {{ padding: 10px; border-bottom: 1px solid #dee2e6; }}
                .info-label {{ font-weight: bold; color: #495057; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 4px; margin: 20px 0; }}
                .footer {{ text-align: center; color: #6c757d; font-size: 12px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="title">📊 {report_type.upper()} Report</div>
                </div>
                
                <div class="info">
                    <div class="info-item">
                        <span class="info-label">Generated:</span> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </div>
                    <div class="info-item">
                        <span class="info-label">Location:</span> {report_path}
                    </div>
                </div>
        """

        if report_summary:
            html += """
                <div class="summary">
                    <h3 style="margin-top: 0;">Report Summary</h3>
            """

            for key, value in report_summary.items():
                html += f"""
                    <div style="padding: 5px 0;">
                        <strong>{key}:</strong> {value}
                    </div>
                """

            html += """
                </div>
            """

        html += """
                <div class="footer">
                    This is an automated message from FinLoom Reporting System.
                </div>
            </div>
        </body>
        </html>
        """

        return html


def create_email_notifier(config: EmailConfig) -> EmailNotifier:
    """创建邮件通知器的便捷函数

    Args:
        config: 邮件配置

    Returns:
        邮件通知器实例
    """
    return EmailNotifier(config)
