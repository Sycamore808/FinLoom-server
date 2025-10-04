"""
Module 06 监控告警模块测试
测试系统监控、性能追踪、告警管理、通知服务和报告生成功能
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.logging_system import setup_logger
from module_06_monitoring_alerting import (
    AlertCategory,
    AlertConfig,
    # 告警系统
    AlertManager,
    AlertSeverity,
    MarketMonitor,
    MonitoringConfig,
    NotificationChannel,
    NotificationConfig,
    # 通知服务
    NotificationManager,
    NotificationPriority,
    NotificationType,
    # 实时监控
    PerformanceMonitor,
    PortfolioMonitor,
    ReportConfig,
    ReportFormat,
    # 报告引擎
    ReportGenerator,
    ReportType,
    # 数据库管理
    get_monitoring_database_manager,
)

# 从alert_system导入AlertRule（用于alert_manager）
from module_06_monitoring_alerting.alert_system import (
    AlertRule as AlertManagerRule,
)

# 从performance_monitor导入AlertRule（用于performance_monitor）
from module_06_monitoring_alerting.real_time_monitoring.performance_monitor import (
    AlertRule as PerformanceMonitorRule,
)
from module_06_monitoring_alerting.real_time_monitoring.performance_tracker import (
    get_performance_tracker,
)
from module_06_monitoring_alerting.real_time_monitoring.system_monitor import (
    get_system_monitor,
)

logger = setup_logger("module06_test")


def test_system_monitor():
    """测试系统监控器"""
    logger.info("=" * 60)
    logger.info("测试系统监控器")
    logger.info("=" * 60)

    try:
        # 获取系统监控器
        system_monitor = get_system_monitor(monitoring_interval=5)

        # 获取系统状态
        status = system_monitor.get_system_status()

        logger.info(f"✓ 系统状态:")
        logger.info(f"  CPU使用率: {status.cpu_percent:.1f}%")
        logger.info(f"  内存使用率: {status.memory_percent:.1f}%")
        logger.info(f"  磁盘使用率: {status.disk_percent:.1f}%")
        logger.info(f"  CPU核心数: {status.cpu_count}")
        logger.info(f"  运行时长: {status.uptime_hours:.2f}小时")
        logger.info(f"  进程数: {status.process_count}")
        logger.info(f"  Python版本: {status.python_version}")
        logger.info(f"  操作系统: {status.os_info}")

        # 获取健康状态
        health = system_monitor.get_health_status()
        logger.info(f"✓ 系统健康状态: {health['overall_status']}")

        for component, info in health["components"].items():
            logger.info(f"  {component}: {info['status']} - {info['message']}")

        logger.info("✓ 系统监控器测试通过\n")
        return True

    except Exception as e:
        logger.error(f"✗ 系统监控器测试失败: {e}")
        return False


def test_performance_monitor():
    """测试性能监控器"""
    logger.info("=" * 60)
    logger.info("测试性能监控器")
    logger.info("=" * 60)

    try:
        # 获取性能监控器
        perf_monitor = PerformanceMonitor(monitoring_interval=5)

        # 添加告警规则
        rule = PerformanceMonitorRule(
            name="cpu_high",
            metric_type="system",
            metric_name="cpu_percent",
            operator=">",
            threshold=80.0,
            duration_seconds=60,
        )
        perf_monitor.add_alert_rule(rule)

        # 获取最新指标
        system_metrics = perf_monitor.get_latest_system_metrics()

        if system_metrics:
            logger.info(f"✓ 系统指标:")
            logger.info(f"  时间戳: {system_metrics.timestamp}")
            logger.info(f"  CPU使用率: {system_metrics.cpu_percent:.1f}%")
            logger.info(f"  内存使用率: {system_metrics.memory_percent:.1f}%")
            logger.info(f"  Python内存: {system_metrics.python_memory_mb:.1f}MB")
            logger.info(f"  活跃线程: {system_metrics.active_threads}")

        # 获取摘要
        summary = perf_monitor.get_metrics_summary(minutes=10)
        logger.info(f"✓ 性能摘要:")
        logger.info(f"  系统指标数: {summary.get('system_metrics_count', 0)}")
        logger.info(f"  活跃告警数: {summary.get('active_alerts', 0)}")
        logger.info(f"  总告警数: {summary.get('total_alerts', 0)}")

        logger.info("✓ 性能监控器测试通过\n")
        return True

    except Exception as e:
        logger.error(f"✗ 性能监控器测试失败: {e}")
        return False


def test_performance_tracker():
    """测试性能追踪器"""
    logger.info("=" * 60)
    logger.info("测试性能追踪器")
    logger.info("=" * 60)

    try:
        # 获取性能追踪器
        tracker = get_performance_tracker()

        # 使用上下文管理器追踪操作
        with tracker.track("test_operation", metadata={"test": "value"}):
            import time

            time.sleep(0.1)  # 模拟操作

        # 手动记录操作
        tracker.record_operation("manual_operation", duration=0.05, success=True)
        tracker.record_operation(
            "failed_operation", duration=0.1, success=False, error_message="测试错误"
        )

        # 获取统计
        stats = tracker.get_stats()
        logger.info(f"✓ 性能统计:")
        for op_name, op_stats in stats.items():
            logger.info(f"  {op_name}:")
            logger.info(f"    执行次数: {op_stats.count}")
            logger.info(f"    平均耗时: {op_stats.avg_duration:.3f}秒")
            logger.info(f"    成功率: {op_stats.success_rate:.1%}")

        # 获取慢操作
        slow_ops = tracker.get_slow_operations(threshold=0.05, limit=5)
        logger.info(f"✓ 慢操作数: {len(slow_ops)}")

        # 获取失败操作
        failed_ops = tracker.get_failed_operations(limit=5)
        logger.info(f"✓ 失败操作数: {len(failed_ops)}")

        # 获取摘要
        summary = tracker.get_summary(minutes=10)
        logger.info(f"✓ 性能摘要:")
        logger.info(f"  总操作数: {summary['total_operations']}")
        logger.info(f"  成功率: {summary['success_rate']:.1%}")
        logger.info(f"  平均耗时: {summary['avg_duration']:.3f}秒")

        logger.info("✓ 性能追踪器测试通过\n")
        return True

    except Exception as e:
        logger.error(f"✗ 性能追踪器测试失败: {e}")
        return False


def test_alert_manager():
    """测试告警管理器"""
    logger.info("=" * 60)
    logger.info("测试告警管理器")
    logger.info("=" * 60)

    try:
        # 创建告警管理器
        alert_config = AlertConfig(
            max_alerts_per_rule=10,
            enable_auto_escalation=True,
            enable_alert_suppression=True,
        )
        alert_manager = AlertManager(alert_config)

        # 添加告警规则
        rule = AlertManagerRule(
            rule_id="cpu_high_rule",
            name="CPU使用率过高",
            description="CPU使用率超过阈值",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            condition="cpu_percent > 80",
            threshold=80.0,
            comparison=">",
            metric="cpu_percent",
            cooldown_seconds=300,
        )
        alert_manager.add_rule(rule)

        # 检查规则
        metrics = {"cpu_percent": 85.0, "memory_percent": 70.0}
        triggered_alerts = alert_manager.check_rules(metrics)

        logger.info(f"✓ 触发的告警数: {len(triggered_alerts)}")
        for alert in triggered_alerts:
            logger.info(f"  告警ID: {alert.alert_id}")
            logger.info(f"  严重级别: {alert.severity.name}")
            logger.info(f"  消息: {alert.message}")

        # 获取活跃告警
        active_alerts = alert_manager.get_active_alerts()
        logger.info(f"✓ 活跃告警数: {len(active_alerts)}")

        # 获取统计
        stats = alert_manager.get_alert_statistics()
        logger.info(f"✓ 告警统计:")
        logger.info(f"  总告警数: {stats.total_alerts}")
        logger.info(f"  升级率: {stats.escalation_rate:.1%}")

        logger.info("✓ 告警管理器测试通过\n")
        return True

    except Exception as e:
        logger.error(f"✗ 告警管理器测试失败: {e}")
        return False


def test_notification_manager():
    """测试通知管理器"""
    logger.info("=" * 60)
    logger.info("测试通知管理器")
    logger.info("=" * 60)

    try:
        # 创建通知管理器
        notification_config = NotificationConfig(
            enabled_channels=[NotificationChannel.EMAIL],
            enable_aggregation=True,
        )
        notification_manager = NotificationManager(notification_config)

        # 发送通知
        notification_id = notification_manager.send_notification(
            type=NotificationType.ALERT,
            priority=NotificationPriority.HIGH,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com",
            subject="测试告警",
            message="这是一条测试告警消息",
            data={"test": "data"},
        )

        logger.info(f"✓ 发送通知: {notification_id}")

        # 获取通知状态
        status = notification_manager.get_notification_status(notification_id)
        if status:
            logger.info(f"✓ 通知状态: {status['status']}")

        # 获取统计
        stats = notification_manager.get_statistics()
        logger.info(f"✓ 通知统计:")
        logger.info(f"  总发送数: {stats.total_sent}")
        logger.info(f"  总送达数: {stats.total_delivered}")
        logger.info(f"  成功率: {stats.success_rate:.1%}")

        logger.info("✓ 通知管理器测试通过\n")
        return True

    except Exception as e:
        logger.error(f"✗ 通知管理器测试失败: {e}")
        return False


def test_report_generator():
    """测试报告生成器"""
    logger.info("=" * 60)
    logger.info("测试报告生成器")
    logger.info("=" * 60)

    try:
        # 创建报告生成器
        report_generator = ReportGenerator(
            template_dir="templates", output_dir="reports"
        )

        # 准备测试数据
        test_data = {
            "portfolio_value": 1000000,
            "cash_balance": 100000,
            "positions_value": 900000,
            "daily_return": 0.015,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.08,
            "positions": [
                {
                    "symbol": "000001",
                    "quantity": 10000,
                    "avg_cost": 15.5,
                    "current_price": 16.0,
                    "market_value": 160000,
                    "unrealized_pnl": 5000,
                    "return_pct": 0.032,
                    "weight": 0.18,
                },
            ],
            "transactions": [],
        }

        # 生成JSON报告
        report_config = ReportConfig(
            report_type=ReportType.DAILY,
            format=ReportFormat.JSON,
            include_charts=False,
        )

        report_path = report_generator.generate_report(report_config, test_data)
        logger.info(f"✓ 生成报告: {report_path}")

        # 验证报告文件存在
        if Path(report_path).exists():
            logger.info(f"✓ 报告文件已生成")
        else:
            logger.warning(f"⚠ 报告文件未找到")

        logger.info("✓ 报告生成器测试通过\n")
        return True

    except Exception as e:
        logger.error(f"✗ 报告生成器测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_database_manager():
    """测试数据库管理器"""
    logger.info("=" * 60)
    logger.info("测试数据库管理器")
    logger.info("=" * 60)

    try:
        # 获取数据库管理器
        db_manager = get_monitoring_database_manager("data/test_module06_monitoring.db")

        # 保存系统健康状态
        success = db_manager.save_health_status(
            timestamp=datetime.now(),
            cpu_usage=45.2,
            memory_usage=62.8,
            disk_usage=35.1,
        )
        logger.info(f"✓ 保存健康状态: {success}")

        # 保存性能指标
        success = db_manager.save_performance_metrics(
            timestamp=datetime.now(),
            operation="test_operation",
            duration=0.123,
            success=True,
        )
        logger.info(f"✓ 保存性能指标: {success}")

        # 保存告警
        success = db_manager.save_alert(
            alert_id="test_alert_001",
            rule_id="test_rule",
            timestamp=datetime.now(),
            severity="high",
            category="system",
            title="测试告警",
            message="这是一条测试告警",
            metric_value=85.0,
            threshold_value=80.0,
        )
        logger.info(f"✓ 保存告警: {success}")

        # 查询健康历史
        health_df = db_manager.get_health_history(limit=10)
        logger.info(f"✓ 查询健康历史: {len(health_df)}条记录")

        # 查询性能历史
        perf_df = db_manager.get_performance_history(limit=10)
        logger.info(f"✓ 查询性能历史: {len(perf_df)}条记录")

        # 查询告警
        alerts = db_manager.get_alerts(limit=10)
        logger.info(f"✓ 查询告警: {len(alerts)}条记录")

        # 获取统计
        alert_stats = db_manager.get_alert_statistics()
        logger.info(f"✓ 告警统计: {alert_stats}")

        perf_summary = db_manager.get_performance_summary()
        logger.info(f"✓ 性能摘要: {len(perf_summary)}个操作")

        logger.info("✓ 数据库管理器测试通过\n")
        return True

    except Exception as e:
        logger.error(f"✗ 数据库管理器测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration():
    """集成测试"""
    logger.info("=" * 60)
    logger.info("集成测试 - 完整工作流")
    logger.info("=" * 60)

    try:
        # 1. 系统监控
        logger.info("步骤 1: 系统监控")
        system_monitor = get_system_monitor()
        system_status = system_monitor.get_system_status()
        logger.info(f"  系统状态已获取")

        # 2. 性能追踪
        logger.info("步骤 2: 性能追踪")
        tracker = get_performance_tracker()
        with tracker.track("integration_test"):
            import time

            time.sleep(0.05)
        logger.info(f"  性能已追踪")

        # 3. 告警检测
        logger.info("步骤 3: 告警检测")
        alert_manager = AlertManager()
        rule = AlertManagerRule(
            rule_id="integration_rule",
            name="集成测试规则",
            description="测试规则",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.INFO,
            condition="test_metric > 0",
            threshold=0.0,
            comparison=">",
            metric="test_metric",
        )
        alert_manager.add_rule(rule)
        alerts = alert_manager.check_rules({"test_metric": 1.0})
        logger.info(f"  检测到 {len(alerts)} 个告警")

        # 4. 通知发送
        logger.info("步骤 4: 通知发送")
        notifier = NotificationManager()
        notif_id = notifier.send_notification(
            type=NotificationType.SYSTEM,
            priority=NotificationPriority.NORMAL,
            channel=NotificationChannel.EMAIL,
            recipient="test@example.com",
            subject="集成测试",
            message="集成测试消息",
        )
        logger.info(f"  通知已发送: {notif_id}")

        # 5. 数据保存
        logger.info("步骤 5: 数据保存")
        db_manager = get_monitoring_database_manager("data/test_module06_monitoring.db")

        db_manager.save_health_status(
            timestamp=datetime.now(),
            cpu_usage=system_status.cpu_percent,
            memory_usage=system_status.memory_percent,
            disk_usage=system_status.disk_percent,
        )

        if alerts:
            db_manager.save_alert(
                alert_id=alerts[0].alert_id,
                rule_id=alerts[0].rule_id,
                timestamp=alerts[0].timestamp,
                severity=alerts[0].severity.name,
                category=alerts[0].category.name,
                title=alerts[0].title,
                message=alerts[0].message,
                metric_value=alerts[0].metric_value,
                threshold_value=alerts[0].threshold_value,
            )

        logger.info(f"  数据已保存到数据库")

        logger.info("✓ 集成测试通过\n")
        return True

    except Exception as e:
        logger.error(f"✗ 集成测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    logger.info("\n" + "=" * 60)
    logger.info("Module 06 监控告警模块测试开始")
    logger.info("=" * 60 + "\n")

    results = {}

    # 运行测试
    results["系统监控"] = test_system_monitor()
    results["性能监控"] = test_performance_monitor()
    results["性能追踪"] = test_performance_tracker()
    results["告警管理"] = test_alert_manager()
    results["通知管理"] = test_notification_manager()
    results["报告生成"] = test_report_generator()
    results["数据库管理"] = test_database_manager()
    results["集成测试"] = test_integration()

    # 输出测试总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        logger.info("✓ 所有测试通过！")
        return 0
    else:
        logger.error(f"✗ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
