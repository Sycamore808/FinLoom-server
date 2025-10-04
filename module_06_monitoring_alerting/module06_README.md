# Module 06 - 监控告警模块 API文档

## 概述

监控告警模块是 FinLoom 量化交易系统的实时监控中枢，负责系统健康监控、性能追踪、异常检测、告警管理和通知服务。该模块与所有其他模块集成，确保系统稳定运行和及时响应市场变化。

## 主要功能

### 1. 实时监控 (Real-time Monitoring)
- **系统监控 (SystemMonitor)**: 监控CPU、内存、磁盘、网络等系统资源
- **性能监控 (PerformanceMonitor)**: 追踪系统和交易指标
- **性能追踪 (PerformanceTracker)**: 记录操作性能和耗时
- **市场监控 (MarketMonitor)**: 实时市场数据和异常监控
- **投资组合监控 (PortfolioMonitor)**: 持仓和盈亏实时追踪

### 2. 告警系统 (Alert System)
- **告警管理器 (AlertManager)**: 统一告警管理和规则引擎
- **告警规则 (AlertRule)**: 自定义告警条件和阈值
- **告警状态管理**: 触发、确认、解决、升级

### 3. 通知服务 (Notification Service)
- **通知管理器 (NotificationManager)**: 多渠道通知管理
- **邮件通知器 (EmailNotifier)**: 通过SMTP发送邮件通知
- **Webhook通知器 (WebhookNotifier)**: 通过HTTP Webhook发送通知
- **通知模板**: 预定义的消息模板

### 4. 报告引擎 (Reporting Engine)
- **报告生成器 (ReportGenerator)**: 生成各类报告
- **报告格式**: HTML、Excel、JSON、Markdown
- **报告类型**: 日报、周报、月报、自定义报告

### 5. 数据持久化 (Data Persistence)
- **数据库管理器 (MonitoringDatabaseManager)**: 监控数据存储到SQLite
- **历史查询**: 查询历史监控数据和统计

## 快速开始

### 导入模块

```python
# 导入 Module 06 核心组件
from module_06_monitoring_alerting import (
    # 实时监控
    PerformanceMonitor,
    SystemMetrics,
    TradingMetrics,
    MarketMonitor,
    PortfolioMonitor,
    MonitoringConfig,
    # 告警系统
    AlertManager,
    AlertConfig,
    AlertSeverity,
    AlertCategory,
    AlertStatus,
    # 通知服务
    NotificationManager,
    NotificationConfig,
    NotificationChannel,
    NotificationPriority,
    NotificationType,
    # 报告引擎
    ReportGenerator,
    ReportType,
    ReportFormat,
    ReportConfig,
    # 数据库管理
    get_monitoring_database_manager,
)

# 导入系统监控器和性能追踪器
from module_06_monitoring_alerting.real_time_monitoring.system_monitor import get_system_monitor
from module_06_monitoring_alerting.real_time_monitoring.performance_tracker import get_performance_tracker

# 导入其他模块（用于集成）
from module_01_data_pipeline import AkshareDataCollector
from module_05_risk_management import PortfolioRiskAnalyzer
```

---

## API 参考

### 1. 系统监控 (System Monitor)

#### 基本用法

```python
from module_06_monitoring_alerting.real_time_monitoring.system_monitor import get_system_monitor

# 获取系统监控器实例（单例模式）
system_monitor = get_system_monitor(monitoring_interval=60)

# 获取系统状态
status = system_monitor.get_system_status()

print(f"CPU使用率: {status.cpu_percent:.1f}%")
print(f"内存使用率: {status.memory_percent:.1f}%")
print(f"磁盘使用率: {status.disk_percent:.1f}%")
print(f"CPU核心数: {status.cpu_count}")
print(f"运行时长: {status.uptime_hours:.2f}小时")

# 获取健康状态
health = system_monitor.get_health_status()
print(f"整体状态: {health['overall_status']}")  # 'healthy', 'warning', 'critical'

for component, info in health['components'].items():
    print(f"{component}: {info['status']} - {info['message']}")
```

#### 异步监控

```python
import asyncio

async def monitor_system():
    # 启动系统监控
    await system_monitor.start_monitoring()

# 在事件循环中运行
# asyncio.run(monitor_system())

# 停止监控
system_monitor.stop_monitoring()
```

#### 获取统计信息

```python
# 获取最近60分钟的统计
stats = system_monitor.get_statistics(minutes=60)

print(f"CPU统计:")
print(f"  平均: {stats['cpu']['avg']:.1f}%")
print(f"  最大: {stats['cpu']['max']:.1f}%")
print(f"  最小: {stats['cpu']['min']:.1f}%")
```

#### 注册回调

```python
def on_status_change(status):
    """状态变化回调"""
    if status.cpu_percent > 80:
        print(f"警告: CPU使用率过高 {status.cpu_percent:.1f}%")

system_monitor.register_callback(on_status_change)
```

---

### 2. 性能监控 (Performance Monitor)

#### 基本用法

```python
from module_06_monitoring_alerting import PerformanceMonitor
from module_06_monitoring_alerting.real_time_monitoring.performance_monitor import AlertRule

# 创建性能监控器
perf_monitor = PerformanceMonitor(monitoring_interval=5)

# 添加告警规则
rule = AlertRule(
    name="cpu_high",
    metric_type="system",  # 'system' 或 'trading'
    metric_name="cpu_percent",
    operator=">",  # '>', '<', '>=', '<=', '==', '!='
    threshold=80.0,
    duration_seconds=60,
    enabled=True,
)
perf_monitor.add_alert_rule(rule)

# 启动监控
perf_monitor.start_monitoring()

# 获取最新指标
system_metrics = perf_monitor.get_latest_system_metrics()
if system_metrics:
    print(f"时间戳: {system_metrics.timestamp}")
    print(f"CPU使用率: {system_metrics.cpu_percent:.1f}%")
    print(f"内存使用率: {system_metrics.memory_percent:.1f}%")
    print(f"Python内存: {system_metrics.python_memory_mb:.1f}MB")

trading_metrics = perf_monitor.get_latest_trading_metrics()
if trading_metrics:
    print(f"总信号数: {trading_metrics.total_signals}")
    print(f"活跃订单: {trading_metrics.active_orders}")

# 停止监控
perf_monitor.stop_monitoring()
```

#### 获取性能摘要

```python
# 获取最近60分钟的摘要
summary = perf_monitor.get_metrics_summary(minutes=60)

print(f"时间范围: {summary['time_range_minutes']}分钟")
print(f"系统指标数: {summary['system_metrics_count']}")
print(f"活跃告警数: {summary['active_alerts']}")
print(f"平均CPU: {summary.get('avg_cpu_percent', 0):.1f}%")
print(f"平均内存: {summary.get('avg_memory_percent', 0):.1f}%")
```

#### 添加回调

```python
def metrics_callback(system_metrics, trading_metrics):
    """指标变化回调"""
    print(f"系统指标更新: CPU {system_metrics.cpu_percent:.1f}%")

def alert_callback(alert):
    """告警回调"""
    print(f"告警触发: {alert.message}")

perf_monitor.add_metrics_callback(metrics_callback)
perf_monitor.add_alert_callback(alert_callback)
```

#### 导出指标

```python
# 导出到JSON文件
perf_monitor.export_metrics("metrics_export.json")
```

---

### 3. 性能追踪 (Performance Tracker)

#### 基本用法

```python
from module_06_monitoring_alerting.real_time_monitoring.performance_tracker import get_performance_tracker

# 获取性能追踪器实例（单例模式）
tracker = get_performance_tracker()

# 使用上下文管理器追踪操作
with tracker.track("data_collection", metadata={"source": "akshare"}):
    collector = AkshareDataCollector()
    data = collector.fetch_stock_history("000001", "20241101", "20241201")

# 手动记录操作
tracker.record_operation(
    operation="manual_task",
    duration=0.123,
    success=True,
    metadata={"note": "测试"}
)
```

#### 获取统计

```python
# 获取所有操作的统计
stats = tracker.get_stats()

for op_name, op_stats in stats.items():
    print(f"{op_name}:")
    print(f"  执行次数: {op_stats.count}")
    print(f"  平均耗时: {op_stats.avg_duration:.3f}秒")
    print(f"  最大耗时: {op_stats.max_duration:.3f}秒")
    print(f"  成功率: {op_stats.success_rate:.1%}")

# 获取特定操作的统计
data_collection_stats = tracker.get_stats("data_collection")
```

#### 获取记录

```python
from datetime import datetime, timedelta

# 获取最近的记录
recent_records = tracker.get_records(limit=10)

# 获取特定操作的记录
op_records = tracker.get_records(
    operation="data_collection",
    start_time=datetime.now() - timedelta(hours=1),
    success_only=True,
    limit=50
)

for record in op_records:
    print(f"{record.operation}: {record.duration:.3f}秒 - {'成功' if record.success else '失败'}")
```

#### 查找慢操作和失败操作

```python
# 获取慢操作（耗时超过1秒）
slow_operations = tracker.get_slow_operations(threshold=1.0, limit=10)

for op in slow_operations:
    print(f"慢操作: {op.operation} - {op.duration:.3f}秒")

# 获取失败操作
failed_operations = tracker.get_failed_operations(limit=10)

for op in failed_operations:
    print(f"失败操作: {op.operation} - {op.error_message}")
```

#### 获取摘要

```python
# 获取最近60分钟的摘要
summary = tracker.get_summary(minutes=60)

print(f"总操作数: {summary['total_operations']}")
print(f"成功率: {summary['success_rate']:.1%}")
print(f"平均耗时: {summary['avg_duration']:.3f}秒")

for op_name, op_info in summary['operations'].items():
    print(f"{op_name}:")
    print(f"  执行次数: {op_info['count']}")
    print(f"  成功率: {op_info['success_rate']:.1%}")
```

---

### 4. 市场监控 (Market Monitor)

#### 基本用法

```python
from module_06_monitoring_alerting import MarketMonitor

# 创建市场监控器
market_monitor = MarketMonitor(
    symbols=['000001', '600036', '000858'],
    benchmark_symbol='000001',
    lookback_period=252
)

# 获取市场状态
regime = market_monitor.calculate_market_regime()
print(f"市场状态: {regime.value}")  # bull, bear, sideways, volatile, crash, rally

condition = market_monitor.calculate_market_condition()
print(f"市场条件: {condition.value}")  # normal, overbought, oversold, etc.
```

#### 检测异常

```python
# 检测市场异常
anomalies = market_monitor.detect_anomalies()

for anomaly in anomalies:
    print(f"异常类型: {anomaly.type}")
    print(f"严重程度: {anomaly.severity:.2f}")
    print(f"描述: {anomaly.description}")
    print(f"影响股票: {anomaly.affected_symbols}")
    print(f"建议操作: {anomaly.recommended_action}")
```

#### 计算相关性和广度

```python
# 计算相关性矩阵
corr_matrix = market_monitor.calculate_correlation_matrix()

# 计算市场广度
breadth = market_monitor.calculate_market_breadth()
print(f"市场广度: {breadth:.2%}")  # 上涨股票比例

# 计算板块表现
sector_perf = market_monitor.calculate_sector_performance()
for sector, performance in sector_perf.items():
    print(f"{sector}: {performance:.2%}")
```

#### 获取市场摘要

```python
# 获取市场摘要
summary = market_monitor.get_market_summary()

print(f"市场状态: {summary['regime']}")
print(f"市场条件: {summary['condition']}")
print(f"波动率: {summary['volatility']:.2%}")
print(f"流动性评分: {summary['liquidity_score']:.2f}")
print(f"风险级别: {summary['risk_level']}")
```

---

### 5. 投资组合监控 (Portfolio Monitor)

#### 基本用法

```python
from module_06_monitoring_alerting import PortfolioMonitor, MonitoringConfig
from common.data_structures import Position

# 配置监控
config = MonitoringConfig(
    frequency=MonitoringConfig.MonitoringFrequency.MINUTE,
    metrics_window=252,
    enable_alerts=True,
    save_snapshots=True,
    snapshot_interval=300,  # 5分钟
)

# 创建监控器
portfolio_monitor = PortfolioMonitor(config)

# 模拟持仓数据
positions = [
    Position(
    symbol='000001',
        quantity=10000,
        avg_cost=15.5,
        current_price=16.0,
        # ... 其他字段
    ),
]

cash_balance = 100000
market_prices = {'000001': 16.0}

# 更新投资组合指标
metrics = portfolio_monitor.update_portfolio_metrics(
    positions=positions,
    cash_balance=cash_balance,
    market_prices=market_prices
)

print(f"组合总价值: {metrics.total_value:,.2f}")
print(f"今日盈亏: {metrics.daily_pnl:,.2f}")
print(f"今日收益率: {metrics.daily_return:.2%}")
print(f"夏普比率: {metrics.sharpe_ratio:.2f}")
print(f"最大回撤: {metrics.max_drawdown:.2%}")
print(f"VaR(95%): {metrics.var_95:.2%}")
```

#### 检测异常

```python
# 检测投资组合异常
anomalies = portfolio_monitor.detect_anomalies()

for anomaly in anomalies:
    print(f"异常类型: {anomaly['type']}")
    print(f"严重程度: {anomaly['severity']}")
    print(f"消息: {anomaly['message']}")
```

#### 获取监控摘要

```python
# 获取监控摘要
summary = portfolio_monitor.get_monitoring_summary()

print(f"状态: {summary['status']}")
print(f"组合价值: {summary['portfolio_value']:,.2f}")
print(f"今日盈亏: {summary['daily_pnl']:,.2f}")
print(f"持仓数量: {summary['n_positions']}")
print(f"前5大持仓: {summary['top_positions']}")
```

#### 注册回调

```python
def on_update(metrics):
    """指标更新回调"""
    print(f"组合价值更新: {metrics.total_value:,.2f}")

def on_alert(alert):
    """告警回调"""
    print(f"投资组合告警: {alert}")

def on_snapshot(snapshot):
    """快照回调"""
    print(f"创建快照: {snapshot.timestamp}")

portfolio_monitor.register_callback("on_update", on_update)
portfolio_monitor.register_callback("on_alert", on_alert)
portfolio_monitor.register_callback("on_snapshot", on_snapshot)
```

---

### 6. 告警管理 (Alert Manager)

#### 基本用法

```python
from module_06_monitoring_alerting import (
    AlertManager,
    AlertConfig,
    AlertSeverity,
    AlertCategory,
)
from module_06_monitoring_alerting.alert_system import AlertRule

# 配置告警管理器
alert_config = AlertConfig(
    max_alerts_per_rule=10,
    alert_retention_days=30,
    enable_auto_escalation=True,
    enable_alert_suppression=True,
    suppression_window=300,  # 抑制窗口（秒）
)

# 创建告警管理器
alert_manager = AlertManager(alert_config)

# 添加告警规则
rule = AlertRule(
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
```

#### 检查规则和触发告警

```python
# 准备指标数据
metrics = {
    "cpu_percent": 85.0,
    "memory_percent": 70.0,
    "disk_usage": 45.0,
}

# 检查规则
triggered_alerts = alert_manager.check_rules(metrics)

for alert in triggered_alerts:
    print(f"告警ID: {alert.alert_id}")
    print(f"规则: {alert.rule_id}")
    print(f"严重级别: {alert.severity.name}")
    print(f"类别: {alert.category.name}")
    print(f"消息: {alert.message}")
    print(f"指标值: {alert.metric_value}")
    print(f"阈值: {alert.threshold_value}")
```

#### 手动触发告警

```python
# 手动触发告警
alert = alert_manager.trigger_alert(
    rule_id="cpu_high_rule",
    metric_value=90.0,
    message="CPU使用率达到90%"
)
```

#### 告警确认和解决

```python
# 确认告警
success = alert_manager.acknowledge_alert(
    alert_id=alert.alert_id,
    acknowledged_by="admin"
)

# 解决告警
success = alert_manager.resolve_alert(
    alert_id=alert.alert_id,
    resolution_notes="已优化程序，CPU使用率降低"
)
```

#### 获取告警

```python
# 获取所有活跃告警
active_alerts = alert_manager.get_active_alerts()

# 按严重级别筛选
critical_alerts = alert_manager.get_active_alerts(severity=AlertSeverity.CRITICAL)

# 按类别筛选
system_alerts = alert_manager.get_active_alerts(category=AlertCategory.SYSTEM)
```

#### 告警统计

```python
# 获取统计信息
stats = alert_manager.get_alert_statistics()

print(f"总告警数: {stats.total_alerts}")
print(f"严重级别分布: {stats.alerts_by_severity}")
print(f"类别分布: {stats.alerts_by_category}")
print(f"状态分布: {stats.alerts_by_status}")
print(f"平均解决时间: {stats.average_resolution_time:.2f}秒")
print(f"升级率: {stats.escalation_rate:.2%}")
```

---

### 7. 通知服务 (Notification Service)

#### 基本用法

```python
from module_06_monitoring_alerting import (
    NotificationManager,
    NotificationConfig,
    NotificationChannel,
    NotificationPriority,
    NotificationType,
)

# 配置通知管理器
notification_config = NotificationConfig(
    enabled_channels=[NotificationChannel.EMAIL],
    rate_limits={NotificationChannel.EMAIL: 100},  # 每分钟最多100封
    retry_attempts=3,
    retry_delay=60,
    enable_aggregation=True,
    aggregation_window=300,
)

# 创建通知管理器
notifier = NotificationManager(notification_config)
```

#### 发送通知

```python
from datetime import datetime

# 发送单个通知
notification_id = notifier.send_notification(
    type=NotificationType.ALERT,
    priority=NotificationPriority.HIGH,
    channel=NotificationChannel.EMAIL,
    recipient="admin@example.com",
    subject="系统告警",
    message="CPU使用率超过85%",
    data={"cpu_percent": 87.5, "timestamp": datetime.now()}
)

print(f"通知ID: {notification_id}")
```

#### 使用模板发送通知

```python
# 使用预定义模板
notification_id = notifier.send_templated_notification(
    template_id="alert_critical",
    priority=NotificationPriority.URGENT,
    channel=NotificationChannel.EMAIL,
    recipient="admin@example.com",
    variables={
        "alert_title": "CPU告警",
        "alert_message": "CPU使用率达到90%",
        "affected_items": "服务器node-01",
        "recommended_action": "检查运行进程",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
)
```

#### 广播通知

```python
# 发送到多个渠道和接收者
notification_ids = notifier.broadcast_notification(
    type=NotificationType.SYSTEM,
    priority=NotificationPriority.NORMAL,
    channels=[NotificationChannel.EMAIL],
    recipients=["admin@example.com", "ops@example.com"],
    subject="系统维护通知",
    message="系统将于今晚22:00进行维护",
)

print(f"发送了 {len(notification_ids)} 条通知")
```

#### 获取通知状态

```python
# 查询通知状态
status = notifier.get_notification_status(notification_id)

if status:
    print(f"状态: {status['status']}")
    if status['status'] == 'delivered':
        print(f"送达时间: {status['sent_at']}")
    elif status['status'] == 'failed':
        print(f"错误: {status['error']}")
```

#### 通知统计

```python
from datetime import datetime, timedelta

# 获取最近24小时的统计
stats = notifier.get_statistics(
    start_time=datetime.now() - timedelta(days=1),
    end_time=datetime.now()
)

print(f"总发送数: {stats.total_sent}")
print(f"总送达数: {stats.total_delivered}")
print(f"总失败数: {stats.total_failed}")
print(f"成功率: {stats.success_rate:.2%}")
print(f"平均送达时间: {stats.average_delivery_time:.2f}秒")
print(f"按渠道: {stats.by_channel}")
print(f"按类型: {stats.by_type}")
```

---

### 8. 报告生成 (Report Generator)

#### 基本用法

```python
from module_06_monitoring_alerting import (
    ReportGenerator,
    ReportType,
    ReportFormat,
    ReportConfig,
)

# 创建报告生成器
report_generator = ReportGenerator(
    template_dir="templates",
    output_dir="reports"
)

# 准备报告数据
report_data = {
    "portfolio_value": 1000000,
    "cash_balance": 100000,
    "positions_value": 900000,
    "start_value": 950000,
    "daily_return": 0.015,
    "sharpe_ratio": 1.5,
    "max_drawdown": -0.08,
    "volatility": 0.15,
    "win_rate": 0.65,
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
```

#### 生成不同格式的报告

```python
# 生成JSON报告
json_config = ReportConfig(
    report_type=ReportType.DAILY,
    format=ReportFormat.JSON,
    include_charts=False,
)
json_report = report_generator.generate_report(json_config, report_data)
print(f"JSON报告: {json_report}")

# 生成HTML报告
html_config = ReportConfig(
    report_type=ReportType.DAILY,
    format=ReportFormat.HTML,
    include_charts=True,
    include_metrics=True,
    include_positions=True,
)
html_report = report_generator.generate_report(html_config, report_data)
print(f"HTML报告: {html_report}")

# 生成Excel报告
excel_config = ReportConfig(
    report_type=ReportType.WEEKLY,
    format=ReportFormat.EXCEL,
    include_positions=True,
    include_transactions=True,
)
excel_report = report_generator.generate_report(excel_config, report_data)
print(f"Excel报告: {excel_report}")

# 生成Markdown报告
md_config = ReportConfig(
    report_type=ReportType.MONTHLY,
    format=ReportFormat.MARKDOWN,
)
md_report = report_generator.generate_report(md_config, report_data)
print(f"Markdown报告: {md_report}")
```

---

### 9. 数据库管理 (Database Manager)

#### 基本用法

```python
from module_06_monitoring_alerting import get_monitoring_database_manager
from datetime import datetime

# 获取数据库管理器实例（单例模式）
db_manager = get_monitoring_database_manager("data/module06_monitoring.db")
```

#### 保存数据

```python
# 保存系统健康状态
db_manager.save_health_status(
    timestamp=datetime.now(),
    cpu_usage=45.2,
    memory_usage=62.8,
    disk_usage=35.1,
    network_sent_mb=1.5,
    network_recv_mb=3.2,
    active_threads=150,
    python_memory_mb=512.0,
    status="healthy"
)

# 保存性能指标
db_manager.save_performance_metrics(
    timestamp=datetime.now(),
    operation="data_collection",
    duration=0.123,
    success=True,
    metadata={"source": "akshare", "symbols": 100}
)

# 保存告警
db_manager.save_alert(
    alert_id="alert_001",
    rule_id="cpu_high_rule",
    timestamp=datetime.now(),
    severity="high",
    category="system",
    title="CPU使用率过高",
    message="CPU使用率达到85%",
    metric_value=85.0,
    threshold_value=80.0,
    status="triggered"
)

# 保存市场事件
db_manager.save_market_event(
    event_id="event_001",
    timestamp=datetime.now(),
    symbol="000001",
    event_type="price_spike",
    severity=0.8,
    description="价格异常波动",
    affected_symbols=["000001", "000002"],
    data={"change_percent": 0.08}
)

# 保存投资组合快照
db_manager.save_portfolio_snapshot(
    timestamp=datetime.now(),
    metrics={
        "total_value": 1000000,
        "cash_balance": 100000,
        "daily_pnl": 5000,
        "daily_return": 0.005,
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.08,
        "num_positions": 10,
        "status": "normal"
    }
)

# 保存通知记录
db_manager.save_notification(
    notification_id="notif_001",
    timestamp=datetime.now(),
    type="alert",
    priority="high",
    channel="email",
    recipient="admin@example.com",
    subject="系统告警",
    message="CPU使用率过高",
    sent_at=datetime.now(),
    delivered=True
)

# 保存报告记录
db_manager.save_report(
    report_id="report_001",
    report_type="daily",
    period_start=datetime.now(),
    period_end=datetime.now(),
    generated_at=datetime.now(),
    format="html",
    file_path="/reports/daily_report.html",
    status="completed"
)
```

#### 查询数据

```python
from datetime import datetime, timedelta
import pandas as pd

# 查询健康历史
health_df = db_manager.get_health_history(
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now(),
    limit=1000
)
print(f"健康记录: {len(health_df)}条")

# 查询性能历史
perf_df = db_manager.get_performance_history(
    operation="data_collection",
    start_date=datetime.now() - timedelta(days=1),
    limit=100
)
print(f"性能记录: {len(perf_df)}条")

# 查询告警
alerts = db_manager.get_alerts(
    severity="high",
    category="system",
    status="triggered",
    start_date=datetime.now() - timedelta(days=7),
    limit=50
)
print(f"告警记录: {len(alerts)}条")

# 查询市场事件
events = db_manager.get_market_events(
    symbol="000001",
    event_type="price_spike",
    start_date=datetime.now() - timedelta(days=7),
    limit=100
)
print(f"市场事件: {len(events)}条")

# 查询投资组合快照
snapshots_df = db_manager.get_portfolio_snapshots(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
    limit=1000
)
print(f"快照记录: {len(snapshots_df)}条")
```

#### 统计和摘要

```python
# 获取告警统计
alert_stats = db_manager.get_alert_statistics(
    start_date=datetime.now() - timedelta(days=30)
)
print(f"告警统计:")
print(f"  总数: {alert_stats.get('total', 0)}")
for severity, count in alert_stats.items():
    if severity != 'total':
        print(f"  {severity}: {count}")

# 获取性能摘要
perf_summary = db_manager.get_performance_summary(
    start_date=datetime.now() - timedelta(days=7)
)
print(f"性能摘要:")
for operation, stats in perf_summary.items():
    print(f"{operation}:")
    print(f"  执行次数: {stats['count']}")
    print(f"  平均耗时: {stats['avg_duration']:.3f}秒")
    print(f"  成功率: {stats['success_rate']:.2%}")
```

#### 数据清理

```python
# 清理30天前的旧数据
deleted = db_manager.cleanup_old_data(days=30)

print(f"清理结果:")
for table, count in deleted.items():
    print(f"  {table}: 删除{count}条记录")
```

---

## 与其他模块集成

### 与 Module 01 (数据管道) 集成

```python
from module_01_data_pipeline import AkshareDataCollector
from module_06_monitoring_alerting.real_time_monitoring.performance_tracker import get_performance_tracker

# 追踪数据采集性能
collector = AkshareDataCollector()
tracker = get_performance_tracker()

with tracker.track("akshare_data_fetch"):
    data = collector.fetch_stock_history("000001", "20241101", "20241201")

# 获取统计
stats = tracker.get_stats("akshare_data_fetch")
print(f"平均耗时: {stats['akshare_data_fetch'].avg_duration:.3f}秒")
```

### 与 Module 05 (风险管理) 集成

```python
from module_05_risk_management import PortfolioRiskAnalyzer
from module_06_monitoring_alerting import (
    AlertManager,
    AlertSeverity,
    AlertCategory,
    NotificationManager,
    NotificationChannel,
    NotificationPriority,
    NotificationType,
)
from module_06_monitoring_alerting.alert_system import AlertRule
import pandas as pd

# 创建风险分析器
risk_analyzer = PortfolioRiskAnalyzer()

# 创建告警管理器
alert_manager = AlertManager()

# 添加风险告警规则
var_rule = AlertRule(
    rule_id="var_breach",
    name="VaR突破",
    description="VaR超过阈值",
    category=AlertCategory.RISK,
    severity=AlertSeverity.HIGH,
    condition="var_95 < -0.05",
    threshold=-0.05,
    comparison="<",
    metric="var_95",
)
alert_manager.add_rule(var_rule)

# 计算风险并检查告警
portfolio = {'000001': {'weight': 0.3}, '600036': {'weight': 0.4}}
returns = pd.DataFrame(...)  # 收益率数据
risk_metrics = risk_analyzer.analyze_portfolio_risk(portfolio, returns)

# 检查告警
alerts = alert_manager.check_rules({"var_95": risk_metrics['var_95']})

# 发送通知
if alerts:
    notifier = NotificationManager()
    for alert in alerts:
        notifier.send_notification(
            type=NotificationType.ALERT,
            priority=NotificationPriority.HIGH,
            channel=NotificationChannel.EMAIL,
            recipient="risk@example.com",
            subject=f"风险告警: {alert.title}",
            message=alert.message
        )
```

---

## 完整示例

```python
import asyncio
from datetime import datetime
from module_06_monitoring_alerting import (
    get_monitoring_database_manager,
    AlertManager,
    AlertConfig,
    AlertSeverity,
    AlertCategory,
    NotificationManager,
    NotificationConfig,
    NotificationChannel,
    NotificationPriority,
    NotificationType,
)
from module_06_monitoring_alerting.alert_system import AlertRule
from module_06_monitoring_alerting.real_time_monitoring.system_monitor import get_system_monitor
from module_06_monitoring_alerting.real_time_monitoring.performance_tracker import get_performance_tracker

async def main():
    """完整的监控流程"""
    
    # 1. 初始化组件
    system_monitor = get_system_monitor()
    tracker = get_performance_tracker()
    db_manager = get_monitoring_database_manager()
    alert_manager = AlertManager(AlertConfig())
    notifier = NotificationManager(NotificationConfig())
    
    # 2. 添加告警规则
    cpu_rule = AlertRule(
        rule_id="cpu_high",
        name="CPU使用率高",
        description="CPU超过80%",
        category=AlertCategory.SYSTEM,
        severity=AlertSeverity.WARNING,
        condition="cpu_percent > 80",
        threshold=80.0,
        comparison=">",
        metric="cpu_percent",
    )
    alert_manager.add_rule(cpu_rule)
    
    # 3. 监控循环
    for _ in range(5):
        # 获取系统状态
        status = system_monitor.get_system_status()
        
        # 保存到数据库
        db_manager.save_health_status(
            timestamp=datetime.now(),
            cpu_usage=status.cpu_percent,
            memory_usage=status.memory_percent,
            disk_usage=status.disk_percent,
        )
        
        # 检查告警
        alerts = alert_manager.check_rules({"cpu_percent": status.cpu_percent})
        
        # 处理告警
        for alert in alerts:
            # 保存告警
            db_manager.save_alert(
                alert_id=alert.alert_id,
                rule_id=alert.rule_id,
                timestamp=alert.timestamp,
                severity=alert.severity.name,
                category=alert.category.name,
                title=alert.title,
                message=alert.message,
                metric_value=alert.metric_value,
                threshold_value=alert.threshold_value,
            )
            
            # 发送通知
            notifier.send_notification(
                type=NotificationType.ALERT,
                priority=NotificationPriority.HIGH,
                channel=NotificationChannel.EMAIL,
                recipient="admin@example.com",
                subject=alert.title,
                message=alert.message,
            )
        
        await asyncio.sleep(10)
    
    print("监控完成")

# 运行
# asyncio.run(main())
```

---

## 测试

### 运行测试

```bash
# 激活conda环境
conda activate study

# 运行Module 06测试
cd /Users/victor/Desktop/25fininnov/FinLoom-server
python tests/module06_test.py
```

### 测试覆盖

测试文件 `tests/module06_test.py` 包含以下测试：

1. 系统监控器测试
2. 性能监控器测试
3. 性能追踪器测试
4. 告警管理器测试
5. 通知管理器测试
6. 报告生成器测试
7. 数据库管理器测试
8. 集成测试

---

## 数据结构

### AlertSeverity (告警严重级别)
```python
class AlertSeverity(Enum):
    INFO = "info"
    LOW = "low"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
```

### AlertCategory (告警类别)
```python
class AlertCategory(Enum):
    SYSTEM = "system"
    PERFORMANCE = "performance"
    RISK = "risk"
    MARKET = "market"
    PORTFOLIO = "portfolio"
    EXECUTION = "execution"
```

### AlertStatus (告警状态)
```python
class AlertStatus(Enum):
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
```

### NotificationChannel (通知渠道)
```python
class NotificationChannel(Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    WEBSOCKET = "websocket"
```

### NotificationPriority (通知优先级)
```python
class NotificationPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
```

### NotificationType (通知类型)
```python
class NotificationType(Enum):
    ALERT = "alert"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SYSTEM = "system"
    MARKET = "market"
    PORTFOLIO = "portfolio"
```

### ReportType (报告类型)
```python
class ReportType(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"
```

### ReportFormat (报告格式)
```python
class ReportFormat(Enum):
    HTML = "html"
    JSON = "json"
    EXCEL = "excel"
    MARKDOWN = "markdown"
```

### MarketRegime (市场状态)
```python
class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRASH = "crash"
    RALLY = "rally"
```

### MarketCondition (市场条件)
```python
class MarketCondition(Enum):
    NORMAL = "normal"
    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
```

---

## 最佳实践

### 1. 系统监控
- 设置合理的监控间隔（建议60秒）
- 注册回调函数处理异常状态
- 定期检查系统健康状态
- 及时响应告警信号

### 2. 性能追踪
- 对关键操作使用性能追踪
- 定期分析慢操作和失败操作
- 设置合理的性能阈值
- 记录详细的元数据便于分析

### 3. 告警管理
- 设置合理的告警阈值避免告警疲劳
- 使用告警抑制避免重复告警
- 及时确认和解决告警
- 定期分析告警统计优化规则

### 4. 通知服务
- 配置合理的速率限制
- 使用模板提高通知质量
- 启用聚合减少通知数量
- 定期检查通知发送状态

### 5. 报告生成
- 选择合适的报告格式
- 定期生成报告保存历史
- 包含关键指标和图表
- 及时分发报告给相关人员

---

## 环境变量配置

```bash
# 数据库路径
MODULE06_DB_PATH="data/module06_monitoring.db"

# 邮件配置
MODULE06_EMAIL_SERVER="smtp.gmail.com"
MODULE06_EMAIL_PORT=587
MODULE06_EMAIL_USER="your_email@example.com"
MODULE06_EMAIL_PASSWORD="your_password"

# Webhook配置
MODULE06_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# 日志级别
MODULE06_LOG_LEVEL="INFO"
```

---

## 常见问题

### 1. 系统监控器无法启动？
确保有足够的系统权限读取系统信息，在某些操作系统上可能需要管理员权限。

### 2. 邮件通知发送失败？
检查SMTP服务器配置、账号密码、网络连接和防火墙设置。

### 3. 数据库操作失败？
确保数据库文件路径存在且有写权限，检查磁盘空间是否充足。

### 4. 告警规则不生效？
检查规则是否已启用，条件表达式是否正确，冷却时间是否合理。

### 5. 性能追踪数据过多？
可以调整`max_records`参数限制记录数量，或定期清理旧数据。

---

## 更新日志

### v1.0.0 (2024-10-04)
- ✅ 实现系统监控器（SystemMonitor）
- ✅ 实现性能监控器（PerformanceMonitor）
- ✅ 实现性能追踪器（PerformanceTracker）
- ✅ 实现市场监控器（MarketMonitor）
- ✅ 实现投资组合监控器（PortfolioMonitor）
- ✅ 实现告警管理器（AlertManager）
- ✅ 实现通知管理器（NotificationManager）
- ✅ 实现邮件通知器（EmailNotifier）
- ✅ 实现Webhook通知器（WebhookNotifier）
- ✅ 实现报告生成器（ReportGenerator）
- ✅ 实现数据库管理器（MonitoringDatabaseManager）
- ✅ 完成单元测试和集成测试
- ✅ 编写完整的API文档

---

## 联系方式

如有问题或建议，请联系开发团队或提交Issue。
