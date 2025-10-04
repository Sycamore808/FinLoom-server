# Module 06 - 监控告警模块

## 概述

监控告警模块是 FinLoom 量化交易系统的实时监控中枢，负责系统健康监控、性能追踪、异常检测、风险预警和智能通知服务。该模块与所有其他模块集成，确保系统稳定运行和及时响应市场变化。

## 主要功能

### 1. 实时系统监控 (Real-time System Monitoring)
- **SystemHealthMonitor**: 系统健康状态监控
- **PerformanceMonitor**: 性能指标追踪
- **ResourceMonitor**: 资源使用监控（CPU、内存、磁盘）
- **APIMonitor**: API响应时间和成功率监控

### 2. 市场监控 (Market Monitoring)
- **MarketWatchdog**: 实时市场数据监控
- **PriceAlertManager**: 价格突破预警
- **VolumeMonitor**: 异常成交量监控
- **VolatilityTracker**: 波动率追踪和预警

### 3. 交易监控 (Trading Monitoring)
- **OrderMonitor**: 订单状态实时监控
- **PositionMonitor**: 持仓变化追踪
- **PnLTracker**: 盈亏实时计算和监控
- **ExecutionQualityMonitor**: 交易执行质量监控

### 4. 风险预警 (Risk Alerting)
- **RiskAlertEngine**: 风险预警引擎
- **LimitBreachDetector**: 风险限额违规检测
- **DrawdownAlertManager**: 回撤预警
- **CorrelationMonitor**: 相关性变化监控

### 5. 通知服务 (Notification Service)
- **NotificationManager**: 统一通知管理
- **EmailNotifier**: 邮件通知
- **WebhookNotifier**: Webhook通知
- **WebSocketBroadcaster**: 实时推送

### 6. 日志管理 (Log Management)
- **LogAggregator**: 日志聚合
- **LogAnalyzer**: 日志分析
- **ErrorTracker**: 错误追踪
- **AuditLogger**: 审计日志

### 7. 报告生成 (Reporting Engine)
- **ReportScheduler**: 定时报告生成
- **DailyReportGenerator**: 日报生成
- **WeeklyReportGenerator**: 周报生成
- **CustomReportBuilder**: 自定义报告构建器

## 快速开始

### 环境配置

```python
# 导入 Module 06 组件
from module_06_monitoring_alerting import (
    SystemHealthMonitor,
    PerformanceMonitor,
    MarketWatchdog,
    RiskAlertEngine,
    NotificationManager,
    ReportScheduler,
    get_monitoring_database_manager
)

# 导入其他模块
from module_01_data_pipeline import AkshareDataCollector
from module_05_risk_management import PortfolioRiskAnalyzer
```

### 基础使用示例

```python
import asyncio
from datetime import datetime, timedelta

# 1. 系统健康监控
from module_06_monitoring_alerting import SystemHealthMonitor, HealthConfig

# 配置健康监控
health_config = HealthConfig(
    check_interval=60,           # 检查间隔（秒）
    cpu_threshold=80.0,          # CPU阈值
    memory_threshold=85.0,       # 内存阈值
    disk_threshold=90.0,         # 磁盘阈值
    enable_auto_recovery=True    # 启用自动恢复
)

# 创建健康监控器
health_monitor = SystemHealthMonitor(health_config)

# 启动监控
await health_monitor.start_monitoring()

# 获取系统状态
health_status = health_monitor.get_health_status()
print(f"系统健康状态: {health_status['overall_status']}")
print(f"  CPU使用率: {health_status['cpu_usage']:.1f}%")
print(f"  内存使用率: {health_status['memory_usage']:.1f}%")
print(f"  磁盘使用率: {health_status['disk_usage']:.1f}%")
print(f"  运行时长: {health_status['uptime']} 小时")

# 2. 性能监控
from module_06_monitoring_alerting import PerformanceMonitor, PerformanceConfig

perf_config = PerformanceConfig(
    track_api_latency=True,
    track_database_queries=True,
    track_model_inference=True,
    slow_query_threshold=1.0     # 慢查询阈值（秒）
)

perf_monitor = PerformanceMonitor(perf_config)

# 记录性能指标
with perf_monitor.track_operation("data_collection"):
    # 执行数据采集
    collector = AkshareDataCollector()
    data = collector.fetch_stock_history("000001", "20241101", "20241201")

# 获取性能统计
perf_stats = perf_monitor.get_statistics()
print(f"\n性能统计:")
print(f"  数据采集平均耗时: {perf_stats['data_collection']['avg_time']:.2f}秒")
print(f"  API调用次数: {perf_stats['api_calls']['total_count']}")
print(f"  慢查询数量: {perf_stats['slow_queries']['count']}")

# 3. 市场监控和预警
from module_06_monitoring_alerting import MarketWatchdog, WatchdogConfig

watchdog_config = WatchdogConfig(
    symbols=['000001', '600036', '000858'],
    price_change_threshold=0.05,      # 5%涨跌幅预警
    volume_surge_multiplier=3.0,      # 3倍成交量异常
    volatility_threshold=0.03,        # 日波动率3%
    check_interval=30                 # 30秒检查一次
)

market_watchdog = MarketWatchdog(watchdog_config)

# 启动市场监控
await market_watchdog.start_watching()

# 设置价格预警
market_watchdog.set_price_alert(
    symbol='000001',
    alert_type='above',
    threshold_price=16.5,
    message='平安银行突破16.5元'
)

market_watchdog.set_price_alert(
    symbol='000001',
    alert_type='below',
    threshold_price=14.5,
    message='平安银行跌破14.5元'
)

# 获取监控状态
watchdog_status = market_watchdog.get_watchdog_status()
print(f"\n市场监控状态:")
print(f"  监控股票数: {watchdog_status['symbols_count']}")
print(f"  活跃预警数: {watchdog_status['active_alerts']}")
print(f"  最近触发: {watchdog_status['recent_triggers']}")

# 4. 风险预警
from module_06_monitoring_alerting import RiskAlertEngine, RiskAlertConfig

risk_alert_config = RiskAlertConfig(
    var_threshold=0.05,              # VaR超过5%预警
    drawdown_threshold=0.10,         # 回撤超过10%预警
    concentration_threshold=0.35,     # 单股仓位超过35%预警
    leverage_threshold=1.5,          # 杠杆超过1.5倍预警
    alert_cooldown=300               # 预警冷却期5分钟
)

risk_alert = RiskAlertEngine(risk_alert_config)

# 监控投资组合风险
portfolio = {
    '000001': {'weight': 0.3, 'shares': 1000, 'cost': 15.5},
    '600036': {'weight': 0.4, 'shares': 800, 'cost': 45.2},
    '000858': {'weight': 0.3, 'shares': 500, 'cost': 180.0}
}

# 检查风险
risk_alerts = await risk_alert.check_portfolio_risk(portfolio)

if risk_alerts:
    print(f"\n⚠️ 风险预警:")
    for alert in risk_alerts:
        print(f"  [{alert['severity']}] {alert['message']}")
        print(f"    触发时间: {alert['timestamp']}")
        print(f"    当前值: {alert['current_value']:.2%}")
        print(f"    阈值: {alert['threshold']:.2%}")

# 5. 通知管理
from module_06_monitoring_alerting import NotificationManager, NotificationConfig

notification_config = NotificationConfig(
    enable_email=True,
    enable_webhook=True,
    enable_websocket=True,
    email_recipients=['trader@example.com'],
    webhook_url='https://your-webhook.com/alerts',
    alert_priority_threshold='medium'  # 只发送中级以上告警
)

notifier = NotificationManager(notification_config)

# 发送通知
await notifier.send_notification(
    title='风险限额违规',
    message='000001持仓比例超过35%限制',
    severity='high',
    data={
        'symbol': '000001',
        'current_weight': 0.38,
        'limit': 0.35,
        'action_required': '减仓'
    }
)

# 批量通知
notifications = [
    {'title': '价格突破', 'message': '600036突破50元', 'severity': 'medium'},
    {'title': '成交量异常', 'message': '000858成交量放大3倍', 'severity': 'low'},
]

await notifier.send_batch_notifications(notifications)

# 6. 日志聚合和分析
from module_06_monitoring_alerting import LogAggregator, LogAnalyzer

log_aggregator = LogAggregator()
log_analyzer = LogAnalyzer()

# 聚合最近1小时的日志
recent_logs = log_aggregator.aggregate_logs(
    start_time=datetime.now() - timedelta(hours=1),
    end_time=datetime.now(),
    modules=['module_01', 'module_03', 'module_05']
)

# 分析日志
log_analysis = log_analyzer.analyze_logs(recent_logs)

print(f"\n日志分析结果:")
print(f"  总日志数: {log_analysis['total_logs']}")
print(f"  错误数: {log_analysis['error_count']}")
print(f"  警告数: {log_analysis['warning_count']}")
print(f"  最频繁错误: {log_analysis['top_errors']}")
print(f"  异常模块: {log_analysis['problematic_modules']}")

# 7. 定时报告生成
from module_06_monitoring_alerting import ReportScheduler, ReportConfig

report_config = ReportConfig(
    daily_report_time='18:00',       # 每日18:00生成日报
    weekly_report_day='Friday',      # 每周五生成周报
    monthly_report_day=1,            # 每月1日生成月报
    report_recipients=['manager@example.com'],
    include_performance=True,
    include_risk_metrics=True,
    include_system_health=True
)

report_scheduler = ReportScheduler(report_config)

# 启动报告调度
await report_scheduler.start_scheduler()

# 手动生成日报
daily_report = await report_scheduler.generate_daily_report(
    date=datetime.now().date()
)

print(f"\n日报生成:")
print(f"  报告日期: {daily_report['date']}")
print(f"  总收益率: {daily_report['total_return']:.2%}")
print(f"  今日PnL: {daily_report['daily_pnl']:.2f}元")
print(f"  系统健康: {daily_report['system_health']}")
print(f"  告警数量: {daily_report['alert_count']}")

# 8. 保存监控数据
monitoring_db = get_monitoring_database_manager()

# 保存系统健康记录
monitoring_db.save_health_status(
    timestamp=datetime.now(),
    cpu_usage=health_status['cpu_usage'],
    memory_usage=health_status['memory_usage'],
    disk_usage=health_status['disk_usage']
)

# 保存性能指标
monitoring_db.save_performance_metrics(
    timestamp=datetime.now(),
    operation='data_collection',
    duration=perf_stats['data_collection']['avg_time'],
    success=True
)

# 保存告警记录
for alert in risk_alerts:
    monitoring_db.save_alert(
        alert_type='risk',
        severity=alert['severity'],
        message=alert['message'],
        timestamp=alert['timestamp'],
        data=alert
    )

print("\n✅ 监控告警系统运行中！")
```

## API 参考

### SystemHealthMonitor

系统健康状态监控。

#### 构造函数
```python
SystemHealthMonitor(config: HealthConfig)
```

#### 配置参数 (HealthConfig)
```python
@dataclass
class HealthConfig:
    check_interval: int = 60             # 检查间隔（秒）
    cpu_threshold: float = 80.0          # CPU阈值
    memory_threshold: float = 85.0       # 内存阈值
    disk_threshold: float = 90.0         # 磁盘阈值
    enable_auto_recovery: bool = True    # 启用自动恢复
    health_check_timeout: int = 30       # 健康检查超时
```

#### 主要方法

**start_monitoring() -> None**
- 启动系统健康监控
- 异步持续监控

**stop_monitoring() -> None**
- 停止监控

**get_health_status() -> Dict[str, Any]**
- 获取当前健康状态
- 返回CPU、内存、磁盘使用情况

**check_module_health(module_name: str) -> bool**
- 检查特定模块健康状态
- 返回True/False

**get_uptime() -> timedelta**
- 获取系统运行时长

#### 使用示例
```python
monitor = SystemHealthMonitor(config)
await monitor.start_monitoring()

status = monitor.get_health_status()
if status['overall_status'] == 'critical':
    print("系统状态严重！")
```

### PerformanceMonitor

性能指标追踪。

#### 构造函数
```python
PerformanceMonitor(config: PerformanceConfig)
```

#### 配置参数 (PerformanceConfig)
```python
@dataclass
class PerformanceConfig:
    track_api_latency: bool = True
    track_database_queries: bool = True
    track_model_inference: bool = True
    slow_query_threshold: float = 1.0     # 秒
    enable_profiling: bool = False
```

#### 主要方法

**track_operation(operation_name: str) -> ContextManager**
- 追踪操作性能
- 使用with语句

**record_metric(metric_name: str, value: float, tags: Dict = None) -> None**
- 记录性能指标

**get_statistics(metric_name: str = None) -> Dict[str, Any]**
- 获取性能统计
- 可选择特定指标

**get_slow_queries() -> List[Dict[str, Any]]**
- 获取慢查询列表

**reset_statistics() -> None**
- 重置统计数据

#### 使用示例
```python
perf_monitor = PerformanceMonitor(config)

with perf_monitor.track_operation("model_prediction"):
    prediction = model.predict(features)

stats = perf_monitor.get_statistics("model_prediction")
```

### MarketWatchdog

实时市场数据监控和预警。

#### 构造函数
```python
MarketWatchdog(config: WatchdogConfig)
```

#### 配置参数 (WatchdogConfig)
```python
@dataclass
class WatchdogConfig:
    symbols: List[str] = None
    price_change_threshold: float = 0.05      # 涨跌幅阈值
    volume_surge_multiplier: float = 3.0      # 成交量异常倍数
    volatility_threshold: float = 0.03        # 波动率阈值
    check_interval: int = 30                  # 检查间隔（秒）
    enable_circuit_breaker: bool = True       # 启用熔断检测
```

#### 主要方法

**start_watching() -> None**
- 启动市场监控
- 异步持续监控

**stop_watching() -> None**
- 停止监控

**set_price_alert(symbol: str, alert_type: str, threshold_price: float, message: str = None) -> str**
- 设置价格预警
- alert_type: 'above', 'below', 'cross'
- 返回alert_id

**remove_price_alert(alert_id: str) -> bool**
- 移除价格预警

**get_watchdog_status() -> Dict[str, Any]**
- 获取监控状态

**get_triggered_alerts(since: datetime = None) -> List[Dict[str, Any]]**
- 获取触发的预警

#### 使用示例
```python
watchdog = MarketWatchdog(config)
await watchdog.start_watching()

# 设置预警
alert_id = watchdog.set_price_alert('000001', 'above', 16.5)

# 获取触发的预警
alerts = watchdog.get_triggered_alerts(since=datetime.now() - timedelta(hours=1))
```

### RiskAlertEngine

风险预警引擎。

#### 构造函数
```python
RiskAlertEngine(config: RiskAlertConfig)
```

#### 配置参数 (RiskAlertConfig)
```python
@dataclass
class RiskAlertConfig:
    var_threshold: float = 0.05              # VaR阈值
    drawdown_threshold: float = 0.10         # 回撤阈值
    concentration_threshold: float = 0.35    # 集中度阈值
    leverage_threshold: float = 1.5          # 杠杆阈值
    alert_cooldown: int = 300                # 预警冷却期（秒）
    severity_levels: Dict[str, float] = None  # 严重性级别
```

#### 主要方法

**check_portfolio_risk(portfolio: Dict) -> List[Dict[str, Any]]**
- 检查投资组合风险
- 返回预警列表

**check_var_breach(portfolio: Dict, current_var: float) -> Optional[Dict]**
- 检查VaR违规

**check_drawdown_breach(portfolio: Dict, current_drawdown: float) -> Optional[Dict]**
- 检查回撤违规

**check_concentration_risk(portfolio: Dict) -> List[Dict]**
- 检查集中度风险

**get_alert_history(start_date: datetime = None) -> List[Dict]**
- 获取预警历史

#### 使用示例
```python
risk_alert = RiskAlertEngine(config)

alerts = await risk_alert.check_portfolio_risk(portfolio)

for alert in alerts:
    if alert['severity'] == 'high':
        print(f"高风险预警: {alert['message']}")
```

### NotificationManager

统一通知管理。

#### 构造函数
```python
NotificationManager(config: NotificationConfig)
```

#### 配置参数 (NotificationConfig)
```python
@dataclass
class NotificationConfig:
    enable_email: bool = True
    enable_webhook: bool = True
    enable_websocket: bool = True
    email_recipients: List[str] = None
    email_server: str = 'smtp.gmail.com'
    email_port: int = 587
    webhook_url: str = None
    alert_priority_threshold: str = 'low'  # 'low', 'medium', 'high'
```

#### 主要方法

**send_notification(title: str, message: str, severity: str = 'info', data: Dict = None) -> bool**
- 发送单个通知
- severity: 'info', 'low', 'medium', 'high', 'critical'

**send_batch_notifications(notifications: List[Dict]) -> Dict[str, int]**
- 批量发送通知
- 返回成功/失败统计

**send_email(to: List[str], subject: str, body: str, html: bool = False) -> bool**
- 发送邮件通知

**send_webhook(url: str, data: Dict) -> bool**
- 发送Webhook通知

**broadcast_websocket(channel: str, message: Dict) -> int**
- WebSocket广播
- 返回接收者数量

**get_notification_history(limit: int = 100) -> List[Dict]**
- 获取通知历史

#### 使用示例
```python
notifier = NotificationManager(config)

# 发送通知
await notifier.send_notification(
    title='交易执行成功',
    message='000001买入1000股',
    severity='info',
    data={'symbol': '000001', 'quantity': 1000}
)

# 发送邮件
notifier.send_email(
    to=['trader@example.com'],
    subject='日报',
    body='今日交易汇总...'
)
```

### ReportScheduler

定时报告生成调度。

#### 构造函数
```python
ReportScheduler(config: ReportConfig)
```

#### 配置参数 (ReportConfig)
```python
@dataclass
class ReportConfig:
    daily_report_time: str = '18:00'
    weekly_report_day: str = 'Friday'
    monthly_report_day: int = 1
    report_recipients: List[str] = None
    report_format: str = 'html'              # 'html', 'pdf', 'json'
    include_performance: bool = True
    include_risk_metrics: bool = True
    include_system_health: bool = True
```

#### 主要方法

**start_scheduler() -> None**
- 启动报告调度

**stop_scheduler() -> None**
- 停止调度

**generate_daily_report(date: date = None) -> Dict[str, Any]**
- 生成日报
- 默认生成当日报告

**generate_weekly_report(week_end_date: date = None) -> Dict[str, Any]**
- 生成周报

**generate_monthly_report(month: int = None, year: int = None) -> Dict[str, Any]**
- 生成月报

**generate_custom_report(start_date: date, end_date: date, metrics: List[str]) -> Dict[str, Any]**
- 生成自定义报告

**get_scheduled_reports() -> List[Dict]**
- 获取已调度的报告任务

#### 使用示例
```python
scheduler = ReportScheduler(config)
await scheduler.start_scheduler()

# 生成日报
daily_report = await scheduler.generate_daily_report()

# 生成自定义报告
custom_report = await scheduler.generate_custom_report(
    start_date=date(2024, 11, 1),
    end_date=date(2024, 11, 30),
    metrics=['pnl', 'sharpe', 'drawdown']
)
```

### LogAggregator & LogAnalyzer

日志聚合和分析。

#### LogAggregator
```python
class LogAggregator:
    def aggregate_logs(self, start_time: datetime, end_time: datetime, modules: List[str] = None) -> List[Dict]
    def filter_logs(self, logs: List[Dict], level: str = None, keyword: str = None) -> List[Dict]
    def export_logs(self, logs: List[Dict], format: str = 'json', filepath: str = None) -> str
```

#### LogAnalyzer
```python
class LogAnalyzer:
    def analyze_logs(self, logs: List[Dict]) -> Dict[str, Any]
    def detect_patterns(self, logs: List[Dict]) -> List[Dict]
    def find_anomalies(self, logs: List[Dict]) -> List[Dict]
    def generate_log_report(self, logs: List[Dict]) -> str
```

#### 使用示例
```python
aggregator = LogAggregator()
analyzer = LogAnalyzer()

# 聚合日志
logs = aggregator.aggregate_logs(
    start_time=datetime.now() - timedelta(days=1),
    end_time=datetime.now()
)

# 分析日志
analysis = analyzer.analyze_logs(logs)
anomalies = analyzer.find_anomalies(logs)
```

## 数据库管理

### MonitoringDatabaseManager

监控数据专用数据库管理。

#### 使用方法
```python
from module_06_monitoring_alerting import get_monitoring_database_manager

monitoring_db = get_monitoring_database_manager()
```

#### 主要方法

**保存监控数据**
- `save_health_status(timestamp: datetime, cpu_usage: float, memory_usage: float, disk_usage: float) -> bool`
- `save_performance_metrics(timestamp: datetime, operation: str, duration: float, success: bool) -> bool`
- `save_alert(alert_type: str, severity: str, message: str, timestamp: datetime, data: Dict = None) -> bool`
- `save_market_event(symbol: str, event_type: str, data: Dict, timestamp: datetime) -> bool`

**查询监控数据**
- `get_health_history(start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame`
- `get_performance_history(operation: str = None, start_date: datetime = None) -> pd.DataFrame`
- `get_alerts(severity: str = None, start_date: datetime = None, limit: int = 100) -> List[Dict]`
- `get_market_events(symbol: str = None, event_type: str = None, start_date: datetime = None) -> List[Dict]`

**统计和报表**
- `get_system_uptime() -> timedelta`
- `get_alert_statistics(start_date: datetime = None) -> Dict[str, int]`
- `get_performance_summary(start_date: datetime = None) -> Dict[str, Any]`
- `get_database_stats() -> Dict[str, Any]`

#### 使用示例
```python
monitoring_db = get_monitoring_database_manager()

# 保存健康状态
monitoring_db.save_health_status(
    timestamp=datetime.now(),
    cpu_usage=45.2,
    memory_usage=62.8,
    disk_usage=35.1
)

# 查询预警
alerts = monitoring_db.get_alerts(
    severity='high',
    start_date=datetime.now() - timedelta(days=7)
)

# 获取统计
stats = monitoring_db.get_alert_statistics(
    start_date=datetime.now() - timedelta(days=30)
)
```

## 与其他模块集成

### 与 Module 01 (数据管道) 集成
```python
# 监控数据采集性能
from module_01_data_pipeline import AkshareDataCollector
from module_06_monitoring_alerting import PerformanceMonitor

perf_monitor = PerformanceMonitor(config)
collector = AkshareDataCollector()

with perf_monitor.track_operation("data_fetch"):
    data = collector.fetch_stock_history("000001", "20241101", "20241201")

# 如果数据采集过慢，发送告警
if perf_monitor.get_last_duration("data_fetch") > 5.0:
    notifier.send_notification(
        title='数据采集缓慢',
        message='数据采集耗时超过5秒',
        severity='medium'
    )
```

### 与 Module 03 (AI模型) 集成
```python
# 监控模型推理性能
from module_03_ai_models import LSTMModel
from module_06_monitoring_alerting import PerformanceMonitor

perf_monitor = PerformanceMonitor(config)
lstm_model = LSTMModel.load_model("risk_predictor")

with perf_monitor.track_operation("model_inference"):
    predictions = lstm_model.predict(features)

# 追踪模型准确率
monitoring_db.save_performance_metrics(
    timestamp=datetime.now(),
    operation='model_inference',
    duration=perf_monitor.get_last_duration("model_inference"),
    success=True
)
```

### 与 Module 05 (风险管理) 集成
```python
# 监控风险指标变化
from module_05_risk_management import PortfolioRiskAnalyzer
from module_06_monitoring_alerting import RiskAlertEngine

risk_analyzer = PortfolioRiskAnalyzer(config)
risk_alert = RiskAlertEngine(alert_config)

# 计算风险
risk_metrics = risk_analyzer.analyze_portfolio_risk(portfolio, returns)

# 检查风险预警
alerts = await risk_alert.check_portfolio_risk(portfolio)

# 如果有预警，发送通知
if alerts:
    for alert in alerts:
        await notifier.send_notification(
            title=f"风险预警: {alert['type']}",
            message=alert['message'],
            severity=alert['severity'],
            data=alert
        )
```

### 与 Module 08 (执行) 集成
```python
# 监控订单执行
from module_08_execution import OrderManager
from module_06_monitoring_alerting import MarketWatchdog

order_manager = OrderManager()
watchdog = MarketWatchdog(config)

# 提交订单
order = order_manager.create_order('000001', 1000, 15.5)
result = order_manager.submit_order(order)

# 记录执行事件
monitoring_db.save_market_event(
    symbol='000001',
    event_type='order_executed',
    data={
        'order_id': order.order_id,
        'quantity': 1000,
        'price': 15.5,
        'status': result['status']
    },
    timestamp=datetime.now()
)
```

## 实时监控仪表板

### WebSocket实时推送

```python
from module_06_monitoring_alerting import WebSocketBroadcaster

broadcaster = WebSocketBroadcaster()

# 实时推送系统状态
async def push_system_status():
    while True:
        status = health_monitor.get_health_status()
        await broadcaster.broadcast('system_status', status)
        await asyncio.sleep(10)

# 实时推送市场预警
async def push_market_alerts():
    async for alert in watchdog.alert_stream():
        await broadcaster.broadcast('market_alert', alert)

# 前端订阅
# ws://localhost:8000/ws/monitoring
```

### 监控API端点

```python
from fastapi import FastAPI
from module_06_monitoring_alerting import get_monitoring_router

app = FastAPI()
app.include_router(get_monitoring_router())

# GET /api/v1/monitoring/health - 系统健康状态
# GET /api/v1/monitoring/performance - 性能指标
# GET /api/v1/monitoring/alerts - 预警列表
# GET /api/v1/monitoring/logs - 日志查询
# POST /api/v1/monitoring/alert/acknowledge - 确认预警
```

## 便捷函数

```python
# 快速健康检查
from module_06_monitoring_alerting import quick_health_check

health = quick_health_check()
if not health['is_healthy']:
    print(f"系统问题: {health['issues']}")

# 快速发送告警
from module_06_monitoring_alerting import send_alert

send_alert("价格突破", "000001突破16.5元", severity='medium')

# 快速生成报告
from module_06_monitoring_alerting import generate_report

report = generate_report('daily', date=datetime.now().date())

# 快速查询日志
from module_06_monitoring_alerting import query_logs

logs = query_logs(level='ERROR', last_hours=24)
```

## 测试和示例

### 运行完整测试
```bash
cd /Users/victor/Desktop/25fininnov/FinLoom-server
python tests/module06_monitoring_alerting_test.py
```

### 测试覆盖内容
- 系统健康监控测试
- 性能追踪测试
- 市场监控和预警测试
- 风险告警测试
- 通知服务测试
- 日志聚合和分析测试
- 报告生成测试
- 数据库操作测试
- WebSocket推送测试
- API端点测试

## 配置说明

### 环境变量
- `MODULE06_DB_PATH`: 监控数据库路径
- `MODULE06_EMAIL_SERVER`: 邮件服务器地址
- `MODULE06_EMAIL_PASSWORD`: 邮件密码
- `MODULE06_WEBHOOK_URL`: Webhook地址
- `MODULE06_LOG_LEVEL`: 日志级别

### 监控配置文件
```yaml
# config/monitoring_config.yaml
system_monitoring:
  check_interval: 60
  cpu_threshold: 80
  memory_threshold: 85
  disk_threshold: 90

market_monitoring:
  price_change_threshold: 0.05
  volume_surge_multiplier: 3.0
  check_interval: 30

notifications:
  email_enabled: true
  webhook_enabled: true
  websocket_enabled: true
  priority_threshold: 'medium'

reporting:
  daily_report_time: '18:00'
  weekly_report_day: 'Friday'
  monthly_report_day: 1
```

## 性能基准

| 操作 | 处理时间 | 内存使用 |
|------|----------|----------|
| 健康检查 | ~10ms | ~2MB |
| 性能指标记录 | ~5ms | ~1MB |
| 预警检查 | ~50ms | ~5MB |
| 通知发送 | ~100ms | ~3MB |
| 日志分析 | ~200ms | ~20MB |
| 报告生成 | ~2s | ~50MB |

## 总结

Module 06 监控告警模块提供了全方位的系统监控和预警能力：

### 功能完整性 ✅
- ✓ 系统健康实时监控
- ✓ 性能指标追踪和分析
- ✓ 市场数据监控和预警
- ✓ 风险实时预警
- ✓ 多渠道通知服务
- ✓ 日志聚合和分析
- ✓ 定时报告生成

### 集成能力 ✅
- ✓ 与所有模块深度集成
- ✓ 实时数据推送
- ✓ REST API接口
- ✓ WebSocket推送

### 实用性 ✅
- ✓ 7×24小时持续监控
- ✓ 智能预警和通知
- ✓ 详细的历史数据
- ✓ 灵活的报告系统

**结论**: Module 06 提供了企业级的监控告警解决方案，确保系统稳定运行和及时响应异常。

