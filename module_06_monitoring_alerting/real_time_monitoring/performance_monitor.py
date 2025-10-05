"""
性能监控模块
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import json

from common.logging_system import setup_logger
from common.exceptions import QuantSystemError

logger = setup_logger("performance_monitor")

@dataclass
class SystemMetrics:
    """系统指标数据结构"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float
    active_threads: int
    python_memory_mb: float

@dataclass
class TradingMetrics:
    """交易指标数据结构"""
    timestamp: datetime
    total_signals: int
    active_orders: int
    filled_orders: int
    cancelled_orders: int
    total_volume: float
    total_pnl: float
    current_positions: int
    cash_balance: float
    portfolio_value: float

@dataclass
class AlertRule:
    """告警规则数据结构"""
    name: str
    metric_type: str  # 'system' or 'trading'
    metric_name: str
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    threshold: float
    duration_seconds: int = 60  # 持续时间
    enabled: bool = True
    callback: Optional[Callable] = None

@dataclass
class Alert:
    """告警数据结构"""
    alert_id: str
    rule_name: str
    message: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class PerformanceMonitor:
    """性能监控器类"""
    
    def __init__(self, monitoring_interval: int = 5):
        """初始化性能监控器
        
        Args:
            monitoring_interval: 监控间隔（秒）
        """
        self.monitoring_interval = monitoring_interval
        self.is_running = False
        self.monitor_thread = None
        
        # 指标存储
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.trading_metrics_history: deque = deque(maxlen=1000)
        
        # 告警系统
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # 网络统计基准
        self._network_baseline = None
        self._last_network_check = None
        
        # 回调函数
        self.metrics_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # 初始化网络基准
        self._init_network_baseline()
        
    def _init_network_baseline(self):
        """初始化网络统计基准"""
        try:
            net_io = psutil.net_io_counters()
            self._network_baseline = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }
            self._last_network_check = time.time()
        except Exception as e:
            logger.warning(f"Failed to initialize network baseline: {e}")
            self._network_baseline = {'bytes_sent': 0, 'bytes_recv': 0}
            self._last_network_check = time.time()
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_running:
            logger.warning("Performance monitoring is already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集系统指标
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # 收集交易指标
                trading_metrics = self._collect_trading_metrics()
                self.trading_metrics_history.append(trading_metrics)
                
                # 检查告警规则
                self._check_alert_rules(system_metrics, trading_metrics)
                
                # 调用回调函数
                self._notify_callbacks(system_metrics, trading_metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # 网络使用情况
            network_sent_mb = 0
            network_recv_mb = 0
            if self._network_baseline:
                try:
                    net_io = psutil.net_io_counters()
                    current_time = time.time()
                    time_diff = current_time - self._last_network_check
                    
                    if time_diff > 0:
                        sent_diff = net_io.bytes_sent - self._network_baseline['bytes_sent']
                        recv_diff = net_io.bytes_recv - self._network_baseline['bytes_recv']
                        
                        network_sent_mb = (sent_diff / time_diff) / (1024 * 1024)
                        network_recv_mb = (recv_diff / time_diff) / (1024 * 1024)
                        
                        # 更新基准
                        self._network_baseline = {
                            'bytes_sent': net_io.bytes_sent,
                            'bytes_recv': net_io.bytes_recv
                        }
                        self._last_network_check = current_time
                except Exception as e:
                    logger.warning(f"Failed to collect network metrics: {e}")
            
            # 活跃线程数
            active_threads = threading.active_count()
            
            # Python进程内存使用
            process = psutil.Process()
            python_memory_mb = process.memory_info().rss / (1024 * 1024)
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                active_threads=active_threads,
                python_memory_mb=python_memory_mb
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                active_threads=0,
                python_memory_mb=0.0
            )
    
    def _collect_trading_metrics(self) -> TradingMetrics:
        """收集交易指标"""
        try:
            # 这里应该从实际的交易系统获取数据
            # 目前返回模拟数据
            return TradingMetrics(
                timestamp=datetime.now(),
                total_signals=0,
                active_orders=0,
                filled_orders=0,
                cancelled_orders=0,
                total_volume=0.0,
                total_pnl=0.0,
                current_positions=0,
                cash_balance=0.0,
                portfolio_value=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to collect trading metrics: {e}")
            return TradingMetrics(
                timestamp=datetime.now(),
                total_signals=0,
                active_orders=0,
                filled_orders=0,
                cancelled_orders=0,
                total_volume=0.0,
                total_pnl=0.0,
                current_positions=0,
                cash_balance=0.0,
                portfolio_value=0.0
            )
    
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则
        
        Args:
            rule: 告警规则
        """
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """移除告警规则
        
        Args:
            rule_name: 规则名称
        """
        self.alert_rules = [rule for rule in self.alert_rules if rule.name != rule_name]
        logger.info(f"Removed alert rule: {rule_name}")
    
    def _check_alert_rules(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics):
        """检查告警规则"""
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            try:
                # 获取指标值
                if rule.metric_type == 'system':
                    metric_value = getattr(system_metrics, rule.metric_name, None)
                elif rule.metric_type == 'trading':
                    metric_value = getattr(trading_metrics, rule.metric_name, None)
                else:
                    continue
                
                if metric_value is None:
                    continue
                
                # 检查条件
                condition_met = self._evaluate_condition(metric_value, rule.operator, rule.threshold)
                
                if condition_met:
                    self._trigger_alert(rule, metric_value)
                else:
                    self._resolve_alert(rule.name)
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule.name}: {e}")
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """评估条件"""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return abs(value - threshold) < 1e-6
        elif operator == '!=':
            return abs(value - threshold) >= 1e-6
        else:
            return False
    
    def _trigger_alert(self, rule: AlertRule, metric_value: float):
        """触发告警"""
        alert_id = f"{rule.name}_{rule.metric_name}"
        
        # 检查是否已存在活跃告警
        if alert_id in self.active_alerts:
            return
        
        # 创建新告警
        alert = Alert(
            alert_id=alert_id,
            rule_name=rule.name,
            message=f"{rule.metric_name} {rule.operator} {rule.threshold} (current: {metric_value:.2f})",
            severity="HIGH",  # 可以根据规则配置
            timestamp=datetime.now()
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Alert triggered: {alert.message}")
        
        # 调用告警回调
        if rule.callback:
            try:
                rule.callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # 调用全局告警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in global alert callback: {e}")
    
    def _resolve_alert(self, rule_name: str):
        """解决告警"""
        alert_id = f"{rule_name}_{rule_name.split('_')[0]}"  # 简化匹配逻辑
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            del self.active_alerts[alert_id]
            logger.info(f"Alert resolved: {alert.message}")
    
    def add_metrics_callback(self, callback: Callable):
        """添加指标回调函数
        
        Args:
            callback: 回调函数
        """
        self.metrics_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """添加告警回调函数
        
        Args:
            callback: 回调函数
        """
        self.alert_callbacks.append(callback)
    
    def _notify_callbacks(self, system_metrics: SystemMetrics, trading_metrics: TradingMetrics):
        """通知回调函数"""
        for callback in self.metrics_callbacks:
            try:
                callback(system_metrics, trading_metrics)
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
    
    def get_latest_system_metrics(self) -> Optional[SystemMetrics]:
        """获取最新的系统指标"""
        return self.system_metrics_history[-1] if self.system_metrics_history else None
    
    def get_latest_trading_metrics(self) -> Optional[TradingMetrics]:
        """获取最新的交易指标"""
        return self.trading_metrics_history[-1] if self.trading_metrics_history else None
    
    def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """获取指标摘要
        
        Args:
            minutes: 时间范围（分钟）
            
        Returns:
            指标摘要
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        # 过滤系统指标
        recent_system_metrics = [
            m for m in self.system_metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        # 过滤交易指标
        recent_trading_metrics = [
            m for m in self.trading_metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        summary = {
            "time_range_minutes": minutes,
            "system_metrics_count": len(recent_system_metrics),
            "trading_metrics_count": len(recent_trading_metrics),
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history)
        }
        
        if recent_system_metrics:
            summary["avg_cpu_percent"] = sum(m.cpu_percent for m in recent_system_metrics) / len(recent_system_metrics)
            summary["avg_memory_percent"] = sum(m.memory_percent for m in recent_system_metrics) / len(recent_system_metrics)
            summary["max_cpu_percent"] = max(m.cpu_percent for m in recent_system_metrics)
            summary["max_memory_percent"] = max(m.memory_percent for m in recent_system_metrics)
        
        return summary
    
    def export_metrics(self, filepath: str):
        """导出指标数据
        
        Args:
            filepath: 文件路径
        """
        try:
            data = {
                "system_metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "cpu_percent": m.cpu_percent,
                        "memory_percent": m.memory_percent,
                        "memory_used_mb": m.memory_used_mb,
                        "memory_available_mb": m.memory_available_mb,
                        "disk_usage_percent": m.disk_usage_percent,
                        "network_sent_mb": m.network_sent_mb,
                        "network_recv_mb": m.network_recv_mb,
                        "active_threads": m.active_threads,
                        "python_memory_mb": m.python_memory_mb
                    }
                    for m in self.system_metrics_history
                ],
                "trading_metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "total_signals": m.total_signals,
                        "active_orders": m.active_orders,
                        "filled_orders": m.filled_orders,
                        "cancelled_orders": m.cancelled_orders,
                        "total_volume": m.total_volume,
                        "total_pnl": m.total_pnl,
                        "current_positions": m.current_positions,
                        "cash_balance": m.cash_balance,
                        "portfolio_value": m.portfolio_value
                    }
                    for m in self.trading_metrics_history
                ],
                "alerts": [
                    {
                        "alert_id": a.alert_id,
                        "rule_name": a.rule_name,
                        "message": a.message,
                        "severity": a.severity,
                        "timestamp": a.timestamp.isoformat(),
                        "resolved": a.resolved,
                        "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None
                    }
                    for a in self.alert_history
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

# 全局性能监控器实例
_global_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor(monitoring_interval: int = 5) -> PerformanceMonitor:
    """获取全局性能监控器实例
    
    Args:
        monitoring_interval: 监控间隔（秒）
        
    Returns:
        性能监控器实例
    """
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor(monitoring_interval)
    return _global_performance_monitor
