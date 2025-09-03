"""
健康监控器模块
负责持续监控系统健康状态
"""

import os
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import psutil
from common.logging_system import setup_logger
from common.exceptions import QuantSystemError

logger = setup_logger("health_monitor")

@dataclass
class HealthMetrics:
    """健康指标数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io_mbps: float
    process_count: int
    thread_count: int
    open_files: int
    error_count: int
    warning_count: int
    
@dataclass
class HealthStatus:
    """健康状态数据类"""
    status: str  # 'HEALTHY', 'WARNING', 'CRITICAL'
    metrics: HealthMetrics
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class HealthMonitor:
    """健康监控器类"""
    
    # 阈值定义
    THRESHOLDS = {
        'cpu_percent': {'warning': 70, 'critical': 90},
        'memory_percent': {'warning': 80, 'critical': 95},
        'disk_percent': {'warning': 85, 'critical': 95},
        'error_count': {'warning': 10, 'critical': 50}
    }
    
    def __init__(self, check_interval: int = 60):
        """初始化健康监控器
        
        Args:
            check_interval: 检查间隔（秒）
        """
        self.check_interval = check_interval
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history: List[HealthMetrics] = []
        self.max_history_size = 1440  # 保留24小时的数据（按分钟计）
        self.callbacks: Dict[str, List[Callable]] = {
            'on_warning': [],
            'on_critical': [],
            'on_recovery': []
        }
        self.last_status = 'HEALTHY'
        
    def start_monitoring(self) -> None:
        """启动监控"""
        if self.is_running:
            logger.warning("Health monitor is already running")
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
        
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
        
    def get_current_status(self) -> HealthStatus:
        """获取当前健康状态
        
        Returns:
            健康状态对象
        """
        metrics = self._collect_metrics()
        status = self._evaluate_health(metrics)
        return status
        
    def get_metrics_history(
        self, 
        duration_minutes: int = 60
    ) -> List[HealthMetrics]:
        """获取历史指标
        
        Args:
            duration_minutes: 历史时长（分钟）
            
        Returns:
            历史指标列表
        """
        if not self.metrics_history:
            return []
            
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
    def register_callback(
        self, 
        event_type: str, 
        callback: Callable[[HealthStatus], None]
    ) -> None:
        """注册回调函数
        
        Args:
            event_type: 事件类型 ('on_warning', 'on_critical', 'on_recovery')
            callback: 回调函数
        """
        if event_type not in self.callbacks:
            raise ValueError(f"Invalid event type: {event_type}")
        self.callbacks[event_type].append(callback)
        
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.is_running:
            try:
                status = self.get_current_status()
                self._handle_status_change(status)
                
                # 保存指标历史
                self.metrics_history.append(status.metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
                    
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                time.sleep(self.check_interval)
                
    def _collect_metrics(self) -> HealthMetrics:
        """收集系统指标
        
        Returns:
            健康指标对象
        """
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # 网络IO
        net_io = psutil.net_io_counters()
        network_io_mbps = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)
        
        # 进程信息
        current_process = psutil.Process()
        process_count = len(psutil.pids())
        thread_count = current_process.num_threads()
        
        try:
            open_files = len(current_process.open_files())
        except:
            open_files = 0
            
        # 错误统计（从日志文件读取）
        error_count = self._count_log_errors()
        warning_count = self._count_log_warnings()
        
        return HealthMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            network_io_mbps=network_io_mbps,
            process_count=process_count,
            thread_count=thread_count,
            open_files=open_files,
            error_count=error_count,
            warning_count=warning_count
        )
        
    def _evaluate_health(self, metrics: HealthMetrics) -> HealthStatus:
        """评估健康状态
        
        Args:
            metrics: 健康指标
            
        Returns:
            健康状态对象
        """
        status = 'HEALTHY'
        issues = []
        recommendations = []
        
        # 检查CPU
        if metrics.cpu_percent >= self.THRESHOLDS['cpu_percent']['critical']:
            status = 'CRITICAL'
            issues.append(f"CPU usage critical: {metrics.cpu_percent:.1f}%")
            recommendations.append("Consider scaling up compute resources")
        elif metrics.cpu_percent >= self.THRESHOLDS['cpu_percent']['warning']:
            if status != 'CRITICAL':
                status = 'WARNING'
            issues.append(f"CPU usage high: {metrics.cpu_percent:.1f}%")
            recommendations.append("Monitor CPU usage closely")
            
        # 检查内存
        if metrics.memory_percent >= self.THRESHOLDS['memory_percent']['critical']:
            status = 'CRITICAL'
            issues.append(f"Memory usage critical: {metrics.memory_percent:.1f}%")
            recommendations.append("Increase system memory or optimize memory usage")
        elif metrics.memory_percent >= self.THRESHOLDS['memory_percent']['warning']:
            if status != 'CRITICAL':
                status = 'WARNING'
            issues.append(f"Memory usage high: {metrics.memory_percent:.1f}%")
            recommendations.append("Consider memory optimization")
            
        # 检查磁盘
        if metrics.disk_percent >= self.THRESHOLDS['disk_percent']['critical']:
            status = 'CRITICAL'
            issues.append(f"Disk usage critical: {metrics.disk_percent:.1f}%")
            recommendations.append("Free up disk space immediately")
        elif metrics.disk_percent >= self.THRESHOLDS['disk_percent']['warning']:
            if status != 'CRITICAL':
                status = 'WARNING'
            issues.append(f"Disk usage high: {metrics.disk_percent:.1f}%")
            recommendations.append("Clean up old logs and data files")
            
        # 检查错误数
        if metrics.error_count >= self.THRESHOLDS['error_count']['critical']:
            status = 'CRITICAL'
            issues.append(f"High error count: {metrics.error_count}")
            recommendations.append("Investigate error logs immediately")
        elif metrics.error_count >= self.THRESHOLDS['error_count']['warning']:
            if status != 'CRITICAL':
                status = 'WARNING'
            issues.append(f"Elevated error count: {metrics.error_count}")
            recommendations.append("Review recent errors")
            
        return HealthStatus(
            status=status,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations
        )
        
    def _handle_status_change(self, status: HealthStatus) -> None:
        """处理状态变化
        
        Args:
            status: 当前状态
        """
        if status.status != self.last_status:
            logger.info(f"Health status changed: {self.last_status} -> {status.status}")
            
            # 触发相应的回调
            if status.status == 'WARNING' and self.last_status == 'HEALTHY':
                for callback in self.callbacks['on_warning']:
                    try:
                        callback(status)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                        
            elif status.status == 'CRITICAL':
                for callback in self.callbacks['on_critical']:
                    try:
                        callback(status)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                        
            elif status.status == 'HEALTHY' and self.last_status != 'HEALTHY':
                for callback in self.callbacks['on_recovery']:
                    try:
                        callback(status)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                        
            self.last_status = status.status
            
    def _count_log_errors(self) -> int:
        """统计日志错误数
        
        Returns:
            最近一小时的错误数
        """
        # 实际实现应该从日志系统获取
        # 这里返回模拟值
        return 0
        
    def _count_log_warnings(self) -> int:
        """统计日志警告数
        
        Returns:
            最近一小时的警告数
        """
        # 实际实现应该从日志系统获取
        # 这里返回模拟值
        return 0

# 全局健康监控器实例
_global_monitor: Optional[HealthMonitor] = None

def get_health_monitor() -> HealthMonitor:
    """获取全局健康监控器实例
    
    Returns:
        健康监控器实例
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = HealthMonitor()
    return _global_monitor

def start_health_monitoring() -> None:
    """启动全局健康监控"""
    monitor = get_health_monitor()
    monitor.start_monitoring()

def stop_health_monitoring() -> None:
    """停止全局健康监控"""
    monitor = get_health_monitor()
    monitor.stop_monitoring()