"""
系统监控器模块
监控系统资源和健康状态
"""

import asyncio
import platform
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import psutil

from common.logging_system import setup_logger

logger = setup_logger("system_monitor")


@dataclass
class SystemStatus:
    """系统状态数据结构"""

    timestamp: datetime
    cpu_percent: float
    cpu_count: int
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_sent_mb: float
    network_recv_mb: float
    boot_time: datetime
    uptime_hours: float
    process_count: int
    python_version: str
    os_info: str


@dataclass
class HealthCheckResult:
    """健康检查结果"""

    component: str
    is_healthy: bool
    status: str
    message: str
    timestamp: datetime
    metrics: Dict[str, Any]


class SystemMonitor:
    """系统监控器类"""

    # 阈值
    CPU_WARNING_THRESHOLD = 70.0
    CPU_CRITICAL_THRESHOLD = 85.0
    MEMORY_WARNING_THRESHOLD = 75.0
    MEMORY_CRITICAL_THRESHOLD = 90.0
    DISK_WARNING_THRESHOLD = 80.0
    DISK_CRITICAL_THRESHOLD = 95.0

    def __init__(self, monitoring_interval: int = 60):
        """初始化系统监控器

        Args:
            monitoring_interval: 监控间隔（秒）
        """
        self.monitoring_interval = monitoring_interval
        self.is_running = False
        self.status_history: deque = deque(maxlen=1000)
        self.health_checks: Dict[str, HealthCheckResult] = {}
        self.callbacks: List[Callable] = []

        # 网络基准
        self._network_baseline = None
        self._last_network_check = None
        self._init_network_baseline()

    def _init_network_baseline(self):
        """初始化网络基准"""
        try:
            net_io = psutil.net_io_counters()
            self._network_baseline = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "time": datetime.now(),
            }
            self._last_network_check = datetime.now()
        except Exception as e:
            logger.warning(f"Failed to initialize network baseline: {e}")
            self._network_baseline = {
                "bytes_sent": 0,
                "bytes_recv": 0,
                "time": datetime.now(),
            }
            self._last_network_check = datetime.now()

    async def start_monitoring(self):
        """启动监控"""
        if self.is_running:
            logger.warning("System monitoring is already running")
            return

        self.is_running = True
        logger.info("Starting system monitoring")

        while self.is_running:
            try:
                # 收集系统状态
                status = self.get_system_status()
                self.status_history.append(status)

                # 执行健康检查
                self._perform_health_checks(status)

                # 触发回调
                await self._trigger_callbacks(status)

                # 等待下一个周期
                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)

    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        logger.info("System monitoring stopped")

    def get_system_status(self) -> SystemStatus:
        """获取系统状态

        Returns:
            系统状态对象
        """
        try:
            # CPU信息
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # 内存信息
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)

            # 磁盘信息
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)

            # 网络信息
            network_sent_mb, network_recv_mb = self._get_network_stats()

            # 启动时间和运行时长
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime_hours = (datetime.now() - boot_time).total_seconds() / 3600

            # 进程数量
            process_count = len(psutil.pids())

            # 系统信息
            python_version = platform.python_version()
            os_info = f"{platform.system()} {platform.release()}"

            status = SystemStatus(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                boot_time=boot_time,
                uptime_hours=uptime_hours,
                process_count=process_count,
                python_version=python_version,
                os_info=os_info,
            )

            return status

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            # 返回默认状态
            return SystemStatus(
                timestamp=datetime.now(),
                cpu_percent=0,
                cpu_count=0,
                memory_percent=0,
                memory_used_gb=0,
                memory_total_gb=0,
                disk_percent=0,
                disk_used_gb=0,
                disk_total_gb=0,
                network_sent_mb=0,
                network_recv_mb=0,
                boot_time=datetime.now(),
                uptime_hours=0,
                process_count=0,
                python_version="unknown",
                os_info="unknown",
            )

    def _get_network_stats(self) -> tuple:
        """获取网络统计

        Returns:
            (发送MB速率, 接收MB速率)
        """
        try:
            if not self._network_baseline:
                return 0.0, 0.0

            net_io = psutil.net_io_counters()
            current_time = datetime.now()

            time_diff = (current_time - self._last_network_check).total_seconds()

            if time_diff > 0:
                sent_diff = net_io.bytes_sent - self._network_baseline["bytes_sent"]
                recv_diff = net_io.bytes_recv - self._network_baseline["bytes_recv"]

                sent_mb_per_sec = (sent_diff / time_diff) / (1024 * 1024)
                recv_mb_per_sec = (recv_diff / time_diff) / (1024 * 1024)

                # 更新基准
                self._network_baseline = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "time": current_time,
                }
                self._last_network_check = current_time

                return sent_mb_per_sec, recv_mb_per_sec
            else:
                return 0.0, 0.0

        except Exception as e:
            logger.warning(f"Failed to get network stats: {e}")
            return 0.0, 0.0

    def _perform_health_checks(self, status: SystemStatus):
        """执行健康检查

        Args:
            status: 系统状态
        """
        # CPU健康检查
        if status.cpu_percent >= self.CPU_CRITICAL_THRESHOLD:
            cpu_health = HealthCheckResult(
                component="cpu",
                is_healthy=False,
                status="critical",
                message=f"CPU usage critical: {status.cpu_percent:.1f}%",
                timestamp=datetime.now(),
                metrics={"cpu_percent": status.cpu_percent},
            )
        elif status.cpu_percent >= self.CPU_WARNING_THRESHOLD:
            cpu_health = HealthCheckResult(
                component="cpu",
                is_healthy=True,
                status="warning",
                message=f"CPU usage high: {status.cpu_percent:.1f}%",
                timestamp=datetime.now(),
                metrics={"cpu_percent": status.cpu_percent},
            )
        else:
            cpu_health = HealthCheckResult(
                component="cpu",
                is_healthy=True,
                status="healthy",
                message=f"CPU usage normal: {status.cpu_percent:.1f}%",
                timestamp=datetime.now(),
                metrics={"cpu_percent": status.cpu_percent},
            )

        self.health_checks["cpu"] = cpu_health

        # 内存健康检查
        if status.memory_percent >= self.MEMORY_CRITICAL_THRESHOLD:
            memory_health = HealthCheckResult(
                component="memory",
                is_healthy=False,
                status="critical",
                message=f"Memory usage critical: {status.memory_percent:.1f}%",
                timestamp=datetime.now(),
                metrics={"memory_percent": status.memory_percent},
            )
        elif status.memory_percent >= self.MEMORY_WARNING_THRESHOLD:
            memory_health = HealthCheckResult(
                component="memory",
                is_healthy=True,
                status="warning",
                message=f"Memory usage high: {status.memory_percent:.1f}%",
                timestamp=datetime.now(),
                metrics={"memory_percent": status.memory_percent},
            )
        else:
            memory_health = HealthCheckResult(
                component="memory",
                is_healthy=True,
                status="healthy",
                message=f"Memory usage normal: {status.memory_percent:.1f}%",
                timestamp=datetime.now(),
                metrics={"memory_percent": status.memory_percent},
            )

        self.health_checks["memory"] = memory_health

        # 磁盘健康检查
        if status.disk_percent >= self.DISK_CRITICAL_THRESHOLD:
            disk_health = HealthCheckResult(
                component="disk",
                is_healthy=False,
                status="critical",
                message=f"Disk usage critical: {status.disk_percent:.1f}%",
                timestamp=datetime.now(),
                metrics={"disk_percent": status.disk_percent},
            )
        elif status.disk_percent >= self.DISK_WARNING_THRESHOLD:
            disk_health = HealthCheckResult(
                component="disk",
                is_healthy=True,
                status="warning",
                message=f"Disk usage high: {status.disk_percent:.1f}%",
                timestamp=datetime.now(),
                metrics={"disk_percent": status.disk_percent},
            )
        else:
            disk_health = HealthCheckResult(
                component="disk",
                is_healthy=True,
                status="healthy",
                message=f"Disk usage normal: {status.disk_percent:.1f}%",
                timestamp=datetime.now(),
                metrics={"disk_percent": status.disk_percent},
            )

        self.health_checks["disk"] = disk_health

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态摘要

        Returns:
            健康状态字典
        """
        if not self.health_checks:
            return {"overall_status": "unknown", "components": {}}

        # 判断整体状态
        critical_count = sum(
            1 for hc in self.health_checks.values() if hc.status == "critical"
        )
        warning_count = sum(
            1 for hc in self.health_checks.values() if hc.status == "warning"
        )

        if critical_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        return {
            "overall_status": overall_status,
            "timestamp": datetime.now(),
            "components": {
                name: {
                    "status": hc.status,
                    "message": hc.message,
                    "is_healthy": hc.is_healthy,
                    "metrics": hc.metrics,
                }
                for name, hc in self.health_checks.items()
            },
        }

    def get_statistics(self, minutes: int = 60) -> Dict[str, Any]:
        """获取统计信息

        Args:
            minutes: 时间范围（分钟）

        Returns:
            统计字典
        """
        if not self.status_history:
            return {}

        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        recent_statuses = [s for s in self.status_history if s.timestamp >= cutoff_time]

        if not recent_statuses:
            return {}

        return {
            "period_minutes": minutes,
            "sample_count": len(recent_statuses),
            "cpu": {
                "avg": sum(s.cpu_percent for s in recent_statuses)
                / len(recent_statuses),
                "max": max(s.cpu_percent for s in recent_statuses),
                "min": min(s.cpu_percent for s in recent_statuses),
            },
            "memory": {
                "avg": sum(s.memory_percent for s in recent_statuses)
                / len(recent_statuses),
                "max": max(s.memory_percent for s in recent_statuses),
                "min": min(s.memory_percent for s in recent_statuses),
            },
            "disk": {
                "avg": sum(s.disk_percent for s in recent_statuses)
                / len(recent_statuses),
                "max": max(s.disk_percent for s in recent_statuses),
                "min": min(s.disk_percent for s in recent_statuses),
            },
        }

    def register_callback(self, callback: Callable):
        """注册状态变化回调

        Args:
            callback: 回调函数
        """
        self.callbacks.append(callback)
        logger.info("Registered system monitor callback")

    async def _trigger_callbacks(self, status: SystemStatus):
        """触发回调函数

        Args:
            status: 系统状态
        """
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(status)
                else:
                    callback(status)
            except Exception as e:
                logger.error(f"Error in callback: {e}")


# 全局实例
_global_system_monitor: Optional[SystemMonitor] = None


def get_system_monitor(monitoring_interval: int = 60) -> SystemMonitor:
    """获取全局系统监控器实例

    Args:
        monitoring_interval: 监控间隔（秒）

    Returns:
        系统监控器实例
    """
    global _global_system_monitor
    if _global_system_monitor is None:
        _global_system_monitor = SystemMonitor(monitoring_interval)
    return _global_system_monitor
