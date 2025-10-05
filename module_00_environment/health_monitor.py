"""
健康监控模块
负责监控系统各个组件的健康状态
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import psutil
import torch
import numpy as np

from common.logging_system import setup_logger
from common.exceptions import QuantSystemError

logger = setup_logger("health_monitor")

class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class ComponentHealth:
    """组件健康状态数据类"""
    component_name: str
    status: HealthStatus
    message: str
    last_check: datetime
    response_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """系统整体健康状态"""
    timestamp: datetime
    overall_status: HealthStatus
    overall_score: float  # 0-100
    components: List[ComponentHealth]
    system_metrics: Dict[str, Any]
    recommendations: List[str]

class HealthMonitor:
    """健康监控器类"""
    
    def __init__(self, check_interval: int = 60):
        """初始化健康监控器
        
        Args:
            check_interval: 检查间隔（秒）
        """
        self.check_interval = check_interval
        self.components: Dict[str, callable] = {}
        self.last_check_time: Optional[datetime] = None
        self.health_history: List[SystemHealth] = []
        
        # 注册默认组件
        self._register_default_components()
        
    def _register_default_components(self):
        """注册默认监控组件"""
        self.register_component("system_resources", self._check_system_resources)
        self.register_component("python_environment", self._check_python_environment)
        self.register_component("gpu_status", self._check_gpu_status)
        self.register_component("database_connection", self._check_database_connection)
        self.register_component("api_endpoints", self._check_api_endpoints)
        self.register_component("model_availability", self._check_model_availability)
        
    def register_component(self, name: str, check_function: callable):
        """注册监控组件
        
        Args:
            name: 组件名称
            check_function: 检查函数
        """
        self.components[name] = check_function
        logger.info(f"Registered health check component: {name}")
        
    def unregister_component(self, name: str):
        """注销监控组件
        
        Args:
            name: 组件名称
        """
        if name in self.components:
            del self.components[name]
            logger.info(f"Unregistered health check component: {name}")
            
    async def check_all_components(self) -> SystemHealth:
        """检查所有组件健康状态
        
        Returns:
            系统健康状态
        """
        logger.info("Starting health check for all components...")
        start_time = time.time()
        
        components_health = []
        recommendations = []
        
        # 并行检查所有组件
        tasks = []
        for name, check_function in self.components.items():
            task = asyncio.create_task(self._check_component(name, check_function))
            tasks.append(task)
            
        # 等待所有检查完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, ComponentHealth):
                components_health.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Component check failed: {result}")
                components_health.append(ComponentHealth(
                    component_name="unknown",
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {str(result)}",
                    last_check=datetime.now(),
                    response_time_ms=0.0
                ))
                
        # 计算系统指标
        system_metrics = self._collect_system_metrics()
        
        # 计算整体健康分数
        overall_score = self._calculate_overall_score(components_health)
        overall_status = self._determine_overall_status(overall_score)
        
        # 生成建议
        recommendations = self._generate_recommendations(components_health, system_metrics)
        
        system_health = SystemHealth(
            timestamp=datetime.now(),
            overall_status=overall_status,
            overall_score=overall_score,
            components=components_health,
            system_metrics=system_metrics,
            recommendations=recommendations
        )
        
        # 保存到历史记录
        self.health_history.append(system_health)
        self.last_check_time = datetime.now()
        
        # 限制历史记录长度
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]
            
        check_duration = (time.time() - start_time) * 1000
        logger.info(f"Health check completed in {check_duration:.2f}ms. Overall score: {overall_score:.1f}")
        
        return system_health
        
    async def _check_component(self, name: str, check_function: callable) -> ComponentHealth:
        """检查单个组件
        
        Args:
            name: 组件名称
            check_function: 检查函数
            
        Returns:
            组件健康状态
        """
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(check_function):
                result = await check_function()
            else:
                result = check_function()
                
            response_time = (time.time() - start_time) * 1000
            
            if isinstance(result, ComponentHealth):
                result.response_time_ms = response_time
                return result
            else:
                # 如果返回的不是ComponentHealth对象，包装它
                return ComponentHealth(
                    component_name=name,
                    status=HealthStatus.HEALTHY,
                    message="Check completed successfully",
                    last_check=datetime.now(),
                    response_time_ms=response_time,
                    metadata={"raw_result": result}
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Health check failed for {name}: {e}")
            
            return ComponentHealth(
                component_name=name,
                status=HealthStatus.CRITICAL,
                message=f"Check failed: {str(e)}",
                last_check=datetime.now(),
                response_time_ms=response_time
            )
            
    def _check_system_resources(self) -> ComponentHealth:
        """检查系统资源"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # 判断状态
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%"
            elif cpu_percent > 70 or memory_percent > 80 or disk_percent > 85:
                status = HealthStatus.WARNING
                message = f"Moderate resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Resource usage normal: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%"
                
            return ComponentHealth(
                component_name="system_resources",
                status=status,
                message=message,
                last_check=datetime.now(),
                response_time_ms=0.0,
                metadata={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "available_memory_gb": memory.available / (1024**3),
                    "available_disk_gb": disk.free / (1024**3)
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check system resources: {str(e)}",
                last_check=datetime.now(),
                response_time_ms=0.0
            )
            
    def _check_python_environment(self) -> ComponentHealth:
        """检查Python环境"""
        try:
            import sys
            import importlib
            
            # 检查Python版本
            python_version = sys.version_info
            if python_version < (3, 9):
                status = HealthStatus.CRITICAL
                message = f"Python version {python_version} is below minimum requirement (3.9)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Python version {python_version} is compatible"
                
            # 检查关键包
            critical_packages = ['numpy', 'pandas', 'torch', 'fastapi']
            missing_packages = []
            
            for package in critical_packages:
                try:
                    importlib.import_module(package)
                except ImportError:
                    missing_packages.append(package)
                    
            if missing_packages:
                status = HealthStatus.CRITICAL
                message += f". Missing packages: {missing_packages}"
                
            return ComponentHealth(
                component_name="python_environment",
                status=status,
                message=message,
                last_check=datetime.now(),
                response_time_ms=0.0,
                metadata={
                    "python_version": str(python_version),
                    "missing_packages": missing_packages
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="python_environment",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check Python environment: {str(e)}",
                last_check=datetime.now(),
                response_time_ms=0.0
            )
            
    def _check_gpu_status(self) -> ComponentHealth:
        """检查GPU状态"""
        try:
            if not torch.cuda.is_available():
                return ComponentHealth(
                    component_name="gpu_status",
                    status=HealthStatus.WARNING,
                    message="CUDA not available, using CPU",
                    last_check=datetime.now(),
                    response_time_ms=0.0,
                    metadata={"cuda_available": False}
                )
                
            # 检查GPU内存
            gpu_count = torch.cuda.device_count()
            gpu_info = []
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory / (1024**3)  # GB
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                memory_free = memory_total - memory_allocated
                
                gpu_info.append({
                    "device_id": i,
                    "name": props.name,
                    "memory_total_gb": memory_total,
                    "memory_allocated_gb": memory_allocated,
                    "memory_cached_gb": memory_cached,
                    "memory_free_gb": memory_free,
                    "utilization": (memory_allocated / memory_total) * 100
                })
                
            # 检查GPU利用率
            max_utilization = max(info["utilization"] for info in gpu_info)
            
            if max_utilization > 95:
                status = HealthStatus.CRITICAL
                message = f"GPU memory usage critical: {max_utilization:.1f}%"
            elif max_utilization > 80:
                status = HealthStatus.WARNING
                message = f"GPU memory usage high: {max_utilization:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"GPU status normal: {gpu_count} device(s) available"
                
            return ComponentHealth(
                component_name="gpu_status",
                status=status,
                message=message,
                last_check=datetime.now(),
                response_time_ms=0.0,
                metadata={
                    "cuda_available": True,
                    "gpu_count": gpu_count,
                    "gpu_info": gpu_info
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="gpu_status",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check GPU status: {str(e)}",
                last_check=datetime.now(),
                response_time_ms=0.0
            )
            
    def _check_database_connection(self) -> ComponentHealth:
        """检查数据库连接"""
        try:
            import duckdb
            
            # 尝试连接数据库
            conn = duckdb.connect(":memory:")
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] == 1:
                return ComponentHealth(
                    component_name="database_connection",
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                    last_check=datetime.now(),
                    response_time_ms=0.0
                )
            else:
                return ComponentHealth(
                    component_name="database_connection",
                    status=HealthStatus.CRITICAL,
                    message="Database connection test failed",
                    last_check=datetime.now(),
                    response_time_ms=0.0
                )
                
        except Exception as e:
            return ComponentHealth(
                component_name="database_connection",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                last_check=datetime.now(),
                response_time_ms=0.0
            )
            
    def _check_api_endpoints(self) -> ComponentHealth:
        """检查API端点"""
        try:
            import requests
            
            # 检查外部API端点
            endpoints = {
                "yahoo_finance": "https://query1.finance.yahoo.com/v8/finance/chart/AAPL",
                "binance": "https://api.binance.com/api/v3/ping"
            }
            
            available_endpoints = 0
            total_endpoints = len(endpoints)
            
            for name, url in endpoints.items():
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code in [200, 401, 403]:  # 401/403表示可达但需认证
                        available_endpoints += 1
                except:
                    pass
                    
            success_rate = available_endpoints / total_endpoints
            
            if success_rate >= 0.8:
                status = HealthStatus.HEALTHY
                message = f"API endpoints available: {available_endpoints}/{total_endpoints}"
            elif success_rate >= 0.5:
                status = HealthStatus.WARNING
                message = f"Some API endpoints unavailable: {available_endpoints}/{total_endpoints}"
            else:
                status = HealthStatus.CRITICAL
                message = f"Most API endpoints unavailable: {available_endpoints}/{total_endpoints}"
                
            return ComponentHealth(
                component_name="api_endpoints",
                status=status,
                message=message,
                last_check=datetime.now(),
                response_time_ms=0.0,
                metadata={
                    "available_endpoints": available_endpoints,
                    "total_endpoints": total_endpoints,
                    "success_rate": success_rate
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="api_endpoints",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check API endpoints: {str(e)}",
                last_check=datetime.now(),
                response_time_ms=0.0
            )
            
    def _check_model_availability(self) -> ComponentHealth:
        """检查模型可用性"""
        try:
            # 这里应该检查FIN-R1模型是否可用
            # 由于模型文件可能不存在，这里返回警告状态
            return ComponentHealth(
                component_name="model_availability",
                status=HealthStatus.WARNING,
                message="FIN-R1 model not configured (this is expected in development)",
                last_check=datetime.now(),
                response_time_ms=0.0,
                metadata={"model_configured": False}
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="model_availability",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check model availability: {str(e)}",
                last_check=datetime.now(),
                response_time_ms=0.0
            )
            
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        try:
            # 系统负载
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            # 网络统计
            net_io = psutil.net_io_counters()
            
            # 磁盘IO
            disk_io = psutil.disk_io_counters()
            
            return {
                "load_average": load_avg,
                "network_bytes_sent": net_io.bytes_sent if net_io else 0,
                "network_bytes_recv": net_io.bytes_recv if net_io else 0,
                "disk_read_bytes": disk_io.read_bytes if disk_io else 0,
                "disk_write_bytes": disk_io.write_bytes if disk_io else 0,
                "uptime_seconds": time.time() - psutil.boot_time() if hasattr(psutil, 'boot_time') else 0
            }
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            return {}
            
    def _calculate_overall_score(self, components: List[ComponentHealth]) -> float:
        """计算整体健康分数
        
        Args:
            components: 组件健康状态列表
            
        Returns:
            0-100的健康分数
        """
        if not components:
            return 0.0
            
        # 权重映射
        weights = {
            "system_resources": 0.25,
            "python_environment": 0.20,
            "gpu_status": 0.15,
            "database_connection": 0.15,
            "api_endpoints": 0.15,
            "model_availability": 0.10
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for component in components:
            weight = weights.get(component.component_name, 0.1)
            total_weight += weight
            
            # 状态到分数的映射
            status_scores = {
                HealthStatus.HEALTHY: 100.0,
                HealthStatus.WARNING: 60.0,
                HealthStatus.CRITICAL: 20.0,
                HealthStatus.UNKNOWN: 50.0
            }
            
            score = status_scores.get(component.status, 50.0)
            total_score += score * weight
            
        return total_score / total_weight if total_weight > 0 else 0.0
        
    def _determine_overall_status(self, score: float) -> HealthStatus:
        """根据分数确定整体状态
        
        Args:
            score: 健康分数
            
        Returns:
            健康状态
        """
        if score >= 80:
            return HealthStatus.HEALTHY
        elif score >= 60:
            return HealthStatus.WARNING
        elif score >= 30:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.UNKNOWN
            
    def _generate_recommendations(self, components: List[ComponentHealth], system_metrics: Dict[str, Any]) -> List[str]:
        """生成改进建议
        
        Args:
            components: 组件健康状态列表
            system_metrics: 系统指标
            
        Returns:
            建议列表
        """
        recommendations = []
        
        for component in components:
            if component.status == HealthStatus.CRITICAL:
                if component.component_name == "system_resources":
                    recommendations.append("Consider upgrading hardware or optimizing resource usage")
                elif component.component_name == "python_environment":
                    recommendations.append("Update Python version or install missing packages")
                elif component.component_name == "gpu_status":
                    recommendations.append("Check GPU drivers and memory usage")
                elif component.component_name == "database_connection":
                    recommendations.append("Check database configuration and connectivity")
                elif component.component_name == "api_endpoints":
                    recommendations.append("Check network connectivity and API access")
                    
            elif component.status == HealthStatus.WARNING:
                if component.component_name == "system_resources":
                    recommendations.append("Monitor resource usage and consider optimization")
                elif component.component_name == "gpu_status":
                    recommendations.append("GPU available but not optimal for performance")
                elif component.component_name == "api_endpoints":
                    recommendations.append("Some API endpoints may be temporarily unavailable")
                    
        return recommendations
        
    def get_health_history(self, hours: int = 24) -> List[SystemHealth]:
        """获取健康历史记录
        
        Args:
            hours: 获取最近多少小时的历史
            
        Returns:
            健康历史记录列表
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [health for health in self.health_history if health.timestamp >= cutoff_time]
        
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康摘要
        
        Returns:
            健康摘要字典
        """
        if not self.health_history:
            return {"message": "No health check history available"}
            
        latest_health = self.health_history[-1]
        recent_health = self.get_health_history(hours=1)
        
        # 计算平均分数
        avg_score = sum(h.overall_score for h in recent_health) / len(recent_health)
        
        # 统计状态分布
        status_counts = {}
        for health in recent_health:
            status = health.overall_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
        return {
            "latest_score": latest_health.overall_score,
            "latest_status": latest_health.overall_status.value,
            "average_score_1h": avg_score,
            "status_distribution_1h": status_counts,
            "total_checks": len(self.health_history),
            "last_check": latest_health.timestamp.isoformat(),
            "recommendations": latest_health.recommendations
        }

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

async def run_health_check() -> SystemHealth:
    """运行健康检查的便捷函数
    
    Returns:
        系统健康状态
    """
    monitor = get_health_monitor()
    return await monitor.check_all_components()