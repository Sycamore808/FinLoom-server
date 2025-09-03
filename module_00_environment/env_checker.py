"""
环境检测器模块
负责检测和验证系统运行环境
"""

import os
import sys
import platform
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import psutil
import torch
import numpy as np
from dataclasses import dataclass
from common.logging_system import setup_logger
from common.exceptions import QuantSystemError

logger = setup_logger("env_checker")

@dataclass
class SystemInfo:
    """系统信息数据类"""
    os_name: str
    os_version: str
    architecture: str
    cpu_count: int
    cpu_freq_mhz: float
    total_memory_gb: float
    available_memory_gb: float
    disk_total_gb: float
    disk_available_gb: float
    
@dataclass
class PythonInfo:
    """Python环境信息"""
    version: str
    executable: str
    virtual_env: Optional[str]
    site_packages: List[str]
    
@dataclass
class GPUInfo:
    """GPU信息数据类"""
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_count: int
    gpu_names: List[str]
    gpu_memory_mb: List[int]
    driver_version: Optional[str]

@dataclass
class NetworkInfo:
    """网络连接信息"""
    internet_connected: bool
    latency_ms: Dict[str, float]
    api_endpoints: Dict[str, bool]

@dataclass
class EnvironmentReport:
    """环境检测完整报告"""
    timestamp: datetime
    system_info: SystemInfo
    python_info: PythonInfo
    gpu_info: GPUInfo
    network_info: NetworkInfo
    dependencies_status: Dict[str, str]
    health_score: float
    issues: List[str]
    recommendations: List[str]

class EnvironmentChecker:
    """环境检测器类"""
    
    REQUIRED_PYTHON_VERSION = (3, 9, 0)
    MIN_MEMORY_GB = 8.0
    MIN_DISK_GB = 50.0
    REQUIRED_PACKAGES = {
        'numpy': '1.24.0',
        'pandas': '2.0.0',
        'polars': '0.19.0',
        'torch': '2.0.0',
        'transformers': '4.35.0',
        'scikit-learn': '1.3.0',
        'fastapi': '0.104.0',
        'pydantic': '2.4.0'
    }
    API_ENDPOINTS = {
        'yahoo_finance': 'https://query1.finance.yahoo.com/v8/finance/chart/AAPL',
        'binance': 'https://api.binance.com/api/v3/ping',
        'polygon': 'https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-01'
    }
    
    def __init__(self):
        """初始化环境检测器"""
        self.issues: List[str] = []
        self.recommendations: List[str] = []
        
    def check_all(self) -> EnvironmentReport:
        """执行完整环境检测
        
        Returns:
            环境检测报告
        """
        logger.info("Starting environment check...")
        
        system_info = self._check_system()
        python_info = self._check_python()
        gpu_info = self._check_gpu()
        network_info = self._check_network()
        dependencies_status = self._check_dependencies()
        
        health_score = self._calculate_health_score(
            system_info, python_info, gpu_info, 
            network_info, dependencies_status
        )
        
        report = EnvironmentReport(
            timestamp=datetime.now(),
            system_info=system_info,
            python_info=python_info,
            gpu_info=gpu_info,
            network_info=network_info,
            dependencies_status=dependencies_status,
            health_score=health_score,
            issues=self.issues.copy(),
            recommendations=self.recommendations.copy()
        )
        
        logger.info(f"Environment check completed. Health score: {health_score:.2f}")
        return report
    
    def _check_system(self) -> SystemInfo:
        """检查系统信息
        
        Returns:
            系统信息对象
        """
        try:
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_info = SystemInfo(
                os_name=platform.system(),
                os_version=platform.version(),
                architecture=platform.machine(),
                cpu_count=psutil.cpu_count(),
                cpu_freq_mhz=cpu_freq.current if cpu_freq else 0.0,
                total_memory_gb=memory.total / (1024**3),
                available_memory_gb=memory.available / (1024**3),
                disk_total_gb=disk.total / (1024**3),
                disk_available_gb=disk.free / (1024**3)
            )
            
            # 检查内存
            if system_info.available_memory_gb < self.MIN_MEMORY_GB:
                self.issues.append(f"Insufficient memory: {system_info.available_memory_gb:.2f} GB available")
                self.recommendations.append(f"Recommend at least {self.MIN_MEMORY_GB} GB available memory")
            
            # 检查磁盘
            if system_info.disk_available_gb < self.MIN_DISK_GB:
                self.issues.append(f"Insufficient disk space: {system_info.disk_available_gb:.2f} GB available")
                self.recommendations.append(f"Recommend at least {self.MIN_DISK_GB} GB available disk space")
                
            return system_info
            
        except Exception as e:
            logger.error(f"Failed to check system info: {e}")
            raise QuantSystemError(f"System check failed: {e}")
    
    def _check_python(self) -> PythonInfo:
        """检查Python环境
        
        Returns:
            Python环境信息
        """
        current_version = sys.version_info[:3]
        
        if current_version < self.REQUIRED_PYTHON_VERSION:
            self.issues.append(
                f"Python version {'.'.join(map(str, current_version))} is below required "
                f"{'.'.join(map(str, self.REQUIRED_PYTHON_VERSION))}"
            )
            self.recommendations.append(
                f"Upgrade to Python {'.'.join(map(str, self.REQUIRED_PYTHON_VERSION))} or higher"
            )
        
        return PythonInfo(
            version='.'.join(map(str, current_version)),
            executable=sys.executable,
            virtual_env=os.environ.get('VIRTUAL_ENV'),
            site_packages=sys.path
        )
    
    def _check_gpu(self) -> GPUInfo:
        """检查GPU和CUDA环境
        
        Returns:
            GPU信息对象
        """
        cuda_available = torch.cuda.is_available()
        
        if not cuda_available:
            self.recommendations.append("GPU not available. Consider using GPU for better performance")
            return GPUInfo(
                cuda_available=False,
                cuda_version=None,
                gpu_count=0,
                gpu_names=[],
                gpu_memory_mb=[],
                driver_version=None
            )
        
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        gpu_memory_mb = [
            torch.cuda.get_device_properties(i).total_memory // (1024**2) 
            for i in range(gpu_count)
        ]
        
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            driver_version = result.stdout.strip() if result.returncode == 0 else None
        except:
            driver_version = None
        
        return GPUInfo(
            cuda_available=True,
            cuda_version=torch.version.cuda,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            gpu_memory_mb=gpu_memory_mb,
            driver_version=driver_version
        )
    
    def _check_network(self) -> NetworkInfo:
        """检查网络连接
        
        Returns:
            网络信息对象
        """
        import requests
        import time
        
        latency_ms = {}
        api_endpoints = {}
        
        # 检查基本网络连接
        try:
            response = requests.get('https://www.google.com', timeout=5)
            internet_connected = response.status_code == 200
        except:
            internet_connected = False
            self.issues.append("No internet connection detected")
        
        # 检查API端点
        for name, url in self.API_ENDPOINTS.items():
            try:
                start_time = time.time()
                response = requests.get(url, timeout=5)
                latency_ms[name] = (time.time() - start_time) * 1000
                api_endpoints[name] = response.status_code in [200, 401, 403]  # 401/403表示可达但需认证
            except:
                latency_ms[name] = -1
                api_endpoints[name] = False
                
        return NetworkInfo(
            internet_connected=internet_connected,
            latency_ms=latency_ms,
            api_endpoints=api_endpoints
        )
    
    def _check_dependencies(self) -> Dict[str, str]:
        """检查Python包依赖
        
        Returns:
            包名到版本的映射
        """
        import importlib.metadata
        
        dependencies_status = {}
        
        for package, required_version in self.REQUIRED_PACKAGES.items():
            try:
                installed_version = importlib.metadata.version(package)
                dependencies_status[package] = installed_version
                
                # 简单版本比较
                if installed_version < required_version:
                    self.issues.append(
                        f"Package {package} version {installed_version} is below required {required_version}"
                    )
                    self.recommendations.append(f"Upgrade {package} to version {required_version} or higher")
                    
            except importlib.metadata.PackageNotFoundError:
                dependencies_status[package] = "NOT_INSTALLED"
                self.issues.append(f"Required package {package} is not installed")
                self.recommendations.append(f"Install {package} version {required_version} or higher")
                
        return dependencies_status
    
    def _calculate_health_score(
        self,
        system_info: SystemInfo,
        python_info: PythonInfo,
        gpu_info: GPUInfo,
        network_info: NetworkInfo,
        dependencies_status: Dict[str, str]
    ) -> float:
        """计算环境健康分数
        
        Args:
            system_info: 系统信息
            python_info: Python信息
            gpu_info: GPU信息
            network_info: 网络信息
            dependencies_status: 依赖状态
            
        Returns:
            0-100的健康分数
        """
        score = 100.0
        
        # 系统资源评分
        if system_info.available_memory_gb < self.MIN_MEMORY_GB:
            score -= 10
        if system_info.disk_available_gb < self.MIN_DISK_GB:
            score -= 10
            
        # Python版本评分
        current_version = tuple(map(int, python_info.version.split('.')))
        if current_version < self.REQUIRED_PYTHON_VERSION:
            score -= 20
            
        # GPU评分
        if not gpu_info.cuda_available:
            score -= 5  # GPU不是必需的，所以扣分较少
            
        # 网络评分
        if not network_info.internet_connected:
            score -= 15
        failed_endpoints = sum(1 for v in network_info.api_endpoints.values() if not v)
        score -= failed_endpoints * 5
        
        # 依赖评分
        missing_packages = sum(1 for v in dependencies_status.values() if v == "NOT_INSTALLED")
        score -= missing_packages * 10
        
        return max(0.0, min(100.0, score))

# 模块级别函数
def run_environment_check() -> EnvironmentReport:
    """运行环境检测的便捷函数
    
    Returns:
        环境检测报告
    """
    checker = EnvironmentChecker()
    return checker.check_all()

def validate_environment(raise_on_error: bool = True) -> bool:
    """验证环境是否满足最低要求
    
    Args:
        raise_on_error: 如果为True，在环境不满足要求时抛出异常
        
    Returns:
        环境是否满足要求
        
    Raises:
        QuantSystemError: 当环境不满足要求且raise_on_error为True时
    """
    report = run_environment_check()
    
    if report.health_score < 60:
        error_msg = f"Environment health score {report.health_score:.2f} is below minimum 60. Issues: {report.issues}"
        if raise_on_error:
            raise QuantSystemError(error_msg)
        else:
            logger.error(error_msg)
            return False
            
    return True