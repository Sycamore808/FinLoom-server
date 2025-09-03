"""
依赖安装器模块
负责自动安装和管理Python包依赖
"""

import os
import sys
import subprocess
import json
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import pkg_resources
from common.logging_system import setup_logger
from common.exceptions import QuantSystemError

logger = setup_logger("dependency_installer")

class DependencyInstaller:
    """依赖安装器类"""
    
    def __init__(self, requirements_file: str = "requirements.txt"):
        """初始化依赖安装器
        
        Args:
            requirements_file: requirements文件路径
        """
        self.requirements_file = requirements_file
        self.pip_index_url = "https://pypi.org/simple"
        self.pip_trusted_host = "pypi.org"
        
    def scan_missing_packages(self) -> Dict[str, str]:
        """扫描缺失的包
        
        Returns:
            缺失包名到所需版本的映射
        """
        missing_packages = {}
        
        # 读取requirements文件
        requirements = self._parse_requirements_file()
        
        for package_name, required_version in requirements.items():
            try:
                installed_version = pkg_resources.get_distribution(package_name).version
                if required_version and not self._version_satisfied(installed_version, required_version):
                    missing_packages[package_name] = required_version
            except pkg_resources.DistributionNotFound:
                missing_packages[package_name] = required_version
                
        logger.info(f"Found {len(missing_packages)} missing or outdated packages")
        return missing_packages
    
    def install_core_dependencies(self) -> bool:
        """安装核心依赖包
        
        Returns:
            是否成功安装所有核心依赖
        """
        core_packages = {
            'numpy': '>=1.24.0',
            'pandas': '>=2.0.0',
            'polars': '>=0.19.0',
            'torch': '>=2.0.0',
            'pydantic': '>=2.4.0'
        }
        
        success = True
        for package, version in core_packages.items():
            if not self._install_package(package, version):
                success = False
                logger.error(f"Failed to install core package: {package}")
                
        return success
    
    def install_module_dependencies(self, module_id: str) -> bool:
        """安装特定模块的依赖
        
        Args:
            module_id: 模块ID，如 'module_00_environment'
            
        Returns:
            是否成功安装
        """
        module_requirements = {
            'module_00_environment': ['psutil>=5.9.0', 'loguru>=0.7.0'],
            'module_01_data_pipeline': ['yfinance>=0.2.30', 'akshare>=1.11.0', 'kafka-python>=2.0.0'],
            'module_02_feature_engineering': ['ta-lib>=0.4.28', 'scipy>=1.10.0'],
            'module_03_ai_models': ['transformers>=4.35.0', 'pytorch-lightning>=2.0.0'],
            'module_04_market_analysis': ['textblob>=0.17.0', 'vaderSentiment>=3.3.0'],
            'module_05_risk_management': ['cvxpy>=1.4.0', 'pyportfolioopt>=1.5.0'],
            'module_06_monitoring_alerting': ['prometheus-client>=0.19.0', 'grafana-api>=1.0.0'],
            'module_07_optimization': ['optuna>=3.4.0', 'hyperopt>=0.2.0'],
            'module_08_execution': ['ccxt>=4.1.0', 'ib_insync>=0.9.0'],
            'module_09_backtesting': ['backtrader>=1.9.0', 'zipline-reloaded>=2.4.0'],
            'module_10_ai_interaction': ['openai>=1.0.0', 'langchain>=0.1.0'],
            'module_11_visualization': ['plotly>=5.17.0', 'dash>=2.14.0']
        }
        
        if module_id not in module_requirements:
            logger.warning(f"No specific requirements found for module: {module_id}")
            return True
            
        requirements = module_requirements[module_id]
        success = True
        
        for requirement in requirements:
            package_name = requirement.split('>=')[0].split('==')[0]
            if not self._install_package_from_requirement(requirement):
                success = False
                logger.error(f"Failed to install {package_name} for module {module_id}")
                
        return success
    
    def resolve_version_conflicts(self) -> List[Tuple[str, str, str]]:
        """解析版本冲突
        
        Returns:
            冲突列表，每个元素为(包名, 已安装版本, 需要版本)
        """
        conflicts = []
        dependency_graph = self._build_dependency_graph()
        
        for package, dependencies in dependency_graph.items():
            for dep_name, dep_version in dependencies.items():
                try:
                    installed = pkg_resources.get_distribution(dep_name).version
                    if not self._version_satisfied(installed, dep_version):
                        conflicts.append((dep_name, installed, dep_version))
                except pkg_resources.DistributionNotFound:
                    pass
                    
        return conflicts
    
    def create_virtual_environment(self, env_name: str = "venv") -> bool:
        """创建虚拟环境
        
        Args:
            env_name: 虚拟环境名称
            
        Returns:
            是否成功创建
        """
        try:
            # 创建虚拟环境
            subprocess.run(
                [sys.executable, '-m', 'venv', env_name],
                check=True,
                capture_output=True
            )
            
            # 激活脚本路径
            if os.name == 'nt':  # Windows
                activate_script = os.path.join(env_name, 'Scripts', 'activate.bat')
            else:  # Unix/Linux/Mac
                activate_script = os.path.join(env_name, 'bin', 'activate')
                
            logger.info(f"Virtual environment created: {env_name}")
            logger.info(f"Activate with: source {activate_script}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return False
    
    def _parse_requirements_file(self) -> Dict[str, str]:
        """解析requirements文件
        
        Returns:
            包名到版本要求的映射
        """
        requirements = {}
        
        if not os.path.exists(self.requirements_file):
            logger.warning(f"Requirements file not found: {self.requirements_file}")
            return requirements
            
        with open(self.requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 解析包名和版本
                    if '>=' in line:
                        name, version = line.split('>=')
                        requirements[name.strip()] = f'>={version.strip()}'
                    elif '==' in line:
                        name, version = line.split('==')
                        requirements[name.strip()] = f'=={version.strip()}'
                    else:
                        requirements[line] = ''
                        
        return requirements
    
    def _install_package(self, package_name: str, version_spec: str = '') -> bool:
        """安装单个包
        
        Args:
            package_name: 包名
            version_spec: 版本要求
            
        Returns:
            是否成功安装
        """
        try:
            install_cmd = [sys.executable, '-m', 'pip', 'install']
            
            if version_spec:
                install_cmd.append(f"{package_name}{version_spec}")
            else:
                install_cmd.append(package_name)
                
            install_cmd.extend([
                '-i', self.pip_index_url,
                '--trusted-host', self.pip_trusted_host
            ])
            
            result = subprocess.run(
                install_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.info(f"Successfully installed {package_name}{version_spec}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package_name}: {e.stderr}")
            return False
    
    def _install_package_from_requirement(self, requirement: str) -> bool:
        """从requirement字符串安装包
        
        Args:
            requirement: requirement字符串，如 'numpy>=1.24.0'
            
        Returns:
            是否成功安装
        """
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', requirement],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Successfully installed {requirement}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {requirement}: {e.stderr}")
            return False
    
    def _version_satisfied(self, installed: str, required: str) -> bool:
        """检查版本是否满足要求
        
        Args:
            installed: 已安装版本
            required: 要求版本
            
        Returns:
            是否满足
        """
        if not required:
            return True
            
        # 简化版本比较，实际应使用packaging库
        if required.startswith('>='):
            required_version = required[2:]
            return installed >= required_version
        elif required.startswith('=='):
            required_version = required[2:]
            return installed == required_version
        else:
            return True
    
    def _build_dependency_graph(self) -> Dict[str, Dict[str, str]]:
        """构建依赖图
        
        Returns:
            包依赖关系图
        """
        graph = {}
        
        for dist in pkg_resources.working_set:
            dependencies = {}
            for req in dist.requires():
                dependencies[req.project_name] = str(req.specifier)
            graph[dist.project_name] = dependencies
            
        return graph

# 模块级别函数
def auto_install_dependencies() -> bool:
    """自动安装所有依赖
    
    Returns:
        是否成功
    """
    installer = DependencyInstaller()
    
    # 首先安装核心依赖
    if not installer.install_core_dependencies():
        logger.error("Failed to install core dependencies")
        return False
    
    # 扫描并安装缺失的包
    missing = installer.scan_missing_packages()
    for package, version in missing.items():
        installer._install_package(package, version)
    
    return True

def verify_installation() -> bool:
    """验证所有依赖是否正确安装
    
    Returns:
        是否所有依赖都已正确安装
    """
    installer = DependencyInstaller()
    missing = installer.scan_missing_packages()
    return len(missing) == 0