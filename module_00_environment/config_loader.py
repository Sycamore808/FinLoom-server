"""
配置加载器模块
负责加载和管理系统的各种配置文件
"""

import os
from typing import Dict, Any
import yaml
from pathlib import Path

from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("config_loader")

class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_dir: str = "config"):
        """初始化配置加载器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise QuantSystemError(f"Config directory not found: {config_dir}")
            
    def load_system_config(self) -> Dict[str, Any]:
        """加载系统配置
        
        Returns:
            系统配置字典
        """
        config_path = self.config_dir / "system_config.yaml"
        return self._load_yaml_config(config_path)
        
    def load_model_config(self) -> Dict[str, Any]:
        """加载模型配置
        
        Returns:
            模型配置字典
        """
        config_path = self.config_dir / "model_config.yaml"
        return self._load_yaml_config(config_path)
        
    def load_trading_config(self) -> Dict[str, Any]:
        """加载交易配置
        
        Returns:
            交易配置字典
        """
        config_path = self.config_dir / "trading_config.yaml"
        return self._load_yaml_config(config_path)
        
    def _load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """加载YAML配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise QuantSystemError(f"Failed to load config: {e}")
            
    def get_config_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """获取配置值（支持嵌套键路径）
        
        Args:
            config: 配置字典
            key_path: 键路径，如 "database.host"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

# 模块级别函数
def load_all_configs(config_dir: str = "config") -> Dict[str, Dict[str, Any]]:
    """加载所有配置文件
    
    Args:
        config_dir: 配置文件目录
        
    Returns:
        所有配置的字典
    """
    loader = ConfigLoader(config_dir)
    
    return {
        "system": loader.load_system_config(),
        "model": loader.load_model_config(),
        "trading": loader.load_trading_config()
    }