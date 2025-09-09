"""
数据管道模块初始化文件
"""

from .data_acquisition.akshare_collector import AkshareDataCollector
from .storage_management.database_manager import DatabaseManager, get_database_manager

__all__ = [
    "AkshareDataCollector",
    "DatabaseManager",
    "get_database_manager"
]
