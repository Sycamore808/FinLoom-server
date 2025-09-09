"""
存储管理模块初始化文件
"""

from .database_manager import DatabaseManager, get_database_manager

__all__ = [
    "DatabaseManager",
    "get_database_manager"
]
