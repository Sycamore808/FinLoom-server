"""
AI模型存储管理模块初始化
"""

from .ai_model_database import AIModelDatabaseManager, get_ai_model_database_manager

__all__ = ["AIModelDatabaseManager", "get_ai_model_database_manager"]
