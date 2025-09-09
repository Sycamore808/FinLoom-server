"""
量化系统异常定义模块
"""

class QuantSystemError(Exception):
    """量化系统基础异常"""
    pass


class DataError(QuantSystemError):
    """数据相关异常"""
    pass


class ModelError(QuantSystemError):
    """模型相关异常"""
    pass


class ExecutionError(QuantSystemError):
    """执行相关异常"""
    pass


class ConfigError(QuantSystemError):
    """配置相关异常"""
    pass


class ValidationError(QuantSystemError):
    """验证相关异常"""
    pass
