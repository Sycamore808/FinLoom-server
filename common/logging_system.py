import logging
import sys
import os
from datetime import datetime

# 尝试导入loguru，如果没有则使用标准logging
try:
    from loguru import logger as loguru_logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False

def setup_logger(module_name: str) -> logging.Logger:
    """设置模块日志器

    Args:
        module_name: 模块名称

    Returns:
        配置好的日志器实例
    """
    if HAS_LOGURU:
        # 使用loguru
        loguru_logger.remove()
        loguru_logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
            level="INFO",
        )
        
        # 确保日志目录存在
        os.makedirs("logs", exist_ok=True)
        loguru_logger.add(
            f"logs/{module_name}.log",
            rotation="100 MB",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
            level="DEBUG",
        )
        return loguru_logger
    else:
        # 使用标准logging
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not logger.handlers:
            # 控制台handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # 文件handler
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler(f"logs/{module_name}.log")
            file_handler.setLevel(logging.DEBUG)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
