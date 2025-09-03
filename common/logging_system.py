import logging
import sys

from loguru import logger


def setup_logger(module_name: str) -> logging.Logger:
    """设置模块日志器

    Args:
        module_name: 模块名称

    Returns:
        配置好的日志器实例
    """
    logger.remove()
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
    )
    logger.add(
        f"logs/{module_name}.log",
        rotation="100 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        level="DEBUG",
    )
    return logger
