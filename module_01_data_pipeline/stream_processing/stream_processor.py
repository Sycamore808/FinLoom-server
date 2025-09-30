# -*- coding: utf-8 -*-
"""
流式数据处理模块
支持注册回调、批量处理等
"""

from typing import Any, Callable, List

from common.exceptions import DataError
from common.logging_system import setup_logger

logger = setup_logger("stream_processor")


class StreamProcessor:
    """
    流式数据处理器，支持注册回调和批量处理
    """

    def __init__(self):
        self.callbacks: List[Callable[[Any], None]] = []

    def register_callback(self, callback: Callable[[Any], None]) -> None:
        self.callbacks.append(callback)
        logger.info(f"Registered callback: {callback}")

    def process(self, data: Any) -> None:
        try:
            for cb in self.callbacks:
                try:
                    cb(data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            logger.info("Stream data processed")
        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            raise DataError(f"Stream processing failed: {e}")
