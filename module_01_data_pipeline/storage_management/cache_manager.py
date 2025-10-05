# -*- coding: utf-8 -*-
"""
缓存管理模块
支持内存缓存、LRU、TTL等机制
"""

import time
from collections import OrderedDict
from typing import Any, Optional

from common.logging_system import setup_logger

logger = setup_logger("cache_manager")


class LRUCache:
    """
    LRU缓存，支持最大容量和TTL
    """

    def __init__(self, capacity: int = 1000, ttl: int = 3600):
        self.capacity = capacity
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamp = dict()

    def get(self, key: Any) -> Optional[Any]:
        now = time.time()
        if key in self.cache:
            if now - self.timestamp[key] > self.ttl:
                logger.info(f"Cache expired for key: {key}")
                self.cache.pop(key)
                self.timestamp.pop(key)
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: Any, value: Any) -> None:
        now = time.time()
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        self.timestamp[key] = now
        if len(self.cache) > self.capacity:
            oldest = next(iter(self.cache))
            logger.info(f"Evicting oldest cache key: {oldest}")
            self.cache.popitem(last=False)
            self.timestamp.pop(oldest, None)

    def clear(self):
        self.cache.clear()
        self.timestamp.clear()
        logger.info("Cache cleared")
