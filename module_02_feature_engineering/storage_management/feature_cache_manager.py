"""
特征缓存管理器
专门用于特征工程数据的内存缓存
"""

import time
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from common.logging_system import setup_logger

logger = setup_logger("feature_cache_manager")


class FeatureCacheManager:
    """特征缓存管理器类"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """初始化特征缓存管理器

        Args:
            max_size: 最大缓存条目数
            ttl: 缓存生存时间(秒)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.access_times: Dict[str, float] = {}

    def _generate_key(self, data_type: str, symbol: str, **kwargs) -> str:
        """生成缓存键"""
        key_parts = [data_type, symbol]
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}={v}")
        return ":".join(key_parts)

    def _is_expired(self, key: str) -> bool:
        """检查缓存是否过期"""
        if key not in self.access_times:
            return True
        return (time.time() - self.access_times[key]) > self.ttl

    def _evict_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key
            for key, access_time in self.access_times.items()
            if (current_time - access_time) > self.ttl
        ]

        for key in expired_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)

    def _evict_lru(self):
        """LRU淘汰"""
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            self.access_times.pop(oldest_key, None)

    def set(self, data_type: str, symbol: str, data: Any, **kwargs) -> None:
        """设置缓存

        Args:
            data_type: 数据类型 (technical_indicators, factors, etc.)
            symbol: 股票代码
            data: 要缓存的数据
            **kwargs: 额外的键值参数
        """
        key = self._generate_key(data_type, symbol, **kwargs)

        self._evict_expired()
        self._evict_lru()

        self.cache[key] = data
        self.access_times[key] = time.time()

        # 移到末尾(最近使用)
        self.cache.move_to_end(key)

    def get(self, data_type: str, symbol: str, **kwargs) -> Optional[Any]:
        """获取缓存

        Args:
            data_type: 数据类型
            symbol: 股票代码
            **kwargs: 额外的键值参数

        Returns:
            缓存的数据或None
        """
        key = self._generate_key(data_type, symbol, **kwargs)

        if key not in self.cache:
            return None

        if self._is_expired(key):
            self.cache.pop(key)
            self.access_times.pop(key, None)
            return None

        # 更新访问时间并移到末尾
        self.access_times[key] = time.time()
        self.cache.move_to_end(key)

        return self.cache[key]

    def clear(self) -> None:
        """清空所有缓存"""
        self.cache.clear()
        self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "hit_rate": getattr(self, "_hit_count", 0)
            / max(getattr(self, "_access_count", 1), 1),
        }
