"""
内存优化器模块
优化内存使用和管理
"""

import gc
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from common.logging_system import setup_logger

logger = setup_logger("memory_optimizer")


@dataclass
class MemoryProfile:
    """内存使用概况"""

    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    object_count: int
    largest_objects: List[Dict[str, Any]]


class MemoryOptimizer:
    """内存优化器"""

    def __init__(self, memory_limit_mb: Optional[float] = None):
        """初始化内存优化器

        Args:
            memory_limit_mb: 内存限制（MB）
        """
        self.memory_limit_mb = memory_limit_mb
        self.cache = {}

    def get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）

        Returns:
            内存使用量（MB）
        """
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # 转换为MB

    def profile_memory(self, top_n: int = 10) -> MemoryProfile:
        """内存使用分析

        Args:
            top_n: 返回最大对象的数量

        Returns:
            内存概况
        """
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        total_memory = memory_info.rss / (1024 * 1024)
        used_memory = total_memory

        # 获取系统可用内存
        virtual_memory = psutil.virtual_memory()
        available_memory = virtual_memory.available / (1024 * 1024)

        # 分析对象
        all_objects = gc.get_objects()
        object_count = len(all_objects)

        # 找出最大的对象
        object_sizes = []
        for obj in all_objects:
            try:
                size = sys.getsizeof(obj)
                obj_type = type(obj).__name__
                object_sizes.append({"type": obj_type, "size_mb": size / (1024 * 1024)})
            except:
                pass

        # 排序并获取前N个
        object_sizes.sort(key=lambda x: x["size_mb"], reverse=True)
        largest_objects = object_sizes[:top_n]

        return MemoryProfile(
            total_memory_mb=total_memory,
            used_memory_mb=used_memory,
            available_memory_mb=available_memory,
            object_count=object_count,
            largest_objects=largest_objects,
        )

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame内存使用

        Args:
            df: 原始DataFrame

        Returns:
            优化后的DataFrame
        """
        optimized_df = df.copy()

        # 优化数值列
        for col in optimized_df.select_dtypes(include=["int"]).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()

            if col_min >= 0:
                if col_max < 255:
                    optimized_df[col] = optimized_df[col].astype(np.uint8)
                elif col_max < 65535:
                    optimized_df[col] = optimized_df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    optimized_df[col] = optimized_df[col].astype(np.uint32)
            else:
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif (
                    col_min > np.iinfo(np.int16).min
                    and col_max < np.iinfo(np.int16).max
                ):
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif (
                    col_min > np.iinfo(np.int32).min
                    and col_max < np.iinfo(np.int32).max
                ):
                    optimized_df[col] = optimized_df[col].astype(np.int32)

        for col in optimized_df.select_dtypes(include=["float"]).columns:
            optimized_df[col] = optimized_df[col].astype(np.float32)

        # 优化字符串列
        for col in optimized_df.select_dtypes(include=["object"]).columns:
            num_unique = optimized_df[col].nunique()
            num_total = len(optimized_df[col])

            if num_unique / num_total < 0.5:  # 如果唯一值少，转换为category
                optimized_df[col] = optimized_df[col].astype("category")

        original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)

        logger.info(
            f"DataFrame memory optimized: {original_memory:.2f}MB -> {optimized_memory:.2f}MB "
            f"(saved {(original_memory - optimized_memory) / original_memory * 100:.1f}%)"
        )

        return optimized_df

    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """清理缓存

        Args:
            cache_type: 缓存类型，None表示清理所有
        """
        if cache_type:
            if cache_type in self.cache:
                del self.cache[cache_type]
                logger.info(f"Cleared cache: {cache_type}")
        else:
            self.cache.clear()
            logger.info("Cleared all caches")

        # 强制垃圾回收
        gc.collect()

    def set_cache(self, key: str, value: Any, size_limit_mb: float = 100) -> bool:
        """设置缓存（带大小限制）

        Args:
            key: 缓存键
            value: 缓存值
            size_limit_mb: 单个缓存大小限制（MB）

        Returns:
            是否成功设置
        """
        value_size = sys.getsizeof(value) / (1024 * 1024)

        if value_size > size_limit_mb:
            logger.warning(
                f"Cache value too large: {value_size:.2f}MB > {size_limit_mb}MB"
            )
            return False

        self.cache[key] = value
        return True

    def get_cache(self, key: str) -> Optional[Any]:
        """获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存值
        """
        return self.cache.get(key)

    def chunk_dataframe(self, df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
        """将DataFrame分块以节省内存

        Args:
            df: 原始DataFrame
            chunk_size: 块大小

        Returns:
            DataFrame块列表
        """
        chunks = []
        n_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx].copy()
            chunks.append(chunk)

        logger.info(f"Split DataFrame into {len(chunks)} chunks")
        return chunks

    def check_memory_limit(self) -> bool:
        """检查是否超过内存限制

        Returns:
            是否在限制内
        """
        if self.memory_limit_mb is None:
            return True

        current_usage = self.get_memory_usage()
        is_within_limit = current_usage <= self.memory_limit_mb

        if not is_within_limit:
            logger.warning(
                f"Memory usage {current_usage:.2f}MB exceeds limit {self.memory_limit_mb}MB"
            )

        return is_within_limit

    def suggest_optimizations(self) -> List[str]:
        """建议内存优化措施

        Returns:
            优化建议列表
        """
        profile = self.profile_memory()
        suggestions = []

        # 检查内存使用
        if profile.used_memory_mb > 1000:  # > 1GB
            suggestions.append("内存使用较高，考虑清理缓存或减少数据加载")

        # 检查大对象
        if profile.largest_objects:
            largest = profile.largest_objects[0]
            if largest["size_mb"] > 100:  # > 100MB
                suggestions.append(
                    f"存在大型{largest['type']}对象（{largest['size_mb']:.2f}MB），"
                    f"考虑分块处理或使用迭代器"
                )

        # 检查对象数量
        if profile.object_count > 100000:
            suggestions.append(
                f"对象数量较多（{profile.object_count}），建议进行垃圾回收"
            )

        if not suggestions:
            suggestions.append("内存使用正常")

        return suggestions
