"""
性能追踪器模块
用于追踪和记录系统各操作的性能指标
"""

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from common.logging_system import setup_logger

logger = setup_logger("performance_tracker")


@dataclass
class PerformanceRecord:
    """性能记录"""

    operation: str
    start_time: datetime
    end_time: datetime
    duration: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """性能统计"""

    operation: str
    count: int
    total_duration: float
    avg_duration: float
    min_duration: float
    max_duration: float
    success_count: int
    failure_count: int
    success_rate: float
    last_execution: datetime


class PerformanceTracker:
    """性能追踪器类"""

    def __init__(self, max_records: int = 10000):
        """初始化性能追踪器

        Args:
            max_records: 最大记录数
        """
        self.max_records = max_records
        self.records: deque = deque(maxlen=max_records)
        self.operation_records: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.stats_cache: Dict[str, PerformanceStats] = {}
        self.cache_timestamp: datetime = datetime.now()
        self.cache_ttl: int = 60  # 缓存TTL（秒）

    @contextmanager
    def track(self, operation: str, metadata: Dict[str, Any] = None):
        """追踪操作性能的上下文管理器

        Args:
            operation: 操作名称
            metadata: 元数据

        Yields:
            性能记录对象（在上下文结束后填充）
        """
        start_time = datetime.now()
        record = PerformanceRecord(
            operation=operation,
            start_time=start_time,
            end_time=start_time,  # 临时值
            duration=0.0,
            success=True,
            metadata=metadata or {},
        )

        try:
            yield record
            record.success = True
        except Exception as e:
            record.success = False
            record.error_message = str(e)
            logger.error(f"Operation '{operation}' failed: {e}")
            raise
        finally:
            end_time = datetime.now()
            record.end_time = end_time
            record.duration = (end_time - start_time).total_seconds()

            # 记录
            self.records.append(record)
            self.operation_records[operation].append(record)

            # 清除缓存
            if operation in self.stats_cache:
                del self.stats_cache[operation]

            logger.debug(
                f"Operation '{operation}' completed in {record.duration:.3f}s "
                f"(success={record.success})"
            )

    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        error_message: str = None,
        metadata: Dict[str, Any] = None,
    ):
        """手动记录操作

        Args:
            operation: 操作名称
            duration: 持续时间（秒）
            success: 是否成功
            error_message: 错误消息
            metadata: 元数据
        """
        now = datetime.now()
        record = PerformanceRecord(
            operation=operation,
            start_time=now - timedelta(seconds=duration),
            end_time=now,
            duration=duration,
            success=success,
            error_message=error_message,
            metadata=metadata or {},
        )

        self.records.append(record)
        self.operation_records[operation].append(record)

        # 清除缓存
        if operation in self.stats_cache:
            del self.stats_cache[operation]

    def get_stats(self, operation: str = None) -> Dict[str, PerformanceStats]:
        """获取性能统计

        Args:
            operation: 操作名称（None表示所有操作）

        Returns:
            统计字典
        """
        # 检查缓存
        if operation and operation in self.stats_cache:
            if (datetime.now() - self.cache_timestamp).total_seconds() < self.cache_ttl:
                return {operation: self.stats_cache[operation]}

        # 计算统计
        if operation:
            operations = [operation] if operation in self.operation_records else []
        else:
            operations = list(self.operation_records.keys())

        stats = {}
        for op in operations:
            op_records = list(self.operation_records[op])

            if not op_records:
                continue

            durations = [r.duration for r in op_records]
            success_count = sum(1 for r in op_records if r.success)
            failure_count = len(op_records) - success_count

            op_stats = PerformanceStats(
                operation=op,
                count=len(op_records),
                total_duration=sum(durations),
                avg_duration=sum(durations) / len(durations),
                min_duration=min(durations),
                max_duration=max(durations),
                success_count=success_count,
                failure_count=failure_count,
                success_rate=success_count / len(op_records)
                if len(op_records) > 0
                else 0,
                last_execution=op_records[-1].end_time,
            )

            stats[op] = op_stats
            self.stats_cache[op] = op_stats

        self.cache_timestamp = datetime.now()
        return stats

    def get_records(
        self,
        operation: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        success_only: bool = False,
        limit: int = None,
    ) -> List[PerformanceRecord]:
        """获取性能记录

        Args:
            operation: 操作名称
            start_time: 开始时间
            end_time: 结束时间
            success_only: 只返回成功的记录
            limit: 限制数量

        Returns:
            记录列表
        """
        # 选择数据源
        if operation:
            records = list(self.operation_records.get(operation, []))
        else:
            records = list(self.records)

        # 过滤
        if start_time:
            records = [r for r in records if r.end_time >= start_time]

        if end_time:
            records = [r for r in records if r.end_time <= end_time]

        if success_only:
            records = [r for r in records if r.success]

        # 排序（最新的在前）
        records.sort(key=lambda x: x.end_time, reverse=True)

        # 限制数量
        if limit:
            records = records[:limit]

        return records

    def get_slow_operations(
        self, operation: str = None, threshold: float = 1.0, limit: int = 10
    ) -> List[PerformanceRecord]:
        """获取慢操作

        Args:
            operation: 操作名称
            threshold: 慢操作阈值（秒）
            limit: 限制数量

        Returns:
            慢操作列表
        """
        records = self.get_records(operation=operation)

        # 过滤慢操作
        slow_records = [r for r in records if r.duration >= threshold]

        # 按持续时间排序
        slow_records.sort(key=lambda x: x.duration, reverse=True)

        return slow_records[:limit]

    def get_failed_operations(
        self, operation: str = None, limit: int = 10
    ) -> List[PerformanceRecord]:
        """获取失败操作

        Args:
            operation: 操作名称
            limit: 限制数量

        Returns:
            失败操作列表
        """
        records = self.get_records(operation=operation)

        # 过滤失败操作
        failed_records = [r for r in records if not r.success]

        return failed_records[:limit]

    def get_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """获取性能摘要

        Args:
            minutes: 时间范围（分钟）

        Returns:
            摘要字典
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_records = [r for r in self.records if r.end_time >= cutoff_time]

        if not recent_records:
            return {
                "period_minutes": minutes,
                "total_operations": 0,
                "success_rate": 0,
                "avg_duration": 0,
                "operations": {},
            }

        # 整体统计
        total_operations = len(recent_records)
        success_count = sum(1 for r in recent_records if r.success)
        success_rate = success_count / total_operations if total_operations > 0 else 0
        avg_duration = sum(r.duration for r in recent_records) / total_operations

        # 按操作统计
        operation_stats = {}
        for op_name in set(r.operation for r in recent_records):
            op_records = [r for r in recent_records if r.operation == op_name]
            op_success = sum(1 for r in op_records if r.success)

            operation_stats[op_name] = {
                "count": len(op_records),
                "success_rate": op_success / len(op_records),
                "avg_duration": sum(r.duration for r in op_records) / len(op_records),
                "max_duration": max(r.duration for r in op_records),
            }

        return {
            "period_minutes": minutes,
            "total_operations": total_operations,
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "operations": operation_stats,
        }

    def clear(self, operation: str = None):
        """清除记录

        Args:
            operation: 操作名称（None表示清除所有）
        """
        if operation:
            if operation in self.operation_records:
                self.operation_records[operation].clear()
            if operation in self.stats_cache:
                del self.stats_cache[operation]
        else:
            self.records.clear()
            self.operation_records.clear()
            self.stats_cache.clear()

        logger.info(f"Cleared performance records for {operation or 'all operations'}")


# 全局实例
_global_performance_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker(max_records: int = 10000) -> PerformanceTracker:
    """获取全局性能追踪器实例

    Args:
        max_records: 最大记录数

    Returns:
        性能追踪器实例
    """
    global _global_performance_tracker
    if _global_performance_tracker is None:
        _global_performance_tracker = PerformanceTracker(max_records)
    return _global_performance_tracker
