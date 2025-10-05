"""
计算资源优化器模块
优化计算资源的分配和使用
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from common.logging_system import setup_logger

logger = setup_logger("compute_optimizer")


@dataclass
class ComputeResource:
    """计算资源定义"""

    resource_id: str
    resource_type: str  # 'CPU', 'GPU', 'Memory'
    capacity: float  # 容量
    cost_per_unit: float  # 单位成本
    availability: float = 1.0  # 可用性（0-1）


@dataclass
class ComputeTask:
    """计算任务定义"""

    task_id: str
    task_type: str
    resource_requirements: Dict[str, float]  # 资源需求
    priority: int = 1  # 优先级
    estimated_duration: float = 1.0  # 预计时长（小时）


class ComputeOptimizer:
    """计算资源优化器"""

    def __init__(
        self,
        resources: List[ComputeResource],
        optimization_objective: str = "min_cost",
    ):
        """初始化计算优化器

        Args:
            resources: 可用资源列表
            optimization_objective: 优化目标 ('min_cost', 'min_time', 'max_utilization')
        """
        self.resources = resources
        self.optimization_objective = optimization_objective
        self.resource_dict = {r.resource_id: r for r in resources}

    def allocate_resources(
        self, tasks: List[ComputeTask]
    ) -> Dict[str, Dict[str, float]]:
        """为任务分配资源

        Args:
            tasks: 任务列表

        Returns:
            资源分配方案 {task_id: {resource_id: allocation}}
        """
        allocation = {}

        # 按优先级排序任务
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)

        # 跟踪资源使用情况
        resource_usage = {r.resource_id: 0.0 for r in self.resources}

        for task in sorted_tasks:
            task_allocation = self._allocate_for_task(task, resource_usage)
            allocation[task.task_id] = task_allocation

            # 更新资源使用
            for resource_id, amount in task_allocation.items():
                resource_usage[resource_id] += amount

        logger.info(f"Allocated resources for {len(tasks)} tasks")
        return allocation

    def _allocate_for_task(
        self, task: ComputeTask, current_usage: Dict[str, float]
    ) -> Dict[str, float]:
        """为单个任务分配资源

        Args:
            task: 任务
            current_usage: 当前资源使用情况

        Returns:
            资源分配
        """
        allocation = {}

        for resource_type, required_amount in task.resource_requirements.items():
            # 找到满足需求的最优资源
            best_resource = None
            best_score = float("inf")

            for resource in self.resources:
                if resource.resource_type == resource_type:
                    # 检查可用容量
                    available = (
                        resource.capacity * resource.availability
                        - current_usage[resource.resource_id]
                    )

                    if available >= required_amount:
                        # 计算分配得分
                        if self.optimization_objective == "min_cost":
                            score = resource.cost_per_unit
                        elif self.optimization_objective == "min_time":
                            score = 1.0 / resource.capacity  # 容量越大越好
                        else:  # max_utilization
                            score = -current_usage[resource.resource_id]

                        if score < best_score:
                            best_score = score
                            best_resource = resource

            if best_resource:
                allocation[best_resource.resource_id] = required_amount
            else:
                logger.warning(
                    f"Cannot allocate {resource_type} for task {task.task_id}"
                )

        return allocation

    def calculate_total_cost(
        self, allocation: Dict[str, Dict[str, float]], tasks: List[ComputeTask]
    ) -> float:
        """计算总成本

        Args:
            allocation: 资源分配方案
            tasks: 任务列表

        Returns:
            总成本
        """
        total_cost = 0.0
        task_dict = {t.task_id: t for t in tasks}

        for task_id, task_allocation in allocation.items():
            task = task_dict[task_id]
            for resource_id, amount in task_allocation.items():
                resource = self.resource_dict[resource_id]
                cost = amount * resource.cost_per_unit * task.estimated_duration
                total_cost += cost

        return total_cost

    def calculate_utilization(
        self, allocation: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """计算资源利用率

        Args:
            allocation: 资源分配方案

        Returns:
            利用率字典
        """
        utilization = {}

        for resource in self.resources:
            total_allocated = 0.0
            for task_allocation in allocation.values():
                if resource.resource_id in task_allocation:
                    total_allocated += task_allocation[resource.resource_id]

            utilization[resource.resource_id] = total_allocated / resource.capacity

        return utilization

    def optimize_batch_processing(
        self, tasks: List[ComputeTask], batch_size: int
    ) -> List[List[str]]:
        """优化批处理任务分组

        Args:
            tasks: 任务列表
            batch_size: 批次大小

        Returns:
            任务分组列表
        """
        # 按资源需求相似度分组
        batches = []
        remaining_tasks = tasks.copy()

        while remaining_tasks:
            # 创建新批次
            batch = [remaining_tasks.pop(0)]
            batch_requirements = self._sum_requirements(batch)

            # 添加相似任务到批次
            i = 0
            while i < len(remaining_tasks) and len(batch) < batch_size:
                task = remaining_tasks[i]
                combined_requirements = self._sum_requirements(batch + [task])

                # 检查资源约束
                if self._check_resource_constraints(combined_requirements):
                    batch.append(task)
                    remaining_tasks.pop(i)
                    batch_requirements = combined_requirements
                else:
                    i += 1

            batches.append([t.task_id for t in batch])

        logger.info(f"Created {len(batches)} batches for {len(tasks)} tasks")
        return batches

    def _sum_requirements(self, tasks: List[ComputeTask]) -> Dict[str, float]:
        """累加任务资源需求

        Args:
            tasks: 任务列表

        Returns:
            总资源需求
        """
        total_requirements = {}

        for task in tasks:
            for resource_type, amount in task.resource_requirements.items():
                if resource_type not in total_requirements:
                    total_requirements[resource_type] = 0.0
                total_requirements[resource_type] += amount

        return total_requirements

    def _check_resource_constraints(self, requirements: Dict[str, float]) -> bool:
        """检查资源约束

        Args:
            requirements: 资源需求

        Returns:
            是否满足约束
        """
        for resource_type, required_amount in requirements.items():
            # 查找该类型的总容量
            total_capacity = sum(
                r.capacity for r in self.resources if r.resource_type == resource_type
            )

            if required_amount > total_capacity:
                return False

        return True
