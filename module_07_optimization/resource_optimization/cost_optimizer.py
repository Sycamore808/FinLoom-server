"""
成本优化器模块
优化系统运行成本
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from common.logging_system import setup_logger

logger = setup_logger("cost_optimizer")


@dataclass
class CostComponent:
    """成本组成"""

    component_id: str
    component_type: str  # 'data', 'compute', 'storage', 'network'
    fixed_cost: float  # 固定成本
    variable_cost: float  # 变动成本（单位）
    volume: float = 0.0  # 使用量


class CostOptimizer:
    """成本优化器"""

    def __init__(self, cost_components: List[CostComponent]):
        """初始化成本优化器

        Args:
            cost_components: 成本组成列表
        """
        self.cost_components = cost_components
        self.component_dict = {c.component_id: c for c in cost_components}

    def calculate_total_cost(self) -> float:
        """计算总成本

        Returns:
            总成本
        """
        total_cost = 0.0

        for component in self.cost_components:
            cost = component.fixed_cost + component.variable_cost * component.volume
            total_cost += cost

        return total_cost

    def calculate_cost_breakdown(self) -> Dict[str, Dict[str, float]]:
        """计算成本明细

        Returns:
            成本明细字典
        """
        breakdown = {}

        # 按类型分组
        type_costs = {}
        for component in self.cost_components:
            comp_type = component.component_type
            if comp_type not in type_costs:
                type_costs[comp_type] = {"fixed": 0.0, "variable": 0.0, "total": 0.0}

            fixed = component.fixed_cost
            variable = component.variable_cost * component.volume
            total = fixed + variable

            type_costs[comp_type]["fixed"] += fixed
            type_costs[comp_type]["variable"] += variable
            type_costs[comp_type]["total"] += total

        # 计算总成本
        total_cost = sum(c["total"] for c in type_costs.values())

        # 添加占比
        for comp_type, costs in type_costs.items():
            costs["percentage"] = costs["total"] / total_cost if total_cost > 0 else 0

        breakdown["by_type"] = type_costs
        breakdown["total_cost"] = total_cost

        return breakdown

    def optimize_volume_allocation(
        self, budget: float, priorities: Dict[str, float]
    ) -> Dict[str, float]:
        """在预算约束下优化使用量分配

        Args:
            budget: 预算
            priorities: 优先级字典 {component_id: priority}

        Returns:
            优化后的使用量分配
        """
        # 计算固定成本
        total_fixed_cost = sum(c.fixed_cost for c in self.cost_components)
        remaining_budget = budget - total_fixed_cost

        if remaining_budget <= 0:
            logger.warning("Budget insufficient for fixed costs")
            return {c.component_id: 0.0 for c in self.cost_components}

        # 归一化优先级
        total_priority = sum(priorities.values())
        if total_priority == 0:
            logger.warning("Total priority is zero")
            return {c.component_id: 0.0 for c in self.cost_components}

        normalized_priorities = {k: v / total_priority for k, v in priorities.items()}

        # 按优先级分配预算
        allocation = {}

        for component in self.cost_components:
            priority = normalized_priorities.get(component.component_id, 0)
            allocated_budget = remaining_budget * priority

            if component.variable_cost > 0:
                volume = allocated_budget / component.variable_cost
            else:
                volume = 0.0

            allocation[component.component_id] = volume

        logger.info(f"Optimized volume allocation within budget {budget}")
        return allocation

    def recommend_cost_reduction(self, target_reduction: float) -> List[Dict[str, any]]:
        """推荐成本削减方案

        Args:
            target_reduction: 目标削减金额

        Returns:
            削减建议列表
        """
        recommendations = []

        # 计算当前成本
        current_cost = self.calculate_total_cost()

        # 计算每个组件的成本效率（成本/单位）
        for component in self.cost_components:
            if component.volume > 0:
                unit_cost = (
                    component.fixed_cost + component.variable_cost * component.volume
                ) / component.volume

                # 计算可以削减的最大金额
                max_reduction = component.variable_cost * component.volume

                if max_reduction > 0:
                    recommendations.append(
                        {
                            "component_id": component.component_id,
                            "component_type": component.component_type,
                            "current_volume": component.volume,
                            "unit_cost": unit_cost,
                            "max_reduction": max_reduction,
                            "reduction_percentage": max_reduction / current_cost,
                        }
                    )

        # 按单位成本降序排序（优先削减成本高的）
        recommendations.sort(key=lambda x: x["unit_cost"], reverse=True)

        # 标记建议
        cumulative_reduction = 0.0
        for i, rec in enumerate(recommendations):
            if cumulative_reduction < target_reduction:
                needed_reduction = min(
                    target_reduction - cumulative_reduction, rec["max_reduction"]
                )
                rec["recommended_reduction"] = needed_reduction
                rec["recommended_volume_reduction"] = (
                    needed_reduction
                    / self.component_dict[rec["component_id"]].variable_cost
                    if self.component_dict[rec["component_id"]].variable_cost > 0
                    else 0
                )
                cumulative_reduction += needed_reduction
            else:
                rec["recommended_reduction"] = 0.0
                rec["recommended_volume_reduction"] = 0.0

        logger.info(f"Generated {len(recommendations)} cost reduction recommendations")
        return recommendations

    def forecast_cost(
        self, volume_growth: Dict[str, float], periods: int
    ) -> List[float]:
        """预测未来成本

        Args:
            volume_growth: 使用量增长率字典
            periods: 预测期数

        Returns:
            预测成本列表
        """
        forecasts = []

        for period in range(periods):
            period_cost = 0.0

            for component in self.cost_components:
                growth_rate = volume_growth.get(component.component_id, 0)
                projected_volume = component.volume * (1 + growth_rate) ** period

                cost = component.fixed_cost + component.variable_cost * projected_volume
                period_cost += cost

            forecasts.append(period_cost)

        logger.info(f"Forecasted costs for {periods} periods")
        return forecasts

    def update_volumes(self, new_volumes: Dict[str, float]) -> None:
        """更新使用量

        Args:
            new_volumes: 新使用量字典
        """
        for component_id, volume in new_volumes.items():
            if component_id in self.component_dict:
                self.component_dict[component_id].volume = volume
