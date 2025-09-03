"""
执行算法模块
实现各种高级交易执行算法
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from common.data_structures import MarketData
from common.exceptions import ExecutionError
from common.logging_system import setup_logger

logger = setup_logger("execution_algorithms")


@dataclass
class ExecutionPlan:
    """执行计划"""

    algorithm: str
    symbol: str
    total_quantity: int
    start_time: datetime
    end_time: datetime
    slices: List[Dict[str, Any]]  # 执行切片
    constraints: Dict[str, Any]
    estimated_cost: float
    estimated_impact: float


@dataclass
class ExecutionSlice:
    """执行切片"""

    slice_id: int
    timestamp: datetime
    quantity: int
    price_limit: Optional[float]
    urgency: float  # 0-1
    participation_rate: float  # 市场参与率


class ExecutionAlgorithm(ABC):
    """执行算法基类"""

    def __init__(self, config: Dict[str, Any]):
        """初始化执行算法

        Args:
            config: 配置字典
        """
        self.config = config
        self.max_participation_rate = config.get("max_participation_rate", 0.1)
        self.min_slice_size = config.get("min_slice_size", 100)
        self.price_improvement_bps = config.get("price_improvement_bps", 2)

    @abstractmethod
    def create_execution_plan(
        self,
        symbol: str,
        quantity: int,
        side: str,
        market_data: MarketData,
        constraints: Dict[str, Any],
    ) -> ExecutionPlan:
        """创建执行计划

        Args:
            symbol: 标的代码
            quantity: 总数量
            side: 买卖方向
            market_data: 市场数据
            constraints: 约束条件

        Returns:
            执行计划
        """
        pass

    @abstractmethod
    def update_execution_plan(
        self, plan: ExecutionPlan, market_data: MarketData, filled_quantity: int
    ) -> ExecutionPlan:
        """更新执行计划

        Args:
            plan: 原执行计划
            market_data: 最新市场数据
            filled_quantity: 已成交数量

        Returns:
            更新后的执行计划
        """
        pass

    def estimate_market_impact(
        self, symbol: str, quantity: int, avg_volume: float, volatility: float
    ) -> float:
        """估算市场冲击

        Args:
            symbol: 标的代码
            quantity: 数量
            avg_volume: 平均成交量
            volatility: 波动率

        Returns:
            预期市场冲击（基点）
        """
        if avg_volume == 0:
            return 0.0

        # 使用平方根模型估算市场冲击
        participation = quantity / avg_volume
        impact_bps = 10 * np.sqrt(participation) * volatility

        return impact_bps

    def calculate_optimal_slice_size(
        self, total_quantity: int, market_volume: float, urgency: float
    ) -> int:
        """计算最优切片大小

        Args:
            total_quantity: 总数量
            market_volume: 市场成交量
            urgency: 紧急程度

        Returns:
            切片大小
        """
        # 基于市场参与率计算
        max_slice = int(market_volume * self.max_participation_rate)

        # 根据紧急程度调整
        urgency_factor = 0.5 + urgency * 1.5  # 0.5x到2x
        optimal_slice = int(max_slice * urgency_factor)

        # 应用最小限制
        optimal_slice = max(optimal_slice, self.min_slice_size)

        # 不超过总量
        optimal_slice = min(optimal_slice, total_quantity)

        return optimal_slice


class TWAPAlgorithm(ExecutionAlgorithm):
    """时间加权平均价格算法"""

    def create_execution_plan(
        self,
        symbol: str,
        quantity: int,
        side: str,
        market_data: MarketData,
        constraints: Dict[str, Any],
    ) -> ExecutionPlan:
        """创建TWAP执行计划"""

        # 获取时间窗口
        duration_minutes = constraints.get("duration_minutes", 60)
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        # 计算切片数量
        num_slices = max(duration_minutes // 5, 1)  # 每5分钟一个切片
        slice_quantity = quantity // num_slices
        remaining = quantity % num_slices

        # 创建执行切片
        slices = []
        for i in range(num_slices):
            slice_time = start_time + timedelta(minutes=i * 5)
            slice_qty = slice_quantity
            if i == num_slices - 1:  # 最后一个切片包含剩余量
                slice_qty += remaining

            slices.append(
                {
                    "slice_id": i,
                    "timestamp": slice_time,
                    "quantity": slice_qty,
                    "participation_rate": self.max_participation_rate,
                }
            )

        # 估算成本和冲击
        estimated_impact = self.estimate_market_impact(
            symbol,
            quantity,
            market_data.volume * num_slices,
            0.02,  # 假设2%日波动率
        )

        return ExecutionPlan(
            algorithm="TWAP",
            symbol=symbol,
            total_quantity=quantity,
            start_time=start_time,
            end_time=end_time,
            slices=slices,
            constraints=constraints,
            estimated_cost=quantity * market_data.close * 0.0005,  # 5bps成本
            estimated_impact=estimated_impact,
        )

    def update_execution_plan(
        self, plan: ExecutionPlan, market_data: MarketData, filled_quantity: int
    ) -> ExecutionPlan:
        """更新TWAP执行计划"""

        # 计算剩余数量
        remaining_quantity = plan.total_quantity - filled_quantity

        if remaining_quantity <= 0:
            return plan

        # 计算剩余时间
        time_elapsed = datetime.now() - plan.start_time
        time_remaining = plan.end_time - datetime.now()

        if time_remaining.total_seconds() <= 0:
            # 时间用完，最后一个切片执行所有剩余
            plan.slices = [
                {
                    "slice_id": 999,
                    "timestamp": datetime.now(),
                    "quantity": remaining_quantity,
                    "participation_rate": min(0.2, self.max_participation_rate * 2),
                }
            ]
        else:
            # 重新分配剩余切片
            remaining_slices = [
                s for s in plan.slices if s["timestamp"] > datetime.now()
            ]
            num_remaining = len(remaining_slices)

            if num_remaining > 0:
                qty_per_slice = remaining_quantity // num_remaining
                remainder = remaining_quantity % num_remaining

                for i, slice_info in enumerate(remaining_slices):
                    slice_info["quantity"] = qty_per_slice
                    if i == num_remaining - 1:
                        slice_info["quantity"] += remainder

        return plan


class VWAPAlgorithm(ExecutionAlgorithm):
    """成交量加权平均价格算法"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.volume_profile = self._load_volume_profile()

    def _load_volume_profile(self) -> Dict[int, float]:
        """加载历史成交量分布

        Returns:
            时段到成交量占比的映射
        """
        # 标准市场成交量分布（U型）
        profile = {
            9: 0.15,  # 9:30-10:30 高
            10: 0.08,  # 10:30-11:30 中
            11: 0.06,  # 11:30-12:30 低
            13: 0.08,  # 13:00-14:00 中
            14: 0.13,  # 14:00-15:00 高
            15: 0.50,  # 尾盘集中
        }
        return profile

    def create_execution_plan(
        self,
        symbol: str,
        quantity: int,
        side: str,
        market_data: MarketData,
        constraints: Dict[str, Any],
    ) -> ExecutionPlan:
        """创建VWAP执行计划"""

        start_time = datetime.now()
        end_time = start_time + timedelta(hours=6)  # 交易日结束

        # 根据成交量分布创建切片
        slices = []
        current_hour = start_time.hour

        for hour, volume_pct in self.volume_profile.items():
            if hour >= current_hour:
                slice_time = start_time.replace(hour=hour, minute=0, second=0)
                slice_quantity = int(quantity * volume_pct)

                slices.append(
                    {
                        "slice_id": hour,
                        "timestamp": slice_time,
                        "quantity": slice_quantity,
                        "participation_rate": self.max_participation_rate,
                        "volume_weight": volume_pct,
                    }
                )

        # 确保总量匹配
        total_sliced = sum(s["quantity"] for s in slices)
        if total_sliced < quantity and slices:
            slices[-1]["quantity"] += quantity - total_sliced

        estimated_impact = self.estimate_market_impact(
            symbol, quantity, market_data.volume, 0.02
        )

        return ExecutionPlan(
            algorithm="VWAP",
            symbol=symbol,
            total_quantity=quantity,
            start_time=start_time,
            end_time=end_time,
            slices=slices,
            constraints=constraints,
            estimated_cost=quantity * market_data.close * 0.0003,  # 3bps成本
            estimated_impact=estimated_impact,
        )

    def update_execution_plan(
        self, plan: ExecutionPlan, market_data: MarketData, filled_quantity: int
    ) -> ExecutionPlan:
        """更新VWAP执行计划"""

        remaining_quantity = plan.total_quantity - filled_quantity

        if remaining_quantity <= 0:
            return plan

        # 获取当日实际成交量分布
        current_hour = datetime.now().hour
        actual_volume = market_data.volume

        # 调整剩余切片的数量
        future_slices = [s for s in plan.slices if s["timestamp"] > datetime.now()]

        if future_slices:
            # 基于实际成交量调整
            total_weight = sum(s["volume_weight"] for s in future_slices)

            for slice_info in future_slices:
                adjusted_pct = (
                    slice_info["volume_weight"] / total_weight
                    if total_weight > 0
                    else 1.0
                )
                slice_info["quantity"] = int(remaining_quantity * adjusted_pct)

        return plan


class ImplementationShortfallAlgorithm(ExecutionAlgorithm):
    """实施缺口算法"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.risk_aversion = config.get("risk_aversion", 0.5)

    def create_execution_plan(
        self,
        symbol: str,
        quantity: int,
        side: str,
        market_data: MarketData,
        constraints: Dict[str, Any],
    ) -> ExecutionPlan:
        """创建IS执行计划"""

        # 计算最优执行轨迹
        urgency = constraints.get("urgency", 0.5)
        duration_minutes = self._calculate_optimal_duration(
            quantity, market_data.volume, urgency
        )

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        # 前载执行策略（早期执行更多）
        num_slices = min(duration_minutes // 2, 30)
        slices = []

        for i in range(num_slices):
            # 指数递减的执行量
            decay_factor = np.exp(-i * 0.1)
            slice_pct = decay_factor / sum(np.exp(-j * 0.1) for j in range(num_slices))

            slice_time = start_time + timedelta(minutes=i * 2)
            slice_quantity = int(quantity * slice_pct)

            # 动态调整参与率
            participation_rate = self.max_participation_rate * (1 + urgency)

            slices.append(
                {
                    "slice_id": i,
                    "timestamp": slice_time,
                    "quantity": slice_quantity,
                    "participation_rate": min(participation_rate, 0.2),
                    "urgency": urgency * decay_factor,
                }
            )

        # 估算成本
        estimated_cost = self._calculate_is_cost(
            quantity, market_data.close, market_data.volume, duration_minutes
        )

        estimated_impact = self.estimate_market_impact(
            symbol, quantity, market_data.volume, 0.02
        )

        return ExecutionPlan(
            algorithm="IS",
            symbol=symbol,
            total_quantity=quantity,
            start_time=start_time,
            end_time=end_time,
            slices=slices,
            constraints=constraints,
            estimated_cost=estimated_cost,
            estimated_impact=estimated_impact,
        )

    def update_execution_plan(
        self, plan: ExecutionPlan, market_data: MarketData, filled_quantity: int
    ) -> ExecutionPlan:
        """更新IS执行计划"""

        remaining_quantity = plan.total_quantity - filled_quantity

        if remaining_quantity <= 0:
            return plan

        # 计算实际vs预期执行进度
        time_elapsed = (datetime.now() - plan.start_time).total_seconds() / 60
        total_duration = (plan.end_time - plan.start_time).total_seconds() / 60

        if total_duration > 0:
            time_progress = time_elapsed / total_duration
            execution_progress = filled_quantity / plan.total_quantity

            # 如果落后于计划，增加紧急度
            if execution_progress < time_progress * 0.9:
                for slice_info in plan.slices:
                    if slice_info["timestamp"] > datetime.now():
                        slice_info["urgency"] = min(
                            slice_info.get("urgency", 0.5) * 1.2, 1.0
                        )
                        slice_info["participation_rate"] = min(
                            slice_info["participation_rate"] * 1.1, 0.25
                        )

        return plan

    def _calculate_optimal_duration(
        self, quantity: int, avg_volume: float, urgency: float
    ) -> int:
        """计算最优执行时长

        Args:
            quantity: 数量
            avg_volume: 平均成交量
            urgency: 紧急程度

        Returns:
            最优时长（分钟）
        """
        # 基于参与率的基础时长
        base_duration = quantity / (avg_volume * self.max_participation_rate / 390)

        # 根据紧急程度调整
        urgency_factor = 2.0 - urgency * 1.5  # 高紧急度缩短时间
        optimal_duration = base_duration * urgency_factor

        # 限制在合理范围内
        return int(max(5, min(optimal_duration, 240)))

    def _calculate_is_cost(
        self, quantity: int, price: float, volume: float, duration: int
    ) -> float:
        """计算IS成本

        Args:
            quantity: 数量
            price: 价格
            volume: 成交量
            duration: 时长

        Returns:
            预期成本
        """
        # 简化的IS成本模型
        spread_cost = quantity * price * 0.0002  # 2bps价差成本
        impact_cost = quantity * price * 0.0001 * np.sqrt(quantity / volume)
        timing_risk = quantity * price * 0.0001 * np.sqrt(duration / 60)

        total_cost = spread_cost + impact_cost + timing_risk * self.risk_aversion

        return total_cost


class AdaptiveAlgorithm(ExecutionAlgorithm):
    """自适应执行算法"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.algorithms = {
            "TWAP": TWAPAlgorithm(config),
            "VWAP": VWAPAlgorithm(config),
            "IS": ImplementationShortfallAlgorithm(config),
        }

    def create_execution_plan(
        self,
        symbol: str,
        quantity: int,
        side: str,
        market_data: MarketData,
        constraints: Dict[str, Any],
    ) -> ExecutionPlan:
        """创建自适应执行计划"""

        # 选择最优算法
        best_algorithm = self._select_optimal_algorithm(
            symbol, quantity, market_data, constraints
        )

        logger.info(f"Selected algorithm: {best_algorithm}")

        # 使用选定算法创建计划
        return self.algorithms[best_algorithm].create_execution_plan(
            symbol, quantity, side, market_data, constraints
        )

    def update_execution_plan(
        self, plan: ExecutionPlan, market_data: MarketData, filled_quantity: int
    ) -> ExecutionPlan:
        """更新自适应执行计划"""

        # 评估当前执行效果
        performance = self._evaluate_execution_performance(
            plan, market_data, filled_quantity
        )

        # 如果表现不佳，考虑切换算法
        if performance < 0.3:  # 表现差
            remaining_quantity = plan.total_quantity - filled_quantity

            # 重新选择算法
            new_algorithm = self._select_optimal_algorithm(
                plan.symbol, remaining_quantity, market_data, plan.constraints
            )

            if new_algorithm != plan.algorithm:
                logger.info(
                    f"Switching algorithm from {plan.algorithm} to {new_algorithm}"
                )

                # 创建新计划
                new_plan = self.algorithms[new_algorithm].create_execution_plan(
                    plan.symbol,
                    remaining_quantity,
                    "BUY",  # 需要从plan中获取
                    market_data,
                    plan.constraints,
                )

                return new_plan

        # 否则更新现有计划
        algorithm = self.algorithms.get(plan.algorithm)
        if algorithm:
            return algorithm.update_execution_plan(plan, market_data, filled_quantity)

        return plan

    def _select_optimal_algorithm(
        self,
        symbol: str,
        quantity: int,
        market_data: MarketData,
        constraints: Dict[str, Any],
    ) -> str:
        """选择最优算法

        Args:
            symbol: 标的代码
            quantity: 数量
            market_data: 市场数据
            constraints: 约束条件

        Returns:
            算法名称
        """
        urgency = constraints.get("urgency", 0.5)
        minimize_impact = constraints.get("minimize_impact", False)
        target_benchmark = constraints.get("target_benchmark", "arrival")

        # 基于约束条件选择
        if urgency > 0.7:
            return "IS"  # 高紧急度用IS
        elif minimize_impact:
            return "TWAP"  # 最小化冲击用TWAP
        elif target_benchmark == "vwap":
            return "VWAP"  # 目标是VWAP
        else:
            # 根据市场条件选择
            if market_data.volume < 1000000:
                return "TWAP"  # 低流动性用TWAP
            else:
                return "VWAP"  # 高流动性用VWAP

    def _evaluate_execution_performance(
        self, plan: ExecutionPlan, market_data: MarketData, filled_quantity: int
    ) -> float:
        """评估执行表现

        Args:
            plan: 执行计划
            market_data: 市场数据
            filled_quantity: 已成交数量

        Returns:
            表现分数（0-1）
        """
        if plan.total_quantity == 0:
            return 1.0

        # 执行进度
        progress = filled_quantity / plan.total_quantity

        # 时间进度
        time_elapsed = (datetime.now() - plan.start_time).total_seconds()
        total_time = (plan.end_time - plan.start_time).total_seconds()
        time_progress = time_elapsed / total_time if total_time > 0 else 1.0

        # 计算表现分数
        if progress >= time_progress * 0.9:
            return 1.0  # 按计划或超前
        else:
            return progress / (time_progress * 0.9)  # 落后程度


# 算法工厂函数
def create_execution_algorithm(
    algorithm_type: str, config: Dict[str, Any]
) -> ExecutionAlgorithm:
    """创建执行算法实例

    Args:
        algorithm_type: 算法类型
        config: 配置字典

    Returns:
        执行算法实例
    """
    algorithms = {
        "TWAP": TWAPAlgorithm,
        "VWAP": VWAPAlgorithm,
        "IS": ImplementationShortfallAlgorithm,
        "ADAPTIVE": AdaptiveAlgorithm,
    }

    if algorithm_type not in algorithms:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")

    return algorithms[algorithm_type](config)


# 执行质量分析
def analyze_execution_quality(
    executed_orders: List[Dict[str, Any]], market_data: pd.DataFrame
) -> Dict[str, float]:
    """分析执行质量

    Args:
        executed_orders: 已执行订单列表
        market_data: 市场数据

    Returns:
        质量指标字典
    """
    if not executed_orders:
        return {}

    metrics = {}

    # 计算平均滑点
    slippages = []
    for order in executed_orders:
        if "arrival_price" in order and "avg_fill_price" in order:
            if order["side"] == "BUY":
                slippage = order["avg_fill_price"] - order["arrival_price"]
            else:
                slippage = order["arrival_price"] - order["avg_fill_price"]
            slippages.append(slippage / order["arrival_price"])

    metrics["avg_slippage_bps"] = np.mean(slippages) * 10000 if slippages else 0

    # 计算VWAP表现
    if not market_data.empty:
        vwap = (market_data["price"] * market_data["volume"]).sum() / market_data[
            "volume"
        ].sum()

        vwap_diff = []
        for order in executed_orders:
            if "avg_fill_price" in order:
                diff = (order["avg_fill_price"] - vwap) / vwap
                if order["side"] == "SELL":
                    diff = -diff
                vwap_diff.append(diff)

        metrics["vwap_performance_bps"] = np.mean(vwap_diff) * 10000 if vwap_diff else 0

    # 计算成交率
    fill_rates = []
    for order in executed_orders:
        if "quantity" in order and "filled_quantity" in order:
            fill_rate = order["filled_quantity"] / order["quantity"]
            fill_rates.append(fill_rate)

    metrics["avg_fill_rate"] = np.mean(fill_rates) if fill_rates else 0

    return metrics
