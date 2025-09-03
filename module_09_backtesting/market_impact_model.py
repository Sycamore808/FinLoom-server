"""
市场冲击模型模块
实现多种市场冲击模型，包括线性、平方根、Almgren-Chriss等
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from common.constants import TRADING_DAYS_PER_YEAR
from common.logging_system import setup_logger
from scipy.optimize import minimize

logger = setup_logger("market_impact_model")


@dataclass
class ImpactParameters:
    """市场冲击参数数据类"""

    permanent_impact: float  # 永久冲击系数
    temporary_impact: float  # 临时冲击系数
    decay_rate: float  # 冲击衰减率
    volatility: float  # 波动率
    daily_volume: float  # 日均成交量
    spread: float  # 买卖价差

    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "permanent_impact": self.permanent_impact,
            "temporary_impact": self.temporary_impact,
            "decay_rate": self.decay_rate,
            "volatility": self.volatility,
            "daily_volume": self.daily_volume,
            "spread": self.spread,
        }


@dataclass
class ImpactEstimate:
    """市场冲击估算结果"""

    total_impact: float  # 总冲击（基点）
    permanent_component: float  # 永久部分
    temporary_component: float  # 临时部分
    execution_cost: float  # 执行成本
    optimal_trajectory: Optional[List[float]]  # 最优执行轨迹
    confidence_interval: Tuple[float, float]  # 置信区间

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_impact": self.total_impact,
            "permanent_component": self.permanent_component,
            "temporary_component": self.temporary_component,
            "execution_cost": self.execution_cost,
            "optimal_trajectory": self.optimal_trajectory,
            "confidence_interval": self.confidence_interval,
        }


class MarketImpactModel(ABC):
    """市场冲击模型基类"""

    @abstractmethod
    def estimate_impact(
        self, order_size: float, parameters: ImpactParameters
    ) -> ImpactEstimate:
        """估算市场冲击

        Args:
            order_size: 订单大小
            parameters: 冲击参数

        Returns:
            冲击估算结果
        """
        pass

    @abstractmethod
    def calibrate(self, historical_trades: pd.DataFrame) -> ImpactParameters:
        """校准模型参数

        Args:
            historical_trades: 历史交易数据

        Returns:
            校准后的参数
        """
        pass


class LinearImpactModel(MarketImpactModel):
    """线性市场冲击模型"""

    def __init__(self):
        """初始化线性冲击模型"""
        self.name = "Linear Impact Model"

    def estimate_impact(
        self, order_size: float, parameters: ImpactParameters
    ) -> ImpactEstimate:
        """估算市场冲击（线性模型）

        Args:
            order_size: 订单大小
            parameters: 冲击参数

        Returns:
            冲击估算结果
        """
        # 计算参与率
        participation_rate = order_size / parameters.daily_volume

        # 线性永久冲击
        permanent = parameters.permanent_impact * participation_rate * 10000

        # 线性临时冲击
        temporary = parameters.temporary_impact * participation_rate * 10000

        # 总冲击
        total = permanent + temporary

        # 执行成本（价格的百分比）
        execution_cost = total / 10000

        # 置信区间（基于历史波动率）
        std_impact = parameters.volatility * np.sqrt(participation_rate) * 10000
        confidence_interval = (total - 2 * std_impact, total + 2 * std_impact)

        return ImpactEstimate(
            total_impact=total,
            permanent_component=permanent,
            temporary_component=temporary,
            execution_cost=execution_cost,
            optimal_trajectory=None,
            confidence_interval=confidence_interval,
        )

    def calibrate(self, historical_trades: pd.DataFrame) -> ImpactParameters:
        """校准线性模型参数

        Args:
            historical_trades: 历史交易数据

        Returns:
            校准后的参数
        """
        # 计算平均日成交量
        daily_volume = historical_trades["volume"].mean()

        # 计算波动率
        returns = historical_trades["price"].pct_change()
        volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # 计算价差
        if "bid" in historical_trades.columns and "ask" in historical_trades.columns:
            spread = (historical_trades["ask"] - historical_trades["bid"]).mean()
        else:
            spread = historical_trades["price"].std() * 0.001  # 估算

        # 通过回归估算冲击系数
        if (
            "impact" in historical_trades.columns
            and "participation_rate" in historical_trades.columns
        ):
            # 线性回归
            X = historical_trades["participation_rate"].values
            y = historical_trades["impact"].values

            # 最小二乘法
            coefficients = np.polyfit(X, y, 1)
            permanent_impact = coefficients[0] * 0.7  # 70%为永久
            temporary_impact = coefficients[0] * 0.3  # 30%为临时
        else:
            # 使用默认值
            permanent_impact = 0.1
            temporary_impact = 0.05

        return ImpactParameters(
            permanent_impact=permanent_impact,
            temporary_impact=temporary_impact,
            decay_rate=0.5,
            volatility=volatility,
            daily_volume=daily_volume,
            spread=spread,
        )


class SquareRootImpactModel(MarketImpactModel):
    """平方根市场冲击模型"""

    def __init__(self):
        """初始化平方根冲击模型"""
        self.name = "Square Root Impact Model"

    def estimate_impact(
        self, order_size: float, parameters: ImpactParameters
    ) -> ImpactEstimate:
        """估算市场冲击（平方根模型）

        Args:
            order_size: 订单大小
            parameters: 冲击参数

        Returns:
            冲击估算结果
        """
        # 计算参与率
        participation_rate = order_size / parameters.daily_volume

        # 平方根永久冲击
        permanent = parameters.permanent_impact * np.sqrt(participation_rate) * 10000

        # 平方根临时冲击
        temporary = parameters.temporary_impact * np.sqrt(participation_rate) * 10000

        # 总冲击
        total = permanent + temporary

        # 执行成本
        execution_cost = total / 10000

        # 置信区间
        std_impact = parameters.volatility * participation_rate**0.25 * 10000
        confidence_interval = (total - 2 * std_impact, total + 2 * std_impact)

        return ImpactEstimate(
            total_impact=total,
            permanent_component=permanent,
            temporary_component=temporary,
            execution_cost=execution_cost,
            optimal_trajectory=None,
            confidence_interval=confidence_interval,
        )

    def calibrate(self, historical_trades: pd.DataFrame) -> ImpactParameters:
        """校准平方根模型参数

        Args:
            historical_trades: 历史交易数据

        Returns:
            校准后的参数
        """
        # 基本参数计算
        daily_volume = historical_trades["volume"].mean()
        returns = historical_trades["price"].pct_change()
        volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # 价差估算
        if "spread" in historical_trades.columns:
            spread = historical_trades["spread"].mean()
        else:
            spread = historical_trades["price"].std() * 0.001

        # 非线性回归估算冲击系数
        if (
            "impact" in historical_trades.columns
            and "participation_rate" in historical_trades.columns
        ):
            X = historical_trades["participation_rate"].values
            y = historical_trades["impact"].values

            # 定义目标函数
            def objective(params):
                permanent, temporary = params
                predicted = permanent * np.sqrt(X) + temporary * np.sqrt(X)
                return np.sum((y - predicted) ** 2)

            # 优化
            result = minimize(objective, x0=[0.1, 0.05], bounds=[(0, 1), (0, 1)])
            permanent_impact = result.x[0]
            temporary_impact = result.x[1]
        else:
            permanent_impact = 0.15
            temporary_impact = 0.08

        return ImpactParameters(
            permanent_impact=permanent_impact,
            temporary_impact=temporary_impact,
            decay_rate=0.5,
            volatility=volatility,
            daily_volume=daily_volume,
            spread=spread,
        )


class AlmgrenChrissModel(MarketImpactModel):
    """Almgren-Chriss市场冲击模型"""

    def __init__(self):
        """初始化Almgren-Chriss模型"""
        self.name = "Almgren-Chriss Model"
        self.risk_aversion = 1e-6  # 风险厌恶参数

    def estimate_impact(
        self,
        order_size: float,
        parameters: ImpactParameters,
        time_horizon: int = 1,
        n_slices: int = 10,
    ) -> ImpactEstimate:
        """估算市场冲击（Almgren-Chriss模型）

        Args:
            order_size: 订单大小
            parameters: 冲击参数
            time_horizon: 执行时间（天）
            n_slices: 时间切片数

        Returns:
            冲击估算结果
        """
        # 时间间隔
        tau = time_horizon / n_slices

        # 计算最优执行轨迹
        trajectory = self._compute_optimal_trajectory(
            order_size, parameters, time_horizon, n_slices
        )

        # 计算各部分冲击
        permanent = 0.0
        temporary = 0.0

        for i, trade_size in enumerate(trajectory):
            # 交易速率
            trade_rate = trade_size / tau
            participation_rate = trade_rate / parameters.daily_volume

            # 永久冲击累积
            permanent += (
                parameters.permanent_impact * trade_size / parameters.daily_volume
            )

            # 临时冲击
            temporary += parameters.temporary_impact * np.sqrt(
                trade_rate / parameters.daily_volume
            )

        # 转换为基点
        permanent *= 10000
        temporary *= 10000
        total = permanent + temporary

        # 执行成本
        execution_cost = total / 10000 + parameters.spread * n_slices

        # 置信区间（考虑执行风险）
        execution_risk = (
            parameters.volatility
            * np.sqrt(time_horizon)
            * order_size
            / parameters.daily_volume
        )
        confidence_interval = (
            total - 2 * execution_risk * 10000,
            total + 2 * execution_risk * 10000,
        )

        return ImpactEstimate(
            total_impact=total,
            permanent_component=permanent,
            temporary_component=temporary,
            execution_cost=execution_cost,
            optimal_trajectory=trajectory,
            confidence_interval=confidence_interval,
        )

    def _compute_optimal_trajectory(
        self,
        order_size: float,
        parameters: ImpactParameters,
        time_horizon: int,
        n_slices: int,
    ) -> List[float]:
        """计算最优执行轨迹

        Args:
            order_size: 订单大小
            parameters: 冲击参数
            time_horizon: 执行时间
            n_slices: 时间切片数

        Returns:
            最优执行轨迹
        """
        # 简化的最优轨迹（指数衰减）
        kappa = 2 * self.risk_aversion * parameters.volatility**2

        trajectory = []
        remaining = order_size

        for i in range(n_slices):
            # 指数衰减策略
            fraction = np.exp(-kappa * i / n_slices)
            trade_size = remaining * fraction / (n_slices - i)

            # 确保不超过剩余量
            trade_size = min(trade_size, remaining)
            trajectory.append(trade_size)
            remaining -= trade_size

        # 确保全部执行
        if remaining > 0:
            trajectory[-1] += remaining

        return trajectory

    def calibrate(self, historical_trades: pd.DataFrame) -> ImpactParameters:
        """校准Almgren-Chriss模型参数

        Args:
            historical_trades: 历史交易数据

        Returns:
            校准后的参数
        """
        # 基本统计
        daily_volume = historical_trades["volume"].mean()
        returns = historical_trades["price"].pct_change()
        volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # 微观结构参数
        if "spread" in historical_trades.columns:
            spread = historical_trades["spread"].mean()
        else:
            # 使用Roll's spread estimator
            price_changes = historical_trades["price"].diff()
            spread = 2 * np.sqrt(-np.cov(price_changes[1:], price_changes[:-1])[0, 1])

        # 冲击参数估算
        if "execution_cost" in historical_trades.columns:
            # 使用实际执行成本校准
            costs = historical_trades["execution_cost"].values
            sizes = historical_trades["trade_size"].values
            rates = sizes / daily_volume

            # 分离永久和临时冲击
            def objective(params):
                perm, temp = params
                predicted_costs = perm * rates + temp * np.sqrt(rates)
                return np.sum((costs - predicted_costs) ** 2)

            result = minimize(objective, x0=[0.1, 0.1], bounds=[(0, 1), (0, 1)])
            permanent_impact = result.x[0]
            temporary_impact = result.x[1]
        else:
            # 使用经验值
            permanent_impact = volatility * 0.5
            temporary_impact = spread * 0.3

        # 衰减率（基于自相关）
        autocorr = returns.autocorr()
        decay_rate = max(0.1, 1 - abs(autocorr))

        return ImpactParameters(
            permanent_impact=permanent_impact,
            temporary_impact=temporary_impact,
            decay_rate=decay_rate,
            volatility=volatility,
            daily_volume=daily_volume,
            spread=spread,
        )


class MarketImpactEstimator:
    """市场冲击估算器（集成多个模型）"""

    def __init__(self):
        """初始化市场冲击估算器"""
        self.models = {
            "linear": LinearImpactModel(),
            "sqrt": SquareRootImpactModel(),
            "almgren_chriss": AlmgrenChrissModel(),
        }
        self.calibrated_parameters: Dict[str, ImpactParameters] = {}

    def calibrate_all_models(
        self, historical_trades: pd.DataFrame
    ) -> Dict[str, ImpactParameters]:
        """校准所有模型

        Args:
            historical_trades: 历史交易数据

        Returns:
            各模型的校准参数
        """
        for name, model in self.models.items():
            logger.info(f"Calibrating {name} model...")
            parameters = model.calibrate(historical_trades)
            self.calibrated_parameters[name] = parameters

        return self.calibrated_parameters

    def estimate_ensemble(
        self, order_size: float, symbol: str, use_models: Optional[List[str]] = None
    ) -> Dict[str, ImpactEstimate]:
        """使用多个模型估算冲击

        Args:
            order_size: 订单大小
            symbol: 标的代码
            use_models: 使用的模型列表

        Returns:
            各模型的估算结果
        """
        if use_models is None:
            use_models = list(self.models.keys())

        results = {}

        for model_name in use_models:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not found")
                continue

            model = self.models[model_name]

            # 获取参数
            if model_name in self.calibrated_parameters:
                parameters = self.calibrated_parameters[model_name]
            else:
                # 使用默认参数
                parameters = ImpactParameters(
                    permanent_impact=0.1,
                    temporary_impact=0.05,
                    decay_rate=0.5,
                    volatility=0.02,
                    daily_volume=1000000,
                    spread=0.001,
                )

            # 估算冲击
            estimate = model.estimate_impact(order_size, parameters)
            results[model_name] = estimate

        return results

    def get_consensus_estimate(
        self,
        estimates: Dict[str, ImpactEstimate],
        weights: Optional[Dict[str, float]] = None,
    ) -> ImpactEstimate:
        """获取共识估算（加权平均）

        Args:
            estimates: 各模型估算结果
            weights: 模型权重

        Returns:
            共识估算结果
        """
        if weights is None:
            # 平均权重
            weights = {name: 1.0 / len(estimates) for name in estimates}

        # 归一化权重
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # 加权平均
        total_impact = sum(
            estimates[name].total_impact * weights.get(name, 0) for name in estimates
        )

        permanent = sum(
            estimates[name].permanent_component * weights.get(name, 0)
            for name in estimates
        )

        temporary = sum(
            estimates[name].temporary_component * weights.get(name, 0)
            for name in estimates
        )

        execution_cost = sum(
            estimates[name].execution_cost * weights.get(name, 0) for name in estimates
        )

        # 置信区间（取最宽的）
        lower_bounds = [est.confidence_interval[0] for est in estimates.values()]
        upper_bounds = [est.confidence_interval[1] for est in estimates.values()]

        confidence_interval = (min(lower_bounds), max(upper_bounds))

        return ImpactEstimate(
            total_impact=total_impact,
            permanent_component=permanent,
            temporary_component=temporary,
            execution_cost=execution_cost,
            optimal_trajectory=None,
            confidence_interval=confidence_interval,
        )


# 模块级别函数
def estimate_market_impact(
    order_size: float, daily_volume: float, volatility: float, model_type: str = "sqrt"
) -> float:
    """估算市场冲击的便捷函数

    Args:
        order_size: 订单大小
        daily_volume: 日均成交量
        volatility: 波动率
        model_type: 模型类型

    Returns:
        市场冲击（基点）
    """
    # 创建参数
    parameters = ImpactParameters(
        permanent_impact=0.1,
        temporary_impact=0.05,
        decay_rate=0.5,
        volatility=volatility,
        daily_volume=daily_volume,
        spread=0.001,
    )

    # 选择模型
    if model_type == "linear":
        model = LinearImpactModel()
    elif model_type == "sqrt":
        model = SquareRootImpactModel()
    elif model_type == "almgren_chriss":
        model = AlmgrenChrissModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 估算冲击
    estimate = model.estimate_impact(order_size, parameters)

    return estimate.total_impact


def calibrate_impact_model(
    historical_trades: pd.DataFrame, model_type: str = "sqrt"
) -> ImpactParameters:
    """校准冲击模型的便捷函数

    Args:
        historical_trades: 历史交易数据
        model_type: 模型类型

    Returns:
        校准后的参数
    """
    if model_type == "linear":
        model = LinearImpactModel()
    elif model_type == "sqrt":
        model = SquareRootImpactModel()
    elif model_type == "almgren_chriss":
        model = AlmgrenChrissModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.calibrate(historical_trades)
