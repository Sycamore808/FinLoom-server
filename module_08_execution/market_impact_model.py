"""
市场冲击模型模块
负责预测和估算交易对市场价格的影响
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from common.data_structures import MarketData
from common.exceptions import ExecutionError
from common.logging_system import setup_logger
from scipy import optimize

logger = setup_logger("market_impact_model")


@dataclass
class ImpactEstimate:
    """市场冲击估算结果"""

    symbol: str
    quantity: int
    side: str
    temporary_impact_bps: float  # 临时冲击（基点）
    permanent_impact_bps: float  # 永久冲击（基点）
    total_impact_bps: float  # 总冲击
    estimated_cost: float  # 预估成本
    confidence_interval: Tuple[float, float]  # 置信区间
    model_type: str  # 使用的模型类型
    parameters: Dict[str, float]  # 模型参数


@dataclass
class MarketMicrostructure:
    """市场微观结构参数"""

    symbol: str
    avg_spread_bps: float  # 平均价差（基点）
    avg_daily_volume: float  # 日均成交量
    volatility: float  # 波动率
    liquidity_score: float  # 流动性评分（0-1）
    tick_size: float  # 最小价格变动单位
    lot_size: int  # 最小交易单位
    market_depth: Dict[float, int]  # 价格深度
    order_book_imbalance: float  # 订单簿失衡度


class MarketImpactModel:
    """市场冲击模型基类"""

    def __init__(self, config: Dict[str, Any]):
        """初始化市场冲击模型

        Args:
            config: 配置字典
        """
        self.config = config
        self.calibration_window = config.get("calibration_window", 30)  # 天
        self.confidence_level = config.get("confidence_level", 0.95)
        self.model_parameters = {}

    def estimate_impact(
        self,
        symbol: str,
        quantity: int,
        side: str,
        market_data: MarketData,
        microstructure: MarketMicrostructure,
        execution_duration: Optional[int] = None,
    ) -> ImpactEstimate:
        """估算市场冲击

        Args:
            symbol: 标的代码
            quantity: 交易数量
            side: 买卖方向
            market_data: 市场数据
            microstructure: 市场微观结构
            execution_duration: 执行时长（分钟）

        Returns:
            冲击估算结果
        """
        # 计算参与率
        participation_rate = self._calculate_participation_rate(
            quantity, microstructure.avg_daily_volume, execution_duration
        )

        # 估算临时冲击
        temp_impact = self._estimate_temporary_impact(
            participation_rate, microstructure.volatility, microstructure.avg_spread_bps
        )

        # 估算永久冲击
        perm_impact = self._estimate_permanent_impact(
            participation_rate, microstructure.liquidity_score
        )

        # 计算总冲击
        total_impact = temp_impact + perm_impact

        # 方向调整（卖出时冲击为负）
        if side == "SELL":
            total_impact = -total_impact

        # 计算预估成本
        estimated_cost = quantity * market_data.close * total_impact / 10000

        # 计算置信区间
        confidence_interval = self._calculate_confidence_interval(
            total_impact, microstructure.volatility
        )

        return ImpactEstimate(
            symbol=symbol,
            quantity=quantity,
            side=side,
            temporary_impact_bps=temp_impact,
            permanent_impact_bps=perm_impact,
            total_impact_bps=total_impact,
            estimated_cost=estimated_cost,
            confidence_interval=confidence_interval,
            model_type=self.__class__.__name__,
            parameters=self.model_parameters.copy(),
        )

    def calibrate(
        self, historical_data: pd.DataFrame, execution_data: pd.DataFrame
    ) -> None:
        """校准模型参数

        Args:
            historical_data: 历史市场数据
            execution_data: 历史执行数据
        """
        logger.info("Calibrating market impact model...")

        # 提取特征
        features = self._extract_calibration_features(historical_data, execution_data)

        # 计算实际冲击
        actual_impacts = self._calculate_actual_impacts(execution_data)

        # 优化参数
        self.model_parameters = self._optimize_parameters(features, actual_impacts)

        logger.info(f"Model calibrated with parameters: {self.model_parameters}")

    def _calculate_participation_rate(
        self, quantity: int, avg_daily_volume: float, execution_duration: Optional[int]
    ) -> float:
        """计算参与率

        Args:
            quantity: 交易数量
            avg_daily_volume: 日均成交量
            execution_duration: 执行时长（分钟）

        Returns:
            参与率
        """
        if avg_daily_volume == 0:
            return 0.0

        if execution_duration:
            # 按时间段计算参与率
            minutes_in_day = 390  # 交易时间
            period_volume = avg_daily_volume * (execution_duration / minutes_in_day)
            return min(quantity / period_volume, 1.0) if period_volume > 0 else 0.0
        else:
            # 按日参与率计算
            return min(quantity / avg_daily_volume, 1.0)

    def _estimate_temporary_impact(
        self, participation_rate: float, volatility: float, spread_bps: float
    ) -> float:
        """估算临时冲击

        Args:
            participation_rate: 参与率
            volatility: 波动率
            spread_bps: 价差（基点）

        Returns:
            临时冲击（基点）
        """
        # 使用平方根模型
        alpha = self.model_parameters.get("temp_alpha", 0.1)
        beta = self.model_parameters.get("temp_beta", 0.5)

        temp_impact = (
            alpha * spread_bps + beta * volatility * np.sqrt(participation_rate) * 10000
        )

        return temp_impact

    def _estimate_permanent_impact(
        self, participation_rate: float, liquidity_score: float
    ) -> float:
        """估算永久冲击

        Args:
            participation_rate: 参与率
            liquidity_score: 流动性评分

        Returns:
            永久冲击（基点）
        """
        # 线性永久冲击模型
        gamma = self.model_parameters.get("perm_gamma", 0.05)

        # 流动性调整
        liquidity_factor = 2.0 - liquidity_score  # 流动性越差，冲击越大

        perm_impact = gamma * participation_rate * liquidity_factor * 10000

        return perm_impact

    def _calculate_confidence_interval(
        self, impact: float, volatility: float
    ) -> Tuple[float, float]:
        """计算置信区间

        Args:
            impact: 冲击估算值
            volatility: 波动率

        Returns:
            (下限, 上限)
        """
        # 使用正态分布假设
        z_score = 1.96  # 95%置信水平
        std_error = impact * volatility * 0.5  # 简化的标准误差

        lower = impact - z_score * std_error
        upper = impact + z_score * std_error

        return (lower, upper)

    def _extract_calibration_features(
        self, historical_data: pd.DataFrame, execution_data: pd.DataFrame
    ) -> pd.DataFrame:
        """提取校准特征

        Args:
            historical_data: 历史市场数据
            execution_data: 执行数据

        Returns:
            特征数据框
        """
        features = pd.DataFrame()

        # 计算参与率
        features["participation_rate"] = (
            execution_data["quantity"] / execution_data["daily_volume"]
        )

        # 计算波动率
        features["volatility"] = execution_data["symbol"].map(
            lambda s: historical_data[historical_data["symbol"] == s]["returns"].std()
        )

        # 其他特征
        features["spread"] = execution_data["spread_bps"]
        features["urgency"] = execution_data.get("urgency", 0.5)

        return features

    def _calculate_actual_impacts(self, execution_data: pd.DataFrame) -> np.ndarray:
        """计算实际冲击

        Args:
            execution_data: 执行数据

        Returns:
            实际冲击数组
        """
        impacts = []

        for _, row in execution_data.iterrows():
            if row["side"] == "BUY":
                impact = (row["avg_fill_price"] - row["arrival_price"]) / row[
                    "arrival_price"
                ]
            else:
                impact = (row["arrival_price"] - row["avg_fill_price"]) / row[
                    "arrival_price"
                ]

            impacts.append(impact * 10000)  # 转换为基点

        return np.array(impacts)

    def _optimize_parameters(
        self, features: pd.DataFrame, actual_impacts: np.ndarray
    ) -> Dict[str, float]:
        """优化模型参数

        Args:
            features: 特征数据
            actual_impacts: 实际冲击

        Returns:
            优化后的参数
        """

        def objective(params):
            """目标函数：最小化预测误差"""
            self.model_parameters = {
                "temp_alpha": params[0],
                "temp_beta": params[1],
                "perm_gamma": params[2],
            }

            predicted = []
            for _, row in features.iterrows():
                temp = self._estimate_temporary_impact(
                    row["participation_rate"], row["volatility"], row["spread"]
                )
                perm = self._estimate_permanent_impact(
                    row["participation_rate"],
                    0.5,  # 默认流动性
                )
                predicted.append(temp + perm)

            predicted = np.array(predicted)

            # 均方误差
            mse = np.mean((predicted - actual_impacts) ** 2)
            return mse

        # 初始参数
        x0 = [0.1, 0.5, 0.05]

        # 参数边界
        bounds = [(0.01, 1.0), (0.1, 2.0), (0.01, 0.5)]

        # 优化
        result = optimize.minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

        return {
            "temp_alpha": result.x[0],
            "temp_beta": result.x[1],
            "perm_gamma": result.x[2],
        }


class AlmgrenChrissModel(MarketImpactModel):
    """Almgren-Chriss市场冲击模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.eta = config.get("eta", 0.01)  # 临时冲击系数
        self.gamma = config.get("gamma", 0.001)  # 永久冲击系数
        self.lambda_risk = config.get("lambda_risk", 1e-6)  # 风险厌恶系数

    def calculate_optimal_trajectory(
        self,
        total_quantity: int,
        time_horizon: int,
        microstructure: MarketMicrostructure,
    ) -> List[int]:
        """计算最优执行轨迹

        Args:
            total_quantity: 总交易量
            time_horizon: 时间范围（分钟）
            microstructure: 市场微观结构

        Returns:
            每个时间段的交易量列表
        """
        n_periods = min(time_horizon, 60)  # 最多60个时间段
        dt = time_horizon / n_periods

        # 计算衰减率
        kappa = np.sqrt(self.lambda_risk / self.eta)

        # 生成执行轨迹
        trajectory = []
        remaining = total_quantity

        for i in range(n_periods):
            # Almgren-Chriss最优解
            fraction = np.sinh(kappa * (n_periods - i)) / np.sinh(kappa * n_periods)
            period_quantity = int(remaining * fraction)

            trajectory.append(period_quantity)
            remaining -= period_quantity

        # 确保总量匹配
        if remaining > 0:
            trajectory[-1] += remaining

        return trajectory

    def estimate_implementation_shortfall(
        self, trajectory: List[int], microstructure: MarketMicrostructure
    ) -> float:
        """估算实施缺口

        Args:
            trajectory: 执行轨迹
            microstructure: 市场微观结构

        Returns:
            预期实施缺口（基点）
        """
        total_quantity = sum(trajectory)
        n_periods = len(trajectory)

        if total_quantity == 0 or n_periods == 0:
            return 0.0

        # 计算永久冲击成本
        perm_cost = 0.5 * self.gamma * total_quantity

        # 计算临时冲击成本
        temp_cost = 0
        for qty in trajectory:
            temp_cost += self.eta * qty * qty / total_quantity

        # 计算波动成本
        vol_cost = 0.5 * microstructure.volatility * np.sqrt(n_periods) * total_quantity

        # 总成本（基点）
        total_cost = (perm_cost + temp_cost + vol_cost) * 10000

        return total_cost


class LinearPropagatorModel(MarketImpactModel):
    """线性传播模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.propagator_decay = config.get("propagator_decay", 0.9)
        self.impact_coefficient = config.get("impact_coefficient", 0.1)

    def estimate_cumulative_impact(
        self, order_sequence: List[Dict[str, Any]], decay_time: int = 30
    ) -> List[float]:
        """估算累积冲击

        Args:
            order_sequence: 订单序列
            decay_time: 衰减时间（分钟）

        Returns:
            每个订单后的累积冲击列表
        """
        cumulative_impacts = []
        current_impact = 0.0

        for i, order in enumerate(order_sequence):
            # 计算该订单的冲击
            instant_impact = self._calculate_instant_impact(order)

            # 考虑之前订单的衰减
            if i > 0:
                time_diff = (
                    order["timestamp"] - order_sequence[i - 1]["timestamp"]
                ).total_seconds() / 60
                decay_factor = self.propagator_decay ** (time_diff / decay_time)
                current_impact *= decay_factor

            # 累加新冲击
            current_impact += instant_impact
            cumulative_impacts.append(current_impact)

        return cumulative_impacts

    def _calculate_instant_impact(self, order: Dict[str, Any]) -> float:
        """计算瞬时冲击

        Args:
            order: 订单信息

        Returns:
            瞬时冲击（基点）
        """
        size = order["quantity"]
        volume = order.get("market_volume", 1000000)

        # 线性冲击模型
        impact = self.impact_coefficient * (size / volume) * 10000

        # 方向调整
        if order["side"] == "SELL":
            impact = -impact

        return impact


class MachineLearningImpactModel(MarketImpactModel):
    """机器学习冲击模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.feature_columns = [
            "participation_rate",
            "volatility",
            "spread_bps",
            "order_book_imbalance",
            "time_of_day",
            "urgency",
        ]

    def train(
        self, training_data: pd.DataFrame, target_column: str = "actual_impact_bps"
    ) -> None:
        """训练模型

        Args:
            training_data: 训练数据
            target_column: 目标列名
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        # 准备特征
        X = training_data[self.feature_columns]
        y = training_data[target_column]

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 训练随机森林
        self.model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )

        self.model.fit(X_scaled, y)

        # 计算特征重要性
        self.feature_importance = dict(
            zip(self.feature_columns, self.model.feature_importances_)
        )

        logger.info(f"Model trained. Feature importance: {self.feature_importance}")

    def predict_impact(self, features: Dict[str, float]) -> float:
        """预测冲击

        Args:
            features: 特征字典

        Returns:
            预测的冲击（基点）
        """
        if self.model is None:
            logger.warning("Model not trained, using default impact model")
            return 0.0

        # 准备特征向量
        feature_vector = np.array(
            [[features.get(col, 0) for col in self.feature_columns]]
        )

        # 标准化
        feature_scaled = self.scaler.transform(feature_vector)

        # 预测
        impact = self.model.predict(feature_scaled)[0]

        return impact


# 模块级别函数
def create_impact_model(model_type: str, config: Dict[str, Any]) -> MarketImpactModel:
    """创建市场冲击模型实例

    Args:
        model_type: 模型类型
        config: 配置字典

    Returns:
        市场冲击模型实例
    """
    models = {
        "base": MarketImpactModel,
        "almgren_chriss": AlmgrenChrissModel,
        "linear_propagator": LinearPropagatorModel,
        "machine_learning": MachineLearningImpactModel,
    }

    if model_type not in models:
        logger.warning(f"Unknown model type {model_type}, using base model")
        model_type = "base"

    return models[model_type](config)


def analyze_market_microstructure(
    symbol: str,
    market_data: pd.DataFrame,
    order_book_data: Optional[pd.DataFrame] = None,
) -> MarketMicrostructure:
    """分析市场微观结构

    Args:
        symbol: 标的代码
        market_data: 市场数据
        order_book_data: 订单簿数据

    Returns:
        市场微观结构对象
    """
    # 计算平均价差
    if "bid" in market_data.columns and "ask" in market_data.columns:
        spreads = (market_data["ask"] - market_data["bid"]) / market_data["mid_price"]
        avg_spread_bps = spreads.mean() * 10000
    else:
        avg_spread_bps = 5.0  # 默认5基点

    # 计算日均成交量
    avg_daily_volume = (
        market_data["volume"].mean() if "volume" in market_data.columns else 1000000
    )

    # 计算波动率
    if "returns" in market_data.columns:
        volatility = market_data["returns"].std() * np.sqrt(252)
    else:
        volatility = 0.02  # 默认2%

    # 计算流动性评分
    liquidity_score = min(1.0, avg_daily_volume / 10000000)  # 简化评分

    # 分析订单簿深度
    market_depth = {}
    if order_book_data is not None and not order_book_data.empty:
        for level in [0.001, 0.002, 0.005, 0.01]:  # 不同价格水平
            depth_at_level = order_book_data[order_book_data["price_level"] <= level][
                "quantity"
            ].sum()
            market_depth[level] = depth_at_level

    # 计算订单簿失衡
    if order_book_data is not None and "side" in order_book_data.columns:
        bid_volume = order_book_data[order_book_data["side"] == "BID"]["quantity"].sum()
        ask_volume = order_book_data[order_book_data["side"] == "ASK"]["quantity"].sum()
        total_volume = bid_volume + ask_volume

        if total_volume > 0:
            order_book_imbalance = (bid_volume - ask_volume) / total_volume
        else:
            order_book_imbalance = 0.0
    else:
        order_book_imbalance = 0.0

    return MarketMicrostructure(
        symbol=symbol,
        avg_spread_bps=avg_spread_bps,
        avg_daily_volume=avg_daily_volume,
        volatility=volatility,
        liquidity_score=liquidity_score,
        tick_size=0.01,  # 默认值
        lot_size=100,  # 默认值
        market_depth=market_depth,
        order_book_imbalance=order_book_imbalance,
    )
