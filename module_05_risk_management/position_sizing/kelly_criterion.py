"""
Kelly准则仓位计算器模块
实现最优仓位大小计算
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.constants import MAX_POSITION_PCT, MIN_POSITION_SIZE
from common.exceptions import ModelError
from common.logging_system import setup_logger
from scipy import optimize

logger = setup_logger("kelly_criterion")


@dataclass
class KellyConfig:
    """Kelly准则配置"""

    kelly_fraction: float = 0.25  # Kelly分数（保守化）
    max_leverage: float = 1.0  # 最大杠杆
    min_bet_size: float = 0.01  # 最小下注比例
    max_bet_size: float = 0.25  # 最大下注比例
    confidence_threshold: float = 0.6
    lookback_periods: int = 100
    use_half_kelly: bool = True  # 使用半Kelly以降低风险
    adjust_for_skewness: bool = True
    correlation_adjustment: bool = True


@dataclass
class KellyResult:
    """Kelly准则计算结果"""

    optimal_fraction: float
    adjusted_fraction: float
    expected_return: float
    win_probability: float
    win_loss_ratio: float
    confidence: float
    risk_metrics: Dict[str, float]
    constraints_applied: List[str]


class KellyCriterion:
    """Kelly准则计算器"""

    def __init__(self, config: Optional[KellyConfig] = None):
        """初始化Kelly准则计算器

        Args:
            config: Kelly配置
        """
        self.config = config or KellyConfig()
        self.historical_results: List[KellyResult] = []
        self.performance_metrics: Dict[str, float] = {}

    def calculate_optimal_position_size(
        self,
        returns: pd.Series,
        signal_strength: float = 1.0,
        current_portfolio_value: float = 100000,
    ) -> KellyResult:
        """计算最优仓位大小

        Args:
            returns: 历史收益率序列
            signal_strength: 信号强度（0-1）
            current_portfolio_value: 当前组合价值

        Returns:
            Kelly计算结果
        """
        logger.info("Calculating optimal position size using Kelly criterion...")

        # 计算基本统计量
        win_probability, avg_win, avg_loss = self._calculate_win_loss_statistics(
            returns
        )

        # 计算基本Kelly分数
        if avg_loss != 0:
            win_loss_ratio = abs(avg_win / avg_loss)
            kelly_fraction = (
                win_probability * win_loss_ratio - (1 - win_probability)
            ) / win_loss_ratio
        else:
            kelly_fraction = 0
            win_loss_ratio = 0

        # 调整Kelly分数
        adjusted_fraction = self._adjust_kelly_fraction(
            kelly_fraction, returns, signal_strength
        )

        # 计算期望收益
        expected_return = win_probability * avg_win - (1 - win_probability) * abs(
            avg_loss
        )

        # 计算置信度
        confidence = self._calculate_confidence(returns, win_probability)

        # 风险指标
        risk_metrics = self._calculate_risk_metrics(returns, adjusted_fraction)

        # 应用的约束
        constraints = self._get_applied_constraints(kelly_fraction, adjusted_fraction)

        result = KellyResult(
            optimal_fraction=kelly_fraction,
            adjusted_fraction=adjusted_fraction,
            expected_return=expected_return,
            win_probability=win_probability,
            win_loss_ratio=win_loss_ratio,
            confidence=confidence,
            risk_metrics=risk_metrics,
            constraints_applied=constraints,
        )

        self.historical_results.append(result)

        logger.info(
            f"Kelly fraction: {kelly_fraction:.4f}, Adjusted: {adjusted_fraction:.4f}"
        )
        return result

    def implement_kelly_criterion(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float = 0.02,
    ) -> np.ndarray:
        """实现多资产Kelly准则

        Args:
            expected_returns: 期望收益率数组
            covariance_matrix: 协方差矩阵
            risk_free_rate: 无风险利率

        Returns:
            最优权重数组
        """
        n_assets = len(expected_returns)

        # 调整期望收益（减去无风险利率）
        excess_returns = expected_returns - risk_free_rate

        # Kelly权重 = 协方差矩阵的逆 × 超额收益
        try:
            inv_cov = np.linalg.inv(covariance_matrix)
            kelly_weights = inv_cov @ excess_returns

            # 应用Kelly分数
            kelly_weights *= self.config.kelly_fraction

            # 标准化权重
            if self.config.max_leverage == 1.0:
                # 无杠杆：确保权重和为1且非负
                kelly_weights = self._normalize_weights(kelly_weights)
            else:
                # 有杠杆：限制总杠杆
                total_exposure = np.sum(np.abs(kelly_weights))
                if total_exposure > self.config.max_leverage:
                    kelly_weights *= self.config.max_leverage / total_exposure

        except np.linalg.LinAlgError:
            logger.error("Singular covariance matrix, using equal weights")
            kelly_weights = np.ones(n_assets) / n_assets

        return kelly_weights

    def calculate_fractional_kelly(
        self,
        strategies: List[Dict[str, float]],
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """计算多策略的分数Kelly

        Args:
            strategies: 策略列表，每个包含win_prob, avg_win, avg_loss
            correlation_matrix: 策略间相关性矩阵

        Returns:
            每个策略的Kelly分数
        """
        kelly_fractions = {}

        for i, strategy in enumerate(strategies):
            # 计算单个策略的Kelly
            win_prob = strategy["win_prob"]
            avg_win = strategy["avg_win"]
            avg_loss = strategy["avg_loss"]

            if avg_loss != 0:
                b = avg_win / abs(avg_loss)
                f = (win_prob * b - (1 - win_prob)) / b
            else:
                f = 0

            # 如果提供了相关性矩阵，进行调整
            if correlation_matrix is not None and self.config.correlation_adjustment:
                f = self._adjust_for_correlation(f, i, correlation_matrix)

            # 应用分数Kelly
            if self.config.use_half_kelly:
                f *= 0.5
            else:
                f *= self.config.kelly_fraction

            # 限制范围
            f = max(self.config.min_bet_size, min(f, self.config.max_bet_size))

            kelly_fractions[f"strategy_{i}"] = f

        return kelly_fractions

    def optimize_kelly_portfolio(
        self, returns_df: pd.DataFrame, constraints: Optional[Dict[str, Any]] = None
    ) -> pd.Series:
        """优化Kelly组合

        Args:
            returns_df: 收益率DataFrame
            constraints: 约束条件

        Returns:
            最优权重Series
        """
        n_assets = len(returns_df.columns)

        # 计算每个资产的Kelly参数
        kelly_params = {}
        for col in returns_df.columns:
            returns = returns_df[col]
            win_prob, avg_win, avg_loss = self._calculate_win_loss_statistics(returns)
            kelly_params[col] = {
                "win_prob": win_prob,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
            }

        # 定义目标函数（最大化几何平均收益）
        def objective(weights):
            portfolio_returns = returns_df @ weights
            # 使用对数收益的平均值近似几何平均
            log_returns = np.log(1 + portfolio_returns)
            return -log_returns.mean()  # 负号因为是最小化

        # 定义约束
        cons = []

        # 权重和为1
        cons.append({"type": "eq", "fun": lambda x: np.sum(x) - 1})

        # 自定义约束
        if constraints:
            if "max_position" in constraints:
                for i in range(n_assets):
                    cons.append(
                        {
                            "type": "ineq",
                            "fun": lambda x, i=i: constraints["max_position"] - x[i],
                        }
                    )

        # 边界（允许做空的话可以设置负值）
        bounds = [(0, self.config.max_bet_size) for _ in range(n_assets)]

        # 初始猜测
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = optimize.minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=cons
        )

        if result.success:
            weights = result.x
        else:
            logger.warning("Optimization failed, using equal weights")
            weights = x0

        return pd.Series(weights, index=returns_df.columns)

    def calculate_kelly_with_uncertainty(
        self, returns: pd.Series, confidence_interval: float = 0.95
    ) -> Tuple[float, float, float]:
        """考虑不确定性的Kelly计算

        Args:
            returns: 收益率序列
            confidence_interval: 置信区间

        Returns:
            (下界Kelly, 中值Kelly, 上界Kelly)
        """
        # Bootstrap估计参数不确定性
        n_bootstrap = 1000
        kelly_estimates = []

        for _ in range(n_bootstrap):
            # 重采样
            sample = returns.sample(n=len(returns), replace=True)

            # 计算Kelly
            win_prob, avg_win, avg_loss = self._calculate_win_loss_statistics(sample)

            if avg_loss != 0:
                b = avg_win / abs(avg_loss)
                f = (win_prob * b - (1 - win_prob)) / b
                kelly_estimates.append(f)

        kelly_estimates = np.array(kelly_estimates)

        # 计算置信区间
        alpha = 1 - confidence_interval
        lower = np.percentile(kelly_estimates, alpha / 2 * 100)
        median = np.median(kelly_estimates)
        upper = np.percentile(kelly_estimates, (1 - alpha / 2) * 100)

        # 应用保守调整
        if self.config.use_half_kelly:
            lower *= 0.5
            median *= 0.5
            upper *= 0.5

        return max(0, lower), max(0, median), max(0, upper)

    def _calculate_win_loss_statistics(
        self, returns: pd.Series
    ) -> Tuple[float, float, float]:
        """计算胜率和盈亏统计

        Args:
            returns: 收益率序列

        Returns:
            (胜率, 平均盈利, 平均亏损)
        """
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        win_probability = (
            len(positive_returns) / len(returns) if len(returns) > 0 else 0
        )
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0

        return win_probability, avg_win, avg_loss

    def _adjust_kelly_fraction(
        self, kelly_fraction: float, returns: pd.Series, signal_strength: float
    ) -> float:
        """调整Kelly分数

        Args:
            kelly_fraction: 原始Kelly分数
            returns: 收益率序列
            signal_strength: 信号强度

        Returns:
            调整后的Kelly分数
        """
        adjusted = kelly_fraction

        # 应用分数Kelly
        if self.config.use_half_kelly:
            adjusted *= 0.5
        else:
            adjusted *= self.config.kelly_fraction

        # 根据信号强度调整
        adjusted *= signal_strength

        # 考虑偏度调整
        if self.config.adjust_for_skewness:
            skewness = returns.skew()
            if skewness < -0.5:  # 负偏
                adjusted *= 0.8
            elif skewness > 1.0:  # 正偏
                adjusted *= 1.1

        # 限制范围
        adjusted = max(
            self.config.min_bet_size, min(adjusted, self.config.max_bet_size)
        )

        return adjusted

    def _calculate_confidence(
        self, returns: pd.Series, win_probability: float
    ) -> float:
        """计算置信度

        Args:
            returns: 收益率序列
            win_probability: 胜率

        Returns:
            置信度分数
        """
        # 基于样本量的置信度
        n = len(returns)
        sample_confidence = min(1.0, n / self.config.lookback_periods)

        # 基于胜率稳定性的置信度
        if n > 30:
            # 计算滚动胜率的标准差
            rolling_win_rate = (
                pd.Series([1 if r > 0 else 0 for r in returns])
                .rolling(window=min(30, n // 3))
                .mean()
            )

            win_rate_stability = 1 / (1 + rolling_win_rate.std() * 10)
        else:
            win_rate_stability = 0.5

        # 综合置信度
        confidence = sample_confidence * 0.5 + win_rate_stability * 0.5

        return confidence

    def _calculate_risk_metrics(
        self, returns: pd.Series, kelly_fraction: float
    ) -> Dict[str, float]:
        """计算风险指标

        Args:
            returns: 收益率序列
            kelly_fraction: Kelly分数

        Returns:
            风险指标字典
        """
        metrics = {}

        # 基本统计
        metrics["volatility"] = returns.std()
        metrics["downside_volatility"] = (
            returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0
        )
        metrics["max_drawdown"] = self._calculate_max_drawdown(returns)

        # Kelly相关风险
        leveraged_returns = returns * kelly_fraction
        metrics["kelly_volatility"] = leveraged_returns.std()
        metrics["kelly_max_drawdown"] = self._calculate_max_drawdown(leveraged_returns)

        # 风险价值
        metrics["var_95"] = np.percentile(leveraged_returns, 5)
        metrics["cvar_95"] = leveraged_returns[
            leveraged_returns <= metrics["var_95"]
        ].mean()

        # Kelly准则的增长率
        growth_rate = kelly_fraction * returns.mean() - 0.5 * (kelly_fraction**2) * (
            returns.var()
        )
        metrics["expected_growth_rate"] = growth_rate

        return metrics

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤

        Args:
            returns: 收益率序列

        Returns:
            最大回撤
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _get_applied_constraints(self, original: float, adjusted: float) -> List[str]:
        """获取应用的约束

        Args:
            original: 原始值
            adjusted: 调整后值

        Returns:
            约束列表
        """
        constraints = []

        if self.config.use_half_kelly:
            constraints.append("half_kelly")

        if adjusted == self.config.min_bet_size:
            constraints.append("min_bet_size")
        elif adjusted == self.config.max_bet_size:
            constraints.append("max_bet_size")

        if self.config.adjust_for_skewness:
            constraints.append("skewness_adjustment")

        return constraints

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """标准化权重

        Args:
            weights: 权重数组

        Returns:
            标准化后的权重
        """
        # 处理负权重（如果不允许做空）
        weights = np.maximum(weights, 0)

        # 标准化到和为1
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum
        else:
            # 如果所有权重都是0，使用均等权重
            weights = np.ones_like(weights) / len(weights)

        return weights

    def _adjust_for_correlation(
        self, kelly_fraction: float, strategy_index: int, correlation_matrix: np.ndarray
    ) -> float:
        """根据相关性调整Kelly分数

        Args:
            kelly_fraction: 原始Kelly分数
            strategy_index: 策略索引
            correlation_matrix: 相关性矩阵

        Returns:
            调整后的Kelly分数
        """
        # 获取与其他策略的平均相关性
        correlations = correlation_matrix[strategy_index]
        avg_correlation = np.mean(np.abs(correlations[correlations != 1]))

        # 高相关性降低Kelly分数
        adjustment_factor = 1 - 0.5 * avg_correlation

        return kelly_fraction * adjustment_factor


# 模块级别函数
def calculate_kelly_position(
    returns: pd.Series, config: Optional[KellyConfig] = None
) -> Dict[str, float]:
    """计算Kelly仓位的便捷函数

    Args:
        returns: 收益率序列
        config: Kelly配置

    Returns:
        仓位信息字典
    """
    calculator = KellyCriterion(config)
    result = calculator.calculate_optimal_position_size(returns)

    return {
        "optimal_fraction": result.optimal_fraction,
        "adjusted_fraction": result.adjusted_fraction,
        "expected_return": result.expected_return,
        "win_probability": result.win_probability,
        "confidence": result.confidence,
    }
