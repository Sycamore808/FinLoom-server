"""
Black-Litterman模型模块
实现贝叶斯投资组合优化框架
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.constants import TRADING_DAYS_PER_YEAR
from common.exceptions import ModelError
from common.logging_system import setup_logger
from scipy import linalg

logger = setup_logger("black_litterman")


class ViewType(Enum):
    """观点类型枚举"""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    FACTOR = "factor"


@dataclass
class MarketView:
    """市场观点"""

    view_id: str
    view_type: ViewType
    assets: List[str]
    weights: List[float]
    expected_return: float
    confidence: float
    description: str
    source: str
    timestamp: datetime


@dataclass
class BlackLittermanConfig:
    """Black-Litterman配置"""

    tau: float = 0.05  # 不确定性缩放参数
    risk_aversion: float = 2.5  # 风险厌恶系数
    confidence_scaling: str = "proportional"  # proportional, inverse_variance
    min_confidence: float = 0.1
    max_confidence: float = 10.0
    use_market_cap_weights: bool = True
    shrinkage: float = 0.0  # 向市场权重收缩
    view_uncertainty_method: str = "idzorek"  # idzorek, he_litterman


@dataclass
class BlackLittermanResult:
    """Black-Litterman结果"""

    posterior_returns: pd.Series
    posterior_covariance: pd.DataFrame
    posterior_weights: pd.Series
    implied_returns: pd.Series
    view_impacts: pd.DataFrame
    uncertainty_matrix: pd.DataFrame
    tilting_factors: pd.Series
    confidence_scores: Dict[str, float]


class BlackLitterman:
    """Black-Litterman模型类"""

    def __init__(self, config: Optional[BlackLittermanConfig] = None):
        """初始化Black-Litterman模型

        Args:
            config: Black-Litterman配置
        """
        self.config = config or BlackLittermanConfig()
        self.market_data_cache: Dict[str, Any] = {}
        self.view_history: List[MarketView] = []

    def calculate_posterior_returns(
        self,
        market_caps: pd.Series,
        returns_covariance: pd.DataFrame,
        views: List[MarketView],
        risk_free_rate: float = 0.02,
    ) -> BlackLittermanResult:
        """计算后验收益率

        Args:
            market_caps: 市值Series
            returns_covariance: 收益率协方差矩阵
            views: 市场观点列表
            risk_free_rate: 无风险利率

        Returns:
            Black-Litterman结果
        """
        logger.info(f"Calculating Black-Litterman posterior with {len(views)} views")

        # 计算市场均衡权重
        market_weights = self._calculate_market_weights(market_caps)

        # 计算隐含均衡收益率
        implied_returns = self._calculate_implied_returns(
            returns_covariance, market_weights, risk_free_rate
        )

        # 构建观点矩阵
        P, Q, omega = self._construct_view_matrices(views, implied_returns)

        # 计算后验分布
        posterior_mean, posterior_cov = self._calculate_posterior_distribution(
            implied_returns, returns_covariance, P, Q, omega
        )

        # 计算后验权重
        posterior_weights = self._calculate_posterior_weights(
            posterior_mean, posterior_cov, risk_free_rate
        )

        # 计算观点影响
        view_impacts = self._calculate_view_impacts(
            implied_returns, posterior_mean, views
        )

        # 计算倾斜因子
        tilting_factors = self._calculate_tilting_factors(
            market_weights, posterior_weights
        )

        # 计算置信度分数
        confidence_scores = self._calculate_confidence_scores(views, omega)

        result = BlackLittermanResult(
            posterior_returns=posterior_mean,
            posterior_covariance=posterior_cov,
            posterior_weights=posterior_weights,
            implied_returns=implied_returns,
            view_impacts=view_impacts,
            uncertainty_matrix=pd.DataFrame(
                omega,
                index=[v.view_id for v in views],
                columns=[v.view_id for v in views],
            ),
            tilting_factors=tilting_factors,
            confidence_scores=confidence_scores,
        )

        logger.info(f"Black-Litterman optimization completed")

        return result

    def create_absolute_view(
        self,
        asset: str,
        expected_return: float,
        confidence: float,
        description: str = "",
    ) -> MarketView:
        """创建绝对观点

        Args:
            asset: 资产名称
            expected_return: 期望收益率
            confidence: 置信度
            description: 描述

        Returns:
            市场观点
        """
        view = MarketView(
            view_id=f"absolute_{asset}_{datetime.now().timestamp()}",
            view_type=ViewType.ABSOLUTE,
            assets=[asset],
            weights=[1.0],
            expected_return=expected_return,
            confidence=self._validate_confidence(confidence),
            description=description or f"Absolute view on {asset}",
            source="user",
            timestamp=datetime.now(),
        )

        self.view_history.append(view)
        return view

    def create_relative_view(
        self,
        long_assets: List[str],
        short_assets: List[str],
        expected_outperformance: float,
        confidence: float,
        description: str = "",
    ) -> MarketView:
        """创建相对观点

        Args:
            long_assets: 看多资产列表
            short_assets: 看空资产列表
            expected_outperformance: 期望超额收益
            confidence: 置信度
            description: 描述

        Returns:
            市场观点
        """
        # 构建权重
        n_long = len(long_assets)
        n_short = len(short_assets)

        assets = long_assets + short_assets
        weights = [1.0 / n_long] * n_long + [-1.0 / n_short] * n_short

        view = MarketView(
            view_id=f"relative_{datetime.now().timestamp()}",
            view_type=ViewType.RELATIVE,
            assets=assets,
            weights=weights,
            expected_return=expected_outperformance,
            confidence=self._validate_confidence(confidence),
            description=description
            or f"Relative view: {long_assets} vs {short_assets}",
            source="user",
            timestamp=datetime.now(),
        )

        self.view_history.append(view)
        return view

    def calculate_implied_equilibrium_returns(
        self,
        covariance_matrix: pd.DataFrame,
        market_weights: pd.Series,
        risk_aversion: Optional[float] = None,
    ) -> pd.Series:
        """计算隐含均衡收益率

        Args:
            covariance_matrix: 协方差矩阵
            market_weights: 市场权重
            risk_aversion: 风险厌恶系数

        Returns:
            隐含收益率Series
        """
        risk_aversion = risk_aversion or self.config.risk_aversion

        # π = δ * Σ * w_mkt
        implied_returns = risk_aversion * (covariance_matrix @ market_weights)

        return pd.Series(implied_returns, index=market_weights.index)

    def apply_idzorek_method(
        self,
        view: MarketView,
        covariance_matrix: pd.DataFrame,
        target_confidence: float = 1.0,
    ) -> float:
        """应用Idzorek方法计算观点不确定性

        Args:
            view: 市场观点
            covariance_matrix: 协方差矩阵
            target_confidence: 目标置信度

        Returns:
            观点方差
        """
        # 提取观点向量
        P = self._create_view_vector(view, covariance_matrix.index)

        # 计算观点组合的方差
        view_variance = P @ covariance_matrix @ P.T

        # 根据置信度调整
        # 置信度越高，不确定性越小
        uncertainty = view_variance * (1 / target_confidence)

        return float(uncertainty)

    def combine_expert_views(
        self,
        expert_views: Dict[str, List[MarketView]],
        expert_weights: Optional[Dict[str, float]] = None,
    ) -> List[MarketView]:
        """组合专家观点

        Args:
            expert_views: 专家观点字典
            expert_weights: 专家权重

        Returns:
            组合后的观点列表
        """
        if expert_weights is None:
            # 等权重
            n_experts = len(expert_views)
            expert_weights = {expert: 1.0 / n_experts for expert in expert_views}

        combined_views = []

        # 按资产分组观点
        asset_views = {}

        for expert, views in expert_views.items():
            weight = expert_weights.get(expert, 1.0)

            for view in views:
                key = tuple(view.assets)

                if key not in asset_views:
                    asset_views[key] = []

                asset_views[key].append({"view": view, "weight": weight})

        # 组合相同资产的观点
        for assets, view_list in asset_views.items():
            # 加权平均期望收益
            weighted_return = sum(
                v["view"].expected_return * v["weight"] for v in view_list
            )
            total_weight = sum(v["weight"] for v in view_list)
            avg_return = weighted_return / total_weight

            # 组合置信度
            combined_confidence = (
                sum(v["view"].confidence * v["weight"] for v in view_list)
                / total_weight
            )

            # 创建组合观点
            combined_view = MarketView(
                view_id=f"combined_{datetime.now().timestamp()}",
                view_type=view_list[0]["view"].view_type,
                assets=list(assets),
                weights=view_list[0]["view"].weights,
                expected_return=avg_return,
                confidence=combined_confidence,
                description=f"Combined view from {len(view_list)} experts",
                source="combined",
                timestamp=datetime.now(),
            )

            combined_views.append(combined_view)

        return combined_views

    def backtest_views(
        self,
        historical_returns: pd.DataFrame,
        views: List[MarketView],
        window: int = 60,
    ) -> pd.DataFrame:
        """回测观点准确性

        Args:
            historical_returns: 历史收益率
            views: 观点列表
            window: 评估窗口

        Returns:
            回测结果DataFrame
        """
        results = []

        for view in views:
            # 构建观点组合
            view_portfolio = pd.Series(0.0, index=historical_returns.columns)

            for asset, weight in zip(view.assets, view.weights):
                if asset in view_portfolio.index:
                    view_portfolio[asset] = weight

            # 计算观点组合收益
            portfolio_returns = historical_returns @ view_portfolio

            # 计算实现的收益
            realized_return = (
                portfolio_returns.tail(window).mean() * TRADING_DAYS_PER_YEAR
            )

            # 计算准确性指标
            accuracy = 1 - abs(realized_return - view.expected_return) / abs(
                view.expected_return
            )
            direction_correct = (realized_return > 0) == (view.expected_return > 0)

            results.append(
                {
                    "view_id": view.view_id,
                    "expected_return": view.expected_return,
                    "realized_return": realized_return,
                    "accuracy": accuracy,
                    "direction_correct": direction_correct,
                    "confidence": view.confidence,
                }
            )

        return pd.DataFrame(results)

    def _calculate_market_weights(self, market_caps: pd.Series) -> pd.Series:
        """计算市场权重

        Args:
            market_caps: 市值Series

        Returns:
            市场权重Series
        """
        if self.config.use_market_cap_weights:
            weights = market_caps / market_caps.sum()
        else:
            # 等权重
            n_assets = len(market_caps)
            weights = pd.Series(1.0 / n_assets, index=market_caps.index)

        # 应用收缩
        if self.config.shrinkage > 0:
            equal_weight = pd.Series(1.0 / len(weights), index=weights.index)
            weights = (
                1 - self.config.shrinkage
            ) * weights + self.config.shrinkage * equal_weight

        return weights

    def _calculate_implied_returns(
        self,
        covariance_matrix: pd.DataFrame,
        market_weights: pd.Series,
        risk_free_rate: float,
    ) -> pd.Series:
        """计算隐含收益率

        Args:
            covariance_matrix: 协方差矩阵
            market_weights: 市场权重
            risk_free_rate: 无风险利率

        Returns:
            隐含收益率Series
        """
        # 计算市场组合的风险溢价
        market_variance = market_weights @ covariance_matrix @ market_weights
        market_risk_premium = self.config.risk_aversion * market_variance

        # 隐含均衡收益率
        implied_returns = self.config.risk_aversion * (
            covariance_matrix @ market_weights
        )

        return pd.Series(implied_returns, index=market_weights.index)

    def _construct_view_matrices(
        self, views: List[MarketView], implied_returns: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构建观点矩阵

        Args:
            views: 观点列表
            implied_returns: 隐含收益率

        Returns:
            (P矩阵, Q向量, Omega矩阵)
        """
        n_views = len(views)
        n_assets = len(implied_returns)

        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        omega_diag = np.zeros(n_views)

        assets_list = list(implied_returns.index)

        for i, view in enumerate(views):
            # 构建P矩阵的行
            for asset, weight in zip(view.assets, view.weights):
                if asset in assets_list:
                    j = assets_list.index(asset)
                    P[i, j] = weight

            # Q向量
            if view.view_type == ViewType.ABSOLUTE:
                Q[i] = view.expected_return
            else:
                # 相对观点：已经是超额收益
                Q[i] = view.expected_return

            # 计算不确定性
            if self.config.view_uncertainty_method == "idzorek":
                view_variance = self._calculate_view_uncertainty_idzorek(
                    P[i], implied_returns.values, view.confidence
                )
            else:
                view_variance = self._calculate_view_uncertainty_he_litterman(
                    P[i], implied_returns.values
                )

            omega_diag[i] = view_variance / view.confidence

        # 构建Omega矩阵（对角矩阵）
        omega = np.diag(omega_diag)

        return P, Q, omega

    def _calculate_posterior_distribution(
        self,
        prior_mean: pd.Series,
        prior_cov: pd.DataFrame,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """计算后验分布

        Args:
            prior_mean: 先验均值
            prior_cov: 先验协方差
            P: 观点矩阵
            Q: 观点向量
            omega: 不确定性矩阵

        Returns:
            (后验均值, 后验协方差)
        """
        tau = self.config.tau

        # 缩放的先验协方差
        scaled_cov = tau * prior_cov.values

        # 后验均值
        # μ_BL = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1π + P'Ω^-1Q]
        inv_scaled_cov = linalg.inv(scaled_cov)
        inv_omega = linalg.inv(omega)

        A = inv_scaled_cov + P.T @ inv_omega @ P
        b = inv_scaled_cov @ prior_mean.values + P.T @ inv_omega @ Q

        posterior_mean = linalg.solve(A, b)

        # 后验协方差
        # Σ_BL = [(τΣ)^-1 + P'Ω^-1P]^-1
        posterior_cov = linalg.inv(A)

        # 添加原始协方差
        posterior_cov = posterior_cov + prior_cov.values

        return (
            pd.Series(posterior_mean, index=prior_mean.index),
            pd.DataFrame(
                posterior_cov, index=prior_cov.index, columns=prior_cov.columns
            ),
        )

    def _calculate_posterior_weights(
        self,
        posterior_returns: pd.Series,
        posterior_cov: pd.DataFrame,
        risk_free_rate: float,
    ) -> pd.Series:
        """计算后验权重

        Args:
            posterior_returns: 后验收益率
            posterior_cov: 后验协方差
            risk_free_rate: 无风险利率

        Returns:
            后验权重Series
        """
        # 使用均值方差优化
        excess_returns = posterior_returns - risk_free_rate

        # w = (1/δ) * Σ^-1 * (μ - rf)
        inv_cov = linalg.inv(posterior_cov.values)
        weights = (1 / self.config.risk_aversion) * inv_cov @ excess_returns.values

        # 标准化权重
        weights = weights / weights.sum()

        return pd.Series(weights, index=posterior_returns.index)

    def _calculate_view_impacts(
        self,
        prior_returns: pd.Series,
        posterior_returns: pd.Series,
        views: List[MarketView],
    ) -> pd.DataFrame:
        """计算观点影响

        Args:
            prior_returns: 先验收益率
            posterior_returns: 后验收益率
            views: 观点列表

        Returns:
            观点影响DataFrame
        """
        impacts = pd.DataFrame(
            index=prior_returns.index, columns=[v.view_id for v in views]
        )

        # 计算总影响
        total_impact = posterior_returns - prior_returns

        # 分解到各个观点（简化方法）
        for i, view in enumerate(views):
            view_impact = pd.Series(0.0, index=prior_returns.index)

            for asset in view.assets:
                if asset in view_impact.index:
                    # 根据置信度分配影响
                    weight = view.confidence / sum(v.confidence for v in views)
                    view_impact[asset] = total_impact[asset] * weight

            impacts[view.view_id] = view_impact

        return impacts

    def _calculate_tilting_factors(
        self, market_weights: pd.Series, posterior_weights: pd.Series
    ) -> pd.Series:
        """计算倾斜因子

        Args:
            market_weights: 市场权重
            posterior_weights: 后验权重

        Returns:
            倾斜因子Series
        """
        # 倾斜因子 = 后验权重 / 市场权重
        tilting = posterior_weights / market_weights

        return tilting

    def _calculate_confidence_scores(
        self, views: List[MarketView], omega: np.ndarray
    ) -> Dict[str, float]:
        """计算置信度分数

        Args:
            views: 观点列表
            omega: 不确定性矩阵

        Returns:
            置信度分数字典
        """
        scores = {}

        for i, view in enumerate(views):
            # 基于不确定性的置信度分数
            uncertainty = omega[i, i]
            score = 1 / (1 + uncertainty)
            scores[view.view_id] = score

        return scores

    def _validate_confidence(self, confidence: float) -> float:
        """验证置信度

        Args:
            confidence: 原始置信度

        Returns:
            验证后的置信度
        """
        return np.clip(
            confidence, self.config.min_confidence, self.config.max_confidence
        )

    def _create_view_vector(
        self, view: MarketView, assets_index: pd.Index
    ) -> np.ndarray:
        """创建观点向量

        Args:
            view: 市场观点
            assets_index: 资产索引

        Returns:
            观点向量
        """
        P = np.zeros(len(assets_index))

        for asset, weight in zip(view.assets, view.weights):
            if asset in assets_index:
                idx = assets_index.get_loc(asset)
                P[idx] = weight

        return P

    def _calculate_view_uncertainty_idzorek(
        self, P_row: np.ndarray, covariance: np.ndarray, confidence: float
    ) -> float:
        """使用Idzorek方法计算观点不确定性

        Args:
            P_row: P矩阵的一行
            covariance: 协方差矩阵
            confidence: 置信度

        Returns:
            观点方差
        """
        # 观点组合的方差
        view_variance = P_row @ covariance @ P_row.T

        # 根据置信度调整
        uncertainty = view_variance * (1 / confidence)

        return uncertainty

    def _calculate_view_uncertainty_he_litterman(
        self, P_row: np.ndarray, covariance: np.ndarray
    ) -> float:
        """使用He-Litterman方法计算观点不确定性

        Args:
            P_row: P矩阵的一行
            covariance: 协方差矩阵

        Returns:
            观点方差
        """
        # He-Litterman: ω = τ * P * Σ * P'
        return self.config.tau * (P_row @ covariance @ P_row.T)


# 模块级别函数
def apply_black_litterman(
    market_caps: pd.Series,
    covariance: pd.DataFrame,
    views: List[MarketView],
    config: Optional[BlackLittermanConfig] = None,
) -> pd.Series:
    """应用Black-Litterman模型的便捷函数

    Args:
        market_caps: 市值
        covariance: 协方差矩阵
        views: 观点列表
        config: 配置

    Returns:
        后验权重Series
    """
    bl = BlackLitterman(config)
    result = bl.calculate_posterior_returns(market_caps, covariance, views)
    return result.posterior_weights
