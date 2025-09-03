"""
风险平价配置器模块
实现风险平价投资组合配置策略
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.constants import TRADING_DAYS_PER_YEAR
from common.exceptions import ModelError
from common.logging_system import setup_logger
from scipy import optimize

logger = setup_logger("risk_parity")


@dataclass
class RiskParityConfig:
    """风险平价配置"""

    target_volatility: float = 0.10  # 目标波动率10%
    rebalance_frequency: str = "monthly"  # 再平衡频率
    min_weight: float = 0.01  # 最小权重
    max_weight: float = 0.40  # 最大权重
    risk_measure: str = "volatility"  # 风险度量方式
    use_leverage: bool = False  # 是否使用杠杆
    max_leverage: float = 1.5  # 最大杠杆
    lookback_days: int = 252  # 历史数据回看天数
    decay_factor: float = 0.94  # 指数衰减因子
    correlation_threshold: float = 0.95  # 相关性阈值


@dataclass
class RiskParityResult:
    """风险平价结果"""

    weights: np.ndarray
    asset_names: List[str]
    risk_contributions: np.ndarray
    risk_contribution_pct: np.ndarray
    portfolio_volatility: float
    portfolio_return: float
    leverage_used: float
    optimization_status: str
    convergence_error: float
    metadata: Dict[str, Any]


class RiskParity:
    """风险平价配置器"""

    def __init__(self, config: Optional[RiskParityConfig] = None):
        """初始化风险平价配置器

        Args:
            config: 风险平价配置
        """
        self.config = config or RiskParityConfig()
        self.optimization_history: List[RiskParityResult] = []

    def apply_risk_parity_allocation(
        self,
        returns_data: pd.DataFrame,
        target_risk_contributions: Optional[np.ndarray] = None,
    ) -> RiskParityResult:
        """应用风险平价配置

        Args:
            returns_data: 收益率数据DataFrame
            target_risk_contributions: 目标风险贡献比例

        Returns:
            风险平价结果
        """
        logger.info("Applying risk parity allocation...")

        n_assets = len(returns_data.columns)
        asset_names = list(returns_data.columns)

        # 计算协方差矩阵
        cov_matrix = self._calculate_covariance_matrix(returns_data)

        # 设置目标风险贡献（默认等权）
        if target_risk_contributions is None:
            target_risk_contributions = np.ones(n_assets) / n_assets
        else:
            # 标准化目标风险贡献
            target_risk_contributions = (
                target_risk_contributions / target_risk_contributions.sum()
            )

        # 优化权重
        optimal_weights = self._optimize_risk_parity_weights(
            cov_matrix, target_risk_contributions
        )

        # 计算风险贡献
        risk_contributions, risk_contribution_pct = self._calculate_risk_contributions(
            optimal_weights, cov_matrix
        )

        # 计算组合指标
        portfolio_volatility = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
        expected_returns = returns_data.mean().values
        portfolio_return = optimal_weights @ expected_returns

        # 应用杠杆（如果启用）
        leverage_used = 1.0
        if self.config.use_leverage and self.config.target_volatility > 0:
            leverage_used = min(
                self.config.target_volatility / portfolio_volatility,
                self.config.max_leverage,
            )
            optimal_weights *= leverage_used
            portfolio_volatility *= leverage_used
            portfolio_return *= leverage_used

        # 计算收敛误差
        convergence_error = np.sum(
            np.abs(risk_contribution_pct - target_risk_contributions)
        )

        result = RiskParityResult(
            weights=optimal_weights,
            asset_names=asset_names,
            risk_contributions=risk_contributions,
            risk_contribution_pct=risk_contribution_pct,
            portfolio_volatility=portfolio_volatility * np.sqrt(TRADING_DAYS_PER_YEAR),
            portfolio_return=portfolio_return * TRADING_DAYS_PER_YEAR,
            leverage_used=leverage_used,
            optimization_status="success" if convergence_error < 0.01 else "suboptimal",
            convergence_error=convergence_error,
            metadata={
                "covariance_matrix": cov_matrix.tolist(),
                "target_contributions": target_risk_contributions.tolist(),
                "optimization_method": "SLSQP",
            },
        )

        self.optimization_history.append(result)
        logger.info(
            f"Risk parity optimization completed. Convergence error: {convergence_error:.6f}"
        )

        return result

    def calculate_equal_risk_contribution(self, cov_matrix: np.ndarray) -> np.ndarray:
        """计算等风险贡献权重

        Args:
            cov_matrix: 协方差矩阵

        Returns:
            等风险贡献权重
        """
        n_assets = len(cov_matrix)
        target_risk_contributions = np.ones(n_assets) / n_assets

        return self._optimize_risk_parity_weights(cov_matrix, target_risk_contributions)

    def calculate_risk_budgeting_weights(
        self, returns_data: pd.DataFrame, risk_budgets: Dict[str, float]
    ) -> pd.Series:
        """计算风险预算权重

        Args:
            returns_data: 收益率数据
            risk_budgets: 风险预算字典

        Returns:
            权重Series
        """
        # 确保所有资产都有风险预算
        asset_names = list(returns_data.columns)
        risk_budget_array = np.array(
            [risk_budgets.get(asset, 1.0 / len(asset_names)) for asset in asset_names]
        )

        # 标准化风险预算
        risk_budget_array = risk_budget_array / risk_budget_array.sum()

        # 计算协方差矩阵
        cov_matrix = self._calculate_covariance_matrix(returns_data)

        # 优化权重
        weights = self._optimize_risk_parity_weights(cov_matrix, risk_budget_array)

        return pd.Series(weights, index=asset_names)

    def calculate_hierarchical_risk_parity(
        self, returns_data: pd.DataFrame
    ) -> pd.Series:
        """计算层次风险平价权重

        Args:
            returns_data: 收益率数据

        Returns:
            HRP权重
        """
        # 计算相关性矩阵
        corr_matrix = returns_data.corr()

        # 计算距离矩阵
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))

        # 层次聚类
        from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
        from scipy.spatial.distance import squareform

        # 转换为压缩距离矩阵
        condensed_dist = squareform(dist_matrix)

        # 层次聚类
        link = linkage(condensed_dist, method="single")

        # 获取排序后的索引
        sorted_indices = self._get_quasi_diag(link)

        # 重排序相关性矩阵
        sorted_corr = corr_matrix.iloc[sorted_indices, sorted_indices]

        # 递归二分配置
        weights = self._recursive_bisection(returns_data.cov().values, sorted_indices)

        # 创建权重Series
        asset_names = list(returns_data.columns)
        weight_dict = {asset_names[i]: weights[i] for i in range(len(weights))}

        return pd.Series(weight_dict)

    def adjust_for_correlation(
        self, weights: np.ndarray, correlation_matrix: np.ndarray
    ) -> np.ndarray:
        """根据相关性调整权重

        Args:
            weights: 原始权重
            correlation_matrix: 相关性矩阵

        Returns:
            调整后的权重
        """
        n_assets = len(weights)
        adjusted_weights = weights.copy()

        # 找出高度相关的资产对
        high_corr_pairs = []
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if abs(correlation_matrix[i, j]) > self.config.correlation_threshold:
                    high_corr_pairs.append((i, j, correlation_matrix[i, j]))

        # 调整高相关资产的权重
        for i, j, corr in high_corr_pairs:
            # 降低高相关资产的总权重
            total_weight = adjusted_weights[i] + adjusted_weights[j]
            reduction_factor = 1 - abs(corr - self.config.correlation_threshold) * 0.5

            new_total = total_weight * reduction_factor
            ratio = adjusted_weights[i] / (adjusted_weights[i] + adjusted_weights[j])

            adjusted_weights[i] = new_total * ratio
            adjusted_weights[j] = new_total * (1 - ratio)

        # 重新标准化权重
        adjusted_weights = adjusted_weights / adjusted_weights.sum()

        return adjusted_weights

    def _calculate_covariance_matrix(self, returns_data: pd.DataFrame) -> np.ndarray:
        """计算协方差矩阵

        Args:
            returns_data: 收益率数据

        Returns:
            协方差矩阵
        """
        if self.config.decay_factor < 1.0:
            # 使用指数加权协方差
            return self._ewma_covariance(returns_data, self.config.decay_factor)
        else:
            # 使用简单协方差
            return returns_data.cov().values

    def _ewma_covariance(
        self, returns_data: pd.DataFrame, decay_factor: float
    ) -> np.ndarray:
        """计算指数加权移动平均协方差

        Args:
            returns_data: 收益率数据
            decay_factor: 衰减因子

        Returns:
            EWMA协方差矩阵
        """
        n_periods = len(returns_data)
        n_assets = len(returns_data.columns)

        # 计算权重
        weights = np.array(
            [(1 - decay_factor) * decay_factor**i for i in range(n_periods - 1, -1, -1)]
        )
        weights = weights / weights.sum()

        # 中心化收益率
        mean_returns = (returns_data * weights.reshape(-1, 1)).sum()
        centered_returns = returns_data - mean_returns

        # 计算加权协方差
        cov_matrix = np.zeros((n_assets, n_assets))
        for i in range(n_periods):
            ret_i = centered_returns.iloc[i].values.reshape(-1, 1)
            cov_matrix += weights[i] * (ret_i @ ret_i.T)

        return cov_matrix

    def _optimize_risk_parity_weights(
        self, cov_matrix: np.ndarray, target_risk_contributions: np.ndarray
    ) -> np.ndarray:
        """优化风险平价权重

        Args:
            cov_matrix: 协方差矩阵
            target_risk_contributions: 目标风险贡献

        Returns:
            最优权重
        """
        n_assets = len(cov_matrix)

        # 目标函数：最小化风险贡献与目标的差异
        def objective(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol

            # 相对风险贡献
            rel_contrib = contrib / contrib.sum()

            # 计算与目标的差异（使用平方误差）
            error = np.sum((rel_contrib - target_risk_contributions) ** 2)

            return error

        # 约束条件
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # 权重和为1
        ]

        # 边界
        bounds = [
            (self.config.min_weight, self.config.max_weight) for _ in range(n_assets)
        ]

        # 初始猜测（等权重）
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )

        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")

        return result.x

    def _calculate_risk_contributions(
        self, weights: np.ndarray, cov_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算风险贡献

        Args:
            weights: 权重
            cov_matrix: 协方差矩阵

        Returns:
            (绝对风险贡献, 相对风险贡献)
        """
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)

        # 边际风险贡献
        marginal_contrib = cov_matrix @ weights

        # 绝对风险贡献
        abs_contrib = weights * marginal_contrib

        # 相对风险贡献
        rel_contrib = abs_contrib / abs_contrib.sum()

        return abs_contrib, rel_contrib

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """获取准对角化排序

        Args:
            link: 链接矩阵

        Returns:
            排序后的索引
        """
        link = link.astype(int)
        sorted_indices = []
        clusters = {i: [i] for i in range(link.shape[0] + 1)}

        for i in range(link.shape[0]):
            cluster1 = int(link[i, 0])
            cluster2 = int(link[i, 1])
            clusters[link.shape[0] + 1 + i] = clusters[cluster1] + clusters[cluster2]
            del clusters[cluster1]
            del clusters[cluster2]

        sorted_indices = clusters[max(clusters.keys())]

        return sorted_indices

    def _recursive_bisection(
        self, cov_matrix: np.ndarray, sorted_indices: List[int]
    ) -> np.ndarray:
        """递归二分配置

        Args:
            cov_matrix: 协方差矩阵
            sorted_indices: 排序后的索引

        Returns:
            权重数组
        """

        def _bisection(indices):
            if len(indices) == 1:
                return np.array([1.0])

            # 分成两组
            n_split = len(indices) // 2
            indices_left = indices[:n_split]
            indices_right = indices[n_split:]

            # 计算两组的协方差
            cov_left = cov_matrix[np.ix_(indices_left, indices_left)]
            cov_right = cov_matrix[np.ix_(indices_right, indices_right)]

            # 计算逆方差权重
            inv_vol_left = 1 / np.sqrt(
                np.ones(len(indices_left)) @ cov_left @ np.ones(len(indices_left))
            )
            inv_vol_right = 1 / np.sqrt(
                np.ones(len(indices_right)) @ cov_right @ np.ones(len(indices_right))
            )

            # 分配权重
            alpha = inv_vol_left / (inv_vol_left + inv_vol_right)

            # 递归
            weights_left = _bisection(indices_left) * alpha
            weights_right = _bisection(indices_right) * (1 - alpha)

            return np.concatenate([weights_left, weights_right])

        # 计算权重
        weights_sorted = _bisection(list(range(len(sorted_indices))))

        # 恢复原始顺序
        weights = np.zeros(len(sorted_indices))
        for i, idx in enumerate(sorted_indices):
            weights[idx] = weights_sorted[i]

        return weights


# 模块级别函数
def calculate_risk_parity_weights(
    returns_data: pd.DataFrame, config: Optional[RiskParityConfig] = None
) -> pd.Series:
    """计算风险平价权重的便捷函数

    Args:
        returns_data: 收益率数据
        config: 风险平价配置

    Returns:
        权重Series
    """
    rp = RiskParity(config)
    result = rp.apply_risk_parity_allocation(returns_data)

    return pd.Series(result.weights, index=result.asset_names)
