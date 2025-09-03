"""
风险归因分析器模块
提供投资组合风险来源的分解和归因分析
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from common.constants import TRADING_DAYS_PER_YEAR
from common.logging_system import setup_logger
from scipy.optimize import minimize
from sklearn.decomposition import PCA

logger = setup_logger("risk_attribution")


@dataclass
class RiskAttributionReport:
    """风险归因报告数据类"""

    total_risk: float
    risk_decomposition: Dict[str, float]
    factor_contributions: Dict[str, float]
    marginal_contributions: Dict[str, float]
    component_var: Dict[str, float]
    correlation_matrix: pd.DataFrame
    principal_components: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_risk": self.total_risk,
            "risk_decomposition": self.risk_decomposition,
            "factor_contributions": self.factor_contributions,
            "marginal_contributions": self.marginal_contributions,
            "component_var": self.component_var,
            "correlation_matrix": self.correlation_matrix.to_dict(),
        }


class RiskAttributor:
    """风险归因分析器类"""

    def __init__(self):
        """初始化风险归因分析器"""
        self.portfolio_returns: Optional[pd.DataFrame] = None
        self.weights: Optional[np.ndarray] = None
        self.factor_returns: Optional[pd.DataFrame] = None

    def attribute_risk(
        self,
        portfolio_returns: pd.DataFrame,
        weights: np.ndarray,
        factor_returns: Optional[pd.DataFrame] = None,
    ) -> RiskAttributionReport:
        """执行风险归因分析

        Args:
            portfolio_returns: 组合中各资产的收益率DataFrame
            weights: 资产权重数组
            factor_returns: 因子收益率DataFrame

        Returns:
            风险归因报告
        """
        self.portfolio_returns = portfolio_returns
        self.weights = weights / weights.sum()  # 标准化权重
        self.factor_returns = factor_returns

        # 计算总风险
        total_risk = self._calculate_portfolio_risk()

        # 风险分解
        risk_decomposition = self._decompose_risk()

        # 边际风险贡献
        marginal_contributions = self._calculate_marginal_contributions()

        # 成分VaR
        component_var = self._calculate_component_var()

        # 相关性矩阵
        correlation_matrix = portfolio_returns.corr()

        # 因子贡献（如果提供了因子数据）
        factor_contributions = {}
        if factor_returns is not None:
            factor_contributions = self._calculate_factor_contributions()

        # 主成分分析
        principal_components = self._perform_pca()

        return RiskAttributionReport(
            total_risk=total_risk,
            risk_decomposition=risk_decomposition,
            factor_contributions=factor_contributions,
            marginal_contributions=marginal_contributions,
            component_var=component_var,
            correlation_matrix=correlation_matrix,
            principal_components=principal_components,
        )

    def _calculate_portfolio_risk(self) -> float:
        """计算组合总风险

        Returns:
            年化风险（标准差）
        """
        returns = self.portfolio_returns
        weights = self.weights

        # 计算协方差矩阵
        cov_matrix = returns.cov() * TRADING_DAYS_PER_YEAR

        # 组合方差
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

        # 组合标准差
        portfolio_risk = np.sqrt(portfolio_variance)

        return portfolio_risk

    def _decompose_risk(self) -> Dict[str, float]:
        """分解风险来源

        Returns:
            风险分解字典
        """
        returns = self.portfolio_returns
        weights = self.weights

        # 计算各资产的风险贡献
        cov_matrix = returns.cov() * TRADING_DAYS_PER_YEAR

        # 单个资产风险
        individual_risks = {}
        for i, asset in enumerate(returns.columns):
            asset_variance = cov_matrix.iloc[i, i]
            asset_risk = np.sqrt(asset_variance)
            individual_risks[f"{asset}_standalone"] = asset_risk

        # 加权风险贡献
        weighted_contributions = {}
        portfolio_risk = self._calculate_portfolio_risk()

        for i, asset in enumerate(returns.columns):
            # 计算资产i对组合风险的贡献
            marginal_contribution = np.dot(cov_matrix.iloc[i], weights) / portfolio_risk
            contribution = weights[i] * marginal_contribution
            weighted_contributions[f"{asset}_contribution"] = contribution

        # 分散化收益
        weighted_sum_risk = sum(
            weights[i] * np.sqrt(cov_matrix.iloc[i, i]) for i in range(len(weights))
        )
        diversification_ratio = weighted_sum_risk / portfolio_risk

        result = {
            "portfolio_risk": portfolio_risk,
            "weighted_average_risk": weighted_sum_risk,
            "diversification_ratio": diversification_ratio,
            "diversification_benefit": weighted_sum_risk - portfolio_risk,
        }
        result.update(individual_risks)
        result.update(weighted_contributions)

        return result

    def _calculate_marginal_contributions(self) -> Dict[str, float]:
        """计算边际风险贡献

        Returns:
            边际贡献字典
        """
        returns = self.portfolio_returns
        weights = self.weights

        cov_matrix = returns.cov() * TRADING_DAYS_PER_YEAR
        portfolio_risk = self._calculate_portfolio_risk()

        marginal_contributions = {}

        for i, asset in enumerate(returns.columns):
            # 边际风险贡献 = ∂σ_p/∂w_i
            marginal_risk = np.dot(cov_matrix.iloc[i], weights) / portfolio_risk
            marginal_contributions[asset] = marginal_risk

            # 百分比贡献
            pct_contribution = (weights[i] * marginal_risk) / portfolio_risk
            marginal_contributions[f"{asset}_pct"] = pct_contribution

        return marginal_contributions

    def _calculate_component_var(
        self, confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """计算成分VaR

        Args:
            confidence_level: 置信水平

        Returns:
            成分VaR字典
        """
        returns = self.portfolio_returns
        weights = self.weights

        # 计算组合收益
        portfolio_return = np.dot(returns, weights)

        # 计算VaR
        from scipy import stats

        z_score = stats.norm.ppf(1 - confidence_level)

        # 组合VaR
        portfolio_var = -np.percentile(portfolio_return, (1 - confidence_level) * 100)

        # 成分VaR
        component_vars = {}

        for i, asset in enumerate(returns.columns):
            # 计算资产i的成分VaR
            asset_returns = returns.iloc[:, i]
            asset_var = -np.percentile(asset_returns, (1 - confidence_level) * 100)

            # 加权成分VaR
            component_var = weights[i] * asset_var
            component_vars[asset] = component_var

            # 相对贡献
            relative_contribution = (
                component_var / portfolio_var if portfolio_var != 0 else 0
            )
            component_vars[f"{asset}_relative"] = relative_contribution

        component_vars["portfolio_var"] = portfolio_var

        return component_vars

    def _calculate_factor_contributions(self) -> Dict[str, float]:
        """计算因子风险贡献

        Returns:
            因子贡献字典
        """
        returns = self.portfolio_returns
        weights = self.weights
        factors = self.factor_returns

        # 计算组合收益
        portfolio_return = np.dot(returns, weights)

        # 因子暴露（通过回归计算）
        from sklearn.linear_model import LinearRegression

        # 对齐数据
        aligned_portfolio = pd.Series(portfolio_return, index=returns.index)
        aligned_factors = factors.reindex(returns.index)

        # 移除NaN
        valid_idx = ~(aligned_portfolio.isna() | aligned_factors.isna().any(axis=1))
        aligned_portfolio = aligned_portfolio[valid_idx]
        aligned_factors = aligned_factors[valid_idx]

        # 回归分析
        model = LinearRegression()
        model.fit(aligned_factors.values, aligned_portfolio.values)

        # 因子载荷
        factor_loadings = dict(zip(factors.columns, model.coef_))

        # 因子风险贡献
        factor_contributions = {}

        # 计算因子协方差矩阵
        factor_cov = aligned_factors.cov() * TRADING_DAYS_PER_YEAR

        # 总因子风险
        factor_variance = 0
        for i, factor1 in enumerate(factors.columns):
            for j, factor2 in enumerate(factors.columns):
                loading1 = model.coef_[i]
                loading2 = model.coef_[j]
                factor_variance += loading1 * loading2 * factor_cov.iloc[i, j]

        factor_risk = np.sqrt(factor_variance)

        # 各因子的风险贡献
        for i, factor in enumerate(factors.columns):
            factor_vol = aligned_factors.iloc[:, i].std() * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
            contribution = abs(model.coef_[i]) * factor_vol
            factor_contributions[factor] = contribution
            factor_contributions[f"{factor}_loading"] = model.coef_[i]

        # 特质风险
        residuals = aligned_portfolio.values - model.predict(aligned_factors.values)
        idiosyncratic_risk = np.std(residuals) * np.sqrt(TRADING_DAYS_PER_YEAR)

        factor_contributions["factor_risk"] = factor_risk
        factor_contributions["idiosyncratic_risk"] = idiosyncratic_risk
        factor_contributions["r_squared"] = model.score(
            aligned_factors.values, aligned_portfolio.values
        )

        return factor_contributions

    def _perform_pca(self, n_components: int = 5) -> Dict[str, Any]:
        """执行主成分分析

        Args:
            n_components: 主成分数量

        Returns:
            PCA结果字典
        """
        returns = self.portfolio_returns

        # 标准化收益率
        standardized_returns = (returns - returns.mean()) / returns.std()

        # PCA分析
        pca = PCA(n_components=min(n_components, len(returns.columns)))
        pca.fit(standardized_returns.fillna(0))

        # 主成分载荷
        components = pd.DataFrame(
            pca.components_,
            columns=returns.columns,
            index=[f"PC{i + 1}" for i in range(pca.n_components_)],
        )

        # 解释方差
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        # 主成分得分
        pc_scores = pca.transform(standardized_returns.fillna(0))

        return {
            "components": components.to_dict(),
            "explained_variance": list(explained_variance),
            "cumulative_variance": list(cumulative_variance),
            "n_components_95pct": int(np.argmax(cumulative_variance >= 0.95) + 1),
        }


# 模块级别函数
def attribute_portfolio_risk(
    portfolio_returns: pd.DataFrame, weights: np.ndarray
) -> RiskAttributionReport:
    """执行组合风险归因的便捷函数

    Args:
        portfolio_returns: 资产收益率DataFrame
        weights: 权重数组

    Returns:
        风险归因报告
    """
    attributor = RiskAttributor()
    return attributor.attribute_risk(portfolio_returns, weights)


def calculate_risk_contributions(
    returns: pd.DataFrame, weights: np.ndarray
) -> pd.DataFrame:
    """计算风险贡献的便捷函数

    Args:
        returns: 收益率DataFrame
        weights: 权重数组

    Returns:
        风险贡献DataFrame
    """
    attributor = RiskAttributor()
    report = attributor.attribute_risk(returns, weights)

    contributions = pd.DataFrame(
        {
            "Weight": weights,
            "Marginal_Risk": list(report.marginal_contributions.values())[
                : len(weights)
            ],
            "Risk_Contribution": [
                weights[i] * list(report.marginal_contributions.values())[i]
                for i in range(len(weights))
            ],
        },
        index=returns.columns,
    )

    contributions["Pct_Contribution"] = (
        contributions["Risk_Contribution"] / contributions["Risk_Contribution"].sum()
    )

    return contributions
