"""
均值方差优化器模块
实现Markowitz均值方差投资组合优化
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.constants import TRADING_DAYS_PER_YEAR
from common.exceptions import ModelError
from common.logging_system import setup_logger
from scipy import optimize

logger = setup_logger("mean_variance_optimizer")


class OptimizationObjective(Enum):
    """优化目标枚举"""

    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"


@dataclass
class MVOConfig:
    """均值方差优化配置"""

    objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE
    risk_free_rate: float = 0.02
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    min_weight: float = 0.0
    max_weight: float = 1.0
    allow_short: bool = False
    regularization: float = 0.0
    shrinkage_target: str = "constant_correlation"
    shrinkage_intensity: float = 0.1
    robust_estimation: bool = True
    constraints: Optional[Dict[str, Any]] = None


@dataclass
class EfficientFrontier:
    """有效前沿数据"""

    returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray
    weights: np.ndarray
    optimal_point: Dict[str, float]
    tangency_portfolio: Dict[str, float]
    min_variance_portfolio: Dict[str, float]


@dataclass
class OptimizationResult:
    """优化结果"""

    weights: np.ndarray
    asset_names: List[str]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_n: float
    max_drawdown_estimate: float
    var_95: float
    cvar_95: float
    optimization_status: str
    metadata: Dict[str, Any]


class MeanVarianceOptimizer:
    """均值方差优化器"""

    def __init__(self, config: Optional[MVOConfig] = None):
        """初始化均值方差优化器

        Args:
            config: MVO配置
        """
        self.config = config or MVOConfig()
        self.optimization_history: List[OptimizationResult] = []

    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """优化投资组合

        Args:
            expected_returns: 期望收益率Series
            cov_matrix: 协方差矩阵DataFrame
            constraints: 额外约束

        Returns:
            优化结果
        """
        logger.info(
            f"Starting portfolio optimization with objective: {self.config.objective.value}"
        )

        n_assets = len(expected_returns)
        asset_names = list(expected_returns.index)

        # 调整期望收益和协方差矩阵
        if self.config.robust_estimation:
            expected_returns_adj = self._shrink_expected_returns(expected_returns)
            cov_matrix_adj = self._shrink_covariance(cov_matrix)
        else:
            expected_returns_adj = expected_returns.values
            cov_matrix_adj = cov_matrix.values

        # 根据目标选择优化方法
        if self.config.objective == OptimizationObjective.MAX_SHARPE:
            weights = self._maximize_sharpe_ratio(expected_returns_adj, cov_matrix_adj)
        elif self.config.objective == OptimizationObjective.MIN_VARIANCE:
            weights = self._minimize_variance(cov_matrix_adj)
        elif self.config.objective == OptimizationObjective.MAX_RETURN:
            weights = self._maximize_return(expected_returns_adj, cov_matrix_adj)
        elif self.config.objective == OptimizationObjective.MAX_DIVERSIFICATION:
            weights = self._maximize_diversification(cov_matrix_adj)
        else:
            raise ValueError(f"Unknown objective: {self.config.objective}")

        # 计算组合统计
        portfolio_return = weights @ expected_returns_adj
        portfolio_volatility = np.sqrt(weights @ cov_matrix_adj @ weights)
        sharpe_ratio = (
            portfolio_return - self.config.risk_free_rate
        ) / portfolio_volatility

        # 计算其他风险指标
        diversification_ratio = self._calculate_diversification_ratio(
            weights, cov_matrix_adj
        )
        effective_n = self._calculate_effective_n(weights)
        max_dd_estimate = self._estimate_max_drawdown(
            portfolio_volatility, sharpe_ratio
        )
        var_95, cvar_95 = self._calculate_var_cvar(
            portfolio_return, portfolio_volatility
        )

        result = OptimizationResult(
            weights=weights,
            asset_names=asset_names,
            expected_return=portfolio_return * TRADING_DAYS_PER_YEAR,
            expected_volatility=portfolio_volatility * np.sqrt(TRADING_DAYS_PER_YEAR),
            sharpe_ratio=sharpe_ratio * np.sqrt(TRADING_DAYS_PER_YEAR),
            diversification_ratio=diversification_ratio,
            effective_n=effective_n,
            max_drawdown_estimate=max_dd_estimate,
            var_95=var_95,
            cvar_95=cvar_95,
            optimization_status="success",
            metadata={
                "objective": self.config.objective.value,
                "risk_free_rate": self.config.risk_free_rate,
                "regularization": self.config.regularization,
            },
        )

        self.optimization_history.append(result)
        logger.info(f"Optimization completed. Sharpe ratio: {result.sharpe_ratio:.4f}")

        return result

    def calculate_efficient_frontier(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        n_portfolios: int = 50,
    ) -> EfficientFrontier:
        """计算有效前沿

        Args:
            expected_returns: 期望收益率
            cov_matrix: 协方差矩阵
            n_portfolios: 前沿上的组合数量

        Returns:
            有效前沿数据
        """
        logger.info(f"Calculating efficient frontier with {n_portfolios} portfolios")

        # 调整输入
        if self.config.robust_estimation:
            returns = self._shrink_expected_returns(expected_returns)
            cov = self._shrink_covariance(cov_matrix)
        else:
            returns = expected_returns.values
            cov = cov_matrix.values

        # 计算最小方差组合
        min_var_weights = self._minimize_variance(cov)
        min_var_return = min_var_weights @ returns
        min_var_vol = np.sqrt(min_var_weights @ cov @ min_var_weights)

        # 计算最大收益组合
        max_ret_weights = self._maximize_return(returns, cov)
        max_ret_return = max_ret_weights @ returns
        max_ret_vol = np.sqrt(max_ret_weights @ cov @ max_ret_weights)

        # 生成目标收益率范围
        target_returns = np.linspace(min_var_return, max_ret_return, n_portfolios)

        # 计算每个目标收益率的最优组合
        frontier_returns = []
        frontier_volatilities = []
        frontier_weights = []
        frontier_sharpes = []

        for target_return in target_returns:
            weights = self._minimize_variance_for_target_return(
                returns, cov, target_return
            )

            if weights is not None:
                portfolio_return = weights @ returns
                portfolio_vol = np.sqrt(weights @ cov @ weights)
                sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

                frontier_returns.append(portfolio_return)
                frontier_volatilities.append(portfolio_vol)
                frontier_weights.append(weights)
                frontier_sharpes.append(sharpe)

        # 找出切线组合（最大夏普）
        max_sharpe_idx = np.argmax(frontier_sharpes)
        tangency_weights = frontier_weights[max_sharpe_idx]

        # 构建结果
        frontier = EfficientFrontier(
            returns=np.array(frontier_returns) * TRADING_DAYS_PER_YEAR,
            volatilities=np.array(frontier_volatilities)
            * np.sqrt(TRADING_DAYS_PER_YEAR),
            sharpe_ratios=np.array(frontier_sharpes) * np.sqrt(TRADING_DAYS_PER_YEAR),
            weights=np.array(frontier_weights),
            optimal_point={
                "return": frontier_returns[max_sharpe_idx] * TRADING_DAYS_PER_YEAR,
                "volatility": frontier_volatilities[max_sharpe_idx]
                * np.sqrt(TRADING_DAYS_PER_YEAR),
                "sharpe": frontier_sharpes[max_sharpe_idx]
                * np.sqrt(TRADING_DAYS_PER_YEAR),
            },
            tangency_portfolio={
                "weights": tangency_weights.tolist(),
                "return": frontier_returns[max_sharpe_idx] * TRADING_DAYS_PER_YEAR,
                "volatility": frontier_volatilities[max_sharpe_idx]
                * np.sqrt(TRADING_DAYS_PER_YEAR),
            },
            min_variance_portfolio={
                "weights": min_var_weights.tolist(),
                "return": min_var_return * TRADING_DAYS_PER_YEAR,
                "volatility": min_var_vol * np.sqrt(TRADING_DAYS_PER_YEAR),
            },
        )

        return frontier

    def apply_black_litterman(
        self,
        market_caps: pd.Series,
        cov_matrix: pd.DataFrame,
        views: pd.DataFrame,
        view_confidence: pd.Series,
        tau: float = 0.05,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """应用Black-Litterman模型

        Args:
            market_caps: 市值Series
            cov_matrix: 协方差矩阵
            views: 观点矩阵
            view_confidence: 观点置信度
            tau: tau参数

        Returns:
            (后验期望收益, 后验协方差矩阵)
        """
        # 计算市场均衡收益
        total_market_cap = market_caps.sum()
        market_weights = market_caps / total_market_cap

        # 隐含均衡收益率
        lambda_mkt = self.config.risk_free_rate + self._calculate_market_risk_premium(
            cov_matrix, market_weights
        )
        pi = lambda_mkt * (cov_matrix @ market_weights)

        # 观点矩阵P和观点向量Q
        P = views.values
        Q = views @ pi

        # 观点误差协方差矩阵Omega
        omega_diag = view_confidence.values**2
        Omega = np.diag(omega_diag)

        # Black-Litterman后验期望收益
        tau_sigma = tau * cov_matrix
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(Omega)

        posterior_mean = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P) @ (
            inv_tau_sigma @ pi + P.T @ inv_omega @ Q
        )

        # 后验协方差
        posterior_cov = (
            cov_matrix
            + tau_sigma
            - tau_sigma
            @ P.T
            @ np.linalg.inv(Omega + P @ tau_sigma @ P.T)
            @ P
            @ tau_sigma
        )

        return pd.Series(posterior_mean, index=cov_matrix.index), pd.DataFrame(
            posterior_cov, index=cov_matrix.index, columns=cov_matrix.columns
        )

    def _maximize_sharpe_ratio(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """最大化夏普比率

        Args:
            expected_returns: 期望收益率
            cov_matrix: 协方差矩阵

        Returns:
            最优权重
        """
        n_assets = len(expected_returns)

        # 目标函数（负夏普比率）
        def neg_sharpe(weights):
            portfolio_return = weights @ expected_returns
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

            # 添加正则化项
            if self.config.regularization > 0:
                regularization = self.config.regularization * np.sum(weights**2)
                return -sharpe + regularization

            return -sharpe

        # 约束和边界
        constraints = self._get_optimization_constraints(n_assets)
        bounds = self._get_optimization_bounds(n_assets)

        # 初始猜测
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = optimize.minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not result.success:
            logger.warning(f"Sharpe optimization did not converge: {result.message}")

        return result.x

    def _minimize_variance(self, cov_matrix: np.ndarray) -> np.ndarray:
        """最小化方差

        Args:
            cov_matrix: 协方差矩阵

        Returns:
            最优权重
        """
        n_assets = len(cov_matrix)

        # 目标函数
        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights

        # 约束和边界
        constraints = self._get_optimization_constraints(n_assets)
        bounds = self._get_optimization_bounds(n_assets)

        # 初始猜测
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = optimize.minimize(
            portfolio_variance,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    def _maximize_return(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """最大化收益（给定风险约束）

        Args:
            expected_returns: 期望收益率
            cov_matrix: 协方差矩阵

        Returns:
            最优权重
        """
        n_assets = len(expected_returns)

        # 目标函数（负收益）
        def neg_return(weights):
            return -weights @ expected_returns

        # 约束
        constraints = self._get_optimization_constraints(n_assets)

        # 添加风险约束
        if self.config.target_volatility is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: self.config.target_volatility**2
                    - x @ cov_matrix @ x,
                }
            )

        bounds = self._get_optimization_bounds(n_assets)

        # 初始猜测
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = optimize.minimize(
            neg_return, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        return result.x

    def _maximize_diversification(self, cov_matrix: np.ndarray) -> np.ndarray:
        """最大化分散化比率

        Args:
            cov_matrix: 协方差矩阵

        Returns:
            最优权重
        """
        n_assets = len(cov_matrix)

        # 计算标准差
        stds = np.sqrt(np.diag(cov_matrix))

        # 目标函数（负分散化比率）
        def neg_diversification(weights):
            weighted_avg_vol = weights @ stds
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            return -weighted_avg_vol / portfolio_vol

        # 约束和边界
        constraints = self._get_optimization_constraints(n_assets)
        bounds = self._get_optimization_bounds(n_assets)

        # 初始猜测
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = optimize.minimize(
            neg_diversification,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    def _minimize_variance_for_target_return(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray, target_return: float
    ) -> Optional[np.ndarray]:
        """为目标收益率最小化方差

        Args:
            expected_returns: 期望收益率
            cov_matrix: 协方差矩阵
            target_return: 目标收益率

        Returns:
            最优权重或None
        """
        n_assets = len(expected_returns)

        # 目标函数
        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights

        # 约束
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {"type": "eq", "fun": lambda x: x @ expected_returns - target_return},
        ]

        bounds = self._get_optimization_bounds(n_assets)

        # 初始猜测
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = optimize.minimize(
            portfolio_variance,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9},
        )

        if result.success:
            return result.x
        else:
            return None

    def _get_optimization_constraints(self, n_assets: int) -> List[Dict[str, Any]]:
        """获取优化约束

        Args:
            n_assets: 资产数量

        Returns:
            约束列表
        """
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # 权重和为1
        ]

        # 添加自定义约束
        if self.config.constraints:
            if "sector_limits" in self.config.constraints:
                # 行业限制
                for sector, limit in self.config.constraints["sector_limits"].items():
                    # 需要sector_mapping来实现
                    pass

            if "max_positions" in self.config.constraints:
                # 最大持仓数限制
                max_pos = self.config.constraints["max_positions"]
                # 这需要使用混合整数规划
                pass

        return constraints

    def _get_optimization_bounds(self, n_assets: int) -> List[Tuple[float, float]]:
        """获取优化边界

        Args:
            n_assets: 资产数量

        Returns:
            边界列表
        """
        if self.config.allow_short:
            # 允许做空
            return [
                (-self.config.max_weight, self.config.max_weight)
                for _ in range(n_assets)
            ]
        else:
            # 只做多
            return [
                (self.config.min_weight, self.config.max_weight)
                for _ in range(n_assets)
            ]

    def _shrink_expected_returns(self, expected_returns: pd.Series) -> np.ndarray:
        """收缩期望收益估计

        Args:
            expected_returns: 原始期望收益

        Returns:
            收缩后的期望收益
        """
        returns = expected_returns.values

        # James-Stein收缩估计
        n = len(returns)
        mean_return = np.mean(returns)

        # 收缩强度
        shrinkage = min(1, (n - 2) / (n * np.sum((returns - mean_return) ** 2)))

        # 收缩后的估计
        shrunk_returns = shrinkage * mean_return + (1 - shrinkage) * returns

        return shrunk_returns

    def _shrink_covariance(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """收缩协方差矩阵估计

        Args:
            cov_matrix: 原始协方差矩阵

        Returns:
            收缩后的协方差矩阵
        """
        cov = cov_matrix.values
        n = len(cov)

        # 计算收缩目标
        if self.config.shrinkage_target == "constant_correlation":
            # 常相关矩阵
            avg_corr = (np.sum(np.corrcoef(cov)) - n) / (n * (n - 1))
            target = np.eye(n) + avg_corr * (np.ones((n, n)) - np.eye(n))
            stds = np.sqrt(np.diag(cov))
            target = np.outer(stds, stds) * target
        elif self.config.shrinkage_target == "diagonal":
            # 对角矩阵
            target = np.diag(np.diag(cov))
        else:
            # 单位矩阵
            target = np.eye(n) * np.trace(cov) / n

        # Ledoit-Wolf收缩
        shrunk_cov = (
            1 - self.config.shrinkage_intensity
        ) * cov + self.config.shrinkage_intensity * target

        return shrunk_cov

    def _calculate_diversification_ratio(
        self, weights: np.ndarray, cov_matrix: np.ndarray
    ) -> float:
        """计算分散化比率

        Args:
            weights: 权重
            cov_matrix: 协方差矩阵

        Returns:
            分散化比率
        """
        stds = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = weights @ stds
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)

        return weighted_avg_vol / portfolio_vol

    def _calculate_effective_n(self, weights: np.ndarray) -> float:
        """计算有效资产数量

        Args:
            weights: 权重

        Returns:
            有效N
        """
        # 基于赫芬达尔指数
        return 1 / np.sum(weights**2)

    def _estimate_max_drawdown(
        self, volatility: float, sharpe_ratio: float, time_horizon: int = 252
    ) -> float:
        """估计最大回撤

        Args:
            volatility: 波动率
            sharpe_ratio: 夏普比率
            time_horizon: 时间范围

        Returns:
            估计的最大回撤
        """
        # 基于布朗运动的近似
        drift = sharpe_ratio * volatility

        # 预期最大回撤的近似公式
        if drift > 0:
            max_dd = -2 * volatility * np.sqrt(time_horizon / (2 * np.pi))
        else:
            max_dd = -volatility * np.sqrt(time_horizon) * 2.5

        return abs(max_dd)

    def _calculate_var_cvar(
        self, expected_return: float, volatility: float, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """计算VaR和CVaR

        Args:
            expected_return: 期望收益
            volatility: 波动率
            confidence_level: 置信水平

        Returns:
            (VaR, CVaR)
        """
        from scipy import stats

        # 假设正态分布
        z_score = stats.norm.ppf(1 - confidence_level)

        # VaR
        var = expected_return + z_score * volatility

        # CVaR (条件VaR)
        pdf_z = stats.norm.pdf(z_score)
        cvar = expected_return - volatility * pdf_z / (1 - confidence_level)

        return var, cvar

    def _calculate_market_risk_premium(
        self, cov_matrix: pd.DataFrame, market_weights: pd.Series
    ) -> float:
        """计算市场风险溢价

        Args:
            cov_matrix: 协方差矩阵
            market_weights: 市场权重

        Returns:
            市场风险溢价
        """
        market_variance = market_weights @ cov_matrix @ market_weights

        # 使用历史平均或设定值
        # 这里使用典型的市场风险溢价
        return 0.05  # 5%年化


# 模块级别函数
def optimize_mean_variance(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    objective: str = "max_sharpe",
    config: Optional[MVOConfig] = None,
) -> pd.Series:
    """均值方差优化的便捷函数

    Args:
        expected_returns: 期望收益率
        cov_matrix: 协方差矩阵
        objective: 优化目标
        config: 配置

    Returns:
        权重Series
    """
    if config is None:
        config = MVOConfig()
    config.objective = OptimizationObjective(objective)

    optimizer = MeanVarianceOptimizer(config)
    result = optimizer.optimize_portfolio(expected_returns, cov_matrix)

    return pd.Series(result.weights, index=result.asset_names)
