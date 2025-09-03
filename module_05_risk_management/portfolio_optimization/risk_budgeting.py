"""
风险预算优化器模块
实现风险预算和风险贡献优化
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.exceptions import ModelError
from common.logging_system import setup_logger
from scipy import optimize

logger = setup_logger("risk_budgeting")


@dataclass
class RiskBudgetConfig:
    """风险预算配置"""

    optimization_method: str = (
        "sequential_quadratic"  # sequential_quadratic, trust_region
    )
    max_iterations: int = 1000
    tolerance: float = 1e-8
    use_log_barrier: bool = True
    barrier_parameter: float = 0.1
    min_weight: float = 0.0
    max_weight: float = 1.0
    allow_short: bool = False
    regularization: float = 0.0


@dataclass
class RiskBudget:
    """风险预算"""

    asset_name: str
    target_risk_contribution: float
    min_risk_contribution: float
    max_risk_contribution: float
    priority: int  # 优先级，用于分层优化


@dataclass
class RiskBudgetingResult:
    """风险预算优化结果"""

    optimal_weights: pd.Series
    risk_contributions: pd.Series
    risk_contribution_pct: pd.Series
    total_portfolio_risk: float
    risk_budgets: pd.Series
    budget_deviation: pd.Series
    convergence_error: float
    optimization_status: str
    iterations: int


class RiskBudgetingOptimizer:
    """风险预算优化器类"""

    def __init__(self, config: Optional[RiskBudgetConfig] = None):
        """初始化风险预算优化器

        Args:
            config: 风险预算配置
        """
        self.config = config or RiskBudgetConfig()
        self.optimization_history: List[RiskBudgetingResult] = []

    def optimize_risk_budget(
        self,
        covariance_matrix: pd.DataFrame,
        risk_budgets: Dict[str, float],
        initial_weights: Optional[pd.Series] = None,
    ) -> RiskBudgetingResult:
        """优化风险预算

        Args:
            covariance_matrix: 协方差矩阵
            risk_budgets: 风险预算字典
            initial_weights: 初始权重

        Returns:
            风险预算优化结果
        """
        logger.info("Starting risk budgeting optimization...")

        assets = list(covariance_matrix.index)
        n_assets = len(assets)

        # 标准化风险预算
        budget_array = np.array(
            [risk_budgets.get(asset, 1.0 / n_assets) for asset in assets]
        )
        budget_array = budget_array / budget_array.sum()

        # 初始权重
        if initial_weights is None:
            x0 = np.ones(n_assets) / n_assets
        else:
            x0 = initial_weights.values

        # 优化
        if self.config.optimization_method == "sequential_quadratic":
            result = self._optimize_sequential_quadratic(
                covariance_matrix.values, budget_array, x0
            )
        else:
            result = self._optimize_trust_region(
                covariance_matrix.values, budget_array, x0
            )

        # 提取结果
        optimal_weights = result.x

        # 计算风险贡献
        risk_contributions, risk_pct = self._calculate_risk_contributions(
            optimal_weights, covariance_matrix.values
        )

        # 计算总风险
        total_risk = np.sqrt(
            optimal_weights @ covariance_matrix.values @ optimal_weights
        )

        # 计算预算偏差
        budget_deviation = risk_pct - budget_array

        # 构建结果
        result_obj = RiskBudgetingResult(
            optimal_weights=pd.Series(optimal_weights, index=assets),
            risk_contributions=pd.Series(risk_contributions, index=assets),
            risk_contribution_pct=pd.Series(risk_pct, index=assets),
            total_portfolio_risk=total_risk,
            risk_budgets=pd.Series(budget_array, index=assets),
            budget_deviation=pd.Series(budget_deviation, index=assets),
            convergence_error=np.sum(np.abs(budget_deviation)),
            optimization_status="success" if result.success else "failed",
            iterations=result.nit,
        )

        self.optimization_history.append(result_obj)

        logger.info(
            f"Risk budgeting completed. Convergence error: {result_obj.convergence_error:.6f}"
        )

        return result_obj

    def optimize_hierarchical_risk_budget(
        self,
        covariance_matrix: pd.DataFrame,
        risk_budgets: List[RiskBudget],
        hierarchy_levels: Dict[str, int],
    ) -> RiskBudgetingResult:
        """优化层次化风险预算

        Args:
            covariance_matrix: 协方差矩阵
            risk_budgets: 风险预算列表
            hierarchy_levels: 层次级别

        Returns:
            风险预算优化结果
        """
        logger.info("Starting hierarchical risk budgeting...")

        # 按优先级分组
        priority_groups = {}
        for budget in risk_budgets:
            if budget.priority not in priority_groups:
                priority_groups[budget.priority] = []
            priority_groups[budget.priority].append(budget)

        # 逐层优化
        allocated_weights = {}
        remaining_risk = 1.0

        for priority in sorted(priority_groups.keys()):
            group_budgets = priority_groups[priority]

            # 该层的总风险预算
            group_total_risk = sum(b.target_risk_contribution for b in group_budgets)
            group_total_risk = min(group_total_risk, remaining_risk)

            # 优化该层
            group_assets = [b.asset_name for b in group_budgets]
            group_cov = covariance_matrix.loc[group_assets, group_assets]

            group_risk_budgets = {
                b.asset_name: b.target_risk_contribution / group_total_risk
                for b in group_budgets
            }

            group_result = self.optimize_risk_budget(group_cov, group_risk_budgets)

            # 分配权重
            for asset, weight in group_result.optimal_weights.items():
                allocated_weights[asset] = weight * group_total_risk

            remaining_risk -= group_total_risk

        # 构建最终权重
        all_assets = list(covariance_matrix.index)
        final_weights = pd.Series(
            [allocated_weights.get(asset, 0.0) for asset in all_assets],
            index=all_assets,
        )

        # 计算最终风险贡献
        risk_contributions, risk_pct = self._calculate_risk_contributions(
            final_weights.values, covariance_matrix.values
        )

        # 构建结果
        target_budgets = pd.Series(
            [
                next(
                    (
                        b.target_risk_contribution
                        for b in risk_budgets
                        if b.asset_name == asset
                    ),
                    0.0,
                )
                for asset in all_assets
            ],
            index=all_assets,
        )

        result = RiskBudgetingResult(
            optimal_weights=final_weights,
            risk_contributions=pd.Series(risk_contributions, index=all_assets),
            risk_contribution_pct=pd.Series(risk_pct, index=all_assets),
            total_portfolio_risk=np.sqrt(
                final_weights @ covariance_matrix @ final_weights
            ),
            risk_budgets=target_budgets,
            budget_deviation=pd.Series(risk_pct, index=all_assets) - target_budgets,
            convergence_error=np.sum(np.abs(risk_pct - target_budgets.values)),
            optimization_status="success",
            iterations=0,
        )

        return result

    def calculate_marginal_risk_contributions(
        self, weights: pd.Series, covariance_matrix: pd.DataFrame
    ) -> pd.Series:
        """计算边际风险贡献

        Args:
            weights: 权重
            covariance_matrix: 协方差矩阵

        Returns:
            边际风险贡献Series
        """
        portfolio_variance = weights @ covariance_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)

        # 边际风险贡献 = ∂σ/∂w = (Σw) / σ
        marginal_risk = (covariance_matrix @ weights) / portfolio_vol

        return pd.Series(marginal_risk, index=weights.index)

    def calculate_component_risk_contributions(
        self, weights: pd.Series, covariance_matrix: pd.DataFrame
    ) -> pd.Series:
        """计算成分风险贡献

        Args:
            weights: 权重
            covariance_matrix: 协方差矩阵

        Returns:
            成分风险贡献Series
        """
        marginal_risk = self.calculate_marginal_risk_contributions(
            weights, covariance_matrix
        )

        # 成分风险贡献 = w * ∂σ/∂w
        component_risk = weights * marginal_risk

        return component_risk

    def enforce_risk_budget_constraints(
        self,
        weights: pd.Series,
        risk_budgets: Dict[str, Tuple[float, float]],
        covariance_matrix: pd.DataFrame,
    ) -> pd.Series:
        """强制执行风险预算约束

        Args:
            weights: 原始权重
            risk_budgets: 风险预算约束（最小值，最大值）
            covariance_matrix: 协方差矩阵

        Returns:
            调整后的权重
        """
        # 计算当前风险贡献
        risk_contributions = self.calculate_component_risk_contributions(
            weights, covariance_matrix
        )
        risk_pct = risk_contributions / risk_contributions.sum()

        # 检查违反约束的资产
        adjusted_weights = weights.copy()

        for asset, (min_risk, max_risk) in risk_budgets.items():
            if asset in risk_pct.index:
                current_risk = risk_pct[asset]

                if current_risk < min_risk:
                    # 增加权重
                    scale_factor = min_risk / current_risk
                    adjusted_weights[asset] *= scale_factor
                elif current_risk > max_risk:
                    # 减少权重
                    scale_factor = max_risk / current_risk
                    adjusted_weights[asset] *= scale_factor

        # 重新标准化
        adjusted_weights = adjusted_weights / adjusted_weights.sum()

        return adjusted_weights

    def _optimize_sequential_quadratic(
        self,
        covariance: np.ndarray,
        target_risk_contributions: np.ndarray,
        x0: np.ndarray,
    ) -> optimize.OptimizeResult:
        """使用序列二次规划优化

        Args:
            covariance: 协方差矩阵
            target_risk_contributions: 目标风险贡献
            x0: 初始权重

        Returns:
            优化结果
        """
        n_assets = len(x0)

        def objective(weights):
            """目标函数：最小化风险贡献与目标的偏差"""
            # 计算风险贡献
            portfolio_vol = np.sqrt(weights @ covariance @ weights)
            marginal_risk = (covariance @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk
            risk_pct = risk_contributions / risk_contributions.sum()

            # 计算偏差
            deviation = risk_pct - target_risk_contributions

            # 加入对数障碍（如果启用）
            if self.config.use_log_barrier:
                barrier = -self.config.barrier_parameter * np.sum(
                    np.log(weights + 1e-10)
                )
                return np.sum(deviation**2) + barrier
            else:
                return np.sum(deviation**2)

        def gradient(weights):
            """梯度"""
            eps = 1e-8
            grad = np.zeros(n_assets)

            for i in range(n_assets):
                weights_plus = weights.copy()
                weights_plus[i] += eps

                weights_minus = weights.copy()
                weights_minus[i] -= eps

                grad[i] = (objective(weights_plus) - objective(weights_minus)) / (
                    2 * eps
                )

            return grad

        # 约束
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # 权重和为1
        ]

        # 边界
        if self.config.allow_short:
            bounds = [
                (-self.config.max_weight, self.config.max_weight)
                for _ in range(n_assets)
            ]
        else:
            bounds = [
                (self.config.min_weight, self.config.max_weight)
                for _ in range(n_assets)
            ]

        # 优化
        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": self.config.max_iterations,
                "ftol": self.config.tolerance,
            },
        )

        return result

    def _optimize_trust_region(
        self,
        covariance: np.ndarray,
        target_risk_contributions: np.ndarray,
        x0: np.ndarray,
    ) -> optimize.OptimizeResult:
        """使用信赖域方法优化

        Args:
            covariance: 协方差矩阵
            target_risk_contributions: 目标风险贡献
            x0: 初始权重

        Returns:
            优化结果
        """
        # 转换为无约束优化问题
        # 使用softmax变换确保权重和为1且非负

        def weights_from_z(z):
            """从无约束变量转换为权重"""
            exp_z = np.exp(z)
            return exp_z / exp_z.sum()

        def objective(z):
            """目标函数"""
            weights = weights_from_z(z)

            # 计算风险贡献
            portfolio_vol = np.sqrt(weights @ covariance @ weights)

            if portfolio_vol < 1e-10:
                return 1e10

            marginal_risk = (covariance @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk
            risk_pct = risk_contributions / risk_contributions.sum()

            # 计算偏差
            deviation = risk_pct - target_risk_contributions

            return np.sum(deviation**2)

        # 初始z值（逆softmax变换）
        z0 = np.log(x0 + 1e-10)

        # 优化
        result_z = optimize.minimize(
            objective,
            z0,
            method="trust-constr",
            options={
                "maxiter": self.config.max_iterations,
                "gtol": self.config.tolerance,
            },
        )

        # 转换回权重
        result = optimize.OptimizeResult()
        result.x = weights_from_z(result_z.x)
        result.fun = result_z.fun
        result.success = result_z.success
        result.nit = result_z.nit
        result.message = result_z.message

        return result

    def _calculate_risk_contributions(
        self, weights: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算风险贡献

        Args:
            weights: 权重
            covariance: 协方差矩阵

        Returns:
            (绝对风险贡献, 相对风险贡献)
        """
        portfolio_vol = np.sqrt(weights @ covariance @ weights)

        if portfolio_vol < 1e-10:
            return np.zeros_like(weights), np.zeros_like(weights)

        # 边际风险贡献
        marginal_risk = (covariance @ weights) / portfolio_vol

        # 绝对风险贡献
        abs_risk_contrib = weights * marginal_risk

        # 相对风险贡献
        total_contrib = abs_risk_contrib.sum()
        if total_contrib > 0:
            rel_risk_contrib = abs_risk_contrib / total_contrib
        else:
            rel_risk_contrib = np.zeros_like(abs_risk_contrib)

        return abs_risk_contrib, rel_risk_contrib


# 模块级别函数
def optimize_risk_budget(
    covariance_matrix: pd.DataFrame,
    risk_budgets: Dict[str, float],
    config: Optional[RiskBudgetConfig] = None,
) -> pd.Series:
    """优化风险预算的便捷函数

    Args:
        covariance_matrix: 协方差矩阵
        risk_budgets: 风险预算
        config: 配置

    Returns:
        最优权重Series
    """
    optimizer = RiskBudgetingOptimizer(config)
    result = optimizer.optimize_risk_budget(covariance_matrix, risk_budgets)
    return result.optimal_weights
