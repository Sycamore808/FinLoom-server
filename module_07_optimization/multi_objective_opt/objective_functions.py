"""
多目标优化的目标函数定义
提供常用的投资组合多目标函数
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from common.logging_system import setup_logger

logger = setup_logger("objective_functions")


class PortfolioObjectives:
    """投资组合多目标函数"""

    def __init__(self, returns_data: pd.DataFrame, risk_free_rate: float = 0.03):
        """初始化投资组合目标函数

        Args:
            returns_data: 收益率数据 (DataFrame, columns=symbols)
            risk_free_rate: 无风险利率
        """
        self.returns_data = returns_data
        self.risk_free_rate = risk_free_rate
        self.n_assets = returns_data.shape[1]

        # 预计算协方差矩阵
        self.cov_matrix = returns_data.cov() * 252
        self.mean_returns = returns_data.mean() * 252

    def return_objective(self, weights: np.ndarray) -> float:
        """收益率目标（最大化）

        Args:
            weights: 权重向量

        Returns:
            年化收益率（取负值用于最小化）
        """
        portfolio_return = np.dot(weights, self.mean_returns)
        return -portfolio_return  # 负值用于最小化框架

    def risk_objective(self, weights: np.ndarray) -> float:
        """风险目标（最小化波动率）

        Args:
            weights: 权重向量

        Returns:
            年化波动率
        """
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        return portfolio_std

    def sharpe_objective(self, weights: np.ndarray) -> float:
        """夏普比率目标（最大化）

        Args:
            weights: 权重向量

        Returns:
            负夏普比率（用于最小化）
        """
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_std = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))

        if portfolio_std == 0:
            return float("inf")

        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
        return -sharpe  # 负值用于最小化

    def max_drawdown_objective(self, weights: np.ndarray) -> float:
        """最大回撤目标（最小化）

        Args:
            weights: 权重向量

        Returns:
            最大回撤
        """
        # 计算投资组合收益序列
        portfolio_returns = (self.returns_data * weights).sum(axis=1)

        # 计算累计收益
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # 计算回撤
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max

        max_dd = abs(drawdown.min())
        return max_dd

    def concentration_objective(self, weights: np.ndarray) -> float:
        """集中度目标（最小化）

        使用Herfindahl指数衡量集中度

        Args:
            weights: 权重向量

        Returns:
            集中度指数
        """
        return np.sum(weights**2)

    def turnover_objective(
        self, weights: np.ndarray, previous_weights: np.ndarray
    ) -> float:
        """换手率目标（最小化）

        Args:
            weights: 当前权重
            previous_weights: 上期权重

        Returns:
            换手率
        """
        return np.sum(np.abs(weights - previous_weights))


def create_portfolio_objectives(
    returns_data: pd.DataFrame,
    objective_names: List[str],
    risk_free_rate: float = 0.03,
    previous_weights: np.ndarray = None,
) -> List:
    """创建投资组合目标函数列表

    Args:
        returns_data: 收益率数据
        objective_names: 目标名称列表
        risk_free_rate: 无风险利率
        previous_weights: 上期权重

    Returns:
        目标函数列表
    """
    portfolio_obj = PortfolioObjectives(returns_data, risk_free_rate)
    objective_functions = []

    for obj_name in objective_names:
        if obj_name == "return":
            objective_functions.append(portfolio_obj.return_objective)
        elif obj_name == "risk":
            objective_functions.append(portfolio_obj.risk_objective)
        elif obj_name == "sharpe":
            objective_functions.append(portfolio_obj.sharpe_objective)
        elif obj_name == "drawdown":
            objective_functions.append(portfolio_obj.max_drawdown_objective)
        elif obj_name == "concentration":
            objective_functions.append(portfolio_obj.concentration_objective)
        elif obj_name == "turnover" and previous_weights is not None:

            def turnover_func(weights):
                return portfolio_obj.turnover_objective(weights, previous_weights)

            objective_functions.append(turnover_func)
        else:
            logger.warning(f"Unknown objective: {obj_name}")

    return objective_functions


def weights_to_params(weights: np.ndarray, symbol_names: List[str]) -> Dict[str, float]:
    """将权重向量转换为参数字典

    Args:
        weights: 权重向量
        symbol_names: 股票代码列表

    Returns:
        参数字典
    """
    return {f"weight_{symbol}": w for symbol, w in zip(symbol_names, weights)}


def params_to_weights(params: Dict[str, Any], symbol_names: List[str]) -> np.ndarray:
    """将参数字典转换为权重向量

    Args:
        params: 参数字典
        symbol_names: 股票代码列表

    Returns:
        权重向量
    """
    weights = np.array([params.get(f"weight_{symbol}", 0) for symbol in symbol_names])

    # 归一化权重
    total_weight = weights.sum()
    if total_weight > 0:
        weights = weights / total_weight

    return weights
