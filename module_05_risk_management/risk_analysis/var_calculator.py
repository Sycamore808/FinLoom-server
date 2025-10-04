"""
VaR和CVaR计算器
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("var_calculator")


@dataclass
class VaRConfig:
    """VaR计算配置"""

    method: str = "historical"  # 'historical', 'parametric', 'monte_carlo'
    confidence_level: float = 0.95
    time_horizon: int = 1
    n_simulations: int = 10000  # 蒙特卡洛模拟次数


class VaRCalculator:
    """专业的风险价值计算工具"""

    def __init__(self, method: str = "historical", confidence_level: float = 0.95):
        """初始化VaR计算器

        Args:
            method: 计算方法 ('historical', 'parametric', 'monte_carlo')
            confidence_level: 置信水平
        """
        self.method = method
        self.confidence_level = confidence_level
        logger.info(
            f"Initialized VaRCalculator with method={method}, confidence={confidence_level}"
        )

    def historical_var(self, returns: pd.Series) -> float:
        """历史模拟法计算VaR

        Args:
            returns: 收益率序列

        Returns:
            VaR值
        """
        try:
            returns = returns.dropna()

            if len(returns) == 0:
                logger.warning("Empty returns series for historical VaR")
                return 0.0

            percentile = (1 - self.confidence_level) * 100
            var = np.percentile(returns, percentile)

            logger.debug(f"Historical VaR ({self.confidence_level * 100}%): {var:.4f}")
            return var

        except Exception as e:
            logger.error(f"Failed to calculate historical VaR: {e}")
            raise QuantSystemError(f"Historical VaR calculation failed: {e}")

    def parametric_var(self, returns: pd.Series) -> float:
        """参数法计算VaR（假设正态分布）

        Args:
            returns: 收益率序列

        Returns:
            VaR值
        """
        try:
            returns = returns.dropna()

            if len(returns) < 2:
                logger.warning("Insufficient data for parametric VaR")
                return 0.0

            mean = returns.mean()
            std = returns.std()

            # 正态分布的分位数
            z_score = stats.norm.ppf(1 - self.confidence_level)
            var = mean + z_score * std

            logger.debug(f"Parametric VaR ({self.confidence_level * 100}%): {var:.4f}")
            return var

        except Exception as e:
            logger.error(f"Failed to calculate parametric VaR: {e}")
            raise QuantSystemError(f"Parametric VaR calculation failed: {e}")

    def monte_carlo_var(self, returns: pd.Series, n_simulations: int = 10000) -> float:
        """蒙特卡洛模拟法计算VaR

        Args:
            returns: 收益率序列
            n_simulations: 模拟次数

        Returns:
            VaR值
        """
        try:
            returns = returns.dropna()

            if len(returns) < 2:
                logger.warning("Insufficient data for Monte Carlo VaR")
                return 0.0

            mean = returns.mean()
            std = returns.std()

            # 生成随机收益率
            simulated_returns = np.random.normal(mean, std, n_simulations)

            # 计算VaR
            percentile = (1 - self.confidence_level) * 100
            var = np.percentile(simulated_returns, percentile)

            logger.debug(
                f"Monte Carlo VaR ({self.confidence_level * 100}%, {n_simulations} sims): {var:.4f}"
            )
            return var

        except Exception as e:
            logger.error(f"Failed to calculate Monte Carlo VaR: {e}")
            raise QuantSystemError(f"Monte Carlo VaR calculation failed: {e}")

    def conditional_var(self, returns: pd.Series) -> float:
        """计算CVaR（条件VaR）

        Args:
            returns: 收益率序列

        Returns:
            CVaR值
        """
        try:
            returns = returns.dropna()

            if len(returns) == 0:
                logger.warning("Empty returns series for CVaR")
                return 0.0

            # 先计算VaR
            if self.method == "historical":
                var = self.historical_var(returns)
            elif self.method == "parametric":
                var = self.parametric_var(returns)
            elif self.method == "monte_carlo":
                var = self.monte_carlo_var(returns)
            else:
                var = self.historical_var(returns)

            # 计算CVaR（超过VaR的平均损失）
            tail_losses = returns[returns <= var]

            if len(tail_losses) == 0:
                cvar = var
            else:
                cvar = tail_losses.mean()

            logger.debug(f"CVaR ({self.confidence_level * 100}%): {cvar:.4f}")
            return cvar

        except Exception as e:
            logger.error(f"Failed to calculate CVaR: {e}")
            raise QuantSystemError(f"CVaR calculation failed: {e}")

    def calculate(self, returns: pd.Series) -> Dict[str, float]:
        """计算VaR和CVaR

        Args:
            returns: 收益率序列

        Returns:
            包含VaR和CVaR的字典
        """
        try:
            if self.method == "historical":
                var = self.historical_var(returns)
            elif self.method == "parametric":
                var = self.parametric_var(returns)
            elif self.method == "monte_carlo":
                var = self.monte_carlo_var(returns)
            else:
                var = self.historical_var(returns)

            cvar = self.conditional_var(returns)

            return {
                "var": var,
                "cvar": cvar,
                "method": self.method,
                "confidence_level": self.confidence_level,
            }

        except Exception as e:
            logger.error(f"Failed to calculate VaR metrics: {e}")
            raise QuantSystemError(f"VaR calculation failed: {e}")

    def calculate_portfolio_var(
        self, returns: pd.DataFrame, weights: np.ndarray
    ) -> Dict[str, float]:
        """计算投资组合VaR

        Args:
            returns: 收益率DataFrame（每列是一个资产）
            weights: 权重数组

        Returns:
            包含投资组合VaR和CVaR的字典
        """
        try:
            # 计算投资组合收益率
            portfolio_returns = (returns * weights).sum(axis=1)

            return self.calculate(portfolio_returns)

        except Exception as e:
            logger.error(f"Failed to calculate portfolio VaR: {e}")
            raise QuantSystemError(f"Portfolio VaR calculation failed: {e}")

    def backtest_var(
        self, returns: pd.Series, var_estimates: pd.Series
    ) -> Dict[str, Any]:
        """回测VaR模型

        Args:
            returns: 实际收益率
            var_estimates: VaR估计值

        Returns:
            回测结果
        """
        try:
            # 对齐数据
            common_index = returns.index.intersection(var_estimates.index)
            returns_aligned = returns.loc[common_index]
            var_aligned = var_estimates.loc[common_index]

            # 计算违规次数
            violations = (returns_aligned < var_aligned).sum()
            total_days = len(returns_aligned)
            violation_rate = violations / total_days if total_days > 0 else 0

            # 理论违规率
            theoretical_rate = 1 - self.confidence_level

            # Kupiec检验
            if violations > 0 and total_days > 0:
                likelihood_ratio = -2 * (
                    np.log(
                        (1 - theoretical_rate) ** (total_days - violations)
                        * theoretical_rate**violations
                    )
                    - np.log(
                        (1 - violation_rate) ** (total_days - violations)
                        * violation_rate**violations
                    )
                )
                # 卡方检验（自由度为1）
                p_value = 1 - stats.chi2.cdf(likelihood_ratio, 1)
            else:
                likelihood_ratio = 0.0
                p_value = 1.0

            result = {
                "violations": int(violations),
                "total_days": int(total_days),
                "violation_rate": float(violation_rate),
                "theoretical_rate": float(theoretical_rate),
                "kupiec_lr": float(likelihood_ratio),
                "kupiec_p_value": float(p_value),
                "pass_test": p_value > 0.05,
            }

            logger.info(
                f"VaR backtest: {violations}/{total_days} violations ({violation_rate:.2%})"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to backtest VaR: {e}")
            raise QuantSystemError(f"VaR backtest failed: {e}")

    def calculate_marginal_var(
        self, returns: pd.DataFrame, weights: np.ndarray, asset_index: int
    ) -> float:
        """计算边际VaR

        Args:
            returns: 收益率DataFrame
            weights: 当前权重
            asset_index: 资产索引

        Returns:
            边际VaR
        """
        try:
            # 原始投资组合VaR
            original_var = self.calculate_portfolio_var(returns, weights)["var"]

            # 小幅增加该资产权重
            delta = 0.01
            new_weights = weights.copy()
            new_weights[asset_index] += delta

            # 重新归一化
            new_weights = new_weights / new_weights.sum()

            # 新投资组合VaR
            new_var = self.calculate_portfolio_var(returns, new_weights)["var"]

            # 边际VaR
            marginal_var = (new_var - original_var) / delta

            return marginal_var

        except Exception as e:
            logger.error(f"Failed to calculate marginal VaR: {e}")
            return 0.0
