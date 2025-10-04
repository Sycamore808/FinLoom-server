"""
投资组合风险分析器
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("risk_analyzer")


@dataclass
class RiskConfig:
    """风险分析配置"""

    confidence_level: float = 0.95  # 置信水平
    time_horizon: int = 1  # 持有期（天）
    calculation_method: str = "historical"  # 'historical', 'parametric', 'monte_carlo'
    rolling_window: int = 252  # 滚动窗口
    annualization_factor: int = 252  # 年化因子
    enable_correlation_adjustment: bool = True
    risk_free_rate: float = 0.03  # 无风险利率


class PortfolioRiskAnalyzer:
    """投资组合风险综合评估工具"""

    def __init__(self, config: Optional[RiskConfig] = None):
        """初始化风险分析器

        Args:
            config: 风险分析配置
        """
        self.config = config or RiskConfig()
        logger.info(f"Initialized PortfolioRiskAnalyzer with config: {self.config}")

    def analyze_portfolio_risk(
        self, portfolio: Dict[str, Dict], returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """综合分析投资组合风险

        Args:
            portfolio: 投资组合字典 {symbol: {'weight': float, 'shares': int, 'cost': float}}
            returns: 收益率DataFrame

        Returns:
            风险指标字典
        """
        try:
            logger.info(f"Analyzing portfolio risk for {len(portfolio)} assets")

            # 计算投资组合收益率
            weights = np.array([pos["weight"] for pos in portfolio.values()])
            portfolio_returns = (returns * weights).sum(axis=1)

            # 计算各项风险指标
            var_95 = self.calculate_var(portfolio_returns, 0.95)
            var_99 = self.calculate_var(portfolio_returns, 0.99)
            cvar_95 = self.calculate_cvar(portfolio_returns, 0.95)
            cvar_99 = self.calculate_cvar(portfolio_returns, 0.99)

            # 计算最大回撤
            drawdown_metrics = self.calculate_max_drawdown_series(
                (1 + portfolio_returns).cumprod()
            )

            # 计算夏普比率
            sharpe = self.calculate_sharpe_ratio(portfolio_returns)

            # 计算索提诺比率
            sortino = self.calculate_sortino_ratio(portfolio_returns)

            # 计算波动率
            volatility = self.calculate_volatility(portfolio_returns)

            # 计算偏度和峰度
            skewness = portfolio_returns.skew()
            kurtosis = portfolio_returns.kurtosis()

            # 风险分解
            risk_decomposition = self.decompose_risk(portfolio, returns)

            risk_metrics = {
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95,
                "cvar_99": cvar_99,
                "max_drawdown": drawdown_metrics["max_drawdown"],
                "max_drawdown_duration": drawdown_metrics["max_duration"],
                "current_drawdown": drawdown_metrics["current_drawdown"],
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "volatility": volatility,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "marginal_var": risk_decomposition["marginal_var"],
                "component_var": risk_decomposition["component_var"],
                "correlation_risk": self._assess_correlation_risk(returns),
                "concentration_risk": self._assess_concentration_risk(portfolio),
            }

            logger.info(
                f"Portfolio risk analysis completed: VaR95={var_95:.4f}, Sharpe={sharpe:.2f}"
            )
            return risk_metrics

        except Exception as e:
            logger.error(f"Failed to analyze portfolio risk: {e}")
            raise QuantSystemError(f"Portfolio risk analysis failed: {e}")

    def calculate_var(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """计算风险价值(VaR)

        Args:
            returns: 收益率序列
            confidence_level: 置信水平

        Returns:
            VaR值
        """
        try:
            returns = returns.dropna()

            if len(returns) == 0:
                return 0.0

            if self.config.calculation_method == "historical":
                var = self._historical_var(returns, confidence_level)
            elif self.config.calculation_method == "parametric":
                var = self._parametric_var(returns, confidence_level)
            elif self.config.calculation_method == "monte_carlo":
                var = self._monte_carlo_var(returns, confidence_level)
            else:
                var = self._historical_var(returns, confidence_level)

            return var

        except Exception as e:
            logger.error(f"Failed to calculate VaR: {e}")
            return 0.0

    def _historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """历史模拟法计算VaR"""
        percentile = (1 - confidence_level) * 100
        return np.percentile(returns, percentile)

    def _parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """参数法计算VaR（假设正态分布）"""
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        return mean + z_score * std

    def _monte_carlo_var(
        self, returns: pd.Series, confidence_level: float, n_simulations: int = 10000
    ) -> float:
        """蒙特卡洛模拟法计算VaR"""
        mean = returns.mean()
        std = returns.std()

        # 生成随机收益率
        simulated_returns = np.random.normal(mean, std, n_simulations)

        percentile = (1 - confidence_level) * 100
        return np.percentile(simulated_returns, percentile)

    def calculate_cvar(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """计算条件风险价值(CVaR)

        Args:
            returns: 收益率序列
            confidence_level: 置信水平

        Returns:
            CVaR值
        """
        try:
            returns = returns.dropna()

            if len(returns) == 0:
                return 0.0

            var = self.calculate_var(returns, confidence_level)
            cvar = returns[returns <= var].mean()

            return cvar if not np.isnan(cvar) else var

        except Exception as e:
            logger.error(f"Failed to calculate CVaR: {e}")
            return 0.0

    def calculate_max_drawdown(self, portfolio_value: pd.Series) -> Dict[str, Any]:
        """计算最大回撤（简化版本，返回单个值）

        Args:
            portfolio_value: 投资组合价值序列

        Returns:
            包含最大回撤的字典
        """
        drawdown_metrics = self.calculate_max_drawdown_series(portfolio_value)
        return {
            "max_drawdown": drawdown_metrics["max_drawdown"],
            "start_date": drawdown_metrics["start_date"],
            "end_date": drawdown_metrics["end_date"],
        }

    def calculate_max_drawdown_series(
        self, portfolio_value: pd.Series
    ) -> Dict[str, Any]:
        """计算最大回撤（完整版本）

        Args:
            portfolio_value: 投资组合价值序列

        Returns:
            回撤指标字典
        """
        try:
            portfolio_value = portfolio_value.dropna()

            if len(portfolio_value) == 0:
                return {
                    "max_drawdown": 0.0,
                    "max_duration": 0,
                    "current_drawdown": 0.0,
                    "start_date": None,
                    "end_date": None,
                    "recovery_date": None,
                }

            # 计算累计最大值
            running_max = portfolio_value.expanding().max()

            # 计算回撤
            drawdown = (portfolio_value - running_max) / running_max

            # 最大回撤
            max_drawdown = drawdown.min()

            # 找到最大回撤的起止日期
            max_dd_idx = drawdown.idxmin()
            start_idx = portfolio_value[:max_dd_idx].idxmax()

            # 恢复日期（如果有）
            recovery_idx = None
            if max_dd_idx < len(portfolio_value) - 1:
                future_values = portfolio_value[max_dd_idx:]
                peak_value = portfolio_value[start_idx]
                recovered = future_values[future_values >= peak_value]
                if len(recovered) > 0:
                    recovery_idx = recovered.index[0]

            # 计算持续时间
            if recovery_idx:
                duration = (
                    (recovery_idx - start_idx).days if hasattr(start_idx, "days") else 0
                )
            else:
                duration = (
                    (portfolio_value.index[-1] - start_idx).days
                    if hasattr(start_idx, "days")
                    else 0
                )

            # 当前回撤
            current_drawdown = drawdown.iloc[-1]

            return {
                "max_drawdown": max_drawdown,
                "max_duration": duration,
                "current_drawdown": current_drawdown,
                "start_date": start_idx,
                "end_date": max_dd_idx,
                "recovery_date": recovery_idx,
            }

        except Exception as e:
            logger.error(f"Failed to calculate max drawdown: {e}")
            return {
                "max_drawdown": 0.0,
                "max_duration": 0,
                "current_drawdown": 0.0,
                "start_date": None,
                "end_date": None,
                "recovery_date": None,
            }

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: Optional[float] = None
    ) -> float:
        """计算夏普比率

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率（年化）

        Returns:
            夏普比率
        """
        try:
            returns = returns.dropna()

            if len(returns) < 2:
                return 0.0

            risk_free = (
                risk_free_rate
                if risk_free_rate is not None
                else self.config.risk_free_rate
            )

            # 年化收益率
            annual_return = returns.mean() * self.config.annualization_factor

            # 年化波动率
            annual_volatility = returns.std() * np.sqrt(
                self.config.annualization_factor
            )

            if annual_volatility == 0:
                return 0.0

            sharpe = (annual_return - risk_free) / annual_volatility

            return sharpe

        except Exception as e:
            logger.error(f"Failed to calculate Sharpe ratio: {e}")
            return 0.0

    def calculate_sortino_ratio(
        self, returns: pd.Series, risk_free_rate: Optional[float] = None
    ) -> float:
        """计算索提诺比率

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率

        Returns:
            索提诺比率
        """
        try:
            returns = returns.dropna()

            if len(returns) < 2:
                return 0.0

            risk_free = (
                risk_free_rate
                if risk_free_rate is not None
                else self.config.risk_free_rate
            )

            # 年化收益率
            annual_return = returns.mean() * self.config.annualization_factor

            # 下行波动率
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return float("inf") if annual_return > risk_free else 0.0

            downside_volatility = downside_returns.std() * np.sqrt(
                self.config.annualization_factor
            )

            if downside_volatility == 0:
                return 0.0

            sortino = (annual_return - risk_free) / downside_volatility

            return sortino

        except Exception as e:
            logger.error(f"Failed to calculate Sortino ratio: {e}")
            return 0.0

    def calculate_volatility(self, returns: pd.Series) -> float:
        """计算波动率

        Args:
            returns: 收益率序列

        Returns:
            年化波动率
        """
        try:
            returns = returns.dropna()

            if len(returns) < 2:
                return 0.0

            volatility = returns.std() * np.sqrt(self.config.annualization_factor)
            return volatility

        except Exception as e:
            logger.error(f"Failed to calculate volatility: {e}")
            return 0.0

    def decompose_risk(
        self, portfolio: Dict[str, Dict], returns: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """风险因子分解

        Args:
            portfolio: 投资组合
            returns: 收益率DataFrame

        Returns:
            风险分解结果
        """
        try:
            symbols = list(portfolio.keys())
            weights = np.array([portfolio[s]["weight"] for s in symbols])

            # 确保returns包含所有symbol
            returns_subset = returns[symbols].dropna()

            if len(returns_subset) == 0:
                return {
                    "marginal_var": {s: 0.0 for s in symbols},
                    "component_var": {s: 0.0 for s in symbols},
                    "incremental_var": {s: 0.0 for s in symbols},
                }

            # 计算协方差矩阵
            cov_matrix = returns_subset.cov() * self.config.annualization_factor

            # 投资组合方差
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)

            # 边际VaR
            marginal_var = {}
            for i, symbol in enumerate(symbols):
                marginal_contribution = np.dot(cov_matrix.iloc[i], weights)
                if portfolio_volatility > 0:
                    marginal_var[symbol] = marginal_contribution / portfolio_volatility
                else:
                    marginal_var[symbol] = 0.0

            # 成分VaR
            component_var = {}
            for symbol in symbols:
                component_var[symbol] = (
                    marginal_var[symbol] * portfolio[symbol]["weight"]
                )

            # 增量VaR（简化计算）
            incremental_var = {}
            for symbol in symbols:
                incremental_var[symbol] = (
                    marginal_var[symbol] * portfolio[symbol]["weight"] * 0.01
                )

            return {
                "marginal_var": marginal_var,
                "component_var": component_var,
                "incremental_var": incremental_var,
            }

        except Exception as e:
            logger.error(f"Failed to decompose risk: {e}")
            return {"marginal_var": {}, "component_var": {}, "incremental_var": {}}

    def _assess_correlation_risk(self, returns: pd.DataFrame) -> float:
        """评估相关性风险

        Args:
            returns: 收益率DataFrame

        Returns:
            相关性风险得分 (0-1)
        """
        try:
            if returns.shape[1] < 2:
                return 0.0

            # 计算相关性矩阵
            corr_matrix = returns.corr()

            # 去除对角线
            np.fill_diagonal(corr_matrix.values, 0)

            # 平均绝对相关性
            avg_corr = corr_matrix.abs().values.mean()

            # 最大相关性
            max_corr = corr_matrix.abs().values.max()

            # 相关性风险得分
            correlation_risk = avg_corr * 0.5 + max_corr * 0.5

            return correlation_risk

        except Exception as e:
            logger.error(f"Failed to assess correlation risk: {e}")
            return 0.0

    def _assess_concentration_risk(self, portfolio: Dict[str, Dict]) -> float:
        """评估集中度风险

        Args:
            portfolio: 投资组合

        Returns:
            集中度风险得分 (0-1)
        """
        try:
            weights = np.array([pos["weight"] for pos in portfolio.values()])

            # 赫芬达尔指数
            herfindahl_index = np.sum(weights**2)

            # 归一化到0-1
            n = len(weights)
            normalized_herfindahl = (
                (herfindahl_index - 1 / n) / (1 - 1 / n) if n > 1 else 0
            )

            return normalized_herfindahl

        except Exception as e:
            logger.error(f"Failed to assess concentration risk: {e}")
            return 0.0


# 便捷函数
def calculate_portfolio_var(
    portfolio: Dict[str, Dict], returns: pd.DataFrame, confidence_level: float = 0.95
) -> float:
    """快速计算投资组合VaR的便捷函数

    Args:
        portfolio: 投资组合
        returns: 收益率DataFrame
        confidence_level: 置信水平

    Returns:
        VaR值
    """
    analyzer = PortfolioRiskAnalyzer()
    weights = np.array([pos["weight"] for pos in portfolio.values()])
    portfolio_returns = (returns * weights).sum(axis=1)
    return analyzer.calculate_var(portfolio_returns, confidence_level)
