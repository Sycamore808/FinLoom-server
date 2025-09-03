"""
绩效分析器模块
提供详细的策略绩效分析和评估功能
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from common.constants import TRADING_DAYS_PER_YEAR
from common.logging_system import setup_logger

logger = setup_logger("performance_analyzer")


@dataclass
class PerformanceReport:
    """绩效报告数据类"""

    summary_stats: Dict[str, float]
    period_returns: Dict[str, pd.Series]
    rolling_metrics: pd.DataFrame
    benchmark_comparison: Optional[Dict[str, float]]
    factor_exposures: Optional[Dict[str, float]]
    regime_analysis: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "summary_stats": self.summary_stats,
            "period_returns": {k: v.to_dict() for k, v in self.period_returns.items()},
            "rolling_metrics": self.rolling_metrics.to_dict(),
            "benchmark_comparison": self.benchmark_comparison,
            "factor_exposures": self.factor_exposures,
            "regime_analysis": self.regime_analysis,
        }


class PerformanceAnalyzer:
    """绩效分析器类"""

    def __init__(self):
        """初始化绩效分析器"""
        self.returns_data: Optional[pd.Series] = None
        self.benchmark_returns: Optional[pd.Series] = None
        self.factor_returns: Optional[pd.DataFrame] = None

    def analyze(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None,
    ) -> PerformanceReport:
        """执行完整的绩效分析

        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            factor_returns: 因子收益率DataFrame

        Returns:
            绩效报告对象
        """
        self.returns_data = returns
        self.benchmark_returns = benchmark_returns
        self.factor_returns = factor_returns

        # 计算汇总统计
        summary_stats = self._calculate_summary_statistics()

        # 计算期间收益
        period_returns = self._calculate_period_returns()

        # 计算滚动指标
        rolling_metrics = self._calculate_rolling_metrics()

        # 基准比较
        benchmark_comparison = None
        if benchmark_returns is not None:
            benchmark_comparison = self._compare_with_benchmark()

        # 因子暴露分析
        factor_exposures = None
        if factor_returns is not None:
            factor_exposures = self._analyze_factor_exposures()

        # 市场状态分析
        regime_analysis = self._analyze_market_regimes()

        return PerformanceReport(
            summary_stats=summary_stats,
            period_returns=period_returns,
            rolling_metrics=rolling_metrics,
            benchmark_comparison=benchmark_comparison,
            factor_exposures=factor_exposures,
            regime_analysis=regime_analysis,
        )

    def _calculate_summary_statistics(self) -> Dict[str, float]:
        """计算汇总统计指标

        Returns:
            统计指标字典
        """
        returns = self.returns_data

        # 基本统计
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1

        # 风险指标
        volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        downside_vol = returns[returns < 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # 风险调整收益
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns)

        # 回撤分析
        max_drawdown, drawdown_duration = self._calculate_max_drawdown(returns)

        # 高阶矩
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # 尾部风险
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        # 稳定性指标
        information_ratio = self._calculate_information_ratio(returns)

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "downside_volatility": downside_vol,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "drawdown_duration_days": drawdown_duration,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "information_ratio": information_ratio,
            "best_day": returns.max(),
            "worst_day": returns.min(),
            "positive_days": (returns > 0).sum(),
            "negative_days": (returns < 0).sum(),
            "hit_rate": (returns > 0).mean(),
        }

    def _calculate_period_returns(self) -> Dict[str, pd.Series]:
        """计算不同时期的收益

        Returns:
            期间收益字典
        """
        returns = self.returns_data

        period_returns = {
            "daily": returns,
            "weekly": returns.resample("W").apply(lambda x: (1 + x).prod() - 1),
            "monthly": returns.resample("M").apply(lambda x: (1 + x).prod() - 1),
            "quarterly": returns.resample("Q").apply(lambda x: (1 + x).prod() - 1),
            "yearly": returns.resample("Y").apply(lambda x: (1 + x).prod() - 1),
        }

        return period_returns

    def _calculate_rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        """计算滚动指标

        Args:
            window: 滚动窗口大小

        Returns:
            滚动指标DataFrame
        """
        returns = self.returns_data

        rolling_metrics = pd.DataFrame(index=returns.index)

        # 滚动收益
        rolling_metrics["rolling_return"] = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        )

        # 滚动波动率
        rolling_metrics["rolling_volatility"] = returns.rolling(window).std() * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )

        # 滚动夏普比率
        rolling_metrics["rolling_sharpe"] = returns.rolling(window).apply(
            lambda x: self._calculate_sharpe_ratio(x)
        )

        # 滚动最大回撤
        rolling_metrics["rolling_max_drawdown"] = returns.rolling(window).apply(
            lambda x: self._calculate_max_drawdown(x)[0]
        )

        # 滚动相关性（如果有基准）
        if self.benchmark_returns is not None:
            rolling_metrics["rolling_correlation"] = returns.rolling(window).corr(
                self.benchmark_returns
            )

            # 滚动贝塔
            rolling_metrics["rolling_beta"] = returns.rolling(window).apply(
                lambda x: self._calculate_beta(x, self.benchmark_returns.loc[x.index])
            )

        return rolling_metrics

    def _compare_with_benchmark(self) -> Dict[str, float]:
        """与基准进行比较

        Returns:
            比较结果字典
        """
        returns = self.returns_data
        benchmark = self.benchmark_returns

        # 对齐数据
        aligned_returns, aligned_benchmark = returns.align(benchmark, join="inner")

        # 超额收益
        excess_returns = aligned_returns - aligned_benchmark

        # 计算指标
        total_excess = (1 + excess_returns).prod() - 1
        tracking_error = excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        information_ratio = (
            excess_returns.mean()
            / excess_returns.std()
            * np.sqrt(TRADING_DAYS_PER_YEAR)
            if excess_returns.std() > 0
            else 0
        )

        # 相关性和贝塔
        correlation = aligned_returns.corr(aligned_benchmark)
        beta = self._calculate_beta(aligned_returns, aligned_benchmark)
        alpha = self._calculate_alpha(aligned_returns, aligned_benchmark)

        # 上下行捕获率
        up_capture, down_capture = self._calculate_capture_ratios(
            aligned_returns, aligned_benchmark
        )

        return {
            "total_excess_return": total_excess,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "correlation": correlation,
            "beta": beta,
            "alpha": alpha,
            "up_capture_ratio": up_capture,
            "down_capture_ratio": down_capture,
            "win_rate_vs_benchmark": (excess_returns > 0).mean(),
        }

    def _analyze_factor_exposures(self) -> Dict[str, float]:
        """分析因子暴露

        Returns:
            因子暴露字典
        """
        returns = self.returns_data
        factors = self.factor_returns

        # 对齐数据
        aligned_returns, aligned_factors = returns.align(factors, join="inner", axis=0)

        # 多因子回归
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(aligned_factors.values, aligned_returns.values)

        # 因子载荷
        factor_loadings = dict(zip(factors.columns, model.coef_))

        # 因子贡献
        factor_contributions = {}
        for factor in factors.columns:
            factor_return = aligned_factors[factor].mean() * TRADING_DAYS_PER_YEAR
            factor_contributions[f"{factor}_contribution"] = (
                factor_loadings[factor] * factor_return
            )

        # R-squared
        r_squared = model.score(aligned_factors.values, aligned_returns.values)

        result = {"r_squared": r_squared, "intercept": model.intercept_}
        result.update(factor_loadings)
        result.update(factor_contributions)

        return result

    def _analyze_market_regimes(self) -> Dict[str, Any]:
        """分析市场状态下的表现

        Returns:
            市场状态分析结果
        """
        returns = self.returns_data

        # 定义市场状态
        # 使用滚动波动率划分高低波动期
        rolling_vol = returns.rolling(20).std()
        median_vol = rolling_vol.median()

        high_vol_periods = rolling_vol > median_vol
        low_vol_periods = ~high_vol_periods

        # 使用累计收益划分牛熊市
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.rolling(252, min_periods=1).max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max

        bear_market = drawdown < -0.2  # 回撤超过20%定义为熊市
        bull_market = ~bear_market

        # 计算不同状态下的表现
        regime_performance = {
            "high_volatility": {
                "return": returns[high_vol_periods].mean() * TRADING_DAYS_PER_YEAR,
                "volatility": returns[high_vol_periods].std()
                * np.sqrt(TRADING_DAYS_PER_YEAR),
                "sharpe": self._calculate_sharpe_ratio(returns[high_vol_periods]),
                "periods": high_vol_periods.sum(),
            },
            "low_volatility": {
                "return": returns[low_vol_periods].mean() * TRADING_DAYS_PER_YEAR,
                "volatility": returns[low_vol_periods].std()
                * np.sqrt(TRADING_DAYS_PER_YEAR),
                "sharpe": self._calculate_sharpe_ratio(returns[low_vol_periods]),
                "periods": low_vol_periods.sum(),
            },
            "bull_market": {
                "return": returns[bull_market].mean() * TRADING_DAYS_PER_YEAR,
                "volatility": returns[bull_market].std()
                * np.sqrt(TRADING_DAYS_PER_YEAR),
                "sharpe": self._calculate_sharpe_ratio(returns[bull_market]),
                "periods": bull_market.sum(),
            },
            "bear_market": {
                "return": returns[bear_market].mean() * TRADING_DAYS_PER_YEAR,
                "volatility": returns[bear_market].std()
                * np.sqrt(TRADING_DAYS_PER_YEAR),
                "sharpe": self._calculate_sharpe_ratio(returns[bear_market])
                if bear_market.sum() > 0
                else 0,
                "periods": bear_market.sum(),
            },
        }

        return regime_performance

    def _calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """计算夏普比率

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率

        Returns:
            夏普比率
        """
        excess_returns = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
        if excess_returns.std() == 0:
            return 0
        return (
            np.sqrt(TRADING_DAYS_PER_YEAR)
            * excess_returns.mean()
            / excess_returns.std()
        )

    def _calculate_sortino_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """计算索提诺比率

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率

        Returns:
            索提诺比率
        """
        excess_returns = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return 0

        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0

        return np.sqrt(TRADING_DAYS_PER_YEAR) * excess_returns.mean() / downside_std

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """计算卡尔玛比率

        Args:
            returns: 收益率序列

        Returns:
            卡尔玛比率
        """
        annual_return = returns.mean() * TRADING_DAYS_PER_YEAR
        max_drawdown, _ = self._calculate_max_drawdown(returns)

        if max_drawdown == 0:
            return 0

        return annual_return / abs(max_drawdown)

    def _calculate_max_drawdown(self, returns: pd.Series) -> Tuple[float, int]:
        """计算最大回撤

        Args:
            returns: 收益率序列

        Returns:
            (最大回撤, 持续天数)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        max_drawdown = drawdown.min()

        # 计算最大回撤持续时间
        duration = 0
        current_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                duration = max(duration, current_duration)
            else:
                current_duration = 0

        return max_drawdown, duration

    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """计算信息比率

        Args:
            returns: 收益率序列

        Returns:
            信息比率
        """
        if self.benchmark_returns is None:
            return 0

        active_returns = returns - self.benchmark_returns
        if active_returns.std() == 0:
            return 0

        return (
            active_returns.mean()
            / active_returns.std()
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

    def _calculate_beta(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """计算贝塔系数

        Args:
            returns: 策略收益率
            benchmark_returns: 基准收益率

        Returns:
            贝塔系数
        """
        covariance = returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()

        if benchmark_variance == 0:
            return 0

        return covariance / benchmark_variance

    def _calculate_alpha(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02,
    ) -> float:
        """计算阿尔法

        Args:
            returns: 策略收益率
            benchmark_returns: 基准收益率
            risk_free_rate: 无风险利率

        Returns:
            年化阿尔法
        """
        beta = self._calculate_beta(returns, benchmark_returns)

        strategy_return = returns.mean() * TRADING_DAYS_PER_YEAR
        benchmark_return = benchmark_returns.mean() * TRADING_DAYS_PER_YEAR

        alpha = strategy_return - (
            risk_free_rate + beta * (benchmark_return - risk_free_rate)
        )

        return alpha

    def _calculate_capture_ratios(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """计算捕获率

        Args:
            returns: 策略收益率
            benchmark_returns: 基准收益率

        Returns:
            (上行捕获率, 下行捕获率)
        """
        # 上行捕获率
        up_market = benchmark_returns > 0
        if up_market.sum() > 0:
            up_capture = (1 + returns[up_market]).prod() / (
                1 + benchmark_returns[up_market]
            ).prod()
        else:
            up_capture = 0

        # 下行捕获率
        down_market = benchmark_returns < 0
        if down_market.sum() > 0:
            down_capture = (1 + returns[down_market]).prod() / (
                1 + benchmark_returns[down_market]
            ).prod()
        else:
            down_capture = 0

        return up_capture, down_capture


# 模块级别函数
def analyze_performance(
    returns: pd.Series, benchmark_returns: Optional[pd.Series] = None
) -> PerformanceReport:
    """分析绩效的便捷函数

    Args:
        returns: 收益率序列
        benchmark_returns: 基准收益率序列

    Returns:
        绩效报告
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.analyze(returns, benchmark_returns)


def compare_strategies(strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
    """比较多个策略的绩效

    Args:
        strategy_returns: 策略名称到收益率序列的映射

    Returns:
        比较结果DataFrame
    """
    results = []

    for name, returns in strategy_returns.items():
        analyzer = PerformanceAnalyzer()
        report = analyzer.analyze(returns)

        result = {"strategy": name}
        result.update(report.summary_stats)
        results.append(result)

    return pd.DataFrame(results)
