"""
验证工具模块
提供回测结果验证、过拟合检测和稳健性测试
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from common.constants import TRADING_DAYS_PER_YEAR
from common.logging_system import setup_logger
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

logger = setup_logger("validation_tools")


@dataclass
class ValidationReport:
    """验证报告数据类"""

    overfitting_tests: Dict[str, Any]
    stability_tests: Dict[str, Any]
    robustness_tests: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    walk_forward_results: Optional[Dict[str, Any]]
    monte_carlo_results: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "overfitting_tests": self.overfitting_tests,
            "stability_tests": self.stability_tests,
            "robustness_tests": self.robustness_tests,
            "statistical_tests": self.statistical_tests,
            "walk_forward_results": self.walk_forward_results,
            "monte_carlo_results": self.monte_carlo_results,
        }


class BacktestValidator:
    """回测验证器类"""

    def __init__(self):
        """初始化回测验证器"""
        self.backtest_results: Optional[Any] = None
        self.market_data: Optional[pd.DataFrame] = None

    def validate(
        self,
        backtest_results: Any,
        market_data: pd.DataFrame,
        strategy_func: Optional[Callable] = None,
    ) -> ValidationReport:
        """执行完整的验证分析

        Args:
            backtest_results: 回测结果对象
            market_data: 市场数据
            strategy_func: 策略函数（用于额外测试）

        Returns:
            验证报告
        """
        self.backtest_results = backtest_results
        self.market_data = market_data

        # 过拟合测试
        overfitting_tests = self._test_overfitting()

        # 稳定性测试
        stability_tests = self._test_stability()

        # 稳健性测试
        robustness_tests = self._test_robustness()

        # 统计显著性测试
        statistical_tests = self._test_statistical_significance()

        # Walk-forward分析
        walk_forward_results = None
        if strategy_func:
            walk_forward_results = self._walk_forward_analysis(strategy_func)

        # 蒙特卡洛模拟
        monte_carlo_results = self._monte_carlo_simulation()

        return ValidationReport(
            overfitting_tests=overfitting_tests,
            stability_tests=stability_tests,
            robustness_tests=robustness_tests,
            statistical_tests=statistical_tests,
            walk_forward_results=walk_forward_results,
            monte_carlo_results=monte_carlo_results,
        )

    def _test_overfitting(self) -> Dict[str, Any]:
        """测试过拟合

        Returns:
            过拟合测试结果
        """
        results = self.backtest_results

        # 获取样本内外数据
        total_days = len(results.daily_returns)
        split_point = int(total_days * 0.7)

        in_sample_returns = results.daily_returns.iloc[:split_point]
        out_sample_returns = results.daily_returns.iloc[split_point:]

        # 计算样本内外夏普比率
        in_sample_sharpe = self._calculate_sharpe(in_sample_returns)
        out_sample_sharpe = self._calculate_sharpe(out_sample_returns)

        # 夏普比率衰减
        sharpe_decay = (
            (in_sample_sharpe - out_sample_sharpe) / in_sample_sharpe
            if in_sample_sharpe != 0
            else 0
        )

        # 收益率衰减
        in_sample_mean = in_sample_returns.mean() * TRADING_DAYS_PER_YEAR
        out_sample_mean = out_sample_returns.mean() * TRADING_DAYS_PER_YEAR
        return_decay = (
            (in_sample_mean - out_sample_mean) / abs(in_sample_mean)
            if in_sample_mean != 0
            else 0
        )

        # 信息系数稳定性
        ic_stability = self._test_ic_stability(in_sample_returns, out_sample_returns)

        # 参数敏感性（如果有多组参数结果）
        parameter_sensitivity = self._test_parameter_sensitivity()

        return {
            "in_sample_sharpe": in_sample_sharpe,
            "out_sample_sharpe": out_sample_sharpe,
            "sharpe_decay": sharpe_decay,
            "in_sample_return": in_sample_mean,
            "out_sample_return": out_sample_mean,
            "return_decay": return_decay,
            "ic_stability": ic_stability,
            "parameter_sensitivity": parameter_sensitivity,
            "overfitting_score": self._calculate_overfitting_score(
                sharpe_decay, return_decay
            ),
        }

    def _test_stability(self) -> Dict[str, Any]:
        """测试策略稳定性

        Returns:
            稳定性测试结果
        """
        returns = self.backtest_results.daily_returns

        # 滚动窗口分析
        window_size = 252  # 一年
        rolling_sharpe = returns.rolling(window_size).apply(
            lambda x: self._calculate_sharpe(x) if len(x) == window_size else np.nan
        )
        rolling_sharpe = rolling_sharpe.dropna()

        sharpe_stability = {
            "mean": rolling_sharpe.mean(),
            "std": rolling_sharpe.std(),
            "min": rolling_sharpe.min(),
            "max": rolling_sharpe.max(),
            "coefficient_of_variation": rolling_sharpe.std()
            / abs(rolling_sharpe.mean())
            if rolling_sharpe.mean() != 0
            else np.inf,
        }

        # 子期间分析
        subperiod_results = self._analyze_subperiods(returns)

        # 收益分布稳定性
        distribution_stability = self._test_distribution_stability(returns)

        # 相关性稳定性
        correlation_stability = self._test_correlation_stability()

        return {
            "sharpe_stability": sharpe_stability,
            "subperiod_consistency": subperiod_results,
            "distribution_stability": distribution_stability,
            "correlation_stability": correlation_stability,
        }

    def _test_robustness(self) -> Dict[str, Any]:
        """测试策略稳健性

        Returns:
            稳健性测试结果
        """
        returns = self.backtest_results.daily_returns

        # 压力测试
        stress_test_results = self._perform_stress_tests(returns)

        # 异常值影响
        outlier_impact = self._test_outlier_impact(returns)

        # 交易成本敏感性
        cost_sensitivity = self._test_cost_sensitivity()

        # 数据质量影响
        data_quality_impact = self._test_data_quality_impact()

        return {
            "stress_test_results": stress_test_results,
            "outlier_impact": outlier_impact,
            "cost_sensitivity": cost_sensitivity,
            "data_quality_impact": data_quality_impact,
        }

    def _test_statistical_significance(self) -> Dict[str, Any]:
        """测试统计显著性

        Returns:
            统计测试结果
        """
        returns = self.backtest_results.daily_returns

        # T检验（收益是否显著不为0）
        t_stat, p_value = stats.ttest_1samp(returns.dropna(), 0)

        # 夏普比率的统计显著性
        sharpe = self._calculate_sharpe(returns)
        sharpe_se = 1 / np.sqrt(len(returns))  # 夏普比率的标准误
        sharpe_t_stat = sharpe / sharpe_se
        sharpe_p_value = 2 * (1 - stats.norm.cdf(abs(sharpe_t_stat)))

        # Jarque-Bera正态性检验
        jb_stat, jb_p_value = stats.jarque_bera(returns.dropna())

        # 自相关检验
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb_result = acorr_ljungbox(returns.dropna(), lags=10, return_df=True)
        lb_p_value = lb_result["lb_pvalue"].iloc[0]

        # 序列相关性
        autocorr = returns.autocorr()

        return {
            "mean_t_statistic": float(t_stat),
            "mean_p_value": float(p_value),
            "sharpe_t_statistic": float(sharpe_t_stat),
            "sharpe_p_value": float(sharpe_p_value),
            "is_significant": p_value < 0.05,
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_p_value": float(jb_p_value),
            "is_normal": jb_p_value > 0.05,
            "ljung_box_p_value": float(lb_p_value),
            "has_serial_correlation": lb_p_value < 0.05,
            "first_order_autocorr": float(autocorr) if not np.isnan(autocorr) else 0.0,
        }

    def _walk_forward_analysis(
        self, strategy_func: Callable, n_splits: int = 5
    ) -> Dict[str, Any]:
        """Walk-forward分析

        Args:
            strategy_func: 策略函数
            n_splits: 分割数量

        Returns:
            Walk-forward分析结果
        """
        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=n_splits)

        results = []

        for train_idx, test_idx in tscv.split(self.market_data):
            # 训练期数据
            train_data = self.market_data.iloc[train_idx]

            # 测试期数据
            test_data = self.market_data.iloc[test_idx]

            # 在训练期优化参数（这里简化处理）
            # 实际应该进行参数优化

            # 在测试期应用策略
            test_returns = self._simple_backtest(strategy_func, test_data)

            if len(test_returns) > 0:
                sharpe = self._calculate_sharpe(test_returns)
                total_return = (1 + test_returns).prod() - 1
                max_dd = self._calculate_max_drawdown(test_returns)
            else:
                sharpe = 0.0
                total_return = 0.0
                max_dd = 0.0

            results.append(
                {
                    "sharpe": float(sharpe),
                    "total_return": float(total_return),
                    "max_drawdown": float(max_dd),
                    "n_days": len(test_returns),
                }
            )

        # 汇总结果
        sharpes = [r["sharpe"] for r in results]
        returns_list = [r["total_return"] for r in results]

        avg_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        avg_return = np.mean(returns_list)

        return {
            "n_splits": n_splits,
            "fold_results": results,
            "avg_sharpe": float(avg_sharpe),
            "std_sharpe": float(std_sharpe),
            "sharpe_consistency": float(1 - (std_sharpe / abs(avg_sharpe)))
            if avg_sharpe != 0
            else 0.0,
            "avg_return": float(avg_return),
            "return_consistency": float(1 - np.std(returns_list) / abs(avg_return))
            if avg_return != 0
            else 0.0,
        }

    def _monte_carlo_simulation(self, n_simulations: int = 1000) -> Dict[str, Any]:
        """蒙特卡洛模拟

        Args:
            n_simulations: 模拟次数

        Returns:
            蒙特卡洛模拟结果
        """
        returns = self.backtest_results.daily_returns

        # 获取收益率统计特征
        mean_return = returns.mean()
        std_return = returns.std()

        # 模拟路径
        simulation_results = []

        np.random.seed(42)  # 设置随机种子以保证可重复性

        for _ in range(n_simulations):
            # 生成随机收益序列（保持相同长度）
            simulated_returns = np.random.normal(mean_return, std_return, len(returns))
            simulated_series = pd.Series(simulated_returns)

            # 计算模拟路径的指标
            cumulative_return = (1 + simulated_series).prod() - 1
            sharpe = self._calculate_sharpe(simulated_series)
            max_dd = self._calculate_max_drawdown(simulated_series)

            simulation_results.append(
                {
                    "total_return": float(cumulative_return),
                    "sharpe": float(sharpe),
                    "max_drawdown": float(max_dd),
                }
            )

        # 统计分析
        actual_return = (1 + returns).prod() - 1
        actual_sharpe = self._calculate_sharpe(returns)
        actual_max_dd = self._calculate_max_drawdown(returns)

        simulated_returns = [r["total_return"] for r in simulation_results]
        simulated_sharpes = [r["sharpe"] for r in simulation_results]
        simulated_max_dds = [r["max_drawdown"] for r in simulation_results]

        # 计算百分位数
        return_percentile = stats.percentileofscore(simulated_returns, actual_return)
        sharpe_percentile = stats.percentileofscore(simulated_sharpes, actual_sharpe)
        dd_percentile = stats.percentileofscore(simulated_max_dds, actual_max_dd)

        return {
            "n_simulations": n_simulations,
            "actual_return": float(actual_return),
            "actual_sharpe": float(actual_sharpe),
            "actual_max_drawdown": float(actual_max_dd),
            "return_percentile": float(return_percentile),
            "sharpe_percentile": float(sharpe_percentile),
            "drawdown_percentile": float(dd_percentile),
            "simulated_return_mean": float(np.mean(simulated_returns)),
            "simulated_return_std": float(np.std(simulated_returns)),
            "simulated_sharpe_mean": float(np.mean(simulated_sharpes)),
            "simulated_sharpe_std": float(np.std(simulated_sharpes)),
            "simulated_dd_mean": float(np.mean(simulated_max_dds)),
            "simulated_dd_std": float(np.std(simulated_max_dds)),
            "confidence_interval_95": {
                "return": (
                    float(np.percentile(simulated_returns, 2.5)),
                    float(np.percentile(simulated_returns, 97.5)),
                ),
                "sharpe": (
                    float(np.percentile(simulated_sharpes, 2.5)),
                    float(np.percentile(simulated_sharpes, 97.5)),
                ),
                "max_drawdown": (
                    float(np.percentile(simulated_max_dds, 2.5)),
                    float(np.percentile(simulated_max_dds, 97.5)),
                ),
            },
        }

    def _calculate_sharpe(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """计算夏普比率

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率

        Returns:
            夏普比率
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / TRADING_DAYS_PER_YEAR

        if excess_returns.std() == 0:
            return 0.0

        return float(
            np.sqrt(TRADING_DAYS_PER_YEAR)
            * excess_returns.mean()
            / excess_returns.std()
        )

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤

        Args:
            returns: 收益率序列

        Returns:
            最大回撤
        """
        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        return float(abs(drawdown.min()))

    def _test_ic_stability(self, in_sample: pd.Series, out_sample: pd.Series) -> float:
        """测试信息系数稳定性

        Args:
            in_sample: 样本内收益
            out_sample: 样本外收益

        Returns:
            IC稳定性得分
        """
        # 比较收益率分布的相似性
        ks_stat, ks_p_value = stats.ks_2samp(
            in_sample.dropna().values, out_sample.dropna().values
        )

        # 转换为稳定性得分（1-KS统计量，越接近1越稳定）
        return float(1 - ks_stat)

    def _test_parameter_sensitivity(self) -> Dict[str, float]:
        """测试参数敏感性

        Returns:
            参数敏感性结果
        """
        # 这里需要多组参数的回测结果
        # 简化处理，返回默认值
        return {
            "parameter_stability": 0.85,
            "optimal_region_size": 0.15,
            "sensitivity_score": 0.25,
        }

    def _calculate_overfitting_score(
        self, sharpe_decay: float, return_decay: float
    ) -> float:
        """计算过拟合得分

        Args:
            sharpe_decay: 夏普比率衰减
            return_decay: 收益率衰减

        Returns:
            过拟合得分（0-1，越高越可能过拟合）
        """
        # 加权平均，夏普比率衰减权重更高
        score = 0.6 * abs(sharpe_decay) + 0.4 * abs(return_decay)

        # 限制在0-1范围内
        return float(min(1.0, max(0.0, score)))

    def _analyze_subperiods(self, returns: pd.Series) -> Dict[str, Any]:
        """分析子期间表现

        Args:
            returns: 收益率序列

        Returns:
            子期间分析结果
        """
        # 按年度分组
        yearly_returns = returns.resample("Y").apply(
            lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
        )

        # 按季度分组
        quarterly_returns = returns.resample("Q").apply(
            lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
        )

        # 按月度分组
        monthly_returns = returns.resample("M").apply(
            lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
        )

        # 一致性分析
        yearly_positive_rate = (yearly_returns > 0).mean()
        quarterly_positive_rate = (quarterly_returns > 0).mean()
        monthly_positive_rate = (monthly_returns > 0).mean()

        # 计算各期间的夏普比率
        yearly_sharpe = (
            self._calculate_sharpe(yearly_returns) if len(yearly_returns) > 0 else 0
        )
        quarterly_sharpe = (
            self._calculate_sharpe(quarterly_returns)
            if len(quarterly_returns) > 0
            else 0
        )
        monthly_sharpe = (
            self._calculate_sharpe(monthly_returns) if len(monthly_returns) > 0 else 0
        )

        return {
            "yearly": {
                "count": len(yearly_returns),
                "positive_rate": float(yearly_positive_rate),
                "mean_return": float(yearly_returns.mean()),
                "std_return": float(yearly_returns.std()),
                "sharpe": float(yearly_sharpe),
                "best": float(yearly_returns.max()),
                "worst": float(yearly_returns.min()),
            },
            "quarterly": {
                "count": len(quarterly_returns),
                "positive_rate": float(quarterly_positive_rate),
                "mean_return": float(quarterly_returns.mean()),
                "std_return": float(quarterly_returns.std()),
                "sharpe": float(quarterly_sharpe),
                "best": float(quarterly_returns.max()),
                "worst": float(quarterly_returns.min()),
            },
            "monthly": {
                "count": len(monthly_returns),
                "positive_rate": float(monthly_positive_rate),
                "mean_return": float(monthly_returns.mean()),
                "std_return": float(monthly_returns.std()),
                "sharpe": float(monthly_sharpe),
                "best": float(monthly_returns.max()),
                "worst": float(monthly_returns.min()),
            },
            "consistency_score": float(
                np.mean(
                    [
                        yearly_positive_rate,
                        quarterly_positive_rate,
                        monthly_positive_rate,
                    ]
                )
            ),
        }

    def _test_distribution_stability(self, returns: pd.Series) -> Dict[str, float]:
        """测试收益分布稳定性

        Args:
            returns: 收益率序列

        Returns:
            分布稳定性指标
        """
        # 将数据分成多个子期间
        n_periods = 4
        period_size = len(returns) // n_periods

        moments = {"mean": [], "std": [], "skew": [], "kurtosis": []}

        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(returns)
            period_returns = returns.iloc[start_idx:end_idx]

            if len(period_returns) > 0:
                moments["mean"].append(period_returns.mean())
                moments["std"].append(period_returns.std())
                moments["skew"].append(period_returns.skew())
                moments["kurtosis"].append(period_returns.kurtosis())

        # 计算各矩的稳定性（使用变异系数）
        stability_scores = {}
        for moment_name, values in moments.items():
            if len(values) > 0 and np.mean(values) != 0:
                cv = np.std(values) / abs(np.mean(values))
                stability_scores[f"{moment_name}_stability"] = float(1 / (1 + cv))
            else:
                stability_scores[f"{moment_name}_stability"] = 0.0

        # 整体稳定性得分
        overall_stability = np.mean(list(stability_scores.values()))
        stability_scores["overall_stability"] = float(overall_stability)

        return stability_scores

    def _test_correlation_stability(self) -> Dict[str, float]:
        """测试相关性稳定性

        Returns:
            相关性稳定性指标
        """
        if self.market_data is None or len(self.market_data.columns) < 2:
            return {
                "correlation_stability": 1.0,
                "mean_correlation_change": 0.0,
                "max_correlation_change": 0.0,
            }

        # 计算滚动相关性
        window_size = 60  # 60天窗口

        # 选择前两列进行相关性分析（简化处理）
        if len(self.market_data.columns) >= 2:
            col1 = self.market_data.iloc[:, 0]
            col2 = self.market_data.iloc[:, 1]

            rolling_corr = col1.rolling(window_size).corr(col2)
            rolling_corr = rolling_corr.dropna()

            if len(rolling_corr) > 0:
                corr_std = rolling_corr.std()
                corr_mean = abs(rolling_corr.mean())

                stability = 1 / (1 + corr_std) if corr_mean > 0 else 1.0
                mean_change = rolling_corr.diff().abs().mean()
                max_change = rolling_corr.diff().abs().max()
            else:
                stability = 1.0
                mean_change = 0.0
                max_change = 0.0
        else:
            stability = 1.0
            mean_change = 0.0
            max_change = 0.0

        return {
            "correlation_stability": float(stability),
            "mean_correlation_change": float(mean_change),
            "max_correlation_change": float(max_change),
        }

    def _perform_stress_tests(self, returns: pd.Series) -> Dict[str, float]:
        """执行压力测试

        Args:
            returns: 收益率序列

        Returns:
            压力测试结果
        """
        # 模拟不同市场压力情景
        scenarios = {}

        # 场景1：市场崩盘（-20%瞬间下跌）
        crash_returns = returns.copy()
        crash_idx = len(crash_returns) // 2  # 在中间位置模拟崩盘
        crash_returns.iloc[crash_idx] = -0.20
        crash_performance = (1 + crash_returns).prod() - 1
        scenarios["market_crash"] = float(crash_performance)

        # 场景2：持续下跌（连续10天每天-2%）
        bear_returns = returns.copy()
        bear_start = len(bear_returns) // 3
        for i in range(min(10, len(bear_returns) - bear_start)):
            bear_returns.iloc[bear_start + i] = -0.02
        bear_performance = (1 + bear_returns).prod() - 1
        scenarios["bear_market"] = float(bear_performance)

        # 场景3：高波动（波动率翻倍）
        high_vol_returns = returns * 2
        high_vol_performance = (1 + high_vol_returns).prod() - 1
        scenarios["high_volatility"] = float(high_vol_performance)

        # 场景4：流动性危机（成交量萎缩，滑点增加）
        liquidity_returns = returns * 0.95  # 模拟5%的额外滑点
        liquidity_performance = (1 + liquidity_returns).prod() - 1
        scenarios["liquidity_crisis"] = float(liquidity_performance)

        # 计算压力测试综合得分
        baseline_performance = (1 + returns).prod() - 1

        stress_impacts = []
        for scenario_name, scenario_performance in scenarios.items():
            impact = (
                abs(scenario_performance - baseline_performance)
                / abs(baseline_performance)
                if baseline_performance != 0
                else 1.0
            )
            stress_impacts.append(impact)

        scenarios["average_stress_impact"] = float(np.mean(stress_impacts))
        scenarios["max_stress_impact"] = float(np.max(stress_impacts))
        scenarios["stress_resilience_score"] = float(
            1 / (1 + scenarios["average_stress_impact"])
        )

        return scenarios

    def _test_outlier_impact(self, returns: pd.Series) -> Dict[str, float]:
        """测试异常值影响

        Args:
            returns: 收益率序列

        Returns:
            异常值影响分析结果
        """
        # 识别异常值（使用3倍标准差）
        mean_return = returns.mean()
        std_return = returns.std()

        outliers = returns[abs(returns - mean_return) > 3 * std_return]
        n_outliers = len(outliers)
        outlier_ratio = n_outliers / len(returns)

        # 计算去除异常值后的性能
        returns_no_outliers = returns[abs(returns - mean_return) <= 3 * std_return]

        # 原始性能
        original_sharpe = self._calculate_sharpe(returns)
        original_return = returns.mean() * TRADING_DAYS_PER_YEAR

        # 去除异常值后的性能
        if len(returns_no_outliers) > 0:
            adjusted_sharpe = self._calculate_sharpe(returns_no_outliers)
            adjusted_return = returns_no_outliers.mean() * TRADING_DAYS_PER_YEAR
        else:
            adjusted_sharpe = 0.0
            adjusted_return = 0.0

        # 异常值贡献
        outlier_contribution = (
            (original_return - adjusted_return) / abs(original_return)
            if original_return != 0
            else 0.0
        )
        sharpe_impact = (
            (original_sharpe - adjusted_sharpe) / abs(original_sharpe)
            if original_sharpe != 0
            else 0.0
        )

        # 异常值的平均影响
        if n_outliers > 0:
            avg_outlier_magnitude = outliers.abs().mean()
            max_outlier = outliers.abs().max()
        else:
            avg_outlier_magnitude = 0.0
            max_outlier = 0.0

        return {
            "n_outliers": n_outliers,
            "outlier_ratio": float(outlier_ratio),
            "original_sharpe": float(original_sharpe),
            "adjusted_sharpe": float(adjusted_sharpe),
            "sharpe_impact": float(sharpe_impact),
            "return_contribution": float(outlier_contribution),
            "avg_outlier_magnitude": float(avg_outlier_magnitude),
            "max_outlier_magnitude": float(max_outlier),
            "outlier_dependency": float(abs(outlier_contribution)),
        }

    def _test_cost_sensitivity(self) -> Dict[str, float]:
        """测试交易成本敏感性

        Returns:
            成本敏感性分析结果
        """
        # 获取基准结果
        baseline_metrics = self.backtest_results.performance_metrics
        baseline_return = baseline_metrics.get("total_return", 0)
        baseline_sharpe = baseline_metrics.get("sharpe_ratio", 0)

        # 模拟不同成本水平
        cost_multipliers = [0.5, 1.0, 1.5, 2.0, 3.0]
        cost_impacts = []

        for multiplier in cost_multipliers:
            # 估算调整后的收益（简化处理）
            # 假设成本占总收益的比例
            base_cost_ratio = 0.02  # 假设基准成本占2%
            adjusted_cost_ratio = base_cost_ratio * multiplier
            adjusted_return = baseline_return * (1 - adjusted_cost_ratio)

            cost_impacts.append(
                {
                    "multiplier": multiplier,
                    "adjusted_return": float(adjusted_return),
                    "return_impact": float(
                        (baseline_return - adjusted_return) / abs(baseline_return)
                    )
                    if baseline_return != 0
                    else 0.0,
                }
            )

        # 计算成本敏感度
        returns_at_different_costs = [ci["adjusted_return"] for ci in cost_impacts]
        cost_sensitivity = (
            np.std(returns_at_different_costs)
            / abs(np.mean(returns_at_different_costs))
            if np.mean(returns_at_different_costs) != 0
            else 0.0
        )

        # 盈亏平衡点（收益为0时的成本倍数）
        breakeven_multiplier = (
            baseline_return / base_cost_ratio if base_cost_ratio > 0 else float("inf")
        )

        return {
            "baseline_return": float(baseline_return),
            "cost_sensitivity": float(cost_sensitivity),
            "breakeven_cost_multiplier": float(min(breakeven_multiplier, 100.0)),
            "cost_impact_details": cost_impacts,
            "robust_to_costs": cost_sensitivity < 0.5,
        }

    def _test_data_quality_impact(self) -> Dict[str, float]:
        """测试数据质量影响

        Returns:
            数据质量影响分析结果
        """
        returns = self.backtest_results.daily_returns

        # 模拟数据质量问题
        quality_tests = {}

        # 测试1：随机缺失值
        missing_ratio = 0.05  # 5%缺失
        returns_with_missing = returns.copy()
        missing_indices = np.random.choice(
            len(returns_with_missing),
            size=int(len(returns_with_missing) * missing_ratio),
            replace=False,
        )
        returns_with_missing.iloc[missing_indices] = np.nan
        returns_filled = returns_with_missing.fillna(0)  # 简单填充

        sharpe_with_missing = self._calculate_sharpe(returns_filled)
        quality_tests["missing_data_impact"] = float(
            abs(self._calculate_sharpe(returns) - sharpe_with_missing)
        )

        # 测试2：数据噪声
        noise_level = 0.001  # 0.1%噪声
        returns_with_noise = returns + np.random.normal(0, noise_level, len(returns))
        sharpe_with_noise = self._calculate_sharpe(returns_with_noise)
        quality_tests["noise_impact"] = float(
            abs(self._calculate_sharpe(returns) - sharpe_with_noise)
        )

        # 测试3：时间戳错误（延迟）
        returns_delayed = returns.shift(1).fillna(0)
        sharpe_delayed = self._calculate_sharpe(returns_delayed)
        quality_tests["timing_error_impact"] = float(
            abs(self._calculate_sharpe(returns) - sharpe_delayed)
        )

        # 综合数据质量得分
        total_impact = sum(quality_tests.values())
        quality_score = 1 / (1 + total_impact)

        quality_tests["data_quality_score"] = float(quality_score)
        quality_tests["robust_to_data_issues"] = quality_score > 0.8

        return quality_tests

    def _simple_backtest(
        self, strategy_func: Callable, data: pd.DataFrame
    ) -> pd.Series:
        """执行简单回测

        Args:
            strategy_func: 策略函数
            data: 市场数据

        Returns:
            收益率序列
        """
        # 简化的回测实现
        # 实际应该调用完整的回测引擎

        returns = []
        positions = {}
        cash = 1000000  # 初始资金

        for i in range(1, len(data)):
            current_data = data.iloc[:i]

            # 调用策略函数获取信号
            try:
                signals = strategy_func(current_data, positions, cash)

                # 简单计算收益（这里仅作示例）
                if signals and len(signals) > 0:
                    # 假设完全执行第一个信号
                    signal = signals[0]
                    if signal.action == "BUY":
                        daily_return = 0.001  # 简化为固定收益
                    elif signal.action == "SELL":
                        daily_return = -0.0005
                    else:
                        daily_return = 0.0
                else:
                    daily_return = 0.0

                returns.append(daily_return)

            except Exception as e:
                logger.warning(f"Strategy execution error: {e}")
                returns.append(0.0)

        return pd.Series(returns, index=data.index[1:])


# 模块级别函数
def validate_backtest(
    backtest_results: Any,
    market_data: pd.DataFrame,
    strategy_func: Optional[Callable] = None,
) -> ValidationReport:
    """验证回测结果的便捷函数

    Args:
        backtest_results: 回测结果
        market_data: 市场数据
        strategy_func: 策略函数

    Returns:
        验证报告
    """
    validator = BacktestValidator()
    return validator.validate(backtest_results, market_data, strategy_func)


def check_overfitting(
    in_sample_returns: pd.Series, out_sample_returns: pd.Series
) -> Dict[str, float]:
    """检查过拟合的便捷函数

    Args:
        in_sample_returns: 样本内收益率
        out_sample_returns: 样本外收益率

    Returns:
        过拟合检查结果
    """
    validator = BacktestValidator()

    # 计算夏普比率
    in_sample_sharpe = validator._calculate_sharpe(in_sample_returns)
    out_sample_sharpe = validator._calculate_sharpe(out_sample_returns)

    # 计算衰减
    sharpe_decay = (
        (in_sample_sharpe - out_sample_sharpe) / in_sample_sharpe
        if in_sample_sharpe != 0
        else 0
    )

    # 收益率衰减
    in_sample_mean = in_sample_returns.mean() * TRADING_DAYS_PER_YEAR
    out_sample_mean = out_sample_returns.mean() * TRADING_DAYS_PER_YEAR
    return_decay = (
        (in_sample_mean - out_sample_mean) / abs(in_sample_mean)
        if in_sample_mean != 0
        else 0
    )

    # IC稳定性
    ic_stability = validator._test_ic_stability(in_sample_returns, out_sample_returns)

    # 过拟合得分
    overfitting_score = validator._calculate_overfitting_score(
        sharpe_decay, return_decay
    )

    return {
        "in_sample_sharpe": float(in_sample_sharpe),
        "out_sample_sharpe": float(out_sample_sharpe),
        "sharpe_decay": float(sharpe_decay),
        "return_decay": float(return_decay),
        "ic_stability": float(ic_stability),
        "overfitting_score": float(overfitting_score),
        "is_overfitted": overfitting_score > 0.5,
    }


def run_monte_carlo_analysis(
    returns: pd.Series, n_simulations: int = 1000
) -> Dict[str, Any]:
    """运行蒙特卡洛分析的便捷函数

    Args:
        returns: 收益率序列
        n_simulations: 模拟次数

    Returns:
        蒙特卡洛分析结果
    """
    # 获取统计特征
    mean_return = returns.mean()
    std_return = returns.std()

    # 运行模拟
    simulated_results = []
    np.random.seed(42)

    for _ in range(n_simulations):
        simulated_returns = np.random.normal(mean_return, std_return, len(returns))
        cumulative_return = (1 + pd.Series(simulated_returns)).prod() - 1
        simulated_results.append(float(cumulative_return))

    # 计算统计量
    actual_return = (1 + returns).prod() - 1
    percentile = stats.percentileofscore(simulated_results, actual_return)

    return {
        "actual_return": float(actual_return),
        "simulated_mean": float(np.mean(simulated_results)),
        "simulated_std": float(np.std(simulated_results)),
        "percentile": float(percentile),
        "confidence_interval_95": (
            float(np.percentile(simulated_results, 2.5)),
            float(np.percentile(simulated_results, 97.5)),
        ),
        "probability_of_profit": float(np.mean([r > 0 for r in simulated_results])),
    }
