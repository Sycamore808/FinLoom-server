"""
历史压力测试模块
基于历史危机事件的压力测试
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.constants import TRADING_DAYS_PER_YEAR
from common.exceptions import ModelError
from common.logging_system import setup_logger

logger = setup_logger("historical_stress_test")


class CrisisEvent(Enum):
    """历史危机事件枚举"""

    BLACK_MONDAY_1987 = "black_monday_1987"
    ASIAN_CRISIS_1997 = "asian_crisis_1997"
    LTCM_COLLAPSE_1998 = "ltcm_collapse_1998"
    DOT_COM_CRASH_2000 = "dot_com_crash_2000"
    SEPTEMBER_11_2001 = "september_11_2001"
    FINANCIAL_CRISIS_2008 = "financial_crisis_2008"
    EUROPEAN_DEBT_2011 = "european_debt_2011"
    CHINA_CRASH_2015 = "china_crash_2015"
    BREXIT_2016 = "brexit_2016"
    COVID_CRASH_2020 = "covid_crash_2020"
    RUSSIA_UKRAINE_2022 = "russia_ukraine_2022"


@dataclass
class CrisisParameters:
    """危机参数"""

    event_name: str
    start_date: datetime
    end_date: datetime
    duration_days: int
    equity_drawdown: float
    bond_performance: float
    commodity_impact: float
    fx_volatility: float
    correlation_breakdown: bool
    liquidity_crisis: bool
    contagion_effects: Dict[str, float]


@dataclass
class StressTestConfig:
    """压力测试配置"""

    crisis_events: List[CrisisEvent] = None
    scaling_factor: float = 1.0  # 冲击放大倍数
    apply_contagion: bool = True
    include_second_order_effects: bool = True
    recovery_period_days: int = 60
    confidence_level: float = 0.99
    use_conditional_correlation: bool = True


@dataclass
class HistoricalStressResult:
    """历史压力测试结果"""

    crisis_event: CrisisEvent
    portfolio_impact: float
    asset_impacts: pd.Series
    worst_day_loss: float
    recovery_time: int
    var_breach: bool
    liquidity_impact: float
    correlation_changes: pd.DataFrame
    contagion_effects: Dict[str, float]
    risk_metrics: Dict[str, float]


class HistoricalStressTest:
    """历史压力测试类"""

    # 历史危机数据库
    CRISIS_DATABASE = {
        CrisisEvent.BLACK_MONDAY_1987: CrisisParameters(
            event_name="Black Monday 1987",
            start_date=datetime(1987, 10, 19),
            end_date=datetime(1987, 10, 30),
            duration_days=10,
            equity_drawdown=-0.22,
            bond_performance=0.02,
            commodity_impact=-0.05,
            fx_volatility=2.5,
            correlation_breakdown=True,
            liquidity_crisis=True,
            contagion_effects={"global": 0.8, "regional": 0.6},
        ),
        CrisisEvent.ASIAN_CRISIS_1997: CrisisParameters(
            event_name="Asian Financial Crisis 1997",
            start_date=datetime(1997, 7, 2),
            end_date=datetime(1997, 12, 31),
            duration_days=180,
            equity_drawdown=-0.35,
            bond_performance=0.05,
            commodity_impact=-0.15,
            fx_volatility=4.0,
            correlation_breakdown=True,
            liquidity_crisis=True,
            contagion_effects={"asia": 0.9, "emerging": 0.7, "developed": 0.3},
        ),
        CrisisEvent.FINANCIAL_CRISIS_2008: CrisisParameters(
            event_name="Global Financial Crisis 2008",
            start_date=datetime(2008, 9, 15),
            end_date=datetime(2009, 3, 9),
            duration_days=175,
            equity_drawdown=-0.55,
            bond_performance=0.08,
            commodity_impact=-0.40,
            fx_volatility=3.5,
            correlation_breakdown=True,
            liquidity_crisis=True,
            contagion_effects={"global": 0.95, "banking": 1.0, "real_estate": 1.0},
        ),
        CrisisEvent.COVID_CRASH_2020: CrisisParameters(
            event_name="COVID-19 Crash 2020",
            start_date=datetime(2020, 2, 20),
            end_date=datetime(2020, 3, 23),
            duration_days=32,
            equity_drawdown=-0.34,
            bond_performance=0.03,
            commodity_impact=-0.50,
            fx_volatility=2.8,
            correlation_breakdown=True,
            liquidity_crisis=True,
            contagion_effects={"global": 1.0, "travel": 1.5, "energy": 1.2},
        ),
    }

    def __init__(self, config: Optional[StressTestConfig] = None):
        """初始化历史压力测试

        Args:
            config: 压力测试配置
        """
        self.config = config or StressTestConfig()
        if self.config.crisis_events is None:
            self.config.crisis_events = list(CrisisEvent)
        self.test_results: List[HistoricalStressResult] = []

    def run_historical_stress_test(
        self,
        portfolio_weights: pd.Series,
        asset_returns: pd.DataFrame,
        crisis_event: CrisisEvent,
    ) -> HistoricalStressResult:
        """运行历史压力测试

        Args:
            portfolio_weights: 投资组合权重
            asset_returns: 资产收益率数据
            crisis_event: 危机事件

        Returns:
            历史压力测试结果
        """
        logger.info(f"Running historical stress test for {crisis_event.value}")

        # 获取危机参数
        crisis_params = self.CRISIS_DATABASE.get(crisis_event)
        if not crisis_params:
            raise ValueError(f"Crisis event {crisis_event} not found in database")

        # 提取危机期间数据
        crisis_data = self._extract_crisis_period_data(
            asset_returns, crisis_params.start_date, crisis_params.end_date
        )

        # 应用危机冲击
        stressed_returns = self._apply_crisis_shocks(crisis_data, crisis_params)

        # 计算投资组合影响
        portfolio_impact = self._calculate_portfolio_impact(
            stressed_returns, portfolio_weights
        )

        # 计算各资产影响
        asset_impacts = self._calculate_asset_impacts(stressed_returns)

        # 计算最糟糕一天的损失
        worst_day_loss = self._calculate_worst_day_loss(
            stressed_returns, portfolio_weights
        )

        # 估计恢复时间
        recovery_time = self._estimate_recovery_time(
            asset_returns, crisis_params.end_date, portfolio_weights
        )

        # 检查VaR突破
        var_breach = self._check_var_breach(
            portfolio_impact, asset_returns, portfolio_weights
        )

        # 计算流动性影响
        liquidity_impact = self._calculate_liquidity_impact(
            crisis_params, portfolio_weights
        )

        # 分析相关性变化
        correlation_changes = self._analyze_correlation_changes(
            asset_returns, crisis_params
        )

        # 评估传染效应
        if self.config.apply_contagion:
            contagion_effects = self._evaluate_contagion_effects(
                crisis_params, portfolio_weights
            )
        else:
            contagion_effects = {}

        # 计算风险指标
        risk_metrics = self._calculate_crisis_risk_metrics(
            stressed_returns, portfolio_weights
        )

        result = HistoricalStressResult(
            crisis_event=crisis_event,
            portfolio_impact=portfolio_impact,
            asset_impacts=asset_impacts,
            worst_day_loss=worst_day_loss,
            recovery_time=recovery_time,
            var_breach=var_breach,
            liquidity_impact=liquidity_impact,
            correlation_changes=correlation_changes,
            contagion_effects=contagion_effects,
            risk_metrics=risk_metrics,
        )

        self.test_results.append(result)

        logger.info(f"Stress test completed. Portfolio impact: {portfolio_impact:.2%}")

        return result

    def run_all_crisis_scenarios(
        self, portfolio_weights: pd.Series, asset_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """运行所有危机场景

        Args:
            portfolio_weights: 投资组合权重
            asset_returns: 资产收益率数据

        Returns:
            所有场景结果汇总
        """
        logger.info(f"Running {len(self.config.crisis_events)} crisis scenarios")

        results = []

        for crisis_event in self.config.crisis_events:
            try:
                result = self.run_historical_stress_test(
                    portfolio_weights, asset_returns, crisis_event
                )

                results.append(
                    {
                        "crisis": crisis_event.value,
                        "portfolio_impact": result.portfolio_impact,
                        "worst_day": result.worst_day_loss,
                        "recovery_days": result.recovery_time,
                        "var_breach": result.var_breach,
                        "liquidity_impact": result.liquidity_impact,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to run stress test for {crisis_event}: {e}")

        return pd.DataFrame(results)

    def analyze_crisis_patterns(self, asset_returns: pd.DataFrame) -> Dict[str, Any]:
        """分析危机模式

        Args:
            asset_returns: 资产收益率数据

        Returns:
            危机模式分析结果
        """
        logger.info("Analyzing historical crisis patterns")

        patterns = {
            "common_factors": [],
            "leading_indicators": {},
            "contagion_paths": [],
            "recovery_patterns": {},
        }

        for crisis_event, crisis_params in self.CRISIS_DATABASE.items():
            # 提取危机前后数据
            pre_crisis = self._extract_pre_crisis_data(
                asset_returns, crisis_params.start_date, lookback_days=60
            )

            post_crisis = self._extract_post_crisis_data(
                asset_returns,
                crisis_params.end_date,
                forward_days=self.config.recovery_period_days,
            )

            # 分析先导指标
            leading_indicators = self._identify_leading_indicators(pre_crisis)
            patterns["leading_indicators"][crisis_event.value] = leading_indicators

            # 分析恢复模式
            recovery_pattern = self._analyze_recovery_pattern(post_crisis)
            patterns["recovery_patterns"][crisis_event.value] = recovery_pattern

        # 识别共同因素
        patterns["common_factors"] = self._identify_common_factors()

        # 分析传染路径
        patterns["contagion_paths"] = self._analyze_contagion_paths()

        return patterns

    def calculate_conditional_stress_test(
        self,
        portfolio_weights: pd.Series,
        asset_returns: pd.DataFrame,
        conditioning_event: str,
    ) -> HistoricalStressResult:
        """计算条件压力测试

        Args:
            portfolio_weights: 投资组合权重
            asset_returns: 资产收益率数据
            conditioning_event: 条件事件

        Returns:
            条件压力测试结果
        """
        logger.info(f"Running conditional stress test on {conditioning_event}")

        # 识别相似的历史事件
        similar_events = self._find_similar_events(conditioning_event)

        # 构建条件分布
        conditional_distribution = self._build_conditional_distribution(
            asset_returns, similar_events
        )

        # 生成压力场景
        stress_scenario = self._generate_conditional_scenario(
            conditional_distribution, self.config.confidence_level
        )

        # 应用压力场景
        stressed_returns = self._apply_stress_scenario(asset_returns, stress_scenario)

        # 计算影响
        portfolio_impact = self._calculate_portfolio_impact(
            stressed_returns, portfolio_weights
        )

        # 构建结果
        result = HistoricalStressResult(
            crisis_event=None,  # 条件事件
            portfolio_impact=portfolio_impact,
            asset_impacts=pd.Series(),
            worst_day_loss=0,
            recovery_time=0,
            var_breach=False,
            liquidity_impact=0,
            correlation_changes=pd.DataFrame(),
            contagion_effects={},
            risk_metrics={},
        )

        return result

    def _extract_crisis_period_data(
        self, asset_returns: pd.DataFrame, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """提取危机期间数据

        Args:
            asset_returns: 资产收益率数据
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            危机期间数据
        """
        # 如果索引不是日期，尝试转换
        if not isinstance(asset_returns.index, pd.DatetimeIndex):
            # 生成模拟的危机期间数据
            crisis_length = (end_date - start_date).days
            if crisis_length > len(asset_returns):
                crisis_length = min(30, len(asset_returns))

            # 使用最近的数据作为危机数据
            return asset_returns.tail(crisis_length)

        # 正常的日期筛选
        mask = (asset_returns.index >= start_date) & (asset_returns.index <= end_date)
        return asset_returns.loc[mask]

    def _apply_crisis_shocks(
        self, crisis_data: pd.DataFrame, crisis_params: CrisisParameters
    ) -> pd.DataFrame:
        """应用危机冲击

        Args:
            crisis_data: 危机期间数据
            crisis_params: 危机参数

        Returns:
            冲击后的收益率
        """
        stressed_returns = crisis_data.copy()

        # 应用缩放因子
        scaling = self.config.scaling_factor

        # 应用资产类别冲击
        for col in stressed_returns.columns:
            if "equity" in col.lower() or "stock" in col.lower():
                stressed_returns[col] *= 1 + crisis_params.equity_drawdown * scaling
            elif "bond" in col.lower():
                stressed_returns[col] *= 1 + crisis_params.bond_performance * scaling
            elif "commodity" in col.lower():
                stressed_returns[col] *= 1 + crisis_params.commodity_impact * scaling
            else:
                # 默认使用股票冲击
                stressed_returns[col] *= (
                    1 + crisis_params.equity_drawdown * scaling * 0.5
                )

        # 应用波动率放大
        if crisis_params.fx_volatility > 1:
            vol_multiplier = crisis_params.fx_volatility
            stressed_returns *= np.random.normal(
                1, vol_multiplier - 1, stressed_returns.shape
            )

        return stressed_returns

    def _calculate_portfolio_impact(
        self, stressed_returns: pd.DataFrame, portfolio_weights: pd.Series
    ) -> float:
        """计算投资组合影响

        Args:
            stressed_returns: 压力测试后的收益率
            portfolio_weights: 投资组合权重

        Returns:
            投资组合总影响
        """
        # 确保权重和收益率列对齐
        common_assets = stressed_returns.columns.intersection(portfolio_weights.index)

        if len(common_assets) == 0:
            # 如果没有匹配的资产，使用平均权重
            portfolio_returns = stressed_returns.mean(axis=1)
        else:
            aligned_weights = portfolio_weights[common_assets]
            aligned_weights = aligned_weights / aligned_weights.sum()
            portfolio_returns = stressed_returns[common_assets] @ aligned_weights

        # 计算累计影响
        cumulative_impact = (1 + portfolio_returns).prod() - 1

        return cumulative_impact

    def _calculate_asset_impacts(self, stressed_returns: pd.DataFrame) -> pd.Series:
        """计算各资产影响

        Args:
            stressed_returns: 压力测试后的收益率

        Returns:
            各资产影响Series
        """
        # 计算每个资产的累计收益
        asset_impacts = (1 + stressed_returns).prod() - 1

        return asset_impacts

    def _calculate_worst_day_loss(
        self, stressed_returns: pd.DataFrame, portfolio_weights: pd.Series
    ) -> float:
        """计算最糟糕一天的损失

        Args:
            stressed_returns: 压力测试后的收益率
            portfolio_weights: 投资组合权重

        Returns:
            最糟糕一天的损失
        """
        # 计算每日组合收益
        common_assets = stressed_returns.columns.intersection(portfolio_weights.index)

        if len(common_assets) == 0:
            daily_returns = stressed_returns.mean(axis=1)
        else:
            aligned_weights = portfolio_weights[common_assets]
            aligned_weights = aligned_weights / aligned_weights.sum()
            daily_returns = stressed_returns[common_assets] @ aligned_weights

        # 找出最小值
        worst_day_loss = daily_returns.min()

        return worst_day_loss

    def _estimate_recovery_time(
        self,
        asset_returns: pd.DataFrame,
        crisis_end_date: datetime,
        portfolio_weights: pd.Series,
    ) -> int:
        """估计恢复时间

        Args:
            asset_returns: 资产收益率数据
            crisis_end_date: 危机结束日期
            portfolio_weights: 投资组合权重

        Returns:
            恢复天数
        """
        # 简化处理：返回配置的恢复期
        return self.config.recovery_period_days

    def _check_var_breach(
        self,
        portfolio_impact: float,
        asset_returns: pd.DataFrame,
        portfolio_weights: pd.Series,
    ) -> bool:
        """检查VaR突破

        Args:
            portfolio_impact: 投资组合影响
            asset_returns: 资产收益率数据
            portfolio_weights: 投资组合权重

        Returns:
            是否突破VaR
        """
        # 计算历史VaR
        common_assets = asset_returns.columns.intersection(portfolio_weights.index)

        if len(common_assets) == 0:
            portfolio_returns = asset_returns.mean(axis=1)
        else:
            aligned_weights = portfolio_weights[common_assets]
            aligned_weights = aligned_weights / aligned_weights.sum()
            portfolio_returns = asset_returns[common_assets] @ aligned_weights

        var_threshold = np.percentile(
            portfolio_returns, (1 - self.config.confidence_level) * 100
        )

        # 检查是否突破
        return portfolio_impact < var_threshold

    def _calculate_liquidity_impact(
        self, crisis_params: CrisisParameters, portfolio_weights: pd.Series
    ) -> float:
        """计算流动性影响

        Args:
            crisis_params: 危机参数
            portfolio_weights: 投资组合权重

        Returns:
            流动性影响
        """
        if not crisis_params.liquidity_crisis:
            return 0.0

        # 基于资产类型估计流动性影响
        liquidity_impact = 0.0

        for asset, weight in portfolio_weights.items():
            if "small_cap" in asset.lower():
                liquidity_impact += weight * 0.20  # 小盘股20%影响
            elif "emerging" in asset.lower():
                liquidity_impact += weight * 0.15  # 新兴市场15%影响
            elif "high_yield" in asset.lower():
                liquidity_impact += weight * 0.10  # 高收益债10%影响
            else:
                liquidity_impact += weight * 0.05  # 其他5%影响

        return liquidity_impact

    def _analyze_correlation_changes(
        self, asset_returns: pd.DataFrame, crisis_params: CrisisParameters
    ) -> pd.DataFrame:
        """分析相关性变化

        Args:
            asset_returns: 资产收益率数据
            crisis_params: 危机参数

        Returns:
            相关性变化DataFrame
        """
        # 计算正常时期相关性
        normal_corr = asset_returns.corr()

        # 模拟危机时期相关性
        if crisis_params.correlation_breakdown:
            # 相关性趋向于1（危机时资产同涨同跌）
            crisis_corr = normal_corr.copy()
            crisis_corr = crisis_corr * 0.5 + 0.5  # 向1收敛

            # 对角线保持为1
            np.fill_diagonal(crisis_corr.values, 1.0)
        else:
            crisis_corr = normal_corr

        # 计算变化
        correlation_changes = crisis_corr - normal_corr

        return correlation_changes

    def _evaluate_contagion_effects(
        self, crisis_params: CrisisParameters, portfolio_weights: pd.Series
    ) -> Dict[str, float]:
        """评估传染效应

        Args:
            crisis_params: 危机参数
            portfolio_weights: 投资组合权重

        Returns:
            传染效应字典
        """
        contagion_effects = {}

        for region, effect_multiplier in crisis_params.contagion_effects.items():
            # 计算该地区的暴露
            region_exposure = 0.0

            for asset, weight in portfolio_weights.items():
                if region.lower() in asset.lower():
                    region_exposure += weight

            # 计算传染影响
            contagion_impact = (
                region_exposure * effect_multiplier * crisis_params.equity_drawdown
            )
            contagion_effects[region] = contagion_impact

        return contagion_effects

    def _calculate_crisis_risk_metrics(
        self, stressed_returns: pd.DataFrame, portfolio_weights: pd.Series
    ) -> Dict[str, float]:
        """计算危机风险指标

        Args:
            stressed_returns: 压力测试后的收益率
            portfolio_weights: 投资组合权重

        Returns:
            风险指标字典
        """
        # 计算组合收益
        common_assets = stressed_returns.columns.intersection(portfolio_weights.index)

        if len(common_assets) == 0:
            portfolio_returns = stressed_returns.mean(axis=1)
        else:
            aligned_weights = portfolio_weights[common_assets]
            aligned_weights = aligned_weights / aligned_weights.sum()
            portfolio_returns = stressed_returns[common_assets] @ aligned_weights

        metrics = {
            "crisis_volatility": portfolio_returns.std()
            * np.sqrt(TRADING_DAYS_PER_YEAR),
            "crisis_skewness": portfolio_returns.skew(),
            "crisis_kurtosis": portfolio_returns.kurtosis(),
            "downside_deviation": portfolio_returns[portfolio_returns < 0].std()
            * np.sqrt(TRADING_DAYS_PER_YEAR),
            "max_daily_loss": portfolio_returns.min(),
            "days_negative": (portfolio_returns < 0).sum(),
            "hit_ratio": (portfolio_returns > 0).mean(),
        }

        return metrics

    def _extract_pre_crisis_data(
        self, asset_returns: pd.DataFrame, crisis_start: datetime, lookback_days: int
    ) -> pd.DataFrame:
        """提取危机前数据

        Args:
            asset_returns: 资产收益率数据
            crisis_start: 危机开始日期
            lookback_days: 回看天数

        Returns:
            危机前数据
        """
        # 简化处理：返回危机前的数据
        if len(asset_returns) > lookback_days:
            return asset_returns.head(lookback_days)
        else:
            return asset_returns

    def _extract_post_crisis_data(
        self, asset_returns: pd.DataFrame, crisis_end: datetime, forward_days: int
    ) -> pd.DataFrame:
        """提取危机后数据

        Args:
            asset_returns: 资产收益率数据
            crisis_end: 危机结束日期
            forward_days: 前瞻天数

        Returns:
            危机后数据
        """
        # 简化处理：返回危机后的数据
        if len(asset_returns) > forward_days:
            return asset_returns.tail(forward_days)
        else:
            return asset_returns

    def _identify_leading_indicators(
        self, pre_crisis_data: pd.DataFrame
    ) -> Dict[str, float]:
        """识别先导指标

        Args:
            pre_crisis_data: 危机前数据

        Returns:
            先导指标字典
        """
        indicators = {}

        # 计算一些简单的先导指标
        indicators["volatility_increase"] = pre_crisis_data.std().mean()
        indicators["correlation_increase"] = pre_crisis_data.corr().values.mean()
        indicators["negative_skew"] = pre_crisis_data.skew().mean()

        return indicators

    def _analyze_recovery_pattern(
        self, post_crisis_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """分析恢复模式

        Args:
            post_crisis_data: 危机后数据

        Returns:
            恢复模式分析
        """
        pattern = {
            "recovery_speed": (1 + post_crisis_data.mean()).prod()
            ** (1 / len(post_crisis_data))
            - 1,
            "volatility_normalization": post_crisis_data.std().mean(),
            "correlation_normalization": post_crisis_data.corr().values.mean(),
        }

        return pattern

    def _identify_common_factors(self) -> List[str]:
        """识别共同因素

        Returns:
            共同因素列表
        """
        # 基于历史经验的共同因素
        return [
            "excessive_leverage",
            "liquidity_crisis",
            "correlation_breakdown",
            "volatility_spike",
            "contagion_effects",
            "policy_uncertainty",
        ]

    def _analyze_contagion_paths(self) -> List[Dict[str, Any]]:
        """分析传染路径

        Returns:
            传染路径列表
        """
        # 基于历史经验的传染路径
        paths = [
            {
                "source": "banking_sector",
                "targets": ["real_estate", "corporate_bonds", "equities"],
                "mechanism": "credit_crunch",
            },
            {
                "source": "emerging_markets",
                "targets": ["commodities", "developed_markets"],
                "mechanism": "capital_flight",
            },
            {
                "source": "currency_crisis",
                "targets": ["trade_partners", "foreign_debt"],
                "mechanism": "devaluation",
            },
        ]

        return paths

    def _find_similar_events(self, conditioning_event: str) -> List[CrisisEvent]:
        """查找相似事件

        Args:
            conditioning_event: 条件事件

        Returns:
            相似事件列表
        """
        # 简单的关键词匹配
        similar = []

        for crisis_event in CrisisEvent:
            if conditioning_event.lower() in crisis_event.value.lower():
                similar.append(crisis_event)

        # 如果没有找到，返回所有事件
        if not similar:
            similar = list(CrisisEvent)

        return similar

    def _build_conditional_distribution(
        self, asset_returns: pd.DataFrame, similar_events: List[CrisisEvent]
    ) -> pd.DataFrame:
        """构建条件分布

        Args:
            asset_returns: 资产收益率数据
            similar_events: 相似事件列表

        Returns:
            条件分布
        """
        # 简化处理：使用历史数据的子集
        n_samples = min(100, len(asset_returns))
        return asset_returns.sample(n=n_samples)

    def _generate_conditional_scenario(
        self, conditional_distribution: pd.DataFrame, confidence_level: float
    ) -> pd.DataFrame:
        """生成条件场景

        Args:
            conditional_distribution: 条件分布
            confidence_level: 置信水平

        Returns:
            条件场景
        """
        # 计算极端分位数
        scenario = conditional_distribution.quantile(1 - confidence_level)

        return scenario

    def _apply_stress_scenario(
        self, asset_returns: pd.DataFrame, stress_scenario: pd.Series
    ) -> pd.DataFrame:
        """应用压力场景

        Args:
            asset_returns: 资产收益率数据
            stress_scenario: 压力场景

        Returns:
            压力测试后的收益率
        """
        # 应用压力因子
        stressed = asset_returns.copy()

        for col in stressed.columns:
            if col in stress_scenario.index:
                stressed[col] *= 1 + stress_scenario[col]

        return stressed


# 模块级别函数
def run_historical_stress_test(
    portfolio_weights: pd.Series,
    asset_returns: pd.DataFrame,
    crisis_event: str = "financial_crisis_2008",
    config: Optional[StressTestConfig] = None,
) -> Dict[str, float]:
    """运行历史压力测试的便捷函数

    Args:
        portfolio_weights: 投资组合权重
        asset_returns: 资产收益率数据
        crisis_event: 危机事件名称
        config: 配置

    Returns:
        压力测试结果字典
    """
    tester = HistoricalStressTest(config)

    # 将字符串转换为枚举
    crisis_enum = CrisisEvent[crisis_event.upper()]

    result = tester.run_historical_stress_test(
        portfolio_weights, asset_returns, crisis_enum
    )

    return {
        "portfolio_impact": result.portfolio_impact,
        "worst_day_loss": result.worst_day_loss,
        "recovery_time": result.recovery_time,
        "var_breach": result.var_breach,
        "liquidity_impact": result.liquidity_impact,
    }
