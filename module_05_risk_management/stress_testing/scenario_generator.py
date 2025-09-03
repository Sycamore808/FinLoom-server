"""
压力测试场景生成器模块
生成各种压力测试场景
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.exceptions import ModelError
from common.logging_system import setup_logger
from scipy import stats

logger = setup_logger("scenario_generator")


class ScenarioType(Enum):
    """场景类型枚举"""

    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    MONTE_CARLO = "monte_carlo"
    FACTOR_SHOCK = "factor_shock"
    REGIME_BASED = "regime_based"
    TAIL_RISK = "tail_risk"


@dataclass
class ScenarioConfig:
    """场景配置"""

    scenario_types: List[ScenarioType] = None
    n_scenarios: int = 100
    confidence_levels: List[float] = None
    time_horizon: int = 20
    include_correlation_breaks: bool = True
    include_volatility_clusters: bool = True
    include_fat_tails: bool = True
    seed: Optional[int] = None


@dataclass
class StressScenario:
    """压力测试场景"""

    scenario_id: str
    scenario_type: ScenarioType
    description: str
    probability: float
    time_horizon: int
    asset_returns: pd.DataFrame
    factor_shocks: Optional[Dict[str, float]]
    correlation_matrix: Optional[pd.DataFrame]
    volatility_multipliers: Optional[pd.Series]
    metadata: Dict[str, Any]


@dataclass
class ScenarioSet:
    """场景集合"""

    scenarios: List[StressScenario]
    baseline_scenario: StressScenario
    worst_case_scenario: StressScenario
    expected_shortfall: float
    tail_risk_metrics: Dict[str, float]
    scenario_probabilities: pd.Series


class ScenarioGenerator:
    """场景生成器"""

    # 历史危机事件
    HISTORICAL_CRISES = {
        "black_monday_1987": {
            "date": "1987-10-19",
            "equity_shock": -0.22,
            "volatility_spike": 3.0,
            "correlation_increase": 0.3,
        },
        "asian_crisis_1997": {
            "date": "1997-07-02",
            "equity_shock": -0.15,
            "volatility_spike": 2.5,
            "correlation_increase": 0.25,
        },
        "dot_com_crash_2000": {
            "date": "2000-03-10",
            "equity_shock": -0.35,
            "volatility_spike": 2.0,
            "correlation_increase": 0.2,
        },
        "financial_crisis_2008": {
            "date": "2008-09-15",
            "equity_shock": -0.40,
            "volatility_spike": 4.0,
            "correlation_increase": 0.4,
        },
        "covid_crash_2020": {
            "date": "2020-03-16",
            "equity_shock": -0.30,
            "volatility_spike": 5.0,
            "correlation_increase": 0.35,
        },
    }

    def __init__(self, config: Optional[ScenarioConfig] = None):
        """初始化场景生成器

        Args:
            config: 场景配置
        """
        self.config = config or ScenarioConfig()
        if self.config.scenario_types is None:
            self.config.scenario_types = list(ScenarioType)
        if self.config.confidence_levels is None:
            self.config.confidence_levels = [0.95, 0.99, 0.999]
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

    def generate_stress_scenarios(
        self, base_returns: pd.DataFrame, factor_loadings: Optional[pd.DataFrame] = None
    ) -> ScenarioSet:
        """生成压力测试场景

        Args:
            base_returns: 基础收益率数据
            factor_loadings: 因子载荷矩阵

        Returns:
            场景集合
        """
        logger.info(f"Generating {self.config.n_scenarios} stress scenarios")

        scenarios = []

        # 生成不同类型的场景
        if ScenarioType.HISTORICAL in self.config.scenario_types:
            historical_scenarios = self._generate_historical_scenarios(base_returns)
            scenarios.extend(historical_scenarios)

        if ScenarioType.HYPOTHETICAL in self.config.scenario_types:
            hypothetical_scenarios = self._generate_hypothetical_scenarios(base_returns)
            scenarios.extend(hypothetical_scenarios)

        if ScenarioType.MONTE_CARLO in self.config.scenario_types:
            mc_scenarios = self._generate_monte_carlo_scenarios(base_returns)
            scenarios.extend(mc_scenarios)

        if (
            ScenarioType.FACTOR_SHOCK in self.config.scenario_types
            and factor_loadings is not None
        ):
            factor_scenarios = self._generate_factor_shock_scenarios(
                base_returns, factor_loadings
            )
            scenarios.extend(factor_scenarios)

        if ScenarioType.TAIL_RISK in self.config.scenario_types:
            tail_scenarios = self._generate_tail_risk_scenarios(base_returns)
            scenarios.extend(tail_scenarios)

        # 创建基线场景
        baseline_scenario = self._create_baseline_scenario(base_returns)

        # 找出最坏情况
        worst_case_scenario = self._identify_worst_case(scenarios)

        # 计算场景概率
        scenario_probs = self._calculate_scenario_probabilities(scenarios)

        # 计算尾部风险指标
        tail_metrics = self._calculate_tail_risk_metrics(scenarios)

        # 计算期望短缺
        expected_shortfall = self._calculate_expected_shortfall(scenarios)

        return ScenarioSet(
            scenarios=scenarios,
            baseline_scenario=baseline_scenario,
            worst_case_scenario=worst_case_scenario,
            expected_shortfall=expected_shortfall,
            tail_risk_metrics=tail_metrics,
            scenario_probabilities=scenario_probs,
        )

    def generate_historical_stress_test(
        self, base_returns: pd.DataFrame, crisis_name: str
    ) -> StressScenario:
        """生成历史压力测试

        Args:
            base_returns: 基础收益率数据
            crisis_name: 危机名称

        Returns:
            压力测试场景
        """
        if crisis_name not in self.HISTORICAL_CRISES:
            raise ValueError(f"Unknown crisis: {crisis_name}")

        crisis_params = self.HISTORICAL_CRISES[crisis_name]

        # 计算基础统计
        mean_returns = base_returns.mean()
        cov_matrix = base_returns.cov()
        corr_matrix = base_returns.corr()

        # 应用冲击
        shocked_returns = mean_returns * (1 + crisis_params["equity_shock"])

        # 调整波动率
        shocked_cov = cov_matrix * crisis_params["volatility_spike"]

        # 调整相关性
        shocked_corr = corr_matrix + crisis_params["correlation_increase"] * (
            1 - corr_matrix
        )
        shocked_corr = np.clip(shocked_corr, -1, 1)

        # 生成场景收益
        n_periods = self.config.time_horizon
        n_assets = len(base_returns.columns)

        # 使用调整后的参数生成收益
        scenario_returns = pd.DataFrame(
            np.random.multivariate_normal(shocked_returns, shocked_cov, n_periods),
            columns=base_returns.columns,
        )

        return StressScenario(
            scenario_id=f"historical_{crisis_name}",
            scenario_type=ScenarioType.HISTORICAL,
            description=f"Historical crisis: {crisis_name}",
            probability=self._estimate_crisis_probability(crisis_params),
            time_horizon=n_periods,
            asset_returns=scenario_returns,
            factor_shocks={"equity": crisis_params["equity_shock"]},
            correlation_matrix=pd.DataFrame(
                shocked_corr, index=base_returns.columns, columns=base_returns.columns
            ),
            volatility_multipliers=pd.Series(
                [crisis_params["volatility_spike"]] * n_assets,
                index=base_returns.columns,
            ),
            metadata=crisis_params,
        )

    def generate_monte_carlo_scenarios(
        self, base_returns: pd.DataFrame, n_simulations: int = 1000
    ) -> List[pd.DataFrame]:
        """生成蒙特卡洛场景

        Args:
            base_returns: 基础收益率数据
            n_simulations: 模拟次数

        Returns:
            场景列表
        """
        scenarios = []

        # 估计分布参数
        if self.config.include_fat_tails:
            # 使用t分布
            params = self._fit_t_distribution(base_returns)
            distribution = "t"
        else:
            # 使用正态分布
            params = {"mean": base_returns.mean(), "cov": base_returns.cov()}
            distribution = "normal"

        # 生成场景
        for i in range(n_simulations):
            if distribution == "t":
                scenario = self._generate_t_distributed_returns(
                    params, self.config.time_horizon, base_returns.columns
                )
            else:
                scenario = pd.DataFrame(
                    np.random.multivariate_normal(
                        params["mean"], params["cov"], self.config.time_horizon
                    ),
                    columns=base_returns.columns,
                )

            scenarios.append(scenario)

        return scenarios

    def _generate_historical_scenarios(
        self, base_returns: pd.DataFrame
    ) -> List[StressScenario]:
        """生成历史场景

        Args:
            base_returns: 基础收益率数据

        Returns:
            历史场景列表
        """
        scenarios = []

        for crisis_name in self.HISTORICAL_CRISES.keys():
            scenario = self.generate_historical_stress_test(base_returns, crisis_name)
            scenarios.append(scenario)

        return scenarios

    def _generate_hypothetical_scenarios(
        self, base_returns: pd.DataFrame
    ) -> List[StressScenario]:
        """生成假设场景

        Args:
            base_returns: 基础收益率数据

        Returns:
            假设场景列表
        """
        scenarios = []

        # 定义假设冲击
        hypothetical_shocks = [
            {
                "name": "interest_rate_spike",
                "description": "Sudden 200bps interest rate increase",
                "shocks": {"rates": 0.02, "equity": -0.15, "credit": -0.10},
            },
            {
                "name": "currency_crisis",
                "description": "Major currency devaluation",
                "shocks": {"fx": -0.30, "equity": -0.20, "commodity": 0.15},
            },
            {
                "name": "geopolitical_event",
                "description": "Major geopolitical crisis",
                "shocks": {"equity": -0.25, "commodity": 0.30, "gold": 0.20},
            },
            {
                "name": "tech_bubble_burst",
                "description": "Technology sector collapse",
                "shocks": {"tech": -0.50, "equity": -0.20, "value": 0.10},
            },
        ]

        for shock_def in hypothetical_shocks:
            scenario = self._create_hypothetical_scenario(base_returns, shock_def)
            scenarios.append(scenario)

        return scenarios

    def _generate_monte_carlo_scenarios(
        self, base_returns: pd.DataFrame
    ) -> List[StressScenario]:
        """生成蒙特卡洛场景

        Args:
            base_returns: 基础收益率数据

        Returns:
            蒙特卡洛场景列表
        """
        scenarios = []

        # 生成多个蒙特卡洛路径
        n_mc_scenarios = min(
            10, self.config.n_scenarios // len(self.config.scenario_types)
        )

        for i in range(n_mc_scenarios):
            # 生成一条路径
            scenario_returns = self._generate_single_mc_path(base_returns)

            scenario = StressScenario(
                scenario_id=f"monte_carlo_{i}",
                scenario_type=ScenarioType.MONTE_CARLO,
                description=f"Monte Carlo simulation #{i}",
                probability=1.0 / n_mc_scenarios,
                time_horizon=self.config.time_horizon,
                asset_returns=scenario_returns,
                factor_shocks=None,
                correlation_matrix=None,
                volatility_multipliers=None,
                metadata={"simulation_index": i},
            )

            scenarios.append(scenario)

        return scenarios

    def _generate_factor_shock_scenarios(
        self, base_returns: pd.DataFrame, factor_loadings: pd.DataFrame
    ) -> List[StressScenario]:
        """生成因子冲击场景

        Args:
            base_returns: 基础收益率数据
            factor_loadings: 因子载荷

        Returns:
            因子冲击场景列表
        """
        scenarios = []

        # 定义因子冲击
        factor_shocks = [
            {"market": -0.20, "size": 0.10, "value": 0.15},
            {"market": -0.30, "momentum": -0.25, "quality": 0.20},
            {"market": -0.15, "low_vol": 0.25, "dividend": 0.10},
        ]

        for i, shocks in enumerate(factor_shocks):
            # 计算资产收益
            asset_returns = pd.DataFrame(
                index=range(self.config.time_horizon), columns=base_returns.columns
            )

            for asset in base_returns.columns:
                # 计算因子贡献的收益
                factor_return = sum(
                    factor_loadings.loc[asset, factor] * shock
                    for factor, shock in shocks.items()
                    if factor in factor_loadings.columns
                )

                # 添加特质风险
                idio_risk = base_returns[asset].std() * np.random.randn(
                    self.config.time_horizon
                )

                asset_returns[asset] = factor_return + idio_risk

            scenario = StressScenario(
                scenario_id=f"factor_shock_{i}",
                scenario_type=ScenarioType.FACTOR_SHOCK,
                description=f"Factor shock scenario #{i}",
                probability=self._calculate_factor_shock_probability(shocks),
                time_horizon=self.config.time_horizon,
                asset_returns=asset_returns,
                factor_shocks=shocks,
                correlation_matrix=None,
                volatility_multipliers=None,
                metadata={"factor_shocks": shocks},
            )

            scenarios.append(scenario)

        return scenarios

    def _generate_tail_risk_scenarios(
        self, base_returns: pd.DataFrame
    ) -> List[StressScenario]:
        """生成尾部风险场景

        Args:
            base_returns: 基础收益率数据

        Returns:
            尾部风险场景列表
        """
        scenarios = []

        for confidence_level in self.config.confidence_levels:
            # 计算VaR水平的收益
            var_returns = base_returns.quantile(1 - confidence_level)

            # 生成尾部场景
            tail_returns = pd.DataFrame(
                index=range(self.config.time_horizon), columns=base_returns.columns
            )

            for asset in base_returns.columns:
                # 从尾部分布采样
                tail_data = base_returns[asset][
                    base_returns[asset] <= var_returns[asset]
                ]

                if len(tail_data) > 0:
                    tail_returns[asset] = np.random.choice(
                        tail_data, self.config.time_horizon, replace=True
                    )
                else:
                    # 如果没有足够的尾部数据，使用极值理论
                    tail_returns[asset] = self._generate_extreme_values(
                        base_returns[asset], confidence_level
                    )

            scenario = StressScenario(
                scenario_id=f"tail_risk_{int(confidence_level * 100)}",
                scenario_type=ScenarioType.TAIL_RISK,
                description=f"Tail risk at {confidence_level * 100}% confidence",
                probability=1 - confidence_level,
                time_horizon=self.config.time_horizon,
                asset_returns=tail_returns,
                factor_shocks=None,
                correlation_matrix=None,
                volatility_multipliers=None,
                metadata={"confidence_level": confidence_level},
            )

            scenarios.append(scenario)

        return scenarios

    def _create_baseline_scenario(self, base_returns: pd.DataFrame) -> StressScenario:
        """创建基线场景

        Args:
            base_returns: 基础收益率数据

        Returns:
            基线场景
        """
        # 使用历史平均
        mean_returns = base_returns.mean()
        cov_matrix = base_returns.cov()

        # 生成基线路径
        baseline_returns = pd.DataFrame(
            np.random.multivariate_normal(
                mean_returns, cov_matrix, self.config.time_horizon
            ),
            columns=base_returns.columns,
        )

        return StressScenario(
            scenario_id="baseline",
            scenario_type=ScenarioType.HISTORICAL,
            description="Baseline scenario based on historical average",
            probability=0.5,  # 中性概率
            time_horizon=self.config.time_horizon,
            asset_returns=baseline_returns,
            factor_shocks=None,
            correlation_matrix=base_returns.corr(),
            volatility_multipliers=pd.Series(1.0, index=base_returns.columns),
            metadata={"type": "baseline"},
        )

    def _create_hypothetical_scenario(
        self, base_returns: pd.DataFrame, shock_definition: Dict[str, Any]
    ) -> StressScenario:
        """创建假设场景

        Args:
            base_returns: 基础收益率数据
            shock_definition: 冲击定义

        Returns:
            假设场景
        """
        # 应用冲击到基础收益
        scenario_returns = base_returns.copy()

        # 简单的冲击传导模型
        for asset in scenario_returns.columns:
            base_shock = shock_definition["shocks"].get("equity", 0)

            # 添加资产特定冲击
            if "tech" in asset.lower() and "tech" in shock_definition["shocks"]:
                base_shock = shock_definition["shocks"]["tech"]

            # 应用冲击
            shocked_mean = scenario_returns[asset].mean() * (1 + base_shock)
            shocked_std = scenario_returns[asset].std() * (1 + abs(base_shock))

            # 生成冲击后的收益
            scenario_returns[asset] = np.random.normal(
                shocked_mean, shocked_std, len(scenario_returns)
            )

        return StressScenario(
            scenario_id=f"hypothetical_{shock_definition['name']}",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description=shock_definition["description"],
            probability=self._estimate_hypothetical_probability(shock_definition),
            time_horizon=len(scenario_returns),
            asset_returns=scenario_returns.head(self.config.time_horizon),
            factor_shocks=shock_definition["shocks"],
            correlation_matrix=None,
            volatility_multipliers=None,
            metadata=shock_definition,
        )

    def _identify_worst_case(self, scenarios: List[StressScenario]) -> StressScenario:
        """识别最坏情况

        Args:
            scenarios: 场景列表

        Returns:
            最坏场景
        """
        worst_loss = float("inf")
        worst_scenario = None

        for scenario in scenarios:
            # 计算累计收益
            cumulative_return = (1 + scenario.asset_returns).prod() - 1
            portfolio_return = cumulative_return.mean()  # 简单平均

            if portfolio_return < worst_loss:
                worst_loss = portfolio_return
                worst_scenario = scenario

        return worst_scenario

    def _calculate_scenario_probabilities(
        self, scenarios: List[StressScenario]
    ) -> pd.Series:
        """计算场景概率

        Args:
            scenarios: 场景列表

        Returns:
            概率Series
        """
        probs = {}

        for scenario in scenarios:
            probs[scenario.scenario_id] = scenario.probability

        # 标准化概率
        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {k: v / total_prob for k, v in probs.items()}

        return pd.Series(probs)

    def _calculate_tail_risk_metrics(
        self, scenarios: List[StressScenario]
    ) -> Dict[str, float]:
        """计算尾部风险指标

        Args:
            scenarios: 场景列表

        Returns:
            尾部风险指标
        """
        metrics = {}

        # 收集所有场景的收益
        all_returns = []
        for scenario in scenarios:
            portfolio_return = scenario.asset_returns.mean(axis=1)  # 简单组合
            all_returns.extend(portfolio_return.values)

        all_returns = np.array(all_returns)

        # 计算尾部指标
        metrics["skewness"] = stats.skew(all_returns)
        metrics["kurtosis"] = stats.kurtosis(all_returns)
        metrics["var_95"] = np.percentile(all_returns, 5)
        metrics["var_99"] = np.percentile(all_returns, 1)
        metrics["expected_shortfall_95"] = all_returns[
            all_returns <= metrics["var_95"]
        ].mean()

        return metrics

    def _calculate_expected_shortfall(self, scenarios: List[StressScenario]) -> float:
        """计算期望短缺

        Args:
            scenarios: 场景列表

        Returns:
            期望短缺
        """
        # 收集加权收益
        weighted_returns = []

        for scenario in scenarios:
            portfolio_return = scenario.asset_returns.mean(axis=1).mean()
            weighted_returns.append(portfolio_return * scenario.probability)

        # 只考虑损失场景
        losses = [r for r in weighted_returns if r < 0]

        if losses:
            return np.mean(losses)
        else:
            return 0.0

    def _estimate_crisis_probability(self, crisis_params: Dict[str, Any]) -> float:
        """估计危机概率

        Args:
            crisis_params: 危机参数

        Returns:
            概率估计
        """
        # 基于冲击严重程度估计概率
        severity = abs(crisis_params.get("equity_shock", 0))

        if severity > 0.3:
            return 0.01  # 1%概率
        elif severity > 0.2:
            return 0.05  # 5%概率
        else:
            return 0.10  # 10%概率

    def _estimate_hypothetical_probability(
        self, shock_definition: Dict[str, Any]
    ) -> float:
        """估计假设场景概率

        Args:
            shock_definition: 冲击定义

        Returns:
            概率估计
        """
        # 简单估计
        return 0.05  # 默认5%

    def _calculate_factor_shock_probability(self, shocks: Dict[str, float]) -> float:
        """计算因子冲击概率

        Args:
            shocks: 因子冲击

        Returns:
            概率
        """
        # 基于冲击大小估计
        avg_shock = np.mean([abs(v) for v in shocks.values()])

        if avg_shock > 0.25:
            return 0.02
        elif avg_shock > 0.15:
            return 0.05
        else:
            return 0.10

    def _fit_t_distribution(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """拟合t分布

        Args:
            returns: 收益率数据

        Returns:
            分布参数
        """
        from scipy import stats

        params = {}

        # 对每个资产拟合t分布
        for col in returns.columns:
            data = returns[col].dropna()

            # 拟合t分布
            df, loc, scale = stats.t.fit(data)

            params[col] = {"df": df, "loc": loc, "scale": scale}

        # 估计相关性结构
        params["correlation"] = returns.corr()

        return params

    def _generate_t_distributed_returns(
        self, params: Dict[str, Any], n_periods: int, columns: pd.Index
    ) -> pd.DataFrame:
        """生成t分布收益

        Args:
            params: 分布参数
            n_periods: 时期数
            columns: 列名

        Returns:
            收益DataFrame
        """
        from scipy import stats

        returns = pd.DataFrame(index=range(n_periods), columns=columns)

        for col in columns:
            if col in params:
                col_params = params[col]
                returns[col] = stats.t.rvs(
                    df=col_params["df"],
                    loc=col_params["loc"],
                    scale=col_params["scale"],
                    size=n_periods,
                )

        return returns

    def _generate_single_mc_path(self, base_returns: pd.DataFrame) -> pd.DataFrame:
        """生成单条蒙特卡洛路径

        Args:
            base_returns: 基础收益率

        Returns:
            模拟路径
        """
        mean_returns = base_returns.mean()
        cov_matrix = base_returns.cov()

        # 可选：添加随机波动率
        if self.config.include_volatility_clusters:
            volatility_multiplier = np.random.gamma(2, 0.5)
            cov_matrix *= volatility_multiplier

        # 生成路径
        path = pd.DataFrame(
            np.random.multivariate_normal(
                mean_returns, cov_matrix, self.config.time_horizon
            ),
            columns=base_returns.columns,
        )

        return path

    def _generate_extreme_values(
        self, data: pd.Series, confidence_level: float
    ) -> np.ndarray:
        """生成极值

        Args:
            data: 数据序列
            confidence_level: 置信水平

        Returns:
            极值数组
        """
        # 使用广义极值分布
        from scipy import stats

        # 拟合GEV分布到尾部数据
        threshold = data.quantile(0.1)  # 使用10%分位数作为阈值
        tail_data = data[data <= threshold]

        if len(tail_data) > 10:
            # 拟合GEV
            c, loc, scale = stats.genextreme.fit(tail_data)

            # 生成极值
            extreme_values = stats.genextreme.rvs(
                c, loc, scale, size=self.config.time_horizon
            )
        else:
            # 使用简单的尾部采样
            extreme_values = np.random.choice(
                data.values, self.config.time_horizon, replace=True
            ) * (1 + np.random.randn(self.config.time_horizon) * 0.1)

        return extreme_values


# 模块级别函数
def generate_stress_scenarios(
    base_returns: pd.DataFrame, config: Optional[ScenarioConfig] = None
) -> ScenarioSet:
    """生成压力测试场景的便捷函数

    Args:
        base_returns: 基础收益率数据
        config: 场景配置

    Returns:
        场景集合
    """
    generator = ScenarioGenerator(config)
    return generator.generate_stress_scenarios(base_returns)
