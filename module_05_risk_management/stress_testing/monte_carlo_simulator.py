"""
蒙特卡洛模拟器模块
实现投资组合风险的蒙特卡洛模拟
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
from scipy import stats

logger = setup_logger("monte_carlo_simulator")


class DistributionType(Enum):
    """分布类型枚举"""

    NORMAL = "normal"
    T_DISTRIBUTION = "t_distribution"
    HISTORICAL = "historical"
    MIXTURE = "mixture"
    JUMP_DIFFUSION = "jump_diffusion"
    REGIME_SWITCHING = "regime_switching"


@dataclass
class SimulationConfig:
    """模拟配置"""

    n_simulations: int = 10000
    time_horizon: int = 252  # 交易日
    distribution_type: DistributionType = DistributionType.NORMAL
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    use_antithetic: bool = True  # 对偶变量法
    use_control_variates: bool = False  # 控制变量法
    random_seed: Optional[int] = None
    parallel_processing: bool = True
    n_jobs: int = -1  # -1表示使用所有CPU


@dataclass
class SimulationPath:
    """模拟路径"""

    path_id: int
    returns: np.ndarray
    prices: np.ndarray
    cumulative_return: float
    max_drawdown: float
    volatility: float
    final_value: float


@dataclass
class MonteCarloResult:
    """蒙特卡洛结果"""

    simulation_paths: List[SimulationPath]
    expected_return: float
    expected_volatility: float
    var_estimates: Dict[float, float]
    cvar_estimates: Dict[float, float]
    probability_of_loss: float
    probability_of_target: Dict[float, float]
    percentile_outcomes: pd.DataFrame
    convergence_data: pd.DataFrame


class MonteCarloSimulator:
    """蒙特卡洛模拟器类"""

    def __init__(self, config: Optional[SimulationConfig] = None):
        """初始化蒙特卡洛模拟器

        Args:
            config: 模拟配置
        """
        self.config = config or SimulationConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        self.simulation_cache: Dict[str, Any] = {}

    def simulate_portfolio_returns(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        weights: pd.Series,
        initial_value: float = 100000,
    ) -> MonteCarloResult:
        """模拟投资组合收益

        Args:
            expected_returns: 期望收益率
            covariance_matrix: 协方差矩阵
            weights: 投资组合权重
            initial_value: 初始价值

        Returns:
            蒙特卡洛模拟结果
        """
        logger.info(
            f"Starting Monte Carlo simulation with {self.config.n_simulations} paths"
        )

        # 计算投资组合参数
        portfolio_return = weights @ expected_returns
        portfolio_variance = weights @ covariance_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)

        # 生成模拟路径
        if self.config.distribution_type == DistributionType.NORMAL:
            paths = self._simulate_normal_paths(
                portfolio_return, portfolio_vol, initial_value
            )
        elif self.config.distribution_type == DistributionType.T_DISTRIBUTION:
            paths = self._simulate_t_distribution_paths(
                portfolio_return, portfolio_vol, initial_value
            )
        elif self.config.distribution_type == DistributionType.HISTORICAL:
            paths = self._simulate_historical_bootstrap(
                expected_returns, weights, initial_value
            )
        elif self.config.distribution_type == DistributionType.JUMP_DIFFUSION:
            paths = self._simulate_jump_diffusion_paths(
                portfolio_return, portfolio_vol, initial_value
            )
        else:
            paths = self._simulate_normal_paths(
                portfolio_return, portfolio_vol, initial_value
            )

        # 计算风险指标
        var_estimates = self._calculate_var(paths)
        cvar_estimates = self._calculate_cvar(paths)
        prob_loss = self._calculate_probability_of_loss(paths)
        prob_target = self._calculate_probability_of_target(paths, initial_value)

        # 计算百分位结果
        percentile_outcomes = self._calculate_percentile_outcomes(paths)

        # 计算收敛数据
        convergence_data = self._calculate_convergence_data(paths)

        # 汇总统计
        all_final_values = [p.final_value for p in paths]
        expected_return_sim = np.mean([p.cumulative_return for p in paths])
        expected_vol_sim = np.std([p.cumulative_return for p in paths])

        result = MonteCarloResult(
            simulation_paths=paths[:100],  # 只保存前100条路径
            expected_return=expected_return_sim,
            expected_volatility=expected_vol_sim,
            var_estimates=var_estimates,
            cvar_estimates=cvar_estimates,
            probability_of_loss=prob_loss,
            probability_of_target=prob_target,
            percentile_outcomes=percentile_outcomes,
            convergence_data=convergence_data,
        )

        logger.info(f"Simulation completed. Expected return: {expected_return_sim:.2%}")

        return result

    def simulate_var_scenarios(
        self,
        portfolio_value: float,
        returns_data: pd.DataFrame,
        weights: pd.Series,
        n_days: int = 10,
    ) -> Dict[str, float]:
        """模拟VaR场景

        Args:
            portfolio_value: 组合价值
            returns_data: 历史收益率数据
            weights: 投资组合权重
            n_days: 时间范围

        Returns:
            VaR估计字典
        """
        logger.info(f"Simulating VaR scenarios for {n_days} days")

        # 计算组合收益率
        portfolio_returns = returns_data @ weights

        # 估计分布参数
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()

        # 模拟未来收益
        simulated_returns = np.random.normal(
            mean_return * n_days,
            std_return * np.sqrt(n_days),
            self.config.n_simulations,
        )

        # 计算潜在损失
        potential_losses = -portfolio_value * simulated_returns

        # 计算VaR
        var_estimates = {}
        for confidence in self.config.confidence_levels:
            var_estimates[confidence] = np.percentile(
                potential_losses, confidence * 100
            )

        return var_estimates

    def bootstrap_confidence_intervals(
        self,
        data: pd.Series,
        statistic_func: callable,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Bootstrap置信区间估计

        Args:
            data: 数据序列
            statistic_func: 统计函数
            n_bootstrap: Bootstrap次数
            confidence_level: 置信水平

        Returns:
            (点估计, 下界, 上界)
        """
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            # 重采样
            sample = data.sample(n=len(data), replace=True)
            stat = statistic_func(sample)
            bootstrap_stats.append(stat)

        # 计算置信区间
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        point_estimate = np.mean(bootstrap_stats)

        return point_estimate, lower, upper

    def importance_sampling_simulation(
        self,
        target_function: callable,
        proposal_params: Dict[str, float],
        target_params: Dict[str, float],
        n_samples: int = 10000,
    ) -> Tuple[float, float]:
        """重要性采样模拟

        Args:
            target_function: 目标函数
            proposal_params: 提议分布参数
            target_params: 目标分布参数
            n_samples: 样本数

        Returns:
            (估计值, 标准误差)
        """
        # 从提议分布采样
        samples = np.random.normal(
            proposal_params["mean"], proposal_params["std"], n_samples
        )

        # 计算重要性权重
        proposal_pdf = stats.norm.pdf(
            samples, proposal_params["mean"], proposal_params["std"]
        )

        target_pdf = stats.norm.pdf(
            samples, target_params["mean"], target_params["std"]
        )

        weights = target_pdf / proposal_pdf

        # 计算加权平均
        function_values = target_function(samples)
        estimate = np.mean(weights * function_values)

        # 计算标准误差
        variance = np.var(weights * function_values)
        std_error = np.sqrt(variance / n_samples)

        return estimate, std_error

    def _simulate_normal_paths(
        self, mean_return: float, volatility: float, initial_value: float
    ) -> List[SimulationPath]:
        """模拟正态分布路径

        Args:
            mean_return: 平均收益率
            volatility: 波动率
            initial_value: 初始价值

        Returns:
            模拟路径列表
        """
        paths = []

        # 日收益率参数
        daily_return = mean_return / TRADING_DAYS_PER_YEAR
        daily_vol = volatility / np.sqrt(TRADING_DAYS_PER_YEAR)

        n_paths = self.config.n_simulations
        if self.config.use_antithetic:
            n_paths = n_paths // 2

        for i in range(n_paths):
            # 生成随机收益率
            random_returns = np.random.normal(
                daily_return, daily_vol, self.config.time_horizon
            )

            # 计算价格路径
            price_path = initial_value * np.exp(np.cumsum(random_returns))
            price_path = np.insert(price_path, 0, initial_value)

            # 创建路径对象
            path = self._create_simulation_path(i, random_returns, price_path)
            paths.append(path)

            # 对偶变量法
            if self.config.use_antithetic:
                anti_returns = 2 * daily_return - random_returns
                anti_price_path = initial_value * np.exp(np.cumsum(anti_returns))
                anti_price_path = np.insert(anti_price_path, 0, initial_value)

                anti_path = self._create_simulation_path(
                    i + n_paths, anti_returns, anti_price_path
                )
                paths.append(anti_path)

        return paths

    def _simulate_t_distribution_paths(
        self, mean_return: float, volatility: float, initial_value: float, df: int = 5
    ) -> List[SimulationPath]:
        """模拟t分布路径

        Args:
            mean_return: 平均收益率
            volatility: 波动率
            initial_value: 初始价值
            df: 自由度

        Returns:
            模拟路径列表
        """
        paths = []

        daily_return = mean_return / TRADING_DAYS_PER_YEAR
        daily_vol = volatility / np.sqrt(TRADING_DAYS_PER_YEAR)

        for i in range(self.config.n_simulations):
            # 生成t分布随机数
            t_random = stats.t.rvs(df, size=self.config.time_horizon)

            # 标准化并调整
            random_returns = daily_return + daily_vol * t_random / np.sqrt(
                df / (df - 2)
            )

            # 计算价格路径
            price_path = initial_value * np.exp(np.cumsum(random_returns))
            price_path = np.insert(price_path, 0, initial_value)

            path = self._create_simulation_path(i, random_returns, price_path)
            paths.append(path)

        return paths

    def _simulate_historical_bootstrap(
        self, returns_data: pd.Series, weights: pd.Series, initial_value: float
    ) -> List[SimulationPath]:
        """历史Bootstrap模拟

        Args:
            returns_data: 历史收益率
            weights: 权重
            initial_value: 初始价值

        Returns:
            模拟路径列表
        """
        paths = []

        # 计算历史组合收益率
        if isinstance(returns_data, pd.DataFrame):
            historical_returns = (returns_data @ weights).values
        else:
            historical_returns = returns_data.values

        for i in range(self.config.n_simulations):
            # Bootstrap采样
            random_returns = np.random.choice(
                historical_returns, size=self.config.time_horizon, replace=True
            )

            # 计算价格路径
            price_path = initial_value * np.exp(np.cumsum(random_returns))
            price_path = np.insert(price_path, 0, initial_value)

            path = self._create_simulation_path(i, random_returns, price_path)
            paths.append(path)

        return paths

    def _simulate_jump_diffusion_paths(
        self,
        mean_return: float,
        volatility: float,
        initial_value: float,
        jump_intensity: float = 0.1,
        jump_mean: float = -0.02,
        jump_std: float = 0.03,
    ) -> List[SimulationPath]:
        """模拟跳跃扩散路径

        Args:
            mean_return: 平均收益率
            volatility: 波动率
            initial_value: 初始价值
            jump_intensity: 跳跃强度
            jump_mean: 跳跃均值
            jump_std: 跳跃标准差

        Returns:
            模拟路径列表
        """
        paths = []

        daily_return = mean_return / TRADING_DAYS_PER_YEAR
        daily_vol = volatility / np.sqrt(TRADING_DAYS_PER_YEAR)
        dt = 1 / TRADING_DAYS_PER_YEAR

        for i in range(self.config.n_simulations):
            returns = []

            for t in range(self.config.time_horizon):
                # 布朗运动部分
                diffusion = (
                    daily_return * dt + daily_vol * np.sqrt(dt) * np.random.normal()
                )

                # 跳跃部分
                n_jumps = np.random.poisson(jump_intensity * dt)
                if n_jumps > 0:
                    jumps = np.sum(np.random.normal(jump_mean, jump_std, n_jumps))
                else:
                    jumps = 0

                returns.append(diffusion + jumps)

            returns = np.array(returns)

            # 计算价格路径
            price_path = initial_value * np.exp(np.cumsum(returns))
            price_path = np.insert(price_path, 0, initial_value)

            path = self._create_simulation_path(i, returns, price_path)
            paths.append(path)

        return paths

    def _create_simulation_path(
        self, path_id: int, returns: np.ndarray, prices: np.ndarray
    ) -> SimulationPath:
        """创建模拟路径对象

        Args:
            path_id: 路径ID
            returns: 收益率数组
            prices: 价格数组

        Returns:
            模拟路径对象
        """
        cumulative_return = (prices[-1] / prices[0]) - 1
        max_drawdown = self._calculate_max_drawdown(prices)
        volatility = np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)

        return SimulationPath(
            path_id=path_id,
            returns=returns,
            prices=prices,
            cumulative_return=cumulative_return,
            max_drawdown=max_drawdown,
            volatility=volatility,
            final_value=prices[-1],
        )

    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """计算最大回撤

        Args:
            prices: 价格数组

        Returns:
            最大回撤
        """
        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax
        return np.min(drawdown)

    def _calculate_var(self, paths: List[SimulationPath]) -> Dict[float, float]:
        """计算VaR

        Args:
            paths: 模拟路径列表

        Returns:
            VaR估计字典
        """
        final_returns = [p.cumulative_return for p in paths]

        var_estimates = {}
        for confidence in self.config.confidence_levels:
            var_estimates[confidence] = -np.percentile(
                final_returns, (1 - confidence) * 100
            )

        return var_estimates

    def _calculate_cvar(self, paths: List[SimulationPath]) -> Dict[float, float]:
        """计算CVaR

        Args:
            paths: 模拟路径列表

        Returns:
            CVaR估计字典
        """
        final_returns = [p.cumulative_return for p in paths]

        cvar_estimates = {}
        for confidence in self.config.confidence_levels:
            var_threshold = np.percentile(final_returns, (1 - confidence) * 100)
            tail_returns = [r for r in final_returns if r <= var_threshold]

            if tail_returns:
                cvar_estimates[confidence] = -np.mean(tail_returns)
            else:
                cvar_estimates[confidence] = 0

        return cvar_estimates

    def _calculate_probability_of_loss(self, paths: List[SimulationPath]) -> float:
        """计算亏损概率

        Args:
            paths: 模拟路径列表

        Returns:
            亏损概率
        """
        losses = sum(1 for p in paths if p.cumulative_return < 0)
        return losses / len(paths)

    def _calculate_probability_of_target(
        self, paths: List[SimulationPath], initial_value: float
    ) -> Dict[float, float]:
        """计算达到目标的概率

        Args:
            paths: 模拟路径列表
            initial_value: 初始价值

        Returns:
            概率字典
        """
        targets = [0.05, 0.10, 0.20, 0.50]  # 5%, 10%, 20%, 50%收益目标
        probabilities = {}

        for target in targets:
            hits = sum(1 for p in paths if p.cumulative_return >= target)
            probabilities[target] = hits / len(paths)

        return probabilities

    def _calculate_percentile_outcomes(
        self, paths: List[SimulationPath]
    ) -> pd.DataFrame:
        """计算百分位结果

        Args:
            paths: 模拟路径列表

        Returns:
            百分位结果DataFrame
        """
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

        final_values = [p.final_value for p in paths]
        cumulative_returns = [p.cumulative_return for p in paths]
        max_drawdowns = [p.max_drawdown for p in paths]

        results = pd.DataFrame(index=percentiles)
        results["final_value"] = [np.percentile(final_values, p) for p in percentiles]
        results["cumulative_return"] = [
            np.percentile(cumulative_returns, p) for p in percentiles
        ]
        results["max_drawdown"] = [np.percentile(max_drawdowns, p) for p in percentiles]

        return results

    def _calculate_convergence_data(self, paths: List[SimulationPath]) -> pd.DataFrame:
        """计算收敛数据

        Args:
            paths: 模拟路径列表

        Returns:
            收敛数据DataFrame
        """
        n_checkpoints = min(20, len(paths) // 100)
        checkpoints = np.linspace(100, len(paths), n_checkpoints, dtype=int)

        convergence_data = []

        for n in checkpoints:
            subset_paths = paths[:n]

            mean_return = np.mean([p.cumulative_return for p in subset_paths])
            std_return = np.std([p.cumulative_return for p in subset_paths])

            convergence_data.append(
                {
                    "n_simulations": n,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "std_error": std_return / np.sqrt(n),
                }
            )

        return pd.DataFrame(convergence_data)


# 模块级别函数
def run_monte_carlo_simulation(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    weights: pd.Series,
    config: Optional[SimulationConfig] = None,
) -> Dict[str, float]:
    """运行蒙特卡洛模拟的便捷函数

    Args:
        expected_returns: 期望收益率
        covariance_matrix: 协方差矩阵
        weights: 权重
        config: 配置

    Returns:
        风险指标字典
    """
    simulator = MonteCarloSimulator(config)
    result = simulator.simulate_portfolio_returns(
        expected_returns, covariance_matrix, weights
    )

    return {
        "expected_return": result.expected_return,
        "expected_volatility": result.expected_volatility,
        "var_95": result.var_estimates.get(0.95, 0),
        "cvar_95": result.cvar_estimates.get(0.95, 0),
        "probability_of_loss": result.probability_of_loss,
    }
