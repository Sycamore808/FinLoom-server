"""
贝叶斯优化器模块
使用高斯过程进行超参数优化
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

warnings.filterwarnings("ignore")

from common.logging_system import setup_logger
from module_07_optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationStatus,
    Parameter,
    Trial,
)

logger = setup_logger("bayesian_optimizer")


class BayesianOptimizer(BaseOptimizer):
    """贝叶斯优化器

    使用高斯过程作为代理模型，通过采集函数选择下一个采样点
    """

    def __init__(
        self,
        parameter_space: List[Parameter],
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = False,
        n_trials: int = 100,
        n_initial_points: int = 10,
        acquisition_function: str = "ei",  # 'ei', 'pi', 'ucb'
        xi: float = 0.01,  # EI和PI的探索参数
        kappa: float = 2.576,  # UCB的探索参数
        random_state: int = 42,
    ):
        """初始化贝叶斯优化器

        Args:
            parameter_space: 参数空间
            objective_function: 目标函数
            maximize: 是否最大化
            n_trials: 总试验次数
            n_initial_points: 初始随机采样点数
            acquisition_function: 采集函数类型
            xi: 探索参数（EI/PI）
            kappa: 探索参数（UCB）
            random_state: 随机种子
        """
        super().__init__(
            parameter_space, objective_function, maximize, n_trials, random_state
        )

        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        self.xi = xi
        self.kappa = kappa

        # 高斯过程回归器
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state,
        )

        # 存储观测数据
        self.X_observed = []
        self.y_observed = []

        # 参数边界和类型
        self._setup_parameter_bounds()

    def _setup_parameter_bounds(self) -> None:
        """设置参数边界"""
        self.bounds = []
        self.param_types = []
        self.param_names = []
        self.categorical_mappings = {}

        for param in self.parameter_space:
            self.param_names.append(param.name)
            self.param_types.append(param.param_type)

            if param.param_type in ["float", "int"]:
                if param.log_scale:
                    self.bounds.append((np.log(param.low), np.log(param.high)))
                else:
                    self.bounds.append((param.low, param.high))
            elif param.param_type in ["categorical", "bool"]:
                # 将分类变量映射到连续空间
                self.categorical_mappings[param.name] = param.choices
                self.bounds.append((0, len(param.choices) - 1))

        self.bounds = np.array(self.bounds)

    def suggest_parameters(self) -> Dict[str, Any]:
        """建议下一组参数

        Returns:
            参数字典
        """
        # 初始阶段使用随机采样
        if len(self.X_observed) < self.n_initial_points:
            x = self._random_sample()
        else:
            # 使用采集函数选择下一个点
            x = self._optimize_acquisition()

        # 转换为参数字典
        params = self._vector_to_params(x)
        return params

    def _random_sample(self) -> np.ndarray:
        """随机采样一个点

        Returns:
            参数向量
        """
        x = np.zeros(len(self.bounds))
        for i, (low, high) in enumerate(self.bounds):
            x[i] = self.random_state.uniform(low, high)
        return x

    def _vector_to_params(self, x: np.ndarray) -> Dict[str, Any]:
        """将向量转换为参数字典

        Args:
            x: 参数向量

        Returns:
            参数字典
        """
        params = {}
        for i, (name, param_type) in enumerate(zip(self.param_names, self.param_types)):
            param = self.get_parameter_by_name(name)

            if param_type == "float":
                if param.log_scale:
                    params[name] = np.exp(x[i])
                else:
                    params[name] = float(x[i])
            elif param_type == "int":
                params[name] = int(np.round(x[i]))
            elif param_type in ["categorical", "bool"]:
                idx = int(np.round(x[i]))
                idx = np.clip(idx, 0, len(self.categorical_mappings[name]) - 1)
                params[name] = self.categorical_mappings[name][idx]

        return params

    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """将参数字典转换为向量

        Args:
            params: 参数字典

        Returns:
            参数向量
        """
        x = np.zeros(len(self.param_names))
        for i, name in enumerate(self.param_names):
            param = self.get_parameter_by_name(name)
            value = params[name]

            if param.param_type == "float":
                if param.log_scale:
                    x[i] = np.log(value)
                else:
                    x[i] = value
            elif param.param_type == "int":
                x[i] = value
            elif param.param_type in ["categorical", "bool"]:
                x[i] = self.categorical_mappings[name].index(value)

        return x

    def _optimize_acquisition(self) -> np.ndarray:
        """优化采集函数找到下一个采样点

        Returns:
            最优参数向量
        """

        # 定义负采集函数（用于最小化）
        def neg_acquisition(x):
            return -self._acquisition(x.reshape(1, -1))

        # 多次随机初始化优化
        best_x = None
        best_acquisition = float("inf")

        for _ in range(10):
            # 随机初始点
            x0 = self._random_sample()

            # 优化采集函数
            result = minimize(
                neg_acquisition, x0, bounds=self.bounds, method="L-BFGS-B"
            )

            if result.fun < best_acquisition:
                best_acquisition = result.fun
                best_x = result.x

        return best_x

    def _acquisition(self, x: np.ndarray) -> float:
        """计算采集函数值

        Args:
            x: 参数向量

        Returns:
            采集函数值
        """
        if self.acquisition_function == "ei":
            return self._expected_improvement(x)
        elif self.acquisition_function == "pi":
            return self._probability_improvement(x)
        elif self.acquisition_function == "ucb":
            return self._upper_confidence_bound(x)
        else:
            raise ValueError(
                f"Unknown acquisition function: {self.acquisition_function}"
            )

    def _expected_improvement(self, x: np.ndarray) -> float:
        """计算期望改进（EI）

        Args:
            x: 参数向量

        Returns:
            EI值
        """
        mu, sigma = self.gp.predict(x, return_std=True)
        mu = mu.flatten()
        sigma = sigma.flatten()

        # 当前最佳值
        if self.maximize:
            y_best = np.max(self.y_observed)
            improvement = mu - y_best - self.xi
        else:
            y_best = np.min(self.y_observed)
            improvement = y_best - mu - self.xi

        # 计算EI
        with np.errstate(divide="warn"):
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei[0]

    def _probability_improvement(self, x: np.ndarray) -> float:
        """计算改进概率（PI）

        Args:
            x: 参数向量

        Returns:
            PI值
        """
        mu, sigma = self.gp.predict(x, return_std=True)
        mu = mu.flatten()
        sigma = sigma.flatten()

        # 当前最佳值
        if self.maximize:
            y_best = np.max(self.y_observed)
            Z = (mu - y_best - self.xi) / sigma
        else:
            y_best = np.min(self.y_observed)
            Z = (y_best - mu - self.xi) / sigma

        pi = norm.cdf(Z)
        return pi[0]

    def _upper_confidence_bound(self, x: np.ndarray) -> float:
        """计算置信上界（UCB）

        Args:
            x: 参数向量

        Returns:
            UCB值
        """
        mu, sigma = self.gp.predict(x, return_std=True)
        mu = mu.flatten()
        sigma = sigma.flatten()

        if self.maximize:
            ucb = mu + self.kappa * sigma
        else:
            ucb = -(mu - self.kappa * sigma)  # 取负值用于最小化

        return ucb[0]

    def _update_optimization_state(self, trial: Trial) -> None:
        """更新优化器状态

        Args:
            trial: 完成的试验
        """
        if (
            trial.status == OptimizationStatus.COMPLETED
            and trial.objective_value is not None
        ):
            # 将参数转换为向量
            x = self._params_to_vector(trial.parameters)
            y = trial.objective_value

            # 如果是最大化问题，已经在基类中处理了符号
            # 这里直接存储
            self.X_observed.append(x)
            self.y_observed.append(y)

            # 更新高斯过程
            if len(self.X_observed) >= self.n_initial_points:
                X = np.array(self.X_observed)
                y = np.array(self.y_observed)

                try:
                    self.gp.fit(X, y)
                    logger.debug(f"GP model updated with {len(X)} observations")
                except Exception as e:
                    logger.warning(f"Failed to update GP model: {e}")
