"""
随机搜索优化器模块
随机采样参数空间进行优化
"""

from typing import Any, Callable, Dict, List

import numpy as np

from common.logging_system import setup_logger
from module_07_optimization.base_optimizer import (
    BaseOptimizer,
    Parameter,
    Trial,
)

logger = setup_logger("random_search")


class RandomSearchOptimizer(BaseOptimizer):
    """随机搜索优化器

    随机采样参数空间，适合高维参数空间的初步探索
    """

    def __init__(
        self,
        parameter_space: List[Parameter],
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = False,
        n_trials: int = 100,
        random_state: int = 42,
    ):
        """初始化随机搜索优化器

        Args:
            parameter_space: 参数空间
            objective_function: 目标函数
            maximize: 是否最大化
            n_trials: 试验次数
            random_state: 随机种子
        """
        super().__init__(
            parameter_space, objective_function, maximize, n_trials, random_state
        )

        logger.info(f"Initialized RandomSearch with {n_trials} trials")

    def suggest_parameters(self) -> Dict[str, Any]:
        """建议下一组参数

        Returns:
            参数字典
        """
        params = {}

        for param in self.parameter_space:
            params[param.name] = param.sample(self.random_state)

        return params

    def _update_optimization_state(self, trial: Trial) -> None:
        """更新优化器状态

        Args:
            trial: 完成的试验
        """
        # 随机搜索不需要更新内部状态
        pass

    def get_best_trials(self, n: int = 10) -> List[Trial]:
        """获取最佳的N个试验

        Args:
            n: 试验数量

        Returns:
            最佳试验列表
        """
        completed_trials = [t for t in self.trials if t.objective_value is not None]

        if not completed_trials:
            return []

        # 排序
        sorted_trials = sorted(
            completed_trials,
            key=lambda t: t.objective_value,
            reverse=self.maximize,
        )

        return sorted_trials[:n]
