"""
网格搜索优化器模块
对参数空间进行全面网格搜索
"""

from typing import Any, Callable, Dict, List

import numpy as np

from common.logging_system import setup_logger
from module_07_optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationStatus,
    Parameter,
)

logger = setup_logger("grid_search")


class GridSearchOptimizer(BaseOptimizer):
    """网格搜索优化器

    对参数空间进行穷举搜索，适合参数空间较小的情况
    """

    def __init__(
        self,
        parameter_space: List[Parameter],
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = False,
        n_grid_points: int = 10,
        random_state: int = 42,
    ):
        """初始化网格搜索优化器

        Args:
            parameter_space: 参数空间
            objective_function: 目标函数
            maximize: 是否最大化
            n_grid_points: 每个参数的网格点数
            random_state: 随机种子
        """
        # 计算总试验次数
        n_trials = self._calculate_n_trials(parameter_space, n_grid_points)

        super().__init__(
            parameter_space, objective_function, maximize, n_trials, random_state
        )

        self.n_grid_points = n_grid_points

        # 生成网格点
        self.grid_points = self._generate_grid()
        self.current_idx = 0

    def _calculate_n_trials(
        self, parameter_space: List[Parameter], n_grid_points: int
    ) -> int:
        """计算总试验次数

        Args:
            parameter_space: 参数空间
            n_grid_points: 网格点数

        Returns:
            总试验次数
        """
        n_trials = 1
        for param in parameter_space:
            if param.param_type in ["float", "int"]:
                n_trials *= n_grid_points
            elif param.param_type in ["categorical", "bool"]:
                n_trials *= len(param.choices)
        return n_trials

    def _generate_grid(self) -> List[Dict[str, Any]]:
        """生成网格点

        Returns:
            网格点列表
        """
        logger.info(f"Generating grid with {self.n_trials} points")

        # 为每个参数生成取值
        param_values = []

        for param in self.parameter_space:
            if param.param_type == "float":
                if param.log_scale:
                    values = np.logspace(
                        np.log10(param.low),
                        np.log10(param.high),
                        self.n_grid_points,
                    )
                else:
                    values = np.linspace(param.low, param.high, self.n_grid_points)
            elif param.param_type == "int":
                values = np.linspace(
                    param.low,
                    param.high,
                    min(self.n_grid_points, param.high - param.low + 1),
                )
                values = np.round(values).astype(int)
                values = np.unique(values)  # 去重
            elif param.param_type in ["categorical", "bool"]:
                values = param.choices

            param_values.append((param.name, values))

        # 生成笛卡尔积
        grid_points = []
        self._recursive_grid(param_values, 0, {}, grid_points)

        logger.info(f"Generated {len(grid_points)} grid points")
        return grid_points

    def _recursive_grid(
        self,
        param_values: List[tuple],
        depth: int,
        current_params: Dict[str, Any],
        grid_points: List[Dict[str, Any]],
    ) -> None:
        """递归生成网格点

        Args:
            param_values: 参数值列表
            depth: 当前深度
            current_params: 当前参数
            grid_points: 网格点列表
        """
        if depth == len(param_values):
            grid_points.append(current_params.copy())
            return

        param_name, values = param_values[depth]
        for value in values:
            current_params[param_name] = value
            self._recursive_grid(param_values, depth + 1, current_params, grid_points)

    def suggest_parameters(self) -> Dict[str, Any]:
        """建议下一组参数

        Returns:
            参数字典
        """
        if self.current_idx < len(self.grid_points):
            params = self.grid_points[self.current_idx]
            self.current_idx += 1
            return params
        else:
            # 所有点已遍历完
            return {}

    def _update_optimization_state(self, trial) -> None:
        """更新优化器状态

        Args:
            trial: 完成的试验
        """
        # 网格搜索不需要更新状态
        pass
