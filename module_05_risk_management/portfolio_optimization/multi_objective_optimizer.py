"""
多目标优化器模块
实现多目标投资组合优化
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.exceptions import ModelError
from common.logging_system import setup_logger
from scipy import optimize

logger = setup_logger("multi_objective_optimizer")


class OptimizationMethod(Enum):
    """优化方法枚举"""

    WEIGHTED_SUM = "weighted_sum"
    EPSILON_CONSTRAINT = "epsilon_constraint"
    GOAL_PROGRAMMING = "goal_programming"
    NSGA_II = "nsga_ii"
    PARETO_FRONTIER = "pareto_frontier"


@dataclass
class Objective:
    """优化目标"""

    name: str
    function: Callable
    weight: float
    direction: str  # minimize or maximize
    target: Optional[float] = None
    constraint: Optional[Tuple[float, float]] = None


@dataclass
class MultiObjectiveConfig:
    """多目标优化配置"""

    method: OptimizationMethod = OptimizationMethod.WEIGHTED_SUM
    population_size: int = 100
    n_generations: int = 200
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    tournament_size: int = 2
    n_pareto_points: int = 50
    epsilon_grid_size: int = 10


@dataclass
class ParetoSolution:
    """帕累托解"""

    weights: np.ndarray
    objective_values: Dict[str, float]
    is_dominated: bool
    crowding_distance: float
    rank: int


@dataclass
class MultiObjectiveResult:
    """多目标优化结果"""

    pareto_solutions: List[ParetoSolution]
    selected_solution: ParetoSolution
    objective_trade_offs: pd.DataFrame
    convergence_history: pd.DataFrame
    hypervolume: float
    spacing_metric: float


class MultiObjectiveOptimizer:
    """多目标优化器类"""

    def __init__(self, config: Optional[MultiObjectiveConfig] = None):
        """初始化多目标优化器

        Args:
            config: 多目标优化配置
        """
        self.config = config or MultiObjectiveConfig()
        self.objectives: List[Objective] = []
        self.constraints: List[Dict[str, Any]] = []
        self.optimization_history: List[MultiObjectiveResult] = []

    def add_objective(
        self,
        name: str,
        function: Callable,
        weight: float = 1.0,
        direction: str = "maximize",
        target: Optional[float] = None,
    ) -> None:
        """添加优化目标

        Args:
            name: 目标名称
            function: 目标函数
            weight: 权重
            direction: 优化方向
            target: 目标值
        """
        objective = Objective(
            name=name,
            function=function,
            weight=weight,
            direction=direction,
            target=target,
        )
        self.objectives.append(objective)
        logger.info(f"Added objective: {name} ({direction})")

    def optimize(
        self,
        n_assets: int,
        bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
    ) -> MultiObjectiveResult:
        """执行多目标优化

        Args:
            n_assets: 资产数量
            bounds: 变量边界
            constraints: 约束条件

        Returns:
            多目标优化结果
        """
        logger.info(
            f"Starting multi-objective optimization with {len(self.objectives)} objectives"
        )

        if bounds is None:
            bounds = [(0, 1) for _ in range(n_assets)]

        if constraints:
            self.constraints = constraints

        # 根据方法选择优化策略
        if self.config.method == OptimizationMethod.WEIGHTED_SUM:
            result = self._optimize_weighted_sum(n_assets, bounds)
        elif self.config.method == OptimizationMethod.EPSILON_CONSTRAINT:
            result = self._optimize_epsilon_constraint(n_assets, bounds)
        elif self.config.method == OptimizationMethod.NSGA_II:
            result = self._optimize_nsga2(n_assets, bounds)
        elif self.config.method == OptimizationMethod.PARETO_FRONTIER:
            result = self._compute_pareto_frontier(n_assets, bounds)
        else:
            result = self._optimize_goal_programming(n_assets, bounds)

        self.optimization_history.append(result)

        logger.info(
            f"Optimization completed. Found {len(result.pareto_solutions)} Pareto solutions"
        )

        return result

    def calculate_hypervolume(
        self,
        solutions: List[ParetoSolution],
        reference_point: Optional[np.ndarray] = None,
    ) -> float:
        """计算超体积指标

        Args:
            solutions: 解列表
            reference_point: 参考点

        Returns:
            超体积
        """
        if not solutions:
            return 0.0

        # 提取目标值
        objectives_array = np.array(
            [
                [sol.objective_values[obj.name] for obj in self.objectives]
                for sol in solutions
            ]
        )

        # 标准化到最小化问题
        for i, obj in enumerate(self.objectives):
            if obj.direction == "maximize":
                objectives_array[:, i] = -objectives_array[:, i]

        # 设置参考点
        if reference_point is None:
            reference_point = np.max(objectives_array, axis=0) + 1

        # 计算超体积（简化版本）
        hypervolume = self._calculate_hypervolume_2d(objectives_array, reference_point)

        return hypervolume

    def select_preferred_solution(
        self,
        pareto_solutions: List[ParetoSolution],
        preferences: Optional[Dict[str, float]] = None,
    ) -> ParetoSolution:
        """选择偏好解

        Args:
            pareto_solutions: 帕累托解列表
            preferences: 偏好权重

        Returns:
            选择的解
        """
        if not pareto_solutions:
            raise ValueError("No Pareto solutions available")

        if preferences is None:
            # 使用默认权重
            preferences = {obj.name: obj.weight for obj in self.objectives}

        # 计算加权得分
        best_score = -float("inf")
        best_solution = pareto_solutions[0]

        for solution in pareto_solutions:
            score = 0
            for obj_name, pref_weight in preferences.items():
                if obj_name in solution.objective_values:
                    obj_value = solution.objective_values[obj_name]
                    # 找到对应的目标
                    obj = next((o for o in self.objectives if o.name == obj_name), None)
                    if obj:
                        if obj.direction == "minimize":
                            score -= obj_value * pref_weight
                        else:
                            score += obj_value * pref_weight

            if score > best_score:
                best_score = score
                best_solution = solution

        return best_solution

    def _optimize_weighted_sum(
        self, n_assets: int, bounds: List[Tuple[float, float]]
    ) -> MultiObjectiveResult:
        """加权和方法优化

        Args:
            n_assets: 资产数量
            bounds: 边界

        Returns:
            优化结果
        """

        def combined_objective(weights):
            """组合目标函数"""
            total_score = 0

            for obj in self.objectives:
                value = obj.function(weights)

                if obj.direction == "minimize":
                    total_score += obj.weight * value
                else:
                    total_score -= obj.weight * value  # 最大化转最小化

            return total_score

        # 初始猜测
        x0 = np.ones(n_assets) / n_assets

        # 约束
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        ] + self.constraints

        # 优化
        result = optimize.minimize(
            combined_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        # 计算各目标值
        objective_values = {obj.name: obj.function(result.x) for obj in self.objectives}

        # 创建解
        solution = ParetoSolution(
            weights=result.x,
            objective_values=objective_values,
            is_dominated=False,
            crowding_distance=0,
            rank=1,
        )

        return MultiObjectiveResult(
            pareto_solutions=[solution],
            selected_solution=solution,
            objective_trade_offs=pd.DataFrame([objective_values]),
            convergence_history=pd.DataFrame(),
            hypervolume=0,
            spacing_metric=0,
        )

    def _optimize_epsilon_constraint(
        self, n_assets: int, bounds: List[Tuple[float, float]]
    ) -> MultiObjectiveResult:
        """ε-约束方法优化

        Args:
            n_assets: 资产数量
            bounds: 边界

        Returns:
            优化结果
        """
        if len(self.objectives) < 2:
            return self._optimize_weighted_sum(n_assets, bounds)

        # 选择主目标
        primary_obj = self.objectives[0]
        secondary_objs = self.objectives[1:]

        pareto_solutions = []

        # 为每个次要目标生成ε值网格
        epsilon_values = {}
        for obj in secondary_objs:
            # 先优化单个目标找出范围
            if obj.direction == "minimize":
                min_val = self._optimize_single_objective(obj, n_assets, bounds)
                max_val = min_val * 2  # 假设最大值
            else:
                max_val = self._optimize_single_objective(obj, n_assets, bounds)
                min_val = 0

            epsilon_values[obj.name] = np.linspace(
                min_val, max_val, self.config.epsilon_grid_size
            )

        # 对每个ε组合求解
        import itertools

        epsilon_combinations = itertools.product(*epsilon_values.values())

        for epsilon_combo in epsilon_combinations:
            # 构建约束
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            ] + self.constraints

            # 添加ε约束
            for i, obj in enumerate(secondary_objs):
                epsilon = epsilon_combo[i]
                if obj.direction == "minimize":
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda x, o=obj, e=epsilon: e - o.function(x),
                        }
                    )
                else:
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda x, o=obj, e=epsilon: o.function(x) - e,
                        }
                    )

            # 优化主目标
            x0 = np.ones(n_assets) / n_assets

            if primary_obj.direction == "minimize":
                result = optimize.minimize(
                    primary_obj.function,
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                )
            else:
                result = optimize.minimize(
                    lambda x: -primary_obj.function(x),
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                )

            if result.success:
                # 计算所有目标值
                objective_values = {
                    obj.name: obj.function(result.x) for obj in self.objectives
                }

                solution = ParetoSolution(
                    weights=result.x,
                    objective_values=objective_values,
                    is_dominated=False,
                    crowding_distance=0,
                    rank=1,
                )

                pareto_solutions.append(solution)

        # 过滤非支配解
        pareto_solutions = self._filter_non_dominated(pareto_solutions)

        # 选择偏好解
        selected = self.select_preferred_solution(pareto_solutions)

        # 构建权衡分析
        trade_offs = pd.DataFrame([sol.objective_values for sol in pareto_solutions])

        return MultiObjectiveResult(
            pareto_solutions=pareto_solutions,
            selected_solution=selected,
            objective_trade_offs=trade_offs,
            convergence_history=pd.DataFrame(),
            hypervolume=self.calculate_hypervolume(pareto_solutions),
            spacing_metric=self._calculate_spacing_metric(pareto_solutions),
        )

    def _optimize_nsga2(
        self, n_assets: int, bounds: List[Tuple[float, float]]
    ) -> MultiObjectiveResult:
        """NSGA-II算法优化

        Args:
            n_assets: 资产数量
            bounds: 边界

        Returns:
            优化结果
        """
        # 初始化种群
        population = self._initialize_population(n_assets, bounds)

        convergence_history = []

        for generation in range(self.config.n_generations):
            # 评估适应度
            self._evaluate_population(population)

            # 非支配排序
            fronts = self._non_dominated_sort(population)

            # 计算拥挤距离
            for front in fronts:
                self._calculate_crowding_distance(front)

            # 记录收敛历史
            best_front = fronts[0] if fronts else []
            if best_front:
                avg_objectives = {
                    obj.name: np.mean(
                        [sol.objective_values[obj.name] for sol in best_front]
                    )
                    for obj in self.objectives
                }
                convergence_history.append(avg_objectives)

            # 选择
            parents = self._tournament_selection(population)

            # 交叉和变异
            offspring = self._crossover_mutation(parents, bounds)

            # 环境选择
            population = self._environmental_selection(population + offspring)

        # 最终非支配排序
        fronts = self._non_dominated_sort(population)
        pareto_solutions = fronts[0] if fronts else []

        # 选择偏好解
        selected = self.select_preferred_solution(pareto_solutions)

        # 构建结果
        trade_offs = pd.DataFrame([sol.objective_values for sol in pareto_solutions])

        return MultiObjectiveResult(
            pareto_solutions=pareto_solutions,
            selected_solution=selected,
            objective_trade_offs=trade_offs,
            convergence_history=pd.DataFrame(convergence_history),
            hypervolume=self.calculate_hypervolume(pareto_solutions),
            spacing_metric=self._calculate_spacing_metric(pareto_solutions),
        )

    def _compute_pareto_frontier(
        self, n_assets: int, bounds: List[Tuple[float, float]]
    ) -> MultiObjectiveResult:
        """计算帕累托前沿

        Args:
            n_assets: 资产数量
            bounds: 边界

        Returns:
            优化结果
        """
        pareto_solutions = []

        # 生成权重组合
        n_points = self.config.n_pareto_points

        if len(self.objectives) == 2:
            # 双目标：简单扫描
            weights_range = np.linspace(0, 1, n_points)

            for w in weights_range:
                # 组合目标
                def combined_obj(x):
                    obj1_val = self.objectives[0].function(x)
                    obj2_val = self.objectives[1].function(x)

                    if self.objectives[0].direction == "minimize":
                        score1 = w * obj1_val
                    else:
                        score1 = -w * obj1_val

                    if self.objectives[1].direction == "minimize":
                        score2 = (1 - w) * obj2_val
                    else:
                        score2 = -(1 - w) * obj2_val

                    return score1 + score2

                # 优化
                x0 = np.ones(n_assets) / n_assets
                constraints = [
                    {"type": "eq", "fun": lambda x: np.sum(x) - 1}
                ] + self.constraints

                result = optimize.minimize(
                    combined_obj,
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                )

                if result.success:
                    objective_values = {
                        obj.name: obj.function(result.x) for obj in self.objectives
                    }

                    solution = ParetoSolution(
                        weights=result.x,
                        objective_values=objective_values,
                        is_dominated=False,
                        crowding_distance=0,
                        rank=1,
                    )

                    pareto_solutions.append(solution)

        else:
            # 多目标：使用NSGA-II
            return self._optimize_nsga2(n_assets, bounds)

        # 过滤非支配解
        pareto_solutions = self._filter_non_dominated(pareto_solutions)

        # 选择偏好解
        selected = self.select_preferred_solution(pareto_solutions)

        # 构建结果
        trade_offs = pd.DataFrame([sol.objective_values for sol in pareto_solutions])

        return MultiObjectiveResult(
            pareto_solutions=pareto_solutions,
            selected_solution=selected,
            objective_trade_offs=trade_offs,
            convergence_history=pd.DataFrame(),
            hypervolume=self.calculate_hypervolume(pareto_solutions),
            spacing_metric=self._calculate_spacing_metric(pareto_solutions),
        )

    def _optimize_goal_programming(
        self, n_assets: int, bounds: List[Tuple[float, float]]
    ) -> MultiObjectiveResult:
        """目标规划优化

        Args:
            n_assets: 资产数量
            bounds: 边界

        Returns:
            优化结果
        """
        # 定义偏差变量
        n_objectives = len(self.objectives)

        # 扩展变量：原始权重 + 正偏差 + 负偏差
        n_vars = n_assets + 2 * n_objectives

        def objective(x):
            """最小化偏差"""
            weights = x[:n_assets]
            deviations = x[n_assets:]

            # 计算加权偏差和
            total_deviation = 0
            for i, obj in enumerate(self.objectives):
                pos_dev = deviations[2 * i]
                neg_dev = deviations[2 * i + 1]
                total_deviation += obj.weight * (pos_dev + neg_dev)

            return total_deviation

        # 约束
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x[:n_assets]) - 1}]

        # 目标约束
        for i, obj in enumerate(self.objectives):
            if obj.target is not None:

                def goal_constraint(x, idx=i, objective=obj):
                    weights = x[:n_assets]
                    pos_dev = x[n_assets + 2 * idx]
                    neg_dev = x[n_assets + 2 * idx + 1]

                    actual_value = objective.function(weights)

                    return actual_value - objective.target + neg_dev - pos_dev

                constraints.append({"type": "eq", "fun": goal_constraint})

        # 扩展边界
        extended_bounds = list(bounds) + [(0, float("inf"))] * (2 * n_objectives)

        # 初始猜测
        x0 = np.zeros(n_vars)
        x0[:n_assets] = 1.0 / n_assets

        # 优化
        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=extended_bounds,
            constraints=constraints,
        )

        # 提取权重
        optimal_weights = result.x[:n_assets]

        # 计算目标值
        objective_values = {
            obj.name: obj.function(optimal_weights) for obj in self.objectives
        }

        solution = ParetoSolution(
            weights=optimal_weights,
            objective_values=objective_values,
            is_dominated=False,
            crowding_distance=0,
            rank=1,
        )

        return MultiObjectiveResult(
            pareto_solutions=[solution],
            selected_solution=solution,
            objective_trade_offs=pd.DataFrame([objective_values]),
            convergence_history=pd.DataFrame(),
            hypervolume=0,
            spacing_metric=0,
        )

    # 辅助方法
    def _optimize_single_objective(
        self, objective: Objective, n_assets: int, bounds: List[Tuple[float, float]]
    ) -> float:
        """优化单个目标

        Args:
            objective: 目标
            n_assets: 资产数量
            bounds: 边界

        Returns:
            最优值
        """
        x0 = np.ones(n_assets) / n_assets
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        if objective.direction == "minimize":
            result = optimize.minimize(
                objective.function,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            return result.fun
        else:
            result = optimize.minimize(
                lambda x: -objective.function(x),
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            return -result.fun

    def _initialize_population(
        self, n_assets: int, bounds: List[Tuple[float, float]]
    ) -> List[ParetoSolution]:
        """初始化种群

        Args:
            n_assets: 资产数量
            bounds: 边界

        Returns:
            初始种群
        """
        population = []

        for _ in range(self.config.population_size):
            # 随机生成权重
            weights = np.random.random(n_assets)

            # 应用边界
            for i, (low, high) in enumerate(bounds):
                weights[i] = low + weights[i] * (high - low)

            # 标准化
            weights = weights / weights.sum()

            solution = ParetoSolution(
                weights=weights,
                objective_values={},
                is_dominated=False,
                crowding_distance=0,
                rank=0,
            )

            population.append(solution)

        return population

    def _evaluate_population(self, population: List[ParetoSolution]) -> None:
        """评估种群

        Args:
            population: 种群
        """
        for solution in population:
            solution.objective_values = {
                obj.name: obj.function(solution.weights) for obj in self.objectives
            }

    def _non_dominated_sort(
        self, population: List[ParetoSolution]
    ) -> List[List[ParetoSolution]]:
        """非支配排序

        Args:
            population: 种群

        Returns:
            分层的非支配前沿
        """
        fronts = []
        current_front = []

        # 计算支配关系
        for i, sol1 in enumerate(population):
            sol1.rank = 0
            dominated_count = 0

            for j, sol2 in enumerate(population):
                if i != j:
                    if self._dominates(sol1, sol2):
                        sol2.is_dominated = True
                    elif self._dominates(sol2, sol1):
                        dominated_count += 1

            if dominated_count == 0:
                sol1.rank = 1
                current_front.append(sol1)

        fronts.append(current_front)

        # 后续前沿
        while current_front:
            next_front = []
            for sol in population:
                if sol.rank == 0:  # 未分配
                    # 检查是否被当前前沿支配
                    dominated = False
                    for front_sol in current_front:
                        if self._dominates(front_sol, sol):
                            dominated = True
                            break

                    if not dominated:
                        sol.rank = len(fronts) + 1
                        next_front.append(sol)

            if next_front:
                fronts.append(next_front)
                current_front = next_front
            else:
                break

        return fronts

    def _dominates(self, sol1: ParetoSolution, sol2: ParetoSolution) -> bool:
        """判断支配关系

        Args:
            sol1: 解1
            sol2: 解2

        Returns:
            sol1是否支配sol2
        """
        better_in_any = False

        for obj in self.objectives:
            val1 = sol1.objective_values.get(obj.name, 0)
            val2 = sol2.objective_values.get(obj.name, 0)

            if obj.direction == "minimize":
                if val1 > val2:
                    return False
                elif val1 < val2:
                    better_in_any = True
            else:
                if val1 < val2:
                    return False
                elif val1 > val2:
                    better_in_any = True

        return better_in_any

    def _calculate_crowding_distance(self, front: List[ParetoSolution]) -> None:
        """计算拥挤距离

        Args:
            front: 前沿
        """
        n = len(front)
        if n <= 2:
            for sol in front:
                sol.crowding_distance = float("inf")
            return

        # 初始化
        for sol in front:
            sol.crowding_distance = 0

        # 对每个目标
        for obj in self.objectives:
            # 按目标值排序
            front.sort(key=lambda x: x.objective_values.get(obj.name, 0))

            # 边界解设为无穷大
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            # 计算中间解的拥挤距离
            obj_range = (
                front[-1].objective_values[obj.name]
                - front[0].objective_values[obj.name]
            )

            if obj_range > 0:
                for i in range(1, n - 1):
                    distance = (
                        front[i + 1].objective_values[obj.name]
                        - front[i - 1].objective_values[obj.name]
                    ) / obj_range
                    front[i].crowding_distance += distance

    def _tournament_selection(
        self, population: List[ParetoSolution]
    ) -> List[ParetoSolution]:
        """锦标赛选择

        Args:
            population: 种群

        Returns:
            选中的父代
        """
        parents = []

        for _ in range(len(population)):
            # 随机选择参赛者
            tournament = np.random.choice(
                population, size=self.config.tournament_size, replace=False
            )

            # 选择最好的
            winner = min(tournament, key=lambda x: (x.rank, -x.crowding_distance))
            parents.append(winner)

        return parents

    def _crossover_mutation(
        self, parents: List[ParetoSolution], bounds: List[Tuple[float, float]]
    ) -> List[ParetoSolution]:
        """交叉和变异

        Args:
            parents: 父代
            bounds: 边界

        Returns:
            子代
        """
        offspring = []

        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # 交叉
            if np.random.random() < self.config.crossover_probability:
                # SBX交叉
                child1_weights, child2_weights = self._sbx_crossover(
                    parent1.weights, parent2.weights, bounds
                )
            else:
                child1_weights = parent1.weights.copy()
                child2_weights = parent2.weights.copy()

            # 变异
            if np.random.random() < self.config.mutation_probability:
                child1_weights = self._polynomial_mutation(child1_weights, bounds)
            if np.random.random() < self.config.mutation_probability:
                child2_weights = self._polynomial_mutation(child2_weights, bounds)

            # 标准化权重
            child1_weights = child1_weights / child1_weights.sum()
            child2_weights = child2_weights / child2_weights.sum()

            # 创建子代
            child1 = ParetoSolution(
                weights=child1_weights,
                objective_values={},
                is_dominated=False,
                crowding_distance=0,
                rank=0,
            )

            child2 = ParetoSolution(
                weights=child2_weights,
                objective_values={},
                is_dominated=False,
                crowding_distance=0,
                rank=0,
            )

            offspring.extend([child1, child2])

        return offspring

    def _sbx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        bounds: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """模拟二进制交叉

        Args:
            parent1: 父代1
            parent2: 父代2
            bounds: 边界

        Returns:
            (子代1, 子代2)
        """
        eta = 20  # 分布指数

        child1 = parent1.copy()
        child2 = parent2.copy()

        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    if parent1[i] < parent2[i]:
                        y1 = parent1[i]
                        y2 = parent2[i]
                    else:
                        y1 = parent2[i]
                        y2 = parent1[i]

                    yl = bounds[i][0]
                    yu = bounds[i][1]

                    beta = 1 + (2 * (y1 - yl) / (y2 - y1))
                    alpha = 2 - beta ** (-(eta + 1))

                    u = np.random.random()

                    if u <= 1 / alpha:
                        betaq = (u * alpha) ** (1 / (eta + 1))
                    else:
                        betaq = (1 / (2 - u * alpha)) ** (1 / (eta + 1))

                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                    child1[i] = np.clip(c1, yl, yu)
                    child2[i] = np.clip(c2, yl, yu)

        return child1, child2

    def _polynomial_mutation(
        self, individual: np.ndarray, bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """多项式变异

        Args:
            individual: 个体
            bounds: 边界

        Returns:
            变异后的个体
        """
        eta = 20  # 分布指数
        mutated = individual.copy()

        for i in range(len(individual)):
            if np.random.random() < 1.0 / len(individual):
                y = individual[i]
                yl = bounds[i][0]
                yu = bounds[i][1]

                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)

                u = np.random.random()

                if u <= 0.5:
                    xy = 1 - delta1
                    val = 2 * u + (1 - 2 * u) * (xy ** (eta + 1))
                    deltaq = val ** (1 / (eta + 1)) - 1
                else:
                    xy = 1 - delta2
                    val = 2 * (1 - u) + 2 * (u - 0.5) * (xy ** (eta + 1))
                    deltaq = 1 - val ** (1 / (eta + 1))

                y = y + deltaq * (yu - yl)
                mutated[i] = np.clip(y, yl, yu)

        return mutated

    def _environmental_selection(
        self, combined_population: List[ParetoSolution]
    ) -> List[ParetoSolution]:
        """环境选择

        Args:
            combined_population: 合并种群

        Returns:
            下一代种群
        """
        # 非支配排序
        fronts = self._non_dominated_sort(combined_population)

        next_population = []

        for front in fronts:
            if len(next_population) + len(front) <= self.config.population_size:
                next_population.extend(front)
            else:
                # 需要部分选择
                remaining = self.config.population_size - len(next_population)

                # 计算拥挤距离
                self._calculate_crowding_distance(front)

                # 按拥挤距离排序
                front.sort(key=lambda x: x.crowding_distance, reverse=True)

                next_population.extend(front[:remaining])
                break

        return next_population

    def _filter_non_dominated(
        self, solutions: List[ParetoSolution]
    ) -> List[ParetoSolution]:
        """过滤非支配解

        Args:
            solutions: 解列表

        Returns:
            非支配解列表
        """
        non_dominated = []

        for i, sol1 in enumerate(solutions):
            is_dominated = False

            for j, sol2 in enumerate(solutions):
                if i != j and self._dominates(sol2, sol1):
                    is_dominated = True
                    break

            if not is_dominated:
                non_dominated.append(sol1)

        return non_dominated

    def _calculate_hypervolume_2d(
        self, points: np.ndarray, reference: np.ndarray
    ) -> float:
        """计算2D超体积（简化版）

        Args:
            points: 目标点
            reference: 参考点

        Returns:
            超体积
        """
        if len(points) == 0:
            return 0.0

        # 按第一个目标排序
        sorted_points = points[points[:, 0].argsort()]

        hypervolume = 0.0
        prev_x = 0

        for point in sorted_points:
            if point[0] < reference[0] and point[1] < reference[1]:
                hypervolume += (point[0] - prev_x) * (reference[1] - point[1])
                prev_x = point[0]

        return hypervolume

    def _calculate_spacing_metric(self, solutions: List[ParetoSolution]) -> float:
        """计算间距度量

        Args:
            solutions: 解列表

        Returns:
            间距度量
        """
        if len(solutions) < 2:
            return 0.0

        # 计算每个解到最近邻的距离
        distances = []

        for i, sol1 in enumerate(solutions):
            min_distance = float("inf")

            for j, sol2 in enumerate(solutions):
                if i != j:
                    # 计算目标空间的欧氏距离
                    distance = 0
                    for obj in self.objectives:
                        val1 = sol1.objective_values[obj.name]
                        val2 = sol2.objective_values[obj.name]
                        distance += (val1 - val2) ** 2
                    distance = np.sqrt(distance)

                    min_distance = min(min_distance, distance)

            distances.append(min_distance)

        # 计算间距度量
        avg_distance = np.mean(distances)
        spacing = np.sqrt(np.mean([(d - avg_distance) ** 2 for d in distances]))

        return spacing


# 模块级别函数
def multi_objective_optimize(
    objectives: List[Dict[str, Any]],
    n_assets: int,
    method: str = "nsga_ii",
    config: Optional[MultiObjectiveConfig] = None,
) -> pd.Series:
    """多目标优化的便捷函数

    Args:
        objectives: 目标列表
        n_assets: 资产数量
        method: 优化方法
        config: 配置

    Returns:
        最优权重Series
    """
    if config is None:
        config = MultiObjectiveConfig()
    config.method = OptimizationMethod(method)

    optimizer = MultiObjectiveOptimizer(config)

    for obj_def in objectives:
        optimizer.add_objective(**obj_def)

    result = optimizer.optimize(n_assets)

    return pd.Series(result.selected_solution.weights)
