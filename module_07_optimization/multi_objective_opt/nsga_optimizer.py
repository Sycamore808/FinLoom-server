"""
NSGA-III多目标优化器模块
实现多目标优化算法
"""

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from common.logging_system import setup_logger
from module_07_optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationStatus,
    Parameter,
    Trial,
)
from scipy.spatial.distance import cdist

logger = setup_logger("nsga_optimizer")


@dataclass
class NSGAIndividual:
    """NSGA个体"""

    genes: np.ndarray  # 基因编码
    objectives: Optional[np.ndarray] = None  # 目标函数值
    rank: int = 0  # Pareto等级
    crowding_distance: float = 0.0  # 拥挤度距离
    dominated_count: int = 0  # 被支配数量
    dominated_set: List[int] = None  # 支配集合

    def __post_init__(self):
        if self.dominated_set is None:
            self.dominated_set = []


class NSGAOptimizer(BaseOptimizer):
    """NSGA-III多目标优化器

    实现非支配排序遗传算法用于多目标优化
    """

    def __init__(
        self,
        parameter_space: List[Parameter],
        objective_functions: List[Callable[[Dict[str, Any]], float]],
        population_size: int = 100,
        n_generations: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        random_state: int = 42,
    ):
        """初始化NSGA优化器

        Args:
            parameter_space: 参数空间
            objective_functions: 目标函数列表
            population_size: 种群大小
            n_generations: 代数
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
            random_state: 随机种子
        """
        # 使用第一个目标函数初始化基类
        super().__init__(
            parameter_space,
            objective_functions[0],
            maximize=False,
            n_trials=population_size * n_generations,
            random_state=random_state,
        )

        self.objective_functions = objective_functions
        self.n_objectives = len(objective_functions)
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # 种群
        self.population: List[NSGAIndividual] = []
        self.pareto_front: List[NSGAIndividual] = []

        # 参数边界
        self._setup_bounds()

    def _setup_bounds(self) -> None:
        """设置参数边界"""
        self.lower_bounds = []
        self.upper_bounds = []

        for param in self.parameter_space:
            if param.param_type in ["float", "int"]:
                self.lower_bounds.append(param.low)
                self.upper_bounds.append(param.high)
            else:
                # 分类变量映射到[0, n-1]
                self.lower_bounds.append(0)
                self.upper_bounds.append(len(param.choices) - 1)

        self.lower_bounds = np.array(self.lower_bounds)
        self.upper_bounds = np.array(self.upper_bounds)

    def optimize(self) -> Dict[str, Any]:
        """执行多目标优化

        Returns:
            优化结果
        """
        logger.info(
            f"Starting NSGA-III optimization with {self.n_generations} generations"
        )

        # 初始化种群
        self._initialize_population()

        # 进化循环
        for generation in range(self.n_generations):
            # 评估种群
            self._evaluate_population()

            # 非支配排序
            fronts = self._non_dominated_sort()

            # 计算拥挤度
            self._calculate_crowding_distance(fronts)

            # 选择
            parents = self._tournament_selection()

            # 交叉和变异
            offspring = self._create_offspring(parents)

            # 环境选择
            self.population = self._environmental_selection(self.population + offspring)

            # 更新Pareto前沿
            self.pareto_front = fronts[0] if fronts else []

            # 进度日志
            if (generation + 1) % 10 == 0:
                logger.info(
                    f"Generation {generation + 1}/{self.n_generations} completed"
                )
                logger.info(f"Pareto front size: {len(self.pareto_front)}")

        # 最终评估
        self._evaluate_population()
        fronts = self._non_dominated_sort()
        self.pareto_front = fronts[0] if fronts else []

        # 创建结果
        result = {
            "pareto_front": self._extract_pareto_front(),
            "all_solutions": self._extract_all_solutions(),
            "n_generations": self.n_generations,
            "population_size": self.population_size,
            "metadata": {
                "n_objectives": self.n_objectives,
                "final_front_size": len(self.pareto_front),
            },
        }

        logger.info(
            f"Optimization completed. Pareto front size: {len(self.pareto_front)}"
        )
        return result

    def _initialize_population(self) -> None:
        """初始化种群"""
        self.population = []

        for _ in range(self.population_size):
            # 随机生成基因
            genes = self.random_state.uniform(self.lower_bounds, self.upper_bounds)

            individual = NSGAIndividual(genes=genes)
            self.population.append(individual)

    def _evaluate_population(self) -> None:
        """评估种群中所有个体"""
        for individual in self.population:
            if individual.objectives is None:
                # 解码参数
                params = self._decode_genes(individual.genes)

                # 评估所有目标函数
                objectives = []
                for obj_func in self.objective_functions:
                    try:
                        value = obj_func(params)
                        objectives.append(value)
                    except Exception as e:
                        logger.warning(f"Objective evaluation failed: {e}")
                        objectives.append(float("inf"))

                individual.objectives = np.array(objectives)

    def _decode_genes(self, genes: np.ndarray) -> Dict[str, Any]:
        """将基因解码为参数

        Args:
            genes: 基因编码

        Returns:
            参数字典
        """
        params = {}

        for i, param in enumerate(self.parameter_space):
            if param.param_type == "float":
                params[param.name] = float(genes[i])
            elif param.param_type == "int":
                params[param.name] = int(np.round(genes[i]))
            elif param.param_type in ["categorical", "bool"]:
                idx = int(np.round(genes[i]))
                idx = np.clip(idx, 0, len(param.choices) - 1)
                params[param.name] = param.choices[idx]

        return params

    def _dominates(self, ind1: NSGAIndividual, ind2: NSGAIndividual) -> bool:
        """判断ind1是否支配ind2

        Args:
            ind1: 个体1
            ind2: 个体2

        Returns:
            是否支配
        """
        better_in_any = False
        for i in range(self.n_objectives):
            if ind1.objectives[i] > ind2.objectives[i]:
                return False
            if ind1.objectives[i] < ind2.objectives[i]:
                better_in_any = True
        return better_in_any

    def _non_dominated_sort(self) -> List[List[NSGAIndividual]]:
        """非支配排序

        Returns:
            Pareto前沿列表
        """
        fronts = []
        current_front = []

        # 计算支配关系
        for i, ind1 in enumerate(self.population):
            ind1.dominated_set = []
            ind1.dominated_count = 0

            for j, ind2 in enumerate(self.population):
                if i != j:
                    if self._dominates(ind1, ind2):
                        ind1.dominated_set.append(j)
                    elif self._dominates(ind2, ind1):
                        ind1.dominated_count += 1

            if ind1.dominated_count == 0:
                ind1.rank = 0
                current_front.append(ind1)

        fronts.append(current_front)

        # 构建后续前沿
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for ind1 in fronts[front_idx]:
                for j in ind1.dominated_set:
                    ind2 = self.population[j]
                    ind2.dominated_count -= 1
                    if ind2.dominated_count == 0:
                        ind2.rank = front_idx + 1
                        next_front.append(ind2)
            front_idx += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    def _calculate_crowding_distance(self, fronts: List[List[NSGAIndividual]]) -> None:
        """计算拥挤度距离

        Args:
            fronts: Pareto前沿列表
        """
        for front in fronts:
            if len(front) <= 2:
                for ind in front:
                    ind.crowding_distance = float("inf")
            else:
                # 初始化距离
                for ind in front:
                    ind.crowding_distance = 0.0

                # 对每个目标函数计算距离
                for m in range(self.n_objectives):
                    # 按目标值排序
                    front.sort(key=lambda x: x.objectives[m])

                    # 边界个体距离无穷大
                    front[0].crowding_distance = float("inf")
                    front[-1].crowding_distance = float("inf")

                    # 计算中间个体的距离
                    obj_range = front[-1].objectives[m] - front[0].objectives[m]
                    if obj_range > 0:
                        for i in range(1, len(front) - 1):
                            distance = (
                                front[i + 1].objectives[m] - front[i - 1].objectives[m]
                            ) / obj_range
                            front[i].crowding_distance += distance

    def _tournament_selection(self) -> List[NSGAIndividual]:
        """锦标赛选择

        Returns:
            选中的父代
        """
        parents = []

        for _ in range(self.population_size):
            # 随机选择两个个体
            idx1, idx2 = self.random_state.choice(
                len(self.population), 2, replace=False
            )
            ind1 = self.population[idx1]
            ind2 = self.population[idx2]

            # 选择更优的个体
            if ind1.rank < ind2.rank:
                parents.append(ind1)
            elif ind1.rank > ind2.rank:
                parents.append(ind2)
            else:
                # 相同等级，选择拥挤度更大的
                if ind1.crowding_distance > ind2.crowding_distance:
                    parents.append(ind1)
                else:
                    parents.append(ind2)

        return parents

    def _create_offspring(self, parents: List[NSGAIndividual]) -> List[NSGAIndividual]:
        """创建子代

        Args:
            parents: 父代列表

        Returns:
            子代列表
        """
        offspring = []

        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]

                # 交叉
                if self.random_state.random() < self.crossover_prob:
                    child1_genes, child2_genes = self._crossover(
                        parent1.genes, parent2.genes
                    )
                else:
                    child1_genes = parent1.genes.copy()
                    child2_genes = parent2.genes.copy()

                # 变异
                child1_genes = self._mutate(child1_genes)
                child2_genes = self._mutate(child2_genes)

                # 创建子代个体
                offspring.append(NSGAIndividual(genes=child1_genes))
                offspring.append(NSGAIndividual(genes=child2_genes))

        return offspring[: self.population_size]

    def _crossover(
        self, parent1_genes: np.ndarray, parent2_genes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """模拟二进制交叉（SBX）

        Args:
            parent1_genes: 父代1基因
            parent2_genes: 父代2基因

        Returns:
            两个子代基因
        """
        child1_genes = parent1_genes.copy()
        child2_genes = parent2_genes.copy()

        eta = 20  # 分布指数

        for i in range(len(parent1_genes)):
            if self.random_state.random() < 0.5:
                # SBX交叉
                u = self.random_state.random()

                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                child1_genes[i] = 0.5 * (
                    (1 + beta) * parent1_genes[i] + (1 - beta) * parent2_genes[i]
                )
                child2_genes[i] = 0.5 * (
                    (1 - beta) * parent1_genes[i] + (1 + beta) * parent2_genes[i]
                )

                # 边界修正
                child1_genes[i] = np.clip(
                    child1_genes[i], self.lower_bounds[i], self.upper_bounds[i]
                )
                child2_genes[i] = np.clip(
                    child2_genes[i], self.lower_bounds[i], self.upper_bounds[i]
                )

        return child1_genes, child2_genes

    def _mutate(self, genes: np.ndarray) -> np.ndarray:
        """多项式变异

        Args:
            genes: 基因

        Returns:
            变异后的基因
        """
        mutated_genes = genes.copy()
        eta = 20  # 分布指数

        for i in range(len(genes)):
            if self.random_state.random() < self.mutation_prob:
                u = self.random_state.random()

                if u <= 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

                mutated_genes[i] = genes[i] + delta * (
                    self.upper_bounds[i] - self.lower_bounds[i]
                )
                mutated_genes[i] = np.clip(
                    mutated_genes[i], self.lower_bounds[i], self.upper_bounds[i]
                )

        return mutated_genes

    def _environmental_selection(
        self, combined_population: List[NSGAIndividual]
    ) -> List[NSGAIndividual]:
        """环境选择

        Args:
            combined_population: 合并的种群

        Returns:
            选中的个体
        """
        # 评估合并种群
        for ind in combined_population:
            if ind.objectives is None:
                params = self._decode_genes(ind.genes)
                objectives = []
                for obj_func in self.objective_functions:
                    try:
                        value = obj_func(params)
                        objectives.append(value)
                    except:
                        objectives.append(float("inf"))
                ind.objectives = np.array(objectives)

        # 临时设置种群为合并种群
        original_population = self.population
        self.population = combined_population

        # 非支配排序
        fronts = self._non_dominated_sort()

        # 计算拥挤度
        self._calculate_crowding_distance(fronts)

        # 恢复原始种群
        self.population = original_population

        # 选择个体
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                # 按拥挤度排序选择
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                remaining = self.population_size - len(new_population)
                new_population.extend(front[:remaining])
                break

        return new_population

    def _extract_pareto_front(self) -> List[Dict[str, Any]]:
        """提取Pareto前沿解

        Returns:
            Pareto前沿解列表
        """
        solutions = []

        for ind in self.pareto_front:
            params = self._decode_genes(ind.genes)
            solution = {
                "parameters": params,
                "objectives": ind.objectives.tolist(),
                "crowding_distance": ind.crowding_distance,
            }
            solutions.append(solution)

        return solutions

    def _extract_all_solutions(self) -> List[Dict[str, Any]]:
        """提取所有解

        Returns:
            所有解列表
        """
        solutions = []

        for ind in self.population:
            params = self._decode_genes(ind.genes)
            solution = {
                "parameters": params,
                "objectives": ind.objectives.tolist()
                if ind.objectives is not None
                else None,
                "rank": ind.rank,
                "crowding_distance": ind.crowding_distance,
            }
            solutions.append(solution)

        return solutions

    def suggest_parameters(self) -> Dict[str, Any]:
        """建议参数（兼容接口）

        Returns:
            参数字典
        """
        # 从当前种群中选择最好的个体
        if self.population and self.population[0].objectives is not None:
            best_ind = min(self.population, key=lambda x: x.objectives[0])
            return self._decode_genes(best_ind.genes)
        else:
            # 随机生成
            genes = self.random_state.uniform(self.lower_bounds, self.upper_bounds)
            return self._decode_genes(genes)

    def _update_optimization_state(self, trial: Trial) -> None:
        """更新优化状态（兼容接口）"""
        pass
