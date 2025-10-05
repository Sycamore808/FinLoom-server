#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
遗传算法因子搜索模块
使用遗传算法优化因子构建
"""

import warnings

warnings.filterwarnings("ignore")

import copy
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GeneticConfig:
    """遗传算法配置"""

    population_size: int = 50  # 种群大小
    generations: int = 100  # 迭代代数
    crossover_rate: float = 0.8  # 交叉概率
    mutation_rate: float = 0.1  # 变异概率
    elitism_rate: float = 0.1  # 精英保留比例
    max_depth: int = 5  # 最大树深度
    tournament_size: int = 3  # 锦标赛选择大小
    early_stopping: int = 20  # 早停代数
    n_jobs: int = 4  # 并行数


@dataclass
class FactorGene:
    """因子基因"""

    operator: str  # 操作符
    operands: List[Any] = field(default_factory=list)  # 操作数
    depth: int = 0  # 深度
    fitness: float = 0.0  # 适应度


class GeneticFactorSearch:
    """遗传算法因子搜索器"""

    def __init__(self, config: Optional[GeneticConfig] = None):
        """初始化遗传算法因子搜索器

        Args:
            config: 遗传算法配置
        """
        self.config = config or GeneticConfig()

        # 定义操作符集合
        self.operators = {
            # 算术操作符
            "add": self._add_op,
            "sub": self._sub_op,
            "mul": self._mul_op,
            "div": self._div_op,
            "pow": self._pow_op,
            # 统计操作符
            "mean": self._mean_op,
            "std": self._std_op,
            "max": self._max_op,
            "min": self._min_op,
            "median": self._median_op,
            "quantile": self._quantile_op,
            # 技术指标操作符
            "sma": self._sma_op,
            "ema": self._ema_op,
            "rsi": self._rsi_op,
            "rank": self._rank_op,
            "zscore": self._zscore_op,
            "winsorize": self._winsorize_op,
            # 时序操作符
            "lag": self._lag_op,
            "diff": self._diff_op,
            "pct_change": self._pct_change_op,
            "rolling_corr": self._rolling_corr_op,
            # 逻辑操作符
            "condition": self._condition_op,
            "abs": self._abs_op,
            "log": self._log_op,
            "sign": self._sign_op,
        }

        # 终端节点（数据列）
        self.terminals = []

        # 适应度历史
        self.fitness_history = []

        # 最佳个体
        self.best_individual = None
        self.best_fitness = float("-inf")

    def search(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        fitness_func: Optional[Callable] = None,
    ) -> Tuple[FactorGene, float]:
        """搜索最优因子

        Args:
            data: 输入数据
            target: 目标变量
            fitness_func: 适应度函数

        Returns:
            最优因子基因和适应度
        """
        try:
            logger.info("Starting genetic factor search...")

            # 设置终端节点
            self.terminals = list(data.columns)

            # 使用默认适应度函数
            if fitness_func is None:
                fitness_func = self._default_fitness_func

            # 初始化种群
            population = self._initialize_population()

            # 评估初始种群
            population = self._evaluate_population(
                population, data, target, fitness_func
            )

            # 记录初始最佳个体
            self._update_best_individual(population)

            # 进化过程
            no_improvement_count = 0

            for generation in range(self.config.generations):
                logger.debug(f"Generation {generation + 1}/{self.config.generations}")

                # 选择
                selected = self._selection(population)

                # 交叉
                offspring = self._crossover(selected)

                # 变异
                offspring = self._mutation(offspring)

                # 评估新个体
                offspring = self._evaluate_population(
                    offspring, data, target, fitness_func
                )

                # 环境选择（精英保留）
                population = self._environmental_selection(population, offspring)

                # 更新最佳个体
                improved = self._update_best_individual(population)

                if improved:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # 记录适应度历史
                avg_fitness = np.mean([ind.fitness for ind in population])
                self.fitness_history.append(
                    {
                        "generation": generation + 1,
                        "best_fitness": self.best_fitness,
                        "avg_fitness": avg_fitness,
                    }
                )

                # 早停检查
                if no_improvement_count >= self.config.early_stopping:
                    logger.info(f"Early stopping at generation {generation + 1}")
                    break

            logger.info(
                f"Genetic search completed. Best fitness: {self.best_fitness:.6f}"
            )
            return self.best_individual, self.best_fitness

        except Exception as e:
            logger.error(f"Failed to perform genetic factor search: {e}")
            return None, 0.0

    def _initialize_population(self) -> List[FactorGene]:
        """初始化种群"""
        population = []

        for _ in range(self.config.population_size):
            gene = self._generate_random_gene(max_depth=self.config.max_depth)
            population.append(gene)

        return population

    def _generate_random_gene(
        self, max_depth: int, current_depth: int = 0
    ) -> FactorGene:
        """生成随机基因"""
        # 如果达到最大深度或随机决定使用终端节点
        if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
            # 返回终端节点
            operator = random.choice(self.terminals)
            return FactorGene(operator=operator, operands=[], depth=current_depth)

        # 选择操作符
        operator = random.choice(list(self.operators.keys()))

        # 根据操作符生成操作数
        operands = []
        if operator in ["add", "sub", "mul", "div", "rolling_corr"]:
            # 二元操作符
            operands = [
                self._generate_random_gene(max_depth, current_depth + 1),
                self._generate_random_gene(max_depth, current_depth + 1),
            ]
        elif operator in ["condition"]:
            # 三元操作符
            operands = [
                self._generate_random_gene(max_depth, current_depth + 1),
                self._generate_random_gene(max_depth, current_depth + 1),
                self._generate_random_gene(max_depth, current_depth + 1),
            ]
        else:
            # 一元操作符
            operands = [self._generate_random_gene(max_depth, current_depth + 1)]

            # 为某些操作符添加参数
            if operator in ["sma", "ema", "lag"]:
                operands.append(random.randint(2, 20))  # 窗口大小
            elif operator == "quantile":
                operands.append(random.uniform(0.1, 0.9))  # 分位数
            elif operator == "pow":
                operands.append(random.uniform(0.5, 3.0))  # 指数

        return FactorGene(operator=operator, operands=operands, depth=current_depth)

    def _evaluate_population(
        self,
        population: List[FactorGene],
        data: pd.DataFrame,
        target: pd.Series,
        fitness_func: Callable,
    ) -> List[FactorGene]:
        """评估种群适应度"""
        try:
            # 并行评估
            with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
                futures = []
                for gene in population:
                    future = executor.submit(
                        self._evaluate_gene, gene, data, target, fitness_func
                    )
                    futures.append(future)

                # 获取结果
                for i, future in enumerate(futures):
                    try:
                        fitness = future.result(timeout=30)  # 30秒超时
                        population[i].fitness = fitness
                    except Exception as e:
                        logger.warning(f"Failed to evaluate gene {i}: {e}")
                        population[i].fitness = float("-inf")

            return population

        except Exception as e:
            logger.error(f"Failed to evaluate population: {e}")
            # 串行备用方案
            for gene in population:
                gene.fitness = self._evaluate_gene(gene, data, target, fitness_func)
            return population

    def _evaluate_gene(
        self,
        gene: FactorGene,
        data: pd.DataFrame,
        target: pd.Series,
        fitness_func: Callable,
    ) -> float:
        """评估单个基因适应度"""
        try:
            # 执行基因表达式
            factor_values = self._execute_gene(gene, data)

            if factor_values is None or len(factor_values) == 0:
                return float("-inf")

            # 计算适应度
            fitness = fitness_func(factor_values, target)

            return fitness if pd.notna(fitness) else float("-inf")

        except Exception as e:
            logger.debug(f"Failed to evaluate gene: {e}")
            return float("-inf")

    def _execute_gene(
        self, gene: FactorGene, data: pd.DataFrame
    ) -> Optional[pd.Series]:
        """执行基因表达式"""
        try:
            # 如果是终端节点
            if gene.operator in self.terminals:
                return data[gene.operator].copy()

            # 如果是操作符
            if gene.operator in self.operators:
                return self.operators[gene.operator](gene, data)

            return None

        except Exception as e:
            logger.debug(f"Failed to execute gene: {e}")
            return None

    def _selection(self, population: List[FactorGene]) -> List[FactorGene]:
        """选择操作"""
        selected = []

        for _ in range(len(population)):
            # 锦标赛选择
            tournament = random.sample(
                population, min(self.config.tournament_size, len(population))
            )
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(copy.deepcopy(winner))

        return selected

    def _crossover(self, population: List[FactorGene]) -> List[FactorGene]:
        """交叉操作"""
        offspring = []

        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[0]

            if random.random() < self.config.crossover_rate:
                child1, child2 = self._subtree_crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            offspring.extend([child1, child2])

        return offspring[: len(population)]

    def _subtree_crossover(
        self, parent1: FactorGene, parent2: FactorGene
    ) -> Tuple[FactorGene, FactorGene]:
        """子树交叉"""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # 随机选择交叉点（简化实现）
        if random.random() < 0.5 and child1.operands and child2.operands:
            if len(child1.operands) > 0 and len(child2.operands) > 0:
                # 交换第一个操作数
                child1.operands[0], child2.operands[0] = (
                    child2.operands[0],
                    child1.operands[0],
                )

        return child1, child2

    def _mutation(self, population: List[FactorGene]) -> List[FactorGene]:
        """变异操作"""
        mutated = []

        for gene in population:
            if random.random() < self.config.mutation_rate:
                mutated_gene = self._mutate_gene(gene)
            else:
                mutated_gene = copy.deepcopy(gene)

            mutated.append(mutated_gene)

        return mutated

    def _mutate_gene(self, gene: FactorGene) -> FactorGene:
        """变异单个基因"""
        mutated = copy.deepcopy(gene)

        # 随机选择变异类型
        mutation_type = random.choice(["operator", "operand", "parameter"])

        if mutation_type == "operator" and mutated.operator in self.operators:
            # 操作符变异
            mutated.operator = random.choice(list(self.operators.keys()))
        elif mutation_type == "operand" and mutated.operands:
            # 操作数变异
            if len(mutated.operands) > 0:
                idx = random.randint(0, len(mutated.operands) - 1)
                if isinstance(mutated.operands[idx], FactorGene):
                    mutated.operands[idx] = self._generate_random_gene(3)
        elif mutation_type == "parameter" and mutated.operands:
            # 参数变异
            for i, operand in enumerate(mutated.operands):
                if isinstance(operand, (int, float)):
                    if isinstance(operand, int):
                        mutated.operands[i] = random.randint(2, 20)
                    else:
                        mutated.operands[i] = random.uniform(0.1, 0.9)

        return mutated

    def _environmental_selection(
        self, population: List[FactorGene], offspring: List[FactorGene]
    ) -> List[FactorGene]:
        """环境选择（精英保留）"""
        # 合并种群
        combined = population + offspring

        # 按适应度排序
        combined.sort(key=lambda x: x.fitness, reverse=True)

        # 保留最优个体
        return combined[: self.config.population_size]

    def _update_best_individual(self, population: List[FactorGene]) -> bool:
        """更新最佳个体"""
        best_in_population = max(population, key=lambda x: x.fitness)

        if best_in_population.fitness > self.best_fitness:
            self.best_individual = copy.deepcopy(best_in_population)
            self.best_fitness = best_in_population.fitness
            return True

        return False

    def _default_fitness_func(
        self, factor_values: pd.Series, target: pd.Series
    ) -> float:
        """默认适应度函数（信息系数）"""
        try:
            # 对齐数据
            aligned_factor = factor_values.dropna()
            aligned_target = target.reindex(aligned_factor.index).dropna()

            if len(aligned_factor) < 10:
                return float("-inf")

            # 计算相关系数
            corr = aligned_factor.corr(aligned_target)

            # 添加复杂度惩罚
            complexity_penalty = 0.001 * self._calculate_complexity(
                self.best_individual or FactorGene("dummy")
            )

            return abs(corr) - complexity_penalty if pd.notna(corr) else float("-inf")

        except Exception as e:
            logger.debug(f"Failed to calculate fitness: {e}")
            return float("-inf")

    def _calculate_complexity(self, gene: FactorGene) -> int:
        """计算基因复杂度"""
        if not gene.operands:
            return 1

        complexity = 1
        for operand in gene.operands:
            if isinstance(operand, FactorGene):
                complexity += self._calculate_complexity(operand)

        return complexity

    def gene_to_formula(self, gene: FactorGene) -> str:
        """将基因转换为公式字符串"""
        if gene.operator in self.terminals:
            return gene.operator

        if not gene.operands:
            return gene.operator

        if gene.operator in ["add", "sub", "mul", "div"]:
            left = (
                self.gene_to_formula(gene.operands[0])
                if isinstance(gene.operands[0], FactorGene)
                else str(gene.operands[0])
            )
            right = (
                self.gene_to_formula(gene.operands[1])
                if isinstance(gene.operands[1], FactorGene)
                else str(gene.operands[1])
            )
            op_symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
            return f"({left} {op_symbol[gene.operator]} {right})"
        else:
            operand_strs = []
            for operand in gene.operands:
                if isinstance(operand, FactorGene):
                    operand_strs.append(self.gene_to_formula(operand))
                else:
                    operand_strs.append(str(operand))
            return f"{gene.operator}({', '.join(operand_strs)})"

    # 操作符实现
    def _add_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        left = self._execute_gene(gene.operands[0], data)
        right = self._execute_gene(gene.operands[1], data)
        return left + right

    def _sub_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        left = self._execute_gene(gene.operands[0], data)
        right = self._execute_gene(gene.operands[1], data)
        return left - right

    def _mul_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        left = self._execute_gene(gene.operands[0], data)
        right = self._execute_gene(gene.operands[1], data)
        return left * right

    def _div_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        left = self._execute_gene(gene.operands[0], data)
        right = self._execute_gene(gene.operands[1], data)
        return left / (right + 1e-8)  # 避免除零

    def _pow_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        base = self._execute_gene(gene.operands[0], data)
        exp = gene.operands[1] if len(gene.operands) > 1 else 2
        return np.power(base, exp)

    def _mean_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        window = gene.operands[1] if len(gene.operands) > 1 else 20
        return series.rolling(window=window).mean()

    def _std_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        window = gene.operands[1] if len(gene.operands) > 1 else 20
        return series.rolling(window=window).std()

    def _max_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        window = gene.operands[1] if len(gene.operands) > 1 else 20
        return series.rolling(window=window).max()

    def _min_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        window = gene.operands[1] if len(gene.operands) > 1 else 20
        return series.rolling(window=window).min()

    def _median_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        window = gene.operands[1] if len(gene.operands) > 1 else 20
        return series.rolling(window=window).median()

    def _quantile_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        quantile = gene.operands[1] if len(gene.operands) > 1 else 0.5
        return series.rolling(window=20).quantile(quantile)

    def _sma_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        window = gene.operands[1] if len(gene.operands) > 1 else 20
        return series.rolling(window=window).mean()

    def _ema_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        span = gene.operands[1] if len(gene.operands) > 1 else 20
        return series.ewm(span=span).mean()

    def _rsi_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        window = gene.operands[1] if len(gene.operands) > 1 else 14

        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def _rank_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        window = gene.operands[1] if len(gene.operands) > 1 else 20
        return series.rolling(window=window).rank()

    def _zscore_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        window = gene.operands[1] if len(gene.operands) > 1 else 20

        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / (rolling_std + 1e-8)

    def _winsorize_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        window = gene.operands[1] if len(gene.operands) > 1 else 20

        def winsorize_func(x):
            q1, q99 = x.quantile([0.01, 0.99])
            return x.clip(lower=q1, upper=q99)

        return series.rolling(window=window).apply(winsorize_func)

    def _lag_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        lag = gene.operands[1] if len(gene.operands) > 1 else 1
        return series.shift(lag)

    def _diff_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        periods = gene.operands[1] if len(gene.operands) > 1 else 1
        return series.diff(periods)

    def _pct_change_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        periods = gene.operands[1] if len(gene.operands) > 1 else 1
        return series.pct_change(periods)

    def _rolling_corr_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        left = self._execute_gene(gene.operands[0], data)
        right = self._execute_gene(gene.operands[1], data)
        window = gene.operands[2] if len(gene.operands) > 2 else 20
        return left.rolling(window=window).corr(right)

    def _condition_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        condition = self._execute_gene(gene.operands[0], data)
        true_val = self._execute_gene(gene.operands[1], data)
        false_val = self._execute_gene(gene.operands[2], data)
        return np.where(condition > 0, true_val, false_val)

    def _abs_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        return np.abs(series)

    def _log_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        return np.log(np.abs(series) + 1e-8)

    def _sign_op(self, gene: FactorGene, data: pd.DataFrame) -> pd.Series:
        series = self._execute_gene(gene.operands[0], data)
        return np.sign(series)


# 便捷函数
def genetic_factor_search(
    data: pd.DataFrame,
    target: pd.Series,
    config: Optional[GeneticConfig] = None,
    fitness_func: Optional[Callable] = None,
) -> Tuple[Optional[FactorGene], float]:
    """遗传算法因子搜索便捷函数

    Args:
        data: 输入数据
        target: 目标变量
        config: 遗传算法配置
        fitness_func: 适应度函数

    Returns:
        最优因子基因和适应度
    """
    searcher = GeneticFactorSearch(config)
    return searcher.search(data, target, fitness_func)
