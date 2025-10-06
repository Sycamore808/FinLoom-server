"""
帕累托前沿分析模块
提供帕累托前沿的分析和可视化功能
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.logging_system import setup_logger

logger = setup_logger("pareto_frontier")


class ParetoFrontier:
    """帕累托前沿分析器"""

    def __init__(self, solutions: List[Dict[str, Any]], objective_names: List[str]):
        """初始化帕累托前沿分析器

        Args:
            solutions: 解列表，每个解包含parameters和objectives
            objective_names: 目标名称列表
        """
        self.solutions = solutions
        self.objective_names = objective_names
        self.n_objectives = len(objective_names)

        # 提取目标矩阵
        self.objective_matrix = self._extract_objectives()

    def _extract_objectives(self) -> np.ndarray:
        """提取目标矩阵

        Returns:
            目标矩阵 (n_solutions, n_objectives)
        """
        objectives = []
        for sol in self.solutions:
            if "objectives" in sol and sol["objectives"] is not None:
                objectives.append(sol["objectives"])
            else:
                logger.warning("Solution missing objectives")
                objectives.append([float("inf")] * self.n_objectives)

        return np.array(objectives)

    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """获取帕累托前沿解

        Returns:
            帕累托最优解列表
        """
        n_solutions = len(self.solutions)
        is_dominated = np.zeros(n_solutions, dtype=bool)

        # 检查每个解是否被支配
        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j and self._dominates(j, i):
                    is_dominated[i] = True
                    break

        # 返回非支配解
        pareto_solutions = [
            self.solutions[i] for i in range(n_solutions) if not is_dominated[i]
        ]

        logger.info(f"Found {len(pareto_solutions)} Pareto optimal solutions")
        return pareto_solutions

    def _dominates(self, i: int, j: int) -> bool:
        """判断解i是否支配解j

        Args:
            i: 解索引i
            j: 解索引j

        Returns:
            是否支配
        """
        obj_i = self.objective_matrix[i]
        obj_j = self.objective_matrix[j]

        # 至少在一个目标上更好，且在所有目标上不更差
        better_in_any = False
        for k in range(self.n_objectives):
            if obj_i[k] > obj_j[k]:
                return False  # 在某个目标上更差
            if obj_i[k] < obj_j[k]:
                better_in_any = True

        return better_in_any

    def calculate_hypervolume(
        self, reference_point: Optional[np.ndarray] = None
    ) -> float:
        """计算超体积指标

        Args:
            reference_point: 参考点，默认为每个目标的最大值+1

        Returns:
            超体积值
        """
        pareto_front = self.get_pareto_front()
        if not pareto_front:
            return 0.0

        # 提取Pareto前沿的目标值
        pareto_objectives = np.array([sol["objectives"] for sol in pareto_front])

        # 设置参考点
        if reference_point is None:
            reference_point = self.objective_matrix.max(axis=0) + 1

        # 简化的超体积计算（仅适用于2D情况）
        if self.n_objectives == 2:
            # 按第一个目标排序
            sorted_indices = np.argsort(pareto_objectives[:, 0])
            sorted_objectives = pareto_objectives[sorted_indices]

            hypervolume = 0.0
            for i in range(len(sorted_objectives)):
                if i == 0:
                    width = reference_point[0] - sorted_objectives[i, 0]
                else:
                    width = sorted_objectives[i - 1, 0] - sorted_objectives[i, 0]

                height = reference_point[1] - sorted_objectives[i, 1]
                hypervolume += width * height

            return max(0.0, hypervolume)
        else:
            logger.warning("Hypervolume calculation only implemented for 2D")
            return 0.0

    def get_extreme_solutions(self) -> Dict[str, Dict[str, Any]]:
        """获取每个目标的极值解

        Returns:
            极值解字典
        """
        extreme_solutions = {}

        for i, obj_name in enumerate(self.objective_names):
            # 找到该目标的最小值解
            best_idx = np.argmin(self.objective_matrix[:, i])
            extreme_solutions[f"best_{obj_name}"] = self.solutions[best_idx]

        return extreme_solutions

    def select_solution_by_preference(
        self, preferences: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """根据偏好选择解

        Args:
            preferences: 目标偏好权重字典

        Returns:
            选中的解
        """
        if not self.solutions:
            return None

        # 归一化偏好权重
        total_weight = sum(preferences.values())
        if total_weight == 0:
            logger.warning("Total preference weight is zero")
            return None

        normalized_prefs = {k: v / total_weight for k, v in preferences.items()}

        # 归一化目标值
        obj_min = self.objective_matrix.min(axis=0)
        obj_max = self.objective_matrix.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1  # 避免除零

        normalized_objectives = (self.objective_matrix - obj_min) / obj_range

        # 计算加权得分
        scores = []
        for i in range(len(self.solutions)):
            score = 0.0
            for j, obj_name in enumerate(self.objective_names):
                weight = normalized_prefs.get(obj_name, 0)
                score += weight * normalized_objectives[i, j]
            scores.append(score)

        # 选择得分最低的解（最小化问题）
        best_idx = np.argmin(scores)
        return self.solutions[best_idx]

    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame

        Returns:
            DataFrame格式的解集
        """
        data = []

        for i, sol in enumerate(self.solutions):
            row = {"solution_id": i}

            # 添加目标值
            if "objectives" in sol and sol["objectives"] is not None:
                for j, obj_name in enumerate(self.objective_names):
                    row[obj_name] = sol["objectives"][j]

            # 添加参数
            if "parameters" in sol:
                for param_name, param_value in sol["parameters"].items():
                    row[param_name] = param_value

            # 添加其他信息
            if "crowding_distance" in sol:
                row["crowding_distance"] = sol["crowding_distance"]
            if "rank" in sol:
                row["rank"] = sol["rank"]

            data.append(row)

        return pd.DataFrame(data)

    def get_diversity_metrics(self) -> Dict[str, float]:
        """计算解集的多样性指标

        Returns:
            多样性指标字典
        """
        if len(self.solutions) < 2:
            return {"diversity": 0.0, "spread": 0.0}

        # 计算解之间的最小距离
        from scipy.spatial.distance import pdist

        distances = pdist(self.objective_matrix, metric="euclidean")

        return {
            "min_distance": float(distances.min()) if len(distances) > 0 else 0.0,
            "mean_distance": float(distances.mean()) if len(distances) > 0 else 0.0,
            "max_distance": float(distances.max()) if len(distances) > 0 else 0.0,
            "std_distance": float(distances.std()) if len(distances) > 0 else 0.0,
        }
