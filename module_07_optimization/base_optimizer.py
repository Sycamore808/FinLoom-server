"""
基础优化器模块
提供所有优化器的基类和通用功能
"""

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("base_optimizer")


class OptimizationStatus(Enum):
    """优化状态枚举"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Parameter:
    """参数定义"""

    name: str
    param_type: str  # 'float', 'int', 'categorical', 'bool'
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None
    log_scale: bool = False

    def __post_init__(self):
        """验证参数定义"""
        if self.param_type in ["float", "int"]:
            if self.low is None or self.high is None:
                raise ValueError(
                    f"Parameter {self.name} requires 'low' and 'high' bounds"
                )
            if self.low >= self.high:
                raise ValueError(
                    f"Parameter {self.name}: 'low' must be less than 'high'"
                )
        elif self.param_type == "categorical":
            if not self.choices:
                raise ValueError(
                    f"Categorical parameter {self.name} requires 'choices'"
                )
        elif self.param_type == "bool":
            self.choices = [True, False]

    def sample(self, random_state: Optional[np.random.RandomState] = None) -> Any:
        """随机采样参数值

        Args:
            random_state: 随机状态

        Returns:
            采样的参数值
        """
        if random_state is None:
            random_state = np.random.RandomState()

        if self.param_type == "float":
            if self.log_scale:
                return np.exp(random_state.uniform(np.log(self.low), np.log(self.high)))
            else:
                return random_state.uniform(self.low, self.high)
        elif self.param_type == "int":
            return random_state.randint(self.low, self.high + 1)
        elif self.param_type in ["categorical", "bool"]:
            return random_state.choice(self.choices)


@dataclass
class Trial:
    """优化试验"""

    trial_id: str
    parameters: Dict[str, Any]
    objective_value: Optional[float] = None
    multi_objectives: Optional[Dict[str, float]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: OptimizationStatus = OptimizationStatus.PENDING
    error_message: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """计算试验耗时"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "trial_id": self.trial_id,
            "parameters": self.parameters,
            "objective_value": self.objective_value,
            "multi_objectives": self.multi_objectives,
            "metrics": self.metrics,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class OptimizationResult:
    """优化结果"""

    optimization_id: str
    best_parameters: Dict[str, Any]
    best_value: float
    all_trials: List[Trial]
    convergence_history: List[float]
    total_time_seconds: float
    n_trials: int
    n_successful_trials: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_top_n_trials(self, n: int = 10) -> List[Trial]:
        """获取最佳的N个试验

        Args:
            n: 试验数量

        Returns:
            最佳试验列表
        """
        successful_trials = [
            t for t in self.all_trials if t.objective_value is not None
        ]
        sorted_trials = sorted(successful_trials, key=lambda t: t.objective_value)
        return sorted_trials[:n]

    def save(self, filepath: str) -> None:
        """保存优化结果

        Args:
            filepath: 文件路径
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "OptimizationResult":
        """加载优化结果

        Args:
            filepath: 文件路径

        Returns:
            优化结果对象
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)


class BaseOptimizer(ABC):
    """基础优化器抽象类"""

    def __init__(
        self,
        parameter_space: List[Parameter],
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = False,
        n_trials: int = 100,
        random_state: int = 42,
    ):
        """初始化优化器

        Args:
            parameter_space: 参数空间定义
            objective_function: 目标函数
            maximize: 是否最大化目标
            n_trials: 试验次数
            random_state: 随机种子
        """
        self.parameter_space = parameter_space
        self.objective_function = objective_function
        self.maximize = maximize
        self.n_trials = n_trials
        self.random_state = np.random.RandomState(random_state)

        # 优化历史
        self.trials: List[Trial] = []
        self.best_parameters: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None
        self.optimization_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    @abstractmethod
    def suggest_parameters(self) -> Dict[str, Any]:
        """建议下一组参数

        Returns:
            参数字典
        """
        pass

    def optimize(self) -> OptimizationResult:
        """执行优化

        Returns:
            优化结果
        """
        logger.info(f"Starting optimization with {self.n_trials} trials")
        start_time = datetime.now()
        convergence_history = []

        for i in range(self.n_trials):
            # 创建试验
            trial = Trial(
                trial_id=f"{self.optimization_id}_{i:04d}",
                parameters=self.suggest_parameters(),
                start_time=datetime.now(),
            )

            try:
                # 评估目标函数
                trial.status = OptimizationStatus.RUNNING
                objective_value = self.objective_function(trial.parameters)

                # 如果需要最大化，取负值
                if self.maximize:
                    objective_value = -objective_value

                trial.objective_value = objective_value
                trial.status = OptimizationStatus.COMPLETED

                # 更新最佳结果
                if self.best_value is None or objective_value < self.best_value:
                    self.best_value = objective_value
                    self.best_parameters = trial.parameters.copy()
                    logger.info(f"New best value: {self.best_value:.6f}")

                convergence_history.append(
                    self.best_value if not self.maximize else -self.best_value
                )

            except Exception as e:
                trial.status = OptimizationStatus.FAILED
                trial.error_message = str(e)
                logger.error(f"Trial {trial.trial_id} failed: {e}")
                convergence_history.append(
                    convergence_history[-1] if convergence_history else float("inf")
                )

            finally:
                trial.end_time = datetime.now()
                self.trials.append(trial)
                self._update_optimization_state(trial)

                # 进度日志
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{self.n_trials} trials completed")

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # 如果是最大化问题，转换回原始值
        if self.maximize and self.best_value is not None:
            self.best_value = -self.best_value

        # 创建结果
        result = OptimizationResult(
            optimization_id=self.optimization_id,
            best_parameters=self.best_parameters or {},
            best_value=self.best_value or float("inf"),
            all_trials=self.trials,
            convergence_history=convergence_history,
            total_time_seconds=total_time,
            n_trials=len(self.trials),
            n_successful_trials=sum(
                1 for t in self.trials if t.status == OptimizationStatus.COMPLETED
            ),
            metadata={
                "optimizer_class": self.__class__.__name__,
                "maximize": self.maximize,
                "parameter_space": [p.__dict__ for p in self.parameter_space],
            },
        )

        logger.info(f"Optimization completed. Best value: {self.best_value:.6f}")
        return result

    @abstractmethod
    def _update_optimization_state(self, trial: Trial) -> None:
        """更新优化器内部状态

        Args:
            trial: 完成的试验
        """
        pass

    def get_parameter_by_name(self, name: str) -> Optional[Parameter]:
        """根据名称获取参数定义

        Args:
            name: 参数名称

        Returns:
            参数定义对象
        """
        for param in self.parameter_space:
            if param.name == name:
                return param
        return None
