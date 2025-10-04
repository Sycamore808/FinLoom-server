"""
Optuna优化器模块
使用Optuna框架进行高级超参数优化
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import optuna
    from optuna import Trial as OptunaTrial
    from optuna.pruners import HyperbandPruner, MedianPruner
    from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    OptunaTrial = Any  # 占位类型
    HyperbandPruner = None
    MedianPruner = None
    CmaEsSampler = None
    RandomSampler = None
    TPESampler = None

from common.logging_system import setup_logger
from module_07_optimization.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationStatus,
    Parameter,
    Trial,
)

logger = setup_logger("optuna_optimizer")

# 设置Optuna日志级别
if OPTUNA_AVAILABLE:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer(BaseOptimizer):
    """Optuna优化器

    使用Optuna框架提供的高级优化算法
    """

    def __init__(
        self,
        parameter_space: List[Parameter],
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = False,
        n_trials: int = 100,
        sampler: str = "tpe",  # 'tpe', 'cmaes', 'random'
        pruner: Optional[str] = "median",  # 'median', 'hyperband', None
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        random_state: int = 42,
    ):
        """初始化Optuna优化器

        Args:
            parameter_space: 参数空间
            objective_function: 目标函数
            maximize: 是否最大化
            n_trials: 试验次数
            sampler: 采样器类型
            pruner: 剪枝器类型
            n_startup_trials: 初始随机试验数
            n_ei_candidates: EI候选数（TPE采样器）
            random_state: 随机种子
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is not installed. Please install it with: pip install optuna"
            )

        super().__init__(
            parameter_space, objective_function, maximize, n_trials, random_state
        )

        # 创建采样器
        if sampler == "tpe":
            self.sampler = TPESampler(
                n_startup_trials=n_startup_trials,
                n_ei_candidates=n_ei_candidates,
                seed=random_state,
            )
        elif sampler == "cmaes":
            self.sampler = CmaEsSampler(
                n_startup_trials=n_startup_trials, seed=random_state
            )
        elif sampler == "random":
            self.sampler = RandomSampler(seed=random_state)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        # 创建剪枝器
        if pruner == "median":
            self.pruner = MedianPruner(
                n_startup_trials=n_startup_trials, n_warmup_steps=5
            )
        elif pruner == "hyperband":
            self.pruner = HyperbandPruner(
                min_resource=1, max_resource=n_trials, reduction_factor=3
            )
        else:
            self.pruner = None

        # 创建Optuna study
        self.study = None
        self.current_trial_params = None

    def optimize(self) -> OptimizationResult:
        """执行优化

        Returns:
            优化结果
        """
        logger.info(f"Starting Optuna optimization with {self.n_trials} trials")

        # 创建study
        direction = "maximize" if self.maximize else "minimize"
        self.study = optuna.create_study(
            direction=direction, sampler=self.sampler, pruner=self.pruner
        )

        # 设置用户属性
        self.study.set_user_attr(
            "parameter_space", [p.__dict__ for p in self.parameter_space]
        )
        self.study.set_user_attr("optimization_id", self.optimization_id)

        # 执行优化
        start_time = datetime.now()

        def wrapped_objective(optuna_trial: OptunaTrial) -> float:
            """包装的目标函数"""
            # 采样参数
            params = self._sample_parameters(optuna_trial)

            # 创建试验对象
            trial = Trial(
                trial_id=f"{self.optimization_id}_{len(self.trials):04d}",
                parameters=params,
                start_time=datetime.now(),
            )

            try:
                # 评估目标函数
                trial.status = OptimizationStatus.RUNNING
                objective_value = self.objective_function(params)
                trial.objective_value = objective_value
                trial.status = OptimizationStatus.COMPLETED

                # 存储额外指标（注意：trial.state在运行期间不可用）
                trial.metrics = {
                    "optuna_trial_number": optuna_trial.number,
                }

                # 报告中间值（用于剪枝）
                if self.pruner and hasattr(objective_value, "__iter__"):
                    for step, intermediate_value in enumerate(objective_value):
                        optuna_trial.report(intermediate_value, step)
                        if optuna_trial.should_prune():
                            raise optuna.TrialPruned()

                return objective_value

            except optuna.TrialPruned:
                trial.status = OptimizationStatus.CANCELLED
                trial.error_message = "Pruned by Optuna"
                raise

            except Exception as e:
                trial.status = OptimizationStatus.FAILED
                trial.error_message = str(e)
                logger.error(f"Trial failed: {e}")
                raise

            finally:
                trial.end_time = datetime.now()
                self.trials.append(trial)

        # 运行优化
        self.study.optimize(
            wrapped_objective, n_trials=self.n_trials, show_progress_bar=True
        )

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # 获取最佳结果
        best_trial = self.study.best_trial
        self.best_parameters = best_trial.params
        self.best_value = best_trial.value

        # 创建结果
        result = OptimizationResult(
            optimization_id=self.optimization_id,
            best_parameters=self.best_parameters,
            best_value=self.best_value,
            all_trials=self.trials,
            convergence_history=self._get_convergence_history(),
            total_time_seconds=total_time,
            n_trials=len(self.trials),
            n_successful_trials=sum(
                1 for t in self.trials if t.status == OptimizationStatus.COMPLETED
            ),
            metadata={
                "optimizer_class": self.__class__.__name__,
                "maximize": self.maximize,
                "sampler": self.sampler.__class__.__name__,
                "pruner": self.pruner.__class__.__name__ if self.pruner else None,
                "optuna_best_trial_number": best_trial.number,
                "optuna_study_name": self.study.study_name,
            },
        )

        logger.info(f"Optimization completed. Best value: {self.best_value:.6f}")
        return result

    def _sample_parameters(self, trial: OptunaTrial) -> Dict[str, Any]:
        """使用Optuna trial采样参数

        Args:
            trial: Optuna试验对象

        Returns:
            参数字典
        """
        params = {}

        for param in self.parameter_space:
            if param.param_type == "float":
                if param.log_scale:
                    params[param.name] = trial.suggest_float(
                        param.name, param.low, param.high, log=True
                    )
                else:
                    params[param.name] = trial.suggest_float(
                        param.name, param.low, param.high
                    )
            elif param.param_type == "int":
                params[param.name] = trial.suggest_int(
                    param.name, param.low, param.high
                )
            elif param.param_type == "categorical":
                params[param.name] = trial.suggest_categorical(
                    param.name, param.choices
                )
            elif param.param_type == "bool":
                params[param.name] = trial.suggest_categorical(
                    param.name, [True, False]
                )

        return params

    def _get_convergence_history(self) -> List[float]:
        """获取收敛历史

        Returns:
            收敛历史列表
        """
        if not self.study:
            return []

        # 获取所有完成的试验
        completed_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if not completed_trials:
            return []

        # 计算累计最佳值
        convergence_history = []
        best_value = float("inf") if not self.maximize else float("-inf")

        for trial in completed_trials:
            if self.maximize:
                if trial.value > best_value:
                    best_value = trial.value
            else:
                if trial.value < best_value:
                    best_value = trial.value
            convergence_history.append(best_value)

        return convergence_history

    def suggest_parameters(self) -> Dict[str, Any]:
        """建议下一组参数（兼容基类接口）

        Returns:
            参数字典
        """
        # 这个方法主要用于兼容，实际Optuna在optimize中直接处理
        if not self.study:
            # 随机采样
            params = {}
            for param in self.parameter_space:
                params[param.name] = param.sample(self.random_state)
            return params
        else:
            # 使用Optuna的建议
            trial = self.study.ask()
            return self._sample_parameters(trial)

    def _update_optimization_state(self, trial: Trial) -> None:
        """更新优化器状态

        Args:
            trial: 完成的试验
        """
        # Optuna内部管理状态，这里不需要额外更新
        pass

    def get_parameter_importance(self) -> Dict[str, float]:
        """获取参数重要性

        Returns:
            参数重要性字典
        """
        if not self.study:
            return {}

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            logger.warning(f"Failed to calculate parameter importance: {e}")
            return {}

    def visualize_optimization(self) -> None:
        """可视化优化过程"""
        if not self.study:
            logger.warning("No study available for visualization")
            return

        try:
            import optuna.visualization as vis

            # 优化历史
            fig = vis.plot_optimization_history(self.study)
            fig.show()

            # 参数重要性
            fig = vis.plot_param_importances(self.study)
            fig.show()

            # 参数关系
            fig = vis.plot_parallel_coordinate(self.study)
            fig.show()

        except ImportError:
            logger.warning("Plotly not installed, cannot create visualizations")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
