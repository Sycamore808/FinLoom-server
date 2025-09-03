"""
策略优化器模块
专门用于交易策略参数优化
"""

import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from common.data_structures import Signal
from common.logging_system import setup_logger
from module_07_optimization.base_optimizer import Parameter
from module_07_optimization.hyperparameter_tuning.optuna_optimizer import (
    OptunaOptimizer,
)

logger = setup_logger("strategy_optimizer")


@dataclass
class StrategyPerformance:
    """策略性能指标"""

    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float

    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_trade_return": self.avg_trade_return,
            "volatility": self.volatility,
            "calmar_ratio": self.calmar_ratio,
        }


class StrategyOptimizer:
    """策略优化器

    优化交易策略参数以获得最佳性能
    """

    def __init__(
        self,
        strategy_class: type,
        market_data: pd.DataFrame,
        optimization_metric: str = "sharpe_ratio",
        test_split: float = 0.2,
        walk_forward_windows: Optional[int] = None,
        n_jobs: int = 1,
    ):
        """初始化策略优化器

        Args:
            strategy_class: 策略类
            market_data: 市场数据
            optimization_metric: 优化指标
            test_split: 测试集比例
            walk_forward_windows: 前进窗口数（用于Walk Forward分析）
            n_jobs: 并行任务数
        """
        self.strategy_class = strategy_class
        self.market_data = market_data
        self.optimization_metric = optimization_metric
        self.test_split = test_split
        self.walk_forward_windows = walk_forward_windows
        self.n_jobs = n_jobs

        # 分割训练/测试数据
        self._split_data()

    def _split_data(self) -> None:
        """分割训练和测试数据"""
        n_samples = len(self.market_data)
        split_idx = int(n_samples * (1 - self.test_split))

        self.train_data = self.market_data.iloc[:split_idx]
        self.test_data = self.market_data.iloc[split_idx:]

        logger.info(
            f"Data split: {len(self.train_data)} train, {len(self.test_data)} test samples"
        )

    def optimize(
        self,
        parameter_space: List[Parameter],
        n_trials: int = 100,
        optimizer_type: str = "optuna",
    ) -> Dict[str, Any]:
        """优化策略参数

        Args:
            parameter_space: 参数空间
            n_trials: 试验次数
            optimizer_type: 优化器类型

        Returns:
            优化结果
        """

        # 定义目标函数
        def objective(params: Dict[str, Any]) -> float:
            return self._evaluate_strategy(params, self.train_data)

        # 创建优化器
        if optimizer_type == "optuna":
            optimizer = OptunaOptimizer(
                parameter_space=parameter_space,
                objective_function=objective,
                maximize=True,  # 最大化性能指标
                n_trials=n_trials,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # 执行优化
        result = optimizer.optimize()

        # 在测试集上评估最佳参数
        test_performance = self._evaluate_strategy_full(
            result.best_parameters, self.test_data
        )

        # Walk Forward分析
        wf_results = None
        if self.walk_forward_windows:
            wf_results = self._walk_forward_analysis(
                parameter_space,
                n_trials=max(10, n_trials // 10),  # 减少每个窗口的试验数
            )

        return {
            "best_parameters": result.best_parameters,
            "train_performance": result.best_value,
            "test_performance": test_performance.to_dict(),
            "optimization_result": result,
            "walk_forward_results": wf_results,
        }

    def _evaluate_strategy(self, params: Dict[str, Any], data: pd.DataFrame) -> float:
        """评估策略性能（返回单一指标）

        Args:
            params: 策略参数
            data: 市场数据

        Returns:
            性能指标值
        """
        performance = self._evaluate_strategy_full(params, data)

        # 返回指定的优化指标
        return getattr(performance, self.optimization_metric)

    def _evaluate_strategy_full(
        self, params: Dict[str, Any], data: pd.DataFrame
    ) -> StrategyPerformance:
        """完整评估策略性能

        Args:
            params: 策略参数
            data: 市场数据

        Returns:
            完整性能指标
        """
        try:
            # 创建策略实例
            strategy = self.strategy_class(**params)

            # 运行策略
            signals = []
            positions = []
            returns = []

            for i in range(len(data)):
                # 获取历史数据
                history = data.iloc[: i + 1]

                # 生成信号
                signal = strategy.generate_signal(history)
                signals.append(signal)

                # 计算收益
                if i > 0 and signal:
                    if signal.action == "BUY":
                        position = signal.quantity
                    elif signal.action == "SELL":
                        position = -signal.quantity
                    else:
                        position = 0

                    # 简单收益计算
                    price_change = data.iloc[i]["close"] - data.iloc[i - 1]["close"]
                    ret = position * price_change / data.iloc[i - 1]["close"]
                    returns.append(ret)
                else:
                    returns.append(0)

            # 计算性能指标
            returns = np.array(returns)

            # 总收益
            total_return = (1 + returns).prod() - 1

            # 年化收益
            n_days = len(data)
            annual_return = (1 + total_return) ** (252 / n_days) - 1

            # 夏普比率
            if returns.std() > 0:
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            else:
                sharpe_ratio = 0

            # 最大回撤
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())

            # 胜率
            winning_trades = sum(1 for r in returns if r > 0)
            total_trades = sum(1 for r in returns if r != 0)
            win_rate = winning_trades / max(1, total_trades)

            # 盈亏比
            profits = returns[returns > 0]
            losses = returns[returns < 0]
            if len(losses) > 0 and losses.sum() != 0:
                profit_factor = abs(profits.sum() / losses.sum())
            else:
                profit_factor = float("inf") if len(profits) > 0 else 0

            # 平均交易收益
            avg_trade_return = returns[returns != 0].mean() if total_trades > 0 else 0

            # 波动率
            volatility = returns.std() * np.sqrt(252)

            # Calmar比率
            calmar_ratio = annual_return / max(max_drawdown, 0.01)

            return StrategyPerformance(
                total_return=total_return,
                annual_return=annual_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=min(profit_factor, 100),  # 限制最大值
                total_trades=total_trades,
                avg_trade_return=avg_trade_return,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
            )

        except Exception as e:
            logger.error(f"Strategy evaluation failed: {e}")
            # 返回最差的性能
            return StrategyPerformance(
                total_return=-1.0,
                annual_return=-1.0,
                sharpe_ratio=-10.0,
                max_drawdown=1.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_return=-1.0,
                volatility=10.0,
                calmar_ratio=-10.0,
            )

    def _walk_forward_analysis(
        self, parameter_space: List[Parameter], n_trials: int = 10
    ) -> List[Dict[str, Any]]:
        """Walk Forward分析

        Args:
            parameter_space: 参数空间
            n_trials: 每个窗口的试验次数

        Returns:
            各窗口的优化结果
        """
        logger.info(
            f"Starting Walk Forward Analysis with {self.walk_forward_windows} windows"
        )

        results = []
        window_size = len(self.market_data) // (self.walk_forward_windows + 1)

        for i in range(self.walk_forward_windows):
            # 定义训练和测试窗口
            train_start = i * window_size
            train_end = (i + 1) * window_size
            test_start = train_end
            test_end = min(test_start + window_size, len(self.market_data))

            train_data = self.market_data.iloc[train_start:train_end]
            test_data = self.market_data.iloc[test_start:test_end]

            # 在训练窗口优化
            def objective(params: Dict[str, Any]) -> float:
                return self._evaluate_strategy(params, train_data)

            optimizer = OptunaOptimizer(
                parameter_space=parameter_space,
                objective_function=objective,
                maximize=True,
                n_trials=n_trials,
            )

            opt_result = optimizer.optimize()

            # 在测试窗口评估
            test_performance = self._evaluate_strategy_full(
                opt_result.best_parameters, test_data
            )

            results.append(
                {
                    "window": i,
                    "train_period": (train_start, train_end),
                    "test_period": (test_start, test_end),
                    "best_parameters": opt_result.best_parameters,
                    "train_performance": opt_result.best_value,
                    "test_performance": test_performance.to_dict(),
                }
            )

            logger.info(f"Window {i + 1}/{self.walk_forward_windows} completed")

        return results

    def parallel_evaluate(
        self, parameter_sets: List[Dict[str, Any]]
    ) -> List[StrategyPerformance]:
        """并行评估多组参数

        Args:
            parameter_sets: 参数集合列表

        Returns:
            性能列表
        """
        if self.n_jobs == 1:
            # 串行执行
            return [
                self._evaluate_strategy_full(params, self.train_data)
                for params in parameter_sets
            ]
        else:
            # 并行执行
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.n_jobs
            ) as executor:
                eval_func = partial(self._evaluate_strategy_full, data=self.train_data)
                results = list(executor.map(eval_func, parameter_sets))
            return results
