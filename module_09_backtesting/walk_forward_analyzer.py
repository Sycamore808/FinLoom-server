"""
Walk-forward分析器模块
实现滚动窗口优化和样本外验证
"""

import json
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("walk_forward_analyzer")


@dataclass
class WalkForwardWindow:
    """Walk-forward窗口数据结构"""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    optimal_params: Dict[str, Any] = field(default_factory=dict)
    in_sample_performance: Dict[str, float] = field(default_factory=dict)
    out_sample_performance: Dict[str, float] = field(default_factory=dict)

    @property
    def train_days(self) -> int:
        """训练期天数"""
        return len(self.train_data)

    @property
    def test_days(self) -> int:
        """测试期天数"""
        return len(self.test_data)

    @property
    def efficiency_ratio(self) -> float:
        """效率比率（样本外/样本内性能）"""
        in_sample_sharpe = self.in_sample_performance.get("sharpe_ratio", 0)
        out_sample_sharpe = self.out_sample_performance.get("sharpe_ratio", 0)

        if in_sample_sharpe == 0:
            return 0.0
        return out_sample_sharpe / in_sample_sharpe


@dataclass
class WalkForwardConfig:
    """Walk-forward配置"""

    total_periods: int  # 总期数
    train_periods: int  # 训练期长度
    test_periods: int  # 测试期长度
    step_periods: int  # 步进长度
    optimization_metric: str = "sharpe_ratio"  # 优化指标
    anchored: bool = False  # 是否锚定起始点
    parallel: bool = True  # 是否并行执行
    max_workers: int = 4  # 最大工作线程数
    save_results: bool = True  # 是否保存结果

    def validate(self) -> bool:
        """验证配置有效性"""
        if self.train_periods <= 0:
            raise ValueError("Train periods must be positive")
        if self.test_periods <= 0:
            raise ValueError("Test periods must be positive")
        if self.step_periods <= 0:
            raise ValueError("Step periods must be positive")
        if self.train_periods + self.test_periods > self.total_periods:
            raise ValueError("Train + test periods exceed total periods")
        return True


@dataclass
class WalkForwardResult:
    """Walk-forward分析结果"""

    config: WalkForwardConfig
    windows: List[WalkForwardWindow]
    aggregate_metrics: Dict[str, float]
    parameter_stability: Dict[str, float]
    performance_decay: float
    robustness_score: float
    optimal_parameters_history: pd.DataFrame
    equity_curve: pd.Series

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "n_windows": len(self.windows),
            "aggregate_metrics": self.aggregate_metrics,
            "parameter_stability": self.parameter_stability,
            "performance_decay": self.performance_decay,
            "robustness_score": self.robustness_score,
            "avg_efficiency_ratio": np.mean([w.efficiency_ratio for w in self.windows]),
        }


class WalkForwardAnalyzer:
    """Walk-forward分析器类"""

    def __init__(self, config: WalkForwardConfig):
        """初始化Walk-forward分析器

        Args:
            config: Walk-forward配置
        """
        config.validate()
        self.config = config
        self.windows: List[WalkForwardWindow] = []
        self.optimization_function: Optional[Callable] = None
        self.backtest_function: Optional[Callable] = None

    def set_optimization_function(
        self, func: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, Any]]
    ) -> None:
        """设置优化函数

        Args:
            func: 优化函数，接收数据和参数范围，返回最优参数
        """
        self.optimization_function = func

    def set_backtest_function(
        self, func: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]]
    ) -> None:
        """设置回测函数

        Args:
            func: 回测函数，接收数据和参数，返回性能指标
        """
        self.backtest_function = func

    def run(
        self, data: pd.DataFrame, parameter_ranges: Dict[str, Any]
    ) -> WalkForwardResult:
        """运行Walk-forward分析

        Args:
            data: 完整数据集
            parameter_ranges: 参数搜索范围

        Returns:
            Walk-forward分析结果
        """
        if self.optimization_function is None:
            raise QuantSystemError("Optimization function not set")
        if self.backtest_function is None:
            raise QuantSystemError("Backtest function not set")

        logger.info(
            f"Starting walk-forward analysis with {self.config.total_periods} periods"
        )

        # 生成窗口
        self.windows = self._generate_windows(data)
        logger.info(f"Generated {len(self.windows)} windows")

        # 执行分析
        if self.config.parallel:
            self._run_parallel(parameter_ranges)
        else:
            self._run_sequential(parameter_ranges)

        # 汇总结果
        result = self._aggregate_results()

        # 保存结果
        if self.config.save_results:
            self._save_results(result)

        logger.info("Walk-forward analysis completed")
        return result

    def _generate_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """生成Walk-forward窗口

        Args:
            data: 完整数据集

        Returns:
            窗口列表
        """
        windows = []
        window_id = 0

        # 计算窗口起始位置
        if self.config.anchored:
            # 锚定模式：训练期始终从头开始
            train_start_idx = 0
            test_start_idx = self.config.train_periods

            while test_start_idx + self.config.test_periods <= len(data):
                train_end_idx = test_start_idx
                test_end_idx = test_start_idx + self.config.test_periods

                window = WalkForwardWindow(
                    window_id=window_id,
                    train_start=data.index[train_start_idx],
                    train_end=data.index[train_end_idx - 1],
                    test_start=data.index[test_start_idx],
                    test_end=data.index[test_end_idx - 1],
                    train_data=data.iloc[train_start_idx:train_end_idx],
                    test_data=data.iloc[test_start_idx:test_end_idx],
                )

                windows.append(window)
                window_id += 1

                # 步进
                test_start_idx += self.config.step_periods

        else:
            # 滚动模式：训练期和测试期都滚动
            current_idx = 0

            while (
                current_idx + self.config.train_periods + self.config.test_periods
                <= len(data)
            ):
                train_start_idx = current_idx
                train_end_idx = current_idx + self.config.train_periods
                test_start_idx = train_end_idx
                test_end_idx = test_start_idx + self.config.test_periods

                window = WalkForwardWindow(
                    window_id=window_id,
                    train_start=data.index[train_start_idx],
                    train_end=data.index[train_end_idx - 1],
                    test_start=data.index[test_start_idx],
                    test_end=data.index[test_end_idx - 1],
                    train_data=data.iloc[train_start_idx:train_end_idx],
                    test_data=data.iloc[test_start_idx:test_end_idx],
                )

                windows.append(window)
                window_id += 1

                # 步进
                current_idx += self.config.step_periods

        return windows

    def _run_sequential(self, parameter_ranges: Dict[str, Any]) -> None:
        """顺序执行Walk-forward分析

        Args:
            parameter_ranges: 参数搜索范围
        """
        for window in self.windows:
            self._process_window(window, parameter_ranges)
            logger.info(f"Processed window {window.window_id + 1}/{len(self.windows)}")

    def _run_parallel(self, parameter_ranges: Dict[str, Any]) -> None:
        """并行执行Walk-forward分析

        Args:
            parameter_ranges: 参数搜索范围
        """
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._process_window, window, parameter_ranges): window
                for window in self.windows
            }

            completed = 0
            for future in as_completed(futures):
                window = futures[future]
                try:
                    future.result()
                    completed += 1
                    logger.info(
                        f"Processed window {window.window_id} ({completed}/{len(self.windows)})"
                    )
                except Exception as e:
                    logger.error(f"Error processing window {window.window_id}: {e}")

    def _process_window(
        self, window: WalkForwardWindow, parameter_ranges: Dict[str, Any]
    ) -> None:
        """处理单个窗口

        Args:
            window: Walk-forward窗口
            parameter_ranges: 参数搜索范围
        """
        # 优化阶段（训练期）
        optimal_params = self.optimization_function(window.train_data, parameter_ranges)
        window.optimal_params = optimal_params

        # 样本内回测
        in_sample_metrics = self.backtest_function(window.train_data, optimal_params)
        window.in_sample_performance = in_sample_metrics

        # 样本外回测（测试期）
        out_sample_metrics = self.backtest_function(window.test_data, optimal_params)
        window.out_sample_performance = out_sample_metrics

    def _aggregate_results(self) -> WalkForwardResult:
        """汇总分析结果

        Returns:
            Walk-forward分析结果
        """
        # 收集所有性能指标
        in_sample_metrics = defaultdict(list)
        out_sample_metrics = defaultdict(list)

        for window in self.windows:
            for key, value in window.in_sample_performance.items():
                in_sample_metrics[key].append(value)
            for key, value in window.out_sample_performance.items():
                out_sample_metrics[key].append(value)

        # 计算汇总指标
        aggregate_metrics = {}
        for key in in_sample_metrics:
            if key in out_sample_metrics:
                aggregate_metrics[f"in_sample_{key}_mean"] = float(
                    np.mean(in_sample_metrics[key])
                )
                aggregate_metrics[f"in_sample_{key}_std"] = float(
                    np.std(in_sample_metrics[key])
                )
                aggregate_metrics[f"out_sample_{key}_mean"] = float(
                    np.mean(out_sample_metrics[key])
                )
                aggregate_metrics[f"out_sample_{key}_std"] = float(
                    np.std(out_sample_metrics[key])
                )

        # 计算参数稳定性
        parameter_stability = self._calculate_parameter_stability()

        # 计算性能衰减
        performance_decay = self._calculate_performance_decay()

        # 计算稳健性得分
        robustness_score = self._calculate_robustness_score()

        # 构建参数历史
        param_history = self._build_parameter_history()

        # 构建权益曲线
        equity_curve = self._build_equity_curve()

        return WalkForwardResult(
            config=self.config,
            windows=self.windows,
            aggregate_metrics=aggregate_metrics,
            parameter_stability=parameter_stability,
            performance_decay=performance_decay,
            robustness_score=robustness_score,
            optimal_parameters_history=param_history,
            equity_curve=equity_curve,
        )

    def _calculate_parameter_stability(self) -> Dict[str, float]:
        """计算参数稳定性

        Returns:
            参数稳定性指标
        """
        # 收集所有窗口的最优参数
        param_values = defaultdict(list)

        for window in self.windows:
            for param_name, param_value in window.optimal_params.items():
                if isinstance(param_value, (int, float)):
                    param_values[param_name].append(param_value)

        # 计算每个参数的稳定性
        stability = {}
        for param_name, values in param_values.items():
            if len(values) > 1:
                # 使用变异系数衡量稳定性
                mean_val = np.mean(values)
                std_val = np.std(values)

                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    stability[f"{param_name}_stability"] = float(1 / (1 + cv))
                else:
                    stability[f"{param_name}_stability"] = 0.0

                # 记录参数范围
                stability[f"{param_name}_min"] = float(np.min(values))
                stability[f"{param_name}_max"] = float(np.max(values))
                stability[f"{param_name}_mean"] = float(mean_val)
                stability[f"{param_name}_std"] = float(std_val)

        # 整体稳定性得分
        if stability:
            stability_scores = [
                v for k, v in stability.items() if k.endswith("_stability")
            ]
            stability["overall_stability"] = (
                float(np.mean(stability_scores)) if stability_scores else 0.0
            )
        else:
            stability["overall_stability"] = 0.0

        return stability

    def _calculate_performance_decay(self) -> float:
        """计算性能衰减

        Returns:
            性能衰减比率
        """
        metric = self.config.optimization_metric

        in_sample_values = []
        out_sample_values = []

        for window in self.windows:
            if metric in window.in_sample_performance:
                in_sample_values.append(window.in_sample_performance[metric])
            if metric in window.out_sample_performance:
                out_sample_values.append(window.out_sample_performance[metric])

        if not in_sample_values or not out_sample_values:
            return 0.0

        # 计算平均衰减
        avg_in_sample = np.mean(in_sample_values)
        avg_out_sample = np.mean(out_sample_values)

        if avg_in_sample == 0:
            return 0.0

        decay = (avg_in_sample - avg_out_sample) / abs(avg_in_sample)
        return float(decay)

    def _calculate_robustness_score(self) -> float:
        """计算稳健性得分

        Returns:
            稳健性得分（0-1）
        """
        scores = []

        # 效率比率得分
        efficiency_ratios = [w.efficiency_ratio for w in self.windows]
        if efficiency_ratios:
            avg_efficiency = np.mean(efficiency_ratios)
            efficiency_score = min(1.0, avg_efficiency)
            scores.append(efficiency_score)

        # 参数稳定性得分
        if hasattr(self, "_calculate_parameter_stability"):
            stability = self._calculate_parameter_stability()
            overall_stability = stability.get("overall_stability", 0)
            scores.append(overall_stability)

        # 性能衰减得分
        decay = self._calculate_performance_decay()
        decay_score = max(0, 1 - abs(decay))
        scores.append(decay_score)

        # 一致性得分（样本外正收益率）
        out_sample_positive = 0
        for window in self.windows:
            if window.out_sample_performance.get("total_return", 0) > 0:
                out_sample_positive += 1

        if self.windows:
            consistency_score = out_sample_positive / len(self.windows)
            scores.append(consistency_score)

        # 综合得分
        robustness = np.mean(scores) if scores else 0.0
        return float(robustness)

    def _build_parameter_history(self) -> pd.DataFrame:
        """构建参数历史DataFrame

        Returns:
            参数历史DataFrame
        """
        param_records = []

        for window in self.windows:
            record = {
                "window_id": window.window_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
            }

            # 添加最优参数
            for param_name, param_value in window.optimal_params.items():
                record[f"param_{param_name}"] = param_value

            # 添加性能指标
            for metric_name, metric_value in window.in_sample_performance.items():
                record[f"in_sample_{metric_name}"] = metric_value

            for metric_name, metric_value in window.out_sample_performance.items():
                record[f"out_sample_{metric_name}"] = metric_value

            # 添加效率比率
            record["efficiency_ratio"] = window.efficiency_ratio

            param_records.append(record)

        return pd.DataFrame(param_records)

    def _build_equity_curve(self) -> pd.Series:
        """构建综合权益曲线

        Returns:
            权益曲线Series
        """
        equity_data = []

        for window in self.windows:
            # 获取测试期收益
            if "daily_returns" in window.out_sample_performance:
                # 如果有日收益数据
                returns = window.out_sample_performance["daily_returns"]
                for date, ret in returns.items():
                    equity_data.append(
                        {"date": date, "return": ret, "window_id": window.window_id}
                    )
            else:
                # 使用总收益平摊到测试期
                total_return = window.out_sample_performance.get("total_return", 0)
                test_days = window.test_days

                if test_days > 0:
                    daily_return = (1 + total_return) ** (1 / test_days) - 1

                    # 生成日期序列
                    date_range = pd.date_range(
                        start=window.test_start, end=window.test_end, freq="D"
                    )

                    for date in date_range:
                        equity_data.append(
                            {
                                "date": date,
                                "return": daily_return,
                                "window_id": window.window_id,
                            }
                        )

        if not equity_data:
            return pd.Series(dtype=float)

        # 构建DataFrame并计算累计收益
        df = pd.DataFrame(equity_data)
        df = df.sort_values("date")

        # 去重（如果有重叠的日期，取最新的窗口）
        df = df.drop_duplicates(subset=["date"], keep="last")

        # 计算累计收益
        equity_curve = (1 + df.set_index("date")["return"]).cumprod()

        return equity_curve

    def _save_results(self, result: WalkForwardResult) -> None:
        """保存分析结果

        Args:
            result: Walk-forward分析结果
        """
        # 保存为pickle文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"walk_forward_results_{timestamp}.pkl"

        with open(filename, "wb") as f:
            pickle.dump(result, f)

        logger.info(f"Results saved to {filename}")

        # 同时保存为JSON（仅保存可序列化部分）
        json_filename = f"walk_forward_summary_{timestamp}.json"

        summary = {
            "config": {
                "total_periods": self.config.total_periods,
                "train_periods": self.config.train_periods,
                "test_periods": self.config.test_periods,
                "step_periods": self.config.step_periods,
                "optimization_metric": self.config.optimization_metric,
                "anchored": self.config.anchored,
            },
            "results": result.to_dict(),
            "windows_summary": [
                {
                    "window_id": w.window_id,
                    "efficiency_ratio": w.efficiency_ratio,
                    "in_sample_sharpe": w.in_sample_performance.get("sharpe_ratio", 0),
                    "out_sample_sharpe": w.out_sample_performance.get(
                        "sharpe_ratio", 0
                    ),
                }
                for w in result.windows
            ],
        }

        with open(json_filename, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Summary saved to {json_filename}")


# 模块级别函数
def create_walk_forward_analyzer(
    train_periods: int = 252,
    test_periods: int = 63,
    step_periods: int = 21,
    anchored: bool = False,
) -> WalkForwardAnalyzer:
    """创建Walk-forward分析器的便捷函数

    Args:
        train_periods: 训练期长度
        test_periods: 测试期长度
        step_periods: 步进长度
        anchored: 是否锚定

    Returns:
        Walk-forward分析器实例
    """
    config = WalkForwardConfig(
        total_periods=train_periods + test_periods,
        train_periods=train_periods,
        test_periods=test_periods,
        step_periods=step_periods,
        anchored=anchored,
    )

    return WalkForwardAnalyzer(config)


def run_walk_forward_analysis(
    data: pd.DataFrame,
    optimization_func: Callable,
    backtest_func: Callable,
    parameter_ranges: Dict[str, Any],
    config: Optional[WalkForwardConfig] = None,
) -> WalkForwardResult:
    """运行Walk-forward分析的便捷函数

    Args:
        data: 市场数据
        optimization_func: 优化函数
        backtest_func: 回测函数
        parameter_ranges: 参数范围
        config: 配置（可选）

    Returns:
        分析结果
    """
    if config is None:
        config = WalkForwardConfig(
            total_periods=len(data), train_periods=252, test_periods=63, step_periods=21
        )

    analyzer = WalkForwardAnalyzer(config)
    analyzer.set_optimization_function(optimization_func)
    analyzer.set_backtest_function(backtest_func)

    return analyzer.run(data, parameter_ranges)


def analyze_walk_forward_stability(result: WalkForwardResult) -> Dict[str, Any]:
    """分析Walk-forward稳定性的便捷函数

    Args:
        result: Walk-forward结果

    Returns:
        稳定性分析字典
    """
    # 效率比率分析
    efficiency_ratios = [w.efficiency_ratio for w in result.windows]

    # 参数漂移分析
    param_history = result.optimal_parameters_history
    param_columns = [col for col in param_history.columns if col.startswith("param_")]

    param_drift = {}
    for col in param_columns:
        values = param_history[col].dropna()
        if len(values) > 1:
            # 计算趋势
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            param_drift[col] = float(slope)

    # 性能一致性
    out_sample_sharpes = [
        w.out_sample_performance.get("sharpe_ratio", 0) for w in result.windows
    ]

    # 计算滚动相关性
    if len(out_sample_sharpes) > 2:
        rolling_corr = []
        window_size = min(3, len(out_sample_sharpes) - 1)
        for i in range(len(out_sample_sharpes) - window_size):
            window_values = out_sample_sharpes[i : i + window_size + 1]
            if len(window_values) > 1:
                corr = np.corrcoef(window_values[:-1], window_values[1:])[0, 1]
                rolling_corr.append(corr)
        avg_correlation = np.mean(rolling_corr) if rolling_corr else 0
    else:
        avg_correlation = 0

    return {
        "avg_efficiency_ratio": float(np.mean(efficiency_ratios)),
        "efficiency_ratio_std": float(np.std(efficiency_ratios)),
        "min_efficiency_ratio": float(np.min(efficiency_ratios)),
        "max_efficiency_ratio": float(np.max(efficiency_ratios)),
        "parameter_drift": param_drift,
        "performance_correlation": float(avg_correlation),
        "robustness_score": float(result.robustness_score),
        "performance_decay": float(result.performance_decay),
        "positive_windows_ratio": float(
            sum(
                1
                for w in result.windows
                if w.out_sample_performance.get("total_return", 0) > 0
            )
            / len(result.windows)
        ),
    }
