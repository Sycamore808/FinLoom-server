"""
策略参数空间定义模块
提供常用的策略参数空间定义
"""

from typing import Dict, List

from module_07_optimization.base_optimizer import Parameter


def create_ma_crossover_space() -> List[Parameter]:
    """创建移动平均交叉策略参数空间

    Returns:
        参数列表
    """
    return [
        Parameter(
            name="short_window",
            param_type="int",
            low=5,
            high=50,
            default=10,
        ),
        Parameter(
            name="long_window",
            param_type="int",
            low=20,
            high=200,
            default=50,
        ),
        Parameter(
            name="stop_loss",
            param_type="float",
            low=0.01,
            high=0.20,
            default=0.05,
        ),
    ]


def create_rsi_strategy_space() -> List[Parameter]:
    """创建RSI策略参数空间

    Returns:
        参数列表
    """
    return [
        Parameter(
            name="rsi_period",
            param_type="int",
            low=5,
            high=30,
            default=14,
        ),
        Parameter(
            name="oversold_threshold",
            param_type="float",
            low=20.0,
            high=40.0,
            default=30.0,
        ),
        Parameter(
            name="overbought_threshold",
            param_type="float",
            low=60.0,
            high=80.0,
            default=70.0,
        ),
        Parameter(
            name="position_size",
            param_type="float",
            low=0.1,
            high=1.0,
            default=0.5,
        ),
    ]


def create_bollinger_bands_space() -> List[Parameter]:
    """创建布林带策略参数空间

    Returns:
        参数列表
    """
    return [
        Parameter(
            name="bb_period",
            param_type="int",
            low=10,
            high=50,
            default=20,
        ),
        Parameter(
            name="bb_std",
            param_type="float",
            low=1.0,
            high=3.0,
            default=2.0,
        ),
        Parameter(
            name="entry_threshold",
            param_type="float",
            low=0.0,
            high=0.5,
            default=0.1,
        ),
        Parameter(
            name="exit_threshold",
            param_type="float",
            low=0.0,
            high=0.5,
            default=0.05,
        ),
    ]


def create_macd_strategy_space() -> List[Parameter]:
    """创建MACD策略参数空间

    Returns:
        参数列表
    """
    return [
        Parameter(
            name="fast_period",
            param_type="int",
            low=8,
            high=20,
            default=12,
        ),
        Parameter(
            name="slow_period",
            param_type="int",
            low=20,
            high=40,
            default=26,
        ),
        Parameter(
            name="signal_period",
            param_type="int",
            low=5,
            high=15,
            default=9,
        ),
        Parameter(
            name="min_signal_strength",
            param_type="float",
            low=0.0,
            high=0.01,
            default=0.001,
        ),
    ]


def create_mean_reversion_space() -> List[Parameter]:
    """创建均值回归策略参数空间

    Returns:
        参数列表
    """
    return [
        Parameter(
            name="lookback_period",
            param_type="int",
            low=10,
            high=100,
            default=20,
        ),
        Parameter(
            name="entry_z_score",
            param_type="float",
            low=1.0,
            high=3.0,
            default=2.0,
        ),
        Parameter(
            name="exit_z_score",
            param_type="float",
            low=0.0,
            high=1.0,
            default=0.5,
        ),
        Parameter(
            name="max_holding_period",
            param_type="int",
            low=1,
            high=20,
            default=5,
        ),
    ]


def create_momentum_strategy_space() -> List[Parameter]:
    """创建动量策略参数空间

    Returns:
        参数列表
    """
    return [
        Parameter(
            name="momentum_period",
            param_type="int",
            low=5,
            high=60,
            default=20,
        ),
        Parameter(
            name="holding_period",
            param_type="int",
            low=1,
            high=20,
            default=5,
        ),
        Parameter(
            name="momentum_threshold",
            param_type="float",
            low=0.0,
            high=0.10,
            default=0.02,
        ),
        Parameter(
            name="rebalance_frequency",
            param_type="int",
            low=1,
            high=10,
            default=5,
        ),
    ]


# 策略参数空间注册表
STRATEGY_SPACES = {
    "ma_crossover": create_ma_crossover_space,
    "rsi": create_rsi_strategy_space,
    "bollinger_bands": create_bollinger_bands_space,
    "macd": create_macd_strategy_space,
    "mean_reversion": create_mean_reversion_space,
    "momentum": create_momentum_strategy_space,
}


def get_strategy_space(strategy_name: str) -> List[Parameter]:
    """获取策略参数空间

    Args:
        strategy_name: 策略名称

    Returns:
        参数列表

    Raises:
        ValueError: 如果策略名称未知
    """
    if strategy_name not in STRATEGY_SPACES:
        available = ", ".join(STRATEGY_SPACES.keys())
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")

    return STRATEGY_SPACES[strategy_name]()
