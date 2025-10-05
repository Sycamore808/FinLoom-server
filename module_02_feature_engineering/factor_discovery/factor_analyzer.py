#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
因子分析器模块
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.exceptions import DataError
from common.logging_system import setup_logger

logger = setup_logger("factor_analyzer")


@dataclass
class FactorConfig:
    """因子分析配置"""

    lookback_period: int = 252  # 回看期
    forward_period: int = 5  # 前向期
    min_periods: int = 60  # 最小有效期数
    neutralize: bool = False  # 是否中性化
    standardize: bool = True  # 是否标准化
    winsorize: float = 0.05  # winsorize比例
    max_epochs: int = 100  # 最大迭代次数

    @property
    def epochs(self) -> int:
        """向后兼容的epochs属性"""
        return self.max_epochs


@dataclass
class FactorResult:
    """因子分析结果"""

    factor_name: str
    factor_values: pd.Series
    ic: float
    ir: float
    rank_ic: float
    turnover: float
    decay: float


class FactorAnalyzer:
    """因子分析器类"""

    def __init__(self):
        """初始化因子分析器"""
        pass

    def calculate_ic(self, factor_values: pd.Series, returns: pd.Series) -> float:
        """计算信息系数(IC)

        Args:
            factor_values: 因子值
            returns: 收益率

        Returns:
            信息系数
        """
        try:
            # 对齐数据
            common_index = factor_values.index.intersection(returns.index)
            factor_aligned = factor_values.loc[common_index]
            returns_aligned = returns.loc[common_index]

            # 计算相关系数
            ic = factor_aligned.corr(returns_aligned)
            return ic if not np.isnan(ic) else 0.0

        except Exception as e:
            logger.error(f"Failed to calculate IC: {e}")
            return 0.0

    def calculate_rank_ic(self, factor_values: pd.Series, returns: pd.Series) -> float:
        """计算排序信息系数(Rank IC)

        Args:
            factor_values: 因子值
            returns: 收益率

        Returns:
            排序信息系数
        """
        try:
            # 对齐数据
            common_index = factor_values.index.intersection(returns.index)
            factor_aligned = factor_values.loc[common_index]
            returns_aligned = returns.loc[common_index]

            # 计算排序相关系数
            rank_ic = factor_aligned.rank().corr(returns_aligned.rank())
            return rank_ic if not np.isnan(rank_ic) else 0.0

        except Exception as e:
            logger.error(f"Failed to calculate Rank IC: {e}")
            return 0.0

    def analyze_factor(
        self, factor_values: pd.Series, returns: pd.Series
    ) -> FactorResult:
        """分析因子

        Args:
            factor_values: 因子值
            returns: 收益率

        Returns:
            因子分析结果
        """
        try:
            ic = self.calculate_ic(factor_values, returns)
            rank_ic = self.calculate_rank_ic(factor_values, returns)

            # 计算信息比率(IR) - 简化版本
            ir = ic / (factor_values.std() + 1e-8)

            # 计算换手率 - 简化版本
            turnover = abs(factor_values.diff()).mean() / (
                abs(factor_values).mean() + 1e-8
            )

            # 计算衰减 - 简化版本
            decay = abs(ic) * 0.1  # 简化的衰减计算

            return FactorResult(
                factor_name="unknown",
                factor_values=factor_values,
                ic=ic,
                ir=ir,
                rank_ic=rank_ic,
                turnover=turnover,
                decay=decay,
            )

        except Exception as e:
            logger.error(f"Failed to analyze factor: {e}")
            raise DataError(f"Factor analysis failed: {e}")
