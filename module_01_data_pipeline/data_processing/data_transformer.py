# -*- coding: utf-8 -*-
"""
数据转换模块
提供标准化、归一化、特征工程等常用数据转换方法
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from common.exceptions import DataError
from common.logging_system import setup_logger

logger = setup_logger("data_transformer")


class DataTransformer:
    """
    数据转换器，支持标准化、归一化、特征生成等
    """

    def __init__(self):
        pass

    def standardize(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        对指定列做标准化（均值为0，方差为1）
        Args:
                df: 输入DataFrame
                columns: 需要标准化的列
        Returns:
                标准化后的DataFrame
        Raises:
                DataError: 数据异常
        """
        try:
            cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
            df_std = df.copy()
            for col in cols:
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:
                    logger.warning(f"Column {col} std=0, skip standardization")
                    continue
                df_std[col] = (df[col] - mean) / std
            logger.info(f"Standardized columns: {cols}")
            return df_std
        except Exception as e:
            logger.error(f"Standardization failed: {e}")
            raise DataError(f"Standardization failed: {e}")

    def minmax_scale(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        对指定列做归一化（0-1）
        Args:
                df: 输入DataFrame
                columns: 需要归一化的列
        Returns:
                归一化后的DataFrame
        Raises:
                DataError: 数据异常
        """
        try:
            cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
            df_scaled = df.copy()
            for col in cols:
                min_v = df[col].min()
                max_v = df[col].max()
                if max_v == min_v:
                    logger.warning(f"Column {col} min=max, skip scaling")
                    continue
                df_scaled[col] = (df[col] - min_v) / (max_v - min_v)
            logger.info(f"MinMax scaled columns: {cols}")
            return df_scaled
        except Exception as e:
            logger.error(f"MinMax scaling failed: {e}")
            raise DataError(f"MinMax scaling failed: {e}")

    def generate_features(
        self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        生成常用技术指标特征（如均线、波动率等）
        Args:
                df: 输入DataFrame，需包含close列
                windows: 均线窗口
        Returns:
                增加特征后的DataFrame
        Raises:
                DataError: 数据异常
        """
        try:
            df_feat = df.copy()
            if "close" not in df_feat.columns:
                raise DataError("Input DataFrame must contain 'close' column")
            for w in windows:
                df_feat[f"ma_{w}"] = df_feat["close"].rolling(window=w).mean()
                df_feat[f"vol_{w}"] = df_feat["close"].rolling(window=w).std()
            logger.info(f"Generated features for windows: {windows}")
            return df_feat
        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            raise DataError(f"Feature generation failed: {e}")
