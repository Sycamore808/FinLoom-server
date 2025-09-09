"""
时间序列特征提取模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

from common.logging_system import setup_logger
from common.exceptions import DataError

logger = setup_logger("time_series_features")

@dataclass
class TimeSeriesFeature:
    """时间序列特征"""
    name: str
    values: pd.Series
    description: str

class TimeSeriesFeatures:
    """时间序列特征提取器类"""
    
    def __init__(self):
        """初始化时间序列特征提取器"""
        pass
    
    def extract_momentum_features(self, data: pd.Series, windows: List[int] = [5, 10, 20]) -> Dict[str, TimeSeriesFeature]:
        """提取动量特征
        
        Args:
            data: 时间序列数据
            windows: 时间窗口列表
            
        Returns:
            动量特征字典
        """
        try:
            features = {}
            
            for window in windows:
                # 收益率
                returns = data.pct_change(window)
                features[f'returns_{window}'] = TimeSeriesFeature(
                    name=f'returns_{window}',
                    values=returns,
                    description=f'{window}-period returns'
                )
                
                # 动量
                momentum = data / data.shift(window) - 1
                features[f'momentum_{window}'] = TimeSeriesFeature(
                    name=f'momentum_{window}',
                    values=momentum,
                    description=f'{window}-period momentum'
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract momentum features: {e}")
            raise DataError(f"Momentum feature extraction failed: {e}")
    
    def extract_volatility_features(self, data: pd.Series, windows: List[int] = [5, 10, 20]) -> Dict[str, TimeSeriesFeature]:
        """提取波动率特征
        
        Args:
            data: 时间序列数据
            windows: 时间窗口列表
            
        Returns:
            波动率特征字典
        """
        try:
            features = {}
            
            for window in windows:
                # 滚动波动率
                volatility = data.rolling(window).std()
                features[f'volatility_{window}'] = TimeSeriesFeature(
                    name=f'volatility_{window}',
                    values=volatility,
                    description=f'{window}-period volatility'
                )
                
                # 滚动变异系数
                cv = data.rolling(window).std() / data.rolling(window).mean()
                features[f'cv_{window}'] = TimeSeriesFeature(
                    name=f'cv_{window}',
                    values=cv,
                    description=f'{window}-period coefficient of variation'
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract volatility features: {e}")
            raise DataError(f"Volatility feature extraction failed: {e}")
    
    def extract_trend_features(self, data: pd.Series, windows: List[int] = [5, 10, 20]) -> Dict[str, TimeSeriesFeature]:
        """提取趋势特征
        
        Args:
            data: 时间序列数据
            windows: 时间窗口列表
            
        Returns:
            趋势特征字典
        """
        try:
            features = {}
            
            for window in windows:
                # 线性趋势斜率
                def calculate_slope(series):
                    if len(series) < 2:
                        return np.nan
                    x = np.arange(len(series))
                    return np.polyfit(x, series, 1)[0]
                
                slope = data.rolling(window).apply(calculate_slope, raw=False)
                features[f'slope_{window}'] = TimeSeriesFeature(
                    name=f'slope_{window}',
                    values=slope,
                    description=f'{window}-period linear trend slope'
                )
                
                # 趋势强度
                trend_strength = abs(slope) / data.rolling(window).std()
                features[f'trend_strength_{window}'] = TimeSeriesFeature(
                    name=f'trend_strength_{window}',
                    values=trend_strength,
                    description=f'{window}-period trend strength'
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract trend features: {e}")
            raise DataError(f"Trend feature extraction failed: {e}")
    
    def extract_all_features(self, data: pd.Series) -> Dict[str, TimeSeriesFeature]:
        """提取所有时间序列特征
        
        Args:
            data: 时间序列数据
            
        Returns:
            所有特征字典
        """
        try:
            all_features = {}
            
            # 提取各类特征
            momentum_features = self.extract_momentum_features(data)
            volatility_features = self.extract_volatility_features(data)
            trend_features = self.extract_trend_features(data)
            
            all_features.update(momentum_features)
            all_features.update(volatility_features)
            all_features.update(trend_features)
            
            logger.info(f"Extracted {len(all_features)} time series features")
            return all_features
            
        except Exception as e:
            logger.error(f"Failed to extract all features: {e}")
            raise DataError(f"Feature extraction failed: {e}")
