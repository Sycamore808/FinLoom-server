"""
统计特征提取模块
提供各种统计指标的计算和特征工程功能
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from common.exceptions import DataError
from common.logging_system import setup_logger

logger = setup_logger("statistical_features")


@dataclass
class StatisticalFeatureConfig:
    """统计特征配置"""

    windows: List[int] = None
    percentiles: List[float] = None
    enable_distribution_features: bool = True
    enable_outlier_features: bool = True
    enable_correlation_features: bool = True

    def __post_init__(self):
        if self.windows is None:
            self.windows = [5, 10, 20, 60]
        if self.percentiles is None:
            self.percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]


class StatisticalFeatures:
    """统计特征提取器"""

    def __init__(self, config: Optional[StatisticalFeatureConfig] = None):
        """初始化统计特征提取器

        Args:
            config: 统计特征配置
        """
        self.config = config or StatisticalFeatureConfig()
        self.scaler_standard = StandardScaler()
        self.scaler_minmax = MinMaxScaler()

    def extract_basic_statistics(
        self, data: pd.Series, window: int = None
    ) -> Dict[str, float]:
        """提取基础统计特征

        Args:
            data: 时间序列数据
            window: 滚动窗口大小，None表示全样本

        Returns:
            统计特征字典
        """
        try:
            if window is not None:
                # 滚动统计
                data_window = data.rolling(window=window)
                features = {
                    f"mean_{window}": data_window.mean().iloc[-1],
                    f"std_{window}": data_window.std().iloc[-1],
                    f"var_{window}": data_window.var().iloc[-1],
                    f"min_{window}": data_window.min().iloc[-1],
                    f"max_{window}": data_window.max().iloc[-1],
                    f"median_{window}": data_window.median().iloc[-1],
                    f"skew_{window}": data_window.skew().iloc[-1],
                    f"kurt_{window}": data_window.kurt().iloc[-1],
                    f"range_{window}": (data_window.max() - data_window.min()).iloc[-1],
                    f"iqr_{window}": (
                        data_window.quantile(0.75) - data_window.quantile(0.25)
                    ).iloc[-1],
                }
            else:
                # 全样本统计
                features = {
                    "mean": data.mean(),
                    "std": data.std(),
                    "var": data.var(),
                    "min": data.min(),
                    "max": data.max(),
                    "median": data.median(),
                    "skew": data.skew(),
                    "kurt": data.kurtosis(),
                    "range": data.max() - data.min(),
                    "iqr": data.quantile(0.75) - data.quantile(0.25),
                }

            # 清理NaN值
            features = {k: v for k, v in features.items() if pd.notna(v)}
            return features

        except Exception as e:
            logger.error(f"Failed to extract basic statistics: {e}")
            return {}

    def extract_percentile_features(
        self, data: pd.Series, window: int = None
    ) -> Dict[str, float]:
        """提取分位数特征

        Args:
            data: 时间序列数据
            window: 滚动窗口大小

        Returns:
            分位数特征字典
        """
        try:
            features = {}

            for percentile in self.config.percentiles:
                if window is not None:
                    value = data.rolling(window=window).quantile(percentile).iloc[-1]
                    features[f"p{int(percentile * 100)}_{window}"] = value
                else:
                    value = data.quantile(percentile)
                    features[f"p{int(percentile * 100)}"] = value

            # 清理NaN值
            features = {k: v for k, v in features.items() if pd.notna(v)}
            return features

        except Exception as e:
            logger.error(f"Failed to extract percentile features: {e}")
            return {}

    def extract_distribution_features(
        self, data: pd.Series, window: int = None
    ) -> Dict[str, float]:
        """提取分布特征

        Args:
            data: 时间序列数据
            window: 滚动窗口大小

        Returns:
            分布特征字典
        """
        if not self.config.enable_distribution_features:
            return {}

        try:
            features = {}

            if window is not None:
                # 滚动分布特征
                data_window = data.rolling(window=window)

                # 正态性检验 (Jarque-Bera)
                recent_data = data.tail(window).dropna()
                if len(recent_data) > 8:  # JB test requires at least 8 observations
                    jb_stat, jb_pvalue = stats.jarque_bera(recent_data)
                    features[f"jb_stat_{window}"] = jb_stat
                    features[f"jb_pvalue_{window}"] = jb_pvalue
                    features[f"is_normal_{window}"] = 1.0 if jb_pvalue > 0.05 else 0.0

                # 偏度和峰度的标准化版本
                skew_val = data_window.skew().iloc[-1]
                kurt_val = data_window.kurt().iloc[-1]

                if pd.notna(skew_val):
                    features[f"skew_normalized_{window}"] = (
                        skew_val / np.sqrt(6.0 / window) if window > 6 else skew_val
                    )
                if pd.notna(kurt_val):
                    features[f"excess_kurtosis_{window}"] = kurt_val - 3  # 超额峰度

            else:
                # 全样本分布特征
                clean_data = data.dropna()
                if len(clean_data) > 8:
                    jb_stat, jb_pvalue = stats.jarque_bera(clean_data)
                    features["jb_stat"] = jb_stat
                    features["jb_pvalue"] = jb_pvalue
                    features["is_normal"] = 1.0 if jb_pvalue > 0.05 else 0.0

                features["skew_normalized"] = data.skew() / np.sqrt(
                    6.0 / len(clean_data)
                )
                features["excess_kurtosis"] = data.kurtosis() - 3

            # 清理NaN值
            features = {k: v for k, v in features.items() if pd.notna(v)}
            return features

        except Exception as e:
            logger.error(f"Failed to extract distribution features: {e}")
            return {}

    def extract_outlier_features(
        self, data: pd.Series, window: int = None
    ) -> Dict[str, float]:
        """提取异常值特征

        Args:
            data: 时间序列数据
            window: 滚动窗口大小

        Returns:
            异常值特征字典
        """
        if not self.config.enable_outlier_features:
            return {}

        try:
            features = {}

            if window is not None:
                # 滚动异常值检测
                data_window = data.rolling(window=window)

                # IQR方法检测异常值
                q1 = data_window.quantile(0.25)
                q3 = data_window.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # 当前值是否为异常值
                current_value = data.iloc[-1]
                is_outlier = (current_value < lower_bound.iloc[-1]) or (
                    current_value > upper_bound.iloc[-1]
                )
                features[f"is_outlier_{window}"] = 1.0 if is_outlier else 0.0

                # Z-score方法
                mean_val = data_window.mean().iloc[-1]
                std_val = data_window.std().iloc[-1]
                if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                    z_score = abs(current_value - mean_val) / std_val
                    features[f"z_score_{window}"] = z_score
                    features[f"is_extreme_{window}"] = 1.0 if z_score > 3 else 0.0

            else:
                # 全样本异常值检测
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outlier_count = ((data < lower_bound) | (data > upper_bound)).sum()
                features["outlier_ratio"] = outlier_count / len(data)

                # Z-score统计
                z_scores = np.abs(stats.zscore(data.dropna()))
                features["extreme_count"] = (z_scores > 3).sum()
                features["extreme_ratio"] = features["extreme_count"] / len(data)

            # 清理NaN值
            features = {k: v for k, v in features.items() if pd.notna(v)}
            return features

        except Exception as e:
            logger.error(f"Failed to extract outlier features: {e}")
            return {}

    def extract_correlation_features(
        self, data_dict: Dict[str, pd.Series], window: int = None
    ) -> Dict[str, float]:
        """提取相关性特征

        Args:
            data_dict: 多个时间序列数据字典
            window: 滚动窗口大小

        Returns:
            相关性特征字典
        """
        if not self.config.enable_correlation_features or len(data_dict) < 2:
            return {}

        try:
            features = {}

            # 转换为DataFrame
            df = pd.DataFrame(data_dict)

            if window is not None:
                # 滚动相关性
                for i, col1 in enumerate(df.columns):
                    for j, col2 in enumerate(df.columns):
                        if i < j:  # 避免重复计算
                            corr = (
                                df[col1].rolling(window=window).corr(df[col2]).iloc[-1]
                            )
                            if pd.notna(corr):
                                features[f"corr_{col1}_{col2}_{window}"] = corr
            else:
                # 全样本相关性
                corr_matrix = df.corr()
                for i, col1 in enumerate(df.columns):
                    for j, col2 in enumerate(df.columns):
                        if i < j:
                            corr = corr_matrix.loc[col1, col2]
                            if pd.notna(corr):
                                features[f"corr_{col1}_{col2}"] = corr

            return features

        except Exception as e:
            logger.error(f"Failed to extract correlation features: {e}")
            return {}

    def extract_stability_features(
        self, data: pd.Series, window: int = 20
    ) -> Dict[str, float]:
        """提取稳定性特征

        Args:
            data: 时间序列数据
            window: 滚动窗口大小

        Returns:
            稳定性特征字典
        """
        try:
            features = {}

            # 滚动变异系数
            rolling_mean = data.rolling(window=window).mean()
            rolling_std = data.rolling(window=window).std()
            cv = rolling_std / rolling_mean
            features[f"cv_{window}"] = cv.iloc[-1] if pd.notna(cv.iloc[-1]) else 0

            # 趋势稳定性
            rolling_data = data.rolling(window=window)

            def calculate_trend_stability(series):
                if len(series) < 3:
                    return np.nan
                x = np.arange(len(series))
                slope, _, r_value, _, _ = stats.linregress(x, series)
                return r_value**2  # R-squared作为趋势稳定性指标

            trend_stability = rolling_data.apply(calculate_trend_stability, raw=False)
            features[f"trend_stability_{window}"] = trend_stability.iloc[-1]

            # 水平稳定性 (标准差的变化)
            std_series = data.rolling(window=window).std()
            std_of_std = std_series.rolling(window=window).std().iloc[-1]
            features[f"std_stability_{window}"] = std_of_std

            # 清理NaN值
            features = {k: v for k, v in features.items() if pd.notna(v)}
            return features

        except Exception as e:
            logger.error(f"Failed to extract stability features: {e}")
            return {}

    def extract_all_statistical_features(
        self, data: Union[pd.Series, Dict[str, pd.Series]]
    ) -> Dict[str, float]:
        """提取所有统计特征

        Args:
            data: 单个时间序列或多个时间序列字典

        Returns:
            所有统计特征字典
        """
        try:
            all_features = {}

            if isinstance(data, dict):
                # 多个时间序列
                # 为每个序列提取特征
                for name, series in data.items():
                    series_features = self._extract_single_series_features(
                        series, prefix=name
                    )
                    all_features.update(series_features)

                # 提取相关性特征
                correlation_features = self.extract_correlation_features(data)
                all_features.update(correlation_features)

            else:
                # 单个时间序列
                single_features = self._extract_single_series_features(data)
                all_features.update(single_features)

            logger.info(f"Extracted {len(all_features)} statistical features")
            return all_features

        except Exception as e:
            logger.error(f"Failed to extract all statistical features: {e}")
            raise DataError(f"Statistical feature extraction failed: {e}")

    def _extract_single_series_features(
        self, data: pd.Series, prefix: str = ""
    ) -> Dict[str, float]:
        """为单个时间序列提取特征

        Args:
            data: 时间序列数据
            prefix: 特征名前缀

        Returns:
            特征字典
        """
        features = {}

        # 全样本特征
        basic_features = self.extract_basic_statistics(data)
        percentile_features = self.extract_percentile_features(data)
        distribution_features = self.extract_distribution_features(data)
        outlier_features = self.extract_outlier_features(data)

        if prefix:
            basic_features = {f"{prefix}_{k}": v for k, v in basic_features.items()}
            percentile_features = {
                f"{prefix}_{k}": v for k, v in percentile_features.items()
            }
            distribution_features = {
                f"{prefix}_{k}": v for k, v in distribution_features.items()
            }
            outlier_features = {f"{prefix}_{k}": v for k, v in outlier_features.items()}

        features.update(basic_features)
        features.update(percentile_features)
        features.update(distribution_features)
        features.update(outlier_features)

        # 滚动窗口特征
        for window in self.config.windows:
            if len(data) >= window:
                window_basic = self.extract_basic_statistics(data, window)
                window_percentile = self.extract_percentile_features(data, window)
                window_distribution = self.extract_distribution_features(data, window)
                window_outlier = self.extract_outlier_features(data, window)
                window_stability = self.extract_stability_features(data, window)

                if prefix:
                    window_basic = {f"{prefix}_{k}": v for k, v in window_basic.items()}
                    window_percentile = {
                        f"{prefix}_{k}": v for k, v in window_percentile.items()
                    }
                    window_distribution = {
                        f"{prefix}_{k}": v for k, v in window_distribution.items()
                    }
                    window_outlier = {
                        f"{prefix}_{k}": v for k, v in window_outlier.items()
                    }
                    window_stability = {
                        f"{prefix}_{k}": v for k, v in window_stability.items()
                    }

                features.update(window_basic)
                features.update(window_percentile)
                features.update(window_distribution)
                features.update(window_outlier)
                features.update(window_stability)

        return features

    def normalize_features(
        self, features: Dict[str, float], method: str = "standard"
    ) -> Dict[str, float]:
        """标准化特征

        Args:
            features: 特征字典
            method: 标准化方法 ("standard", "minmax")

        Returns:
            标准化后的特征字典
        """
        try:
            if not features:
                return features

            # 转换为数组
            feature_names = list(features.keys())
            feature_values = np.array(list(features.values())).reshape(-1, 1)

            # 标准化
            if method == "standard":
                normalized_values = self.scaler_standard.fit_transform(
                    feature_values
                ).flatten()
            elif method == "minmax":
                normalized_values = self.scaler_minmax.fit_transform(
                    feature_values
                ).flatten()
            else:
                raise ValueError(f"Unknown normalization method: {method}")

            # 转换回字典
            normalized_features = dict(zip(feature_names, normalized_values))

            # 清理NaN值
            normalized_features = {
                k: v for k, v in normalized_features.items() if pd.notna(v)
            }

            return normalized_features

        except Exception as e:
            logger.error(f"Failed to normalize features: {e}")
            return features


# 便捷函数
def extract_statistical_features(
    data: Union[pd.Series, Dict[str, pd.Series]],
    config: Optional[StatisticalFeatureConfig] = None,
) -> Dict[str, float]:
    """提取统计特征的便捷函数

    Args:
        data: 单个时间序列或多个时间序列字典
        config: 统计特征配置

    Returns:
        统计特征字典
    """
    extractor = StatisticalFeatures(config)
    return extractor.extract_all_statistical_features(data)
