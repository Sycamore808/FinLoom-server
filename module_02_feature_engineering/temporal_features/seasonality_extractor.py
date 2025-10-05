#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
季节性提取模块
检测和分析时间序列中的季节性模式
"""

import warnings

warnings.filterwarnings("ignore")

import calendar
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
    from scipy.fft import fft, fftfreq
    from scipy.signal import find_peaks, periodogram
except ImportError:
    stats = None
    fft = None
    fftfreq = None
    periodogram = None
    find_peaks = None

try:
    from statsmodels.tsa.seasonal import STL, seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    seasonal_decompose = None
    STL = None
    adfuller = None

logger = logging.getLogger(__name__)


@dataclass
class SeasonalityConfig:
    """季节性配置"""

    method: str = "stl"  # 分解方法
    seasonal_periods: List[int] = None  # 季节性周期
    trend_window: int = 252  # 趋势窗口
    seasonal_window: int = "periodic"  # 季节性窗口
    detect_multiple_seasons: bool = True  # 检测多重季节性
    significance_level: float = 0.05  # 显著性水平
    min_period: int = 2  # 最小周期
    max_period: int = 252  # 最大周期


class SeasonalityExtractor:
    """季节性提取器"""

    def __init__(self, config: Optional[SeasonalityConfig] = None):
        """初始化季节性提取器

        Args:
            config: 季节性配置
        """
        self.config = config or SeasonalityConfig()

        if self.config.seasonal_periods is None:
            # 默认检测的季节性周期
            self.config.seasonal_periods = [5, 7, 21, 63, 252]  # 周、月、季、年

        self.seasonal_components = {}
        self.trend_components = {}
        self.residual_components = {}
        self.seasonal_strengths = {}

        if seasonal_decompose is None:
            logger.warning(
                "statsmodels not available. Some seasonality features will be limited."
            )

    def extract_seasonality(
        self, data: pd.Series, method: Optional[str] = None
    ) -> Dict[str, pd.Series]:
        """提取季节性特征

        Args:
            data: 时间序列数据
            method: 分解方法

        Returns:
            季节性分解结果
        """
        try:
            method = method or self.config.method
            logger.info(f"Extracting seasonality using {method} method...")

            if method == "stl":
                return self._stl_decomposition(data)
            elif method == "classical":
                return self._classical_decomposition(data)
            elif method == "x13":
                return self._x13_decomposition(data)
            elif method == "fourier":
                return self._fourier_decomposition(data)
            elif method == "custom":
                return self._custom_decomposition(data)
            else:
                raise ValueError(f"Unknown decomposition method: {method}")

        except Exception as e:
            logger.error(f"Failed to extract seasonality: {e}")
            return {}

    def _stl_decomposition(self, data: pd.Series) -> Dict[str, pd.Series]:
        """STL分解"""
        try:
            if STL is None:
                logger.warning("Using classical decomposition as STL alternative")
                return self._classical_decomposition(data)

            # 确定季节性周期
            period = self._determine_primary_period(data)

            if period < 2:
                logger.warning(
                    "Cannot determine seasonal period, using classical method"
                )
                return self._classical_decomposition(data)

            # STL分解
            stl = STL(
                data,
                seasonal=min(period, 13),
                trend=self.config.trend_window,
                robust=True,
            )
            result = stl.fit()

            decomposition = {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "residual": result.resid,
                "observed": data,
            }

            # 计算季节性强度
            self._calculate_seasonal_strength(decomposition)

            return decomposition

        except Exception as e:
            logger.error(f"Failed to run STL decomposition: {e}")
            return {}

    def _classical_decomposition(self, data: pd.Series) -> Dict[str, pd.Series]:
        """经典分解"""
        try:
            if seasonal_decompose is None:
                logger.error("statsmodels not available")
                return {}

            # 确定季节性周期
            period = self._determine_primary_period(data)

            if period < 2:
                logger.warning("Cannot determine seasonal period")
                return {}

            # 经典分解
            decomposition = seasonal_decompose(data, model="additive", period=period)

            result = {
                "trend": decomposition.trend,
                "seasonal": decomposition.seasonal,
                "residual": decomposition.resid,
                "observed": data,
            }

            # 计算季节性强度
            self._calculate_seasonal_strength(result)

            return result

        except Exception as e:
            logger.error(f"Failed to run classical decomposition: {e}")
            return {}

    def _x13_decomposition(self, data: pd.Series) -> Dict[str, pd.Series]:
        """X-13分解（简化实现）"""
        try:
            # 使用STL作为X-13的替代
            logger.warning("Using STL as X-13 alternative")
            return self._stl_decomposition(data)

        except Exception as e:
            logger.error(f"Failed to run X-13 decomposition: {e}")
            return {}

    def _fourier_decomposition(self, data: pd.Series) -> Dict[str, pd.Series]:
        """傅里叶分解"""
        try:
            if fft is None:
                logger.error("scipy.fft not available")
                return {}

            # 傅里叶变换
            n = len(data)
            fft_values = fft(data.values)
            frequencies = fftfreq(n, d=1)

            # 找到主要频率成分
            power_spectrum = np.abs(fft_values) ** 2
            significant_freqs = frequencies[
                power_spectrum > np.percentile(power_spectrum, 95)
            ]

            # 重构季节性成分
            seasonal_component = np.zeros(n)
            for freq in significant_freqs[:10]:  # 取前10个主要频率
                if freq != 0:  # 排除直流分量
                    period = 1 / abs(freq)
                    if self.config.min_period <= period <= self.config.max_period:
                        amplitude = power_spectrum[frequencies == freq][0]
                        phase = np.angle(fft_values[frequencies == freq][0])
                        seasonal_component += amplitude * np.cos(
                            2 * np.pi * freq * np.arange(n) + phase
                        )

            # 计算趋势（移动平均）
            trend = data.rolling(window=self.config.trend_window, center=True).mean()

            # 计算残差
            seasonal_series = pd.Series(seasonal_component, index=data.index)
            residual = data - trend - seasonal_series

            return {
                "trend": trend,
                "seasonal": seasonal_series,
                "residual": residual,
                "observed": data,
            }

        except Exception as e:
            logger.error(f"Failed to run Fourier decomposition: {e}")
            return {}

    def _custom_decomposition(self, data: pd.Series) -> Dict[str, pd.Series]:
        """自定义分解方法"""
        try:
            # 检测多个季节性周期
            seasonal_components = {}

            for period in self.config.seasonal_periods:
                if period < len(data) // 2:
                    seasonal_comp = self._extract_seasonal_component(data, period)
                    if seasonal_comp is not None:
                        seasonal_components[period] = seasonal_comp

            # 合并季节性成分
            if seasonal_components:
                combined_seasonal = sum(seasonal_components.values())
            else:
                combined_seasonal = pd.Series(0, index=data.index)

            # 计算趋势
            trend = data.rolling(window=self.config.trend_window, center=True).mean()

            # 计算残差
            residual = data - trend - combined_seasonal

            return {
                "trend": trend,
                "seasonal": combined_seasonal,
                "residual": residual,
                "observed": data,
                "seasonal_components": seasonal_components,
            }

        except Exception as e:
            logger.error(f"Failed to run custom decomposition: {e}")
            return {}

    def _determine_primary_period(self, data: pd.Series) -> int:
        """确定主要季节性周期"""
        try:
            if periodogram is None:
                # 简单的自相关方法
                return self._autocorr_period_detection(data)

            # 使用周期图方法
            frequencies, power = periodogram(data.dropna().values)

            # 找到最强的非零频率
            non_zero_mask = frequencies > 0
            max_power_idx = np.argmax(power[non_zero_mask])
            dominant_freq = frequencies[non_zero_mask][max_power_idx]

            period = int(1 / dominant_freq)

            # 验证周期是否合理
            if self.config.min_period <= period <= self.config.max_period:
                return period
            else:
                return self._autocorr_period_detection(data)

        except Exception as e:
            logger.error(f"Failed to determine primary period: {e}")
            return 7  # 默认周期

    def _autocorr_period_detection(self, data: pd.Series) -> int:
        """基于自相关的周期检测"""
        try:
            # 计算自相关
            data_clean = data.dropna()
            if len(data_clean) < 20:
                return 7

            max_lag = min(len(data_clean) // 4, self.config.max_period)
            autocorr = [data_clean.autocorr(lag=i) for i in range(1, max_lag)]

            # 找到局部最大值
            if find_peaks is not None:
                peaks, _ = find_peaks(autocorr, height=0.1, distance=2)
                if len(peaks) > 0:
                    # 返回最强的周期
                    peak_values = [autocorr[p] for p in peaks]
                    best_peak_idx = np.argmax(peak_values)
                    return peaks[best_peak_idx] + 1

            # 简单方法：找最大自相关
            max_autocorr_idx = np.argmax(autocorr)
            return max_autocorr_idx + 1

        except Exception as e:
            logger.error(f"Failed to detect period using autocorrelation: {e}")
            return 7

    def _extract_seasonal_component(
        self, data: pd.Series, period: int
    ) -> Optional[pd.Series]:
        """提取特定周期的季节性成分"""
        try:
            if period >= len(data) // 2:
                return None

            # 创建季节性指示器
            seasonal_pattern = np.zeros(period)
            seasonal_counts = np.zeros(period)

            # 计算每个季节位置的平均值
            for i, value in enumerate(data.dropna()):
                season_idx = i % period
                seasonal_pattern[season_idx] += value
                seasonal_counts[season_idx] += 1

            # 避免除零
            seasonal_counts[seasonal_counts == 0] = 1
            seasonal_pattern /= seasonal_counts

            # 去除趋势
            seasonal_pattern -= seasonal_pattern.mean()

            # 扩展到完整时间序列
            full_seasonal = np.tile(seasonal_pattern, len(data) // period + 1)[
                : len(data)
            ]

            return pd.Series(full_seasonal, index=data.index)

        except Exception as e:
            logger.error(
                f"Failed to extract seasonal component for period {period}: {e}"
            )
            return None

    def _calculate_seasonal_strength(self, decomposition: Dict[str, pd.Series]):
        """计算季节性强度"""
        try:
            seasonal = decomposition.get("seasonal")
            residual = decomposition.get("residual")

            if seasonal is None or residual is None:
                return

            # 季节性强度 = 1 - Var(残差) / Var(去季节性数据)
            var_residual = residual.var()
            var_deseasonalized = (decomposition["observed"] - seasonal).var()

            seasonal_strength = (
                1 - var_residual / var_deseasonalized if var_deseasonalized > 0 else 0
            )
            self.seasonal_strengths["overall"] = max(0, min(1, seasonal_strength))

        except Exception as e:
            logger.error(f"Failed to calculate seasonal strength: {e}")

    def detect_calendar_effects(self, data: pd.Series) -> Dict[str, pd.Series]:
        """检测日历效应

        Args:
            data: 时间序列数据

        Returns:
            日历效应特征
        """
        try:
            logger.info("Detecting calendar effects...")

            calendar_features = {}

            # 月份效应
            monthly_effects = data.groupby(data.index.month).mean()
            monthly_dummies = pd.get_dummies(data.index.month)
            for month in range(1, 13):
                if month in monthly_dummies.columns:
                    month_name = calendar.month_name[month]
                    calendar_features[f"month_{month_name}"] = monthly_dummies[
                        month
                    ].reindex(data.index, fill_value=0)

            # 星期效应
            if hasattr(data.index, "dayofweek"):
                weekly_effects = data.groupby(data.index.dayofweek).mean()
                weekly_dummies = pd.get_dummies(data.index.dayofweek)
                weekdays = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                for i, weekday in enumerate(weekdays):
                    if i in weekly_dummies.columns:
                        calendar_features[f"weekday_{weekday}"] = weekly_dummies[
                            i
                        ].reindex(data.index, fill_value=0)

            # 季度效应
            quarterly_effects = data.groupby(data.index.quarter).mean()
            quarterly_dummies = pd.get_dummies(data.index.quarter)
            for quarter in range(1, 5):
                if quarter in quarterly_dummies.columns:
                    calendar_features[f"quarter_{quarter}"] = quarterly_dummies[
                        quarter
                    ].reindex(data.index, fill_value=0)

            # 年末效应
            if hasattr(data.index, "dayofyear"):
                year_end_effect = (data.index.dayofyear > 350).astype(int)
                calendar_features["year_end"] = pd.Series(
                    year_end_effect, index=data.index
                )

            # 月初/月末效应
            month_start_effect = (data.index.day <= 5).astype(int)
            month_end_effect = (data.index.day >= 25).astype(int)
            calendar_features["month_start"] = pd.Series(
                month_start_effect, index=data.index
            )
            calendar_features["month_end"] = pd.Series(
                month_end_effect, index=data.index
            )

            return calendar_features

        except Exception as e:
            logger.error(f"Failed to detect calendar effects: {e}")
            return {}

    def test_seasonality(self, data: pd.Series) -> Dict[str, Any]:
        """测试季节性显著性

        Args:
            data: 时间序列数据

        Returns:
            季节性测试结果
        """
        try:
            logger.info("Testing seasonality significance...")

            test_results = {}

            # Kruskal-Wallis测试（月份）
            if stats is not None and len(data) > 12:
                monthly_groups = [
                    data[data.index.month == month].dropna() for month in range(1, 13)
                ]
                monthly_groups = [group for group in monthly_groups if len(group) > 0]

                if len(monthly_groups) >= 3:
                    kw_stat, kw_pvalue = stats.kruskal(*monthly_groups)
                    test_results["monthly_kruskal_wallis"] = {
                        "statistic": kw_stat,
                        "p_value": kw_pvalue,
                        "significant": kw_pvalue < self.config.significance_level,
                    }

            # ANOVA测试（季度）
            if stats is not None and len(data) > 4:
                quarterly_groups = [
                    data[data.index.quarter == quarter].dropna()
                    for quarter in range(1, 5)
                ]
                quarterly_groups = [
                    group for group in quarterly_groups if len(group) > 0
                ]

                if len(quarterly_groups) >= 2:
                    f_stat, f_pvalue = stats.f_oneway(*quarterly_groups)
                    test_results["quarterly_anova"] = {
                        "statistic": f_stat,
                        "p_value": f_pvalue,
                        "significant": f_pvalue < self.config.significance_level,
                    }

            # 自相关测试
            for period in self.config.seasonal_periods:
                if period < len(data) // 2:
                    autocorr = data.autocorr(lag=period)
                    if pd.notna(autocorr):
                        # 简单的显著性测试（大样本近似）
                        n = len(data.dropna())
                        se = 1 / np.sqrt(n)
                        z_score = autocorr / se
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                        test_results[f"autocorr_period_{period}"] = {
                            "autocorr": autocorr,
                            "z_score": z_score,
                            "p_value": p_value,
                            "significant": p_value < self.config.significance_level,
                        }

            return test_results

        except Exception as e:
            logger.error(f"Failed to test seasonality: {e}")
            return {}

    def get_seasonal_features(self, data: pd.Series) -> pd.DataFrame:
        """获取季节性特征

        Args:
            data: 时间序列数据

        Returns:
            季节性特征DataFrame
        """
        try:
            # 提取季节性分解
            decomposition = self.extract_seasonality(data)

            if not decomposition:
                return pd.DataFrame(index=data.index)

            features_df = pd.DataFrame(index=data.index)

            # 基础分解成分
            if "trend" in decomposition:
                features_df["trend"] = decomposition["trend"]
                features_df["detrended"] = data - decomposition["trend"]

            if "seasonal" in decomposition:
                features_df["seasonal"] = decomposition["seasonal"]
                features_df["seasonal_strength"] = abs(decomposition["seasonal"])
                features_df["deseasonalized"] = data - decomposition["seasonal"]

            if "residual" in decomposition:
                features_df["residual"] = decomposition["residual"]

            # 日历效应
            calendar_effects = self.detect_calendar_effects(data)
            for name, effect in calendar_effects.items():
                features_df[name] = effect

            # 周期性特征
            for period in self.config.seasonal_periods:
                if period < len(data) // 2:
                    cycle_feature = np.sin(2 * np.pi * np.arange(len(data)) / period)
                    features_df[f"cycle_sin_{period}"] = cycle_feature

                    cycle_feature_cos = np.cos(
                        2 * np.pi * np.arange(len(data)) / period
                    )
                    features_df[f"cycle_cos_{period}"] = cycle_feature_cos

            return features_df

        except Exception as e:
            logger.error(f"Failed to get seasonal features: {e}")
            return pd.DataFrame(index=data.index)


# 便捷函数
def extract_seasonality(data: pd.Series, method: str = "stl") -> Dict[str, pd.Series]:
    """提取季节性的便捷函数

    Args:
        data: 时间序列数据
        method: 分解方法

    Returns:
        季节性分解结果
    """
    config = SeasonalityConfig(method=method)
    extractor = SeasonalityExtractor(config)
    return extractor.extract_seasonality(data, method)
