#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
制度特征模块
检测和分析市场制度/状态变化
"""

import warnings

warnings.filterwarnings("ignore")

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
except ImportError:
    GaussianMixture = None
    KMeans = None
    StandardScaler = None
    PCA = None

try:
    from scipy import stats
    from scipy.signal import find_peaks
except ImportError:
    stats = None
    find_peaks = None

logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    """制度检测配置"""

    n_regimes: int = 3  # 制度数量
    lookback_window: int = 252  # 回看窗口
    min_regime_length: int = 20  # 最小制度长度
    regime_method: str = "hmm"  # 检测方法
    volatility_window: int = 60  # 波动率窗口
    return_window: int = 20  # 收益率窗口
    transition_threshold: float = 0.7  # 转换阈值
    stability_window: int = 30  # 稳定性窗口


class RegimeFeatures:
    """制度特征提取器"""

    def __init__(self, config: Optional[RegimeConfig] = None):
        """初始化制度特征提取器

        Args:
            config: 制度检测配置
        """
        self.config = config or RegimeConfig()

        self.regime_states = None
        self.regime_probabilities = None
        self.regime_characteristics = {}

        if GaussianMixture is None:
            logger.warning(
                "scikit-learn not available. Some regime features will be limited."
            )

    def detect_market_regimes(
        self, data: pd.DataFrame, features: Optional[List[str]] = None
    ) -> Optional[pd.Series]:
        """检测市场制度

        Args:
            data: 市场数据
            features: 用于制度检测的特征列表

        Returns:
            制度状态序列
        """
        try:
            logger.info("Detecting market regimes...")

            # 准备特征数据
            if features is None:
                features = ["returns", "volatility", "volume"]

            feature_data = self._prepare_regime_features(data, features)

            if feature_data.empty:
                logger.warning("No valid feature data for regime detection")
                return None

            # 根据方法检测制度
            if self.config.regime_method == "hmm":
                regime_states = self._hmm_regime_detection(feature_data)
            elif self.config.regime_method == "gmm":
                regime_states = self._gmm_regime_detection(feature_data)
            elif self.config.regime_method == "kmeans":
                regime_states = self._kmeans_regime_detection(feature_data)
            elif self.config.regime_method == "threshold":
                regime_states = self._threshold_regime_detection(feature_data)
            else:
                raise ValueError(f"Unknown regime method: {self.config.regime_method}")

            # 平滑制度序列
            regime_states = self._smooth_regime_states(regime_states)

            self.regime_states = regime_states
            self._analyze_regime_characteristics(feature_data, regime_states)

            logger.info(f"Detected {len(self.regime_characteristics)} market regimes")
            return regime_states

        except Exception as e:
            logger.error(f"Failed to detect market regimes: {e}")
            return None

    def _prepare_regime_features(
        self, data: pd.DataFrame, features: List[str]
    ) -> pd.DataFrame:
        """准备制度检测特征"""
        try:
            feature_data = pd.DataFrame(index=data.index)

            # 收益率特征
            if "returns" in features and "close" in data.columns:
                returns = data["close"].pct_change()
                feature_data["returns"] = returns
                feature_data["returns_ma"] = returns.rolling(
                    self.config.return_window
                ).mean()
                feature_data["returns_std"] = returns.rolling(
                    self.config.return_window
                ).std()

            # 波动率特征
            if "volatility" in features and "close" in data.columns:
                returns = data["close"].pct_change()
                volatility = returns.rolling(
                    self.config.volatility_window
                ).std() * np.sqrt(252)
                feature_data["volatility"] = volatility
                feature_data["volatility_ma"] = volatility.rolling(
                    self.config.volatility_window
                ).mean()

            # 成交量特征
            if "volume" in features and "volume" in data.columns:
                volume = data["volume"]
                feature_data["volume"] = volume
                feature_data["volume_ma"] = volume.rolling(
                    self.config.volatility_window
                ).mean()
                feature_data["volume_ratio"] = volume / feature_data["volume_ma"]

            # 价格趋势特征
            if "trend" in features and "close" in data.columns:
                close = data["close"]
                feature_data["price_ma20"] = close.rolling(20).mean()
                feature_data["price_ma60"] = close.rolling(60).mean()
                feature_data["trend_ratio"] = (
                    feature_data["price_ma20"] / feature_data["price_ma60"]
                )

            # VIX类特征
            if "vix" in features and "high" in data.columns and "low" in data.columns:
                high_low_ratio = (data["high"] - data["low"]) / data["close"]
                feature_data["vix_proxy"] = high_low_ratio.rolling(
                    self.config.volatility_window
                ).mean()

            # 动量特征
            if "momentum" in features and "close" in data.columns:
                close = data["close"]
                feature_data["momentum_5d"] = close.pct_change(5)
                feature_data["momentum_20d"] = close.pct_change(20)
                feature_data["momentum_60d"] = close.pct_change(60)

            return feature_data.dropna()

        except Exception as e:
            logger.error(f"Failed to prepare regime features: {e}")
            return pd.DataFrame()

    def _hmm_regime_detection(self, feature_data: pd.DataFrame) -> pd.Series:
        """使用HMM检测制度（简化实现）"""
        try:
            if GaussianMixture is None:
                logger.warning("Using GMM as HMM alternative")
                return self._gmm_regime_detection(feature_data)

            # 标准化特征
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_data.fillna(0))

            # 使用GMM作为HMM的简化替代
            gmm = GaussianMixture(
                n_components=self.config.n_regimes,
                random_state=42,
                covariance_type="full",
            )

            regime_states = gmm.fit_predict(features_scaled)
            self.regime_probabilities = gmm.predict_proba(features_scaled)

            return pd.Series(regime_states, index=feature_data.index)

        except Exception as e:
            logger.error(f"Failed to run HMM regime detection: {e}")
            return pd.Series(0, index=feature_data.index)

    def _gmm_regime_detection(self, feature_data: pd.DataFrame) -> pd.Series:
        """使用GMM检测制度"""
        try:
            if GaussianMixture is None:
                logger.error("GaussianMixture not available")
                return pd.Series(0, index=feature_data.index)

            # 标准化特征
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_data.fillna(0))

            # GMM拟合
            gmm = GaussianMixture(
                n_components=self.config.n_regimes,
                random_state=42,
                covariance_type="full",
            )

            regime_states = gmm.fit_predict(features_scaled)
            self.regime_probabilities = gmm.predict_proba(features_scaled)

            return pd.Series(regime_states, index=feature_data.index)

        except Exception as e:
            logger.error(f"Failed to run GMM regime detection: {e}")
            return pd.Series(0, index=feature_data.index)

    def _kmeans_regime_detection(self, feature_data: pd.DataFrame) -> pd.Series:
        """使用KMeans检测制度"""
        try:
            if KMeans is None:
                logger.error("KMeans not available")
                return pd.Series(0, index=feature_data.index)

            # 标准化特征
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_data.fillna(0))

            # KMeans聚类
            kmeans = KMeans(
                n_clusters=self.config.n_regimes, random_state=42, n_init=10
            )

            regime_states = kmeans.fit_predict(features_scaled)

            return pd.Series(regime_states, index=feature_data.index)

        except Exception as e:
            logger.error(f"Failed to run KMeans regime detection: {e}")
            return pd.Series(0, index=feature_data.index)

    def _threshold_regime_detection(self, feature_data: pd.DataFrame) -> pd.Series:
        """使用阈值方法检测制度"""
        try:
            regime_states = pd.Series(1, index=feature_data.index)  # 默认正常制度

            # 基于波动率的制度分类
            if "volatility" in feature_data.columns:
                volatility = feature_data["volatility"]
                vol_threshold_high = volatility.quantile(0.8)
                vol_threshold_low = volatility.quantile(0.2)

                # 高波动制度
                regime_states[volatility > vol_threshold_high] = 2
                # 低波动制度
                regime_states[volatility < vol_threshold_low] = 0

            # 基于收益率的制度调整
            if "returns" in feature_data.columns:
                returns = feature_data["returns"]
                return_threshold = returns.std() * 2

                # 极端负收益制度
                regime_states[returns < -return_threshold] = 3 % self.config.n_regimes

            return regime_states

        except Exception as e:
            logger.error(f"Failed to run threshold regime detection: {e}")
            return pd.Series(0, index=feature_data.index)

    def _smooth_regime_states(self, regime_states: pd.Series) -> pd.Series:
        """平滑制度状态序列"""
        try:
            smoothed = regime_states.copy()

            # 滑动窗口平滑
            for i in range(len(regime_states)):
                start_idx = max(0, i - self.config.stability_window // 2)
                end_idx = min(
                    len(regime_states), i + self.config.stability_window // 2 + 1
                )

                window_states = regime_states.iloc[start_idx:end_idx]
                most_common_state = (
                    window_states.mode().iloc[0]
                    if not window_states.mode().empty
                    else regime_states.iloc[i]
                )

                # 如果当前状态与窗口内最常见状态不同，且持续时间较短，则修正
                current_state = regime_states.iloc[i]
                if current_state != most_common_state:
                    # 检查当前状态的持续长度
                    state_length = 1
                    for j in range(
                        i + 1,
                        min(len(regime_states), i + self.config.min_regime_length),
                    ):
                        if regime_states.iloc[j] == current_state:
                            state_length += 1
                        else:
                            break

                    # 如果持续时间过短，则修正为最常见状态
                    if state_length < self.config.min_regime_length:
                        smoothed.iloc[i] = most_common_state

            return smoothed

        except Exception as e:
            logger.error(f"Failed to smooth regime states: {e}")
            return regime_states

    def _analyze_regime_characteristics(
        self, feature_data: pd.DataFrame, regime_states: pd.Series
    ):
        """分析制度特征"""
        try:
            self.regime_characteristics = {}

            for regime in regime_states.unique():
                regime_mask = regime_states == regime
                regime_data = feature_data[regime_mask]

                if regime_data.empty:
                    continue

                characteristics = {}

                # 基础统计
                for col in regime_data.columns:
                    if pd.api.types.is_numeric_dtype(regime_data[col]):
                        characteristics[f"{col}_mean"] = regime_data[col].mean()
                        characteristics[f"{col}_std"] = regime_data[col].std()
                        characteristics[f"{col}_median"] = regime_data[col].median()

                # 制度持续时间
                regime_durations = self._calculate_regime_durations(
                    regime_states, regime
                )
                characteristics["avg_duration"] = (
                    np.mean(regime_durations) if regime_durations else 0
                )
                characteristics["max_duration"] = (
                    max(regime_durations) if regime_durations else 0
                )
                characteristics["min_duration"] = (
                    min(regime_durations) if regime_durations else 0
                )

                # 制度频率
                characteristics["frequency"] = (regime_states == regime).sum() / len(
                    regime_states
                )

                self.regime_characteristics[regime] = characteristics

        except Exception as e:
            logger.error(f"Failed to analyze regime characteristics: {e}")

    def _calculate_regime_durations(
        self, regime_states: pd.Series, regime: int
    ) -> List[int]:
        """计算制度持续时间"""
        durations = []
        current_duration = 0

        for state in regime_states:
            if state == regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0

        # 处理最后一段
        if current_duration > 0:
            durations.append(current_duration)

        return durations

    def get_regime_transition_probabilities(self) -> Optional[pd.DataFrame]:
        """获取制度转换概率矩阵

        Returns:
            转换概率矩阵
        """
        try:
            if self.regime_states is None:
                return None

            n_regimes = len(self.regime_states.unique())
            transition_matrix = np.zeros((n_regimes, n_regimes))

            # 计算转换次数
            for i in range(len(self.regime_states) - 1):
                current_regime = self.regime_states.iloc[i]
                next_regime = self.regime_states.iloc[i + 1]
                transition_matrix[current_regime, next_regime] += 1

            # 转换为概率
            for i in range(n_regimes):
                row_sum = transition_matrix[i].sum()
                if row_sum > 0:
                    transition_matrix[i] /= row_sum

            return pd.DataFrame(
                transition_matrix,
                index=[f"Regime_{i}" for i in range(n_regimes)],
                columns=[f"Regime_{i}" for i in range(n_regimes)],
            )

        except Exception as e:
            logger.error(f"Failed to calculate transition probabilities: {e}")
            return None

    def get_current_regime_probability(
        self, lookback_days: int = 30
    ) -> Optional[Dict[int, float]]:
        """获取当前制度概率

        Args:
            lookback_days: 回看天数

        Returns:
            各制度的概率
        """
        try:
            if self.regime_states is None:
                return None

            # 最近的制度状态
            recent_states = self.regime_states.tail(lookback_days)

            # 计算各制度的频率
            regime_probs = {}
            total_days = len(recent_states)

            for regime in recent_states.unique():
                count = (recent_states == regime).sum()
                regime_probs[regime] = count / total_days

            return regime_probs

        except Exception as e:
            logger.error(f"Failed to calculate current regime probability: {e}")
            return None

    def detect_regime_changes(self, sensitivity: float = 0.8) -> Optional[pd.Series]:
        """检测制度变化点

        Args:
            sensitivity: 敏感度

        Returns:
            制度变化标识序列
        """
        try:
            if self.regime_states is None:
                return None

            # 检测制度变化
            regime_changes = (self.regime_states != self.regime_states.shift(1)).astype(
                int
            )

            # 过滤短期噪音
            smoothed_changes = regime_changes.rolling(window=5, center=True).mean()
            significant_changes = (smoothed_changes > sensitivity).astype(int)

            return significant_changes

        except Exception as e:
            logger.error(f"Failed to detect regime changes: {e}")
            return None

    def get_regime_features(self) -> Optional[pd.DataFrame]:
        """获取制度特征

        Returns:
            制度特征DataFrame
        """
        try:
            if self.regime_states is None:
                return None

            features_df = pd.DataFrame(index=self.regime_states.index)

            # 当前制度
            features_df["current_regime"] = self.regime_states

            # 制度持续时间
            regime_duration = pd.Series(0, index=self.regime_states.index)
            current_regime = None
            duration = 0

            for i, regime in enumerate(self.regime_states):
                if regime == current_regime:
                    duration += 1
                else:
                    current_regime = regime
                    duration = 1
                regime_duration.iloc[i] = duration

            features_df["regime_duration"] = regime_duration

            # 制度稳定性（最近N天制度一致性）
            stability_window = 10
            regime_stability = self.regime_states.rolling(
                window=stability_window
            ).apply(lambda x: (x == x.iloc[-1]).mean())
            features_df["regime_stability"] = regime_stability

            # 制度变化标识
            features_df["regime_change"] = (
                self.regime_states != self.regime_states.shift(1)
            ).astype(int)

            # 制度概率（如果有的话）
            if self.regime_probabilities is not None:
                for i in range(self.regime_probabilities.shape[1]):
                    features_df[f"regime_{i}_prob"] = self.regime_probabilities[:, i]

            return features_df

        except Exception as e:
            logger.error(f"Failed to get regime features: {e}")
            return None


# 便捷函数
def detect_market_regimes(
    data: pd.DataFrame, n_regimes: int = 3, method: str = "gmm"
) -> Optional[pd.Series]:
    """检测市场制度的便捷函数

    Args:
        data: 市场数据
        n_regimes: 制度数量
        method: 检测方法

    Returns:
        制度状态序列
    """
    config = RegimeConfig(n_regimes=n_regimes, regime_method=method)
    extractor = RegimeFeatures(config)
    return extractor.detect_market_regimes(data)
