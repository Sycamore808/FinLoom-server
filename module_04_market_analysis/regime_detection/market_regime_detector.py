"""
市场状态检测器模块
识别和预测市场状态转换
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.constants import TRADING_DAYS_PER_YEAR
from common.exceptions import ModelError
from common.logging_system import setup_logger
try:
    from hmmlearn import hmm
except ImportError:
    hmm = None
try:
    from scipy import stats
except ImportError:
    stats = None

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
except ImportError:
    GaussianMixture = None
    StandardScaler = None

logger = setup_logger("market_regime_detector")


class MarketRegime(Enum):
    """市场状态枚举"""

    BULL_QUIET = "bull_quiet"  # 牛市低波动
    BULL_VOLATILE = "bull_volatile"  # 牛市高波动
    BEAR_QUIET = "bear_quiet"  # 熊市低波动
    BEAR_VOLATILE = "bear_volatile"  # 熊市高波动
    SIDEWAYS = "sideways"  # 横盘整理
    CRISIS = "crisis"  # 危机状态
    RECOVERY = "recovery"  # 恢复期


@dataclass
class RegimeDetectionConfig:
    """状态检测配置"""

    n_regimes: int = 4
    lookback_window: int = 60
    vol_window: int = 20
    trend_window: int = 50
    min_regime_duration: int = 5
    transition_threshold: float = 0.7
    use_hmm: bool = True
    use_clustering: bool = True
    features: List[str] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.features is None:
            self.features = [
                "returns",
                "volatility",
                "volume",
                "trend",
                "momentum",
                "correlation",
                "skewness",
                "kurtosis",
            ]


@dataclass
class RegimeState:
    """市场状态"""

    regime: MarketRegime
    probability: float
    start_date: datetime
    duration_days: int
    characteristics: Dict[str, float]
    transition_probs: Dict[MarketRegime, float]
    confidence: float


@dataclass
class RegimeTransition:
    """状态转换"""

    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_date: datetime
    probability: float
    trigger_factors: Dict[str, float]


class MarketRegimeDetector:
    """市场状态检测器"""

    # 状态特征阈值
    REGIME_THRESHOLDS = {
        "bull_threshold": 0.1,  # 年化收益率10%以上为牛市
        "bear_threshold": -0.1,  # 年化收益率-10%以下为熊市
        "high_vol_threshold": 0.25,  # 年化波动率25%以上为高波动
        "crisis_vol_threshold": 0.4,  # 年化波动率40%以上为危机
    }

    def __init__(self, config: Optional[RegimeDetectionConfig] = None):
        """初始化市场状态检测器

        Args:
            config: 状态检测配置
        """
        self.config = config or RegimeDetectionConfig()
        self.hmm_model: Optional[hmm.GaussianHMM] = None
        self.gmm_model: Optional[GaussianMixture] = None
        self.scaler = StandardScaler() if StandardScaler else None
        self.current_regime: Optional[RegimeState] = None
        self.regime_history: List[RegimeState] = []
        self.transition_matrix: Optional[np.ndarray] = None

        # 初始化模型
        if self.config.use_hmm and hmm is not None:
            self._initialize_hmm()
        if self.config.use_clustering and GaussianMixture is not None:
            self._initialize_gmm()

    def detect_market_regime(
        self, market_data: pd.DataFrame, symbols: Optional[List[str]] = None
    ) -> RegimeState:
        """检测当前市场状态

        Args:
            market_data: 市场数据DataFrame
            symbols: 股票代码列表（可选）

        Returns:
            当前市场状态
        """
        logger.info("Detecting market regime...")

        # 计算市场特征
        features = self._calculate_market_features(market_data, symbols)

        # 使用多种方法检测状态
        regimes = []
        probabilities = []

        # 基于规则的检测
        rule_regime, rule_prob = self._rule_based_detection(features)
        regimes.append(rule_regime)
        probabilities.append(rule_prob)

        # HMM检测
        if self.config.use_hmm and self.hmm_model is not None and hmm is not None:
            hmm_regime, hmm_prob = self._hmm_detection(features)
            regimes.append(hmm_regime)
            probabilities.append(hmm_prob)

        # 聚类检测
        if self.config.use_clustering and self.gmm_model is not None and GaussianMixture is not None:
            cluster_regime, cluster_prob = self._clustering_detection(features)
            regimes.append(cluster_regime)
            probabilities.append(cluster_prob)

        # 综合判断
        final_regime = self._ensemble_detection(regimes, probabilities)

        # 计算状态特征
        characteristics = self._calculate_regime_characteristics(features)

        # 计算转换概率
        transition_probs = self._calculate_transition_probabilities(
            final_regime, features
        )

        # 创建状态对象
        regime_state = RegimeState(
            regime=final_regime,
            probability=np.mean(probabilities),
            start_date=market_data.index[-1]
            if isinstance(market_data.index, pd.DatetimeIndex)
            else datetime.now(),
            duration_days=self._calculate_regime_duration(final_regime),
            characteristics=characteristics,
            transition_probs=transition_probs,
            confidence=self._calculate_confidence(probabilities),
        )

        # 更新当前状态
        self.current_regime = regime_state
        self.regime_history.append(regime_state)

        logger.info(
            f"Detected regime: {final_regime.value} with probability {regime_state.probability:.2f}"
        )
        return regime_state

    def identify_trend_patterns(
        self, price_data: pd.Series, window: Optional[int] = None
    ) -> Dict[str, Any]:
        """识别趋势模式

        Args:
            price_data: 价格序列
            window: 窗口大小

        Returns:
            趋势模式字典
        """
        window = window or self.config.trend_window

        patterns = {}

        # 计算移动平均
        ma_short = price_data.rolling(window=window // 2).mean()
        ma_long = price_data.rolling(window=window).mean()

        # 识别趋势
        current_price = price_data.iloc[-1]
        patterns["trend"] = (
            "uptrend" if ma_short.iloc[-1] > ma_long.iloc[-1] else "downtrend"
        )

        # 计算趋势强度
        trend_strength = abs(ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
        patterns["trend_strength"] = trend_strength

        # 识别支撑和阻力
        patterns["support"] = price_data.rolling(window=window).min().iloc[-1]
        patterns["resistance"] = price_data.rolling(window=window).max().iloc[-1]

        # 识别图表形态
        patterns["chart_patterns"] = self._identify_chart_patterns(price_data, window)

        # 计算趋势持续时间
        trend_changes = (ma_short > ma_long).astype(int).diff()
        last_change_idx = (
            trend_changes[trend_changes != 0].index[-1]
            if any(trend_changes != 0)
            else price_data.index[0]
        )
        patterns["trend_duration"] = len(price_data.loc[last_change_idx:])

        return patterns

    def calculate_volatility_clusters(
        self, returns: pd.Series, method: str = "garch"
    ) -> pd.DataFrame:
        """计算波动率聚类

        Args:
            returns: 收益率序列
            method: 方法 ('garch', 'ewma', 'realized')

        Returns:
            波动率聚类DataFrame
        """
        vol_clusters = pd.DataFrame(index=returns.index)

        if method == "garch":
            # 简化的GARCH(1,1)估计
            vol_clusters["volatility"] = self._estimate_garch_volatility(returns)

        elif method == "ewma":
            # EWMA波动率
            vol_clusters["volatility"] = returns.ewm(span=self.config.vol_window).std()

        elif method == "realized":
            # 已实现波动率
            vol_clusters["volatility"] = returns.rolling(
                window=self.config.vol_window
            ).std()

        else:
            raise ValueError(f"Unknown volatility method: {method}")

        # 识别波动率状态
        vol_mean = vol_clusters["volatility"].mean()
        vol_std = vol_clusters["volatility"].std()

        vol_clusters["vol_regime"] = pd.cut(
            vol_clusters["volatility"],
            bins=[0, vol_mean - vol_std, vol_mean, vol_mean + vol_std, np.inf],
            labels=["very_low", "low", "normal", "high"],
        )

        # 计算波动率持续性
        vol_clusters["vol_persistence"] = vol_clusters["volatility"].autocorr(lag=1)

        return vol_clusters

    def predict_regime_transitions(
        self, current_features: Dict[str, float], horizon: int = 5
    ) -> List[RegimeTransition]:
        """预测状态转换

        Args:
            current_features: 当前市场特征
            horizon: 预测时间范围（天）

        Returns:
            预测的状态转换列表
        """
        if self.current_regime is None:
            raise ModelError("No current regime detected")

        transitions = []

        # 使用转换矩阵预测
        if self.transition_matrix is not None:
            current_regime_idx = self._regime_to_index(self.current_regime.regime)

            # 计算未来状态概率
            future_probs = self.transition_matrix[current_regime_idx]

            for future_regime_idx, prob in enumerate(future_probs):
                if prob > self.config.transition_threshold:
                    future_regime = self._index_to_regime(future_regime_idx)

                    if future_regime != self.current_regime.regime:
                        transition = RegimeTransition(
                            from_regime=self.current_regime.regime,
                            to_regime=future_regime,
                            transition_date=datetime.now() + timedelta(days=horizon),
                            probability=prob,
                            trigger_factors=self._identify_trigger_factors(
                                current_features, future_regime
                            ),
                        )
                        transitions.append(transition)

        # 基于特征趋势预测
        feature_based_transitions = self._predict_from_features(
            current_features, horizon
        )
        transitions.extend(feature_based_transitions)

        # 按概率排序
        transitions.sort(key=lambda x: x.probability, reverse=True)

        return transitions[:3]  # 返回最可能的3个转换

    def assess_market_stress(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """评估市场压力

        Args:
            market_data: 市场数据

        Returns:
            市场压力指标
        """
        stress_indicators = {}

        # 计算各种压力指标
        returns = market_data["close"].pct_change()

        # 1. 波动率压力
        current_vol = returns.rolling(window=20).std() * np.sqrt(252)
        historical_vol = returns.rolling(window=252).std() * np.sqrt(252)
        stress_indicators["volatility_stress"] = (
            current_vol.iloc[-1] / historical_vol.mean() - 1
        )

        # 2. 相关性压力（如果有多个资产）
        if len(market_data.columns) > 1:
            corr_matrix = market_data.pct_change().rolling(window=60).corr()
            avg_corr = corr_matrix.values[
                np.triu_indices_from(corr_matrix.values, k=1)
            ].mean()
            stress_indicators["correlation_stress"] = avg_corr

        # 3. 流动性压力（使用成交量）
        if "volume" in market_data.columns:
            volume_ma = market_data["volume"].rolling(window=20).mean()
            current_volume = market_data["volume"].iloc[-1]
            stress_indicators["liquidity_stress"] = 1 - (
                current_volume / volume_ma.iloc[-1]
            )

        # 4. 尾部风险
        stress_indicators["tail_risk"] = self._calculate_tail_risk(returns)

        # 5. 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        stress_indicators["max_drawdown"] = drawdown.min()

        # 6. 综合压力指数
        stress_weights = {
            "volatility_stress": 0.3,
            "correlation_stress": 0.2,
            "liquidity_stress": 0.2,
            "tail_risk": 0.2,
            "max_drawdown": 0.1,
        }

        composite_stress = sum(
            stress_indicators.get(k, 0) * v for k, v in stress_weights.items()
        )
        stress_indicators["composite_stress"] = composite_stress

        # 分类压力等级
        if composite_stress < -0.2:
            stress_indicators["stress_level"] = "low"
        elif composite_stress < 0.2:
            stress_indicators["stress_level"] = "normal"
        elif composite_stress < 0.5:
            stress_indicators["stress_level"] = "elevated"
        else:
            stress_indicators["stress_level"] = "high"

        return stress_indicators

    def _initialize_hmm(self) -> None:
        """初始化HMM模型"""
        if hmm is not None:
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.config.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42,
            )
        else:
            self.hmm_model = None

    def _initialize_gmm(self) -> None:
        """初始化GMM模型"""
        if GaussianMixture is not None:
            self.gmm_model = GaussianMixture(
                n_components=self.config.n_regimes, covariance_type="full", random_state=42
            )
        else:
            self.gmm_model = None

    def _calculate_market_features(
        self, market_data: pd.DataFrame, symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """计算市场特征

        Args:
            market_data: 市场数据
            symbols: 股票代码列表

        Returns:
            特征DataFrame
        """
        features = pd.DataFrame(index=market_data.index)

        # 基础特征
        if "close" in market_data.columns:
            returns = market_data["close"].pct_change()
            features["returns"] = returns
            features["volatility"] = returns.rolling(
                window=self.config.vol_window
            ).std()

        if "volume" in market_data.columns:
            features["volume"] = market_data["volume"]
            features["volume_ma"] = market_data["volume"].rolling(window=20).mean()

        # 技术特征
        if "close" in market_data.columns:
            # 趋势
            ma_short = market_data["close"].rolling(window=20).mean()
            ma_long = market_data["close"].rolling(window=50).mean()
            features["trend"] = (ma_short - ma_long) / ma_long

            # 动量
            features["momentum"] = market_data["close"].pct_change(periods=20)

            # RSI
            features["rsi"] = self._calculate_rsi(market_data["close"])

        # 统计特征
        if "returns" in features.columns:
            features["skewness"] = returns.rolling(window=60).skew()
            features["kurtosis"] = returns.rolling(window=60).kurt()

        # 填充缺失值
        features = features.fillna(method="ffill").fillna(0)

        return features

    def _rule_based_detection(
        self, features: pd.DataFrame
    ) -> Tuple[MarketRegime, float]:
        """基于规则的状态检测

        Args:
            features: 特征DataFrame

        Returns:
            检测到的状态和概率
        """
        latest = features.iloc[-1]

        # 计算年化指标
        annual_return = latest["returns"] * 252 if "returns" in latest else 0
        annual_vol = (
            latest["volatility"] * np.sqrt(252) if "volatility" in latest else 0
        )

        # 判断市场方向
        if annual_return > self.REGIME_THRESHOLDS["bull_threshold"]:
            direction = "bull"
        elif annual_return < self.REGIME_THRESHOLDS["bear_threshold"]:
            direction = "bear"
        else:
            direction = "sideways"

        # 判断波动水平
        if annual_vol > self.REGIME_THRESHOLDS["crisis_vol_threshold"]:
            vol_level = "crisis"
        elif annual_vol > self.REGIME_THRESHOLDS["high_vol_threshold"]:
            vol_level = "volatile"
        else:
            vol_level = "quiet"

        # 组合判断
        if vol_level == "crisis":
            regime = MarketRegime.CRISIS
        elif direction == "bull" and vol_level == "quiet":
            regime = MarketRegime.BULL_QUIET
        elif direction == "bull" and vol_level == "volatile":
            regime = MarketRegime.BULL_VOLATILE
        elif direction == "bear" and vol_level == "quiet":
            regime = MarketRegime.BEAR_QUIET
        elif direction == "bear" and vol_level == "volatile":
            regime = MarketRegime.BEAR_VOLATILE
        else:
            regime = MarketRegime.SIDEWAYS

        # 计算置信度
        confidence = min(abs(annual_return) * 2, 1.0)  # 简化的置信度计算

        return regime, confidence

    def _hmm_detection(self, features: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """HMM状态检测

        Args:
            features: 特征DataFrame

        Returns:
            检测到的状态和概率
        """
        # 准备特征
        feature_cols = [col for col in self.config.features if col in features.columns]
        X = features[feature_cols].values

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 训练或预测
        if not hasattr(self.hmm_model, "means_"):
            # 训练模型
            self.hmm_model.fit(X_scaled)

        # 预测状态
        states = self.hmm_model.predict(X_scaled)
        probs = self.hmm_model.predict_proba(X_scaled)

        # 获取最新状态
        current_state = states[-1]
        current_prob = probs[-1, current_state]

        # 映射到市场状态
        regime = self._map_state_to_regime(current_state, X_scaled[-1])

        return regime, current_prob

    def _clustering_detection(
        self, features: pd.DataFrame
    ) -> Tuple[MarketRegime, float]:
        """聚类状态检测

        Args:
            features: 特征DataFrame

        Returns:
            检测到的状态和概率
        """
        # 准备特征
        feature_cols = [col for col in self.config.features if col in features.columns]
        X = features[feature_cols].values

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 训练或预测
        if not hasattr(self.gmm_model, "means_"):
            # 训练模型
            self.gmm_model.fit(X_scaled)

        # 预测聚类
        clusters = self.gmm_model.predict(X_scaled)
        probs = self.gmm_model.predict_proba(X_scaled)

        # 获取最新聚类
        current_cluster = clusters[-1]
        current_prob = probs[-1, current_cluster]

        # 映射到市场状态
        regime = self._map_cluster_to_regime(current_cluster, X_scaled[-1])

        return regime, current_prob

    def _ensemble_detection(
        self, regimes: List[MarketRegime], probabilities: List[float]
    ) -> MarketRegime:
        """集成多种检测方法

        Args:
            regimes: 检测到的状态列表
            probabilities: 对应的概率列表

        Returns:
            最终状态
        """
        # 加权投票
        regime_scores = defaultdict(float)

        for regime, prob in zip(regimes, probabilities):
            regime_scores[regime] += prob

        # 选择得分最高的状态
        final_regime = max(regime_scores.items(), key=lambda x: x[1])[0]

        return final_regime

    def _calculate_regime_characteristics(
        self, features: pd.DataFrame
    ) -> Dict[str, float]:
        """计算状态特征

        Args:
            features: 特征DataFrame

        Returns:
            状态特征字典
        """
        latest = features.iloc[-1]

        characteristics = {}

        for col in features.columns:
            if col in latest:
                characteristics[col] = float(latest[col])

        # 添加额外统计
        if len(features) > 30:
            recent = features.tail(30)
            characteristics["mean_return"] = (
                recent["returns"].mean() if "returns" in recent else 0
            )
            characteristics["mean_volatility"] = (
                recent["volatility"].mean() if "volatility" in recent else 0
            )
            characteristics["trend_strength"] = (
                abs(recent["trend"].mean()) if "trend" in recent else 0
            )

        return characteristics

    def _calculate_transition_probabilities(
        self, current_regime: MarketRegime, features: pd.DataFrame
    ) -> Dict[MarketRegime, float]:
        """计算转换概率

        Args:
            current_regime: 当前状态
            features: 特征DataFrame

        Returns:
            到各状态的转换概率
        """
        transition_probs = {}

        # 使用历史转换矩阵
        if self.transition_matrix is not None:
            current_idx = self._regime_to_index(current_regime)

            for regime in MarketRegime:
                target_idx = self._regime_to_index(regime)
                transition_probs[regime] = self.transition_matrix[
                    current_idx, target_idx
                ]

        else:
            # 默认概率
            for regime in MarketRegime:
                if regime == current_regime:
                    transition_probs[regime] = 0.7  # 保持当前状态的概率
                else:
                    transition_probs[regime] = 0.3 / (len(MarketRegime) - 1)

        return transition_probs

    def _calculate_regime_duration(self, regime: MarketRegime) -> int:
        """计算状态持续时间

        Args:
            regime: 市场状态

        Returns:
            持续天数
        """
        if not self.regime_history:
            return 1

        # 从历史中查找最后一次不同状态
        duration = 1
        for i in range(len(self.regime_history) - 1, -1, -1):
            if self.regime_history[i].regime == regime:
                duration += 1
            else:
                break

        return duration

    def _calculate_confidence(self, probabilities: List[float]) -> float:
        """计算置信度

        Args:
            probabilities: 概率列表

        Returns:
            置信度分数
        """
        if not probabilities:
            return 0.0

        # 使用概率的一致性作为置信度
        mean_prob = np.mean(probabilities)
        std_prob = np.std(probabilities)

        # 一致性越高，置信度越高
        confidence = mean_prob * (1 - std_prob)

        return min(max(confidence, 0.0), 1.0)

    def _regime_to_index(self, regime: MarketRegime) -> int:
        """将状态转换为索引

        Args:
            regime: 市场状态

        Returns:
            索引
        """
        regime_list = list(MarketRegime)
        return regime_list.index(regime)

    def _index_to_regime(self, index: int) -> MarketRegime:
        """将索引转换为状态

        Args:
            index: 索引

        Returns:
            市场状态
        """
        regime_list = list(MarketRegime)
        return regime_list[index % len(regime_list)]

    def _map_state_to_regime(self, state: int, features: np.ndarray) -> MarketRegime:
        """将HMM状态映射到市场状态

        Args:
            state: HMM状态
            features: 特征向量

        Returns:
            市场状态
        """
        # 简化映射（实际应用中应该基于状态特征）
        regime_map = {
            0: MarketRegime.BEAR_VOLATILE,
            1: MarketRegime.BEAR_QUIET,
            2: MarketRegime.SIDEWAYS,
            3: MarketRegime.BULL_QUIET,
            4: MarketRegime.BULL_VOLATILE,
            5: MarketRegime.CRISIS,
            6: MarketRegime.RECOVERY,
        }

        return regime_map.get(state, MarketRegime.SIDEWAYS)

    def _map_cluster_to_regime(
        self, cluster: int, features: np.ndarray
    ) -> MarketRegime:
        """将聚类映射到市场状态

        Args:
            cluster: 聚类标签
            features: 特征向量

        Returns:
            市场状态
        """
        # 基于聚类中心特征判断
        # 简化实现
        return self._map_state_to_regime(cluster, features)

    def _identify_trigger_factors(
        self, current_features: Dict[str, float], target_regime: MarketRegime
    ) -> Dict[str, float]:
        """识别触发因素

        Args:
            current_features: 当前特征
            target_regime: 目标状态

        Returns:
            触发因素字典
        """
        trigger_factors = {}

        # 识别可能导致状态转换的因素
        if "volatility" in current_features:
            trigger_factors["volatility_change"] = current_features["volatility"]

        if "momentum" in current_features:
            trigger_factors["momentum_shift"] = current_features["momentum"]

        if "volume" in current_features:
            trigger_factors["volume_spike"] = current_features["volume"]

        return trigger_factors

    def _predict_from_features(
        self, current_features: Dict[str, float], horizon: int
    ) -> List[RegimeTransition]:
        """基于特征预测状态转换

        Args:
            current_features: 当前特征
            horizon: 预测时间范围

        Returns:
            预测的转换列表
        """
        transitions = []

        # 简化的特征趋势预测
        # 实际应用中应使用更复杂的预测模型

        # 如果波动率上升，可能转向高波动状态
        if current_features.get("volatility", 0) > 0.2:
            if self.current_regime.regime in [
                MarketRegime.BULL_QUIET,
                MarketRegime.BEAR_QUIET,
            ]:
                target_regime = (
                    MarketRegime.BULL_VOLATILE
                    if "bull" in self.current_regime.regime.value
                    else MarketRegime.BEAR_VOLATILE
                )

                transition = RegimeTransition(
                    from_regime=self.current_regime.regime,
                    to_regime=target_regime,
                    transition_date=datetime.now() + timedelta(days=horizon),
                    probability=0.6,
                    trigger_factors={
                        "volatility_increase": current_features["volatility"]
                    },
                )
                transitions.append(transition)

        return transitions

    def _identify_chart_patterns(self, price_data: pd.Series, window: int) -> List[str]:
        """识别图表形态

        Args:
            price_data: 价格序列
            window: 窗口大小

        Returns:
            识别到的形态列表
        """
        patterns = []

        # 简化的形态识别
        recent_prices = price_data.tail(window)

        # 头肩顶/底
        if self._is_head_and_shoulders(recent_prices):
            patterns.append("head_and_shoulders")

        # 双顶/双底
        if self._is_double_top_bottom(recent_prices):
            patterns.append("double_top_bottom")

        # 三角形
        if self._is_triangle_pattern(recent_prices):
            patterns.append("triangle")

        return patterns

    def _is_head_and_shoulders(self, prices: pd.Series) -> bool:
        """判断是否为头肩形态

        Args:
            prices: 价格序列

        Returns:
            是否为头肩形态
        """
        # 简化判断
        if len(prices) < 5:
            return False

        # 查找局部极值点
        peaks = []
        for i in range(1, len(prices) - 1):
            if (
                prices.iloc[i] > prices.iloc[i - 1]
                and prices.iloc[i] > prices.iloc[i + 1]
            ):
                peaks.append(i)

        # 判断是否有三个峰值且中间最高
        if len(peaks) >= 3:
            if (
                prices.iloc[peaks[1]] > prices.iloc[peaks[0]]
                and prices.iloc[peaks[1]] > prices.iloc[peaks[2]]
            ):
                return True

        return False

    def _is_double_top_bottom(self, prices: pd.Series) -> bool:
        """判断是否为双顶/双底形态

        Args:
            prices: 价格序列

        Returns:
            是否为双顶/双底形态
        """
        # 简化判断
        if len(prices) < 4:
            return False

        # 查找两个相近的极值点
        highs = prices.rolling(window=3).max()
        lows = prices.rolling(window=3).min()

        # 检查是否有两个相近的高点或低点
        high_peaks = prices[prices == highs]
        low_troughs = prices[prices == lows]

        if len(high_peaks) >= 2:
            peak_values = high_peaks.values[-2:]
            if abs(peak_values[0] - peak_values[1]) / peak_values[0] < 0.03:
                return True

        if len(low_troughs) >= 2:
            trough_values = low_troughs.values[-2:]
            if abs(trough_values[0] - trough_values[1]) / trough_values[0] < 0.03:
                return True

        return False

    def _is_triangle_pattern(self, prices: pd.Series) -> bool:
        """判断是否为三角形形态

        Args:
            prices: 价格序列

        Returns:
            是否为三角形形态
        """
        # 简化判断：检查价格范围是否收敛
        if len(prices) < 10:
            return False

        # 计算滚动范围
        rolling_range = prices.rolling(window=5).max() - prices.rolling(window=5).min()

        # 检查范围是否递减（收敛）
        if rolling_range.iloc[-1] < rolling_range.iloc[-5] * 0.7:
            return True

        return False

    def _calculate_tail_risk(self, returns: pd.Series) -> float:
        """计算尾部风险

        Args:
            returns: 收益率序列

        Returns:
            尾部风险值
        """
        # 计算VaR和CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        # 尾部风险指标
        tail_risk = abs(cvar_95) / returns.std() if returns.std() > 0 else 0

        return tail_risk

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标

        Args:
            prices: 价格序列
            period: 周期

        Returns:
            RSI序列
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _estimate_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """估计GARCH波动率

        Args:
            returns: 收益率序列

        Returns:
            GARCH波动率序列
        """
        # 简化的GARCH(1,1)实现
        # 实际应用中应使用arch包

        omega = 0.00001  # 长期方差
        alpha = 0.1  # ARCH系数
        beta = 0.85  # GARCH系数

        volatility = pd.Series(index=returns.index)
        volatility.iloc[0] = returns.std()

        for i in range(1, len(returns)):
            volatility.iloc[i] = np.sqrt(
                omega
                + alpha * returns.iloc[i - 1] ** 2
                + beta * volatility.iloc[i - 1] ** 2
            )

        return volatility


# 模块级别函数
def detect_current_market_regime(
    market_data: pd.DataFrame, config: Optional[RegimeDetectionConfig] = None
) -> Dict[str, Any]:
    """检测当前市场状态的便捷函数

    Args:
        market_data: 市场数据
        config: 检测配置

    Returns:
        市场状态信息字典
    """
    detector = MarketRegimeDetector(config)
    regime_state = detector.detect_market_regime(market_data)

    return {
        "regime": regime_state.regime.value,
        "probability": regime_state.probability,
        "characteristics": regime_state.characteristics,
        "confidence": regime_state.confidence,
        "stress_level": detector.assess_market_stress(market_data),
    }
