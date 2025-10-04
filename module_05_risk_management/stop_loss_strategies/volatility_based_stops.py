"""
波动率止损策略模块
基于市场波动率的动态止损
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.constants import TRADING_DAYS_PER_YEAR
from common.exceptions import ModelError
from common.logging_system import setup_logger
from scipy import stats

logger = setup_logger("volatility_based_stops")


class VolatilityMeasure(Enum):
    """波动率度量方式"""

    STANDARD_DEVIATION = "std"
    ATR = "atr"
    GARCH = "garch"
    REALIZED_VOLATILITY = "realized"
    IMPLIED_VOLATILITY = "implied"
    RANGE_BASED = "range"


@dataclass
class VolatilityStopConfig:
    """波动率止损配置"""

    volatility_measure: VolatilityMeasure = VolatilityMeasure.STANDARD_DEVIATION
    lookback_period: int = 20
    volatility_multiplier: float = 2.0
    use_ewma: bool = True
    ewma_span: int = 20
    min_stop_distance: float = 0.01  # 1%最小止损距离
    max_stop_distance: float = 0.10  # 10%最大止损距离
    adapt_to_regime: bool = True
    garch_p: int = 1
    garch_q: int = 1
    confidence_level: float = 0.95


@dataclass
class VolatilityState:
    """波动率状态"""

    current_volatility: float
    historical_volatility: pd.Series
    volatility_regime: str  # low, normal, high, extreme
    volatility_percentile: float
    forecast_volatility: Optional[float]
    volatility_term_structure: Optional[Dict[int, float]]


@dataclass
class VolatilityStopResult:
    """波动率止损结果"""

    symbol: str
    stop_price: float
    stop_distance: float
    stop_distance_pct: float
    volatility_used: float
    volatility_type: str
    confidence_interval: Tuple[float, float]
    expected_hit_probability: float


class VolatilityBasedStops:
    """波动率止损策略类"""

    def __init__(self, config: Optional[VolatilityStopConfig] = None):
        """初始化波动率止损

        Args:
            config: 波动率止损配置
        """
        self.config = config or VolatilityStopConfig()
        self.volatility_cache: Dict[str, VolatilityState] = {}
        self.garch_models: Dict[str, Any] = {}

    def calculate_volatility_stop(
        self,
        symbol: str,
        current_price: float,
        price_data: pd.DataFrame,
        position_type: str = "long",
    ) -> VolatilityStopResult:
        """计算波动率止损

        Args:
            symbol: 标的代码
            current_price: 当前价格
            price_data: 价格数据
            position_type: 仓位类型

        Returns:
            波动率止损结果
        """
        logger.info(f"Calculating volatility-based stop for {symbol}")

        # 计算波动率
        volatility = self._calculate_volatility(price_data)

        # 获取波动率状态
        vol_state = self._get_volatility_state(symbol, price_data)

        # 根据波动率体制调整
        if self.config.adapt_to_regime:
            volatility = self._adjust_for_regime(
                volatility, vol_state.volatility_regime
            )

        # 计算止损距离
        stop_distance = self._calculate_stop_distance(
            current_price, volatility, vol_state
        )

        # 应用限制
        stop_distance = self._apply_limits(stop_distance, current_price)

        # 计算止损价格
        if position_type == "long":
            stop_price = current_price - stop_distance
        else:
            stop_price = current_price + stop_distance

        # 计算置信区间
        confidence_interval = self._calculate_confidence_interval(
            stop_price, volatility, self.config.confidence_level
        )

        # 估计触发概率
        hit_probability = self._estimate_hit_probability(
            current_price, stop_price, volatility
        )

        result = VolatilityStopResult(
            symbol=symbol,
            stop_price=stop_price,
            stop_distance=stop_distance,
            stop_distance_pct=stop_distance / current_price,
            volatility_used=volatility,
            volatility_type=self.config.volatility_measure.value,
            confidence_interval=confidence_interval,
            expected_hit_probability=hit_probability,
        )

        logger.info(
            f"Volatility stop: {stop_price:.2f} (distance: {stop_distance_pct:.2%})"
        )

        return result

    def calculate_dynamic_volatility(
        self, returns: pd.Series, method: str = "ewma"
    ) -> pd.Series:
        """计算动态波动率

        Args:
            returns: 收益率序列
            method: 计算方法

        Returns:
            动态波动率序列
        """
        if method == "ewma":
            # 指数加权移动平均
            return returns.ewm(span=self.config.ewma_span, adjust=False).std()

        elif method == "rolling":
            # 滚动窗口
            return returns.rolling(window=self.config.lookback_period).std()

        elif method == "expanding":
            # 扩展窗口
            return returns.expanding(min_periods=2).std()

        elif method == "garch":
            # GARCH模型
            return self._fit_garch_volatility(returns)

        else:
            raise ValueError(f"Unknown method: {method}")

    def calculate_realized_volatility(
        self,
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        open_prices: Optional[pd.Series] = None,
    ) -> float:
        """计算已实现波动率

        Args:
            high_prices: 最高价序列
            low_prices: 最低价序列
            close_prices: 收盘价序列
            open_prices: 开盘价序列（可选）

        Returns:
            已实现波动率
        """
        # Parkinson估计器
        parkinson_vol = np.sqrt(
            np.mean(np.log(high_prices / low_prices) ** 2) / (4 * np.log(2))
        )

        if open_prices is not None:
            # Garman-Klass估计器
            term1 = 0.5 * np.log(high_prices / low_prices) ** 2
            term2 = (2 * np.log(2) - 1) * np.log(close_prices / open_prices) ** 2
            gk_vol = np.sqrt(np.mean(term1 - term2))

            # Rogers-Satchell估计器
            rs_vol = np.sqrt(
                np.mean(
                    np.log(high_prices / close_prices)
                    * np.log(high_prices / open_prices)
                    + np.log(low_prices / close_prices)
                    * np.log(low_prices / open_prices)
                )
            )

            # 组合估计
            realized_vol = (parkinson_vol + gk_vol + rs_vol) / 3
        else:
            realized_vol = parkinson_vol

        # 年化
        return realized_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    def fit_garch_model(
        self, returns: pd.Series, p: int = 1, q: int = 1
    ) -> Dict[str, Any]:
        """拟合GARCH模型

        Args:
            returns: 收益率序列
            p: ARCH阶数
            q: GARCH阶数

        Returns:
            GARCH模型参数
        """
        from arch import arch_model

        # 缩放收益率（避免数值问题）
        scaled_returns = returns * 100

        # 拟合GARCH模型
        model = arch_model(scaled_returns, vol="Garch", p=p, q=q)
        res = model.fit(disp="off")

        # 提取参数
        params = {
            "omega": res.params["omega"],
            "alpha": res.params[f"alpha[{p}]"]
            if f"alpha[{p}]" in res.params
            else res.params["alpha[1]"],
            "beta": res.params[f"beta[{q}]"]
            if f"beta[{q}]" in res.params
            else res.params["beta[1]"],
            "conditional_volatility": res.conditional_volatility / 100,
            "aic": res.aic,
            "bic": res.bic,
        }

        return params

    def calculate_volatility_cone(
        self, returns: pd.Series, periods: List[int] = None
    ) -> pd.DataFrame:
        """计算波动率锥

        Args:
            returns: 收益率序列
            periods: 时间周期列表

        Returns:
            波动率锥DataFrame
        """
        if periods is None:
            periods = [5, 10, 20, 60, 120, 252]

        percentiles = [10, 25, 50, 75, 90]

        cone = pd.DataFrame(index=periods, columns=percentiles)

        for period in periods:
            rolling_vol = returns.rolling(window=period).std() * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )

            for percentile in percentiles:
                cone.loc[period, percentile] = np.percentile(
                    rolling_vol.dropna(), percentile
                )

        return cone

    def detect_volatility_regime(
        self, volatility: pd.Series, n_regimes: int = 3
    ) -> pd.Series:
        """检测波动率体制

        Args:
            volatility: 波动率序列
            n_regimes: 体制数量

        Returns:
            体制标签序列
        """
        from sklearn.cluster import KMeans

        # 准备数据
        vol_data = volatility.values.reshape(-1, 1)

        # K-means聚类
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regimes = kmeans.fit_predict(vol_data)

        # 排序体制（低到高）
        centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centers)

        # 映射到标签
        regime_map = {sorted_indices[0]: "low", sorted_indices[-1]: "high"}
        if n_regimes == 3:
            regime_map[sorted_indices[1]] = "normal"
        elif n_regimes == 4:
            regime_map[sorted_indices[1]] = "normal_low"
            regime_map[sorted_indices[2]] = "normal_high"

        # 转换为标签
        regime_labels = pd.Series(index=volatility.index)
        for i, regime in enumerate(regimes):
            regime_labels.iloc[i] = regime_map.get(regime, "normal")

        return regime_labels

    def _calculate_volatility(self, price_data: pd.DataFrame) -> float:
        """计算波动率

        Args:
            price_data: 价格数据

        Returns:
            波动率值
        """
        if self.config.volatility_measure == VolatilityMeasure.STANDARD_DEVIATION:
            returns = price_data["close"].pct_change()
            if self.config.use_ewma:
                vol = returns.ewm(span=self.config.ewma_span).std().iloc[-1]
            else:
                vol = returns.rolling(window=self.config.lookback_period).std().iloc[-1]

        elif self.config.volatility_measure == VolatilityMeasure.ATR:
            vol = self._calculate_atr_volatility(price_data)

        elif self.config.volatility_measure == VolatilityMeasure.REALIZED_VOLATILITY:
            vol = self.calculate_realized_volatility(
                price_data["high"],
                price_data["low"],
                price_data["close"],
                price_data.get("open"),
            )

        elif self.config.volatility_measure == VolatilityMeasure.GARCH:
            returns = price_data["close"].pct_change().dropna()
            garch_params = self.fit_garch_model(returns)
            vol = garch_params["conditional_volatility"].iloc[-1]

        elif self.config.volatility_measure == VolatilityMeasure.RANGE_BASED:
            vol = self._calculate_range_volatility(price_data)

        else:
            # 默认使用标准差
            returns = price_data["close"].pct_change()
            vol = returns.std()

        # 年化（如果需要）
        if vol < 1:  # 假设日波动率
            vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)

        return vol

    def _get_volatility_state(
        self, symbol: str, price_data: pd.DataFrame
    ) -> VolatilityState:
        """获取波动率状态

        Args:
            symbol: 标的代码
            price_data: 价格数据

        Returns:
            波动率状态
        """
        returns = price_data["close"].pct_change()

        # 计算历史波动率
        hist_vol = self.calculate_dynamic_volatility(returns)

        # 当前波动率
        current_vol = hist_vol.iloc[-1]

        # 波动率百分位
        vol_percentile = stats.percentileofscore(hist_vol.dropna(), current_vol)

        # 判断体制
        if vol_percentile < 25:
            regime = "low"
        elif vol_percentile < 75:
            regime = "normal"
        elif vol_percentile < 95:
            regime = "high"
        else:
            regime = "extreme"

        state = VolatilityState(
            current_volatility=current_vol,
            historical_volatility=hist_vol,
            volatility_regime=regime,
            volatility_percentile=vol_percentile,
            forecast_volatility=None,
            volatility_term_structure=None,
        )

        # 缓存状态
        self.volatility_cache[symbol] = state

        return state

    def _calculate_stop_distance(
        self, current_price: float, volatility: float, vol_state: VolatilityState
    ) -> float:
        """计算止损距离

        Args:
            current_price: 当前价格
            volatility: 波动率
            vol_state: 波动率状态

        Returns:
            止损距离
        """
        # 基础距离
        base_distance = current_price * volatility * self.config.volatility_multiplier

        # 根据波动率体制调整
        if vol_state.volatility_regime == "extreme":
            # 极端波动时增加距离
            base_distance *= 1.5
        elif vol_state.volatility_regime == "low":
            # 低波动时减少距离
            base_distance *= 0.8

        return base_distance

    def _adjust_for_regime(self, volatility: float, regime: str) -> float:
        """根据体制调整波动率

        Args:
            volatility: 原始波动率
            regime: 波动率体制

        Returns:
            调整后的波动率
        """
        adjustments = {"low": 0.8, "normal": 1.0, "high": 1.2, "extreme": 1.5}

        return volatility * adjustments.get(regime, 1.0)

    def _apply_limits(self, stop_distance: float, current_price: float) -> float:
        """应用止损距离限制

        Args:
            stop_distance: 原始止损距离
            current_price: 当前价格

        Returns:
            限制后的止损距离
        """
        min_distance = current_price * self.config.min_stop_distance
        max_distance = current_price * self.config.max_stop_distance

        return np.clip(stop_distance, min_distance, max_distance)

    def _calculate_confidence_interval(
        self, stop_price: float, volatility: float, confidence_level: float
    ) -> Tuple[float, float]:
        """计算置信区间

        Args:
            stop_price: 止损价格
            volatility: 波动率
            confidence_level: 置信水平

        Returns:
            (下界, 上界)
        """
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = stop_price * volatility * z_score

        return (stop_price - margin, stop_price + margin)

    def _estimate_hit_probability(
        self, current_price: float, stop_price: float, volatility: float
    ) -> float:
        """估计触发概率

        Args:
            current_price: 当前价格
            stop_price: 止损价格
            volatility: 波动率

        Returns:
            触发概率
        """
        # 使用正态分布近似
        distance = abs(current_price - stop_price) / current_price
        z_score = distance / volatility

        # 单尾概率
        probability = stats.norm.cdf(-z_score)

        return probability

    def _calculate_atr_volatility(self, price_data: pd.DataFrame) -> float:
        """计算ATR波动率

        Args:
            price_data: 价格数据

        Returns:
            ATR波动率
        """
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.config.lookback_period).mean().iloc[-1]

        # 转换为百分比
        return atr / close.iloc[-1]

    def _calculate_range_volatility(self, price_data: pd.DataFrame) -> float:
        """计算范围波动率

        Args:
            price_data: 价格数据

        Returns:
            范围波动率
        """
        high = price_data["high"]
        low = price_data["low"]

        # 日内范围
        daily_range = (high - low) / ((high + low) / 2)

        # 平均范围
        avg_range = (
            daily_range.rolling(window=self.config.lookback_period).mean().iloc[-1]
        )

        return avg_range

    def _fit_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """拟合GARCH波动率

        Args:
            returns: 收益率序列

        Returns:
            条件波动率序列
        """
        try:
            from arch import arch_model

            # 拟合GARCH模型
            model = arch_model(
                returns * 100, vol="Garch", p=self.config.garch_p, q=self.config.garch_q
            )
            res = model.fit(disp="off")

            # 返回条件波动率
            return res.conditional_volatility / 100

        except ImportError:
            logger.warning("arch package not available, using standard deviation")
            return returns.rolling(window=self.config.lookback_period).std()


# 模块级别函数
def calculate_volatility_stop(
    symbol: str,
    current_price: float,
    price_data: pd.DataFrame,
    config: Optional[VolatilityStopConfig] = None,
) -> Dict[str, float]:
    """计算波动率止损的便捷函数

    Args:
        symbol: 标的代码
        current_price: 当前价格
        price_data: 价格数据
        config: 配置

    Returns:
        止损信息字典
    """
    vol_stops = VolatilityBasedStops(config)
    result = vol_stops.calculate_volatility_stop(symbol, current_price, price_data)

    return {
        "stop_price": result.stop_price,
        "stop_distance_pct": result.stop_distance_pct,
        "volatility": result.volatility_used,
        "hit_probability": result.expected_hit_probability,
    }
