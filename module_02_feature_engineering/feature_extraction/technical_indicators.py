"""
技术指标特征提取器模块
计算各类技术分析指标
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import talib
from common.exceptions import DataError
from common.logging_system import setup_logger

logger = setup_logger("technical_indicators")


@dataclass
class IndicatorConfig:
    """指标配置"""

    ma_periods: List[int] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    cci_period: int = 20

    def __post_init__(self):
        """初始化后处理"""
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 50, 100, 200]


class TechnicalIndicators:
    """技术指标计算器"""

    # 指标分类
    INDICATOR_CATEGORIES = {
        "trend": ["sma", "ema", "wma", "macd", "adx", "sar"],
        "momentum": ["rsi", "stoch", "williams", "roc", "mom", "cci"],
        "volatility": ["bb", "atr", "natr", "trange"],
        "volume": ["obv", "ad", "adosc", "mfi", "vwap"],
        "pattern": ["cdl_patterns", "support_resistance"],
    }

    def __init__(self, config: Optional[IndicatorConfig] = None):
        """初始化技术指标计算器

        Args:
            config: 指标配置
        """
        self.config = config or IndicatorConfig()
        self.calculated_indicators: Dict[str, pd.DataFrame] = {}

    def calculate_all_indicators(
        self, df: pd.DataFrame, categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """计算所有指标

        Args:
            df: OHLCV数据DataFrame
            categories: 要计算的指标类别列表

        Returns:
            包含所有指标的DataFrame
        """
        if categories is None:
            categories = list(self.INDICATOR_CATEGORIES.keys())

        result_dfs = [df.copy()]

        for category in categories:
            if category == "trend":
                result_dfs.append(self.calculate_trend_indicators(df))
            elif category == "momentum":
                result_dfs.append(self.calculate_momentum_indicators(df))
            elif category == "volatility":
                result_dfs.append(self.calculate_volatility_indicators(df))
            elif category == "volume":
                result_dfs.append(self.calculate_volume_indicators(df))
            elif category == "pattern":
                result_dfs.append(self.calculate_pattern_indicators(df))

        # 合并所有指标
        result = pd.concat(result_dfs, axis=1)

        # 删除重复列
        result = result.loc[:, ~result.columns.duplicated()]

        logger.info(f"Calculated {len(result.columns)} technical indicators")
        return result

    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势指标

        Args:
            df: OHLCV数据

        Returns:
            趋势指标DataFrame
        """
        indicators = pd.DataFrame(index=df.index)

        # 移动平均线
        for period in self.config.ma_periods:
            indicators[f"sma_{period}"] = talib.SMA(df["close"], timeperiod=period)
            indicators[f"ema_{period}"] = talib.EMA(df["close"], timeperiod=period)
            indicators[f"wma_{period}"] = talib.WMA(df["close"], timeperiod=period)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            df["close"],
            fastperiod=self.config.macd_fast,
            slowperiod=self.config.macd_slow,
            signalperiod=self.config.macd_signal,
        )
        indicators["macd"] = macd
        indicators["macd_signal"] = macd_signal
        indicators["macd_hist"] = macd_hist

        # ADX (Average Directional Index)
        indicators["adx"] = talib.ADX(
            df["high"], df["low"], df["close"], timeperiod=self.config.adx_period
        )
        indicators["plus_di"] = talib.PLUS_DI(
            df["high"], df["low"], df["close"], timeperiod=self.config.adx_period
        )
        indicators["minus_di"] = talib.MINUS_DI(
            df["high"], df["low"], df["close"], timeperiod=self.config.adx_period
        )

        # SAR (Parabolic Stop and Reverse)
        indicators["sar"] = talib.SAR(df["high"], df["low"])

        # 趋势线斜率
        for period in [20, 50]:
            indicators[f"trend_slope_{period}"] = self._calculate_trend_slope(
                df["close"], period
            )

        return indicators

    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量指标

        Args:
            df: OHLCV数据

        Returns:
            动量指标DataFrame
        """
        indicators = pd.DataFrame(index=df.index)

        # RSI
        indicators["rsi"] = talib.RSI(df["close"], timeperiod=self.config.rsi_period)
        indicators["rsi_ma"] = indicators["rsi"].rolling(window=9).mean()

        # Stochastic
        slowk, slowd = talib.STOCH(
            df["high"],
            df["low"],
            df["close"],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3,
        )
        indicators["stoch_k"] = slowk
        indicators["stoch_d"] = slowd

        # Williams %R
        indicators["williams_r"] = talib.WILLR(
            df["high"], df["low"], df["close"], timeperiod=14
        )

        # ROC (Rate of Change)
        indicators["roc"] = talib.ROC(df["close"], timeperiod=10)

        # Momentum
        indicators["mom"] = talib.MOM(df["close"], timeperiod=10)

        # CCI (Commodity Channel Index)
        indicators["cci"] = talib.CCI(
            df["high"], df["low"], df["close"], timeperiod=self.config.cci_period
        )

        # MFI (Money Flow Index)
        if "volume" in df.columns:
            indicators["mfi"] = talib.MFI(
                df["high"], df["low"], df["close"], df["volume"], timeperiod=14
            )

        return indicators

    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率指标

        Args:
            df: OHLCV数据

        Returns:
            波动率指标DataFrame
        """
        indicators = pd.DataFrame(index=df.index)

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df["close"],
            timeperiod=self.config.bb_period,
            nbdevup=self.config.bb_std,
            nbdevdn=self.config.bb_std,
        )
        indicators["bb_upper"] = upper
        indicators["bb_middle"] = middle
        indicators["bb_lower"] = lower
        indicators["bb_width"] = upper - lower
        indicators["bb_percent"] = (df["close"] - lower) / (upper - lower)

        # ATR (Average True Range)
        indicators["atr"] = talib.ATR(
            df["high"], df["low"], df["close"], timeperiod=self.config.atr_period
        )

        # NATR (Normalized ATR)
        indicators["natr"] = talib.NATR(
            df["high"], df["low"], df["close"], timeperiod=self.config.atr_period
        )

        # True Range
        indicators["trange"] = talib.TRANGE(df["high"], df["low"], df["close"])

        # 历史波动率
        indicators["hist_volatility"] = (
            df["close"].pct_change().rolling(window=20).std()
        )

        # Parkinson波动率
        indicators["parkinson_vol"] = self._calculate_parkinson_volatility(
            df["high"], df["low"]
        )

        # Garman-Klass波动率
        indicators["gk_vol"] = self._calculate_garman_klass_volatility(
            df["open"], df["high"], df["low"], df["close"]
        )

        return indicators

    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量指标

        Args:
            df: OHLCV数据

        Returns:
            成交量指标DataFrame
        """
        indicators = pd.DataFrame(index=df.index)

        if "volume" not in df.columns:
            logger.warning("Volume data not available")
            return indicators

        # OBV (On Balance Volume)
        indicators["obv"] = talib.OBV(df["close"], df["volume"])

        # AD (Accumulation/Distribution)
        indicators["ad"] = talib.AD(df["high"], df["low"], df["close"], df["volume"])

        # ADOSC (Chaikin A/D Oscillator)
        indicators["adosc"] = talib.ADOSC(
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            fastperiod=3,
            slowperiod=10,
        )

        # VWAP
        indicators["vwap"] = self._calculate_vwap(
            df["high"], df["low"], df["close"], df["volume"]
        )

        # Volume Rate of Change
        indicators["vroc"] = talib.ROC(df["volume"], timeperiod=10)

        # Volume Moving Average
        indicators["volume_sma"] = talib.SMA(df["volume"], timeperiod=20)

        # Volume Ratio
        indicators["volume_ratio"] = df["volume"] / indicators["volume_sma"]

        return indicators

    def calculate_pattern_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算形态指标

        Args:
            df: OHLCV数据

        Returns:
            形态指标DataFrame
        """
        indicators = pd.DataFrame(index=df.index)

        # 蜡烛图形态
        cdl_patterns = {
            "cdl_doji": talib.CDLDOJI,
            "cdl_hammer": talib.CDLHAMMER,
            "cdl_hanging_man": talib.CDLHANGINGMAN,
            "cdl_engulfing": talib.CDLENGULFING,
            "cdl_morning_star": talib.CDLMORNINGSTAR,
            "cdl_evening_star": talib.CDLEVENINGSTAR,
            "cdl_three_white_soldiers": talib.CDL3WHITESOLDIERS,
            "cdl_three_black_crows": talib.CDL3BLACKCROWS,
        }

        for name, func in cdl_patterns.items():
            try:
                indicators[name] = func(df["open"], df["high"], df["low"], df["close"])
            except:
                indicators[name] = 0

        # 支撑阻力位
        support, resistance = self._calculate_support_resistance(
            df["high"], df["low"], df["close"]
        )
        indicators["support"] = support
        indicators["resistance"] = resistance

        # 价格通道
        indicators["price_channel_upper"] = df["high"].rolling(window=20).max()
        indicators["price_channel_lower"] = df["low"].rolling(window=20).min()
        indicators["price_channel_mid"] = (
            indicators["price_channel_upper"] + indicators["price_channel_lower"]
        ) / 2

        return indicators

    def _calculate_trend_slope(self, prices: pd.Series, period: int) -> pd.Series:
        """计算趋势斜率

        Args:
            prices: 价格序列
            period: 周期

        Returns:
            斜率序列
        """

        def calculate_slope(x):
            if len(x) < 2:
                return np.nan
            y = np.arange(len(x))
            slope = np.polyfit(y, x, 1)[0]
            return slope

        return prices.rolling(window=period).apply(calculate_slope)

    def _calculate_vwap(
        self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """计算VWAP

        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            volume: 成交量

        Returns:
            VWAP序列
        """
        typical_price = (high + low + close) / 3
        cumulative_tpv = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()

        vwap = cumulative_tpv / cumulative_volume
        return vwap

    def _calculate_parkinson_volatility(
        self, high: pd.Series, low: pd.Series, window: int = 20
    ) -> pd.Series:
        """计算Parkinson波动率

        Args:
            high: 最高价
            low: 最低价
            window: 窗口大小

        Returns:
            波动率序列
        """
        log_hl = np.log(high / low)
        factor = 1 / (4 * np.log(2))

        return log_hl.pow(2).rolling(window=window).mean().apply(np.sqrt) * np.sqrt(
            factor
        )

    def _calculate_garman_klass_volatility(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """计算Garman-Klass波动率

        Args:
            open_price: 开盘价
            high: 最高价
            low: 最低价
            close: 收盘价
            window: 窗口大小

        Returns:
            波动率序列
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open_price)

        gk = 0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)

        return gk.rolling(window=window).mean().apply(np.sqrt)

    def _calculate_support_resistance(
        self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """计算支撑阻力位

        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            window: 窗口大小

        Returns:
            支撑位和阻力位序列
        """
        # 简化的支撑阻力计算
        support = low.rolling(window=window).min()
        resistance = high.rolling(window=window).max()

        return support, resistance


# 模块级别函数
def extract_technical_features(
    df: pd.DataFrame, config: Optional[IndicatorConfig] = None
) -> pd.DataFrame:
    """提取技术指标特征的便捷函数

    Args:
        df: OHLCV数据
        config: 指标配置

    Returns:
        包含技术指标的DataFrame
    """
    calculator = TechnicalIndicators(config)
    return calculator.calculate_all_indicators(df)
