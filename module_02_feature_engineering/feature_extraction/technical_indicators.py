"""
技术指标计算模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from common.logging_system import setup_logger
from common.exceptions import DataError

logger = setup_logger("technical_indicators")

@dataclass
class IndicatorConfig:
    """技术指标配置"""
    name: str
    parameters: Dict[str, Union[int, float]]
    enabled: bool = True

class TechnicalIndicators:
    """技术指标计算器类"""
    
    def __init__(self):
        """初始化技术指标计算器"""
        self.indicators = {}
        
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """计算简单移动平均线
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            简单移动平均线
        """
        try:
            return data.rolling(window=period).mean()
        except Exception as e:
            logger.error(f"Failed to calculate SMA: {e}")
            raise DataError(f"SMA calculation failed: {e}")
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均线
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            指数移动平均线
        """
        try:
            return data.ewm(span=period).mean()
        except Exception as e:
            logger.error(f"Failed to calculate EMA: {e}")
            raise DataError(f"EMA calculation failed: {e}")
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """计算相对强弱指数
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            RSI指标
        """
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            raise DataError(f"RSI calculation failed: {e}")
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """计算MACD指标
        
        Args:
            data: 价格数据
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        Returns:
            MACD指标字典
        """
        try:
            ema_fast = self.calculate_ema(data, fast)
            ema_slow = self.calculate_ema(data, slow)
            macd_line = ema_fast - ema_slow
            signal_line = self.calculate_ema(macd_line, signal)
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            logger.error(f"Failed to calculate MACD: {e}")
            raise DataError(f"MACD calculation failed: {e}")
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """计算布林带
        
        Args:
            data: 价格数据
            period: 周期
            std_dev: 标准差倍数
            
        Returns:
            布林带指标字典
        """
        try:
            sma = self.calculate_sma(data, period)
            std = data.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return {
                'upper': upper_band,
                'middle': sma,
                'lower': lower_band
            }
        except Exception as e:
            logger.error(f"Failed to calculate Bollinger Bands: {e}")
            raise DataError(f"Bollinger Bands calculation failed: {e}")
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算平均真实波幅
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            ATR指标
        """
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr
        except Exception as e:
            logger.error(f"Failed to calculate ATR: {e}")
            raise DataError(f"ATR calculation failed: {e}")
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """计算随机指标
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            k_period: K值周期
            d_period: D值周期
            
        Returns:
            随机指标字典
        """
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return {
                'k': k_percent,
                'd': d_percent
            }
        except Exception as e:
            logger.error(f"Failed to calculate Stochastic: {e}")
            raise DataError(f"Stochastic calculation failed: {e}")
    
    def calculate_all_indicators(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标
        
        Args:
            ohlcv_data: OHLCV数据
            
        Returns:
            包含所有指标的DataFrame
        """
        try:
            result = ohlcv_data.copy()
            
            # 确保有必要的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in ohlcv_data.columns:
                    raise DataError(f"Missing required column: {col}")
            
            # 计算各种指标
            result['sma_5'] = self.calculate_sma(ohlcv_data['close'], 5)
            result['sma_10'] = self.calculate_sma(ohlcv_data['close'], 10)
            result['sma_20'] = self.calculate_sma(ohlcv_data['close'], 20)
            result['sma_50'] = self.calculate_sma(ohlcv_data['close'], 50)
            
            result['ema_12'] = self.calculate_ema(ohlcv_data['close'], 12)
            result['ema_26'] = self.calculate_ema(ohlcv_data['close'], 26)
            
            result['rsi'] = self.calculate_rsi(ohlcv_data['close'])
            
            # MACD
            macd_data = self.calculate_macd(ohlcv_data['close'])
            result['macd'] = macd_data['macd']
            result['macd_signal'] = macd_data['signal']
            result['macd_histogram'] = macd_data['histogram']
            
            # 布林带
            bb_data = self.calculate_bollinger_bands(ohlcv_data['close'])
            result['bb_upper'] = bb_data['upper']
            result['bb_middle'] = bb_data['middle']
            result['bb_lower'] = bb_data['lower']
            
            # ATR
            result['atr'] = self.calculate_atr(
                ohlcv_data['high'], 
                ohlcv_data['low'], 
                ohlcv_data['close']
            )
            
            # 随机指标
            stoch_data = self.calculate_stochastic(
                ohlcv_data['high'], 
                ohlcv_data['low'], 
                ohlcv_data['close']
            )
            result['stoch_k'] = stoch_data['k']
            result['stoch_d'] = stoch_data['d']
            
            logger.info(f"Calculated {len(result.columns) - len(ohlcv_data.columns)} technical indicators")
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate all indicators: {e}")
            raise DataError(f"Technical indicators calculation failed: {e}")

# 便捷函数
def calculate_technical_indicators(ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标的便捷函数
    
    Args:
        ohlcv_data: OHLCV数据
        
    Returns:
        包含技术指标的DataFrame
    """
    calculator = TechnicalIndicators()
    return calculator.calculate_all_indicators(ohlcv_data)