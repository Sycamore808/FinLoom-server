"""
信号生成器模块
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from common.data_structures import Signal
from common.logging_system import setup_logger
from common.exceptions import QuantSystemError

logger = setup_logger("signal_generator")

class SignalType(Enum):
    """信号类型枚举"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class SignalPriority(Enum):
    """信号优先级枚举"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10

@dataclass
class EnhancedSignal:
    """增强信号数据结构"""
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    quantity: int
    price: float
    confidence: float
    priority: SignalPriority
    strategy_name: str
    metadata: Dict[str, Any]
    expected_return: Optional[float] = None
    risk_score: Optional[float] = None
    holding_period: Optional[int] = None

class SignalGenerator:
    """信号生成器类"""

    def __init__(self):
        """初始化信号生成器"""
        self.signal_counter = 0
        self.active_signals: Dict[str, EnhancedSignal] = {}
        
    def _generate_signal_id(self, strategy_name: str) -> str:
        """生成信号ID"""
        self.signal_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{strategy_name}_{timestamp}_{self.signal_counter}"
    
    def generate_ma_crossover_signal(
        self,
        symbol: str, 
        data: pd.DataFrame,
        short_window: int = 5,
        long_window: int = 20
    ) -> Optional[EnhancedSignal]:
        """生成移动平均线交叉信号

        Args:
            symbol: 股票代码
            data: 价格数据
            short_window: 短期均线窗口
            long_window: 长期均线窗口

        Returns:
            交易信号
        """
        try:
            if len(data) < long_window:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # 计算移动平均线
            short_ma = data['close'].rolling(window=short_window).mean()
            long_ma = data['close'].rolling(window=long_window).mean()
            
            # 获取最新值
            current_short = short_ma.iloc[-1]
            current_long = long_ma.iloc[-1]
            prev_short = short_ma.iloc[-2]
            prev_long = long_ma.iloc[-2]
            
            # 判断交叉
            signal_type = None
            confidence = 0.0
            
            if prev_short <= prev_long and current_short > current_long:
                # 金叉 - 买入信号
                signal_type = SignalType.BUY
                confidence = min(0.9, 0.5 + (current_short - current_long) / current_long * 10)
            elif prev_short >= prev_long and current_short < current_long:
                # 死叉 - 卖出信号
                signal_type = SignalType.SELL
                confidence = min(0.9, 0.5 + (current_long - current_short) / current_long * 10)
            
            if signal_type is None:
                return None
            
            # 计算信号强度
            ma_spread = abs(current_short - current_long) / current_long
            volume_ratio = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
            
            # 调整置信度
            confidence *= (1 + ma_spread * 2)  # 均线差距越大，信号越强
            confidence *= min(1.5, volume_ratio)  # 成交量放大增强信号
            
            confidence = min(0.95, max(0.1, confidence))
            
            # 计算预期收益率
            expected_return = self._calculate_expected_return(data, signal_type)
            
            # 计算风险评分
            risk_score = self._calculate_risk_score(data)
            
            signal = EnhancedSignal(
                signal_id=self._generate_signal_id("MA_CROSSOVER"),
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                quantity=100,  # 默认数量
                price=data['close'].iloc[-1],
                confidence=confidence,
                priority=SignalPriority.HIGH if confidence > 0.7 else SignalPriority.NORMAL,
                strategy_name="MA_CROSSOVER",
                metadata={
                    "short_window": short_window,
                    "long_window": long_window,
                    "short_ma": current_short,
                    "long_ma": current_long,
                    "ma_spread": ma_spread,
                    "volume_ratio": volume_ratio
                },
                expected_return=expected_return,
                risk_score=risk_score,
                holding_period=20  # 默认持有20天
            )
            
            logger.info(f"Generated MA crossover signal for {symbol}: {signal_type.value} (confidence: {confidence:.3f})")
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate MA crossover signal for {symbol}: {e}")
            return None
    
    def generate_rsi_signal(
        self,
        symbol: str, 
        data: pd.DataFrame,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70
    ) -> Optional[EnhancedSignal]:
        """生成RSI信号

        Args:
            symbol: 股票代码
            data: 价格数据
            rsi_period: RSI周期
            oversold: 超卖阈值
            overbought: 超买阈值

        Returns:
            交易信号
        """
        try:
            if len(data) < rsi_period + 1:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # 计算RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            
            signal_type = None
            confidence = 0.0
            
            # 判断RSI信号
            if current_rsi < oversold and prev_rsi >= oversold:
                # 从超卖区域反弹 - 买入信号
                signal_type = SignalType.BUY
                confidence = min(0.9, (oversold - current_rsi) / oversold * 2)
            elif current_rsi > overbought and prev_rsi <= overbought:
                # 从超买区域回落 - 卖出信号
                signal_type = SignalType.SELL
                confidence = min(0.9, (current_rsi - overbought) / (100 - overbought) * 2)
            
            if signal_type is None:
                return None
            
            # 计算预期收益率和风险评分
            expected_return = self._calculate_expected_return(data, signal_type)
            risk_score = self._calculate_risk_score(data)
            
            signal = EnhancedSignal(
                signal_id=self._generate_signal_id("RSI"),
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                quantity=100,
                price=data['close'].iloc[-1],
                confidence=confidence,
                priority=SignalPriority.HIGH if confidence > 0.7 else SignalPriority.NORMAL,
                strategy_name="RSI",
                metadata={
                    "rsi_period": rsi_period,
                    "current_rsi": current_rsi,
                    "prev_rsi": prev_rsi,
                    "oversold": oversold,
                    "overbought": overbought
                },
                expected_return=expected_return,
                risk_score=risk_score,
                holding_period=10  # RSI信号通常持有时间较短
            )
            
            logger.info(f"Generated RSI signal for {symbol}: {signal_type.value} (RSI: {current_rsi:.2f}, confidence: {confidence:.3f})")
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate RSI signal for {symbol}: {e}")
            return None
    
    def generate_bollinger_bands_signal(
        self,
        symbol: str, 
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Optional[EnhancedSignal]:
        """生成布林带信号

        Args:
            symbol: 股票代码
            data: 价格数据
            period: 布林带周期
            std_dev: 标准差倍数

        Returns:
            交易信号
        """
        try:
            if len(data) < period:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # 计算布林带
            sma = data['close'].rolling(window=period).mean()
            std = data['close'].rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            current_price = data['close'].iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_middle = sma.iloc[-1]
            
            signal_type = None
            confidence = 0.0
            
            # 判断布林带信号
            if current_price <= current_lower:
                # 价格触及下轨 - 买入信号
                signal_type = SignalType.BUY
                confidence = min(0.9, (current_lower - current_price) / current_lower * 5)
            elif current_price >= current_upper:
                # 价格触及上轨 - 卖出信号
                signal_type = SignalType.SELL
                confidence = min(0.9, (current_price - current_upper) / current_upper * 5)
            
            if signal_type is None:
                return None
            
            # 计算预期收益率和风险评分
            expected_return = self._calculate_expected_return(data, signal_type)
            risk_score = self._calculate_risk_score(data)
            
            signal = EnhancedSignal(
                signal_id=self._generate_signal_id("BOLLINGER"),
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                quantity=100,
                price=current_price,
                confidence=confidence,
                priority=SignalPriority.HIGH if confidence > 0.7 else SignalPriority.NORMAL,
                strategy_name="BOLLINGER_BANDS",
                metadata={
                    "period": period,
                    "std_dev": std_dev,
                    "upper_band": current_upper,
                    "lower_band": current_lower,
                    "middle_band": current_middle,
                    "band_width": (current_upper - current_lower) / current_middle
                },
                expected_return=expected_return,
                risk_score=risk_score,
                holding_period=15
            )
            
            logger.info(f"Generated Bollinger Bands signal for {symbol}: {signal_type.value} (confidence: {confidence:.3f})")
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate Bollinger Bands signal for {symbol}: {e}")
            return None
    
    def generate_multi_signal(
        self, 
        symbol: str, 
        data: pd.DataFrame,
        strategies: List[str] = None
    ) -> List[EnhancedSignal]:
        """生成多重信号

        Args:
            symbol: 股票代码
            data: 价格数据
            strategies: 策略列表

        Returns:
            信号列表
        """
        if strategies is None:
            strategies = ["MA_CROSSOVER", "RSI", "BOLLINGER"]
        
        signals = []
        
        for strategy in strategies:
            try:
                if strategy == "MA_CROSSOVER":
                    signal = self.generate_ma_crossover_signal(symbol, data)
                elif strategy == "RSI":
                    signal = self.generate_rsi_signal(symbol, data)
                elif strategy == "BOLLINGER":
                    signal = self.generate_bollinger_bands_signal(symbol, data)
                else:
                    continue
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Failed to generate {strategy} signal for {symbol}: {e}")
                continue
        
        return signals
    
    def _calculate_expected_return(self, data: pd.DataFrame, signal_type: SignalType) -> float:
        """计算预期收益率

        Args:
            data: 价格数据
            signal_type: 信号类型

        Returns:
            预期收益率
        """
        try:
            # 计算历史收益率
            returns = data['close'].pct_change().dropna()
            
            if signal_type == SignalType.BUY:
                # 买入信号：计算正收益率的平均值
                positive_returns = returns[returns > 0]
                expected_return = positive_returns.mean() if len(positive_returns) > 0 else 0.0
            else:
                # 卖出信号：计算负收益率的平均值（绝对值）
                negative_returns = returns[returns < 0]
                expected_return = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0.0
            
            return expected_return * 100  # 转换为百分比
            
        except Exception as e:
            logger.error(f"Failed to calculate expected return: {e}")
            return 0.0
    
    def _calculate_risk_score(self, data: pd.DataFrame) -> float:
        """计算风险评分

        Args:
            data: 价格数据

        Returns:
            风险评分 (0-1)
        """
        try:
            # 计算波动率
            returns = data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # 计算最大回撤
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # 综合风险评分
            risk_score = min(1.0, volatility * 10 + max_drawdown * 2)
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Failed to calculate risk score: {e}")
            return 0.5  # 默认中等风险
    
    def filter_signals(
        self,
        signals: List[EnhancedSignal],
        min_confidence: float = 0.3,
        max_risk: float = 0.8
    ) -> List[EnhancedSignal]:
        """过滤信号

        Args:
            signals: 信号列表
            min_confidence: 最小置信度
            max_risk: 最大风险

        Returns:
            过滤后的信号列表
        """
        filtered_signals = []
        
        for signal in signals:
            if signal.confidence >= min_confidence and (signal.risk_score or 0.5) <= max_risk:
                filtered_signals.append(signal)
        
        # 按置信度排序
        filtered_signals.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Filtered {len(signals)} signals to {len(filtered_signals)} high-quality signals")
        return filtered_signals
    
    def convert_to_signal(self, enhanced_signal: EnhancedSignal) -> Signal:
        """将增强信号转换为标准信号

        Args:
            enhanced_signal: 增强信号

        Returns:
            标准信号
        """
        return Signal(
            signal_id=enhanced_signal.signal_id,
            timestamp=enhanced_signal.timestamp,
            symbol=enhanced_signal.symbol,
            action=enhanced_signal.signal_type.value,
            quantity=enhanced_signal.quantity,
            price=enhanced_signal.price,
            confidence=enhanced_signal.confidence,
            strategy_name=enhanced_signal.strategy_name,
            metadata=enhanced_signal.metadata
        )

# 便捷函数
def generate_trading_signals(
    symbol: str, 
    data: pd.DataFrame,
    strategies: List[str] = None
) -> List[Signal]:
    """生成交易信号的便捷函数

    Args:
        symbol: 股票代码
        data: 价格数据
        strategies: 策略列表

    Returns:
        标准信号列表
    """
    generator = SignalGenerator()
    enhanced_signals = generator.generate_multi_signal(symbol, data, strategies)
    filtered_signals = generator.filter_signals(enhanced_signals)

    return [generator.convert_to_signal(signal) for signal in filtered_signals]