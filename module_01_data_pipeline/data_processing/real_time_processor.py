"""
实时数据处理器模块
负责实时处理市场数据，生成交易信号和风险指标
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import threading
import time

from common.logging_system import setup_logger
from common.exceptions import DataError
from module_02_feature_engineering.feature_extraction.technical_indicators import TechnicalIndicators
from module_05_risk_management.position_sizing.kelly_criterion import KellyCriterion

logger = setup_logger("real_time_processor")

@dataclass
class MarketSignal:
    """市场信号数据结构"""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 信号强度 0-1
    price: float
    confidence: float
    indicators: Dict[str, float]
    risk_score: float

@dataclass
class PortfolioMetrics:
    """投资组合指标"""
    timestamp: datetime
    total_value: float
    daily_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    positions: Dict[str, Dict[str, float]]

class RealTimeProcessor:
    """实时数据处理器类"""
    
    def __init__(self, config: Dict):
        """初始化实时处理器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.technical_indicators = TechnicalIndicators()
        self.kelly_criterion = KellyCriterion()
        
        # 数据缓存
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        self.signal_cache: Dict[str, List[MarketSignal]] = {}
        self.portfolio_metrics: Optional[PortfolioMetrics] = None
        
        # 回调函数
        self.signal_callbacks: List[Callable] = []
        self.metrics_callbacks: List[Callable] = []
        
        # 运行状态
        self.is_running = False
        self.processing_thread = None
        
    def add_signal_callback(self, callback: Callable):
        """添加信号回调函数
        
        Args:
            callback: 信号回调函数
        """
        self.signal_callbacks.append(callback)
        
    def add_metrics_callback(self, callback: Callable):
        """添加指标回调函数
        
        Args:
            callback: 指标回调函数
        """
        self.metrics_callbacks.append(callback)
    
    def update_market_data(self, symbol: str, data: pd.DataFrame):
        """更新市场数据
        
        Args:
            symbol: 股票代码
            data: 市场数据
        """
        try:
            # 更新缓存
            if symbol in self.market_data_cache:
                # 合并新数据
                self.market_data_cache[symbol] = pd.concat([
                    self.market_data_cache[symbol], 
                    data
                ]).drop_duplicates().sort_index()
            else:
                self.market_data_cache[symbol] = data.copy()
            
            # 保持最近1000条记录
            if len(self.market_data_cache[symbol]) > 1000:
                self.market_data_cache[symbol] = self.market_data_cache[symbol].tail(1000)
            
            logger.info(f"Updated market data for {symbol}: {len(data)} records")
            
        except Exception as e:
            logger.error(f"Failed to update market data for {symbol}: {e}")
            raise DataError(f"Market data update failed: {e}")
    
    def generate_signals(self, symbol: str) -> List[MarketSignal]:
        """生成交易信号
        
        Args:
            symbol: 股票代码
            
        Returns:
            交易信号列表
        """
        try:
            if symbol not in self.market_data_cache:
                return []
            
            data = self.market_data_cache[symbol]
            if len(data) < 50:  # 需要足够的历史数据
                return []
            
            signals = []
            
            # 计算技术指标
            indicators_data = self.technical_indicators.calculate_all_indicators(data)
            
            # 获取最新数据
            latest_data = indicators_data.iloc[-1]
            latest_price = latest_data['close']
            
            # 生成各种信号
            signals.extend(self._generate_sma_signals(symbol, indicators_data, latest_price))
            signals.extend(self._generate_rsi_signals(symbol, indicators_data, latest_price))
            signals.extend(self._generate_macd_signals(symbol, indicators_data, latest_price))
            signals.extend(self._generate_bollinger_signals(symbol, indicators_data, latest_price))
            
            # 计算风险评分
            for signal in signals:
                signal.risk_score = self._calculate_risk_score(symbol, signal)
            
            # 缓存信号
            if symbol not in self.signal_cache:
                self.signal_cache[symbol] = []
            
            self.signal_cache[symbol].extend(signals)
            
            # 保持最近100个信号
            if len(self.signal_cache[symbol]) > 100:
                self.signal_cache[symbol] = self.signal_cache[symbol][-100:]
            
            # 触发回调
            for callback in self.signal_callbacks:
                try:
                    callback(symbol, signals)
                except Exception as e:
                    logger.error(f"Signal callback error: {e}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to generate signals for {symbol}: {e}")
            return []
    
    def _generate_sma_signals(self, symbol: str, data: pd.DataFrame, price: float) -> List[MarketSignal]:
        """生成简单移动平均信号"""
        signals = []
        
        try:
            sma_5 = data['sma_5'].iloc[-1]
            sma_20 = data['sma_20'].iloc[-1]
            sma_50 = data['sma_50'].iloc[-1]
            
            # 金叉死叉信号
            if sma_5 > sma_20 and data['sma_5'].iloc[-2] <= data['sma_20'].iloc[-2]:
                signal = MarketSignal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    signal_type='BUY',
                    strength=0.7,
                    price=price,
                    confidence=0.8,
                    indicators={'sma_5': sma_5, 'sma_20': sma_20},
                    risk_score=0.0
                )
                signals.append(signal)
            
            elif sma_5 < sma_20 and data['sma_5'].iloc[-2] >= data['sma_20'].iloc[-2]:
                signal = MarketSignal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    signal_type='SELL',
                    strength=0.7,
                    price=price,
                    confidence=0.8,
                    indicators={'sma_5': sma_5, 'sma_20': sma_20},
                    risk_score=0.0
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"Failed to generate SMA signals: {e}")
        
        return signals
    
    def _generate_rsi_signals(self, symbol: str, data: pd.DataFrame, price: float) -> List[MarketSignal]:
        """生成RSI信号"""
        signals = []
        
        try:
            rsi = data['rsi'].iloc[-1]
            
            # 超买超卖信号
            if rsi < 30:  # 超卖
                signal = MarketSignal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    signal_type='BUY',
                    strength=0.8,
                    price=price,
                    confidence=0.9,
                    indicators={'rsi': rsi},
                    risk_score=0.0
                )
                signals.append(signal)
            
            elif rsi > 70:  # 超买
                signal = MarketSignal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    signal_type='SELL',
                    strength=0.8,
                    price=price,
                    confidence=0.9,
                    indicators={'rsi': rsi},
                    risk_score=0.0
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"Failed to generate RSI signals: {e}")
        
        return signals
    
    def _generate_macd_signals(self, symbol: str, data: pd.DataFrame, price: float) -> List[MarketSignal]:
        """生成MACD信号"""
        signals = []
        
        try:
            macd = data['macd'].iloc[-1]
            macd_signal = data['macd_signal'].iloc[-1]
            macd_histogram = data['macd_histogram'].iloc[-1]
            
            # MACD金叉死叉
            if macd > macd_signal and data['macd'].iloc[-2] <= data['macd_signal'].iloc[-2]:
                signal = MarketSignal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    signal_type='BUY',
                    strength=0.6,
                    price=price,
                    confidence=0.7,
                    indicators={'macd': macd, 'macd_signal': macd_signal},
                    risk_score=0.0
                )
                signals.append(signal)
            
            elif macd < macd_signal and data['macd'].iloc[-2] >= data['macd_signal'].iloc[-2]:
                signal = MarketSignal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    signal_type='SELL',
                    strength=0.6,
                    price=price,
                    confidence=0.7,
                    indicators={'macd': macd, 'macd_signal': macd_signal},
                    risk_score=0.0
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"Failed to generate MACD signals: {e}")
        
        return signals
    
    def _generate_bollinger_signals(self, symbol: str, data: pd.DataFrame, price: float) -> List[MarketSignal]:
        """生成布林带信号"""
        signals = []
        
        try:
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            bb_middle = data['bb_middle'].iloc[-1]
            
            # 布林带突破信号
            if price < bb_lower:  # 价格触及下轨
                signal = MarketSignal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    signal_type='BUY',
                    strength=0.9,
                    price=price,
                    confidence=0.85,
                    indicators={'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_middle': bb_middle},
                    risk_score=0.0
                )
                signals.append(signal)
            
            elif price > bb_upper:  # 价格触及上轨
                signal = MarketSignal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    signal_type='SELL',
                    strength=0.9,
                    price=price,
                    confidence=0.85,
                    indicators={'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_middle': bb_middle},
                    risk_score=0.0
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"Failed to generate Bollinger signals: {e}")
        
        return signals
    
    def _calculate_risk_score(self, symbol: str, signal: MarketSignal) -> float:
        """计算风险评分
        
        Args:
            symbol: 股票代码
            signal: 交易信号
            
        Returns:
            风险评分 (0-1, 1表示最高风险)
        """
        try:
            if symbol not in self.market_data_cache:
                return 0.5
            
            data = self.market_data_cache[symbol]
            if len(data) < 20:
                return 0.5
            
            # 计算波动率
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # 计算最大回撤
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # 综合风险评分
            risk_score = min(1.0, (volatility * 0.4 + max_drawdown * 0.6))
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Failed to calculate risk score: {e}")
            return 0.5
    
    def calculate_portfolio_metrics(self, positions: Dict[str, Dict]) -> PortfolioMetrics:
        """计算投资组合指标
        
        Args:
            positions: 持仓信息
            
        Returns:
            投资组合指标
        """
        try:
            total_value = 0
            total_cost = 0
            position_metrics = {}
            
            for symbol, position in positions.items():
                if symbol in self.market_data_cache:
                    data = self.market_data_cache[symbol]
                    if len(data) > 0:
                        current_price = data['close'].iloc[-1]
                        quantity = position.get('quantity', 0)
                        cost_price = position.get('cost_price', 0)
                        
                        market_value = quantity * current_price
                        cost_value = quantity * cost_price
                        pnl = market_value - cost_value
                        pnl_rate = pnl / cost_value if cost_value > 0 else 0
                        
                        total_value += market_value
                        total_cost += cost_value
                        
                        position_metrics[symbol] = {
                            'market_value': market_value,
                            'cost_value': cost_value,
                            'pnl': pnl,
                            'pnl_rate': pnl_rate,
                            'current_price': current_price
                        }
            
            # 计算组合指标
            daily_return = (total_value - total_cost) / total_cost if total_cost > 0 else 0
            
            # 计算波动率（简化版本）
            volatility = 0.15  # 默认15%
            
            # 计算夏普比率（简化版本）
            sharpe_ratio = daily_return / volatility if volatility > 0 else 0
            
            # 计算最大回撤（简化版本）
            max_drawdown = -0.05  # 默认-5%
            
            # 计算VaR（简化版本）
            var_95 = -0.02  # 默认-2%
            
            metrics = PortfolioMetrics(
                timestamp=datetime.now(),
                total_value=total_value,
                daily_return=daily_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                positions=position_metrics
            )
            
            self.portfolio_metrics = metrics
            
            # 触发回调
            for callback in self.metrics_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"Metrics callback error: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio metrics: {e}")
            raise DataError(f"Portfolio metrics calculation failed: {e}")
    
    def start_processing(self):
        """启动实时处理"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Real-time processing started")
    
    def stop_processing(self):
        """停止实时处理"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        
        logger.info("Real-time processing stopped")
    
    def _processing_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                # 为每个有数据的股票生成信号
                for symbol in self.market_data_cache.keys():
                    if self.is_running:
                        self.generate_signals(symbol)
                
                # 等待下次处理
                time.sleep(5)  # 每5秒处理一次
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(10)  # 出错时等待更长时间
    
    def get_latest_signals(self, symbol: str, limit: int = 10) -> List[MarketSignal]:
        """获取最新信号
        
        Args:
            symbol: 股票代码
            limit: 返回数量限制
            
        Returns:
            最新信号列表
        """
        if symbol not in self.signal_cache:
            return []
        
        return self.signal_cache[symbol][-limit:]
    
    def get_portfolio_metrics(self) -> Optional[PortfolioMetrics]:
        """获取投资组合指标"""
        return self.portfolio_metrics
