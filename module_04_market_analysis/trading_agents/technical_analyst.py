"""
技术分析师
分析价格图表、技术指标和交易模式
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .base_agent import BaseAgent, AgentAnalysis, AgentRecommendation, RecommendationType, DebateContext, DebateResponse, create_analysis_id
from common.logging_system import setup_logger

logger = setup_logger("technical_analyst")


class TechnicalAnalyst(BaseAgent):
    """技术分析师"""
    
    def __init__(self):
        super().__init__(
            name="技术分析师",
            agent_type="technical_analyst",
            expertise="技术指标分析, 图表形态识别, 趋势分析, 支撑阻力位分析",
            confidence_threshold=0.65
        )
        self.price_data_cache = {}
        self.technical_indicators_cache = {}
        self.pattern_recognition_cache = {}
    
    async def analyze(
        self, 
        symbols: List[str], 
        market_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """分析技术指标和图表形态"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting technical analysis for symbols: {symbols}")
            
            # 1. 获取价格数据
            price_data = await self._get_price_data(symbols, market_data)
            
            # 2. 计算技术指标
            technical_indicators = self._calculate_technical_indicators(price_data)
            
            # 3. 识别图表形态
            chart_patterns = self._identify_chart_patterns(price_data)
            
            # 4. 分析支撑阻力位
            support_resistance = self._analyze_support_resistance(price_data)
            
            # 5. 趋势分析
            trend_analysis = self._analyze_trends(price_data, technical_indicators)
            
            # 6. 生成推荐
            recommendation = self._generate_recommendation(
                technical_indicators, chart_patterns, support_resistance, trend_analysis
            )
            
            # 7. 计算分析耗时
            analysis_duration = (datetime.now() - start_time).total_seconds()
            
            # 8. 创建分析结果
            analysis = AgentAnalysis(
                agent_name=self.name,
                agent_type=self.agent_type,
                analysis_id=create_analysis_id(),
                timestamp=datetime.now(),
                symbols=symbols,
                recommendation=recommendation,
                key_factors=self._extract_key_factors(technical_indicators, chart_patterns, trend_analysis),
                risk_factors=self._extract_risk_factors(technical_indicators, chart_patterns),
                market_outlook=self._generate_market_outlook(trend_analysis, technical_indicators),
                additional_insights={
                    'technical_indicators': technical_indicators,
                    'chart_patterns': chart_patterns,
                    'support_resistance': support_resistance,
                    'trend_analysis': trend_analysis,
                    'rsi': technical_indicators.get('rsi', {}),
                    'macd': technical_indicators.get('macd', {}),
                    'bollinger_bands': technical_indicators.get('bollinger_bands', {})
                },
                data_sources=['价格数据', '成交量数据', '技术指标'],
                analysis_duration=analysis_duration
            )
            
            logger.info(f"Technical analysis completed for {symbols} in {analysis_duration:.2f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbols}: {e}")
            return self._create_default_analysis(symbols, str(e))
    
    async def debate(
        self, 
        other_analyses: List[AgentAnalysis],
        debate_context: DebateContext
    ) -> DebateResponse:
        """与其他智能体进行辩论"""
        try:
            # 分析其他智能体的观点
            opposing_views = []
            supporting_views = []
            
            for analysis in other_analyses:
                if analysis.recommendation.recommendation_type in [RecommendationType.BUY, RecommendationType.STRONG_BUY]:
                    supporting_views.append(analysis)
                elif analysis.recommendation.recommendation_type in [RecommendationType.SELL, RecommendationType.STRONG_SELL]:
                    opposing_views.append(analysis)
            
            # 基于技术分析生成辩论响应
            if len(supporting_views) > len(opposing_views):
                position = "支持"
                arguments = [
                    "技术指标显示买入信号",
                    "价格突破关键阻力位",
                    "成交量配合价格上涨"
                ]
            elif len(opposing_views) > len(supporting_views):
                position = "反对"
                arguments = [
                    "技术指标显示卖出信号",
                    "价格跌破关键支撑位",
                    "成交量萎缩显示买盘不足"
                ]
            else:
                position = "中立"
                arguments = [
                    "技术指标信号不明确",
                    "价格在关键位置震荡",
                    "需要等待更明确的信号"
                ]
            
            return DebateResponse(
                agent_name=self.name,
                position=position,
                arguments=arguments,
                evidence={
                    'technical_signals': self._get_current_technical_signals(),
                    'price_action': self._analyze_price_action(),
                    'volume_analysis': self._analyze_volume_patterns()
                },
                counter_arguments=self._generate_counter_arguments(other_analyses),
                updated_confidence=self._calculate_debate_confidence(other_analyses),
                willingness_to_compromise=0.5  # 技术分析相对灵活
            )
            
        except Exception as e:
            logger.error(f"Debate failed for {self.name}: {e}")
            return DebateResponse(
                agent_name=self.name,
                position="中立",
                arguments=["技术分析过程中出现错误"],
                evidence={},
                updated_confidence=0.3
            )
    
    async def _get_price_data(self, symbols: List[str], market_data: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        """获取价格数据"""
        try:
            price_data = {}
            
            for symbol in symbols:
                if market_data and symbol in market_data:
                    # 使用提供的市场数据
                    data = market_data[symbol]
                    if isinstance(data, pd.DataFrame):
                        price_data[symbol] = data
                    else:
                        # 转换为DataFrame格式
                        price_data[symbol] = self._convert_to_dataframe(data)
                else:
                    # 生成模拟价格数据
                    price_data[symbol] = self._generate_mock_price_data(symbol)
            
            # 缓存数据
            cache_key = f"price_{','.join(sorted(symbols))}_{datetime.now().strftime('%Y%m%d')}"
            self.price_data_cache[cache_key] = price_data
            
            return price_data
            
        except Exception as e:
            logger.error(f"Failed to get price data: {e}")
            return {}
    
    def _convert_to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """将数据转换为DataFrame格式"""
        try:
            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to convert data to DataFrame: {e}")
            return pd.DataFrame()
    
    def _generate_mock_price_data(self, symbol: str) -> pd.DataFrame:
        """生成模拟价格数据"""
        try:
            # 生成100天的模拟数据
            dates = pd.date_range(start=datetime.now() - timedelta(days=100), end=datetime.now(), freq='D')
            
            # 生成价格序列（带趋势和波动）
            base_price = 10.0
            trend = 0.001  # 轻微上升趋势
            volatility = 0.02
            
            prices = []
            current_price = base_price
            
            for i, date in enumerate(dates):
                # 添加趋势和随机波动
                change = trend + np.random.normal(0, volatility)
                current_price *= (1 + change)
                prices.append(current_price)
            
            # 生成OHLC数据
            data = []
            for i, (date, close) in enumerate(zip(dates, prices)):
                high = close * (1 + abs(np.random.normal(0, 0.01)))
                low = close * (1 - abs(np.random.normal(0, 0.01)))
                open_price = prices[i-1] if i > 0 else close
                volume = np.random.randint(1000000, 10000000)
                
                data.append({
                    'date': date,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Failed to generate mock price data: {e}")
            return pd.DataFrame()
    
    def _calculate_technical_indicators(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """计算技术指标"""
        indicators = {}
        
        for symbol, df in price_data.items():
            if df.empty:
                continue
            
            symbol_indicators = {}
            
            # 移动平均线
            symbol_indicators['sma_5'] = df['close'].rolling(window=5).mean()
            symbol_indicators['sma_20'] = df['close'].rolling(window=20).mean()
            symbol_indicators['sma_50'] = df['close'].rolling(window=50).mean()
            
            # RSI
            symbol_indicators['rsi'] = self._calculate_rsi(df['close'])
            
            # MACD
            symbol_indicators['macd'] = self._calculate_macd(df['close'])
            
            # 布林带
            symbol_indicators['bollinger_bands'] = self._calculate_bollinger_bands(df['close'])
            
            # KDJ
            symbol_indicators['kdj'] = self._calculate_kdj(df)
            
            # 成交量指标
            symbol_indicators['volume_ma'] = df['volume'].rolling(window=20).mean()
            symbol_indicators['volume_ratio'] = df['volume'] / symbol_indicators['volume_ma']
            
            indicators[symbol] = symbol_indicators
        
        # 缓存指标
        cache_key = f"indicators_{datetime.now().strftime('%Y%m%d%H')}"
        self.technical_indicators_cache[cache_key] = indicators
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            logger.error(f"Failed to calculate MACD: {e}")
            return {}
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """计算布林带"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return {
                'upper': upper_band,
                'middle': sma,
                'lower': lower_band
            }
        except Exception as e:
            logger.error(f"Failed to calculate Bollinger Bands: {e}")
            return {}
    
    def _calculate_kdj(self, df: pd.DataFrame, k_period: int = 9, d_period: int = 3, j_period: int = 3) -> Dict[str, pd.Series]:
        """计算KDJ指标"""
        try:
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            
            rsv = (df['close'] - low_min) / (high_max - low_min) * 100
            k = rsv.ewm(alpha=1/d_period).mean()
            d = k.ewm(alpha=1/j_period).mean()
            j = 3 * k - 2 * d
            
            return {
                'k': k,
                'd': d,
                'j': j
            }
        except Exception as e:
            logger.error(f"Failed to calculate KDJ: {e}")
            return {}
    
    def _identify_chart_patterns(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """识别图表形态"""
        patterns = {}
        
        for symbol, df in price_data.items():
            if df.empty:
                continue
            
            symbol_patterns = []
            
            # 识别各种图表形态
            if self._is_head_and_shoulders(df):
                symbol_patterns.append("头肩顶/底")
            
            if self._is_double_top_bottom(df):
                symbol_patterns.append("双顶/双底")
            
            if self._is_triangle(df):
                symbol_patterns.append("三角形")
            
            if self._is_wedge(df):
                symbol_patterns.append("楔形")
            
            if self._is_flag_pennant(df):
                symbol_patterns.append("旗形/三角旗")
            
            patterns[symbol] = symbol_patterns
        
        # 缓存形态识别结果
        cache_key = f"patterns_{datetime.now().strftime('%Y%m%d%H')}"
        self.pattern_recognition_cache[cache_key] = patterns
        
        return patterns
    
    def _is_head_and_shoulders(self, df: pd.DataFrame) -> bool:
        """判断是否为头肩形态"""
        try:
            if len(df) < 20:
                return False
            
            # 简化的头肩形态识别
            highs = df['high'].rolling(window=5).max()
            peaks = df[df['high'] == highs]
            
            if len(peaks) >= 3:
                # 检查是否有三个峰值
                peak_values = peaks['high'].values[-3:]
                if len(peak_values) == 3:
                    # 中间峰值应该是最高的
                    if peak_values[1] > peak_values[0] and peak_values[1] > peak_values[2]:
                        return True
            
            return False
        except Exception:
            return False
    
    def _is_double_top_bottom(self, df: pd.DataFrame) -> bool:
        """判断是否为双顶/双底形态"""
        try:
            if len(df) < 20:
                return False
            
            # 简化的双顶/双底识别
            highs = df['high'].rolling(window=5).max()
            lows = df['low'].rolling(window=5).min()
            
            high_peaks = df[df['high'] == highs]
            low_troughs = df[df['low'] == lows]
            
            # 检查双顶
            if len(high_peaks) >= 2:
                peak_values = high_peaks['high'].values[-2:]
                if abs(peak_values[0] - peak_values[1]) / peak_values[0] < 0.03:
                    return True
            
            # 检查双底
            if len(low_troughs) >= 2:
                trough_values = low_troughs['low'].values[-2:]
                if abs(trough_values[0] - trough_values[1]) / trough_values[0] < 0.03:
                    return True
            
            return False
        except Exception:
            return False
    
    def _is_triangle(self, df: pd.DataFrame) -> bool:
        """判断是否为三角形形态"""
        try:
            if len(df) < 20:
                return False
            
            # 检查价格范围是否收敛
            recent_data = df.tail(20)
            price_range = recent_data['high'].max() - recent_data['low'].min()
            early_range = df.iloc[-40:-20]['high'].max() - df.iloc[-40:-20]['low'].min()
            
            # 如果价格范围缩小超过30%，可能是三角形
            if price_range < early_range * 0.7:
                return True
            
            return False
        except Exception:
            return False
    
    def _is_wedge(self, df: pd.DataFrame) -> bool:
        """判断是否为楔形形态"""
        try:
            if len(df) < 20:
                return False
            
            # 简化的楔形识别
            recent_data = df.tail(20)
            
            # 检查趋势线是否收敛
            highs = recent_data['high']
            lows = recent_data['low']
            
            # 计算趋势线斜率
            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
            
            # 如果两条趋势线斜率相反且收敛，可能是楔形
            if (high_slope < 0 and low_slope > 0) or (high_slope > 0 and low_slope < 0):
                return True
            
            return False
        except Exception:
            return False
    
    def _is_flag_pennant(self, df: pd.DataFrame) -> bool:
        """判断是否为旗形/三角旗形态"""
        try:
            if len(df) < 30:
                return False
            
            # 简化的旗形识别
            # 检查是否有明显的趋势后跟随横盘整理
            first_half = df.iloc[-30:-15]
            second_half = df.iloc[-15:]
            
            # 前半部分应该有明显趋势
            first_trend = (first_half['close'].iloc[-1] - first_half['close'].iloc[0]) / first_half['close'].iloc[0]
            
            # 后半部分应该横盘整理
            second_volatility = second_half['close'].std() / second_half['close'].mean()
            
            if abs(first_trend) > 0.05 and second_volatility < 0.02:
                return True
            
            return False
        except Exception:
            return False
    
    def _analyze_support_resistance(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """分析支撑阻力位"""
        support_resistance = {}
        
        for symbol, df in price_data.items():
            if df.empty:
                continue
            
            # 计算支撑位和阻力位
            recent_data = df.tail(50)
            
            # 阻力位：近期高点
            resistance = recent_data['high'].max()
            
            # 支撑位：近期低点
            support = recent_data['low'].min()
            
            # 关键价位
            current_price = df['close'].iloc[-1]
            
            support_resistance[symbol] = {
                'resistance': resistance,
                'support': support,
                'current_price': current_price,
                'resistance_distance': (resistance - current_price) / current_price,
                'support_distance': (current_price - support) / current_price
            }
        
        return support_resistance
    
    def _analyze_trends(self, price_data: Dict[str, pd.DataFrame], indicators: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """分析趋势"""
        trend_analysis = {}
        
        for symbol, df in price_data.items():
            if df.empty or symbol not in indicators:
                continue
            
            symbol_indicators = indicators[symbol]
            
            # 短期趋势（5日均线）
            sma_5 = symbol_indicators.get('sma_5', pd.Series())
            sma_20 = symbol_indicators.get('sma_20', pd.Series())
            sma_50 = symbol_indicators.get('sma_50', pd.Series())
            
            current_price = df['close'].iloc[-1]
            
            # 趋势判断
            short_trend = "上升" if sma_5.iloc[-1] > sma_20.iloc[-1] else "下降"
            medium_trend = "上升" if sma_20.iloc[-1] > sma_50.iloc[-1] else "下降"
            long_trend = "上升" if sma_50.iloc[-1] > sma_50.iloc[-10] else "下降"
            
            # 趋势强度
            trend_strength = abs(sma_5.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]
            
            # 趋势一致性
            trend_consistency = 1.0
            if short_trend != medium_trend:
                trend_consistency -= 0.3
            if medium_trend != long_trend:
                trend_consistency -= 0.3
            
            trend_analysis[symbol] = {
                'short_trend': short_trend,
                'medium_trend': medium_trend,
                'long_trend': long_trend,
                'trend_strength': trend_strength,
                'trend_consistency': trend_consistency,
                'price_vs_ma5': (current_price - sma_5.iloc[-1]) / sma_5.iloc[-1] if not sma_5.empty else 0,
                'price_vs_ma20': (current_price - sma_20.iloc[-1]) / sma_20.iloc[-1] if not sma_20.empty else 0,
                'price_vs_ma50': (current_price - sma_50.iloc[-1]) / sma_50.iloc[-1] if not sma_50.empty else 0
            }
        
        return trend_analysis
    
    def _generate_recommendation(
        self, 
        technical_indicators: Dict[str, Any], 
        chart_patterns: Dict[str, List[str]], 
        support_resistance: Dict[str, Dict[str, float]],
        trend_analysis: Dict[str, Dict[str, Any]]
    ) -> AgentRecommendation:
        """生成技术分析推荐"""
        try:
            total_score = 0
            factor_count = 0
            
            # 分析每个股票的技术指标
            for symbol in technical_indicators.keys():
                symbol_score = 0
                symbol_factors = 0
                
                # RSI分析
                if symbol in technical_indicators and 'rsi' in technical_indicators[symbol]:
                    rsi = technical_indicators[symbol]['rsi']
                    if not rsi.empty:
                        current_rsi = rsi.iloc[-1]
                        if current_rsi < 30:
                            symbol_score += 0.3  # 超卖，买入信号
                        elif current_rsi > 70:
                            symbol_score -= 0.3  # 超买，卖出信号
                        symbol_factors += 1
                
                # MACD分析
                if symbol in technical_indicators and 'macd' in technical_indicators[symbol]:
                    macd_data = technical_indicators[symbol]['macd']
                    if isinstance(macd_data, dict) and 'macd' in macd_data and 'signal' in macd_data:
                        macd_line = macd_data['macd']
                        signal_line = macd_data['signal']
                        if not macd_line.empty and not signal_line.empty:
                            if macd_line.iloc[-1] > signal_line.iloc[-1]:
                                symbol_score += 0.2  # MACD金叉
                            else:
                                symbol_score -= 0.2  # MACD死叉
                            symbol_factors += 1
                
                # 趋势分析
                if symbol in trend_analysis:
                    trend_data = trend_analysis[symbol]
                    if trend_data['trend_consistency'] > 0.7:
                        if trend_data['short_trend'] == "上升":
                            symbol_score += 0.2
                        else:
                            symbol_score -= 0.2
                        symbol_factors += 1
                
                # 支撑阻力分析
                if symbol in support_resistance:
                    sr_data = support_resistance[symbol]
                    if sr_data['support_distance'] < 0.05:  # 接近支撑位
                        symbol_score += 0.1
                    elif sr_data['resistance_distance'] < 0.05:  # 接近阻力位
                        symbol_score -= 0.1
                    symbol_factors += 1
                
                # 图表形态分析
                if symbol in chart_patterns:
                    patterns = chart_patterns[symbol]
                    if "头肩顶/底" in patterns or "双顶/双底" in patterns:
                        symbol_score -= 0.2  # 反转形态
                    elif "三角形" in patterns or "楔形" in patterns:
                        symbol_score += 0.1  # 整理形态
                    symbol_factors += 1
                
                if symbol_factors > 0:
                    total_score += symbol_score / symbol_factors
                    factor_count += 1
            
            # 计算综合评分
            if factor_count > 0:
                avg_score = total_score / factor_count
            else:
                avg_score = 0.0
            
            # 根据评分确定推荐
            if avg_score > 0.3:
                recommendation_type = RecommendationType.BUY
                confidence = min(0.5 + avg_score * 0.5, 0.9)
                reasoning = "技术指标显示买入信号，趋势向上"
            elif avg_score < -0.3:
                recommendation_type = RecommendationType.SELL
                confidence = min(0.5 + abs(avg_score) * 0.5, 0.9)
                reasoning = "技术指标显示卖出信号，趋势向下"
            else:
                recommendation_type = RecommendationType.HOLD
                confidence = 0.6
                reasoning = "技术指标信号不明确，建议观望"
            
            # 计算目标价格
            target_price = None
            stop_loss = None
            take_profit = None
            
            if recommendation_type in [RecommendationType.BUY, RecommendationType.STRONG_BUY]:
                # 基于技术分析计算目标价格
                for symbol, sr_data in support_resistance.items():
                    if sr_data['resistance_distance'] > 0.1:  # 有足够的上涨空间
                        target_price = sr_data['resistance']
                        stop_loss = sr_data['support']
                        take_profit = sr_data['resistance']
                        break
            
            return AgentRecommendation(
                recommendation_type=recommendation_type,
                confidence=confidence,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                supporting_data={
                    'technical_score': avg_score,
                    'indicators_count': factor_count,
                    'chart_patterns': chart_patterns,
                    'trend_analysis': trend_analysis
                },
                risk_level=self._assess_risk_level(avg_score),
                time_horizon="short"
            )
            
        except Exception as e:
            logger.error(f"Failed to generate technical recommendation: {e}")
            return AgentRecommendation(
                recommendation_type=RecommendationType.HOLD,
                confidence=0.3,
                reasoning=f"技术分析推荐生成失败: {e}",
                risk_level="high"
            )
    
    def _assess_risk_level(self, score: float) -> str:
        """评估风险等级"""
        if abs(score) > 0.5:
            return "medium"  # 技术信号明确，风险中等
        elif abs(score) > 0.2:
            return "low"     # 技术信号较明确，风险较低
        else:
            return "high"    # 技术信号不明确，风险较高
    
    def _extract_key_factors(
        self, 
        technical_indicators: Dict[str, Any], 
        chart_patterns: Dict[str, List[str]], 
        trend_analysis: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """提取关键因素"""
        factors = []
        
        # 技术指标因素
        for symbol, indicators in technical_indicators.items():
            if 'rsi' in indicators and not indicators['rsi'].empty:
                rsi = indicators['rsi'].iloc[-1]
                if rsi < 30:
                    factors.append(f"{symbol} RSI超卖")
                elif rsi > 70:
                    factors.append(f"{symbol} RSI超买")
            
            if 'macd' in indicators and isinstance(indicators['macd'], dict):
                macd_data = indicators['macd']
                if 'macd' in macd_data and 'signal' in macd_data:
                    if not macd_data['macd'].empty and not macd_data['signal'].empty:
                        if macd_data['macd'].iloc[-1] > macd_data['signal'].iloc[-1]:
                            factors.append(f"{symbol} MACD金叉")
        
        # 趋势因素
        for symbol, trend_data in trend_analysis.items():
            if trend_data['trend_consistency'] > 0.8:
                factors.append(f"{symbol} 趋势一致性强")
        
        # 图表形态因素
        for symbol, patterns in chart_patterns.items():
            if patterns:
                factors.append(f"{symbol} 出现{patterns[0]}形态")
        
        return factors[:5]
    
    def _extract_risk_factors(
        self, 
        technical_indicators: Dict[str, Any], 
        chart_patterns: Dict[str, List[str]]
    ) -> List[str]:
        """提取风险因素"""
        risks = []
        
        # 技术指标风险
        for symbol, indicators in technical_indicators.items():
            if 'rsi' in indicators and not indicators['rsi'].empty:
                rsi = indicators['rsi'].iloc[-1]
                if rsi > 80:
                    risks.append(f"{symbol} RSI极度超买")
                elif rsi < 20:
                    risks.append(f"{symbol} RSI极度超卖")
        
        # 图表形态风险
        for symbol, patterns in chart_patterns.items():
            if "头肩顶/底" in patterns:
                risks.append(f"{symbol} 出现反转形态")
            if "双顶/双底" in patterns:
                risks.append(f"{symbol} 出现双顶/双底形态")
        
        return risks[:5]
    
    def _generate_market_outlook(
        self, 
        trend_analysis: Dict[str, Dict[str, Any]], 
        technical_indicators: Dict[str, Any]
    ) -> str:
        """生成市场展望"""
        if not trend_analysis:
            return "技术分析数据不足，无法提供明确的市场展望。"
        
        # 统计趋势方向
        up_trends = sum(1 for data in trend_analysis.values() if data['short_trend'] == "上升")
        down_trends = sum(1 for data in trend_analysis.values() if data['short_trend'] == "下降")
        total = len(trend_analysis)
        
        if up_trends > total * 0.6:
            return "技术分析显示多数股票处于上升趋势，市场情绪较为乐观，建议关注技术突破机会。"
        elif down_trends > total * 0.6:
            return "技术分析显示多数股票处于下降趋势，市场情绪较为悲观，建议谨慎操作。"
        else:
            return "技术分析显示市场趋势分化，个股表现差异较大，建议精选个股操作。"
    
    def _create_default_analysis(self, symbols: List[str], error_msg: str) -> AgentAnalysis:
        """创建默认分析结果"""
        return AgentAnalysis(
            agent_name=self.name,
            agent_type=self.agent_type,
            analysis_id=create_analysis_id(),
            timestamp=datetime.now(),
            symbols=symbols,
            recommendation=AgentRecommendation(
                recommendation_type=RecommendationType.HOLD,
                confidence=0.3,
                reasoning=f"技术分析失败: {error_msg}",
                risk_level="high"
            ),
            key_factors=["分析失败"],
            risk_factors=["数据获取失败", "技术指标计算错误"],
            market_outlook="无法提供市场展望",
            additional_insights={'error': error_msg}
        )
    
    def _get_current_technical_signals(self) -> Dict[str, Any]:
        """获取当前技术信号"""
        return {
            'rsi_signal': 'neutral',
            'macd_signal': 'neutral',
            'trend_signal': 'neutral'
        }
    
    def _analyze_price_action(self) -> Dict[str, Any]:
        """分析价格行为"""
        return {
            'price_momentum': 'neutral',
            'volatility': 'normal',
            'volume_trend': 'stable'
        }
    
    def _analyze_volume_patterns(self) -> Dict[str, Any]:
        """分析成交量模式"""
        return {
            'volume_trend': 'normal',
            'volume_price_confirmation': 'neutral',
            'accumulation_distribution': 'balanced'
        }
    
    def _generate_counter_arguments(self, other_analyses: List[AgentAnalysis]) -> List[str]:
        """生成反驳论点"""
        counter_arguments = []
        
        for analysis in other_analyses:
            if analysis.agent_type == 'fundamental_analyst':
                counter_arguments.append("基本面分析可能忽略了短期技术面变化")
            elif analysis.agent_type == 'news_analyst':
                counter_arguments.append("新闻情绪可能滞后于技术面信号")
            elif analysis.agent_type == 'sentiment_analyst':
                counter_arguments.append("市场情绪可能偏离技术面趋势")
        
        return counter_arguments[:3]
    
    def _calculate_debate_confidence(self, other_analyses: List[AgentAnalysis]) -> float:
        """计算辩论后的置信度"""
        base_confidence = self.get_confidence_score()
        
        # 技术分析在短期内有较高置信度
        technical_analyses = [a for a in other_analyses if a.agent_type == 'technical_analyst']
        
        if len(technical_analyses) > 0:
            # 如果有其他技术分析，考虑一致性
            avg_confidence = sum(a.recommendation.confidence for a in technical_analyses) / len(technical_analyses)
            return (base_confidence + avg_confidence) / 2
        else:
            # 如果没有其他技术分析，保持中等置信度
            return base_confidence
