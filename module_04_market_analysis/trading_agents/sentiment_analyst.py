"""
情绪分析师
分析市场情绪、投资者情绪和社交媒体情绪
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .base_agent import BaseAgent, AgentAnalysis, AgentRecommendation, RecommendationType, DebateContext, DebateResponse, create_analysis_id
from common.logging_system import setup_logger

logger = setup_logger("sentiment_analyst")


class SentimentAnalyst(BaseAgent):
    """情绪分析师"""
    
    def __init__(self):
        super().__init__(
            name="情绪分析师",
            agent_type="sentiment_analyst",
            expertise="市场情绪分析, 投资者情绪分析, 社交媒体情绪分析, 情绪指标计算",
            confidence_threshold=0.6
        )
        self.sentiment_data_cache = {}
        self.social_media_cache = {}
        self.fear_greed_cache = {}
    
    async def analyze(
        self, 
        symbols: List[str], 
        market_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """分析市场情绪"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting sentiment analysis for symbols: {symbols}")
            
            # 1. 获取市场情绪数据
            market_sentiment = await self._get_market_sentiment(symbols)
            
            # 2. 获取社交媒体情绪
            social_sentiment = await self._get_social_media_sentiment(symbols)
            
            # 3. 计算恐惧贪婪指数
            fear_greed_index = await self._calculate_fear_greed_index(symbols)
            
            # 4. 分析投资者情绪
            investor_sentiment = await self._analyze_investor_sentiment(symbols)
            
            # 5. 计算情绪指标
            sentiment_indicators = self._calculate_sentiment_indicators(
                market_sentiment, social_sentiment, fear_greed_index, investor_sentiment
            )
            
            # 6. 生成推荐
            recommendation = self._generate_recommendation(sentiment_indicators, symbols)
            
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
                key_factors=self._extract_key_factors(sentiment_indicators),
                risk_factors=self._extract_risk_factors(sentiment_indicators),
                market_outlook=self._generate_market_outlook(sentiment_indicators),
                additional_insights={
                    'market_sentiment': market_sentiment,
                    'social_sentiment': social_sentiment,
                    'fear_greed_index': fear_greed_index,
                    'investor_sentiment': investor_sentiment,
                    'sentiment_indicators': sentiment_indicators,
                    'sentiment_score': sentiment_indicators.get('overall_sentiment', 0.0),
                    'sentiment_trend': sentiment_indicators.get('sentiment_trend', 'neutral')
                },
                data_sources=['市场情绪数据', '社交媒体数据', '投资者情绪数据', '恐惧贪婪指数'],
                analysis_duration=analysis_duration
            )
            
            logger.info(f"Sentiment analysis completed for {symbols} in {analysis_duration:.2f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbols}: {e}")
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
            
            # 基于情绪分析生成辩论响应
            current_sentiment = self._get_current_sentiment()
            
            if current_sentiment > 0.3:
                position = "支持"
                arguments = [
                    "市场情绪积极，投资者信心增强",
                    "社交媒体情绪偏向乐观",
                    "恐惧贪婪指数显示贪婪情绪"
                ]
            elif current_sentiment < -0.3:
                position = "反对"
                arguments = [
                    "市场情绪悲观，投资者信心不足",
                    "社交媒体情绪偏向悲观",
                    "恐惧贪婪指数显示恐惧情绪"
                ]
            else:
                position = "中立"
                arguments = [
                    "市场情绪中性，投资者观望情绪浓厚",
                    "社交媒体情绪分化",
                    "恐惧贪婪指数处于中性区间"
                ]
            
            return DebateResponse(
                agent_name=self.name,
                position=position,
                arguments=arguments,
                evidence={
                    'market_sentiment': current_sentiment,
                    'fear_greed_index': self._get_fear_greed_index(),
                    'social_media_sentiment': self._get_social_media_sentiment_summary()
                },
                counter_arguments=self._generate_counter_arguments(other_analyses),
                updated_confidence=self._calculate_debate_confidence(other_analyses),
                willingness_to_compromise=0.7  # 情绪分析相对灵活
            )
            
        except Exception as e:
            logger.error(f"Debate failed for {self.name}: {e}")
            return DebateResponse(
                agent_name=self.name,
                position="中立",
                arguments=["情绪分析过程中出现错误"],
                evidence={},
                updated_confidence=0.3
            )
    
    async def _get_market_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """获取市场情绪数据"""
        try:
            # 模拟市场情绪数据
            market_sentiment = {
                'overall_sentiment': np.random.uniform(-0.5, 0.5),
                'bullish_percentage': np.random.uniform(30, 70),
                'bearish_percentage': np.random.uniform(20, 50),
                'neutral_percentage': np.random.uniform(10, 30),
                'sentiment_momentum': np.random.uniform(-0.2, 0.2),
                'volatility_sentiment': np.random.uniform(-0.3, 0.3),
                'volume_sentiment': np.random.uniform(-0.2, 0.2),
                'price_sentiment': np.random.uniform(-0.4, 0.4),
                'market_breadth': np.random.uniform(0.3, 0.8),
                'advance_decline_ratio': np.random.uniform(0.5, 2.0)
            }
            
            # 缓存数据
            cache_key = f"market_sentiment_{','.join(sorted(symbols))}_{datetime.now().strftime('%Y%m%d%H')}"
            self.sentiment_data_cache[cache_key] = market_sentiment
            
            return market_sentiment
            
        except Exception as e:
            logger.error(f"Failed to get market sentiment: {e}")
            return {}
    
    async def _get_social_media_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """获取社交媒体情绪"""
        try:
            # 模拟社交媒体情绪数据
            social_sentiment = {
                'twitter_sentiment': np.random.uniform(-0.4, 0.4),
                'weibo_sentiment': np.random.uniform(-0.3, 0.3),
                'stock_forum_sentiment': np.random.uniform(-0.5, 0.5),
                'reddit_sentiment': np.random.uniform(-0.3, 0.3),
                'overall_social_sentiment': np.random.uniform(-0.4, 0.4),
                'mention_volume': np.random.randint(1000, 10000),
                'sentiment_velocity': np.random.uniform(-0.2, 0.2),
                'influencer_sentiment': np.random.uniform(-0.3, 0.3),
                'retail_sentiment': np.random.uniform(-0.4, 0.4),
                'institutional_sentiment': np.random.uniform(-0.2, 0.2)
            }
            
            # 缓存数据
            cache_key = f"social_sentiment_{','.join(sorted(symbols))}_{datetime.now().strftime('%Y%m%d%H')}"
            self.social_media_cache[cache_key] = social_sentiment
            
            return social_sentiment
            
        except Exception as e:
            logger.error(f"Failed to get social media sentiment: {e}")
            return {}
    
    async def _calculate_fear_greed_index(self, symbols: List[str]) -> Dict[str, Any]:
        """计算恐惧贪婪指数"""
        try:
            # 模拟恐惧贪婪指数计算
            fear_greed_index = {
                'current_index': np.random.randint(20, 80),
                'previous_index': np.random.randint(20, 80),
                'change': np.random.uniform(-10, 10),
                'trend': np.random.choice(['increasing', 'decreasing', 'stable']),
                'components': {
                    'put_call_ratio': np.random.uniform(0.5, 2.0),
                    'junk_bond_demand': np.random.uniform(0.3, 0.8),
                    'market_volatility': np.random.uniform(0.1, 0.4),
                    'safe_haven_demand': np.random.uniform(0.2, 0.7),
                    'market_momentum': np.random.uniform(-0.3, 0.3)
                },
                'interpretation': self._interpret_fear_greed_index(np.random.randint(20, 80))
            }
            
            # 缓存数据
            cache_key = f"fear_greed_{datetime.now().strftime('%Y%m%d%H')}"
            self.fear_greed_cache[cache_key] = fear_greed_index
            
            return fear_greed_index
            
        except Exception as e:
            logger.error(f"Failed to calculate fear greed index: {e}")
            return {}
    
    async def _analyze_investor_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """分析投资者情绪"""
        try:
            # 模拟投资者情绪分析
            investor_sentiment = {
                'retail_sentiment': np.random.uniform(-0.4, 0.4),
                'institutional_sentiment': np.random.uniform(-0.3, 0.3),
                'hedge_fund_sentiment': np.random.uniform(-0.3, 0.3),
                'mutual_fund_sentiment': np.random.uniform(-0.2, 0.2),
                'pension_fund_sentiment': np.random.uniform(-0.2, 0.2),
                'foreign_investor_sentiment': np.random.uniform(-0.3, 0.3),
                'margin_debt_sentiment': np.random.uniform(-0.2, 0.2),
                'short_interest_sentiment': np.random.uniform(-0.3, 0.3),
                'options_sentiment': np.random.uniform(-0.4, 0.4),
                'overall_investor_sentiment': np.random.uniform(-0.3, 0.3)
            }
            
            return investor_sentiment
            
        except Exception as e:
            logger.error(f"Failed to analyze investor sentiment: {e}")
            return {}
    
    def _calculate_sentiment_indicators(
        self, 
        market_sentiment: Dict[str, Any], 
        social_sentiment: Dict[str, Any], 
        fear_greed_index: Dict[str, Any],
        investor_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算情绪指标"""
        try:
            # 综合情绪分数
            overall_sentiment = 0.0
            sentiment_components = []
            
            # 市场情绪权重
            if market_sentiment:
                market_score = market_sentiment.get('overall_sentiment', 0.0)
                overall_sentiment += market_score * 0.3
                sentiment_components.append(('market', market_score, 0.3))
            
            # 社交媒体情绪权重
            if social_sentiment:
                social_score = social_sentiment.get('overall_social_sentiment', 0.0)
                overall_sentiment += social_score * 0.25
                sentiment_components.append(('social', social_score, 0.25))
            
            # 恐惧贪婪指数权重
            if fear_greed_index:
                fg_index = fear_greed_index.get('current_index', 50)
                fg_score = (fg_index - 50) / 50  # 转换为-1到1的分数
                overall_sentiment += fg_score * 0.25
                sentiment_components.append(('fear_greed', fg_score, 0.25))
            
            # 投资者情绪权重
            if investor_sentiment:
                investor_score = investor_sentiment.get('overall_investor_sentiment', 0.0)
                overall_sentiment += investor_score * 0.2
                sentiment_components.append(('investor', investor_score, 0.2))
            
            # 计算情绪趋势
            sentiment_trend = self._calculate_sentiment_trend(overall_sentiment)
            
            # 计算情绪强度
            sentiment_strength = abs(overall_sentiment)
            
            # 计算情绪一致性
            sentiment_consistency = self._calculate_sentiment_consistency(sentiment_components)
            
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_trend': sentiment_trend,
                'sentiment_strength': sentiment_strength,
                'sentiment_consistency': sentiment_consistency,
                'sentiment_components': sentiment_components,
                'market_sentiment': market_sentiment.get('overall_sentiment', 0.0),
                'social_sentiment': social_sentiment.get('overall_social_sentiment', 0.0),
                'fear_greed_score': (fear_greed_index.get('current_index', 50) - 50) / 50,
                'investor_sentiment': investor_sentiment.get('overall_investor_sentiment', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate sentiment indicators: {e}")
            return {
                'overall_sentiment': 0.0,
                'sentiment_trend': 'neutral',
                'sentiment_strength': 0.0,
                'sentiment_consistency': 0.0,
                'sentiment_components': [],
                'error': str(e)
            }
    
    def _calculate_sentiment_trend(self, current_sentiment: float) -> str:
        """计算情绪趋势"""
        if current_sentiment > 0.2:
            return "bullish"
        elif current_sentiment < -0.2:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_sentiment_consistency(self, sentiment_components: List[Tuple[str, float, float]]) -> float:
        """计算情绪一致性"""
        if not sentiment_components:
            return 0.0
        
        # 计算各组件情绪方向的一致性
        positive_count = sum(1 for _, score, _ in sentiment_components if score > 0.1)
        negative_count = sum(1 for _, score, _ in sentiment_components if score < -0.1)
        total_count = len(sentiment_components)
        
        # 一致性 = 最大方向占比
        max_direction = max(positive_count, negative_count)
        consistency = max_direction / total_count
        
        return consistency
    
    def _interpret_fear_greed_index(self, index: int) -> str:
        """解释恐惧贪婪指数"""
        if index <= 20:
            return "极度恐惧"
        elif index <= 40:
            return "恐惧"
        elif index <= 60:
            return "中性"
        elif index <= 80:
            return "贪婪"
        else:
            return "极度贪婪"
    
    def _generate_recommendation(
        self, 
        sentiment_indicators: Dict[str, Any], 
        symbols: List[str]
    ) -> AgentRecommendation:
        """生成情绪分析推荐"""
        try:
            overall_sentiment = sentiment_indicators.get('overall_sentiment', 0.0)
            sentiment_strength = sentiment_indicators.get('sentiment_strength', 0.0)
            sentiment_consistency = sentiment_indicators.get('sentiment_consistency', 0.0)
            
            # 计算综合置信度
            confidence = (sentiment_strength * 0.4 + sentiment_consistency * 0.6)
            confidence = min(max(confidence, 0.3), 0.9)
            
            # 根据情绪确定推荐
            if overall_sentiment > 0.3 and sentiment_consistency > 0.6:
                recommendation_type = RecommendationType.BUY
                reasoning = "市场情绪积极，投资者信心增强，情绪一致性高"
            elif overall_sentiment < -0.3 and sentiment_consistency > 0.6:
                recommendation_type = RecommendationType.SELL
                reasoning = "市场情绪悲观，投资者信心不足，情绪一致性高"
            elif overall_sentiment > 0.1:
                recommendation_type = RecommendationType.BUY
                reasoning = "市场情绪偏向积极，但一致性有待提高"
            elif overall_sentiment < -0.1:
                recommendation_type = RecommendationType.SELL
                reasoning = "市场情绪偏向悲观，但一致性有待提高"
            else:
                recommendation_type = RecommendationType.HOLD
                reasoning = "市场情绪中性，投资者观望情绪浓厚"
            
            # 计算目标价格（简化实现）
            target_price = None
            stop_loss = None
            take_profit = None
            
            if recommendation_type in [RecommendationType.BUY, RecommendationType.STRONG_BUY]:
                # 基于情绪强度计算目标价格
                base_price = 12.0  # 示例基础价格
                sentiment_multiplier = 1 + overall_sentiment * 0.1
                target_price = base_price * sentiment_multiplier
                stop_loss = target_price * 0.9
                take_profit = target_price * 1.15
            
            return AgentRecommendation(
                recommendation_type=recommendation_type,
                confidence=confidence,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                supporting_data={
                    'overall_sentiment': overall_sentiment,
                    'sentiment_strength': sentiment_strength,
                    'sentiment_consistency': sentiment_consistency,
                    'sentiment_components': sentiment_indicators.get('sentiment_components', [])
                },
                risk_level=self._assess_risk_level(sentiment_indicators),
                time_horizon="short"
            )
            
        except Exception as e:
            logger.error(f"Failed to generate sentiment recommendation: {e}")
            return AgentRecommendation(
                recommendation_type=RecommendationType.HOLD,
                confidence=0.3,
                reasoning=f"情绪分析推荐生成失败: {e}",
                risk_level="high"
            )
    
    def _assess_risk_level(self, sentiment_indicators: Dict[str, Any]) -> str:
        """评估风险等级"""
        sentiment_consistency = sentiment_indicators.get('sentiment_consistency', 0.0)
        sentiment_strength = sentiment_indicators.get('sentiment_strength', 0.0)
        
        # 情绪不一致或极端情绪都增加风险
        if sentiment_consistency < 0.5:
            return "high"  # 情绪不一致，风险高
        elif sentiment_strength > 0.8:
            return "high"  # 极端情绪，风险高
        elif sentiment_consistency > 0.7 and sentiment_strength < 0.6:
            return "low"   # 情绪一致且温和，风险低
        else:
            return "medium"  # 其他情况，风险中等
    
    def _extract_key_factors(self, sentiment_indicators: Dict[str, Any]) -> List[str]:
        """提取关键因素"""
        factors = []
        
        overall_sentiment = sentiment_indicators.get('overall_sentiment', 0.0)
        sentiment_trend = sentiment_indicators.get('sentiment_trend', 'neutral')
        sentiment_consistency = sentiment_indicators.get('sentiment_consistency', 0.0)
        
        # 情绪方向因素
        if overall_sentiment > 0.3:
            factors.append("市场情绪积极")
        elif overall_sentiment < -0.3:
            factors.append("市场情绪悲观")
        
        # 情绪趋势因素
        if sentiment_trend == "bullish":
            factors.append("情绪趋势向上")
        elif sentiment_trend == "bearish":
            factors.append("情绪趋势向下")
        
        # 一致性因素
        if sentiment_consistency > 0.7:
            factors.append("情绪一致性高")
        elif sentiment_consistency < 0.4:
            factors.append("情绪分化严重")
        
        # 恐惧贪婪指数因素
        fear_greed_score = sentiment_indicators.get('fear_greed_score', 0.0)
        if fear_greed_score > 0.3:
            factors.append("恐惧贪婪指数显示贪婪")
        elif fear_greed_score < -0.3:
            factors.append("恐惧贪婪指数显示恐惧")
        
        return factors[:5]
    
    def _extract_risk_factors(self, sentiment_indicators: Dict[str, Any]) -> List[str]:
        """提取风险因素"""
        risks = []
        
        sentiment_consistency = sentiment_indicators.get('sentiment_consistency', 0.0)
        sentiment_strength = sentiment_indicators.get('sentiment_strength', 0.0)
        
        # 情绪一致性风险
        if sentiment_consistency < 0.5:
            risks.append("情绪分化严重，市场不确定性高")
        
        # 极端情绪风险
        if sentiment_strength > 0.8:
            risks.append("情绪过于极端，可能存在反转风险")
        
        # 社交媒体风险
        social_sentiment = sentiment_indicators.get('social_sentiment', 0.0)
        if abs(social_sentiment) > 0.5:
            risks.append("社交媒体情绪波动较大")
        
        # 投资者情绪风险
        investor_sentiment = sentiment_indicators.get('investor_sentiment', 0.0)
        if abs(investor_sentiment) > 0.4:
            risks.append("投资者情绪波动较大")
        
        return risks[:5]
    
    def _generate_market_outlook(self, sentiment_indicators: Dict[str, Any]) -> str:
        """生成市场展望"""
        overall_sentiment = sentiment_indicators.get('overall_sentiment', 0.0)
        sentiment_trend = sentiment_indicators.get('sentiment_trend', 'neutral')
        sentiment_consistency = sentiment_indicators.get('sentiment_consistency', 0.0)
        
        if overall_sentiment > 0.3 and sentiment_consistency > 0.6:
            return "市场情绪积极，投资者信心增强，情绪一致性高，有利于市场上涨。"
        elif overall_sentiment < -0.3 and sentiment_consistency > 0.6:
            return "市场情绪悲观，投资者信心不足，情绪一致性高，市场可能面临压力。"
        elif sentiment_consistency < 0.5:
            return "市场情绪分化严重，投资者观点不一致，市场不确定性较高，建议谨慎操作。"
        else:
            return "市场情绪相对中性，投资者观望情绪浓厚，建议关注情绪变化趋势。"
    
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
                reasoning=f"情绪分析失败: {error_msg}",
                risk_level="high"
            ),
            key_factors=["分析失败"],
            risk_factors=["数据获取失败", "情绪分析错误"],
            market_outlook="无法提供市场展望",
            additional_insights={'error': error_msg}
        )
    
    def _get_current_sentiment(self) -> float:
        """获取当前情绪"""
        # 从缓存中获取最新的情绪数据
        if self.sentiment_data_cache:
            latest_data = max(self.sentiment_data_cache.values(), key=lambda x: x.get('timestamp', datetime.min))
            return latest_data.get('overall_sentiment', 0.0)
        return 0.0
    
    def _get_fear_greed_index(self) -> Dict[str, Any]:
        """获取恐惧贪婪指数"""
        if self.fear_greed_cache:
            latest_data = max(self.fear_greed_cache.values(), key=lambda x: x.get('timestamp', datetime.min))
            return latest_data
        return {}
    
    def _get_social_media_sentiment_summary(self) -> Dict[str, Any]:
        """获取社交媒体情绪摘要"""
        if self.social_media_cache:
            latest_data = max(self.social_media_cache.values(), key=lambda x: x.get('timestamp', datetime.min))
            return {
                'overall_sentiment': latest_data.get('overall_social_sentiment', 0.0),
                'mention_volume': latest_data.get('mention_volume', 0)
            }
        return {}
    
    def _generate_counter_arguments(self, other_analyses: List[AgentAnalysis]) -> List[str]:
        """生成反驳论点"""
        counter_arguments = []
        
        for analysis in other_analyses:
            if analysis.agent_type == 'fundamental_analyst':
                counter_arguments.append("基本面分析可能忽略了市场情绪变化")
            elif analysis.agent_type == 'technical_analyst':
                counter_arguments.append("技术分析可能滞后于情绪面信号")
            elif analysis.agent_type == 'news_analyst':
                counter_arguments.append("新闻分析可能过于关注短期情绪")
        
        return counter_arguments[:3]
    
    def _calculate_debate_confidence(self, other_analyses: List[AgentAnalysis]) -> float:
        """计算辩论后的置信度"""
        base_confidence = self.get_confidence_score()
        
        # 情绪分析在短期内有较高置信度
        sentiment_analyses = [a for a in other_analyses if a.agent_type == 'sentiment_analyst']
        
        if len(sentiment_analyses) > 0:
            # 如果有其他情绪分析，考虑一致性
            avg_confidence = sum(a.recommendation.confidence for a in sentiment_analyses) / len(sentiment_analyses)
            return (base_confidence + avg_confidence) / 2
        else:
            # 如果没有其他情绪分析，保持中等置信度
            return base_confidence
