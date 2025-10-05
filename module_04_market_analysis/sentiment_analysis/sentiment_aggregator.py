"""
情感聚合器模块
聚合多种情感分析结果，提供综合情感指标
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from common.logging_system import setup_logger

logger = setup_logger("sentiment_aggregator")


@dataclass
class SentimentSource:
    """情感数据源"""
    source_name: str
    sentiment_score: float  # -1到1之间
    confidence: float  # 0到1之间
    weight: float  # 权重
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class AggregatedSentiment:
    """聚合情感结果"""
    overall_sentiment: float
    confidence: float
    sentiment_trend: str  # 'positive', 'negative', 'neutral', 'mixed'
    volatility: float
    source_contributions: Dict[str, float]
    key_insights: List[str]
    risk_factors: List[str]
    timestamp: datetime


class SentimentAggregator:
    """情感聚合器"""
    
    def __init__(self, default_weights: Optional[Dict[str, float]] = None):
        """初始化情感聚合器"""
        self.default_weights = default_weights or {
            'news_sentiment': 0.3,
            'social_media': 0.25,
            'analyst_reports': 0.2,
            'market_indicators': 0.15,
            'fear_greed_index': 0.1
        }
        
        self.sentiment_history: List[AggregatedSentiment] = []
        self.source_history: Dict[str, List[SentimentSource]] = {}
        
        logger.info("SentimentAggregator initialized")
    
    def aggregate_sentiments(
        self, 
        sentiment_sources: List[SentimentSource],
        custom_weights: Optional[Dict[str, float]] = None
    ) -> AggregatedSentiment:
        """聚合多个情感源"""
        try:
            if not sentiment_sources:
                return self._create_default_sentiment("无情感数据源")
            
            weights = custom_weights or self.default_weights
            
            # 计算加权平均情感分数
            weighted_sentiment = 0.0
            total_weight = 0.0
            source_contributions = {}
            
            for source in sentiment_sources:
                weight = weights.get(source.source_name, 0.1)
                contribution = source.sentiment_score * source.confidence * weight
                
                weighted_sentiment += contribution
                total_weight += weight * source.confidence
                source_contributions[source.source_name] = contribution
            
            # 归一化
            if total_weight > 0:
                overall_sentiment = weighted_sentiment / total_weight
            else:
                overall_sentiment = 0.0
            
            # 计算整体置信度
            confidence = self._calculate_aggregate_confidence(sentiment_sources, weights)
            
            # 计算情感趋势
            sentiment_trend = self._calculate_sentiment_trend(overall_sentiment)
            
            # 计算波动率
            volatility = self._calculate_sentiment_volatility(sentiment_sources)
            
            # 提取关键洞察
            key_insights = self._extract_key_insights(sentiment_sources, overall_sentiment)
            
            # 识别风险因素
            risk_factors = self._identify_risk_factors(sentiment_sources, overall_sentiment)
            
            # 创建聚合结果
            aggregated = AggregatedSentiment(
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                sentiment_trend=sentiment_trend,
                volatility=volatility,
                source_contributions=source_contributions,
                key_insights=key_insights,
                risk_factors=risk_factors,
                timestamp=datetime.now()
            )
            
            # 记录历史
            self.sentiment_history.append(aggregated)
            for source in sentiment_sources:
                if source.source_name not in self.source_history:
                    self.source_history[source.source_name] = []
                self.source_history[source.source_name].append(source)
            
            logger.info(f"Aggregated sentiment: {overall_sentiment:.3f} (confidence: {confidence:.3f})")
            return aggregated
            
        except Exception as e:
            logger.error(f"Sentiment aggregation failed: {e}")
            return self._create_default_sentiment(f"聚合失败: {e}")
    
    def _calculate_aggregate_confidence(
        self, 
        sources: List[SentimentSource], 
        weights: Dict[str, float]
    ) -> float:
        """计算聚合置信度"""
        if not sources:
            return 0.0
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for source in sources:
            weight = weights.get(source.source_name, 0.1)
            weighted_confidence += source.confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _calculate_sentiment_trend(self, sentiment: float) -> str:
        """计算情感趋势"""
        if sentiment > 0.3:
            return "positive"
        elif sentiment < -0.3:
            return "negative"
        elif abs(sentiment) < 0.1:
            return "neutral"
        else:
            return "mixed"
    
    def _calculate_sentiment_volatility(self, sources: List[SentimentSource]) -> float:
        """计算情感波动率"""
        if len(sources) < 2:
            return 0.0
        
        sentiments = [s.sentiment_score for s in sources]
        return np.std(sentiments)
    
    def _extract_key_insights(
        self, 
        sources: List[SentimentSource], 
        overall_sentiment: float
    ) -> List[str]:
        """提取关键洞察"""
        insights = []
        
        if abs(overall_sentiment) > 0.5:
            insights.append(f"强烈{'积极' if overall_sentiment > 0 else '消极'}情感信号")
        
        sentiments = [s.sentiment_score for s in sources]
        if len(sentiments) > 1:
            sentiment_std = np.std(sentiments)
            if sentiment_std < 0.2:
                insights.append("各数据源情感高度一致")
            elif sentiment_std > 0.5:
                insights.append("各数据源情感存在较大分歧")
        
        high_confidence_sources = [s for s in sources if s.confidence > 0.8]
        if high_confidence_sources:
            insights.append(f"{len(high_confidence_sources)}个高置信度数据源")
        
        return insights
    
    def _identify_risk_factors(
        self, 
        sources: List[SentimentSource], 
        overall_sentiment: float
    ) -> List[str]:
        """识别风险因素"""
        risks = []
        
        low_confidence_sources = [s for s in sources if s.confidence < 0.5]
        if low_confidence_sources:
            risks.append(f"{len(low_confidence_sources)}个低置信度数据源")
        
        volatility = self._calculate_sentiment_volatility(sources)
        if volatility > 0.4:
            risks.append("情感波动率过高")
        
        if abs(overall_sentiment) > 0.8:
            risks.append("极端情感状态")
        
        if len(sources) < 3:
            risks.append("数据源数量不足")
        
        return risks
    
    def _create_default_sentiment(self, reason: str) -> AggregatedSentiment:
        """创建默认情感结果"""
        return AggregatedSentiment(
            overall_sentiment=0.0,
            confidence=0.0,
            sentiment_trend="neutral",
            volatility=0.0,
            source_contributions={},
            key_insights=[reason],
            risk_factors=["数据不足"],
            timestamp=datetime.now()
        )