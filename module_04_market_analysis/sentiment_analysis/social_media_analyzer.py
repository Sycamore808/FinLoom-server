"""
社交媒体分析器模块
分析社交媒体平台上的金融相关内容和情感
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import re
import numpy as np
import pandas as pd
from common.logging_system import setup_logger

logger = setup_logger("social_media_analyzer")


@dataclass
class SocialMediaPost:
    """社交媒体帖子"""
    post_id: str
    platform: str  # 'twitter', 'reddit', 'weibo', 'zhihu'
    content: str
    author: str
    timestamp: datetime
    engagement_metrics: Dict[str, int]  # likes, retweets, comments等
    metadata: Dict[str, Any]


@dataclass
class SocialMediaSentiment:
    """社交媒体情感分析结果"""
    platform: str
    overall_sentiment: float  # -1到1之间
    confidence: float
    post_count: int
    engagement_weighted_sentiment: float
    trending_topics: List[str]
    influential_posts: List[Dict[str, Any]]
    timestamp: datetime


class SocialMediaAnalyzer:
    """社交媒体分析器"""
    
    def __init__(self):
        """初始化社交媒体分析器"""
        self.platform_weights = {
            'twitter': 0.3,
            'reddit': 0.25,
            'weibo': 0.2,
            'zhihu': 0.15,
            'other': 0.1
        }
        
        # 金融相关关键词
        self.financial_keywords = [
            '股票', '股市', '投资', '基金', '债券', '期货', '期权',
            'stock', 'market', 'invest', 'trading', 'finance',
            '涨', '跌', '买入', '卖出', '持有', '看涨', '看跌'
        ]
        
        # 情感词典
        self.positive_words = [
            '涨', '上涨', '利好', '看好', '推荐', '买入', '看涨',
            'bullish', 'buy', 'up', 'rise', 'gain', 'positive'
        ]
        
        self.negative_words = [
            '跌', '下跌', '利空', '看空', '卖出', '看跌', '风险',
            'bearish', 'sell', 'down', 'fall', 'loss', 'negative'
        ]
        
        logger.info("SocialMediaAnalyzer initialized")
    
    def analyze_platform_sentiment(
        self, 
        posts: List[SocialMediaPost],
        platform: str
    ) -> SocialMediaSentiment:
        """分析特定平台的情感
        
        Args:
            posts: 帖子列表
            platform: 平台名称
            
        Returns:
            平台情感分析结果
        """
        try:
            if not posts:
                return self._create_default_sentiment(platform, "无帖子数据")
            
            # 过滤金融相关帖子
            financial_posts = self._filter_financial_posts(posts)
            
            if not financial_posts:
                return self._create_default_sentiment(platform, "无金融相关帖子")
            
            # 计算基础情感分数
            sentiment_scores = []
            engagement_weights = []
            
            for post in financial_posts:
                sentiment = self._analyze_post_sentiment(post.content)
                engagement = self._calculate_engagement_score(post.engagement_metrics)
                
                sentiment_scores.append(sentiment)
                engagement_weights.append(engagement)
            
            # 计算加权平均情感
            if engagement_weights:
                weighted_sentiment = np.average(sentiment_scores, weights=engagement_weights)
            else:
                weighted_sentiment = np.mean(sentiment_scores)
            
            # 计算置信度
            confidence = self._calculate_confidence(sentiment_scores, len(financial_posts))
            
            # 提取热门话题
            trending_topics = self._extract_trending_topics(financial_posts)
            
            # 识别有影响力的帖子
            influential_posts = self._identify_influential_posts(financial_posts)
            
            return SocialMediaSentiment(
                platform=platform,
                overall_sentiment=weighted_sentiment,
                confidence=confidence,
                post_count=len(financial_posts),
                engagement_weighted_sentiment=weighted_sentiment,
                trending_topics=trending_topics,
                influential_posts=influential_posts,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Platform sentiment analysis failed for {platform}: {e}")
            return self._create_default_sentiment(platform, f"分析失败: {e}")
    
    def analyze_multi_platform_sentiment(
        self, 
        platform_posts: Dict[str, List[SocialMediaPost]]
    ) -> Dict[str, SocialMediaSentiment]:
        """分析多平台情感
        
        Args:
            platform_posts: 平台到帖子列表的映射
            
        Returns:
            各平台情感分析结果
        """
        results = {}
        
        for platform, posts in platform_posts.items():
            results[platform] = self.analyze_platform_sentiment(posts, platform)
        
        return results
    
    def aggregate_platform_sentiments(
        self, 
        platform_sentiments: Dict[str, SocialMediaSentiment]
    ) -> Dict[str, Any]:
        """聚合多平台情感
        
        Args:
            platform_sentiments: 平台情感结果
            
        Returns:
            聚合后的情感指标
        """
        if not platform_sentiments:
            return {"overall_sentiment": 0.0, "confidence": 0.0}
        
        # 计算加权平均情感
        weighted_sentiment = 0.0
        total_weight = 0.0
        platform_contributions = {}
        
        for platform, sentiment in platform_sentiments.items():
            weight = self.platform_weights.get(platform, 0.1)
            contribution = sentiment.overall_sentiment * sentiment.confidence * weight
            
            weighted_sentiment += contribution
            total_weight += weight * sentiment.confidence
            platform_contributions[platform] = contribution
        
        # 归一化
        if total_weight > 0:
            overall_sentiment = weighted_sentiment / total_weight
        else:
            overall_sentiment = 0.0
        
        # 计算整体置信度
        confidences = [s.confidence for s in platform_sentiments.values()]
        overall_confidence = np.mean(confidences)
        
        # 收集所有热门话题
        all_trending_topics = []
        for sentiment in platform_sentiments.values():
            all_trending_topics.extend(sentiment.trending_topics)
        
        # 去重并排序
        unique_topics = list(set(all_trending_topics))
        
        return {
            "overall_sentiment": overall_sentiment,
            "confidence": overall_confidence,
            "platform_contributions": platform_contributions,
            "trending_topics": unique_topics,
            "platform_count": len(platform_sentiments),
            "total_posts": sum(s.post_count for s in platform_sentiments.values())
        }
    
    def _filter_financial_posts(self, posts: List[SocialMediaPost]) -> List[SocialMediaPost]:
        """过滤金融相关帖子"""
        financial_posts = []
        
        for post in posts:
            content_lower = post.content.lower()
            
            # 检查是否包含金融关键词
            if any(keyword in content_lower for keyword in self.financial_keywords):
                financial_posts.append(post)
        
        return financial_posts
    
    def _analyze_post_sentiment(self, content: str) -> float:
        """分析单个帖子的情感"""
        content_lower = content.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in content_lower)
        negative_count = sum(1 for word in self.negative_words if word in content_lower)
        
        # 计算情感分数
        total_words = len(content.split())
        if total_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / max(total_words, 1)
        
        # 限制在-1到1之间
        return max(-1.0, min(1.0, sentiment_score))
    
    def _calculate_engagement_score(self, metrics: Dict[str, int]) -> float:
        """计算参与度分数"""
        # 不同平台的参与度指标权重
        weights = {
            'likes': 1.0,
            'retweets': 2.0,
            'comments': 1.5,
            'shares': 2.0,
            'replies': 1.5
        }
        
        total_score = 0.0
        for metric, count in metrics.items():
            weight = weights.get(metric, 1.0)
            total_score += count * weight
        
        # 使用对数缩放避免极端值
        return np.log1p(total_score)
    
    def _calculate_confidence(
        self, 
        sentiment_scores: List[float], 
        post_count: int
    ) -> float:
        """计算置信度"""
        if not sentiment_scores:
            return 0.0
        
        # 基于样本数量和一致性
        sample_confidence = min(post_count / 100.0, 1.0)  # 最多100个帖子达到最大置信度
        
        # 基于情感分数的一致性
        if len(sentiment_scores) > 1:
            consistency = 1.0 - np.std(sentiment_scores)
            consistency = max(0.0, consistency)
        else:
            consistency = 0.5
        
        return (sample_confidence + consistency) / 2.0
    
    def _extract_trending_topics(self, posts: List[SocialMediaPost]) -> List[str]:
        """提取热门话题"""
        # 简单的关键词频率统计
        word_freq = {}
        
        for post in posts:
            # 提取可能的股票代码和关键词
            words = re.findall(r'\b[A-Z]{2,5}\b|\b[0-9]{6}\b', post.content)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 返回频率最高的前10个
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
    
    def _identify_influential_posts(self, posts: List[SocialMediaPost]) -> List[Dict[str, Any]]:
        """识别有影响力的帖子"""
        influential_posts = []
        
        for post in posts:
            engagement_score = self._calculate_engagement_score(post.engagement_metrics)
            
            # 如果参与度分数超过阈值，认为是有影响力的帖子
            if engagement_score > 5.0:  # 可调整的阈值
                influential_posts.append({
                    "post_id": post.post_id,
                    "content": post.content[:100] + "..." if len(post.content) > 100 else post.content,
                    "engagement_score": engagement_score,
                    "platform": post.platform,
                    "timestamp": post.timestamp.isoformat()
                })
        
        # 按参与度分数排序
        influential_posts.sort(key=lambda x: x["engagement_score"], reverse=True)
        return influential_posts[:5]  # 返回前5个
    
    def _create_default_sentiment(self, platform: str, reason: str) -> SocialMediaSentiment:
        """创建默认情感结果"""
        return SocialMediaSentiment(
            platform=platform,
            overall_sentiment=0.0,
            confidence=0.0,
            post_count=0,
            engagement_weighted_sentiment=0.0,
            trending_topics=[],
            influential_posts=[],
            timestamp=datetime.now()
        )


# 便捷函数
def analyze_social_media_sentiment(
    posts_data: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """分析社交媒体情感的便捷函数
    
    Args:
        posts_data: 帖子数据，格式为 {platform: [post_dict, ...]}
        
    Returns:
        社交媒体情感分析结果
    """
    analyzer = SocialMediaAnalyzer()
    
    # 转换数据格式
    platform_posts = {}
    for platform, posts in posts_data.items():
        social_posts = []
        for post_dict in posts:
            post = SocialMediaPost(
                post_id=post_dict.get("id", ""),
                platform=platform,
                content=post_dict.get("content", ""),
                author=post_dict.get("author", ""),
                timestamp=post_dict.get("timestamp", datetime.now()),
                engagement_metrics=post_dict.get("engagement", {}),
                metadata=post_dict
            )
            social_posts.append(post)
        platform_posts[platform] = social_posts
    
    # 分析各平台情感
    platform_sentiments = analyzer.analyze_multi_platform_sentiment(platform_posts)
    
    # 聚合结果
    aggregated = analyzer.aggregate_platform_sentiments(platform_sentiments)
    
    return {
        "platform_sentiments": {
            platform: {
                "overall_sentiment": sentiment.overall_sentiment,
                "confidence": sentiment.confidence,
                "post_count": sentiment.post_count,
                "trending_topics": sentiment.trending_topics
            }
            for platform, sentiment in platform_sentiments.items()
        },
        "aggregated": aggregated
    }