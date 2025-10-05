"""
FIN-R1 情感分析模块 - 集成Trading Agents功能
使用本地FIN-R1模型进行金融文本情感分析，集成情绪分析师智能体功能
集成Module 1的真实数据源
"""

import asyncio
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Module 1 data integration
try:
    from module_01_data_pipeline import (
        AkshareDataCollector,
        ChineseAlternativeDataCollector,
        get_database_manager,
    )
except ImportError:
    AkshareDataCollector = None
    ChineseAlternativeDataCollector = None
    get_database_manager = None

from common.logging_system import setup_logger

from ..storage_management.market_analysis_database import get_market_analysis_db

# 添加FIN-R1模型路径
FIN_R1_PATH = ".Fin-R1"
if os.path.exists(FIN_R1_PATH):
    sys.path.append(FIN_R1_PATH)

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    torch = None

logger = setup_logger("fin_r1_sentiment")


@dataclass
class SentimentResult:
    """情感分析结果"""

    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    probability_scores: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class BatchSentimentResult:
    """批量情感分析结果"""

    results: List[SentimentResult]
    overall_sentiment: str
    average_confidence: float
    sentiment_distribution: Dict[str, float]
    processing_time: float


class FINR1SentimentAnalyzer:
    """FIN-R1情感分析器 - 集成Trading Agents功能"""

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        初始化FIN-R1情感分析器

        Args:
            model_path: 模型路径，默认使用本地路径
            device: 计算设备 ('cpu', 'cuda', 'auto')
        """
        self.model_path = model_path or FIN_R1_PATH
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        self.max_length = 512

        # Trading Agents 集成
        self.data_collector = (
            ChineseAlternativeDataCollector()
            if ChineseAlternativeDataCollector
            else None
        )
        self.db_manager = get_market_analysis_db() if get_market_analysis_db else None
        self.sentiment_cache = {}

        # 情感标签映射
        self.label_mapping = {0: "negative", 1: "neutral", 2: "positive"}

        self._load_model()
        logger.info(
            "Initialized FINR1SentimentAnalyzer with Trading Agents integration"
        )

    def _get_device(self, device: str) -> str:
        """获取计算设备"""
        if device == "auto":
            if torch and torch.cuda.is_available():
                return "cuda"
            elif torch and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_model(self) -> None:
        """加载FIN-R1模型"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"FIN-R1 model path not found: {self.model_path}")
                logger.info("Using fallback sentiment analysis method")
                return

            if AutoTokenizer is None or AutoModelForSequenceClassification is None:
                logger.error("transformers library not available")
                logger.info("Using fallback sentiment analysis method")
                return

            logger.info(f"Loading FIN-R1 model from {self.model_path}")

            # 加载分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            )

            # 移动到指定设备
            if torch:
                self.model.to(self.device)
                self.model.eval()

            logger.info(f"FIN-R1 model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load FIN-R1 model: {str(e)}")
            logger.info("Using fallback sentiment analysis method")
            self.tokenizer = None
            self.model = None

    def analyze_sentiment(self, text: str, **kwargs) -> SentimentResult:
        """
        分析单个文本的情感

        Args:
            text: 待分析文本
            **kwargs: 额外参数

        Returns:
            情感分析结果
        """
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                sentiment="neutral",
                confidence=0.0,
                probability_scores={
                    "positive": 0.33,
                    "neutral": 0.34,
                    "negative": 0.33,
                },
                timestamp=datetime.now(),
                metadata={"method": "empty_text"},
            )

        # 使用FIN-R1模型
        if self.model is not None and self.tokenizer is not None:
            return self._analyze_with_finr1(text, **kwargs)
        else:
            # 使用后备方法
            return self._analyze_with_fallback(text, **kwargs)

    def analyze_batch(
        self, texts: List[str], batch_size: int = 32, **kwargs
    ) -> BatchSentimentResult:
        """
        批量分析文本情感

        Args:
            texts: 文本列表
            batch_size: 批次大小
            **kwargs: 额外参数

        Returns:
            批量情感分析结果
        """
        start_time = datetime.now()
        results = []

        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            if self.model is not None and self.tokenizer is not None:
                batch_results = self._analyze_batch_with_finr1(batch_texts, **kwargs)
            else:
                batch_results = [
                    self._analyze_with_fallback(text, **kwargs) for text in batch_texts
                ]

            results.extend(batch_results)

        # 计算整体统计
        processing_time = (datetime.now() - start_time).total_seconds()

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_confidence = 0

        for result in results:
            sentiment_counts[result.sentiment] += 1
            total_confidence += result.confidence

        total_texts = len(results)
        sentiment_distribution = {
            k: v / total_texts for k, v in sentiment_counts.items()
        }

        # 确定整体情感
        overall_sentiment = max(sentiment_distribution.items(), key=lambda x: x[1])[0]

        return BatchSentimentResult(
            results=results,
            overall_sentiment=overall_sentiment,
            average_confidence=total_confidence / total_texts if total_texts > 0 else 0,
            sentiment_distribution=sentiment_distribution,
            processing_time=processing_time,
        )

    def _analyze_with_finr1(self, text: str, **kwargs) -> SentimentResult:
        """
        使用FIN-R1模型分析情感

        Args:
            text: 待分析文本
            **kwargs: 额外参数

        Returns:
            情感分析结果
        """
        try:
            # 文本预处理
            cleaned_text = self._preprocess_text(text)

            # 分词
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            )

            # 移动到设备
            if torch:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)

            # 获取预测结果
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

            # 构建概率分数字典
            prob_scores = {}
            for idx, label in self.label_mapping.items():
                prob_scores[label] = probabilities[0][idx].item()

            sentiment = self.label_mapping[predicted_class]

            return SentimentResult(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                probability_scores=prob_scores,
                timestamp=datetime.now(),
                metadata={
                    "method": "fin_r1",
                    "model_path": self.model_path,
                    "device": self.device,
                },
            )

        except Exception as e:
            logger.error(f"FIN-R1 analysis failed: {str(e)}")
            return self._analyze_with_fallback(text, **kwargs)

    def _analyze_batch_with_finr1(
        self, texts: List[str], **kwargs
    ) -> List[SentimentResult]:
        """
        使用FIN-R1模型批量分析情感

        Args:
            texts: 文本列表
            **kwargs: 额外参数

        Returns:
            情感分析结果列表
        """
        try:
            # 文本预处理
            cleaned_texts = [self._preprocess_text(text) for text in texts]

            # 批量分词
            inputs = self.tokenizer(
                cleaned_texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            )

            # 移动到设备
            if torch:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 批量预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)

            results = []
            for i, text in enumerate(texts):
                predicted_class = torch.argmax(probabilities[i]).item()
                confidence = probabilities[i][predicted_class].item()

                # 构建概率分数字典
                prob_scores = {}
                for idx, label in self.label_mapping.items():
                    prob_scores[label] = probabilities[i][idx].item()

                sentiment = self.label_mapping[predicted_class]

                result = SentimentResult(
                    text=text,
                    sentiment=sentiment,
                    confidence=confidence,
                    probability_scores=prob_scores,
                    timestamp=datetime.now(),
                    metadata={
                        "method": "fin_r1_batch",
                        "model_path": self.model_path,
                        "device": self.device,
                    },
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"FIN-R1 batch analysis failed: {str(e)}")
            return [self._analyze_with_fallback(text, **kwargs) for text in texts]

    def _analyze_with_fallback(self, text: str, **kwargs) -> SentimentResult:
        """
        使用后备方法分析情感（基于关键词）

        Args:
            text: 待分析文本
            **kwargs: 额外参数

        Returns:
            情感分析结果
        """
        # 简单的关键词情感分析
        positive_keywords = [
            "涨",
            "上涨",
            "增长",
            "利好",
            "买入",
            "看好",
            "上升",
            "突破",
            "强势",
            "牛市",
            "profit",
            "gain",
            "rise",
            "bull",
            "positive",
            "growth",
            "increase",
            "up",
        ]

        negative_keywords = [
            "跌",
            "下跌",
            "下降",
            "利空",
            "卖出",
            "看空",
            "下滑",
            "熊市",
            "亏损",
            "暴跌",
            "loss",
            "fall",
            "bear",
            "negative",
            "decline",
            "decrease",
            "down",
            "crash",
        ]

        text_lower = text.lower()

        positive_score = sum(
            1 for keyword in positive_keywords if keyword in text_lower
        )
        negative_score = sum(
            1 for keyword in negative_keywords if keyword in text_lower
        )

        if positive_score > negative_score:
            sentiment = "positive"
            confidence = min(0.6 + 0.1 * (positive_score - negative_score), 0.9)
        elif negative_score > positive_score:
            sentiment = "negative"
            confidence = min(0.6 + 0.1 * (negative_score - positive_score), 0.9)
        else:
            sentiment = "neutral"
            confidence = 0.5

        # 构建概率分数
        total_keywords = positive_score + negative_score + 1  # +1 for neutral baseline
        prob_scores = {
            "positive": (positive_score + 0.1) / total_keywords,
            "negative": (negative_score + 0.1) / total_keywords,
            "neutral": 1.0 / total_keywords,
        }

        # 归一化概率
        total_prob = sum(prob_scores.values())
        prob_scores = {k: v / total_prob for k, v in prob_scores.items()}

        return SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            probability_scores=prob_scores,
            timestamp=datetime.now(),
            metadata={"method": "keyword_fallback"},
        )

    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本

        Args:
            text: 原始文本

        Returns:
            预处理后的文本
        """
        # 基本清理
        text = text.strip()

        # 移除多余空格
        text = " ".join(text.split())

        # 限制长度
        if len(text) > 2000:  # 保守估计，避免超过模型限制
            text = text[:2000] + "..."

        return text

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        return {
            "model_path": self.model_path,
            "device": self.device,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "max_length": self.max_length,
            "label_mapping": self.label_mapping,
        }


# 便捷函数
def analyze_text_sentiment(
    text: str, model_path: Optional[str] = None
) -> SentimentResult:
    """
    分析单个文本情感的便捷函数

    Args:
        text: 待分析文本
        model_path: 模型路径

    Returns:
        情感分析结果
    """
    analyzer = FINR1SentimentAnalyzer(model_path=model_path)
    return analyzer.analyze_sentiment(text)


def analyze_batch_sentiment(
    texts: List[str], model_path: Optional[str] = None, batch_size: int = 32
) -> BatchSentimentResult:
    """
    批量分析文本情感的便捷函数

    Args:
        texts: 文本列表
        model_path: 模型路径
        batch_size: 批次大小

    Returns:
        批量情感分析结果
    """
    analyzer = FINR1SentimentAnalyzer(model_path=model_path)
    return analyzer.analyze_batch(texts, batch_size=batch_size)


# =============================================================================
# Trading Agents Integration - Enhanced Sentiment Analysis Functionality
# =============================================================================


class TradingAgentsSentimentAnalyzer:
    """集成Trading Agents功能的情感分析器"""

    def __init__(self):
        """初始化情感分析器"""
        self.fin_r1_analyzer = FINR1SentimentAnalyzer()
        self.data_collector = (
            ChineseAlternativeDataCollector()
            if ChineseAlternativeDataCollector
            else None
        )
        self.db_manager = get_market_analysis_db() if get_market_analysis_db else None
        self.sentiment_cache = {}
        logger.info(
            "Initialized TradingAgentsSentimentAnalyzer with real data integration"
        )

    async def analyze_stock_sentiment(
        self, symbols: List[str], days: int = 7
    ) -> Dict[str, Any]:
        """分析股票相关新闻情感"""
        try:
            logger.info(f"Starting stock sentiment analysis for {symbols}")

            if not self.data_collector:
                logger.warning("Data collector not available, using fallback")
                return self._create_default_sentiment_result(symbols)

            sentiment_results = {}

            for symbol in symbols:
                news_data = self.data_collector.fetch_stock_news(symbol, limit=50)

                if not news_data.empty:
                    stock_sentiment = await self._analyze_news_sentiment(
                        news_data, symbol
                    )
                    sentiment_results[symbol] = stock_sentiment
                    await self._save_sentiment_result(stock_sentiment, symbol)
                else:
                    sentiment_results[symbol] = self._create_neutral_sentiment()

            return {
                "overall_sentiment": self._calculate_overall_sentiment(
                    sentiment_results
                ),
                "individual_stocks": sentiment_results,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_source": "stock_news",
                "analysis_period_days": days,
            }

        except Exception as e:
            logger.error(f"Stock sentiment analysis failed: {e}")
            return self._create_default_sentiment_result(symbols)


# Import and apply trading agents methods
try:
    from .trading_agents_sentiment_methods import add_trading_agents_methods

    TradingAgentsSentimentAnalyzer = add_trading_agents_methods(
        TradingAgentsSentimentAnalyzer
    )
except ImportError:
    logger.warning("Could not import trading agents methods")


# Global instance for easy access
_sentiment_analyzer_instance = None


def get_sentiment_analyzer() -> TradingAgentsSentimentAnalyzer:
    """获取全局情感分析器实例"""
    global _sentiment_analyzer_instance
    if _sentiment_analyzer_instance is None:
        _sentiment_analyzer_instance = TradingAgentsSentimentAnalyzer()
    return _sentiment_analyzer_instance


# Convenience functions for trading agents integration
async def analyze_symbol_sentiment(symbols: List[str]) -> Dict[str, Any]:
    """分析股票情感的便捷函数"""
    analyzer = get_sentiment_analyzer()
    return await analyzer.analyze_stock_sentiment(symbols)


async def analyze_market_sentiment() -> Dict[str, Any]:
    """分析市场情感的便捷函数"""
    analyzer = get_sentiment_analyzer()
    return await analyzer.analyze_market_sentiment()
