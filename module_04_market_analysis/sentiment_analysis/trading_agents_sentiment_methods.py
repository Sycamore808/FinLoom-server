"""
Trading Agents Sentiment Analysis Methods
Continuation of sentiment analysis functionality from fin_r1_sentiment.py
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from common.logging_system import setup_logger

logger = setup_logger("trading_agents_sentiment_methods")


def add_trading_agents_methods(cls):
    """Decorator to add trading agents methods to TradingAgentsSentimentAnalyzer"""

    async def analyze_market_sentiment(self) -> Dict[str, Any]:
        """分析市场整体情感"""
        try:
            logger.info("Starting market sentiment analysis")

            if not self.data_collector:
                logger.warning("Data collector not available, using fallback")
                return self._create_default_market_sentiment()

            # 获取新闻联播数据
            news_data = self.data_collector.fetch_news_data(limit=30)

            # 获取板块数据进行情绪判断
            sector_data = self.data_collector.fetch_sector_performance()

            # 分析新闻情感
            news_sentiment = await self._analyze_news_sentiment(news_data, "market")

            # 分析板块表现情感
            sector_sentiment = self._analyze_sector_sentiment(sector_data)

            # 综合情感分析
            market_sentiment = {
                "news_sentiment": news_sentiment,
                "sector_sentiment": sector_sentiment,
                "overall_sentiment": (
                    news_sentiment["sentiment_score"]
                    + sector_sentiment["sentiment_score"]
                )
                / 2,
                "confidence": min(
                    news_sentiment["confidence"], sector_sentiment["confidence"]
                ),
                "analysis_timestamp": datetime.now().isoformat(),
                "data_sources": ["news_cctv", "sector_performance"],
            }

            # 保存结果
            await self._save_sentiment_result(market_sentiment, "market")

            return market_sentiment

        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            return self._create_default_market_sentiment()

    async def _analyze_news_sentiment(
        self, news_data: pd.DataFrame, symbol: str
    ) -> Dict[str, Any]:
        """分析新闻情感"""
        try:
            if news_data.empty:
                return self._create_neutral_sentiment()

            # 提取新闻文本
            texts = []
            for col in ["content", "title", "标题", "内容"]:
                if col in news_data.columns:
                    texts.extend(news_data[col].fillna("").tolist())

            if not texts:
                return self._create_neutral_sentiment()

            # 使用FIN-R1模型分析情感
            batch_result = self.fin_r1_analyzer.analyze_batch(texts, batch_size=16)

            # 计算整体情感分数
            sentiment_score = self._convert_sentiment_to_score(
                batch_result.overall_sentiment, batch_result.average_confidence
            )

            return {
                "sentiment_score": sentiment_score,
                "sentiment_label": batch_result.overall_sentiment,
                "confidence": batch_result.average_confidence,
                "keywords": self._extract_keywords_from_texts(texts),
                "news_count": len(news_data),
                "analysis_method": "fin_r1_enhanced",
                "sentiment_distribution": batch_result.sentiment_distribution,
                "processing_time": batch_result.processing_time,
            }

        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return self._create_neutral_sentiment()

    def _analyze_sector_sentiment(self, sector_data: pd.DataFrame) -> Dict[str, Any]:
        """分析板块情感"""
        try:
            if sector_data.empty:
                return self._create_neutral_sentiment()

            # 基于涨跌幅分析情感
            if "pct_chg" in sector_data.columns:
                avg_change = sector_data["pct_chg"].mean()
                positive_count = (sector_data["pct_chg"] > 0).sum()
                total_count = len(sector_data)

                # 计算情感分数
                change_sentiment = np.tanh(avg_change / 2.0)  # 归一化到[-1, 1]
                ratio_sentiment = (
                    positive_count / total_count - 0.5
                ) * 2  # 归一化到[-1, 1]

                sentiment_score = (change_sentiment + ratio_sentiment) / 2
                confidence = min(0.8, 0.5 + abs(sentiment_score) * 0.5)
            else:
                sentiment_score = 0.0
                positive_count = 0
                total_count = 0
                avg_change = 0.0
                confidence = 0.5

            return {
                "sentiment_score": sentiment_score,
                "sentiment_label": self._score_to_label(sentiment_score),
                "confidence": confidence,
                "positive_sectors": int(positive_count),
                "total_sectors": int(total_count),
                "avg_change": float(avg_change),
                "analysis_method": "sector_performance",
            }

        except Exception as e:
            logger.error(f"Sector sentiment analysis failed: {e}")
            return self._create_neutral_sentiment()

    def _convert_sentiment_to_score(
        self, sentiment_label: str, confidence: float
    ) -> float:
        """将情感标签转换为分数"""
        score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        base_score = score_map.get(sentiment_label, 0.0)
        return base_score * confidence

    def _extract_keywords_from_texts(self, texts: List[str]) -> List[str]:
        """从文本列表中提取关键词"""
        keywords = [
            "上涨",
            "下跌",
            "增长",
            "下降",
            "利好",
            "利空",
            "突破",
            "跌破",
            "强势",
            "弱势",
            "乐观",
            "悲观",
            "买入",
            "卖出",
            "推荐",
            "风险",
            "增持",
            "减持",
            "买进",
            "卖出",
            "看好",
            "看空",
            "牛市",
            "熊市",
        ]

        combined_text = " ".join(texts)
        found_keywords = [keyword for keyword in keywords if keyword in combined_text]
        return found_keywords[:15]  # 最多返回15个关键词

    def _score_to_label(self, score: float) -> str:
        """将分数转换为标签"""
        if score > 0.2:
            return "positive"
        elif score < -0.2:
            return "negative"
        else:
            return "neutral"

    def _calculate_overall_sentiment(
        self, sentiment_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算整体情感"""
        if not sentiment_results:
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
            }

        scores = [result["sentiment_score"] for result in sentiment_results.values()]
        confidences = [result["confidence"] for result in sentiment_results.values()]

        avg_score = np.mean(scores)
        avg_confidence = np.mean(confidences)

        return {
            "sentiment_score": avg_score,
            "sentiment_label": self._score_to_label(avg_score),
            "confidence": avg_confidence,
            "stock_count": len(sentiment_results),
        }

    def _create_neutral_sentiment(self) -> Dict[str, Any]:
        """创建中性情感结果"""
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "confidence": 0.5,
            "keywords": [],
            "analysis_method": "default",
        }

    def _create_default_sentiment_result(self, symbols: List[str]) -> Dict[str, Any]:
        """创建默认情感结果"""
        return {
            "overall_sentiment": self._create_neutral_sentiment(),
            "individual_stocks": {
                symbol: self._create_neutral_sentiment() for symbol in symbols
            },
            "analysis_timestamp": datetime.now().isoformat(),
            "data_source": "default",
        }

    def _create_default_market_sentiment(self) -> Dict[str, Any]:
        """创建默认市场情感"""
        neutral = self._create_neutral_sentiment()
        return {
            "news_sentiment": neutral,
            "sector_sentiment": neutral,
            "overall_sentiment": 0.0,
            "confidence": 0.5,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    async def _save_sentiment_result(
        self, sentiment_data: Dict[str, Any], symbol: str
    ) -> None:
        """保存情感分析结果到数据库"""
        try:
            if not self.db_manager:
                logger.warning("Database manager not available, skipping save")
                return

            analysis_id = f"sentiment_{uuid.uuid4().hex[:8]}"

            db_data = {
                "analysis_id": analysis_id,
                "symbol": symbol,
                "text_source": sentiment_data.get("data_source", "unknown"),
                "sentiment_score": sentiment_data.get("sentiment_score", 0.0),
                "sentiment_label": sentiment_data.get("sentiment_label", "neutral"),
                "confidence": sentiment_data.get("confidence", 0.5),
                "keywords": sentiment_data.get("keywords", []),
                "analysis_method": sentiment_data.get(
                    "analysis_method", "fin_r1_enhanced"
                ),
                "source_data": sentiment_data,
                "timestamp": datetime.now(),
            }

            success = self.db_manager.save_sentiment_analysis(db_data)
            if success:
                logger.info(f"Saved sentiment analysis for {symbol}")
            else:
                logger.warning(f"Failed to save sentiment analysis for {symbol}")

        except Exception as e:
            logger.error(f"Failed to save sentiment result: {e}")

    # Add all methods to the class
    cls.analyze_market_sentiment = analyze_market_sentiment
    cls._analyze_news_sentiment = _analyze_news_sentiment
    cls._analyze_sector_sentiment = _analyze_sector_sentiment
    cls._convert_sentiment_to_score = _convert_sentiment_to_score
    cls._extract_keywords_from_texts = _extract_keywords_from_texts
    cls._score_to_label = _score_to_label
    cls._calculate_overall_sentiment = _calculate_overall_sentiment
    cls._create_neutral_sentiment = _create_neutral_sentiment
    cls._create_default_sentiment_result = _create_default_sentiment_result
    cls._create_default_market_sentiment = _create_default_market_sentiment
    cls._save_sentiment_result = _save_sentiment_result

    return cls
