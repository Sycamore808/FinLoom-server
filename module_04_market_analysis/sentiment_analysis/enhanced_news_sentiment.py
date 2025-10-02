"""
Enhanced News Sentiment Analyzer - Integrating Fundamental Analysis
结合基本面分析的新闻情感分析器，集成Trading Agents的新闻分析师功能
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from module_01_data_pipeline import (
        AkshareDataCollector,
        ChineseAlternativeDataCollector,
        ChineseFundamentalCollector,
    )
except ImportError:
    AkshareDataCollector = None
    ChineseAlternativeDataCollector = None
    ChineseFundamentalCollector = None

from common.logging_system import setup_logger

from ..storage_management.market_analysis_database import get_market_analysis_db

logger = setup_logger("enhanced_news_sentiment")


class EnhancedNewsSentimentAnalyzer:
    """增强的新闻情感分析器，整合基本面分析功能"""

    def __init__(self):
        """初始化分析器"""
        self.news_collector = (
            ChineseAlternativeDataCollector()
            if ChineseAlternativeDataCollector
            else None
        )
        self.fundamental_collector = (
            ChineseFundamentalCollector() if ChineseFundamentalCollector else None
        )
        self.db_manager = get_market_analysis_db()
        self.cache = {}
        logger.info(
            "Initialized EnhancedNewsSentimentAnalyzer with real data integration"
        )

    async def analyze_comprehensive_sentiment(
        self, symbols: List[str]
    ) -> Dict[str, Any]:
        """综合分析新闻情感和基本面数据

        Args:
            symbols: 股票代码列表

        Returns:
            综合分析结果
        """
        try:
            logger.info(f"Starting comprehensive sentiment analysis for {symbols}")

            results = {}

            for symbol in symbols:
                # 并行获取新闻和基本面数据
                news_task = self._get_news_sentiment(symbol)
                fundamental_task = self._get_fundamental_sentiment(symbol)

                news_sentiment, fundamental_sentiment = await asyncio.gather(
                    news_task, fundamental_task, return_exceptions=True
                )

                # 处理异常情况
                if isinstance(news_sentiment, Exception):
                    logger.error(
                        f"News sentiment analysis failed for {symbol}: {news_sentiment}"
                    )
                    news_sentiment = self._create_neutral_sentiment("news")

                if isinstance(fundamental_sentiment, Exception):
                    logger.error(
                        f"Fundamental sentiment analysis failed for {symbol}: {fundamental_sentiment}"
                    )
                    fundamental_sentiment = self._create_neutral_sentiment(
                        "fundamental"
                    )

                # 综合分析
                comprehensive_result = self._combine_sentiments(
                    symbol, news_sentiment, fundamental_sentiment
                )

                results[symbol] = comprehensive_result

                # 保存结果
                await self._save_analysis_result(comprehensive_result, symbol)

            # 计算整体市场情感
            overall_sentiment = self._calculate_overall_sentiment(results)

            return {
                "individual_results": results,
                "overall_sentiment": overall_sentiment,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_sources": ["news", "fundamentals"],
                "analysis_type": "comprehensive",
            }

        except Exception as e:
            logger.error(f"Comprehensive sentiment analysis failed: {e}")
            return self._create_error_result(symbols, str(e))

    async def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """获取新闻情感数据"""
        try:
            if not self.news_collector:
                return self._create_neutral_sentiment("news")

            # 获取股票相关新闻
            stock_news = self.news_collector.fetch_stock_news(symbol, limit=30)

            # 获取股票详细信息
            stock_detail = self.news_collector.fetch_detail(symbol)

            # 分析新闻情感
            news_sentiment = self._analyze_news_texts(stock_news)

            # 分析公司信息情感
            company_sentiment = self._analyze_company_info(stock_detail)

            # 综合新闻情感
            combined_sentiment = self._combine_news_sentiments(
                news_sentiment, company_sentiment
            )

            return {
                "sentiment_type": "news",
                "sentiment_score": combined_sentiment["score"],
                "sentiment_label": combined_sentiment["label"],
                "confidence": combined_sentiment["confidence"],
                "news_count": len(stock_news) if not stock_news.empty else 0,
                "key_factors": combined_sentiment["factors"],
                "data_quality": "high" if not stock_news.empty else "low",
            }

        except Exception as e:
            logger.error(f"News sentiment analysis failed for {symbol}: {e}")
            return self._create_neutral_sentiment("news")

    async def _get_fundamental_sentiment(self, symbol: str) -> Dict[str, Any]:
        """获取基本面情感数据"""
        try:
            if not self.fundamental_collector:
                return self._create_neutral_sentiment("fundamental")

            # 获取财务数据
            financial_indicators = (
                self.fundamental_collector.fetch_financial_indicators(symbol)
            )
            balance_sheet = self.fundamental_collector.fetch_financial_statements(
                symbol, "资产负债表"
            )
            income_statement = self.fundamental_collector.fetch_financial_statements(
                symbol, "利润表"
            )

            # 分析财务健康度
            financial_health = self._analyze_financial_health(
                financial_indicators, balance_sheet, income_statement
            )

            # 获取分红数据
            dividend_data = self.fundamental_collector.fetch_dividend_history(symbol)
            dividend_sentiment = self._analyze_dividend_sentiment(dividend_data)

            # 综合基本面情感
            fundamental_sentiment = self._combine_fundamental_sentiments(
                financial_health, dividend_sentiment
            )

            return {
                "sentiment_type": "fundamental",
                "sentiment_score": fundamental_sentiment["score"],
                "sentiment_label": fundamental_sentiment["label"],
                "confidence": fundamental_sentiment["confidence"],
                "financial_health": financial_health,
                "dividend_sentiment": dividend_sentiment,
                "key_indicators": fundamental_sentiment["indicators"],
                "data_quality": "high",
            }

        except Exception as e:
            logger.error(f"Fundamental sentiment analysis failed for {symbol}: {e}")
            return self._create_neutral_sentiment("fundamental")

    def _analyze_news_texts(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """分析新闻文本情感"""
        if news_data.empty:
            return {"score": 0.0, "label": "neutral", "confidence": 0.0, "factors": []}

        # 提取新闻标题和内容
        texts = []
        if "标题" in news_data.columns:
            texts.extend(news_data["标题"].fillna("").tolist())
        if "title" in news_data.columns:
            texts.extend(news_data["title"].fillna("").tolist())
        if "内容" in news_data.columns:
            texts.extend(news_data["内容"].fillna("").tolist())

        if not texts:
            return {"score": 0.0, "label": "neutral", "confidence": 0.0, "factors": []}

        # 简单关键词情感分析（在实际应用中应使用FIN-R1模型）
        positive_keywords = [
            "利好",
            "上涨",
            "增长",
            "突破",
            "买入",
            "看好",
            "强势",
            "创新高",
            "盈利",
            "业绩",
            "合作",
            "扩张",
            "投资",
            "发展",
            "机遇",
        ]

        negative_keywords = [
            "利空",
            "下跌",
            "下降",
            "跌破",
            "卖出",
            "看空",
            "弱势",
            "创新低",
            "亏损",
            "风险",
            "问题",
            "困难",
            "下滑",
            "压力",
            "挑战",
        ]

        combined_text = " ".join(texts)
        positive_count = sum(
            1 for keyword in positive_keywords if keyword in combined_text
        )
        negative_count = sum(
            1 for keyword in negative_keywords if keyword in combined_text
        )

        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            return {"score": 0.0, "label": "neutral", "confidence": 0.5, "factors": []}

        sentiment_score = (positive_count - negative_count) / total_keywords
        confidence = min(0.8, 0.5 + total_keywords * 0.1)

        # 提取关键因素
        factors = []
        if positive_count > 0:
            factors.extend([kw for kw in positive_keywords if kw in combined_text][:3])
        if negative_count > 0:
            factors.extend([kw for kw in negative_keywords if kw in combined_text][:3])

        return {
            "score": sentiment_score,
            "label": self._score_to_label(sentiment_score),
            "confidence": confidence,
            "factors": factors,
        }

    def _analyze_company_info(self, company_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析公司信息情感"""
        if not company_info:
            return {"score": 0.0, "label": "neutral", "confidence": 0.0, "factors": []}

        # 分析公司基本信息的正面因素
        positive_factors = []
        negative_factors = []

        # 分析主营业务
        main_business = company_info.get("main_operation_business", "")
        if main_business:
            if any(
                keyword in main_business
                for keyword in ["科技", "创新", "研发", "智能", "数字"]
            ):
                positive_factors.append("科技创新业务")
            if any(keyword in main_business for keyword in ["传统", "落后", "过时"]):
                negative_factors.append("传统业务模式")

        # 分析公司规模和地位
        if company_info.get("market_cap", 0) > 10000000000:  # 100亿以上市值
            positive_factors.append("大型企业规模")

        # 计算情感分数
        pos_score = len(positive_factors)
        neg_score = len(negative_factors)
        total_score = pos_score + neg_score

        if total_score == 0:
            return {"score": 0.0, "label": "neutral", "confidence": 0.5, "factors": []}

        sentiment_score = (
            (pos_score - neg_score) / max(total_score, 1) * 0.3
        )  # 降低权重
        confidence = min(0.6, 0.4 + total_score * 0.1)

        return {
            "score": sentiment_score,
            "label": self._score_to_label(sentiment_score),
            "confidence": confidence,
            "factors": positive_factors + negative_factors,
        }

    def _analyze_financial_health(
        self,
        indicators: Dict,
        balance_sheet: pd.DataFrame,
        income_statement: pd.DataFrame,
    ) -> Dict[str, Any]:
        """分析财务健康度"""
        try:
            health_score = 0.0
            factors = []

            # 分析关键财务指标
            if indicators:
                # ROE分析
                roe = indicators.get("roe", 0)
                if roe > 0.15:
                    health_score += 0.3
                    factors.append("高ROE")
                elif roe < 0.05:
                    health_score -= 0.2
                    factors.append("低ROE")

                # PE比率分析
                pe_ratio = indicators.get("pe_ratio", 0)
                if 10 < pe_ratio < 25:
                    health_score += 0.2
                    factors.append("合理估值")
                elif pe_ratio > 50:
                    health_score -= 0.2
                    factors.append("高估值")

                # 债务比率分析
                debt_ratio = indicators.get("debt_ratio", 0)
                if debt_ratio < 0.4:
                    health_score += 0.2
                    factors.append("低负债")
                elif debt_ratio > 0.7:
                    health_score -= 0.3
                    factors.append("高负债")

            # 标准化分数到[-1, 1]
            health_score = max(-1.0, min(1.0, health_score))

            return {
                "score": health_score,
                "label": self._score_to_label(health_score),
                "confidence": 0.8 if indicators else 0.3,
                "factors": factors,
            }

        except Exception as e:
            logger.error(f"Financial health analysis failed: {e}")
            return {"score": 0.0, "label": "neutral", "confidence": 0.3, "factors": []}

    def _analyze_dividend_sentiment(
        self, dividend_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """分析分红情感"""
        try:
            if dividend_data.empty:
                return {
                    "score": 0.0,
                    "label": "neutral",
                    "confidence": 0.3,
                    "factors": [],
                }

            # 分析最近分红记录
            recent_dividends = dividend_data.head(3)  # 最近3次分红

            dividend_score = 0.0
            factors = []

            if len(recent_dividends) > 0:
                # 检查分红连续性
                if len(recent_dividends) >= 2:
                    dividend_score += 0.3
                    factors.append("持续分红")

                # 分析分红率
                avg_yield = recent_dividends.get(
                    "dividend_yield", pd.Series([0])
                ).mean()
                if avg_yield > 0.03:  # 超过3%
                    dividend_score += 0.2
                    factors.append("高股息率")
                elif avg_yield > 0.01:  # 1-3%
                    dividend_score += 0.1
                    factors.append("稳定股息")

            return {
                "score": dividend_score,
                "label": self._score_to_label(dividend_score),
                "confidence": 0.6,
                "factors": factors,
            }

        except Exception as e:
            logger.error(f"Dividend sentiment analysis failed: {e}")
            return {"score": 0.0, "label": "neutral", "confidence": 0.3, "factors": []}

    def _combine_news_sentiments(
        self, news_sentiment: Dict, company_sentiment: Dict
    ) -> Dict[str, Any]:
        """综合新闻情感"""
        # 新闻情感权重更高
        news_weight = 0.7
        company_weight = 0.3

        combined_score = (
            news_sentiment["score"] * news_weight
            + company_sentiment["score"] * company_weight
        )

        combined_confidence = (
            news_sentiment["confidence"] * news_weight
            + company_sentiment["confidence"] * company_weight
        )

        combined_factors = news_sentiment["factors"] + company_sentiment["factors"]

        return {
            "score": combined_score,
            "label": self._score_to_label(combined_score),
            "confidence": combined_confidence,
            "factors": combined_factors[:5],  # 限制因素数量
        }

    def _combine_fundamental_sentiments(
        self, financial_health: Dict, dividend_sentiment: Dict
    ) -> Dict[str, Any]:
        """综合基本面情感"""
        # 财务健康度权重更高
        health_weight = 0.8
        dividend_weight = 0.2

        combined_score = (
            financial_health["score"] * health_weight
            + dividend_sentiment["score"] * dividend_weight
        )

        combined_confidence = (
            financial_health["confidence"] * health_weight
            + dividend_sentiment["confidence"] * dividend_weight
        )

        # 合并关键指标
        indicators = {
            "financial_health": financial_health["score"],
            "dividend_quality": dividend_sentiment["score"],
            "combined_score": combined_score,
        }

        return {
            "score": combined_score,
            "label": self._score_to_label(combined_score),
            "confidence": combined_confidence,
            "indicators": indicators,
        }

    def _combine_sentiments(
        self, symbol: str, news_sentiment: Dict, fundamental_sentiment: Dict
    ) -> Dict[str, Any]:
        """综合新闻和基本面情感"""
        # 权重分配：新闻40%，基本面60%
        news_weight = 0.4
        fundamental_weight = 0.6

        final_score = (
            news_sentiment["sentiment_score"] * news_weight
            + fundamental_sentiment["sentiment_score"] * fundamental_weight
        )

        final_confidence = (
            news_sentiment["confidence"] * news_weight
            + fundamental_sentiment["confidence"] * fundamental_weight
        )

        # 合并关键因素
        all_factors = []
        if "key_factors" in news_sentiment:
            all_factors.extend(
                [f"新闻: {f}" for f in news_sentiment["key_factors"][:3]]
            )
        if "key_indicators" in fundamental_sentiment:
            health_score = fundamental_sentiment["key_indicators"]["financial_health"]
            if health_score > 0.2:
                all_factors.append("基本面: 财务健康")
            elif health_score < -0.2:
                all_factors.append("基本面: 财务风险")

        return {
            "symbol": symbol,
            "final_sentiment_score": final_score,
            "final_sentiment_label": self._score_to_label(final_score),
            "final_confidence": final_confidence,
            "news_sentiment": news_sentiment,
            "fundamental_sentiment": fundamental_sentiment,
            "key_factors": all_factors,
            "recommendation": self._generate_recommendation(
                final_score, final_confidence
            ),
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def _generate_recommendation(self, score: float, confidence: float) -> str:
        """生成投资建议"""
        if confidence < 0.5:
            return "数据不足，建议谨慎观察"

        if score > 0.3:
            return "积极信号，可考虑关注"
        elif score > 0.1:
            return "温和积极，可适度关注"
        elif score < -0.3:
            return "消极信号，建议谨慎"
        elif score < -0.1:
            return "偏消极，建议观望"
        else:
            return "中性信号，建议持续观察"

    def _score_to_label(self, score: float) -> str:
        """分数转标签"""
        if score > 0.2:
            return "positive"
        elif score < -0.2:
            return "negative"
        else:
            return "neutral"

    def _calculate_overall_sentiment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算整体情感"""
        if not results:
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
            }

        scores = [result["final_sentiment_score"] for result in results.values()]
        confidences = [result["final_confidence"] for result in results.values()]

        avg_score = np.mean(scores)
        avg_confidence = np.mean(confidences)

        return {
            "sentiment_score": avg_score,
            "sentiment_label": self._score_to_label(avg_score),
            "confidence": avg_confidence,
            "stock_count": len(results),
        }

    def _create_neutral_sentiment(self, sentiment_type: str) -> Dict[str, Any]:
        """创建中性情感"""
        base_result = {
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "confidence": 0.5,
        }

        if sentiment_type == "news":
            base_result.update(
                {
                    "sentiment_type": "news",
                    "news_count": 0,
                    "key_factors": [],
                    "data_quality": "low",
                }
            )
        elif sentiment_type == "fundamental":
            base_result.update(
                {
                    "sentiment_type": "fundamental",
                    "financial_health": {
                        "score": 0.0,
                        "label": "neutral",
                        "confidence": 0.5,
                        "factors": [],
                    },
                    "dividend_sentiment": {
                        "score": 0.0,
                        "label": "neutral",
                        "confidence": 0.5,
                        "factors": [],
                    },
                    "key_indicators": {
                        "financial_health": 0.0,
                        "dividend_quality": 0.0,
                        "combined_score": 0.0,
                    },
                    "data_quality": "low",
                }
            )

        return base_result

    def _create_error_result(
        self, symbols: List[str], error_msg: str
    ) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            "individual_results": {
                symbol: self._create_neutral_sentiment("error") for symbol in symbols
            },
            "overall_sentiment": {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
            },
            "analysis_timestamp": datetime.now().isoformat(),
            "error": error_msg,
            "status": "failed",
        }

    async def _save_analysis_result(self, result: Dict[str, Any], symbol: str) -> None:
        """保存分析结果"""
        try:
            if not self.db_manager:
                logger.warning("Database manager not available")
                return

            analysis_id = f"enhanced_sentiment_{uuid.uuid4().hex[:8]}"

            db_data = {
                "analysis_id": analysis_id,
                "symbol": symbol,
                "text_source": "comprehensive",
                "sentiment_score": result["final_sentiment_score"],
                "sentiment_label": result["final_sentiment_label"],
                "confidence": result["final_confidence"],
                "keywords": result["key_factors"],
                "analysis_method": "enhanced_news_fundamental",
                "source_data": result,
                "timestamp": datetime.now(),
            }

            success = self.db_manager.save_sentiment_analysis(db_data)
            if success:
                logger.info(f"Saved enhanced sentiment analysis for {symbol}")

        except Exception as e:
            logger.error(f"Failed to save analysis result: {e}")
