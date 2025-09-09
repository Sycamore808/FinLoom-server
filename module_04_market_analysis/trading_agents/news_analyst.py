"""
国内新闻分析师
集成国内新闻数据源和Fin-R1模型进行新闻情感分析
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
try:
    from module_01_data_pipeline.data_acquisition.akshare_collector import AkshareDataCollector
except ImportError:
    AkshareDataCollector = None

try:
    from module_04_market_analysis.sentiment_analysis.news_sentiment_analyzer import NewsSentimentAnalyzer, NewsArticle, SentimentConfig
except ImportError:
    NewsSentimentAnalyzer = None
    NewsArticle = None
    SentimentConfig = None

try:
    from module_10_ai_interaction.fin_r1_integration import FINR1Integration
except ImportError:
    FINR1Integration = None
from .base_agent import BaseAgent, AgentAnalysis, AgentRecommendation, RecommendationType, DebateContext, DebateResponse, create_analysis_id
from common.logging_system import setup_logger

logger = setup_logger("domestic_news_analyst")

class DomesticNewsAnalyst(BaseAgent):
    """
    国内新闻分析师：负责收集国内金融新闻，并利用Fin-R1模型进行情感和内容分析。
    """
    def __init__(self, fin_r1_config: Dict[str, Any] = None, sentiment_config: SentimentConfig = None):
        super().__init__(
            name="DomesticNewsAnalyst",
            agent_type="news_analyst",
            expertise="国内新闻分析和情感分析"
        )
        self.data_collector = AkshareDataCollector() if AkshareDataCollector else None
        self.fin_r1 = FINR1Integration(fin_r1_config or {}) if FINR1Integration else None
        self.sentiment_analyzer = NewsSentimentAnalyzer(sentiment_config) if NewsSentimentAnalyzer else None
        self.last_recommendation = None  # 添加last_recommendation属性
        self.analysis_count = 0  # 添加analysis_count属性
        self.last_analysis_time = None  # 添加last_analysis_time属性
        logger.info("DomesticNewsAnalyst initialized with FIN-R1 and NewsSentimentAnalyzer.")

    async def analyze(
        self, 
        symbols: List[str], 
        market_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """
        执行国内新闻分析。
        Args:
            symbols: 股票代码列表
            market_data: 市场数据（可选）
            context: 分析上下文（可选）
        Returns:
            AgentAnalysis: 包含新闻分析结果的AgentAnalysis对象。
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting domestic news analysis for symbols: {symbols}")
            
            # 从context中获取参数，如果没有则使用默认值
            target_symbols = symbols
            lookback_days = context.get("news_lookback_days", 7) if context else 7

            # 1. 收集国内新闻数据
            domestic_news = await self._fetch_domestic_news(target_symbols, lookback_days)
            if not domestic_news:
                logger.warning("No domestic news found for analysis.")
                return self._create_default_analysis(symbols, "未发现相关国内新闻")

            # 2. 使用NewsSentimentAnalyzer进行初步情感分析
            news_articles_for_sentiment = [
                NewsArticle(
                    article_id=str(idx),
                    title=n.get("title", ""),
                    content=n.get("content", ""),
                    source=n.get("source", "unknown"),
                    timestamp=n.get("timestamp", datetime.now()),
                    url=n.get("url", ""),
                    symbols=n.get("symbols", []),
                    metadata=n
                ) for idx, n in enumerate(domestic_news)
            ]
            sentiment_results_by_symbol = self.sentiment_analyzer.analyze_news_sentiment(news_articles_for_sentiment, symbols=target_symbols)

            # 3. 结合Fin-R1进行深度内容和情感总结
            finr1_analysis_summary = await self._process_news_with_finr1(domestic_news, sentiment_results_by_symbol, target_symbols)

            # 4. 生成推荐
            overall_sentiment = finr1_analysis_summary.get("overall_sentiment", 0.0)
            confidence = finr1_analysis_summary.get("confidence", 0.7)
            
            recommendation = self._generate_recommendation(finr1_analysis_summary, symbols)
            
            # 5. 计算分析耗时
            analysis_duration = (datetime.now() - start_time).total_seconds()
            
            # 6. 创建分析结果
            analysis = AgentAnalysis(
                agent_name=self.name,
                agent_type=self.agent_type,
                analysis_id=create_analysis_id(),
                timestamp=datetime.now(),
                symbols=symbols,
                recommendation=recommendation,
                key_factors=self._extract_key_factors(finr1_analysis_summary),
                risk_factors=self._extract_risk_factors(finr1_analysis_summary),
                market_outlook=self._generate_market_outlook(finr1_analysis_summary),
                additional_insights={
                    "news_count": len(domestic_news),
                    "sentiment_results": {s: [sr.__dict__ for sr in res] for s, res in sentiment_results_by_symbol.items()},
                    "finr1_insights": finr1_analysis_summary.get("insights", {})
                },
                data_sources=["国内新闻", "Fin-R1模型", "情感分析"],
                analysis_duration=analysis_duration
            )
            
            # 7. 更新统计信息
            self.analysis_count += 1
            self.last_analysis_time = datetime.now()
            self.last_recommendation = recommendation  # 更新last_recommendation
            
            logger.info(f"Domestic news analysis completed for {symbols} in {analysis_duration:.2f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Domestic news analysis failed: {e}")
            return self._create_default_analysis(symbols, f"新闻分析失败: {e}")

    async def _fetch_domestic_news(self, symbols: List[str], lookback_days: int) -> List[Dict[str, Any]]:
        """
        模拟或实际从国内数据源获取新闻。
        实际实现中，这里会调用Akshare或其他国内新闻API。
        """
        logger.info(f"Fetching domestic news for symbols: {symbols} for last {lookback_days} days.")
        # Placeholder for actual data collection
        # Example: Use Akshare to get news, or a dedicated news API
        mock_news = [
            {"title": "某公司发布利好财报", "content": "公司Q3利润超预期，股价大涨。", "source": "新华网", "timestamp": datetime.now(), "url": "http://example.com/news1", "symbols": ["000001"]},
            {"title": "行业政策收紧，对某板块构成压力", "content": "监管部门出台新规，相关股票承压。", "source": "财经网", "timestamp": datetime.now() - timedelta(days=1), "url": "http://example.com/news2", "symbols": ["600036"]},
        ]
        # Filter mock news by symbols if provided
        if symbols:
            filtered_news = [n for n in mock_news if any(s in n.get("symbols", []) for s in symbols)]
            return filtered_news
        return mock_news

    async def _process_news_with_finr1(self, news_data: List[Dict[str, Any]], sentiment_results: Dict[str, List], target_symbols: List[str]) -> Dict[str, Any]:
        """
        利用Fin-R1模型对新闻进行深度分析和总结。
        """
        logger.info("Processing news with Fin-R1 for deeper insights.")
        news_texts = [n.get("title", "") + " " + n.get("content", "") for n in news_data]
        combined_text = "\n".join(news_texts)

        prompt = f"根据以下国内金融新闻和初步情感分析结果，请使用Fin-R1模型对市场情绪和相关股票（{', '.join(target_symbols) if target_symbols else '整体市场'}）进行综合分析和总结，并给出整体情感分数（-1到1）和置信度（0到1）。\n\n新闻内容：\n{combined_text}\n\n初步情感分析结果：\n{sentiment_results}"

        try:
            # 调用FIN-R1进行分析
            finr1_response = await self.fin_r1.process_request(prompt)
            # 假设FIN-R1返回一个包含summary, overall_sentiment, confidence, insights的字典
            return finr1_response.get("analysis_result", {
                "summary": "Fin-R1新闻分析完成。",
                "overall_sentiment": 0.0,
                "confidence": 0.7,
                "insights": {"raw_response": finr1_response}
            })
        except Exception as e:
            logger.error(f"Fin-R1 news processing failed: {e}")
            return {
                "summary": f"Fin-R1新闻分析失败: {e}",
                "overall_sentiment": 0.0,
                "confidence": 0.3,
                "insights": {}
            }

    async def debate(
        self, 
        other_analyses: List[AgentAnalysis],
        debate_context: DebateContext
    ) -> DebateResponse:
        """参与辩论"""
        try:
            # 分析其他智能体的观点
            supporting_arguments = []
            counter_arguments = []
            
            for analysis in other_analyses:
                if analysis.agent_type == "sentiment_analyst":
                    # 与情绪分析师的观点对比
                    if self.last_recommendation and analysis.recommendation.recommendation_type == self.last_recommendation.recommendation_type:
                        supporting_arguments.append(f"情绪分析师也支持{analysis.recommendation.recommendation_type.value}观点")
                    else:
                        counter_arguments.append(f"情绪分析师持{analysis.recommendation.recommendation_type.value}观点，与新闻分析不一致")
                
                elif analysis.agent_type == "fundamental_analyst":
                    # 与基本面分析师的观点对比
                    if self.last_recommendation and analysis.recommendation.recommendation_type == self.last_recommendation.recommendation_type:
                        supporting_arguments.append("基本面分析支持新闻分析结论")
                    else:
                        counter_arguments.append("基本面分析与新闻分析存在分歧")
            
            # 计算辩论后的置信度
            updated_confidence = self.get_confidence_score()
            if supporting_arguments:
                updated_confidence = min(updated_confidence + 0.1, 1.0)
            if counter_arguments:
                updated_confidence = max(updated_confidence - 0.1, 0.0)
            
            # 确定立场
            position = "中立"
            if self.last_recommendation:
                if self.last_recommendation.recommendation_type in [RecommendationType.BUY, RecommendationType.STRONG_BUY]:
                    position = "支持"
                elif self.last_recommendation.recommendation_type in [RecommendationType.SELL, RecommendationType.STRONG_SELL]:
                    position = "反对"
            
            return DebateResponse(
                agent_name=self.name,
                position=position,
                arguments=supporting_arguments,
                evidence={"news_analysis": "基于国内新闻和Fin-R1模型分析"},
                counter_arguments=counter_arguments,
                updated_confidence=updated_confidence,
                willingness_to_compromise=0.6
            )
            
        except Exception as e:
            logger.error(f"News analyst debate failed: {e}")
            return DebateResponse(
                agent_name=self.name,
                position="中立",
                arguments=[],
                evidence={},
                counter_arguments=[],
                updated_confidence=0.5,
                willingness_to_compromise=0.8
            )

    def _generate_recommendation(self, finr1_analysis: Dict[str, Any], symbols: List[str]) -> AgentRecommendation:
        """生成推荐"""
        overall_sentiment = finr1_analysis.get("overall_sentiment", 0.0)
        confidence = finr1_analysis.get("confidence", 0.7)
        
        # 根据情感分数确定推荐类型
        if overall_sentiment > 0.3:
            recommendation_type = RecommendationType.BUY
        elif overall_sentiment > 0.1:
            recommendation_type = RecommendationType.BUY
        elif overall_sentiment < -0.3:
            recommendation_type = RecommendationType.SELL
        elif overall_sentiment < -0.1:
            recommendation_type = RecommendationType.SELL
        else:
            recommendation_type = RecommendationType.HOLD
        
        # 根据置信度调整推荐强度
        if confidence > 0.8 and abs(overall_sentiment) > 0.5:
            if recommendation_type == RecommendationType.BUY:
                recommendation_type = RecommendationType.STRONG_BUY
            elif recommendation_type == RecommendationType.SELL:
                recommendation_type = RecommendationType.STRONG_SELL
        
        reasoning = finr1_analysis.get("summary", "基于国内新闻分析")
        
        return AgentRecommendation(
            recommendation_type=recommendation_type,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data=finr1_analysis,
            risk_level=self._assess_risk_level(overall_sentiment),
            time_horizon="short"
        )

    def _extract_key_factors(self, finr1_analysis: Dict[str, Any]) -> List[str]:
        """提取关键因素"""
        factors = []
        
        insights = finr1_analysis.get("insights", {})
        if insights:
            factors.append("Fin-R1深度分析洞察")
        
        sentiment = finr1_analysis.get("overall_sentiment", 0.0)
        if abs(sentiment) > 0.5:
            factors.append("强烈情感信号")
        elif abs(sentiment) > 0.2:
            factors.append("中等情感信号")
        
        factors.append("国内新闻覆盖")
        factors.append("多源情感分析")
        
        return factors

    def _extract_risk_factors(self, finr1_analysis: Dict[str, Any]) -> List[str]:
        """提取风险因素"""
        risks = []
        
        confidence = finr1_analysis.get("confidence", 0.7)
        if confidence < 0.6:
            risks.append("分析置信度较低")
        
        insights = finr1_analysis.get("insights", {})
        if not insights:
            risks.append("Fin-R1分析结果不完整")
        
        risks.append("新闻数据时效性风险")
        risks.append("情感分析模型局限性")
        
        return risks

    def _generate_market_outlook(self, finr1_analysis: Dict[str, Any]) -> str:
        """生成市场展望"""
        sentiment = finr1_analysis.get("overall_sentiment", 0.0)
        
        if sentiment > 0.3:
            return "国内新闻情绪积极，市场前景乐观"
        elif sentiment > 0.1:
            return "国内新闻情绪偏正面，市场谨慎乐观"
        elif sentiment < -0.3:
            return "国内新闻情绪消极，市场前景悲观"
        elif sentiment < -0.1:
            return "国内新闻情绪偏负面，市场谨慎悲观"
        else:
            return "国内新闻情绪中性，市场保持观望"

    def _assess_risk_level(self, sentiment: float) -> str:
        """评估风险等级"""
        if abs(sentiment) > 0.5:
            return "high"
        elif abs(sentiment) > 0.2:
            return "medium"
        else:
            return "low"

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
                reasoning=error_msg,
                risk_level="high",
                time_horizon="short"
            ),
            key_factors=["数据获取失败"],
            risk_factors=["数据源不可用", "分析置信度低"],
            market_outlook="无法获取有效新闻数据",
            additional_insights={"error": error_msg},
            data_sources=["国内新闻"],
            analysis_duration=0.0
        )
