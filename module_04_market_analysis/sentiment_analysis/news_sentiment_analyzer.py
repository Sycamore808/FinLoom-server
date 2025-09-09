"""
新闻情感分析器模块
分析金融新闻的情感倾向
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common.constants import TIMEOUT_SECONDS
from common.exceptions import DataError
from common.logging_system import setup_logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = setup_logger("news_sentiment_analyzer")


@dataclass
class NewsArticle:
    """新闻文章数据结构"""

    article_id: str
    title: str
    content: str
    source: str
    timestamp: datetime
    url: str
    symbols: List[str]  # 相关股票代码
    metadata: Dict[str, Any]


@dataclass
class SentimentResult:
    """情感分析结果"""

    article_id: str
    overall_sentiment: float  # -1到1之间
    sentiment_label: str  # 'positive', 'neutral', 'negative'
    confidence: float  # 0到1之间
    entity_sentiments: Dict[str, float]  # 实体级别情感
    aspect_sentiments: Dict[str, float]  # 方面级别情感
    keywords: List[str]
    timestamp: datetime


@dataclass
class SentimentConfig:
    """情感分析配置"""

    model_name: str = "ProsusAI/finbert"
    batch_size: int = 16
    max_length: int = 512
    use_gpu: bool = True
    confidence_threshold: float = 0.7
    sentiment_smoothing_window: int = 5
    entity_recognition: bool = True
    aspect_extraction: bool = True
    keyword_extraction_count: int = 10


class NewsSentimentAnalyzer:
    """新闻情感分析器"""

    # 情感标签映射
    SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}

    # 金融实体类型
    FINANCIAL_ENTITIES = [
        "company",
        "stock",
        "index",
        "commodity",
        "currency",
        "bond",
        "executive",
        "regulator",
    ]

    # 金融方面词典
    FINANCIAL_ASPECTS = {
        "earnings": ["earnings", "revenue", "profit", "income", "eps"],
        "growth": ["growth", "expansion", "increase", "surge", "rise"],
        "risk": ["risk", "volatility", "uncertainty", "exposure", "threat"],
        "valuation": ["valuation", "price", "value", "worth", "premium"],
        "management": ["ceo", "management", "board", "executive", "leadership"],
        "product": ["product", "service", "launch", "innovation", "technology"],
        "market": ["market", "share", "competition", "industry", "sector"],
        "regulation": ["regulation", "compliance", "law", "policy", "government"],
    }

    def __init__(self, config: Optional[SentimentConfig] = None):
        """初始化新闻情感分析器

        Args:
            config: 情感分析配置
        """
        self.config = config or SentimentConfig()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        )

        # 尝试加载预训练模型，如果失败则使用离线模式
        self.model = None
        self.tokenizer = None
        self.offline_mode = False
        
        try:
            # 尝试加载预训练模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                local_files_only=False,
                trust_remote_code=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                local_files_only=False,
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            logger.info(
                f"NewsSentimentAnalyzer initialized with model: {self.config.model_name}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load model {self.config.model_name}: {str(e)}. "
                "Switching to offline mode with rule-based sentiment analysis."
            )
            self.offline_mode = True
            # 初始化简单的词汇表用于离线情感分析
            self._init_offline_sentiment_lexicon()

        # 缓存
        self.sentiment_cache: Dict[str, SentimentResult] = {}
        self.entity_cache: Dict[str, List[str]] = {}

    def _init_offline_sentiment_lexicon(self):
        """初始化离线情感分析词汇表"""
        # 正面情感词汇
        self.positive_words = {
            '上涨', '增长', '盈利', '利好', '突破', '强势', '看好', '推荐', '买入',
            '收益', '利润', '成功', '优秀', '良好', '积极', '乐观', '强劲', '稳定',
            '提升', '改善', '创新', '领先', '优势', '机会', '潜力', '发展', '进步'
        }
        
        # 负面情感词汇
        self.negative_words = {
            '下跌', '下降', '亏损', '利空', '跌破', '弱势', '看空', '卖出', '风险',
            '损失', '失败', '困难', '问题', '危机', '担忧', '悲观', '疲软', '波动',
            '恶化', '衰退', '落后', '劣势', '威胁', '挑战', '压力', '困难', '问题'
        }
        
        # 中性词汇
        self.neutral_words = {
            '维持', '保持', '稳定', '持平', '调整', '变化', '报告', '数据', '分析',
            '市场', '股票', '公司', '行业', '经济', '政策', '影响', '因素', '情况'
        }
        
        logger.info("Offline sentiment lexicon initialized")

    def analyze_news_sentiment(
        self, articles: List[NewsArticle], symbols: Optional[List[str]] = None
    ) -> Dict[str, List[SentimentResult]]:
        """分析新闻情感

        Args:
            articles: 新闻文章列表
            symbols: 关注的股票代码列表（可选）

        Returns:
            股票代码到情感结果列表的映射
        """
        logger.info(f"Analyzing sentiment for {len(articles)} articles")

        results_by_symbol: Dict[str, List[SentimentResult]] = defaultdict(list)

        # 批处理文章
        for i in range(0, len(articles), self.config.batch_size):
            batch = articles[i : i + self.config.batch_size]
            batch_results = self._analyze_batch(batch)

            # 按股票代码组织结果
            for article, result in zip(batch, batch_results):
                # 如果指定了symbols，只保留相关的
                if symbols:
                    relevant_symbols = [s for s in article.symbols if s in symbols]
                else:
                    relevant_symbols = article.symbols

                for symbol in relevant_symbols:
                    results_by_symbol[symbol].append(result)

        # 对每个股票的情感结果进行时间排序
        for symbol in results_by_symbol:
            results_by_symbol[symbol].sort(key=lambda x: x.timestamp)

        logger.info(
            f"Completed sentiment analysis for {len(results_by_symbol)} symbols"
        )
        return dict(results_by_symbol)

    def process_social_media_signals(
        self, posts: List[Dict[str, Any]], platform: str = "twitter"
    ) -> pd.DataFrame:
        """处理社交媒体信号

        Args:
            posts: 社交媒体帖子列表
            platform: 平台名称

        Returns:
            情感分析结果DataFrame
        """
        results = []

        for post in posts:
            # 提取文本内容
            text = post.get("text", "")
            if not text:
                continue

            # 清理文本
            cleaned_text = self._clean_social_media_text(text, platform)

            # 创建临时文章对象
            temp_article = NewsArticle(
                article_id=post.get("id", ""),
                title="",
                content=cleaned_text,
                source=platform,
                timestamp=post.get("timestamp", datetime.now()),
                url=post.get("url", ""),
                symbols=self._extract_tickers(cleaned_text),
                metadata=post,
            )

            # 分析情感
            sentiment = self._analyze_single_article(temp_article)

            results.append(
                {
                    "post_id": post.get("id"),
                    "platform": platform,
                    "timestamp": sentiment.timestamp,
                    "sentiment": sentiment.overall_sentiment,
                    "sentiment_label": sentiment.sentiment_label,
                    "confidence": sentiment.confidence,
                    "text": cleaned_text[:100],  # 保存前100字符
                    "symbols": temp_article.symbols,
                }
            )

        return pd.DataFrame(results)

    def extract_analyst_consensus(
        self, analyst_reports: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """提取分析师共识

        Args:
            analyst_reports: 分析师报告列表

        Returns:
            股票代码到共识指标的映射
        """
        consensus_by_symbol: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for report in analyst_reports:
            symbol = report.get("symbol")
            if not symbol:
                continue

            # 提取评级
            rating = report.get("rating", "")
            rating_score = self._convert_rating_to_score(rating)
            if rating_score is not None:
                consensus_by_symbol[symbol]["rating"].append(rating_score)

            # 提取目标价
            target_price = report.get("target_price")
            if target_price:
                consensus_by_symbol[symbol]["target_price"].append(float(target_price))

            # 分析报告文本情感
            report_text = report.get("summary", "") + " " + report.get("analysis", "")
            if report_text.strip():
                temp_article = NewsArticle(
                    article_id=f"report_{report.get('id', '')}",
                    title=report.get("title", ""),
                    content=report_text,
                    source="analyst_report",
                    timestamp=report.get("date", datetime.now()),
                    url="",
                    symbols=[symbol],
                    metadata=report,
                )

                sentiment = self._analyze_single_article(temp_article)
                consensus_by_symbol[symbol]["sentiment"].append(
                    sentiment.overall_sentiment
                )

        # 计算共识指标
        consensus_results = {}
        for symbol, metrics in consensus_by_symbol.items():
            consensus_results[symbol] = {}

            # 平均评级
            if metrics["rating"]:
                consensus_results[symbol]["avg_rating"] = np.mean(metrics["rating"])
                consensus_results[symbol]["rating_std"] = np.std(metrics["rating"])

            # 平均目标价
            if metrics["target_price"]:
                consensus_results[symbol]["avg_target_price"] = np.mean(
                    metrics["target_price"]
                )
                consensus_results[symbol]["target_price_std"] = np.std(
                    metrics["target_price"]
                )

            # 平均情感
            if metrics["sentiment"]:
                consensus_results[symbol]["avg_sentiment"] = np.mean(
                    metrics["sentiment"]
                )
                consensus_results[symbol]["sentiment_std"] = np.std(
                    metrics["sentiment"]
                )

        return consensus_results

    def calculate_sentiment_momentum(
        self, sentiment_history: pd.DataFrame, window: int = 20
    ) -> pd.Series:
        """计算情感动量

        Args:
            sentiment_history: 历史情感数据
            window: 窗口大小

        Returns:
            情感动量序列
        """
        if "sentiment" not in sentiment_history.columns:
            raise DataError("sentiment_history must contain 'sentiment' column")

        # 计算移动平均
        ma_short = sentiment_history["sentiment"].rolling(window=window // 2).mean()
        ma_long = sentiment_history["sentiment"].rolling(window=window).mean()

        # 计算动量
        momentum = ma_short - ma_long

        # 计算变化率
        sentiment_roc = sentiment_history["sentiment"].pct_change(periods=window)

        # 综合动量指标
        combined_momentum = 0.7 * momentum + 0.3 * sentiment_roc

        return combined_momentum

    def generate_sentiment_indicators(
        self, symbol: str, lookback_days: int = 30
    ) -> Dict[str, float]:
        """生成情感指标

        Args:
            symbol: 股票代码
            lookback_days: 回看天数

        Returns:
            情感指标字典
        """
        indicators = {}

        # 获取历史情感数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # 从缓存中获取相关情感结果
        symbol_sentiments = [
            result
            for article_id, result in self.sentiment_cache.items()
            if symbol in result.entity_sentiments
            and start_date <= result.timestamp <= end_date
        ]

        if not symbol_sentiments:
            logger.warning(f"No sentiment data found for {symbol}")
            return indicators

        # 按时间排序
        symbol_sentiments.sort(key=lambda x: x.timestamp)

        # 计算基本统计
        sentiments = [s.overall_sentiment for s in symbol_sentiments]
        indicators["mean_sentiment"] = np.mean(sentiments)
        indicators["std_sentiment"] = np.std(sentiments)
        indicators["median_sentiment"] = np.median(sentiments)

        # 计算趋势
        if len(sentiments) > 1:
            x = np.arange(len(sentiments))
            slope, intercept = np.polyfit(x, sentiments, 1)
            indicators["sentiment_trend"] = slope

        # 计算情感分布
        positive_count = sum(1 for s in sentiments if s > 0.2)
        negative_count = sum(1 for s in sentiments if s < -0.2)
        neutral_count = len(sentiments) - positive_count - negative_count

        indicators["positive_ratio"] = positive_count / len(sentiments)
        indicators["negative_ratio"] = negative_count / len(sentiments)
        indicators["neutral_ratio"] = neutral_count / len(sentiments)

        # 计算情感强度
        indicators["avg_positive_strength"] = (
            np.mean([s for s in sentiments if s > 0]) if positive_count > 0 else 0
        )
        indicators["avg_negative_strength"] = (
            np.mean([s for s in sentiments if s < 0]) if negative_count > 0 else 0
        )

        # 计算置信度
        confidences = [s.confidence for s in symbol_sentiments]
        indicators["avg_confidence"] = np.mean(confidences)

        # 计算方面情感
        aspect_sentiments = defaultdict(list)
        for sentiment_result in symbol_sentiments:
            for aspect, score in sentiment_result.aspect_sentiments.items():
                aspect_sentiments[aspect].append(score)

        for aspect, scores in aspect_sentiments.items():
            indicators[f"aspect_{aspect}_sentiment"] = np.mean(scores)

        return indicators

    def _analyze_batch(self, articles: List[NewsArticle]) -> List[SentimentResult]:
        """批量分析文章

        Args:
            articles: 文章列表

        Returns:
            情感结果列表
        """
        if self.offline_mode:
            return self._analyze_batch_offline(articles)
        
        results = []

        # 准备输入文本
        texts = []
        for article in articles:
            # 组合标题和内容
            text = f"{article.title} {article.content}"
            # 截断到最大长度
            if len(text) > self.config.max_length * 3:  # 粗略估计token长度
                text = text[: self.config.max_length * 3]
            texts.append(text)

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # 处理结果
        for i, article in enumerate(articles):
            pred = predictions[i].cpu().numpy()

            # 获取预测标签和置信度
            label_idx = np.argmax(pred)
            confidence = float(pred[label_idx])

            # 计算整体情感分数（-1到1）
            sentiment_score = float(pred[2] - pred[0])  # positive - negative

            # 提取实体情感
            entity_sentiments = {}
            if self.config.entity_recognition:
                entity_sentiments = self._extract_entity_sentiments(
                    article.content, sentiment_score
                )

            # 提取方面情感
            aspect_sentiments = {}
            if self.config.aspect_extraction:
                aspect_sentiments = self._extract_aspect_sentiments(
                    article.content, sentiment_score
                )

            # 提取关键词
            keywords = self._extract_keywords(
                article.content, self.config.keyword_extraction_count
            )

            result = SentimentResult(
                article_id=article.article_id,
                overall_sentiment=sentiment_score,
                sentiment_label=self.SENTIMENT_LABELS[label_idx],
                confidence=confidence,
                entity_sentiments=entity_sentiments,
                aspect_sentiments=aspect_sentiments,
                keywords=keywords,
                timestamp=article.timestamp,
            )

            results.append(result)

            # 缓存结果
            self.sentiment_cache[article.article_id] = result

        return results

    def _analyze_batch_offline(self, articles: List[NewsArticle]) -> List[SentimentResult]:
        """离线模式批量分析文章

        Args:
            articles: 文章列表

        Returns:
            情感结果列表
        """
        results = []
        
        for article in articles:
            # 组合标题和内容
            text = f"{article.title} {article.content}"
            
            # 使用基于词汇表的情感分析
            sentiment_score = self._calculate_offline_sentiment(text)
            
            # 确定情感标签
            if sentiment_score > 0.1:
                sentiment_label = "positive"
                confidence = min(0.8, abs(sentiment_score))
            elif sentiment_score < -0.1:
                sentiment_label = "negative"
                confidence = min(0.8, abs(sentiment_score))
            else:
                sentiment_label = "neutral"
                confidence = 0.6
            
            # 提取关键词
            keywords = self._extract_keywords_offline(text)
            
            # 创建结果
            result = SentimentResult(
                article_id=article.article_id,
                overall_sentiment=sentiment_score,
                sentiment_label=sentiment_label,
                confidence=confidence,
                entity_sentiments={},  # 离线模式暂不支持实体情感
                aspect_sentiments={},  # 离线模式暂不支持方面情感
                keywords=keywords,
                timestamp=article.timestamp,
            )
            
            results.append(result)
            
            # 缓存结果
            self.sentiment_cache[article.article_id] = result
        
        return results

    def _calculate_offline_sentiment(self, text: str) -> float:
        """计算离线情感分数
        
        Args:
            text: 输入文本
            
        Returns:
            情感分数 (-1到1之间)
        """
        # 简单的基于词汇表的情感分析
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)
        neutral_count = sum(1 for word in self.neutral_words if word in text)
        
        total_words = positive_count + negative_count + neutral_count
        
        if total_words == 0:
            return 0.0
        
        # 计算情感分数
        sentiment_score = (positive_count - negative_count) / total_words
        
        # 限制在-1到1之间
        return max(-1.0, min(1.0, sentiment_score))
    
    def _extract_keywords_offline(self, text: str) -> List[str]:
        """离线模式提取关键词
        
        Args:
            text: 输入文本
            
        Returns:
            关键词列表
        """
        # 简单的关键词提取，基于情感词汇
        keywords = []
        
        # 添加情感词汇作为关键词
        for word in self.positive_words.union(self.negative_words).union(self.neutral_words):
            if word in text:
                keywords.append(word)
        
        # 限制关键词数量
        return keywords[:10]

    def _analyze_single_article(self, article: NewsArticle) -> SentimentResult:
        """分析单篇文章

        Args:
            article: 文章对象

        Returns:
            情感结果
        """
        return self._analyze_batch([article])[0]

    def _clean_social_media_text(self, text: str, platform: str) -> str:
        """清理社交媒体文本

        Args:
            text: 原始文本
            platform: 平台名称

        Returns:
            清理后的文本
        """
        # 移除URLs
        text = re.sub(r"http\S+|www.\S+", "", text)

        # 移除@mentions (Twitter)
        if platform == "twitter":
            text = re.sub(r"@\w+", "", text)

        # 移除hashtags但保留文本
        text = re.sub(r"#(\w+)", r"\1", text)

        # 移除多余空格
        text = " ".join(text.split())

        return text.strip()

    def _extract_tickers(self, text: str) -> List[str]:
        """从文本中提取股票代码

        Args:
            text: 文本内容

        Returns:
            股票代码列表
        """
        # 简单的ticker提取规则
        # 匹配$符号后的大写字母
        dollar_tickers = re.findall(r"\$([A-Z]{1,5})\b", text)

        # 匹配括号中的大写字母组合
        paren_tickers = re.findall(r"\(([A-Z]{1,5})\)", text)

        # 匹配独立的2-5个大写字母（可能是ticker）
        potential_tickers = re.findall(r"\b([A-Z]{2,5})\b", text)

        # 合并并去重
        all_tickers = list(set(dollar_tickers + paren_tickers + potential_tickers))

        # 过滤掉常见的非ticker词
        non_tickers = {"CEO", "CFO", "IPO", "USD", "EUR", "GBP", "API", "AI", "ML"}
        tickers = [t for t in all_tickers if t not in non_tickers]

        return tickers

    def _convert_rating_to_score(self, rating: str) -> Optional[float]:
        """将评级转换为数值分数

        Args:
            rating: 评级字符串

        Returns:
            数值分数（-1到1）
        """
        rating_map = {
            "strong buy": 1.0,
            "buy": 0.5,
            "outperform": 0.5,
            "hold": 0.0,
            "neutral": 0.0,
            "underperform": -0.5,
            "sell": -0.5,
            "strong sell": -1.0,
        }

        rating_lower = rating.lower().strip()
        return rating_map.get(rating_lower)

    def _extract_entity_sentiments(
        self, text: str, overall_sentiment: float
    ) -> Dict[str, float]:
        """提取实体级别情感

        Args:
            text: 文本内容
            overall_sentiment: 整体情感分数

        Returns:
            实体到情感分数的映射
        """
        entity_sentiments = {}

        # 简化的实体识别（实际应用中应使用NER模型）
        # 这里使用正则表达式识别可能的公司名称
        potential_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)

        for entity in potential_entities[:10]:  # 限制实体数量
            # 简单地将整体情感赋给每个实体
            # 实际应用中应该分析实体周围的上下文
            entity_sentiments[entity] = overall_sentiment

        return entity_sentiments

    def _extract_aspect_sentiments(
        self, text: str, overall_sentiment: float
    ) -> Dict[str, float]:
        """提取方面级别情感

        Args:
            text: 文本内容
            overall_sentiment: 整体情感分数

        Returns:
            方面到情感分数的映射
        """
        aspect_sentiments = {}
        text_lower = text.lower()

        for aspect, keywords in self.FINANCIAL_ASPECTS.items():
            # 检查是否包含该方面的关键词
            if any(keyword in text_lower for keyword in keywords):
                # 简化处理：使用整体情感
                # 实际应用中应该分析方面相关的句子
                aspect_sentiments[aspect] = overall_sentiment

        return aspect_sentiments

    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键词

        Args:
            text: 文本内容
            top_k: 返回的关键词数量

        Returns:
            关键词列表
        """
        # 简单的关键词提取（基于词频）
        # 实际应用中应使用TF-IDF或其他更复杂的方法

        # 分词并转小写
        words = re.findall(r"\b[a-z]+\b", text.lower())

        # 过滤停用词
        stop_words = {
            "the",
            "is",
            "at",
            "which",
            "on",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "with",
            "to",
            "for",
            "of",
            "as",
            "from",
            "by",
        }
        words = [w for w in words if w not in stop_words and len(w) > 3]

        # 统计词频
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1

        # 返回高频词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]


# 模块级别函数
def analyze_market_sentiment(
    news_articles: List[Dict[str, Any]], config: Optional[SentimentConfig] = None
) -> pd.DataFrame:
    """分析市场情感的便捷函数

    Args:
        news_articles: 新闻文章列表（字典格式）
        config: 情感分析配置

    Returns:
        情感分析结果DataFrame
    """
    # 转换为NewsArticle对象
    articles = []
    for article_dict in news_articles:
        article = NewsArticle(
            article_id=article_dict.get("id", ""),
            title=article_dict.get("title", ""),
            content=article_dict.get("content", ""),
            source=article_dict.get("source", ""),
            timestamp=article_dict.get("timestamp", datetime.now()),
            url=article_dict.get("url", ""),
            symbols=article_dict.get("symbols", []),
            metadata=article_dict,
        )
        articles.append(article)

    # 创建分析器并分析
    analyzer = NewsSentimentAnalyzer(config)
    results = analyzer.analyze_news_sentiment(articles)

    # 转换为DataFrame
    data = []
    for symbol, sentiment_results in results.items():
        for result in sentiment_results:
            data.append(
                {
                    "symbol": symbol,
                    "article_id": result.article_id,
                    "timestamp": result.timestamp,
                    "sentiment": result.overall_sentiment,
                    "sentiment_label": result.sentiment_label,
                    "confidence": result.confidence,
                }
            )

    return pd.DataFrame(data)
