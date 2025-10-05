"""
NLP处理器模块
负责自然语言处理相关功能
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import jieba
import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("nlp_processor")


@dataclass
class TextEntity:
    """文本实体数据结构"""

    entity_type: str  # 'MONEY', 'DATE', 'PERCENT', 'ORG', 'PRODUCT'
    value: str
    normalized_value: Any
    start_pos: int
    end_pos: int
    confidence: float


@dataclass
class SentimentResult:
    """情感分析结果"""

    sentiment: str  # 'positive', 'negative', 'neutral'
    score: float  # -1 到 1
    confidence: float
    aspects: Dict[str, float]  # 细分维度情感


class NLPProcessor:
    """NLP处理器类"""

    # 金融词典
    FINANCIAL_TERMS = {
        "股票": ["stock", "equity"],
        "债券": ["bond", "fixed_income"],
        "基金": ["fund", "mutual_fund"],
        "期货": ["futures", "derivative"],
        "期权": ["option"],
        "外汇": ["forex", "fx"],
        "商品": ["commodity"],
        "加密货币": ["crypto", "cryptocurrency"],
        "ETF": ["etf", "exchange_traded_fund"],
        "REIT": ["reit", "real_estate_investment_trust"],
    }

    # 情感词典
    SENTIMENT_WORDS = {
        "positive": [
            "看好",
            "上涨",
            "增长",
            "突破",
            "创新高",
            "强势",
            "利好",
            "买入",
            "增持",
            "推荐",
            "优秀",
            "领先",
            "成长",
            "盈利",
        ],
        "negative": [
            "看空",
            "下跌",
            "下降",
            "跌破",
            "创新低",
            "弱势",
            "利空",
            "卖出",
            "减持",
            "警惕",
            "风险",
            "亏损",
            "衰退",
            "下滑",
        ],
        "neutral": ["震荡", "横盘", "观望", "等待", "维持", "平稳", "正常"],
    }

    # 实体识别模式
    ENTITY_PATTERNS = {
        "MONEY": [
            r"(\d+(?:\.\d+)?)\s*(?:元|美元|人民币|块|万|亿|千|百)",
            r"￥\s*(\d+(?:\.\d+)?)",
            r"\$\s*(\d+(?:\.\d+)?)",
        ],
        "PERCENT": [
            r"(\d+(?:\.\d+)?)\s*%",
            r"百分之\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*个百分点",
        ],
        "DATE": [
            r"(\d{4})[年\-/](\d{1,2})[月\-/](\d{1,2})[日号]?",
            r"(\d{1,2})[月\-/](\d{1,2})[日号]",
            r"(今天|明天|昨天|前天|大前天)",
            r"(本周|下周|上周|本月|下月|上月|今年|明年|去年)",
            r"(\d+)\s*(天|周|月|年)前",
            r"最近\s*(\d+)\s*(天|周|月|年)",
        ],
        "ORG": [
            r"([\u4e00-\u9fa5]+(?:公司|集团|银行|证券|基金|保险|信托))",
            r"([A-Z][A-Za-z\s&]+(?:Inc|Corp|Ltd|LLC|Company|Bank|Securities))",
        ],
    }

    def __init__(self):
        """初始化NLP处理器"""
        self._init_jieba()
        self.tfidf_vectorizer = None

    def _init_jieba(self):
        """初始化jieba分词器"""
        # 添加金融词汇
        for term in self.FINANCIAL_TERMS.keys():
            jieba.add_word(term)

        # 添加情感词汇
        for words in self.SENTIMENT_WORDS.values():
            for word in words:
                jieba.add_word(word)

    def tokenize(self, text: str, use_pos_tagging: bool = False) -> List[str]:
        """分词

        Args:
            text: 输入文本
            use_pos_tagging: 是否使用词性标注

        Returns:
            分词结果
        """
        if use_pos_tagging:
            import jieba.posseg as pseg

            return [(word, flag) for word, flag in pseg.cut(text)]
        else:
            return list(jieba.cut(text))

    def extract_keywords(
        self, text: str, top_k: int = 10, method: str = "tfidf"
    ) -> List[Tuple[str, float]]:
        """提取关键词

        Args:
            text: 输入文本
            top_k: 返回前k个关键词
            method: 提取方法 ('tfidf' 或 'textrank')

        Returns:
            关键词及权重列表
        """
        if method == "tfidf":
            return jieba.analyse.extract_tags(text, topK=top_k, withWeight=True)
        elif method == "textrank":
            return jieba.analyse.textrank(text, topK=top_k, withWeight=True)
        else:
            raise ValueError(f"Unknown method: {method}")

    def extract_entities(self, text: str) -> List[TextEntity]:
        """提取实体

        Args:
            text: 输入文本

        Returns:
            实体列表
        """
        entities = []

        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    entity = TextEntity(
                        entity_type=entity_type,
                        value=match.group(0),
                        normalized_value=self._normalize_entity(entity_type, match),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.9,
                    )
                    entities.append(entity)

        # 去重和排序
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.start_pos)

        return entities

    def analyze_sentiment(self, text: str, domain: str = "finance") -> SentimentResult:
        """情感分析

        Args:
            text: 输入文本
            domain: 领域

        Returns:
            情感分析结果
        """
        # 分词
        words = self.tokenize(text)

        # 计算情感分数
        positive_score = 0
        negative_score = 0
        neutral_score = 0

        for word in words:
            if word in self.SENTIMENT_WORDS["positive"]:
                positive_score += 1
            elif word in self.SENTIMENT_WORDS["negative"]:
                negative_score += 1
            elif word in self.SENTIMENT_WORDS["neutral"]:
                neutral_score += 1

        total_sentiment_words = positive_score + negative_score + neutral_score

        if total_sentiment_words == 0:
            # 没有情感词，判定为中性
            return SentimentResult(
                sentiment="neutral", score=0.0, confidence=0.3, aspects={}
            )

        # 计算情感倾向
        if positive_score > negative_score * 1.5:
            sentiment = "positive"
            score = positive_score / total_sentiment_words
        elif negative_score > positive_score * 1.5:
            sentiment = "negative"
            score = -negative_score / total_sentiment_words
        else:
            sentiment = "neutral"
            score = (positive_score - negative_score) / total_sentiment_words

        confidence = min(1.0, total_sentiment_words / len(words) * 2)

        # 分析细分维度（简化版）
        aspects = self._analyze_aspects(text, words)

        return SentimentResult(
            sentiment=sentiment, score=score, confidence=confidence, aspects=aspects
        )

    def calculate_text_similarity(
        self, text1: str, text2: str, method: str = "cosine"
    ) -> float:
        """计算文本相似度

        Args:
            text1: 文本1
            text2: 文本2
            method: 相似度计算方法

        Returns:
            相似度分数（0-1）
        """
        if method == "cosine":
            return self._cosine_similarity(text1, text2)
        elif method == "jaccard":
            return self._jaccard_similarity(text1, text2)
        else:
            raise ValueError(f"Unknown method: {method}")

    def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """文本摘要

        Args:
            text: 输入文本
            max_sentences: 最大句子数

        Returns:
            摘要文本
        """
        # 分句
        sentences = self._split_sentences(text)

        if len(sentences) <= max_sentences:
            return text

        # 提取关键句（基于TextRank）
        keywords = self.extract_keywords(text, top_k=5, method="textrank")
        keyword_set = set([kw[0] for kw in keywords])

        # 对句子打分
        sentence_scores = []
        for sentence in sentences:
            words = set(self.tokenize(sentence))
            score = len(words.intersection(keyword_set))
            sentence_scores.append((sentence, score))

        # 选择得分最高的句子
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        selected_sentences = [s[0] for s in sentence_scores[:max_sentences]]

        # 按原始顺序排列
        summary_sentences = []
        for sentence in sentences:
            if sentence in selected_sentences:
                summary_sentences.append(sentence)

        return "。".join(summary_sentences) + "。"

    def classify_intent(
        self, text: str, intent_categories: List[str] = None
    ) -> Tuple[str, float]:
        """意图分类

        Args:
            text: 输入文本
            intent_categories: 意图类别列表

        Returns:
            (意图类别, 置信度)
        """
        if intent_categories is None:
            intent_categories = [
                "query_price",  # 查询价格
                "query_performance",  # 查询业绩
                "buy_signal",  # 买入信号
                "sell_signal",  # 卖出信号
                "risk_assessment",  # 风险评估
                "strategy_advice",  # 策略建议
                "market_analysis",  # 市场分析
                "education",  # 教育咨询
                "other",  # 其他
            ]

        # 简单的基于关键词的分类
        intent_keywords = {
            "query_price": ["价格", "多少钱", "股价", "现价", "价位"],
            "query_performance": ["收益", "业绩", "表现", "赚", "亏", "盈利"],
            "buy_signal": ["买", "买入", "建仓", "加仓", "抄底"],
            "sell_signal": ["卖", "卖出", "平仓", "减仓", "止损", "止盈"],
            "risk_assessment": ["风险", "安全", "稳定", "波动", "危险"],
            "strategy_advice": ["策略", "建议", "怎么办", "如何", "方案"],
            "market_analysis": ["市场", "行情", "趋势", "分析", "走势"],
            "education": ["什么是", "为什么", "如何", "怎么样", "解释"],
        }

        scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[intent] = score

        # 找出得分最高的意图
        if max(scores.values()) == 0:
            return "other", 0.3

        best_intent = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_intent] / 3)  # 简单的置信度计算

        return best_intent, confidence

    def _normalize_entity(self, entity_type: str, match: re.Match) -> Any:
        """归一化实体值

        Args:
            entity_type: 实体类型
            match: 正则匹配对象

        Returns:
            归一化后的值
        """
        if entity_type == "MONEY":
            # 提取数值
            text = match.group(0)
            value = float(re.search(r"\d+(?:\.\d+)?", text).group())

            # 转换单位
            if "万" in text:
                value *= 10000
            elif "亿" in text:
                value *= 100000000
            elif "千" in text or "k" in text.lower():
                value *= 1000
            elif "百" in text:
                value *= 100
            elif "$" in text:
                value *= 7  # 简单汇率转换

            return value

        elif entity_type == "PERCENT":
            text = match.group(0)
            value = float(re.search(r"\d+(?:\.\d+)?", text).group())
            return value / 100

        elif entity_type == "DATE":
            # 简单的日期归一化
            return match.group(0)

        else:
            return match.group(0)

    def _deduplicate_entities(self, entities: List[TextEntity]) -> List[TextEntity]:
        """去除重复实体

        Args:
            entities: 实体列表

        Returns:
            去重后的实体列表
        """
        unique_entities = []
        seen_spans = set()

        for entity in entities:
            span = (entity.start_pos, entity.end_pos)
            if span not in seen_spans:
                unique_entities.append(entity)
                seen_spans.add(span)

        return unique_entities

    def _analyze_aspects(self, text: str, words: List[str]) -> Dict[str, float]:
        """分析细分维度情感

        Args:
            text: 文本
            words: 分词结果

        Returns:
            各维度情感分数
        """
        aspects = {"risk": 0.0, "return": 0.0, "market": 0.0, "company": 0.0}

        # 风险相关
        risk_words = ["风险", "波动", "不确定", "危险", "安全"]
        aspects["risk"] = sum(1 for w in words if w in risk_words) / len(words)

        # 收益相关
        return_words = ["收益", "盈利", "赚钱", "回报", "利润"]
        aspects["return"] = sum(1 for w in words if w in return_words) / len(words)

        # 市场相关
        market_words = ["市场", "行情", "大盘", "指数", "板块"]
        aspects["market"] = sum(1 for w in words if w in market_words) / len(words)

        # 公司相关
        company_words = ["公司", "企业", "管理", "业务", "产品"]
        aspects["company"] = sum(1 for w in words if w in company_words) / len(words)

        return aspects

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """计算余弦相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数
        """
        # 分词
        words1 = set(self.tokenize(text1))
        words2 = set(self.tokenize(text2))

        # 构建词汇表
        vocab = list(words1.union(words2))

        # 构建向量
        vec1 = [1 if word in words1 else 0 for word in vocab]
        vec2 = [1 if word in words2 else 0 for word in vocab]

        # 计算余弦相似度
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """计算Jaccard相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数
        """
        words1 = set(self.tokenize(text1))
        words2 = set(self.tokenize(text2))

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)

    def _split_sentences(self, text: str) -> List[str]:
        """分句

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        # 使用多个分隔符分句
        sentences = re.split(r"[。！？\n]", text)
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences


# 模块级别函数
def create_nlp_processor() -> NLPProcessor:
    """创建NLP处理器实例

    Returns:
        NLP处理器实例
    """
    return NLPProcessor()


def process_user_input(text: str) -> Dict[str, Any]:
    """处理用户输入的便捷函数

    Args:
        text: 用户输入文本

    Returns:
        处理结果字典
    """
    processor = NLPProcessor()

    return {
        "tokens": processor.tokenize(text),
        "keywords": processor.extract_keywords(text, top_k=5),
        "entities": [e.__dict__ for e in processor.extract_entities(text)],
        "sentiment": processor.analyze_sentiment(text).__dict__,
        "intent": processor.classify_intent(text),
    }
