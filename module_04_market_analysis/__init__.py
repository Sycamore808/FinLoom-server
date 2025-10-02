"""
市场分析模块初始化文件
增强版 - 集成Trading Agents功能到各个子模块
"""

from .correlation_analysis.correlation_analyzer import (
    CorrelationAnalyzer,
    CorrelationResult,
    analyze_market_correlation,
)

# Enhanced sentiment analysis with integrated trading agents functionality
try:
    from .sentiment_analysis.fin_r1_sentiment import (
        analyze_market_sentiment,
        analyze_symbol_sentiment,
        get_sentiment_analyzer,
    )
except ImportError:
    get_sentiment_analyzer = None
    analyze_symbol_sentiment = None
    analyze_market_sentiment = None

# Enhanced news sentiment analysis
try:
    from .sentiment_analysis.enhanced_news_sentiment import (
        EnhancedNewsSentimentAnalyzer,
    )
except ImportError:
    EnhancedNewsSentimentAnalyzer = None

# Storage management
from .storage_management.market_analysis_database import get_market_analysis_db

__all__ = [
    # 相关性分析
    "CorrelationAnalyzer",
    "CorrelationResult",
    "analyze_market_correlation",
    # 增强的情感分析 (集成Trading Agents功能)
    "get_sentiment_analyzer",
    "analyze_symbol_sentiment",
    "analyze_market_sentiment",
    "EnhancedNewsSentimentAnalyzer",
    # 数据库管理
    "get_market_analysis_db",
]
