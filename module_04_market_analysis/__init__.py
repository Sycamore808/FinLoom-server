"""
市场分析模块初始化文件
融合TradingAgents多智能体框架
"""

from .correlation_analysis.correlation_analyzer import (
    CorrelationAnalyzer,
    CorrelationResult,
    analyze_market_correlation
)

from .trading_agents import (
    BaseAgent,
    AgentAnalysis,
    AgentRecommendation,
    FundamentalAnalyst,
    TechnicalAnalyst,
    DomesticNewsAnalyst,
    SentimentAnalyst,
    RiskManager,
    DebateEngine,
    DebateResult,
    ConsensusBuilder,
    AgentCoordinator
)

__all__ = [
    # 相关性分析
    "CorrelationAnalyzer",
    "CorrelationResult", 
    "analyze_market_correlation",
    
    # TradingAgents多智能体框架
    "BaseAgent",
    "AgentAnalysis",
    "AgentRecommendation",
    "FundamentalAnalyst",
    "TechnicalAnalyst",
    "DomesticNewsAnalyst",
    "SentimentAnalyst",
    "RiskManager",
    "DebateEngine",
    "DebateResult",
    "ConsensusBuilder",
    "AgentCoordinator"
]