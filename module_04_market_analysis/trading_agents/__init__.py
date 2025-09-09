"""
TradingAgents 多智能体交易框架
融合GitHub TradingAgents项目，适配国内市场和Fin-R1模型
"""

from .base_agent import BaseAgent, AgentAnalysis, AgentRecommendation
from .fundamental_analyst import FundamentalAnalyst
from .technical_analyst import TechnicalAnalyst
from .news_analyst import DomesticNewsAnalyst
from .sentiment_analyst import SentimentAnalyst
from .risk_manager import RiskManager
from .debate_engine import DebateEngine, DebateResult
from .consensus_builder import ConsensusBuilder
from .agent_coordinator import AgentCoordinator

__all__ = [
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
