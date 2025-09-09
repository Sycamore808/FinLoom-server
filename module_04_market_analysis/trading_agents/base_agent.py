"""
智能体基类模块
定义所有智能体的基础接口和通用功能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import asyncio
import uuid

from common.logging_system import setup_logger

logger = setup_logger("base_agent")


class RecommendationType(Enum):
    """推荐类型枚举"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class AgentRecommendation:
    """智能体推荐"""
    recommendation_type: RecommendationType
    confidence: float  # 0-1之间
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "medium"  # low, medium, high
    time_horizon: str = "short"  # short, medium, long


@dataclass
class AgentAnalysis:
    """智能体分析结果"""
    agent_name: str
    agent_type: str
    analysis_id: str
    timestamp: datetime
    symbols: List[str]
    recommendation: AgentRecommendation
    key_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    market_outlook: str = ""
    additional_insights: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    analysis_duration: float = 0.0  # 分析耗时（秒）


@dataclass
class DebateContext:
    """辩论上下文"""
    topic: str
    symbols: List[str]
    market_conditions: Dict[str, Any]
    time_horizon: str
    risk_tolerance: str


@dataclass
class DebateResponse:
    """辩论响应"""
    agent_name: str
    position: str  # 支持、反对、中立
    arguments: List[str]
    evidence: Dict[str, Any]
    counter_arguments: List[str] = field(default_factory=list)
    updated_confidence: float = 0.0
    willingness_to_compromise: float = 0.5  # 0-1之间


class BaseAgent(ABC):
    """智能体基类"""
    
    def __init__(
        self, 
        name: str, 
        agent_type: str,
        expertise: str,
        confidence_threshold: float = 0.6
    ):
        """初始化智能体
        
        Args:
            name: 智能体名称
            agent_type: 智能体类型
            expertise: 专业领域
            confidence_threshold: 置信度阈值
        """
        self.name = name
        self.agent_type = agent_type
        self.expertise = expertise
        self.confidence_threshold = confidence_threshold
        self.analysis_history: List[AgentAnalysis] = []
        self.performance_metrics: Dict[str, float] = {}
        self.is_active = True
        self.last_analysis_time: Optional[datetime] = None
        
        logger.info(f"Initialized agent: {self.name} ({self.agent_type})")
    
    @abstractmethod
    async def analyze(
        self, 
        symbols: List[str], 
        market_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """分析市场数据并生成推荐
        
        Args:
            symbols: 股票代码列表
            market_data: 市场数据
            context: 分析上下文
            
        Returns:
            分析结果
        """
        pass
    
    @abstractmethod
    async def debate(
        self, 
        other_analyses: List[AgentAnalysis],
        debate_context: DebateContext
    ) -> DebateResponse:
        """与其他智能体进行辩论
        
        Args:
            other_analyses: 其他智能体的分析结果
            debate_context: 辩论上下文
            
        Returns:
            辩论响应
        """
        pass
    
    async def update_analysis(
        self, 
        symbols: List[str], 
        new_data: Dict[str, Any]
    ) -> AgentAnalysis:
        """更新分析结果
        
        Args:
            symbols: 股票代码列表
            new_data: 新数据
            
        Returns:
            更新的分析结果
        """
        try:
            # 获取最新分析
            analysis = await self.analyze(symbols, new_data)
            
            # 更新历史记录
            self.analysis_history.append(analysis)
            self.last_analysis_time = datetime.now()
            
            # 保持历史记录在合理范围内
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
            
            logger.info(f"Agent {self.name} updated analysis for {symbols}")
            return analysis
            
        except Exception as e:
            logger.error(f"Agent {self.name} failed to update analysis: {e}")
            raise
    
    def get_confidence_score(self) -> float:
        """获取智能体置信度分数
        
        Returns:
            置信度分数 (0-1)
        """
        if not self.analysis_history:
            return 0.0
        
        # 基于历史分析的平均置信度
        recent_analyses = self.analysis_history[-10:]  # 最近10次分析
        avg_confidence = sum(
            analysis.recommendation.confidence for analysis in recent_analyses
        ) / len(recent_analyses)
        
        return avg_confidence
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标
        
        Returns:
            性能指标字典
        """
        if not self.analysis_history:
            return {}
        
        # 计算各种性能指标
        total_analyses = len(self.analysis_history)
        avg_confidence = self.get_confidence_score()
        
        # 计算推荐准确性（简化版本）
        accuracy = self._calculate_accuracy()
        
        # 计算响应时间
        avg_response_time = self._calculate_avg_response_time()
        
        return {
            "total_analyses": total_analyses,
            "avg_confidence": avg_confidence,
            "accuracy": accuracy,
            "avg_response_time": avg_response_time,
            "last_analysis": self.last_analysis_time.isoformat() if self.last_analysis_time else None
        }
    
    def _calculate_accuracy(self) -> float:
        """计算推荐准确性（简化实现）
        
        Returns:
            准确性分数 (0-1)
        """
        # 这里应该根据实际的市场表现来计算准确性
        # 简化实现：基于置信度的一致性
        if len(self.analysis_history) < 5:
            return 0.5
        
        recent_analyses = self.analysis_history[-5:]
        confidences = [a.recommendation.confidence for a in recent_analyses]
        
        # 计算置信度的稳定性
        confidence_std = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
        stability = 1.0 / (1.0 + confidence_std)
        
        return min(stability, 0.9)  # 最高90%的准确性
    
    def _calculate_avg_response_time(self) -> float:
        """计算平均响应时间
        
        Returns:
            平均响应时间（秒）
        """
        if not self.analysis_history:
            return 0.0
        
        response_times = [a.analysis_duration for a in self.analysis_history if a.analysis_duration > 0]
        
        if not response_times:
            return 0.0
        
        return sum(response_times) / len(response_times)
    
    def is_ready_for_analysis(self) -> bool:
        """检查智能体是否准备好进行分析
        
        Returns:
            是否准备好
        """
        # 如果智能体是活跃的，就认为是准备好的
        # 对于新初始化的智能体，即使没有历史分析也应该可以进行分析
        return self.is_active
    
    def get_expertise_areas(self) -> List[str]:
        """获取专业领域列表
        
        Returns:
            专业领域列表
        """
        return self.expertise.split(", ")
    
    def can_analyze_symbol(self, symbol: str) -> bool:
        """检查是否能分析指定股票
        
        Args:
            symbol: 股票代码
            
        Returns:
            是否能分析
        """
        # 基础实现：所有智能体都能分析所有股票
        # 子类可以重写此方法实现更精确的过滤
        return True
    
    def get_analysis_summary(self, limit: int = 5) -> Dict[str, Any]:
        """获取分析摘要
        
        Args:
            limit: 返回的分析数量限制
            
        Returns:
            分析摘要
        """
        recent_analyses = self.analysis_history[-limit:] if self.analysis_history else []
        
        summary = {
            "agent_name": self.name,
            "agent_type": self.agent_type,
            "total_analyses": len(self.analysis_history),
            "recent_analyses": [
                {
                    "analysis_id": a.analysis_id,
                    "symbols": a.symbols,
                    "recommendation": a.recommendation.recommendation_type.value,
                    "confidence": a.recommendation.confidence,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in recent_analyses
            ],
            "performance_metrics": self.get_performance_metrics()
        }
        
        return summary
    
    def __str__(self) -> str:
        return f"{self.name} ({self.agent_type})"
    
    def __repr__(self) -> str:
        return f"BaseAgent(name='{self.name}', type='{self.agent_type}', expertise='{self.expertise}')"


# 便捷函数
def create_analysis_id() -> str:
    """创建分析ID"""
    return f"analysis_{uuid.uuid4().hex[:8]}"


def calculate_consensus_confidence(analyses: List[AgentAnalysis]) -> float:
    """计算共识置信度
    
    Args:
        analyses: 分析结果列表
        
    Returns:
        共识置信度 (0-1)
    """
    if not analyses:
        return 0.0
    
    # 基于各智能体置信度的加权平均
    total_confidence = sum(a.recommendation.confidence for a in analyses)
    avg_confidence = total_confidence / len(analyses)
    
    # 考虑一致性（推荐类型的一致性）
    recommendations = [a.recommendation.recommendation_type for a in analyses]
    unique_recommendations = set(recommendations)
    consistency = 1.0 - (len(unique_recommendations) - 1) / (len(RecommendationType) - 1)
    
    # 综合置信度
    consensus_confidence = avg_confidence * consistency
    
    return min(max(consensus_confidence, 0.0), 1.0)
