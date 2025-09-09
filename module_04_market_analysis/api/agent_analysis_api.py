"""
智能体分析API接口
提供单个智能体分析的REST API
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..trading_agents import (
    AgentCoordinator,
    FundamentalAnalyst,
    TechnicalAnalyst,
    DomesticNewsAnalyst,
    SentimentAnalyst,
    RiskManager
)
from common.logging_system import setup_logger

logger = setup_logger("agent_analysis_api")

# 创建API路由器
router = APIRouter(prefix="/api/v1/agents", tags=["agent_analysis"])

# 全局智能体协调器
agent_coordinator: Optional[AgentCoordinator] = None


class SingleAgentAnalysisRequest(BaseModel):
    """单智能体分析请求模型"""
    symbols: List[str] = Field(..., description="股票代码列表")
    market_data: Optional[Dict[str, Any]] = Field(None, description="市场数据")
    context: Optional[Dict[str, Any]] = Field(None, description="分析上下文")
    timeout: float = Field(60.0, description="超时时间（秒）")


class SingleAgentAnalysisResponse(BaseModel):
    """单智能体分析响应模型"""
    agent_name: str
    agent_type: str
    symbols: List[str]
    recommendation: Dict[str, Any]
    key_factors: List[str]
    risk_factors: List[str]
    market_outlook: str
    additional_insights: Dict[str, Any]
    timestamp: str
    execution_time: float
    confidence: float


async def initialize_agents():
    """初始化智能体"""
    global agent_coordinator
    
    if agent_coordinator is None:
        logger.info("Initializing trading agents for single analysis...")
        
        # 创建智能体
        agents = [
            FundamentalAnalyst(),
            TechnicalAnalyst(),
            DomesticNewsAnalyst(),
            SentimentAnalyst(),
            RiskManager()
        ]
        
        # 创建协调器
        agent_coordinator = AgentCoordinator(agents=agents)
        
        logger.info(f"Initialized {len(agents)} trading agents for single analysis")


@router.post("/{agent_name}/analyze", response_model=SingleAgentAnalysisResponse)
async def analyze_with_single_agent(agent_name: str, request: SingleAgentAnalysisRequest):
    """使用单个智能体进行分析"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        # 获取指定智能体
        agent = agent_coordinator.get_agent(agent_name)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        logger.info(f"Starting single agent analysis with {agent_name} for symbols: {request.symbols}")
        
        # 执行分析
        start_time = datetime.now()
        analysis = await agent.analyze(
            symbols=request.symbols,
            market_data=request.market_data,
            context=request.context
        )
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 转换响应格式
        response = SingleAgentAnalysisResponse(
            agent_name=analysis.agent_name,
            agent_type=analysis.agent_type,
            symbols=analysis.symbols,
            recommendation={
                "type": analysis.recommendation.recommendation_type.value,
                "confidence": analysis.recommendation.confidence,
                "reasoning": analysis.recommendation.reasoning,
                "target_price": analysis.recommendation.target_price,
                "stop_loss": analysis.recommendation.stop_loss,
                "take_profit": analysis.recommendation.take_profit,
                "risk_level": analysis.recommendation.risk_level,
                "time_horizon": analysis.recommendation.time_horizon
            },
            key_factors=analysis.key_factors,
            risk_factors=analysis.risk_factors,
            market_outlook=analysis.market_outlook,
            additional_insights=analysis.additional_insights,
            timestamp=analysis.timestamp.isoformat(),
            execution_time=execution_time,
            confidence=analysis.recommendation.confidence
        )
        
        logger.info(f"Single agent analysis completed with {agent_name} in {execution_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Single agent analysis failed with {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_name}/status")
async def get_agent_status(agent_name: str):
    """获取智能体状态"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        # 获取指定智能体
        agent = agent_coordinator.get_agent(agent_name)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        # 获取状态信息
        is_ready = agent.is_ready_for_analysis()
        confidence = agent.get_confidence_score()
        performance = agent.get_performance_metrics()
        
        return {
            "agent_name": agent_name,
            "is_ready": is_ready,
            "confidence_score": confidence,
            "is_active": agent.is_active,
            "last_analysis": agent.last_analysis_time.isoformat() if agent.last_analysis_time else None,
            "performance_metrics": performance,
            "expertise_areas": agent.get_expertise_areas()
        }
        
    except Exception as e:
        logger.error(f"Failed to get agent status for {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_name}/performance")
async def get_agent_performance(agent_name: str):
    """获取智能体性能指标"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        # 获取指定智能体
        agent = agent_coordinator.get_agent(agent_name)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        # 获取性能指标
        performance = agent.get_performance_metrics()
        analysis_summary = agent.get_analysis_summary(limit=10)
        
        return {
            "agent_name": agent_name,
            "performance_metrics": performance,
            "recent_analyses": analysis_summary
        }
        
    except Exception as e:
        logger.error(f"Failed to get agent performance for {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_name}/health-check")
async def check_agent_health(agent_name: str):
    """检查智能体健康状态"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        # 获取指定智能体
        agent = agent_coordinator.get_agent(agent_name)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        # 健康检查
        is_ready = agent.is_ready_for_analysis()
        confidence = agent.get_confidence_score()
        
        return {
            "agent_name": agent_name,
            "is_ready": is_ready,
            "confidence_score": confidence,
            "is_active": agent.is_active,
            "last_analysis": agent.last_analysis_time.isoformat() if agent.last_analysis_time else None,
            "health_status": "healthy" if is_ready and confidence > 0.5 else "degraded"
        }
        
    except Exception as e:
        logger.error(f"Failed to check agent health for {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available")
async def get_available_agents():
    """获取可用的智能体列表"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        # 获取可用智能体
        available_agents = agent_coordinator.get_available_agents()
        
        agents_info = []
        for agent in available_agents:
            agents_info.append({
                "name": agent.name,
                "type": agent.agent_type,
                "expertise": agent.expertise,
                "is_ready": agent.is_ready_for_analysis(),
                "confidence_score": agent.get_confidence_score(),
                "is_active": agent.is_active
            })
        
        return {
            "available_agents": agents_info,
            "total_count": len(available_agents)
        }
        
    except Exception as e:
        logger.error(f"Failed to get available agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def get_agent_types():
    """获取智能体类型信息"""
    return {
        "agent_types": [
            {
                "name": "FundamentalAnalyst",
                "type": "fundamental_analyst",
                "description": "基本面分析师，分析财务数据和估值",
                "expertise": ["财务分析", "估值分析", "行业分析", "宏观经济"]
            },
            {
                "name": "TechnicalAnalyst", 
                "type": "technical_analyst",
                "description": "技术分析师，分析价格图表和技术指标",
                "expertise": ["技术指标", "图表形态", "趋势分析", "支撑阻力"]
            },
            {
                "name": "DomesticNewsAnalyst",
                "type": "news_analyst", 
                "description": "国内新闻分析师，分析新闻情感和内容",
                "expertise": ["新闻分析", "情感分析", "Fin-R1模型", "内容理解"]
            },
            {
                "name": "SentimentAnalyst",
                "type": "sentiment_analyst",
                "description": "情绪分析师，分析市场情绪和投资者行为",
                "expertise": ["市场情绪", "社交媒体", "恐惧贪婪指数", "投资者行为"]
            },
            {
                "name": "RiskManager",
                "type": "risk_manager",
                "description": "风险管理师，评估各种风险因素",
                "expertise": ["市场风险", "信用风险", "流动性风险", "压力测试"]
            }
        ]
    }
