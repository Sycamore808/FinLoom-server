"""
市场分析API接口
提供多智能体市场分析的REST API
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks
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

logger = setup_logger("market_analysis_api")

# 创建API路由器
router = APIRouter(prefix="/api/v1/market", tags=["market_analysis"])

# 全局智能体协调器
agent_coordinator: Optional[AgentCoordinator] = None


class AnalysisRequest(BaseModel):
    """分析请求模型"""
    symbols: List[str] = Field(..., description="股票代码列表")
    market_data: Optional[Dict[str, Any]] = Field(None, description="市场数据")
    context: Optional[Dict[str, Any]] = Field(None, description="分析上下文")
    priority: str = Field("normal", description="优先级: low, normal, high")
    timeout: float = Field(300.0, description="超时时间（秒）")


class AnalysisResponse(BaseModel):
    """分析响应模型"""
    request_id: str
    symbols: List[str]
    consensus_recommendation: str
    consensus_confidence: float
    consensus_reasoning: str
    key_insights: List[str]
    risk_assessment: Dict[str, Any]
    individual_analyses: List[Dict[str, Any]]
    debate_result: Optional[Dict[str, Any]]
    execution_time: float
    timestamp: str
    status: str


class AgentStatusResponse(BaseModel):
    """智能体状态响应模型"""
    agents: List[Dict[str, Any]]
    coordinator_stats: Dict[str, Any]
    health_status: Dict[str, Any]


async def initialize_agents():
    """初始化智能体"""
    global agent_coordinator
    
    if agent_coordinator is None:
        logger.info("Initializing trading agents...")
        
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
        
        logger.info(f"Initialized {len(agents)} trading agents")


@router.on_event("startup")
async def startup_event():
    """启动事件"""
    await initialize_agents()


@router.post("/analysis/agents", response_model=AnalysisResponse)
async def analyze_with_agents(request: AnalysisRequest):
    """使用多智能体进行市场分析"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        logger.info(f"Starting multi-agent analysis for symbols: {request.symbols}")
        
        # 执行分析
        result = await agent_coordinator.analyze_market(
            symbols=request.symbols,
            market_data=request.market_data,
            context=request.context,
            priority=request.priority,
            timeout=request.timeout
        )
        
        # 转换响应格式
        response = AnalysisResponse(
            request_id=result.request_id,
            symbols=result.symbols,
            consensus_recommendation=result.consensus_recommendation.value,
            consensus_confidence=result.consensus_confidence,
            consensus_reasoning=result.consensus_reasoning,
            key_insights=result.key_insights,
            risk_assessment=result.risk_assessment,
            individual_analyses=[
                {
                    "agent_name": analysis.agent_name,
                    "agent_type": analysis.agent_type,
                    "recommendation": analysis.recommendation.recommendation_type.value,
                    "confidence": analysis.recommendation.confidence,
                    "reasoning": analysis.recommendation.reasoning,
                    "key_factors": analysis.key_factors,
                    "risk_factors": analysis.risk_factors,
                    "timestamp": analysis.timestamp.isoformat()
                }
                for analysis in result.individual_analyses
            ],
            debate_result={
                "consensus_score": result.debate_result.consensus_score,
                "rounds_completed": len(result.debate_result.rounds),
                "final_consensus": result.debate_result.final_consensus,
                "key_insights": result.debate_result.key_insights,
                "remaining_disagreements": result.debate_result.remaining_disagreements
            } if result.debate_result else None,
            execution_time=result.execution_time,
            timestamp=result.timestamp.isoformat(),
            status=result.status
        )
        
        logger.info(f"Multi-agent analysis completed for {request.symbols}")
        return response
        
    except Exception as e:
        logger.error(f"Multi-agent analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/status", response_model=AgentStatusResponse)
async def get_agents_status():
    """获取智能体状态"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        # 获取智能体状态
        agent_status = agent_coordinator.get_agent_status()
        
        # 获取性能统计
        performance_stats = agent_coordinator.get_performance_stats()
        
        # 健康检查
        health_status = await agent_coordinator.health_check()
        
        response = AgentStatusResponse(
            agents=agent_status,
            coordinator_stats=performance_stats,
            health_status=health_status
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get agents status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/history")
async def get_analysis_history(limit: int = 10):
    """获取分析历史"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        history = agent_coordinator.get_analysis_history(limit)
        return {"history": history}
        
    except Exception as e:
        logger.error(f"Failed to get analysis history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/active")
async def get_active_requests():
    """获取活跃请求"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        active_requests = agent_coordinator.get_active_requests()
        return {"active_requests": active_requests}
        
    except Exception as e:
        logger.error(f"Failed to get active requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_name}/analyze")
async def analyze_with_single_agent(agent_name: str, request: AnalysisRequest):
    """使用单个智能体进行分析"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        # 获取指定智能体
        agent = agent_coordinator.get_agent(agent_name)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        # 执行分析
        analysis = await agent.analyze(
            symbols=request.symbols,
            market_data=request.market_data,
            context=request.context
        )
        
        # 转换响应格式
        response = {
            "agent_name": analysis.agent_name,
            "agent_type": analysis.agent_type,
            "symbols": analysis.symbols,
            "recommendation": {
                "type": analysis.recommendation.recommendation_type.value,
                "confidence": analysis.recommendation.confidence,
                "reasoning": analysis.recommendation.reasoning,
                "target_price": analysis.recommendation.target_price,
                "stop_loss": analysis.recommendation.stop_loss,
                "take_profit": analysis.recommendation.take_profit,
                "risk_level": analysis.recommendation.risk_level,
                "time_horizon": analysis.recommendation.time_horizon
            },
            "key_factors": analysis.key_factors,
            "risk_factors": analysis.risk_factors,
            "market_outlook": analysis.market_outlook,
            "additional_insights": analysis.additional_insights,
            "timestamp": analysis.timestamp.isoformat(),
            "execution_time": analysis.analysis_duration
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Single agent analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_name}/performance")
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
        
        return {
            "agent_name": agent_name,
            "performance_metrics": performance
        }
        
    except Exception as e:
        logger.error(f"Failed to get agent performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_name}/health-check")
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
            "last_analysis": agent.last_analysis_time.isoformat() if agent.last_analysis_time else None
        }
        
    except Exception as e:
        logger.error(f"Failed to check agent health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/consensus/history")
async def get_consensus_history(limit: int = 10):
    """获取共识历史"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        # 获取共识历史
        consensus_history = agent_coordinator.consensus_builder.get_consensus_history(limit)
        
        return {"consensus_history": consensus_history}
        
    except Exception as e:
        logger.error(f"Failed to get consensus history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debate/history")
async def get_debate_history(limit: int = 10):
    """获取辩论历史"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        # 获取辩论历史
        debate_history = agent_coordinator.debate_engine.get_debate_history(limit)
        
        return {"debate_history": debate_history}
        
    except Exception as e:
        logger.error(f"Failed to get debate history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debate/active")
async def get_active_debates():
    """获取活跃辩论"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        # 获取活跃辩论
        active_debates = agent_coordinator.debate_engine.get_active_debates()
        
        return {"active_debates": active_debates}
        
    except Exception as e:
        logger.error(f"Failed to get active debates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debate/statistics")
async def get_debate_statistics():
    """获取辩论统计"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        # 获取辩论统计
        debate_stats = agent_coordinator.debate_engine.get_debate_statistics()
        
        return {"debate_statistics": debate_stats}
        
    except Exception as e:
        logger.error(f"Failed to get debate statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
