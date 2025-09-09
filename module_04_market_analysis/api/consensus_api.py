"""
共识构建API接口
提供多智能体共识和辩论的REST API
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..trading_agents import (
    AgentCoordinator,
    DebateEngine,
    ConsensusBuilder
)
from common.logging_system import setup_logger

logger = setup_logger("consensus_api")

# 创建API路由器
router = APIRouter(prefix="/api/v1/consensus", tags=["consensus"])

# 全局智能体协调器
agent_coordinator: Optional[AgentCoordinator] = None


class ConsensusRequest(BaseModel):
    """共识请求模型"""
    symbols: List[str] = Field(..., description="股票代码列表")
    market_data: Optional[Dict[str, Any]] = Field(None, description="市场数据")
    context: Optional[Dict[str, Any]] = Field(None, description="分析上下文")
    enable_debate: bool = Field(True, description="是否启用辩论")
    max_debate_rounds: int = Field(3, description="最大辩论轮数")
    consensus_threshold: float = Field(0.7, description="共识阈值")


class ConsensusResponse(BaseModel):
    """共识响应模型"""
    consensus_id: str
    symbols: List[str]
    final_recommendation: str
    consensus_confidence: float
    consensus_reasoning: str
    key_insights: List[str]
    risk_factors: List[str]
    supporting_evidence: Dict[str, Any]
    disagreement_areas: List[str]
    consensus_score: float
    participant_count: int
    debate_rounds: int
    timestamp: str
    execution_time: float


class DebateRequest(BaseModel):
    """辩论请求模型"""
    symbols: List[str] = Field(..., description="股票代码列表")
    market_data: Optional[Dict[str, Any]] = Field(None, description="市场数据")
    context: Optional[Dict[str, Any]] = Field(None, description="分析上下文")
    max_rounds: int = Field(3, description="最大辩论轮数")
    consensus_threshold: float = Field(0.7, description="共识阈值")


class DebateResponse(BaseModel):
    """辩论响应模型"""
    debate_id: str
    symbols: List[str]
    rounds: List[Dict[str, Any]]
    final_consensus: Dict[str, Any]
    consensus_score: float
    final_recommendation: str
    confidence: float
    reasoning: str
    key_insights: List[str]
    remaining_disagreements: List[str]
    debate_duration: float
    timestamp: str


async def initialize_agents():
    """初始化智能体"""
    global agent_coordinator
    
    if agent_coordinator is None:
        logger.info("Initializing trading agents for consensus...")
        
        # 创建智能体
        from ..trading_agents import (
            FundamentalAnalyst,
            TechnicalAnalyst,
            DomesticNewsAnalyst,
            SentimentAnalyst,
            RiskManager
        )
        
        agents = [
            FundamentalAnalyst(),
            TechnicalAnalyst(),
            DomesticNewsAnalyst(),
            SentimentAnalyst(),
            RiskManager()
        ]
        
        # 创建协调器
        agent_coordinator = AgentCoordinator(agents=agents)
        
        logger.info(f"Initialized {len(agents)} trading agents for consensus")


@router.post("/build", response_model=ConsensusResponse)
async def build_consensus(request: ConsensusRequest):
    """构建多智能体共识"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        logger.info(f"Building consensus for symbols: {request.symbols}")
        
        start_time = datetime.now()
        
        # 1. 获取可用智能体
        available_agents = agent_coordinator.get_available_agents()
        if len(available_agents) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 agents for consensus")
        
        # 2. 执行各智能体分析
        individual_analyses = []
        for agent in available_agents:
            try:
                analysis = await agent.analyze(
                    symbols=request.symbols,
                    market_data=request.market_data,
                    context=request.context
                )
                individual_analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Agent {agent.name} analysis failed: {e}")
                continue
        
        if not individual_analyses:
            raise HTTPException(status_code=500, detail="No successful agent analyses")
        
        # 3. 构建共识
        consensus_result = agent_coordinator.consensus_builder.build_consensus(individual_analyses)
        
        # 4. 如果启用辩论，进行辩论
        debate_rounds = 0
        if request.enable_debate and len(individual_analyses) > 1:
            from ..trading_agents import DebateContext
            
            debate_context = DebateContext(
                topic=f"Analysis consensus for {', '.join(request.symbols)}",
                symbols=request.symbols,
                market_conditions=request.market_data or {},
                time_horizon=request.context.get("time_horizon", "short") if request.context else "short",
                risk_tolerance=request.context.get("risk_tolerance", "medium") if request.context else "medium"
            )
            
            debate_result = await agent_coordinator.debate_engine.conduct_debate(
                available_agents, individual_analyses, debate_context
            )
            debate_rounds = len(debate_result.rounds)
            
            # 更新共识结果
            if debate_result.consensus_score > consensus_result.get("consensus_score", 0):
                consensus_result.update({
                    "recommendation": debate_result.final_recommendation,
                    "confidence": debate_result.confidence,
                    "reasoning": debate_result.reasoning,
                    "key_insights": debate_result.key_insights,
                    "consensus_score": debate_result.consensus_score
                })
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 5. 构建响应
        response = ConsensusResponse(
            consensus_id=f"consensus_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbols=request.symbols,
            final_recommendation=consensus_result.get("recommendation", "hold"),
            consensus_confidence=consensus_result.get("confidence", 0.5),
            consensus_reasoning=consensus_result.get("reasoning", "基于多智能体分析"),
            key_insights=consensus_result.get("key_insights", []),
            risk_factors=consensus_result.get("risk_factors", []),
            supporting_evidence=consensus_result.get("supporting_evidence", {}),
            disagreement_areas=consensus_result.get("disagreement_areas", []),
            consensus_score=consensus_result.get("consensus_score", 0.5),
            participant_count=len(individual_analyses),
            debate_rounds=debate_rounds,
            timestamp=datetime.now().isoformat(),
            execution_time=execution_time
        )
        
        logger.info(f"Consensus built for {request.symbols} in {execution_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Consensus building failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/debate", response_model=DebateResponse)
async def conduct_debate(request: DebateRequest):
    """进行多智能体辩论"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        logger.info(f"Conducting debate for symbols: {request.symbols}")
        
        start_time = datetime.now()
        
        # 1. 获取可用智能体
        available_agents = agent_coordinator.get_available_agents()
        if len(available_agents) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 agents for debate")
        
        # 2. 执行各智能体分析
        individual_analyses = []
        for agent in available_agents:
            try:
                analysis = await agent.analyze(
                    symbols=request.symbols,
                    market_data=request.market_data,
                    context=request.context
                )
                individual_analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Agent {agent.name} analysis failed: {e}")
                continue
        
        if not individual_analyses:
            raise HTTPException(status_code=500, detail="No successful agent analyses")
        
        # 3. 进行辩论
        from ..trading_agents import DebateContext
        
        debate_context = DebateContext(
            topic=f"Debate for {', '.join(request.symbols)}",
            symbols=request.symbols,
            market_conditions=request.market_data or {},
            time_horizon=request.context.get("time_horizon", "short") if request.context else "short",
            risk_tolerance=request.context.get("risk_tolerance", "medium") if request.context else "medium"
        )
        
        debate_result = await agent_coordinator.debate_engine.conduct_debate(
            available_agents, individual_analyses, debate_context
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 4. 构建响应
        response = DebateResponse(
            debate_id=debate_result.debate_id,
            symbols=request.symbols,
            rounds=[
                {
                    "round_number": round.round_number,
                    "timestamp": round.timestamp.isoformat(),
                    "participants": round.participants,
                    "consensus_score": round.consensus_score,
                    "key_arguments": round.key_arguments,
                    "evidence_summary": round.evidence_summary
                }
                for round in debate_result.rounds
            ],
            final_consensus=debate_result.final_consensus,
            consensus_score=debate_result.consensus_score,
            final_recommendation=debate_result.final_recommendation.value,
            confidence=debate_result.confidence,
            reasoning=debate_result.reasoning,
            key_insights=debate_result.key_insights,
            remaining_disagreements=debate_result.remaining_disagreements,
            debate_duration=execution_time,
            timestamp=debate_result.timestamp.isoformat()
        )
        
        logger.info(f"Debate completed for {request.symbols} in {execution_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Debate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
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


@router.get("/statistics")
async def get_consensus_statistics():
    """获取共识统计"""
    try:
        if agent_coordinator is None:
            await initialize_agents()
        
        # 获取共识统计
        consensus_stats = agent_coordinator.consensus_builder.get_consensus_statistics()
        
        return {"consensus_statistics": consensus_stats}
        
    except Exception as e:
        logger.error(f"Failed to get consensus statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
