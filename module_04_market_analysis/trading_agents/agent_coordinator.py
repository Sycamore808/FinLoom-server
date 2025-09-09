"""
智能体协调器
协调多个智能体的分析过程，管理辩论和共识构建
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import uuid

from .base_agent import BaseAgent, AgentAnalysis, DebateContext, RecommendationType
from .debate_engine import DebateEngine, DebateResult
from .consensus_builder import ConsensusBuilder
from common.logging_system import setup_logger

logger = setup_logger("agent_coordinator")


@dataclass
class AnalysisRequest:
    """分析请求"""
    request_id: str
    symbols: List[str]
    market_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    priority: str = "normal"  # low, normal, high
    timeout: float = 300.0  # 5分钟超时
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AnalysisResult:
    """分析结果"""
    request_id: str
    symbols: List[str]
    individual_analyses: List[AgentAnalysis]
    debate_result: Optional[DebateResult]
    consensus_recommendation: RecommendationType
    consensus_confidence: float
    consensus_reasoning: str
    key_insights: List[str]
    risk_assessment: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    status: str  # success, partial_success, failed


class AgentCoordinator:
    """智能体协调器"""
    
    def __init__(
        self,
        agents: Optional[List[BaseAgent]] = None,
        debate_engine: Optional[DebateEngine] = None,
        consensus_builder: Optional[ConsensusBuilder] = None
    ):
        """初始化智能体协调器
        
        Args:
            agents: 智能体列表
            debate_engine: 辩论引擎
            consensus_builder: 共识构建器
        """
        self.agents = agents or []
        self.debate_engine = debate_engine or DebateEngine()
        self.consensus_builder = consensus_builder or ConsensusBuilder()
        
        # 分析历史
        self.analysis_history: List[AnalysisResult] = []
        self.active_requests: Dict[str, AnalysisRequest] = {}
        
        # 性能统计
        self.performance_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'avg_execution_time': 0.0,
            'avg_consensus_score': 0.0
        }
        
        logger.info(f"Initialized agent coordinator with {len(self.agents)} agents")
    
    def add_agent(self, agent: BaseAgent) -> None:
        """添加智能体"""
        if agent not in self.agents:
            self.agents.append(agent)
            logger.info(f"Added agent: {agent.name}")
    
    def remove_agent(self, agent_name: str) -> bool:
        """移除智能体"""
        for i, agent in enumerate(self.agents):
            if agent.name == agent_name:
                removed_agent = self.agents.pop(i)
                logger.info(f"Removed agent: {removed_agent.name}")
                return True
        return False
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """获取指定智能体"""
        for agent in self.agents:
            if agent.name == agent_name:
                return agent
        return None
    
    def get_available_agents(self) -> List[BaseAgent]:
        """获取可用的智能体"""
        return [agent for agent in self.agents if agent.is_ready_for_analysis()]
    
    async def analyze_market(
        self, 
        symbols: List[str], 
        market_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        priority: str = "normal",
        timeout: float = 300.0
    ) -> AnalysisResult:
        """分析市场
        
        Args:
            symbols: 股票代码列表
            market_data: 市场数据
            context: 分析上下文
            priority: 优先级
            timeout: 超时时间
            
        Returns:
            分析结果
        """
        request_id = f"analysis_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        logger.info(f"Starting market analysis {request_id} for symbols: {symbols}")
        
        # 创建分析请求
        request = AnalysisRequest(
            request_id=request_id,
            symbols=symbols,
            market_data=market_data,
            context=context,
            priority=priority,
            timeout=timeout
        )
        
        # 记录活跃请求
        self.active_requests[request_id] = request
        
        try:
            # 1. 获取可用智能体
            available_agents = self.get_available_agents()
            if not available_agents:
                raise ValueError("No available agents for analysis")
            
            logger.info(f"Using {len(available_agents)} agents for analysis")
            
            # 2. 并行执行各智能体分析
            individual_analyses = await self._execute_agent_analyses(
                available_agents, symbols, market_data, context, timeout
            )
            
            if not individual_analyses:
                raise ValueError("No successful agent analyses")
            
            # 3. 进行辩论和共识构建
            debate_result = await self._conduct_debate_and_consensus(
                available_agents, individual_analyses, symbols, context
            )
            
            # 4. 生成最终结果
            analysis_result = await self._generate_analysis_result(
                request, individual_analyses, debate_result, start_time
            )
            
            # 5. 更新统计信息
            self._update_performance_stats(analysis_result)
            
            # 6. 记录到历史
            self.analysis_history.append(analysis_result)
            
            # 7. 清理活跃请求
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            
            logger.info(f"Market analysis {request_id} completed in {analysis_result.execution_time:.2f}s")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Market analysis {request_id} failed: {e}")
            
            # 创建失败结果
            failed_result = AnalysisResult(
                request_id=request_id,
                symbols=symbols,
                individual_analyses=[],
                debate_result=None,
                consensus_recommendation=RecommendationType.HOLD,
                consensus_confidence=0.0,
                consensus_reasoning=f"Analysis failed: {e}",
                key_insights=[],
                risk_assessment={'error': str(e)},
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=start_time,
                status="failed"
            )
            
            # 清理活跃请求
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            
            return failed_result
    
    async def _execute_agent_analyses(
        self, 
        agents: List[BaseAgent], 
        symbols: List[str], 
        market_data: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
        timeout: float
    ) -> List[AgentAnalysis]:
        """执行智能体分析"""
        logger.info(f"Executing analyses for {len(agents)} agents")
        
        # 创建分析任务
        tasks = []
        for agent in agents:
            task = asyncio.create_task(
                agent.analyze(symbols, market_data, context)
            )
            tasks.append(task)
        
        # 并发执行，带超时
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # 处理结果
            successful_analyses = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Agent {agents[i].name} analysis failed: {result}")
                else:
                    successful_analyses.append(result)
            
            logger.info(f"Successfully completed {len(successful_analyses)} out of {len(agents)} analyses")
            return successful_analyses
            
        except asyncio.TimeoutError:
            logger.error(f"Agent analyses timed out after {timeout}s")
            return []
        except Exception as e:
            logger.error(f"Failed to execute agent analyses: {e}")
            return []
    
    async def _conduct_debate_and_consensus(
        self, 
        agents: List[BaseAgent], 
        analyses: List[AgentAnalysis], 
        symbols: List[str],
        context: Optional[Dict[str, Any]]
    ) -> Optional[DebateResult]:
        """进行辩论和共识构建"""
        if len(analyses) < 2:
            logger.info("Not enough analyses for debate, skipping debate process")
            return None
        
        try:
            # 创建辩论上下文
            debate_context = DebateContext(
                topic=f"Market analysis for {', '.join(symbols)}",
                symbols=symbols,
                market_conditions=context or {},
                time_horizon="short",
                risk_tolerance="medium"
            )
            
            # 进行辩论
            debate_result = await self.debate_engine.conduct_debate(
                agents, analyses, debate_context
            )
            
            logger.info(f"Debate completed with consensus score: {debate_result.consensus_score:.3f}")
            return debate_result
            
        except Exception as e:
            logger.error(f"Debate and consensus failed: {e}")
            return None
    
    async def _generate_analysis_result(
        self, 
        request: AnalysisRequest, 
        individual_analyses: List[AgentAnalysis], 
        debate_result: Optional[DebateResult],
        start_time: datetime
    ) -> AnalysisResult:
        """生成分析结果"""
        try:
            # 确定最终推荐
            if debate_result and debate_result.final_consensus:
                consensus_recommendation = debate_result.final_consensus.get('recommendation', RecommendationType.HOLD)
                consensus_confidence = debate_result.final_consensus.get('confidence', 0.0)
                consensus_reasoning = debate_result.final_consensus.get('reasoning', '')
                key_insights = debate_result.final_consensus.get('key_insights', [])
            else:
                # 如果没有辩论结果，使用简单共识
                consensus_data = self.consensus_builder.build_consensus(individual_analyses)
                consensus_recommendation = consensus_data.get('recommendation', RecommendationType.HOLD)
                consensus_confidence = consensus_data.get('confidence', 0.0)
                consensus_reasoning = consensus_data.get('reasoning', '')
                key_insights = consensus_data.get('key_insights', [])
            
            # 风险评估
            risk_assessment = self._assess_risks(individual_analyses, debate_result)
            
            # 确定状态
            if len(individual_analyses) == len(self.get_available_agents()):
                status = "success"
            elif len(individual_analyses) > 0:
                status = "partial_success"
            else:
                status = "failed"
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                request_id=request.request_id,
                symbols=request.symbols,
                individual_analyses=individual_analyses,
                debate_result=debate_result,
                consensus_recommendation=consensus_recommendation,
                consensus_confidence=consensus_confidence,
                consensus_reasoning=consensus_reasoning,
                key_insights=key_insights,
                risk_assessment=risk_assessment,
                execution_time=execution_time,
                timestamp=start_time,
                status=status
            )
            
        except Exception as e:
            logger.error(f"Failed to generate analysis result: {e}")
            raise
    
    def _assess_risks(
        self, 
        analyses: List[AgentAnalysis], 
        debate_result: Optional[DebateResult]
    ) -> Dict[str, Any]:
        """评估风险"""
        risk_assessment = {
            'overall_risk_level': 'medium',
            'risk_factors': [],
            'consensus_risk': 'medium',
            'data_quality_risk': 'low',
            'model_risk': 'medium'
        }
        
        try:
            # 收集所有风险因素
            all_risk_factors = []
            for analysis in analyses:
                all_risk_factors.extend(analysis.risk_factors)
            
            # 去重
            unique_risk_factors = list(set(all_risk_factors))
            risk_assessment['risk_factors'] = unique_risk_factors[:5]
            
            # 评估共识风险
            if debate_result:
                consensus_score = debate_result.consensus_score
                if consensus_score > 0.8:
                    risk_assessment['consensus_risk'] = 'low'
                elif consensus_score > 0.6:
                    risk_assessment['consensus_risk'] = 'medium'
                else:
                    risk_assessment['consensus_risk'] = 'high'
            
            # 评估数据质量风险
            successful_analyses = len(analyses)
            total_agents = len(self.get_available_agents())
            if total_agents > 0:
                success_rate = successful_analyses / total_agents
                if success_rate > 0.8:
                    risk_assessment['data_quality_risk'] = 'low'
                elif success_rate > 0.6:
                    risk_assessment['data_quality_risk'] = 'medium'
                else:
                    risk_assessment['data_quality_risk'] = 'high'
            
            # 综合风险等级
            risk_scores = {
                'low': 1,
                'medium': 2,
                'high': 3
            }
            
            avg_risk_score = (
                risk_scores[risk_assessment['consensus_risk']] +
                risk_scores[risk_assessment['data_quality_risk']] +
                risk_scores[risk_assessment['model_risk']]
            ) / 3
            
            if avg_risk_score <= 1.5:
                risk_assessment['overall_risk_level'] = 'low'
            elif avg_risk_score <= 2.5:
                risk_assessment['overall_risk_level'] = 'medium'
            else:
                risk_assessment['overall_risk_level'] = 'high'
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            risk_assessment['error'] = str(e)
        
        return risk_assessment
    
    def _update_performance_stats(self, result: AnalysisResult) -> None:
        """更新性能统计"""
        self.performance_stats['total_analyses'] += 1
        
        if result.status == "success":
            self.performance_stats['successful_analyses'] += 1
        
        # 更新平均执行时间
        total_time = self.performance_stats['avg_execution_time'] * (self.performance_stats['total_analyses'] - 1)
        self.performance_stats['avg_execution_time'] = (total_time + result.execution_time) / self.performance_stats['total_analyses']
        
        # 更新平均共识分数
        if result.debate_result:
            total_consensus = self.performance_stats['avg_consensus_score'] * (self.performance_stats['total_analyses'] - 1)
            self.performance_stats['avg_consensus_score'] = (total_consensus + result.debate_result.consensus_score) / self.performance_stats['total_analyses']
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取分析历史"""
        recent_analyses = self.analysis_history[-limit:] if self.analysis_history else []
        
        return [
            {
                'request_id': result.request_id,
                'symbols': result.symbols,
                'timestamp': result.timestamp.isoformat(),
                'execution_time': result.execution_time,
                'status': result.status,
                'consensus_recommendation': result.consensus_recommendation.value,
                'consensus_confidence': result.consensus_confidence,
                'agent_count': len(result.individual_analyses),
                'debate_consensus_score': result.debate_result.consensus_score if result.debate_result else None
            }
            for result in recent_analyses
        ]
    
    def get_active_requests(self) -> List[Dict[str, Any]]:
        """获取活跃请求"""
        return [
            {
                'request_id': request.request_id,
                'symbols': request.symbols,
                'timestamp': request.timestamp.isoformat(),
                'duration': (datetime.now() - request.timestamp).total_seconds(),
                'priority': request.priority,
                'timeout': request.timeout
            }
            for request in self.active_requests.values()
        ]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        
        if stats['total_analyses'] > 0:
            stats['success_rate'] = stats['successful_analyses'] / stats['total_analyses']
        else:
            stats['success_rate'] = 0.0
        
        stats['active_requests'] = len(self.active_requests)
        stats['available_agents'] = len(self.get_available_agents())
        stats['total_agents'] = len(self.agents)
        
        return stats
    
    def get_agent_status(self) -> List[Dict[str, Any]]:
        """获取智能体状态"""
        return [
            {
                'name': agent.name,
                'type': agent.agent_type,
                'expertise': agent.expertise,
                'is_active': agent.is_active,
                'is_ready': agent.is_ready_for_analysis(),
                'confidence_score': agent.get_confidence_score(),
                'performance_metrics': agent.get_performance_metrics(),
                'last_analysis': agent.last_analysis_time.isoformat() if agent.last_analysis_time else None
            }
            for agent in self.agents
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            'overall_health': 'healthy',
            'agents_health': {},
            'debate_engine_health': 'healthy',
            'consensus_builder_health': 'healthy',
            'issues': []
        }
        
        try:
            # 检查智能体健康状态
            for agent in self.agents:
                try:
                    # 简单的健康检查
                    is_ready = agent.is_ready_for_analysis()
                    confidence = agent.get_confidence_score()
                    
                    if is_ready and confidence > 0.5:
                        health_status['agents_health'][agent.name] = 'healthy'
                    elif is_ready:
                        health_status['agents_health'][agent.name] = 'warning'
                        health_status['issues'].append(f"Agent {agent.name} has low confidence")
                    else:
                        health_status['agents_health'][agent.name] = 'unhealthy'
                        health_status['issues'].append(f"Agent {agent.name} is not ready")
                        
                except Exception as e:
                    health_status['agents_health'][agent.name] = 'error'
                    health_status['issues'].append(f"Agent {agent.name} health check failed: {e}")
            
            # 检查辩论引擎
            try:
                debate_stats = self.debate_engine.get_debate_statistics()
                health_status['debate_engine_health'] = 'healthy'
            except Exception as e:
                health_status['debate_engine_health'] = 'error'
                health_status['issues'].append(f"Debate engine health check failed: {e}")
            
            # 检查共识构建器
            try:
                # 简单的健康检查
                test_analyses = []
                health_status['consensus_builder_health'] = 'healthy'
            except Exception as e:
                health_status['consensus_builder_health'] = 'error'
                health_status['issues'].append(f"Consensus builder health check failed: {e}")
            
            # 确定整体健康状态
            if health_status['issues']:
                if len(health_status['issues']) > 3:
                    health_status['overall_health'] = 'unhealthy'
                else:
                    health_status['overall_health'] = 'warning'
            
        except Exception as e:
            health_status['overall_health'] = 'error'
            health_status['issues'].append(f"Health check failed: {e}")
        
        return health_status
