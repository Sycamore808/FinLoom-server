"""
辩论引擎
管理多智能体之间的辩论过程和决策机制
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from .base_agent import BaseAgent, AgentAnalysis, DebateContext, DebateResponse, RecommendationType
from common.logging_system import setup_logger

logger = setup_logger("debate_engine")


@dataclass
class DebateRound:
    """辩论轮次"""
    round_number: int
    timestamp: datetime
    participants: List[str]
    responses: List[DebateResponse]
    consensus_score: float
    key_arguments: List[str]
    evidence_summary: Dict[str, Any]


@dataclass
class DebateResult:
    """辩论结果"""
    debate_id: str
    context: DebateContext
    rounds: List[DebateRound]
    final_consensus: Dict[str, Any]
    consensus_score: float
    final_recommendation: RecommendationType
    confidence: float
    reasoning: str
    key_insights: List[str]
    remaining_disagreements: List[str]
    debate_duration: float
    timestamp: datetime


class DebateEngine:
    """辩论引擎"""
    
    def __init__(
        self,
        max_rounds: int = 3,
        consensus_threshold: float = 0.7,
        min_participants: int = 2,
        debate_timeout: float = 300.0  # 5分钟超时
    ):
        """初始化辩论引擎
        
        Args:
            max_rounds: 最大辩论轮数
            consensus_threshold: 共识阈值
            min_participants: 最少参与者数量
            debate_timeout: 辩论超时时间（秒）
        """
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.min_participants = min_participants
        self.debate_timeout = debate_timeout
        self.active_debates: Dict[str, DebateResult] = {}
        self.debate_history: List[DebateResult] = []
        
        logger.info(f"Initialized debate engine with max_rounds={max_rounds}, consensus_threshold={consensus_threshold}")
    
    async def conduct_debate(
        self, 
        agents: List[BaseAgent], 
        initial_analyses: List[AgentAnalysis],
        context: DebateContext
    ) -> DebateResult:
        """进行多轮辩论
        
        Args:
            agents: 参与辩论的智能体列表
            initial_analyses: 初始分析结果
            context: 辩论上下文
            
        Returns:
            辩论结果
        """
        debate_id = f"debate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"Starting debate {debate_id} with {len(agents)} agents")
        
        try:
            # 验证输入
            if len(agents) < self.min_participants:
                raise ValueError(f"Need at least {self.min_participants} agents for debate")
            
            if len(agents) != len(initial_analyses):
                raise ValueError("Number of agents must match number of initial analyses")
            
            # 初始化辩论结果
            debate_result = DebateResult(
                debate_id=debate_id,
                context=context,
                rounds=[],
                final_consensus={},
                consensus_score=0.0,
                final_recommendation=RecommendationType.HOLD,
                confidence=0.0,
                reasoning="",
                key_insights=[],
                remaining_disagreements=[],
                debate_duration=0.0,
                timestamp=start_time
            )
            
            # 记录活跃辩论
            self.active_debates[debate_id] = debate_result
            
            # 进行多轮辩论
            current_analyses = initial_analyses.copy()
            
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"Starting debate round {round_num}")
                
                # 进行一轮辩论
                round_result = await self._conduct_debate_round(
                    agents, current_analyses, context, round_num
                )
                
                # 添加到辩论结果
                debate_result.rounds.append(round_result)
                
                # 更新当前分析结果
                current_analyses = self._update_analyses_from_responses(
                    current_analyses, round_result.responses
                )
                
                # 检查是否达成共识
                if round_result.consensus_score >= self.consensus_threshold:
                    logger.info(f"Consensus reached in round {round_num}")
                    break
                
                # 检查是否超时
                elapsed_time = (datetime.now() - start_time).total_seconds()
                if elapsed_time > self.debate_timeout:
                    logger.warning(f"Debate {debate_id} timed out after {elapsed_time:.2f}s")
                    break
            
            # 生成最终共识
            final_consensus = await self._generate_final_consensus(
                agents, current_analyses, debate_result.rounds
            )
            
            # 更新辩论结果
            debate_result.final_consensus = final_consensus
            debate_result.consensus_score = final_consensus.get('consensus_score', 0.0)
            debate_result.final_recommendation = final_consensus.get('recommendation', RecommendationType.HOLD)
            debate_result.confidence = final_consensus.get('confidence', 0.0)
            debate_result.reasoning = final_consensus.get('reasoning', '')
            debate_result.key_insights = final_consensus.get('key_insights', [])
            debate_result.remaining_disagreements = final_consensus.get('disagreements', [])
            debate_result.debate_duration = (datetime.now() - start_time).total_seconds()
            
            # 记录到历史
            self.debate_history.append(debate_result)
            
            # 清理活跃辩论
            if debate_id in self.active_debates:
                del self.active_debates[debate_id]
            
            logger.info(f"Debate {debate_id} completed in {debate_result.debate_duration:.2f}s with consensus score {debate_result.consensus_score:.3f}")
            
            return debate_result
            
        except Exception as e:
            logger.error(f"Debate {debate_id} failed: {e}")
            
            # 创建失败结果
            failed_result = DebateResult(
                debate_id=debate_id,
                context=context,
                rounds=[],
                final_consensus={'error': str(e)},
                consensus_score=0.0,
                final_recommendation=RecommendationType.HOLD,
                confidence=0.0,
                reasoning=f"Debate failed: {e}",
                key_insights=[],
                remaining_disagreements=[],
                debate_duration=(datetime.now() - start_time).total_seconds(),
                timestamp=start_time
            )
            
            # 清理活跃辩论
            if debate_id in self.active_debates:
                del self.active_debates[debate_id]
            
            return failed_result
    
    async def _conduct_debate_round(
        self, 
        agents: List[BaseAgent], 
        current_analyses: List[AgentAnalysis],
        context: DebateContext,
        round_number: int
    ) -> DebateRound:
        """进行一轮辩论"""
        round_start_time = datetime.now()
        
        # 收集所有智能体的辩论响应
        responses = []
        tasks = []
        
        for i, agent in enumerate(agents):
            # 获取其他智能体的分析结果
            other_analyses = [analysis for j, analysis in enumerate(current_analyses) if j != i]
            
            # 创建辩论任务
            task = agent.debate(other_analyses, context)
            tasks.append(task)
        
        # 并发执行所有辩论任务
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常响应
            valid_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Agent {agents[i].name} debate failed: {response}")
                    # 创建默认响应
                    default_response = DebateResponse(
                        agent_name=agents[i].name,
                        position="中立",
                        arguments=["辩论过程中出现错误"],
                        evidence={},
                        updated_confidence=0.3
                    )
                    valid_responses.append(default_response)
                else:
                    valid_responses.append(response)
            
            responses = valid_responses
            
        except Exception as e:
            logger.error(f"Failed to conduct debate round {round_number}: {e}")
            # 创建默认响应
            responses = [
                DebateResponse(
                    agent_name=agent.name,
                    position="中立",
                    arguments=["辩论失败"],
                    evidence={},
                    updated_confidence=0.3
                )
                for agent in agents
            ]
        
        # 计算共识分数
        consensus_score = self._calculate_round_consensus(responses)
        
        # 提取关键论点
        key_arguments = self._extract_key_arguments(responses)
        
        # 汇总证据
        evidence_summary = self._summarize_evidence(responses)
        
        round_duration = (datetime.now() - round_start_time).total_seconds()
        
        logger.info(f"Debate round {round_number} completed in {round_duration:.2f}s with consensus score {consensus_score:.3f}")
        
        return DebateRound(
            round_number=round_number,
            timestamp=round_start_time,
            participants=[agent.name for agent in agents],
            responses=responses,
            consensus_score=consensus_score,
            key_arguments=key_arguments,
            evidence_summary=evidence_summary
        )
    
    def _update_analyses_from_responses(
        self, 
        current_analyses: List[AgentAnalysis], 
        responses: List[DebateResponse]
    ) -> List[AgentAnalysis]:
        """根据辩论响应更新分析结果"""
        updated_analyses = []
        
        for i, (analysis, response) in enumerate(zip(current_analyses, responses)):
            # 创建更新的分析结果
            updated_analysis = AgentAnalysis(
                agent_name=analysis.agent_name,
                agent_type=analysis.agent_type,
                analysis_id=analysis.analysis_id,
                timestamp=analysis.timestamp,
                symbols=analysis.symbols,
                recommendation=analysis.recommendation,
                key_factors=analysis.key_factors,
                risk_factors=analysis.risk_factors,
                market_outlook=analysis.market_outlook,
                additional_insights={
                    **analysis.additional_insights,
                    'debate_response': {
                        'position': response.position,
                        'arguments': response.arguments,
                        'updated_confidence': response.updated_confidence,
                        'willingness_to_compromise': response.willingness_to_compromise
                    }
                },
                data_sources=analysis.data_sources,
                analysis_duration=analysis.analysis_duration
            )
            
            # 更新置信度
            updated_analysis.recommendation.confidence = response.updated_confidence
            
            updated_analyses.append(updated_analysis)
        
        return updated_analyses
    
    async def _generate_final_consensus(
        self, 
        agents: List[BaseAgent], 
        final_analyses: List[AgentAnalysis],
        rounds: List[DebateRound]
    ) -> Dict[str, Any]:
        """生成最终共识"""
        try:
            # 统计推荐类型分布
            recommendation_counts = {}
            total_confidence = 0
            
            for analysis in final_analyses:
                rec_type = analysis.recommendation.recommendation_type
                if rec_type not in recommendation_counts:
                    recommendation_counts[rec_type] = {'count': 0, 'confidence': 0}
                
                recommendation_counts[rec_type]['count'] += 1
                recommendation_counts[rec_type]['confidence'] += analysis.recommendation.confidence
                total_confidence += analysis.recommendation.confidence
            
            # 确定最终推荐
            if recommendation_counts:
                # 选择支持者最多的推荐类型
                final_recommendation = max(
                    recommendation_counts.items(), 
                    key=lambda x: x[1]['count']
                )[0]
                
                # 计算平均置信度
                avg_confidence = total_confidence / len(final_analyses)
                
                # 计算共识分数
                consensus_score = self._calculate_final_consensus_score(
                    recommendation_counts, len(final_analyses)
                )
            else:
                final_recommendation = RecommendationType.HOLD
                avg_confidence = 0.5
                consensus_score = 0.0
            
            # 提取关键洞察
            key_insights = self._extract_final_insights(final_analyses, rounds)
            
            # 识别剩余分歧
            disagreements = self._identify_remaining_disagreements(final_analyses)
            
            # 生成推理
            reasoning = self._generate_consensus_reasoning(
                final_recommendation, recommendation_counts, key_insights
            )
            
            return {
                'recommendation': final_recommendation,
                'confidence': avg_confidence,
                'consensus_score': consensus_score,
                'reasoning': reasoning,
                'key_insights': key_insights,
                'disagreements': disagreements,
                'recommendation_distribution': recommendation_counts,
                'participant_count': len(agents),
                'rounds_completed': len(rounds)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate final consensus: {e}")
            return {
                'recommendation': RecommendationType.HOLD,
                'confidence': 0.3,
                'consensus_score': 0.0,
                'reasoning': f"Consensus generation failed: {e}",
                'key_insights': [],
                'disagreements': ["Consensus generation failed"],
                'error': str(e)
            }
    
    def _calculate_round_consensus(self, responses: List[DebateResponse]) -> float:
        """计算单轮共识分数"""
        if not responses:
            return 0.0
        
        # 统计立场分布
        positions = {}
        total_confidence = 0
        
        for response in responses:
            position = response.position
            if position not in positions:
                positions[position] = 0
            positions[position] += 1
            total_confidence += response.updated_confidence
        
        # 计算立场一致性
        if len(positions) == 1:
            # 所有智能体立场一致
            position_consensus = 1.0
        else:
            # 计算最大立场占比
            max_position_count = max(positions.values())
            position_consensus = max_position_count / len(responses)
        
        # 计算平均置信度
        avg_confidence = total_confidence / len(responses)
        
        # 综合共识分数
        consensus_score = (position_consensus * 0.6 + avg_confidence * 0.4)
        
        return min(max(consensus_score, 0.0), 1.0)
    
    def _calculate_final_consensus_score(
        self, 
        recommendation_counts: Dict[RecommendationType, Dict[str, float]], 
        total_agents: int
    ) -> float:
        """计算最终共识分数"""
        if not recommendation_counts:
            return 0.0
        
        # 计算推荐类型的一致性
        max_count = max(count['count'] for count in recommendation_counts.values())
        type_consensus = max_count / total_agents
        
        # 计算平均置信度
        total_confidence = sum(count['confidence'] for count in recommendation_counts.values())
        avg_confidence = total_confidence / total_agents
        
        # 综合共识分数
        consensus_score = (type_consensus * 0.7 + avg_confidence * 0.3)
        
        return min(max(consensus_score, 0.0), 1.0)
    
    def _extract_key_arguments(self, responses: List[DebateResponse]) -> List[str]:
        """提取关键论点"""
        all_arguments = []
        
        for response in responses:
            all_arguments.extend(response.arguments)
        
        # 去重并排序
        unique_arguments = list(set(all_arguments))
        
        # 按出现频率排序
        argument_counts = {}
        for arg in all_arguments:
            argument_counts[arg] = argument_counts.get(arg, 0) + 1
        
        sorted_arguments = sorted(
            unique_arguments, 
            key=lambda x: argument_counts[x], 
            reverse=True
        )
        
        return sorted_arguments[:5]  # 返回前5个关键论点
    
    def _summarize_evidence(self, responses: List[DebateResponse]) -> Dict[str, Any]:
        """汇总证据"""
        evidence_summary = {
            'total_evidence_items': 0,
            'evidence_categories': {},
            'strongest_evidence': [],
            'conflicting_evidence': []
        }
        
        for response in responses:
            evidence = response.evidence
            evidence_summary['total_evidence_items'] += len(evidence)
            
            for category, data in evidence.items():
                if category not in evidence_summary['evidence_categories']:
                    evidence_summary['evidence_categories'][category] = []
                evidence_summary['evidence_categories'][category].append(data)
        
        return evidence_summary
    
    def _extract_final_insights(
        self, 
        final_analyses: List[AgentAnalysis], 
        rounds: List[DebateRound]
    ) -> List[str]:
        """提取最终洞察"""
        insights = []
        
        # 从最终分析中提取洞察
        for analysis in final_analyses:
            insights.extend(analysis.key_factors)
        
        # 从辩论轮次中提取洞察
        for round_data in rounds:
            insights.extend(round_data.key_arguments)
        
        # 去重并排序
        unique_insights = list(set(insights))
        
        return unique_insights[:10]  # 返回前10个洞察
    
    def _identify_remaining_disagreements(self, final_analyses: List[AgentAnalysis]) -> List[str]:
        """识别剩余分歧"""
        disagreements = []
        
        # 统计推荐类型分布
        recommendation_types = [analysis.recommendation.recommendation_type for analysis in final_analyses]
        unique_recommendations = set(recommendation_types)
        
        if len(unique_recommendations) > 1:
            disagreements.append(f"推荐类型不一致: {', '.join([r.value for r in unique_recommendations])}")
        
        # 检查置信度差异
        confidences = [analysis.recommendation.confidence for analysis in final_analyses]
        if confidences:
            confidence_std = np.std(confidences)
            if confidence_std > 0.3:
                disagreements.append("智能体置信度差异较大")
        
        return disagreements
    
    def _generate_consensus_reasoning(
        self, 
        final_recommendation: RecommendationType,
        recommendation_counts: Dict[RecommendationType, Dict[str, float]],
        key_insights: List[str]
    ) -> str:
        """生成共识推理"""
        reasoning_parts = []
        
        # 推荐类型说明
        rec_type_names = {
            RecommendationType.STRONG_BUY: "强烈买入",
            RecommendationType.BUY: "买入",
            RecommendationType.HOLD: "持有",
            RecommendationType.SELL: "卖出",
            RecommendationType.STRONG_SELL: "强烈卖出"
        }
        
        reasoning_parts.append(f"经过多轮辩论，智能体达成共识：{rec_type_names.get(final_recommendation, '持有')}")
        
        # 支持度说明
        if final_recommendation in recommendation_counts:
            support_count = recommendation_counts[final_recommendation]['count']
            total_count = sum(count['count'] for count in recommendation_counts.values())
            support_ratio = support_count / total_count
            reasoning_parts.append(f"支持度：{support_ratio:.1%} ({support_count}/{total_count})")
        
        # 关键洞察
        if key_insights:
            reasoning_parts.append(f"关键洞察：{', '.join(key_insights[:3])}")
        
        return "。".join(reasoning_parts) + "。"
    
    def get_debate_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取辩论历史"""
        recent_debates = self.debate_history[-limit:] if self.debate_history else []
        
        return [
            {
                'debate_id': debate.debate_id,
                'timestamp': debate.timestamp.isoformat(),
                'duration': debate.debate_duration,
                'consensus_score': debate.consensus_score,
                'final_recommendation': debate.final_recommendation.value,
                'confidence': debate.confidence,
                'participants': [round_data.participants for round_data in debate.rounds],
                'rounds_completed': len(debate.rounds)
            }
            for debate in recent_debates
        ]
    
    def get_active_debates(self) -> List[Dict[str, Any]]:
        """获取活跃辩论"""
        return [
            {
                'debate_id': debate_id,
                'timestamp': debate.timestamp.isoformat(),
                'duration': (datetime.now() - debate.timestamp).total_seconds(),
                'rounds_completed': len(debate.rounds),
                'current_consensus': debate.rounds[-1].consensus_score if debate.rounds else 0.0
            }
            for debate_id, debate in self.active_debates.items()
        ]
    
    def get_debate_statistics(self) -> Dict[str, Any]:
        """获取辩论统计信息"""
        if not self.debate_history:
            return {
                'total_debates': 0,
                'avg_consensus_score': 0.0,
                'avg_duration': 0.0,
                'success_rate': 0.0
            }
        
        total_debates = len(self.debate_history)
        avg_consensus_score = sum(debate.consensus_score for debate in self.debate_history) / total_debates
        avg_duration = sum(debate.debate_duration for debate in self.debate_history) / total_debates
        successful_debates = sum(1 for debate in self.debate_history if debate.consensus_score >= self.consensus_threshold)
        success_rate = successful_debates / total_debates
        
        return {
            'total_debates': total_debates,
            'avg_consensus_score': avg_consensus_score,
            'avg_duration': avg_duration,
            'success_rate': success_rate,
            'active_debates': len(self.active_debates)
        }
