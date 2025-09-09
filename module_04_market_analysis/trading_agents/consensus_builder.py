"""
共识构建器
构建多智能体分析的共识结果
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import Counter

from .base_agent import AgentAnalysis, RecommendationType
from common.logging_system import setup_logger

logger = setup_logger("consensus_builder")


@dataclass
class ConsensusConfig:
    """共识配置"""
    min_agents: int = 2
    confidence_threshold: float = 0.6
    agreement_threshold: float = 0.7
    use_weighted_voting: bool = True
    weight_by_confidence: bool = True
    weight_by_expertise: bool = True
    expertise_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.expertise_weights is None:
            self.expertise_weights = {
                'fundamental_analyst': 1.0,
                'technical_analyst': 0.9,
                'news_analyst': 0.8,
                'sentiment_analyst': 0.8,
                'risk_manager': 0.9
            }


@dataclass
class ConsensusResult:
    """共识结果"""
    recommendation: RecommendationType
    confidence: float
    reasoning: str
    key_insights: List[str]
    risk_factors: List[str]
    supporting_evidence: Dict[str, Any]
    disagreement_areas: List[str]
    consensus_score: float
    participant_count: int
    timestamp: datetime


class ConsensusBuilder:
    """共识构建器"""
    
    def __init__(self, config: Optional[ConsensusConfig] = None):
        """初始化共识构建器
        
        Args:
            config: 共识配置
        """
        self.config = config or ConsensusConfig()
        self.consensus_history: List[ConsensusResult] = []
        
        logger.info("Initialized consensus builder")
    
    def build_consensus(self, analyses: List[AgentAnalysis]) -> Dict[str, Any]:
        """构建共识
        
        Args:
            analyses: 智能体分析结果列表
            
        Returns:
            共识结果字典
        """
        try:
            if len(analyses) < self.config.min_agents:
                logger.warning(f"Not enough analyses for consensus: {len(analyses)} < {self.config.min_agents}")
                return self._create_default_consensus(analyses, "Insufficient analyses")
            
            logger.info(f"Building consensus from {len(analyses)} analyses")
            
            # 1. 计算推荐类型共识
            recommendation_consensus = self._calculate_recommendation_consensus(analyses)
            
            # 2. 计算置信度共识
            confidence_consensus = self._calculate_confidence_consensus(analyses)
            
            # 3. 提取关键洞察
            key_insights = self._extract_key_insights(analyses)
            
            # 4. 识别风险因素
            risk_factors = self._identify_risk_factors(analyses)
            
            # 5. 汇总支持证据
            supporting_evidence = self._summarize_evidence(analyses)
            
            # 6. 识别分歧领域
            disagreement_areas = self._identify_disagreements(analyses)
            
            # 7. 计算整体共识分数
            consensus_score = self._calculate_consensus_score(
                recommendation_consensus, confidence_consensus, analyses
            )
            
            # 8. 生成推理
            reasoning = self._generate_reasoning(
                recommendation_consensus, confidence_consensus, key_insights, disagreement_areas
            )
            
            # 9. 创建共识结果
            consensus_result = ConsensusResult(
                recommendation=recommendation_consensus['recommendation'],
                confidence=confidence_consensus['weighted_confidence'],
                reasoning=reasoning,
                key_insights=key_insights,
                risk_factors=risk_factors,
                supporting_evidence=supporting_evidence,
                disagreement_areas=disagreement_areas,
                consensus_score=consensus_score,
                participant_count=len(analyses),
                timestamp=datetime.now()
            )
            
            # 10. 记录到历史
            self.consensus_history.append(consensus_result)
            
            logger.info(f"Consensus built with score {consensus_score:.3f}")
            
            return {
                'recommendation': consensus_result.recommendation,
                'confidence': consensus_result.confidence,
                'reasoning': consensus_result.reasoning,
                'key_insights': consensus_result.key_insights,
                'risk_factors': consensus_result.risk_factors,
                'supporting_evidence': consensus_result.supporting_evidence,
                'disagreement_areas': consensus_result.disagreement_areas,
                'consensus_score': consensus_result.consensus_score,
                'participant_count': consensus_result.participant_count,
                'timestamp': consensus_result.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Consensus building failed: {e}")
            return self._create_default_consensus(analyses, str(e))
    
    def _calculate_recommendation_consensus(self, analyses: List[AgentAnalysis]) -> Dict[str, Any]:
        """计算推荐类型共识"""
        try:
            # 收集推荐类型
            recommendations = [analysis.recommendation.recommendation_type for analysis in analyses]
            
            # 计算权重
            weights = self._calculate_weights(analyses)
            
            # 加权投票
            weighted_votes = {}
            for i, rec_type in enumerate(recommendations):
                weight = weights[i]
                if rec_type not in weighted_votes:
                    weighted_votes[rec_type] = 0
                weighted_votes[rec_type] += weight
            
            # 确定获胜推荐
            winning_recommendation = max(weighted_votes.items(), key=lambda x: x[1])[0]
            winning_weight = weighted_votes[winning_recommendation]
            total_weight = sum(weighted_votes.values())
            
            # 计算支持度
            support_ratio = winning_weight / total_weight if total_weight > 0 else 0
            
            # 计算一致性
            consistency = self._calculate_recommendation_consistency(recommendations)
            
            return {
                'recommendation': winning_recommendation,
                'support_ratio': support_ratio,
                'consistency': consistency,
                'weighted_votes': weighted_votes,
                'total_weight': total_weight
            }
            
        except Exception as e:
            logger.error(f"Recommendation consensus calculation failed: {e}")
            return {
                'recommendation': RecommendationType.HOLD,
                'support_ratio': 0.0,
                'consistency': 0.0,
                'weighted_votes': {},
                'total_weight': 0.0
            }
    
    def _calculate_confidence_consensus(self, analyses: List[AgentAnalysis]) -> Dict[str, Any]:
        """计算置信度共识"""
        try:
            confidences = [analysis.recommendation.confidence for analysis in analyses]
            weights = self._calculate_weights(analyses)
            
            # 加权平均置信度
            weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / sum(weights)
            
            # 置信度一致性
            confidence_std = np.std(confidences)
            confidence_consistency = 1.0 / (1.0 + confidence_std)
            
            # 平均置信度
            avg_confidence = np.mean(confidences)
            
            # 置信度分布
            confidence_distribution = {
                'high': sum(1 for c in confidences if c > 0.8),
                'medium': sum(1 for c in confidences if 0.5 <= c <= 0.8),
                'low': sum(1 for c in confidences if c < 0.5)
            }
            
            return {
                'weighted_confidence': weighted_confidence,
                'avg_confidence': avg_confidence,
                'consistency': confidence_consistency,
                'std': confidence_std,
                'distribution': confidence_distribution
            }
            
        except Exception as e:
            logger.error(f"Confidence consensus calculation failed: {e}")
            return {
                'weighted_confidence': 0.5,
                'avg_confidence': 0.5,
                'consistency': 0.5,
                'std': 0.0,
                'distribution': {'high': 0, 'medium': 0, 'low': 0}
            }
    
    def _calculate_weights(self, analyses: List[AgentAnalysis]) -> List[float]:
        """计算权重"""
        weights = []
        
        for analysis in analyses:
            weight = 1.0
            
            # 基于置信度的权重
            if self.config.weight_by_confidence:
                weight *= analysis.recommendation.confidence
            
            # 基于专业领域的权重
            if self.config.weight_by_expertise:
                expertise_weight = self.config.expertise_weights.get(analysis.agent_type, 1.0)
                weight *= expertise_weight
            
            weights.append(weight)
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return weights
    
    def _calculate_recommendation_consistency(self, recommendations: List[RecommendationType]) -> float:
        """计算推荐类型一致性"""
        if not recommendations:
            return 0.0
        
        # 统计推荐类型分布
        rec_counts = Counter(recommendations)
        max_count = max(rec_counts.values())
        
        # 一致性 = 最大支持数 / 总数量
        consistency = max_count / len(recommendations)
        
        return consistency
    
    def _extract_key_insights(self, analyses: List[AgentAnalysis]) -> List[str]:
        """提取关键洞察"""
        all_insights = []
        
        # 收集所有关键因素
        for analysis in analyses:
            all_insights.extend(analysis.key_factors)
        
        # 去重
        unique_insights = list(set(all_insights))
        
        # 按出现频率排序
        insight_counts = Counter(all_insights)
        sorted_insights = sorted(
            unique_insights, 
            key=lambda x: insight_counts[x], 
            reverse=True
        )
        
        return sorted_insights[:10]  # 返回前10个洞察
    
    def _identify_risk_factors(self, analyses: List[AgentAnalysis]) -> List[str]:
        """识别风险因素"""
        all_risks = []
        
        # 收集所有风险因素
        for analysis in analyses:
            all_risks.extend(analysis.risk_factors)
        
        # 去重
        unique_risks = list(set(all_risks))
        
        # 按出现频率排序
        risk_counts = Counter(all_risks)
        sorted_risks = sorted(
            unique_risks, 
            key=lambda x: risk_counts[x], 
            reverse=True
        )
        
        return sorted_risks[:8]  # 返回前8个风险因素
    
    def _summarize_evidence(self, analyses: List[AgentAnalysis]) -> Dict[str, Any]:
        """汇总支持证据"""
        evidence_summary = {
            'data_sources': set(),
            'analysis_types': set(),
            'key_metrics': {},
            'supporting_arguments': []
        }
        
        for analysis in analyses:
            # 数据源
            evidence_summary['data_sources'].update(analysis.data_sources)
            
            # 分析类型
            evidence_summary['analysis_types'].add(analysis.agent_type)
            
            # 关键指标
            if 'additional_insights' in analysis.additional_insights:
                insights = analysis.additional_insights['additional_insights']
                for key, value in insights.items():
                    if isinstance(value, (int, float)):
                        if key not in evidence_summary['key_metrics']:
                            evidence_summary['key_metrics'][key] = []
                        evidence_summary['key_metrics'][key].append(value)
            
            # 支持论点
            evidence_summary['supporting_arguments'].extend(analysis.key_factors)
        
        # 转换集合为列表
        evidence_summary['data_sources'] = list(evidence_summary['data_sources'])
        evidence_summary['analysis_types'] = list(evidence_summary['analysis_types'])
        
        # 计算关键指标的平均值
        for key, values in evidence_summary['key_metrics'].items():
            if values:
                evidence_summary['key_metrics'][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        
        return evidence_summary
    
    def _identify_disagreements(self, analyses: List[AgentAnalysis]) -> List[str]:
        """识别分歧领域"""
        disagreements = []
        
        # 推荐类型分歧
        recommendations = [analysis.recommendation.recommendation_type for analysis in analyses]
        unique_recommendations = set(recommendations)
        
        if len(unique_recommendations) > 1:
            rec_names = [r.value for r in unique_recommendations]
            disagreements.append(f"推荐类型分歧: {', '.join(rec_names)}")
        
        # 置信度分歧
        confidences = [analysis.recommendation.confidence for analysis in analyses]
        if confidences:
            confidence_std = np.std(confidences)
            if confidence_std > 0.3:
                disagreements.append("置信度差异较大")
        
        # 风险因素分歧
        all_risks = []
        for analysis in analyses:
            all_risks.extend(analysis.risk_factors)
        
        risk_counts = Counter(all_risks)
        conflicting_risks = [risk for risk, count in risk_counts.items() if count < len(analyses) / 2]
        
        if conflicting_risks:
            disagreements.append(f"风险因素分歧: {', '.join(conflicting_risks[:3])}")
        
        return disagreements
    
    def _calculate_consensus_score(
        self, 
        recommendation_consensus: Dict[str, Any], 
        confidence_consensus: Dict[str, Any], 
        analyses: List[AgentAnalysis]
    ) -> float:
        """计算整体共识分数"""
        try:
            # 推荐类型一致性权重
            rec_consistency = recommendation_consensus.get('consistency', 0.0)
            rec_support = recommendation_consensus.get('support_ratio', 0.0)
            
            # 置信度一致性权重
            conf_consistency = confidence_consensus.get('consistency', 0.0)
            conf_level = confidence_consensus.get('weighted_confidence', 0.0)
            
            # 参与度权重
            participation_score = min(len(analyses) / 5.0, 1.0)  # 最多5个智能体
            
            # 综合共识分数
            consensus_score = (
                rec_consistency * 0.3 +      # 推荐一致性
                rec_support * 0.2 +          # 推荐支持度
                conf_consistency * 0.2 +     # 置信度一致性
                conf_level * 0.2 +           # 置信度水平
                participation_score * 0.1    # 参与度
            )
            
            return min(max(consensus_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Consensus score calculation failed: {e}")
            return 0.0
    
    def _generate_reasoning(
        self, 
        recommendation_consensus: Dict[str, Any], 
        confidence_consensus: Dict[str, Any], 
        key_insights: List[str], 
        disagreement_areas: List[str]
    ) -> str:
        """生成推理"""
        reasoning_parts = []
        
        # 推荐说明
        rec_type = recommendation_consensus.get('recommendation', RecommendationType.HOLD)
        rec_type_names = {
            RecommendationType.STRONG_BUY: "强烈买入",
            RecommendationType.BUY: "买入",
            RecommendationType.HOLD: "持有",
            RecommendationType.SELL: "卖出",
            RecommendationType.STRONG_SELL: "强烈卖出"
        }
        
        reasoning_parts.append(f"基于多智能体分析，达成共识：{rec_type_names.get(rec_type, '持有')}")
        
        # 支持度说明
        support_ratio = recommendation_consensus.get('support_ratio', 0.0)
        reasoning_parts.append(f"支持度：{support_ratio:.1%}")
        
        # 置信度说明
        confidence = confidence_consensus.get('weighted_confidence', 0.0)
        reasoning_parts.append(f"置信度：{confidence:.1%}")
        
        # 关键洞察
        if key_insights:
            reasoning_parts.append(f"关键洞察：{', '.join(key_insights[:3])}")
        
        # 分歧说明
        if disagreement_areas:
            reasoning_parts.append(f"存在分歧：{', '.join(disagreement_areas[:2])}")
        
        return "。".join(reasoning_parts) + "。"
    
    def _create_default_consensus(self, analyses: List[AgentAnalysis], error_msg: str) -> Dict[str, Any]:
        """创建默认共识"""
        return {
            'recommendation': RecommendationType.HOLD,
            'confidence': 0.3,
            'reasoning': f"共识构建失败: {error_msg}",
            'key_insights': [],
            'risk_factors': [],
            'supporting_evidence': {},
            'disagreement_areas': [error_msg],
            'consensus_score': 0.0,
            'participant_count': len(analyses),
            'timestamp': datetime.now().isoformat(),
            'error': error_msg
        }
    
    def get_consensus_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取共识历史"""
        recent_consensus = self.consensus_history[-limit:] if self.consensus_history else []
        
        return [
            {
                'timestamp': result.timestamp.isoformat(),
                'recommendation': result.recommendation.value,
                'confidence': result.confidence,
                'consensus_score': result.consensus_score,
                'participant_count': result.participant_count,
                'key_insights_count': len(result.key_insights),
                'disagreement_count': len(result.disagreement_areas)
            }
            for result in recent_consensus
        ]
    
    def get_consensus_statistics(self) -> Dict[str, Any]:
        """获取共识统计"""
        if not self.consensus_history:
            return {
                'total_consensus': 0,
                'avg_consensus_score': 0.0,
                'avg_confidence': 0.0,
                'recommendation_distribution': {},
                'success_rate': 0.0
            }
        
        total_consensus = len(self.consensus_history)
        avg_consensus_score = sum(r.consensus_score for r in self.consensus_history) / total_consensus
        avg_confidence = sum(r.confidence for r in self.consensus_history) / total_consensus
        
        # 推荐类型分布
        rec_distribution = Counter(r.recommendation for r in self.consensus_history)
        rec_distribution = {k.value: v for k, v in rec_distribution.items()}
        
        # 成功率（共识分数 > 0.6）
        successful_consensus = sum(1 for r in self.consensus_history if r.consensus_score > 0.6)
        success_rate = successful_consensus / total_consensus
        
        return {
            'total_consensus': total_consensus,
            'avg_consensus_score': avg_consensus_score,
            'avg_confidence': avg_confidence,
            'recommendation_distribution': rec_distribution,
            'success_rate': success_rate
        }
