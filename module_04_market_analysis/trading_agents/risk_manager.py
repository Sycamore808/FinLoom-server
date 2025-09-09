"""
风险管理师
评估投资风险、制定风控策略和风险预警
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from .base_agent import BaseAgent, AgentAnalysis, AgentRecommendation, RecommendationType, DebateContext, DebateResponse, create_analysis_id
from common.logging_system import setup_logger

logger = setup_logger("risk_manager")


class RiskManager(BaseAgent):
    """风险管理师"""
    
    def __init__(self):
        super().__init__(
            name="风险管理师",
            agent_type="risk_manager",
            expertise="风险评估, 风险控制, 风险预警, 压力测试, 风险量化",
            confidence_threshold=0.7
        )
        self.risk_metrics_cache = {}
        self.risk_scenarios_cache = {}
        self.var_cache = {}
    
    async def analyze(
        self, 
        symbols: List[str], 
        market_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """分析投资风险"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting risk analysis for symbols: {symbols}")
            
            # 1. 计算市场风险指标
            market_risk = await self._calculate_market_risk(symbols, market_data)
            
            # 2. 计算信用风险
            credit_risk = await self._calculate_credit_risk(symbols)
            
            # 3. 计算流动性风险
            liquidity_risk = await self._calculate_liquidity_risk(symbols, market_data)
            
            # 4. 计算操作风险
            operational_risk = await self._calculate_operational_risk(symbols)
            
            # 5. 进行压力测试
            stress_test = await self._perform_stress_test(symbols, market_data)
            
            # 6. 计算VaR和CVaR
            var_analysis = await self._calculate_var_cvar(symbols, market_data)
            
            # 7. 综合风险评估
            overall_risk = self._assess_overall_risk(
                market_risk, credit_risk, liquidity_risk, operational_risk, stress_test, var_analysis
            )
            
            # 8. 生成推荐
            recommendation = self._generate_recommendation(overall_risk, symbols)
            
            # 9. 计算分析耗时
            analysis_duration = (datetime.now() - start_time).total_seconds()
            
            # 10. 创建分析结果
            analysis = AgentAnalysis(
                agent_name=self.name,
                agent_type=self.agent_type,
                analysis_id=create_analysis_id(),
                timestamp=datetime.now(),
                symbols=symbols,
                recommendation=recommendation,
                key_factors=self._extract_key_factors(overall_risk),
                risk_factors=self._extract_risk_factors(overall_risk),
                market_outlook=self._generate_market_outlook(overall_risk),
                additional_insights={
                    'market_risk': market_risk,
                    'credit_risk': credit_risk,
                    'liquidity_risk': liquidity_risk,
                    'operational_risk': operational_risk,
                    'stress_test': stress_test,
                    'var_analysis': var_analysis,
                    'overall_risk': overall_risk,
                    'risk_score': overall_risk.get('overall_risk_score', 0.5),
                    'risk_level': overall_risk.get('risk_level', 'medium')
                },
                data_sources=['市场数据', '财务数据', '交易数据', '宏观经济数据'],
                analysis_duration=analysis_duration
            )
            
            logger.info(f"Risk analysis completed for {symbols} in {analysis_duration:.2f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Risk analysis failed for {symbols}: {e}")
            return self._create_default_analysis(symbols, str(e))
    
    async def debate(
        self, 
        other_analyses: List[AgentAnalysis],
        debate_context: DebateContext
    ) -> DebateResponse:
        """与其他智能体进行辩论"""
        try:
            # 分析其他智能体的观点
            opposing_views = []
            supporting_views = []
            
            for analysis in other_analyses:
                if analysis.recommendation.recommendation_type in [RecommendationType.BUY, RecommendationType.STRONG_BUY]:
                    supporting_views.append(analysis)
                elif analysis.recommendation.recommendation_type in [RecommendationType.SELL, RecommendationType.STRONG_SELL]:
                    opposing_views.append(analysis)
            
            # 基于风险评估生成辩论响应
            current_risk = self._get_current_risk_level()
            
            if current_risk > 0.7:
                position = "反对"
                arguments = [
                    "风险水平过高，不建议投资",
                    "存在多重风险因素",
                    "建议降低仓位或暂停投资"
                ]
            elif current_risk < 0.3:
                position = "支持"
                arguments = [
                    "风险水平较低，可以投资",
                    "风险控制措施有效",
                    "建议适度增加仓位"
                ]
            else:
                position = "中立"
                arguments = [
                    "风险水平中等，需要谨慎",
                    "建议控制仓位规模",
                    "密切关注风险变化"
                ]
            
            return DebateResponse(
                agent_name=self.name,
                position=position,
                arguments=arguments,
                evidence={
                    'risk_level': current_risk,
                    'risk_factors': self._get_current_risk_factors(),
                    'var_estimate': self._get_var_estimate()
                },
                counter_arguments=self._generate_counter_arguments(other_analyses),
                updated_confidence=self._calculate_debate_confidence(other_analyses),
                willingness_to_compromise=0.4  # 风险管理相对保守
            )
            
        except Exception as e:
            logger.error(f"Debate failed for {self.name}: {e}")
            return DebateResponse(
                agent_name=self.name,
                position="中立",
                arguments=["风险评估过程中出现错误"],
                evidence={},
                updated_confidence=0.3
            )
    
    async def _calculate_market_risk(self, symbols: List[str], market_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """计算市场风险"""
        try:
            # 模拟市场风险计算
            market_risk = {
                'volatility': np.random.uniform(0.15, 0.35),  # 年化波动率
                'beta': np.random.uniform(0.8, 1.3),  # Beta系数
                'correlation_with_market': np.random.uniform(0.6, 0.9),  # 与市场相关性
                'systematic_risk': np.random.uniform(0.1, 0.3),  # 系统性风险
                'unsystematic_risk': np.random.uniform(0.05, 0.15),  # 非系统性风险
                'market_cap_risk': np.random.uniform(0.1, 0.4),  # 市值风险
                'sector_risk': np.random.uniform(0.1, 0.3),  # 行业风险
                'geographic_risk': np.random.uniform(0.05, 0.2),  # 地域风险
                'currency_risk': np.random.uniform(0.02, 0.1),  # 汇率风险
                'interest_rate_risk': np.random.uniform(0.1, 0.3),  # 利率风险
                'commodity_risk': np.random.uniform(0.05, 0.2),  # 商品风险
                'overall_market_risk': np.random.uniform(0.2, 0.6)  # 综合市场风险
            }
            
            # 缓存数据
            cache_key = f"market_risk_{','.join(sorted(symbols))}_{datetime.now().strftime('%Y%m%d%H')}"
            self.risk_metrics_cache[cache_key] = market_risk
            
            return market_risk
            
        except Exception as e:
            logger.error(f"Failed to calculate market risk: {e}")
            return {}
    
    async def _calculate_credit_risk(self, symbols: List[str]) -> Dict[str, Any]:
        """计算信用风险"""
        try:
            # 模拟信用风险计算
            credit_risk = {
                'default_probability': np.random.uniform(0.01, 0.05),  # 违约概率
                'credit_rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B']),  # 信用评级
                'credit_spread': np.random.uniform(0.5, 3.0),  # 信用利差
                'recovery_rate': np.random.uniform(0.3, 0.7),  # 回收率
                'credit_risk_score': np.random.uniform(0.1, 0.4),  # 信用风险分数
                'counterparty_risk': np.random.uniform(0.05, 0.2),  # 交易对手风险
                'concentration_risk': np.random.uniform(0.1, 0.3),  # 集中度风险
                'sovereign_risk': np.random.uniform(0.02, 0.1),  # 主权风险
                'overall_credit_risk': np.random.uniform(0.1, 0.4)  # 综合信用风险
            }
            
            return credit_risk
            
        except Exception as e:
            logger.error(f"Failed to calculate credit risk: {e}")
            return {}
    
    async def _calculate_liquidity_risk(self, symbols: List[str], market_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """计算流动性风险"""
        try:
            # 模拟流动性风险计算
            liquidity_risk = {
                'bid_ask_spread': np.random.uniform(0.001, 0.01),  # 买卖价差
                'volume_risk': np.random.uniform(0.1, 0.4),  # 成交量风险
                'market_depth': np.random.uniform(0.3, 0.8),  # 市场深度
                'liquidity_ratio': np.random.uniform(0.5, 2.0),  # 流动性比率
                'turnover_ratio': np.random.uniform(0.1, 1.0),  # 换手率
                'price_impact': np.random.uniform(0.01, 0.05),  # 价格冲击
                'time_to_liquidate': np.random.uniform(1, 10),  # 清算时间（天）
                'liquidity_cost': np.random.uniform(0.001, 0.01),  # 流动性成本
                'overall_liquidity_risk': np.random.uniform(0.1, 0.5)  # 综合流动性风险
            }
            
            return liquidity_risk
            
        except Exception as e:
            logger.error(f"Failed to calculate liquidity risk: {e}")
            return {}
    
    async def _calculate_operational_risk(self, symbols: List[str]) -> Dict[str, Any]:
        """计算操作风险"""
        try:
            # 模拟操作风险计算
            operational_risk = {
                'model_risk': np.random.uniform(0.05, 0.2),  # 模型风险
                'data_risk': np.random.uniform(0.02, 0.1),  # 数据风险
                'system_risk': np.random.uniform(0.01, 0.05),  # 系统风险
                'human_risk': np.random.uniform(0.02, 0.08),  # 人为风险
                'regulatory_risk': np.random.uniform(0.01, 0.1),  # 监管风险
                'legal_risk': np.random.uniform(0.01, 0.05),  # 法律风险
                'reputation_risk': np.random.uniform(0.01, 0.08),  # 声誉风险
                'technology_risk': np.random.uniform(0.02, 0.1),  # 技术风险
                'overall_operational_risk': np.random.uniform(0.05, 0.2)  # 综合操作风险
            }
            
            return operational_risk
            
        except Exception as e:
            logger.error(f"Failed to calculate operational risk: {e}")
            return {}
    
    async def _perform_stress_test(self, symbols: List[str], market_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """进行压力测试"""
        try:
            # 模拟压力测试
            stress_test = {
                'market_crash_scenario': {
                    'probability': 0.05,
                    'impact': np.random.uniform(-0.3, -0.5),
                    'description': '市场崩盘情景'
                },
                'recession_scenario': {
                    'probability': 0.1,
                    'impact': np.random.uniform(-0.2, -0.4),
                    'description': '经济衰退情景'
                },
                'interest_rate_shock': {
                    'probability': 0.15,
                    'impact': np.random.uniform(-0.1, -0.3),
                    'description': '利率冲击情景'
                },
                'currency_crisis': {
                    'probability': 0.08,
                    'impact': np.random.uniform(-0.15, -0.25),
                    'description': '货币危机情景'
                },
                'sector_downturn': {
                    'probability': 0.2,
                    'impact': np.random.uniform(-0.1, -0.2),
                    'description': '行业下行情景'
                },
                'worst_case_scenario': {
                    'probability': 0.01,
                    'impact': np.random.uniform(-0.4, -0.6),
                    'description': '最坏情况情景'
                },
                'expected_shortfall': np.random.uniform(0.05, 0.15),  # 期望损失
                'maximum_drawdown': np.random.uniform(0.1, 0.3),  # 最大回撤
                'stress_test_score': np.random.uniform(0.2, 0.7)  # 压力测试分数
            }
            
            # 缓存数据
            cache_key = f"stress_test_{','.join(sorted(symbols))}_{datetime.now().strftime('%Y%m%d%H')}"
            self.risk_scenarios_cache[cache_key] = stress_test
            
            return stress_test
            
        except Exception as e:
            logger.error(f"Failed to perform stress test: {e}")
            return {}
    
    async def _calculate_var_cvar(self, symbols: List[str], market_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """计算VaR和CVaR"""
        try:
            # 模拟VaR和CVaR计算
            var_analysis = {
                'var_95': np.random.uniform(0.02, 0.08),  # 95% VaR
                'var_99': np.random.uniform(0.03, 0.12),  # 99% VaR
                'cvar_95': np.random.uniform(0.03, 0.10),  # 95% CVaR
                'cvar_99': np.random.uniform(0.04, 0.15),  # 99% CVaR
                'var_method': 'historical_simulation',  # VaR计算方法
                'confidence_level': 0.95,  # 置信水平
                'holding_period': 1,  # 持有期（天）
                'var_breach_count': np.random.randint(0, 5),  # VaR突破次数
                'var_breach_rate': np.random.uniform(0.01, 0.05),  # VaR突破率
                'tail_risk': np.random.uniform(0.02, 0.08),  # 尾部风险
                'overall_var_score': np.random.uniform(0.1, 0.6)  # 综合VaR分数
            }
            
            # 缓存数据
            cache_key = f"var_{','.join(sorted(symbols))}_{datetime.now().strftime('%Y%m%d%H')}"
            self.var_cache[cache_key] = var_analysis
            
            return var_analysis
            
        except Exception as e:
            logger.error(f"Failed to calculate VaR/CVaR: {e}")
            return {}
    
    def _assess_overall_risk(
        self, 
        market_risk: Dict[str, Any], 
        credit_risk: Dict[str, Any], 
        liquidity_risk: Dict[str, Any],
        operational_risk: Dict[str, Any], 
        stress_test: Dict[str, Any], 
        var_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """综合风险评估"""
        try:
            # 计算各类风险分数
            risk_scores = {
                'market_risk': market_risk.get('overall_market_risk', 0.5),
                'credit_risk': credit_risk.get('overall_credit_risk', 0.5),
                'liquidity_risk': liquidity_risk.get('overall_liquidity_risk', 0.5),
                'operational_risk': operational_risk.get('overall_operational_risk', 0.5),
                'stress_test': stress_test.get('stress_test_score', 0.5),
                'var_risk': var_analysis.get('overall_var_score', 0.5)
            }
            
            # 计算加权平均风险分数
            weights = {
                'market_risk': 0.3,
                'credit_risk': 0.2,
                'liquidity_risk': 0.15,
                'operational_risk': 0.15,
                'stress_test': 0.1,
                'var_risk': 0.1
            }
            
            overall_risk_score = sum(
                risk_scores[risk_type] * weights[risk_type] 
                for risk_type in risk_scores.keys()
            )
            
            # 确定风险等级
            if overall_risk_score > 0.7:
                risk_level = "high"
            elif overall_risk_score > 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            # 计算风险分散度
            risk_dispersion = np.std(list(risk_scores.values()))
            
            # 识别主要风险
            main_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                'overall_risk_score': overall_risk_score,
                'risk_level': risk_level,
                'risk_scores': risk_scores,
                'risk_weights': weights,
                'risk_dispersion': risk_dispersion,
                'main_risks': main_risks,
                'risk_breakdown': {
                    'market_risk': market_risk,
                    'credit_risk': credit_risk,
                    'liquidity_risk': liquidity_risk,
                    'operational_risk': operational_risk,
                    'stress_test': stress_test,
                    'var_analysis': var_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Overall risk assessment failed: {e}")
            return {
                'overall_risk_score': 0.5,
                'risk_level': 'medium',
                'error': str(e)
            }
    
    def _generate_recommendation(
        self, 
        overall_risk: Dict[str, Any], 
        symbols: List[str]
    ) -> AgentRecommendation:
        """生成风险管理推荐"""
        try:
            risk_score = overall_risk.get('overall_risk_score', 0.5)
            risk_level = overall_risk.get('risk_level', 'medium')
            main_risks = overall_risk.get('main_risks', [])
            
            # 根据风险等级确定推荐
            if risk_score > 0.7:
                recommendation_type = RecommendationType.SELL
                confidence = min(risk_score, 0.9)
                reasoning = f"风险水平过高({risk_level})，主要风险：{', '.join([r[0] for r in main_risks[:2]])}"
            elif risk_score < 0.3:
                recommendation_type = RecommendationType.BUY
                confidence = min(1 - risk_score, 0.9)
                reasoning = f"风险水平较低({risk_level})，风险控制良好"
            else:
                recommendation_type = RecommendationType.HOLD
                confidence = 0.6
                reasoning = f"风险水平中等({risk_level})，建议谨慎操作"
            
            # 计算目标价格（基于风险调整）
            target_price = None
            stop_loss = None
            take_profit = None
            
            if recommendation_type in [RecommendationType.BUY, RecommendationType.STRONG_BUY]:
                base_price = 12.0  # 示例基础价格
                risk_adjustment = 1 - risk_score * 0.2  # 风险调整
                target_price = base_price * risk_adjustment
                stop_loss = target_price * (1 - risk_score * 0.1)  # 基于风险设置止损
                take_profit = target_price * (1 + (1 - risk_score) * 0.1)  # 基于风险设置止盈
            
            return AgentRecommendation(
                recommendation_type=recommendation_type,
                confidence=confidence,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                supporting_data={
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'main_risks': main_risks,
                    'risk_breakdown': overall_risk.get('risk_breakdown', {})
                },
                risk_level=risk_level,
                time_horizon="medium"
            )
            
        except Exception as e:
            logger.error(f"Failed to generate risk recommendation: {e}")
            return AgentRecommendation(
                recommendation_type=RecommendationType.HOLD,
                confidence=0.3,
                reasoning=f"风险管理推荐生成失败: {e}",
                risk_level="high"
            )
    
    def _extract_key_factors(self, overall_risk: Dict[str, Any]) -> List[str]:
        """提取关键因素"""
        factors = []
        
        risk_level = overall_risk.get('risk_level', 'medium')
        main_risks = overall_risk.get('main_risks', [])
        
        # 风险等级因素
        if risk_level == "low":
            factors.append("整体风险水平较低")
        elif risk_level == "high":
            factors.append("整体风险水平较高")
        
        # 主要风险因素
        for risk_type, score in main_risks[:3]:
            if score > 0.6:
                risk_names = {
                    'market_risk': '市场风险',
                    'credit_risk': '信用风险',
                    'liquidity_risk': '流动性风险',
                    'operational_risk': '操作风险',
                    'stress_test': '压力测试风险',
                    'var_risk': 'VaR风险'
                }
                factors.append(f"{risk_names.get(risk_type, risk_type)}较高")
        
        # 风险分散度因素
        risk_dispersion = overall_risk.get('risk_dispersion', 0.0)
        if risk_dispersion > 0.2:
            factors.append("风险分散度较高")
        elif risk_dispersion < 0.1:
            factors.append("风险分散度较低")
        
        return factors[:5]
    
    def _extract_risk_factors(self, overall_risk: Dict[str, Any]) -> List[str]:
        """提取风险因素"""
        risks = []
        
        risk_breakdown = overall_risk.get('risk_breakdown', {})
        
        # 市场风险因素
        market_risk = risk_breakdown.get('market_risk', {})
        if market_risk.get('volatility', 0.2) > 0.3:
            risks.append("市场波动率过高")
        if market_risk.get('beta', 1.0) > 1.2:
            risks.append("Beta系数过高，系统性风险大")
        
        # 信用风险因素
        credit_risk = risk_breakdown.get('credit_risk', {})
        if credit_risk.get('default_probability', 0.02) > 0.03:
            risks.append("违约概率较高")
        if credit_risk.get('credit_rating', 'A') in ['BB', 'B', 'CCC']:
            risks.append("信用评级较低")
        
        # 流动性风险因素
        liquidity_risk = risk_breakdown.get('liquidity_risk', {})
        if liquidity_risk.get('bid_ask_spread', 0.005) > 0.01:
            risks.append("买卖价差过大")
        if liquidity_risk.get('volume_risk', 0.2) > 0.3:
            risks.append("成交量风险较高")
        
        # 操作风险因素
        operational_risk = risk_breakdown.get('operational_risk', {})
        if operational_risk.get('model_risk', 0.1) > 0.15:
            risks.append("模型风险较高")
        if operational_risk.get('regulatory_risk', 0.05) > 0.08:
            risks.append("监管风险较高")
        
        return risks[:5]
    
    def _generate_market_outlook(self, overall_risk: Dict[str, Any]) -> str:
        """生成市场展望"""
        risk_level = overall_risk.get('risk_level', 'medium')
        risk_score = overall_risk.get('overall_risk_score', 0.5)
        main_risks = overall_risk.get('main_risks', [])
        
        if risk_level == "low":
            return f"整体风险水平较低(风险分数: {risk_score:.2f})，风险控制措施有效，市场环境相对稳定，适合投资。"
        elif risk_level == "high":
            main_risk_names = [r[0] for r in main_risks[:2]]
            return f"整体风险水平较高(风险分数: {risk_score:.2f})，主要风险包括{', '.join(main_risk_names)}，建议谨慎投资或降低仓位。"
        else:
            return f"整体风险水平中等(风险分数: {risk_score:.2f})，需要密切关注风险变化，建议适度控制仓位规模。"
    
    def _create_default_analysis(self, symbols: List[str], error_msg: str) -> AgentAnalysis:
        """创建默认分析结果"""
        return AgentAnalysis(
            agent_name=self.name,
            agent_type=self.agent_type,
            analysis_id=create_analysis_id(),
            timestamp=datetime.now(),
            symbols=symbols,
            recommendation=AgentRecommendation(
                recommendation_type=RecommendationType.HOLD,
                confidence=0.3,
                reasoning=f"风险分析失败: {error_msg}",
                risk_level="high"
            ),
            key_factors=["分析失败"],
            risk_factors=["数据获取失败", "风险分析错误"],
            market_outlook="无法提供市场展望",
            additional_insights={'error': error_msg}
        )
    
    def _get_current_risk_level(self) -> float:
        """获取当前风险水平"""
        # 从缓存中获取最新的风险数据
        if self.risk_metrics_cache:
            latest_data = max(self.risk_metrics_cache.values(), key=lambda x: x.get('timestamp', datetime.min))
            return latest_data.get('overall_market_risk', 0.5)
        return 0.5
    
    def _get_current_risk_factors(self) -> List[str]:
        """获取当前风险因素"""
        return ["市场风险", "信用风险", "流动性风险"]
    
    def _get_var_estimate(self) -> Dict[str, float]:
        """获取VaR估计"""
        if self.var_cache:
            latest_data = max(self.var_cache.values(), key=lambda x: x.get('timestamp', datetime.min))
            return {
                'var_95': latest_data.get('var_95', 0.05),
                'var_99': latest_data.get('var_99', 0.08)
            }
        return {'var_95': 0.05, 'var_99': 0.08}
    
    def _generate_counter_arguments(self, other_analyses: List[AgentAnalysis]) -> List[str]:
        """生成反驳论点"""
        counter_arguments = []
        
        for analysis in other_analyses:
            if analysis.agent_type == 'fundamental_analyst':
                counter_arguments.append("基本面分析可能忽略了风险因素")
            elif analysis.agent_type == 'technical_analyst':
                counter_arguments.append("技术分析可能低估了市场风险")
            elif analysis.agent_type == 'sentiment_analyst':
                counter_arguments.append("情绪分析可能过于乐观")
            elif analysis.agent_type == 'news_analyst':
                counter_arguments.append("新闻分析可能忽略了系统性风险")
        
        return counter_arguments[:3]
    
    def _calculate_debate_confidence(self, other_analyses: List[AgentAnalysis]) -> float:
        """计算辩论后的置信度"""
        base_confidence = self.get_confidence_score()
        
        # 风险管理通常具有较高的置信度
        risk_analyses = [a for a in other_analyses if a.agent_type == 'risk_manager']
        
        if len(risk_analyses) > 0:
            # 如果有其他风险分析，考虑一致性
            avg_confidence = sum(a.recommendation.confidence for a in risk_analyses) / len(risk_analyses)
            return (base_confidence + avg_confidence) / 2
        else:
            # 如果没有其他风险分析，保持较高置信度
            return min(base_confidence * 1.1, 1.0)
