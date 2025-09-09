"""
基本面分析师
分析公司财务数据、行业趋势和宏观经济因素
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from .base_agent import BaseAgent, AgentAnalysis, AgentRecommendation, RecommendationType, DebateContext, DebateResponse, create_analysis_id
from common.logging_system import setup_logger

logger = setup_logger("fundamental_analyst")


class FundamentalAnalyst(BaseAgent):
    """基本面分析师"""
    
    def __init__(self):
        super().__init__(
            name="基本面分析师",
            agent_type="fundamental_analyst",
            expertise="财务分析, 行业分析, 宏观经济分析, 估值分析",
            confidence_threshold=0.7
        )
        self.financial_metrics_cache = {}
        self.industry_data_cache = {}
        self.macro_indicators_cache = {}
    
    async def analyze(
        self, 
        symbols: List[str], 
        market_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """分析基本面数据"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting fundamental analysis for symbols: {symbols}")
            
            # 1. 获取财务数据
            financial_data = await self._get_financial_data(symbols)
            
            # 2. 获取行业数据
            industry_data = await self._get_industry_data(symbols)
            
            # 3. 获取宏观经济指标
            macro_data = await self._get_macro_indicators()
            
            # 4. 进行估值分析
            valuation_analysis = self._perform_valuation_analysis(financial_data, industry_data)
            
            # 5. 生成推荐
            recommendation = self._generate_recommendation(
                financial_data, industry_data, macro_data, valuation_analysis
            )
            
            # 6. 计算分析耗时
            analysis_duration = (datetime.now() - start_time).total_seconds()
            
            # 7. 创建分析结果
            analysis = AgentAnalysis(
                agent_name=self.name,
                agent_type=self.agent_type,
                analysis_id=create_analysis_id(),
                timestamp=datetime.now(),
                symbols=symbols,
                recommendation=recommendation,
                key_factors=self._extract_key_factors(financial_data, industry_data, valuation_analysis),
                risk_factors=self._extract_risk_factors(financial_data, industry_data, macro_data),
                market_outlook=self._generate_market_outlook(macro_data, industry_data),
                additional_insights={
                    'financial_metrics': financial_data,
                    'industry_analysis': industry_data,
                    'macro_indicators': macro_data,
                    'valuation_analysis': valuation_analysis,
                    'pe_ratio': valuation_analysis.get('pe_ratio', 0),
                    'pb_ratio': valuation_analysis.get('pb_ratio', 0),
                    'roe': financial_data.get('roe', 0),
                    'debt_ratio': financial_data.get('debt_ratio', 0)
                },
                data_sources=['财务报表', '行业数据', '宏观经济指标'],
                analysis_duration=analysis_duration
            )
            
            logger.info(f"Fundamental analysis completed for {symbols} in {analysis_duration:.2f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Fundamental analysis failed for {symbols}: {e}")
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
            
            # 基于基本面分析生成辩论响应
            if len(supporting_views) > len(opposing_views):
                position = "支持"
                arguments = [
                    "基本面数据支持投资价值",
                    "行业前景良好",
                    "估值水平合理"
                ]
            elif len(opposing_views) > len(supporting_views):
                position = "反对"
                arguments = [
                    "基本面数据存在风险",
                    "行业面临挑战",
                    "估值水平偏高"
                ]
            else:
                position = "中立"
                arguments = [
                    "基本面数据中性",
                    "需要更多财务数据支持",
                    "建议关注季度财报"
                ]
            
            return DebateResponse(
                agent_name=self.name,
                position=position,
                arguments=arguments,
                evidence={
                    'financial_health': self._assess_financial_health(),
                    'industry_outlook': self._assess_industry_outlook(),
                    'valuation_metrics': self._get_valuation_metrics()
                },
                counter_arguments=self._generate_counter_arguments(other_analyses),
                updated_confidence=self._calculate_debate_confidence(other_analyses),
                willingness_to_compromise=0.6
            )
            
        except Exception as e:
            logger.error(f"Debate failed for {self.name}: {e}")
            return DebateResponse(
                agent_name=self.name,
                position="中立",
                arguments=["基本面分析过程中出现错误"],
                evidence={},
                updated_confidence=0.3
            )
    
    async def _get_financial_data(self, symbols: List[str]) -> Dict[str, Any]:
        """获取财务数据"""
        try:
            # 这里应该从实际数据源获取财务数据
            # 目前使用模拟数据
            
            financial_data = {}
            for symbol in symbols:
                # 模拟财务指标
                financial_data[symbol] = {
                    'revenue': np.random.uniform(1000, 10000),  # 营业收入（万元）
                    'net_profit': np.random.uniform(100, 1000),  # 净利润（万元）
                    'total_assets': np.random.uniform(5000, 50000),  # 总资产（万元）
                    'total_liabilities': np.random.uniform(2000, 20000),  # 总负债（万元）
                    'roe': np.random.uniform(0.05, 0.25),  # ROE
                    'roa': np.random.uniform(0.03, 0.15),  # ROA
                    'debt_ratio': np.random.uniform(0.2, 0.6),  # 负债率
                    'current_ratio': np.random.uniform(1.0, 3.0),  # 流动比率
                    'quick_ratio': np.random.uniform(0.8, 2.5),  # 速动比率
                    'gross_margin': np.random.uniform(0.2, 0.5),  # 毛利率
                    'net_margin': np.random.uniform(0.05, 0.2),  # 净利率
                    'eps': np.random.uniform(0.5, 3.0),  # 每股收益
                    'book_value': np.random.uniform(5, 20),  # 每股净资产
                    'revenue_growth': np.random.uniform(-0.1, 0.3),  # 营收增长率
                    'profit_growth': np.random.uniform(-0.2, 0.4),  # 利润增长率
                }
            
            # 缓存数据
            cache_key = f"financial_{','.join(sorted(symbols))}_{datetime.now().strftime('%Y%m%d')}"
            self.financial_metrics_cache[cache_key] = financial_data
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Failed to get financial data: {e}")
            return {}
    
    async def _get_industry_data(self, symbols: List[str]) -> Dict[str, Any]:
        """获取行业数据"""
        try:
            # 模拟行业数据
            industry_data = {
                'industry_name': '金融服务',
                'market_size': 500000,  # 市场规模（亿元）
                'growth_rate': 0.08,  # 行业增长率
                'competition_level': 'high',  # 竞争程度
                'barriers_to_entry': 'high',  # 进入壁垒
                'regulatory_environment': 'strict',  # 监管环境
                'technology_trend': 'digitalization',  # 技术趋势
                'key_drivers': [
                    '数字化转型',
                    '监管政策变化',
                    '利率环境',
                    '经济周期'
                ],
                'risks': [
                    '监管风险',
                    '信用风险',
                    '市场风险',
                    '技术风险'
                ],
                'pe_ratio_industry': 12.5,  # 行业平均PE
                'pb_ratio_industry': 1.2,  # 行业平均PB
                'roe_industry': 0.12,  # 行业平均ROE
            }
            
            # 缓存数据
            cache_key = f"industry_{datetime.now().strftime('%Y%m%d')}"
            self.industry_data_cache[cache_key] = industry_data
            
            return industry_data
            
        except Exception as e:
            logger.error(f"Failed to get industry data: {e}")
            return {}
    
    async def _get_macro_indicators(self) -> Dict[str, Any]:
        """获取宏观经济指标"""
        try:
            # 模拟宏观经济数据
            macro_data = {
                'gdp_growth': 0.05,  # GDP增长率
                'inflation_rate': 0.02,  # 通胀率
                'interest_rate': 0.035,  # 基准利率
                'unemployment_rate': 0.05,  # 失业率
                'currency_rate': 7.2,  # 汇率
                'stock_market_index': 3200,  # 股市指数
                'bond_yield': 0.03,  # 债券收益率
                'money_supply_growth': 0.08,  # 货币供应量增长率
                'fiscal_policy': 'neutral',  # 财政政策
                'monetary_policy': 'neutral',  # 货币政策
                'economic_cycle': 'recovery',  # 经济周期
                'business_confidence': 0.6,  # 商业信心指数
                'consumer_confidence': 0.65,  # 消费者信心指数
            }
            
            # 缓存数据
            cache_key = f"macro_{datetime.now().strftime('%Y%m%d')}"
            self.macro_indicators_cache[cache_key] = macro_data
            
            return macro_data
            
        except Exception as e:
            logger.error(f"Failed to get macro indicators: {e}")
            return {}
    
    def _perform_valuation_analysis(
        self, 
        financial_data: Dict[str, Any], 
        industry_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """进行估值分析"""
        valuation_analysis = {}
        
        for symbol, metrics in financial_data.items():
            # 计算估值指标
            pe_ratio = 15.0  # 模拟PE比率
            pb_ratio = 1.5   # 模拟PB比率
            peg_ratio = pe_ratio / (metrics.get('profit_growth', 0.1) * 100)  # PEG比率
            
            # 计算内在价值（简化DCF模型）
            intrinsic_value = self._calculate_intrinsic_value(metrics)
            
            # 估值结论
            if pe_ratio < industry_data.get('pe_ratio_industry', 15) * 0.8:
                valuation_conclusion = "低估"
            elif pe_ratio > industry_data.get('pe_ratio_industry', 15) * 1.2:
                valuation_conclusion = "高估"
            else:
                valuation_conclusion = "合理"
            
            valuation_analysis[symbol] = {
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'peg_ratio': peg_ratio,
                'intrinsic_value': intrinsic_value,
                'current_price': 12.0,  # 模拟当前价格
                'valuation_conclusion': valuation_conclusion,
                'margin_of_safety': (intrinsic_value - 12.0) / intrinsic_value if intrinsic_value > 0 else 0
            }
        
        return valuation_analysis
    
    def _calculate_intrinsic_value(self, metrics: Dict[str, Any]) -> float:
        """计算内在价值（简化DCF模型）"""
        try:
            # 获取基础数据
            eps = metrics.get('eps', 1.0)
            growth_rate = metrics.get('profit_growth', 0.1)
            roe = metrics.get('roe', 0.12)
            
            # 简化DCF计算
            # 假设未来5年增长率，然后永续增长
            future_eps = eps * (1 + growth_rate) ** 5
            terminal_value = future_eps / (0.1 - 0.03)  # 假设折现率10%，永续增长率3%
            
            # 折现到现值
            discount_rate = 0.1
            intrinsic_value = terminal_value / (1 + discount_rate) ** 5
            
            return max(intrinsic_value, 0)
            
        except Exception as e:
            logger.error(f"Failed to calculate intrinsic value: {e}")
            return 0.0
    
    def _generate_recommendation(
        self, 
        financial_data: Dict[str, Any], 
        industry_data: Dict[str, Any], 
        macro_data: Dict[str, Any],
        valuation_analysis: Dict[str, Any]
    ) -> AgentRecommendation:
        """生成投资推荐"""
        try:
            # 综合评分
            total_score = 0
            factor_count = 0
            
            # 财务健康度评分
            for symbol, metrics in financial_data.items():
                financial_score = self._calculate_financial_score(metrics)
                total_score += financial_score
                factor_count += 1
            
            # 行业前景评分
            industry_score = self._calculate_industry_score(industry_data)
            total_score += industry_score
            factor_count += 1
            
            # 宏观经济评分
            macro_score = self._calculate_macro_score(macro_data)
            total_score += macro_score
            factor_count += 1
            
            # 估值评分
            valuation_score = self._calculate_valuation_score(valuation_analysis)
            total_score += valuation_score
            factor_count += 1
            
            # 计算综合评分
            if factor_count > 0:
                avg_score = total_score / factor_count
            else:
                avg_score = 0.5
            
            # 根据评分确定推荐
            if avg_score > 0.7:
                recommendation_type = RecommendationType.BUY
                confidence = min(avg_score, 0.9)
                reasoning = "基本面数据强劲，估值合理，行业前景良好"
            elif avg_score < 0.3:
                recommendation_type = RecommendationType.SELL
                confidence = min(1 - avg_score, 0.9)
                reasoning = "基本面数据疲弱，估值偏高，存在风险因素"
            else:
                recommendation_type = RecommendationType.HOLD
                confidence = 0.6
                reasoning = "基本面数据中性，建议观望等待更好时机"
            
            # 计算目标价格
            target_price = None
            stop_loss = None
            take_profit = None
            
            if recommendation_type in [RecommendationType.BUY, RecommendationType.STRONG_BUY]:
                # 基于内在价值计算目标价格
                for symbol, valuation in valuation_analysis.items():
                    if valuation.get('intrinsic_value', 0) > 0:
                        target_price = valuation['intrinsic_value']
                        stop_loss = target_price * 0.85  # 15%止损
                        take_profit = target_price * 1.2  # 20%止盈
                        break
            
            return AgentRecommendation(
                recommendation_type=recommendation_type,
                confidence=confidence,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                supporting_data={
                    'financial_score': financial_score if 'financial_score' in locals() else 0,
                    'industry_score': industry_score,
                    'macro_score': macro_score,
                    'valuation_score': valuation_score,
                    'overall_score': avg_score
                },
                risk_level=self._assess_risk_level(avg_score),
                time_horizon="long"
            )
            
        except Exception as e:
            logger.error(f"Failed to generate recommendation: {e}")
            return AgentRecommendation(
                recommendation_type=RecommendationType.HOLD,
                confidence=0.3,
                reasoning=f"推荐生成失败: {e}",
                risk_level="high"
            )
    
    def _calculate_financial_score(self, metrics: Dict[str, Any]) -> float:
        """计算财务健康度评分"""
        score = 0.5  # 基础分
        
        # ROE评分
        roe = metrics.get('roe', 0)
        if roe > 0.15:
            score += 0.2
        elif roe > 0.1:
            score += 0.1
        elif roe < 0.05:
            score -= 0.2
        
        # 负债率评分
        debt_ratio = metrics.get('debt_ratio', 0.5)
        if debt_ratio < 0.3:
            score += 0.1
        elif debt_ratio > 0.6:
            score -= 0.1
        
        # 增长率评分
        profit_growth = metrics.get('profit_growth', 0)
        if profit_growth > 0.2:
            score += 0.2
        elif profit_growth > 0.1:
            score += 0.1
        elif profit_growth < 0:
            score -= 0.2
        
        return max(0, min(1, score))
    
    def _calculate_industry_score(self, industry_data: Dict[str, Any]) -> float:
        """计算行业前景评分"""
        score = 0.5  # 基础分
        
        # 增长率评分
        growth_rate = industry_data.get('growth_rate', 0.05)
        if growth_rate > 0.1:
            score += 0.2
        elif growth_rate > 0.05:
            score += 0.1
        elif growth_rate < 0:
            score -= 0.2
        
        # 竞争程度评分
        competition = industry_data.get('competition_level', 'medium')
        if competition == 'low':
            score += 0.1
        elif competition == 'high':
            score -= 0.1
        
        # 监管环境评分
        regulatory = industry_data.get('regulatory_environment', 'neutral')
        if regulatory == 'favorable':
            score += 0.1
        elif regulatory == 'strict':
            score -= 0.1
        
        return max(0, min(1, score))
    
    def _calculate_macro_score(self, macro_data: Dict[str, Any]) -> float:
        """计算宏观经济评分"""
        score = 0.5  # 基础分
        
        # GDP增长率评分
        gdp_growth = macro_data.get('gdp_growth', 0.05)
        if gdp_growth > 0.06:
            score += 0.2
        elif gdp_growth > 0.04:
            score += 0.1
        elif gdp_growth < 0.02:
            score -= 0.2
        
        # 利率环境评分
        interest_rate = macro_data.get('interest_rate', 0.035)
        if interest_rate < 0.03:
            score += 0.1  # 低利率有利于股市
        elif interest_rate > 0.05:
            score -= 0.1
        
        # 经济周期评分
        economic_cycle = macro_data.get('economic_cycle', 'neutral')
        if economic_cycle == 'expansion':
            score += 0.2
        elif economic_cycle == 'recession':
            score -= 0.2
        
        return max(0, min(1, score))
    
    def _calculate_valuation_score(self, valuation_analysis: Dict[str, Any]) -> float:
        """计算估值评分"""
        if not valuation_analysis:
            return 0.5
        
        total_score = 0
        count = 0
        
        for symbol, valuation in valuation_analysis.items():
            score = 0.5  # 基础分
            
            # 估值结论评分
            conclusion = valuation.get('valuation_conclusion', '合理')
            if conclusion == '低估':
                score += 0.3
            elif conclusion == '高估':
                score -= 0.3
            
            # 安全边际评分
            margin = valuation.get('margin_of_safety', 0)
            if margin > 0.2:
                score += 0.2
            elif margin > 0.1:
                score += 0.1
            elif margin < -0.1:
                score -= 0.2
            
            total_score += score
            count += 1
        
        return total_score / count if count > 0 else 0.5
    
    def _assess_risk_level(self, score: float) -> str:
        """评估风险等级"""
        if score > 0.7:
            return "low"
        elif score > 0.4:
            return "medium"
        else:
            return "high"
    
    def _extract_key_factors(
        self, 
        financial_data: Dict[str, Any], 
        industry_data: Dict[str, Any], 
        valuation_analysis: Dict[str, Any]
    ) -> List[str]:
        """提取关键因素"""
        factors = []
        
        # 财务因素
        for symbol, metrics in financial_data.items():
            if metrics.get('roe', 0) > 0.15:
                factors.append(f"{symbol} ROE表现优异")
            if metrics.get('profit_growth', 0) > 0.2:
                factors.append(f"{symbol} 利润增长强劲")
            if metrics.get('debt_ratio', 0.5) < 0.3:
                factors.append(f"{symbol} 负债率较低")
        
        # 行业因素
        if industry_data.get('growth_rate', 0) > 0.1:
            factors.append("行业增长前景良好")
        if industry_data.get('competition_level') == 'low':
            factors.append("行业竞争程度较低")
        
        # 估值因素
        for symbol, valuation in valuation_analysis.items():
            if valuation.get('valuation_conclusion') == '低估':
                factors.append(f"{symbol} 估值偏低")
        
        return factors[:5]
    
    def _extract_risk_factors(
        self, 
        financial_data: Dict[str, Any], 
        industry_data: Dict[str, Any], 
        macro_data: Dict[str, Any]
    ) -> List[str]:
        """提取风险因素"""
        risks = []
        
        # 财务风险
        for symbol, metrics in financial_data.items():
            if metrics.get('debt_ratio', 0.5) > 0.6:
                risks.append(f"{symbol} 负债率较高")
            if metrics.get('profit_growth', 0) < 0:
                risks.append(f"{symbol} 利润负增长")
            if metrics.get('current_ratio', 1.5) < 1.0:
                risks.append(f"{symbol} 流动性风险")
        
        # 行业风险
        risks.extend(industry_data.get('risks', []))
        
        # 宏观风险
        if macro_data.get('gdp_growth', 0.05) < 0.03:
            risks.append("经济增长放缓")
        if macro_data.get('inflation_rate', 0.02) > 0.05:
            risks.append("通胀压力上升")
        
        return risks[:5]
    
    def _generate_market_outlook(
        self, 
        macro_data: Dict[str, Any], 
        industry_data: Dict[str, Any]
    ) -> str:
        """生成市场展望"""
        gdp_growth = macro_data.get('gdp_growth', 0.05)
        industry_growth = industry_data.get('growth_rate', 0.08)
        
        if gdp_growth > 0.06 and industry_growth > 0.1:
            return "宏观经济环境良好，行业增长强劲，基本面支撑市场上涨。"
        elif gdp_growth < 0.03 or industry_growth < 0.05:
            return "宏观经济面临挑战，行业增长放缓，基本面存在压力。"
        else:
            return "宏观经济环境稳定，行业增长适中，基本面相对中性。"
    
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
                reasoning=f"基本面分析失败: {error_msg}",
                risk_level="high"
            ),
            key_factors=["分析失败"],
            risk_factors=["数据获取失败", "财务分析错误"],
            market_outlook="无法提供市场展望",
            additional_insights={'error': error_msg}
        )
    
    def _assess_financial_health(self) -> Dict[str, Any]:
        """评估财务健康状况"""
        return {
            'overall_health': 'good',
            'liquidity': 'adequate',
            'solvency': 'strong',
            'profitability': 'moderate'
        }
    
    def _assess_industry_outlook(self) -> Dict[str, Any]:
        """评估行业前景"""
        return {
            'growth_prospects': 'positive',
            'competitive_position': 'stable',
            'regulatory_environment': 'neutral'
        }
    
    def _get_valuation_metrics(self) -> Dict[str, Any]:
        """获取估值指标"""
        return {
            'pe_ratio': 15.0,
            'pb_ratio': 1.5,
            'valuation_level': 'fair'
        }
    
    def _generate_counter_arguments(self, other_analyses: List[AgentAnalysis]) -> List[str]:
        """生成反驳论点"""
        counter_arguments = []
        
        for analysis in other_analyses:
            if analysis.agent_type == 'technical_analyst':
                counter_arguments.append("技术分析可能忽略了基本面变化")
            elif analysis.agent_type == 'news_analyst':
                counter_arguments.append("新闻情绪可能过于短期化")
            elif analysis.agent_type == 'sentiment_analyst':
                counter_arguments.append("市场情绪可能偏离基本面价值")
        
        return counter_arguments[:3]
    
    def _calculate_debate_confidence(self, other_analyses: List[AgentAnalysis]) -> float:
        """计算辩论后的置信度"""
        base_confidence = self.get_confidence_score()
        
        # 基本面分析通常具有较高的长期置信度
        fundamental_analyses = [a for a in other_analyses if a.agent_type == 'fundamental_analyst']
        
        if len(fundamental_analyses) > 0:
            # 如果有其他基本面分析，考虑一致性
            avg_confidence = sum(a.recommendation.confidence for a in fundamental_analyses) / len(fundamental_analyses)
            return (base_confidence + avg_confidence) / 2
        else:
            # 如果没有其他基本面分析，保持较高置信度
            return min(base_confidence * 1.1, 1.0)
