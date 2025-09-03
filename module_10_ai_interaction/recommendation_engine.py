"""
推荐引擎模块
负责生成个性化的投资推荐
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("recommendation_engine")


@dataclass
class InvestmentRecommendation:
    """投资推荐数据结构"""

    recommendation_id: str
    timestamp: datetime
    recommendation_type: str  # 'portfolio', 'strategy', 'asset', 'risk_adjustment'
    title: str
    description: str
    confidence_score: float
    expected_return: Optional[float] = None
    risk_level: Optional[str] = None
    time_horizon: Optional[str] = None
    action_items: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    alternatives: List["InvestmentRecommendation"] = field(default_factory=list)


@dataclass
class PortfolioRecommendation:
    """组合推荐数据结构"""

    portfolio_id: str
    name: str
    description: str
    asset_allocation: Dict[str, float]
    expected_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    suitability_score: float
    pros: List[str]
    cons: List[str]
    implementation_steps: List[str]


class RecommendationEngine:
    """推荐引擎类"""

    # 预定义的投资组合模板
    PORTFOLIO_TEMPLATES = {
        "conservative_balanced": {
            "name": "稳健平衡型组合",
            "description": "适合风险承受能力较低的投资者，注重资产保值",
            "allocation": {
                "bonds": 0.60,
                "stocks": 0.30,
                "commodities": 0.05,
                "cash": 0.05,
            },
            "expected_return": 0.06,
            "volatility": 0.08,
        },
        "moderate_growth": {
            "name": "稳健成长型组合",
            "description": "平衡风险与收益，适合中等风险偏好的投资者",
            "allocation": {
                "stocks": 0.50,
                "bonds": 0.35,
                "commodities": 0.10,
                "cash": 0.05,
            },
            "expected_return": 0.10,
            "volatility": 0.12,
        },
        "aggressive_growth": {
            "name": "积极成长型组合",
            "description": "追求高收益，能承受较大波动的投资者",
            "allocation": {
                "stocks": 0.70,
                "bonds": 0.15,
                "commodities": 0.10,
                "alternative": 0.05,
            },
            "expected_return": 0.15,
            "volatility": 0.20,
        },
        "income_focused": {
            "name": "收益导向型组合",
            "description": "注重稳定现金流收入",
            "allocation": {
                "dividend_stocks": 0.40,
                "bonds": 0.40,
                "reits": 0.15,
                "cash": 0.05,
            },
            "expected_return": 0.08,
            "volatility": 0.10,
        },
    }

    # 策略推荐规则
    STRATEGY_RULES = {
        "bull_market": ["momentum", "growth", "trend_following"],
        "bear_market": ["defensive", "value", "mean_reversion"],
        "sideways_market": ["range_trading", "arbitrage", "neutral"],
        "high_volatility": ["volatility_trading", "options", "hedging"],
        "low_volatility": ["carry_trade", "yield_enhancement"],
    }

    def __init__(self):
        """初始化推荐引擎"""
        self.recommendation_history: List[InvestmentRecommendation] = []
        self.user_feedback: Dict[str, Any] = {}

    def generate_portfolio_recommendations(
        self,
        user_profile: Dict[str, Any],
        market_conditions: Dict[str, Any],
        num_recommendations: int = 3,
    ) -> List[PortfolioRecommendation]:
        """生成投资组合推荐

        Args:
            user_profile: 用户画像
            market_conditions: 市场状况
            num_recommendations: 推荐数量

        Returns:
            组合推荐列表
        """
        recommendations = []

        # 获取用户风险偏好
        risk_tolerance = user_profile.get("risk_tolerance", "moderate")
        investment_horizon = user_profile.get("investment_horizon", "medium_term")
        investment_goals = user_profile.get("goals", [])

        # 筛选合适的组合模板
        suitable_templates = self._filter_suitable_templates(
            risk_tolerance, investment_horizon, investment_goals
        )

        # 生成推荐
        for template_key, template in suitable_templates[:num_recommendations]:
            # 根据市场条件调整配置
            adjusted_allocation = self._adjust_allocation_for_market(
                template["allocation"], market_conditions
            )

            # 计算适合度分数
            suitability_score = self._calculate_suitability_score(
                template, user_profile, market_conditions
            )

            # 生成优缺点分析
            pros, cons = self._analyze_portfolio_pros_cons(
                template, user_profile, market_conditions
            )

            recommendation = PortfolioRecommendation(
                portfolio_id=f"portfolio_{template_key}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                name=template["name"],
                description=template["description"],
                asset_allocation=adjusted_allocation,
                expected_metrics={
                    "expected_return": template["expected_return"],
                    "volatility": template["volatility"],
                    "sharpe_ratio": template["expected_return"]
                    / template["volatility"],
                },
                risk_metrics={
                    "var_95": template["volatility"] * 1.65,
                    "max_drawdown": template["volatility"] * 2.5,
                    "beta": 1.0,  # 简化处理
                },
                suitability_score=suitability_score,
                pros=pros,
                cons=cons,
                implementation_steps=self._generate_implementation_steps(template_key),
            )

            recommendations.append(recommendation)

        return recommendations

    def generate_strategy_recommendations(
        self,
        market_analysis: Dict[str, Any],
        current_portfolio: Dict[str, Any],
        risk_constraints: Dict[str, Any],
    ) -> List[InvestmentRecommendation]:
        """生成策略推荐

        Args:
            market_analysis: 市场分析结果
            current_portfolio: 当前组合
            risk_constraints: 风险约束

        Returns:
            策略推荐列表
        """
        recommendations = []

        # 识别市场状态
        market_regime = market_analysis.get("regime", "unknown")
        volatility_level = market_analysis.get("volatility_level", "medium")

        # 获取适用策略
        applicable_strategies = self._get_applicable_strategies(
            market_regime, volatility_level
        )

        for strategy in applicable_strategies:
            # 评估策略与当前组合的兼容性
            compatibility_score = self._assess_strategy_compatibility(
                strategy, current_portfolio, risk_constraints
            )

            if compatibility_score > 0.5:  # 只推荐兼容性高的策略
                recommendation = InvestmentRecommendation(
                    recommendation_id=f"strategy_{strategy}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    recommendation_type="strategy",
                    title=f"建议采用{self._get_strategy_name(strategy)}策略",
                    description=self._get_strategy_description(
                        strategy, market_analysis
                    ),
                    confidence_score=compatibility_score,
                    expected_return=self._estimate_strategy_return(
                        strategy, market_analysis
                    ),
                    risk_level=self._assess_strategy_risk(strategy),
                    time_horizon=self._get_strategy_horizon(strategy),
                    action_items=self._get_strategy_actions(strategy),
                    supporting_data={
                        "market_regime": market_regime,
                        "compatibility_score": compatibility_score,
                        "historical_performance": self._get_strategy_history(strategy),
                    },
                )

                recommendations.append(recommendation)

        return sorted(recommendations, key=lambda x: x.confidence_score, reverse=True)

    def generate_risk_adjustment_recommendations(
        self,
        portfolio_metrics: Dict[str, Any],
        risk_metrics: Dict[str, Any],
        risk_limits: Dict[str, Any],
    ) -> List[InvestmentRecommendation]:
        """生成风险调整建议

        Args:
            portfolio_metrics: 组合指标
            risk_metrics: 风险指标
            risk_limits: 风险限制

        Returns:
            风险调整建议列表
        """
        recommendations = []

        # 检查风险超限
        risk_breaches = self._check_risk_breaches(risk_metrics, risk_limits)

        for breach in risk_breaches:
            # 生成调整建议
            adjustment = self._generate_risk_adjustment(breach, portfolio_metrics)

            recommendation = InvestmentRecommendation(
                recommendation_id=f"risk_adj_{breach['type']}_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                recommendation_type="risk_adjustment",
                title=f"风险调整建议：{breach['description']}",
                description=adjustment["description"],
                confidence_score=adjustment["urgency"],
                risk_level="high" if adjustment["urgency"] > 0.8 else "medium",
                action_items=adjustment["actions"],
                supporting_data={
                    "breach_details": breach,
                    "adjustment_impact": adjustment["expected_impact"],
                },
            )

            recommendations.append(recommendation)

        # 添加预防性建议
        preventive_recommendations = self._generate_preventive_recommendations(
            portfolio_metrics, risk_metrics
        )
        recommendations.extend(preventive_recommendations)

        return recommendations

    def personalize_recommendations(
        self,
        base_recommendations: List[InvestmentRecommendation],
        user_preferences: Dict[str, Any],
        historical_behavior: Dict[str, Any],
    ) -> List[InvestmentRecommendation]:
        """个性化推荐

        Args:
            base_recommendations: 基础推荐列表
            user_preferences: 用户偏好
            historical_behavior: 历史行为

        Returns:
            个性化后的推荐列表
        """
        personalized = []

        for recommendation in base_recommendations:
            # 计算个性化分数
            personalization_score = self._calculate_personalization_score(
                recommendation, user_preferences, historical_behavior
            )

            # 调整置信度
            recommendation.confidence_score *= personalization_score

            # 添加个性化说明
            if personalization_score > 0.8:
                recommendation.description += "\n（特别推荐：基于您的历史偏好）"
            elif personalization_score < 0.5:
                recommendation.description += "\n（注意：可能不完全符合您的偏好）"

            personalized.append(recommendation)

        # 重新排序
        personalized.sort(key=lambda x: x.confidence_score, reverse=True)

        return personalized

    def _filter_suitable_templates(
        self, risk_tolerance: str, investment_horizon: str, investment_goals: List[str]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """筛选合适的组合模板

        Args:
            risk_tolerance: 风险承受能力
            investment_horizon: 投资期限
            investment_goals: 投资目标

        Returns:
            合适的模板列表
        """
        suitable = []

        for key, template in self.PORTFOLIO_TEMPLATES.items():
            suitability = 0

            # 匹配风险等级
            if risk_tolerance == "conservative" and "conservative" in key:
                suitability += 3
            elif risk_tolerance == "moderate" and "moderate" in key:
                suitability += 3
            elif risk_tolerance == "aggressive" and "aggressive" in key:
                suitability += 3

            # 匹配投资目标
            if "income" in investment_goals and "income" in key:
                suitability += 2
            if "growth" in investment_goals and "growth" in key:
                suitability += 2

            if suitability > 0:
                suitable.append((key, template))

        # 按适合度排序
        suitable.sort(key=lambda x: x[1].get("expected_return", 0), reverse=True)

        return suitable

    def _adjust_allocation_for_market(
        self, base_allocation: Dict[str, float], market_conditions: Dict[str, Any]
    ) -> Dict[str, float]:
        """根据市场条件调整配置

        Args:
            base_allocation: 基础配置
            market_conditions: 市场条件

        Returns:
            调整后的配置
        """
        adjusted = base_allocation.copy()

        # 根据市场状态调整
        market_trend = market_conditions.get("trend", "neutral")

        if market_trend == "bullish":
            # 增加股票配置
            if "stocks" in adjusted:
                adjusted["stocks"] *= 1.1
            if "bonds" in adjusted:
                adjusted["bonds"] *= 0.9
        elif market_trend == "bearish":
            # 减少股票配置
            if "stocks" in adjusted:
                adjusted["stocks"] *= 0.9
            if "bonds" in adjusted:
                adjusted["bonds"] *= 1.1
            if "cash" in adjusted:
                adjusted["cash"] *= 1.2

        # 归一化
        total = sum(adjusted.values())
        for key in adjusted:
            adjusted[key] /= total

        return adjusted

    def _calculate_suitability_score(
        self,
        template: Dict[str, Any],
        user_profile: Dict[str, Any],
        market_conditions: Dict[str, Any],
    ) -> float:
        """计算适合度分数

        Args:
            template: 组合模板
            user_profile: 用户画像
            market_conditions: 市场条件

        Returns:
            适合度分数（0-1）
        """
        score = 0.5  # 基础分数

        # 风险匹配度
        user_risk = user_profile.get("risk_tolerance", "moderate")
        template_risk = template.get("risk_level", "moderate")
        if user_risk == template_risk:
            score += 0.2

        # 收益目标匹配度
        target_return = user_profile.get("target_return", 0.1)
        expected_return = template.get("expected_return", 0.1)
        return_diff = abs(target_return - expected_return)
        if return_diff < 0.02:
            score += 0.2
        elif return_diff < 0.05:
            score += 0.1

        # 市场条件适应度
        market_volatility = market_conditions.get("volatility", "medium")
        if market_volatility == "high" and template.get("volatility", 0.15) < 0.15:
            score += 0.1
        elif market_volatility == "low" and template.get("volatility", 0.15) > 0.10:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _analyze_portfolio_pros_cons(
        self,
        template: Dict[str, Any],
        user_profile: Dict[str, Any],
        market_conditions: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        """分析组合优缺点

        Args:
            template: 组合模板
            user_profile: 用户画像
            market_conditions: 市场条件

        Returns:
            (优点列表, 缺点列表)
        """
        pros = []
        cons = []

        # 分析收益特征
        if template.get("expected_return", 0) > 0.12:
            pros.append("预期收益较高")
        elif template.get("expected_return", 0) < 0.08:
            cons.append("预期收益相对较低")

        # 分析风险特征
        if template.get("volatility", 0) < 0.10:
            pros.append("波动性低，风险可控")
        elif template.get("volatility", 0) > 0.20:
            cons.append("波动性较大，需承受一定风险")

        # 分析流动性
        if template.get("allocation", {}).get("cash", 0) > 0.05:
            pros.append("保持一定现金比例，流动性好")
        else:
            cons.append("现金比例较低，流动性受限")

        # 分析分散化
        if len(template.get("allocation", {})) > 3:
            pros.append("资产配置分散，降低集中风险")
        else:
            cons.append("资产集中度较高")

        return pros, cons

    def _generate_implementation_steps(self, template_key: str) -> List[str]:
        """生成实施步骤

        Args:
            template_key: 模板键

        Returns:
            实施步骤列表
        """
        steps = [
            "评估当前持仓情况",
            "计算调仓成本",
            "制定分批建仓计划",
            f"按照{template_key}配置逐步调整仓位",
            "设置止损和再平衡规则",
            "定期监控和评估效果",
        ]

        return steps

    def _get_applicable_strategies(
        self, market_regime: str, volatility_level: str
    ) -> List[str]:
        """获取适用策略

        Args:
            market_regime: 市场状态
            volatility_level: 波动率水平

        Returns:
            策略列表
        """
        strategies = []

        # 根据市场状态选择策略
        if market_regime in self.STRATEGY_RULES:
            strategies.extend(self.STRATEGY_RULES[market_regime])

        # 根据波动率调整
        if volatility_level == "high":
            strategies.extend(self.STRATEGY_RULES.get("high_volatility", []))
        elif volatility_level == "low":
            strategies.extend(self.STRATEGY_RULES.get("low_volatility", []))

        # 去重
        return list(set(strategies))

    def _assess_strategy_compatibility(
        self,
        strategy: str,
        current_portfolio: Dict[str, Any],
        risk_constraints: Dict[str, Any],
    ) -> float:
        """评估策略兼容性

        Args:
            strategy: 策略名称
            current_portfolio: 当前组合
            risk_constraints: 风险约束

        Returns:
            兼容性分数（0-1）
        """
        score = 0.7  # 基础分数

        # 检查风险约束
        if (
            strategy in ["momentum", "growth"]
            and risk_constraints.get("max_drawdown", 1.0) < 0.15
        ):
            score -= 0.3

        # 检查与现有持仓的兼容性
        current_strategy = current_portfolio.get("dominant_strategy", "")
        if current_strategy and current_strategy != strategy:
            # 检查策略是否冲突
            if current_strategy in ["value", "defensive"] and strategy in [
                "momentum",
                "growth",
            ]:
                score -= 0.2

        return max(0.0, min(1.0, score))

    def _get_strategy_name(self, strategy: str) -> str:
        """获取策略中文名称

        Args:
            strategy: 策略标识

        Returns:
            中文名称
        """
        strategy_names = {
            "momentum": "动量追踪",
            "value": "价值投资",
            "growth": "成长股投资",
            "mean_reversion": "均值回归",
            "trend_following": "趋势跟踪",
            "defensive": "防御型",
            "arbitrage": "套利",
            "neutral": "市场中性",
            "volatility_trading": "波动率交易",
            "carry_trade": "利差交易",
        }

        return strategy_names.get(strategy, strategy)

    def _get_strategy_description(
        self, strategy: str, market_analysis: Dict[str, Any]
    ) -> str:
        """获取策略描述

        Args:
            strategy: 策略标识
            market_analysis: 市场分析

        Returns:
            策略描述
        """
        descriptions = {
            "momentum": f"在当前{market_analysis.get('trend', '市场')}环境下，追踪强势股票的上涨动能",
            "value": "寻找被低估的优质资产，等待价值回归",
            "mean_reversion": "利用价格偏离均值后的回归特性进行交易",
            "trend_following": "顺应市场主要趋势，持有趋势方向的仓位",
        }

        return descriptions.get(
            strategy, f"实施{self._get_strategy_name(strategy)}策略"
        )

    def _estimate_strategy_return(
        self, strategy: str, market_analysis: Dict[str, Any]
    ) -> float:
        """估算策略收益

        Args:
            strategy: 策略标识
            market_analysis: 市场分析

        Returns:
            预期收益率
        """
        base_returns = {
            "momentum": 0.15,
            "value": 0.12,
            "growth": 0.18,
            "mean_reversion": 0.10,
            "defensive": 0.06,
            "arbitrage": 0.08,
        }

        base_return = base_returns.get(strategy, 0.10)

        # 根据市场条件调整
        market_favorability = market_analysis.get("strategy_favorability", {}).get(
            strategy, 1.0
        )

        return base_return * market_favorability

    def _assess_strategy_risk(self, strategy: str) -> str:
        """评估策略风险等级

        Args:
            strategy: 策略标识

        Returns:
            风险等级
        """
        risk_levels = {
            "momentum": "high",
            "growth": "high",
            "value": "medium",
            "mean_reversion": "medium",
            "defensive": "low",
            "arbitrage": "low",
            "neutral": "low",
        }

        return risk_levels.get(strategy, "medium")

    def _get_strategy_horizon(self, strategy: str) -> str:
        """获取策略时间跨度

        Args:
            strategy: 策略标识

        Returns:
            时间跨度
        """
        horizons = {
            "momentum": "short_term",
            "value": "long_term",
            "growth": "medium_term",
            "mean_reversion": "short_term",
            "trend_following": "medium_term",
        }

        return horizons.get(strategy, "medium_term")

    def _get_strategy_actions(self, strategy: str) -> List[str]:
        """获取策略执行动作

        Args:
            strategy: 策略标识

        Returns:
            动作列表
        """
        actions = {
            "momentum": [
                "筛选近期涨幅居前的股票",
                "设置严格的止损位",
                "及时跟踪动量变化",
            ],
            "value": ["分析基本面指标", "寻找低估值标的", "耐心等待价值实现"],
            "mean_reversion": [
                "识别超买超卖信号",
                "设置回归目标位",
                "控制单次交易规模",
            ],
        }

        return actions.get(strategy, ["执行策略规则", "监控市场变化", "及时调整仓位"])

    def _get_strategy_history(self, strategy: str) -> Dict[str, Any]:
        """获取策略历史表现

        Args:
            strategy: 策略标识

        Returns:
            历史表现数据
        """
        # 模拟历史数据
        return {
            "avg_annual_return": 0.12,
            "max_drawdown": 0.15,
            "win_rate": 0.55,
            "sharpe_ratio": 1.2,
        }

    def _check_risk_breaches(
        self, risk_metrics: Dict[str, Any], risk_limits: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """检查风险超限

        Args:
            risk_metrics: 风险指标
            risk_limits: 风险限制

        Returns:
            超限列表
        """
        breaches = []

        for metric, limit in risk_limits.items():
            current_value = risk_metrics.get(metric)
            if current_value and current_value > limit:
                breaches.append(
                    {
                        "type": metric,
                        "current": current_value,
                        "limit": limit,
                        "severity": (current_value - limit) / limit,
                        "description": f"{metric}超过限制",
                    }
                )

        return breaches

    def _generate_risk_adjustment(
        self, breach: Dict[str, Any], portfolio_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成风险调整方案

        Args:
            breach: 超限信息
            portfolio_metrics: 组合指标

        Returns:
            调整方案
        """
        adjustment = {
            "description": "",
            "actions": [],
            "urgency": 0.5,
            "expected_impact": {},
        }

        if breach["type"] == "max_drawdown":
            adjustment["description"] = "当前回撤超过限制，需要降低风险敞口"
            adjustment["actions"] = [
                "减少高风险仓位",
                "增加防御性资产配置",
                "设置更严格的止损",
            ]
            adjustment["urgency"] = min(1.0, 0.5 + breach["severity"])
            adjustment["expected_impact"] = {
                "risk_reduction": 0.3,
                "return_impact": -0.05,
            }
        elif breach["type"] == "concentration_risk":
            adjustment["description"] = "持仓过于集中，需要增加分散化"
            adjustment["actions"] = ["减少集中持仓", "增加持仓数量", "平衡行业配置"]
            adjustment["urgency"] = 0.6
            adjustment["expected_impact"] = {
                "risk_reduction": 0.2,
                "return_impact": -0.02,
            }

        return adjustment

    def _generate_preventive_recommendations(
        self, portfolio_metrics: Dict[str, Any], risk_metrics: Dict[str, Any]
    ) -> List[InvestmentRecommendation]:
        """生成预防性建议

        Args:
            portfolio_metrics: 组合指标
            risk_metrics: 风险指标

        Returns:
            预防性建议列表
        """
        recommendations = []

        # 检查潜在风险
        if (
            risk_metrics.get("volatility", 0)
            > risk_metrics.get("historical_avg_volatility", 0) * 1.5
        ):
            recommendation = InvestmentRecommendation(
                recommendation_id=f"preventive_vol_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                recommendation_type="risk_adjustment",
                title="波动率上升预警",
                description="市场波动率显著上升，建议提前做好风险防范",
                confidence_score=0.7,
                risk_level="medium",
                action_items=["检查止损设置", "考虑增加对冲仓位", "准备流动性储备"],
            )
            recommendations.append(recommendation)

        return recommendations

    def _calculate_personalization_score(
        self,
        recommendation: InvestmentRecommendation,
        user_preferences: Dict[str, Any],
        historical_behavior: Dict[str, Any],
    ) -> float:
        """计算个性化分数

        Args:
            recommendation: 推荐对象
            user_preferences: 用户偏好
            historical_behavior: 历史行为

        Returns:
            个性化分数（0-1）
        """
        score = 0.7  # 基础分数

        # 检查是否符合用户偏好的资产类型
        preferred_assets = user_preferences.get("preferred_assets", [])
        if any(asset in recommendation.description for asset in preferred_assets):
            score += 0.15

        # 检查是否符合历史行为模式
        historical_strategies = historical_behavior.get(
            "frequently_used_strategies", []
        )
        if recommendation.recommendation_type == "strategy":
            if any(
                strategy in recommendation.title for strategy in historical_strategies
            ):
                score += 0.15

        return min(1.0, score)


# 模块级别函数
def create_recommendation_engine() -> RecommendationEngine:
    """创建推荐引擎实例

    Returns:
        推荐引擎实例
    """
    return RecommendationEngine()


def generate_default_recommendations(
    user_profile: Dict[str, Any],
) -> List[PortfolioRecommendation]:
    """生成默认推荐

    Args:
        user_profile: 用户画像

    Returns:
        推荐列表
    """
    engine = RecommendationEngine()
    return engine.generate_portfolio_recommendations(
        user_profile,
        {"trend": "neutral", "volatility": "medium"},
        num_recommendations=3,
    )
