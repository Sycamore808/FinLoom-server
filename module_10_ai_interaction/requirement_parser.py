"""
需求解析器模块
负责解析用户的投资需求并提取关键信息
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("requirement_parser")


class RiskTolerance(Enum):
    """风险承受能力枚举"""

    CONSERVATIVE = "conservative"  # 保守型
    MODERATE = "moderate"  # 稳健型
    AGGRESSIVE = "aggressive"  # 激进型
    VERY_AGGRESSIVE = "very_aggressive"  # 非常激进型


class InvestmentHorizon(Enum):
    """投资期限枚举"""

    SHORT_TERM = "short_term"  # 短期（<1年）
    MEDIUM_TERM = "medium_term"  # 中期（1-3年）
    LONG_TERM = "long_term"  # 长期（3-5年）
    VERY_LONG_TERM = "very_long_term"  # 超长期（>5年）


@dataclass
class InvestmentGoal:
    """投资目标数据结构"""

    goal_type: str  # 'wealth_growth', 'income', 'preservation', 'speculation'
    target_return: Optional[float] = None
    priority: int = 1
    description: str = ""


@dataclass
class InvestmentConstraint:
    """投资约束数据结构"""

    constraint_type: str  # 'sector', 'asset_class', 'esg', 'liquidity'
    constraint_value: Any
    is_hard_constraint: bool = True
    description: str = ""


@dataclass
class ParsedRequirement:
    """解析后的需求数据结构"""

    timestamp: datetime
    raw_input: str
    investment_amount: Optional[float] = None
    investment_horizon: Optional[InvestmentHorizon] = None
    risk_tolerance: Optional[RiskTolerance] = None
    investment_goals: List[InvestmentGoal] = field(default_factory=list)
    constraints: List[InvestmentConstraint] = field(default_factory=list)
    preferred_assets: List[str] = field(default_factory=list)
    excluded_assets: List[str] = field(default_factory=list)
    target_sectors: List[str] = field(default_factory=list)
    excluded_sectors: List[str] = field(default_factory=list)
    max_drawdown: Optional[float] = None
    min_liquidity: Optional[float] = None
    tax_considerations: bool = False
    esg_preferences: Optional[str] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    clarification_needed: List[str] = field(default_factory=list)


class RequirementParser:
    """需求解析器类"""

    # 关键词映射
    RISK_KEYWORDS = {
        RiskTolerance.CONSERVATIVE: [
            "保守",
            "稳定",
            "低风险",
            "安全",
            "保本",
            "稳健偏保守",
        ],
        RiskTolerance.MODERATE: ["稳健", "平衡", "中等风险", "适中", "均衡"],
        RiskTolerance.AGGRESSIVE: ["激进", "积极", "高风险", "进取", "成长型"],
        RiskTolerance.VERY_AGGRESSIVE: ["非常激进", "极高风险", "投机", "高收益"],
    }

    HORIZON_KEYWORDS = {
        InvestmentHorizon.SHORT_TERM: ["短期", "几个月", "半年", "一年内", "12个月内"],
        InvestmentHorizon.MEDIUM_TERM: ["中期", "1-3年", "两年", "三年", "几年"],
        InvestmentHorizon.LONG_TERM: ["长期", "3-5年", "五年", "长线"],
        InvestmentHorizon.VERY_LONG_TERM: ["超长期", "5年以上", "十年", "养老", "退休"],
    }

    GOAL_KEYWORDS = {
        "wealth_growth": ["增值", "增长", "成长", "资本利得"],
        "income": ["收入", "分红", "股息", "现金流"],
        "preservation": ["保值", "保本", "避险", "对冲通胀"],
        "speculation": ["投机", "短线", "波段", "套利"],
    }

    SECTOR_MAPPING = {
        "科技": ["technology", "tech", "TMT"],
        "医疗": ["healthcare", "medical", "biotech"],
        "金融": ["financial", "banking", "insurance"],
        "消费": ["consumer", "retail", "FMCG"],
        "能源": ["energy", "oil", "gas", "renewable"],
        "工业": ["industrial", "manufacturing"],
        "房地产": ["real_estate", "REIT", "property"],
        "材料": ["materials", "chemicals", "metals"],
        "公用事业": ["utilities", "infrastructure"],
        "通信": ["communication", "telecom", "media"],
    }

    def __init__(self):
        """初始化需求解析器"""
        self.nlp_model = None  # 可以集成spaCy或其他NLP模型

    def parse_investment_goals(self, text: str) -> Tuple[List[InvestmentGoal], float]:
        """解析投资目标

        Args:
            text: 输入文本

        Returns:
            (投资目标列表, 置信度分数)
        """
        goals = []
        confidence = 0.0

        text_lower = text.lower()

        for goal_type, keywords in self.GOAL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    goal = InvestmentGoal(
                        goal_type=goal_type,
                        priority=1 if goal_type == "wealth_growth" else 2,
                        description=f"Detected from keyword: {keyword}",
                    )

                    # 尝试提取目标收益率
                    target_return = self._extract_percentage(text, keyword)
                    if target_return:
                        goal.target_return = target_return

                    goals.append(goal)
                    confidence = max(confidence, 0.8)
                    break

        # 如果没有明确的目标，设置默认目标
        if not goals:
            goals.append(
                InvestmentGoal(
                    goal_type="wealth_growth", priority=1, description="Default goal"
                )
            )
            confidence = 0.3

        return goals, confidence

    def extract_risk_preferences(
        self, text: str
    ) -> Tuple[Optional[RiskTolerance], float]:
        """提取风险偏好

        Args:
            text: 输入文本

        Returns:
            (风险承受能力, 置信度分数)
        """
        text_lower = text.lower()
        best_match = None
        best_confidence = 0.0

        for risk_level, keywords in self.RISK_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # 计算关键词在文本中的相关性
                    confidence = self._calculate_keyword_relevance(text_lower, keyword)
                    if confidence > best_confidence:
                        best_match = risk_level
                        best_confidence = confidence

        # 如果没有明确的风险偏好，尝试从其他线索推断
        if not best_match:
            if "不想亏" in text or "保本" in text:
                best_match = RiskTolerance.CONSERVATIVE
                best_confidence = 0.7
            elif "收益" in text and "风险" in text:
                best_match = RiskTolerance.MODERATE
                best_confidence = 0.5

        return best_match, best_confidence

    def identify_constraints(self, text: str) -> List[InvestmentConstraint]:
        """识别投资约束

        Args:
            text: 输入文本

        Returns:
            约束列表
        """
        constraints = []
        text_lower = text.lower()

        # 检查行业限制
        if "不投" in text or "避免" in text or "不要" in text:
            # 提取否定的内容
            patterns = [
                r"不投[资买入]*(.+?)(?:[，。,.]|$)",
                r"避免(.+?)(?:[，。,.]|$)",
                r"不要[投买](.+?)(?:[，。,.]|$)",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    constraints.append(
                        InvestmentConstraint(
                            constraint_type="exclusion",
                            constraint_value=match.strip(),
                            is_hard_constraint=True,
                            description=f"Exclude: {match.strip()}",
                        )
                    )

        # 检查ESG偏好
        esg_keywords = ["环保", "绿色", "可持续", "ESG", "社会责任"]
        for keyword in esg_keywords:
            if keyword in text_lower:
                constraints.append(
                    InvestmentConstraint(
                        constraint_type="esg",
                        constraint_value="positive",
                        is_hard_constraint=False,
                        description="ESG preference detected",
                    )
                )
                break

        # 检查流动性要求
        if "流动性" in text or "随时取出" in text or "灵活" in text:
            constraints.append(
                InvestmentConstraint(
                    constraint_type="liquidity",
                    constraint_value="high",
                    is_hard_constraint=False,
                    description="High liquidity requirement",
                )
            )

        # 检查杠杆限制
        if "不加杠杆" in text or "无杠杆" in text:
            constraints.append(
                InvestmentConstraint(
                    constraint_type="leverage",
                    constraint_value=1.0,
                    is_hard_constraint=True,
                    description="No leverage",
                )
            )

        return constraints

    def map_to_system_parameters(self, parsed_req: ParsedRequirement) -> Dict[str, Any]:
        """映射到系统参数

        Args:
            parsed_req: 解析后的需求

        Returns:
            系统参数字典
        """
        # 默认参数
        system_params = {
            "risk_profile": {
                "risk_tolerance": "moderate",
                "max_drawdown": 0.15,
                "position_limit": 0.1,
                "leverage": 1.0,
            },
            "strategy_config": {
                "strategy_mix": {
                    "trend_following": 0.3,
                    "mean_reversion": 0.3,
                    "momentum": 0.2,
                    "value": 0.2,
                },
                "rebalance_frequency": "weekly",
                "min_holding_period": 5,
            },
            "optimization_targets": {
                "primary_objective": "sharpe_ratio",
                "secondary_objectives": ["max_drawdown", "volatility"],
                "optimization_horizon": 252,
            },
            "execution_settings": {
                "order_type": "limit",
                "execution_algo": "vwap",
                "urgency": "normal",
                "max_participation_rate": 0.1,
            },
        }

        # 根据风险偏好调整参数
        if parsed_req.risk_tolerance:
            risk_params = self._get_risk_parameters(parsed_req.risk_tolerance)
            system_params["risk_profile"].update(risk_params)

            # 调整策略组合
            strategy_mix = self._get_strategy_mix(parsed_req.risk_tolerance)
            system_params["strategy_config"]["strategy_mix"] = strategy_mix

        # 根据投资期限调整参数
        if parsed_req.investment_horizon:
            horizon_params = self._get_horizon_parameters(parsed_req.investment_horizon)
            system_params["strategy_config"].update(horizon_params)

        # 应用约束条件
        for constraint in parsed_req.constraints:
            self._apply_constraint(system_params, constraint)

        # 设置最大回撤
        if parsed_req.max_drawdown:
            system_params["risk_profile"]["max_drawdown"] = parsed_req.max_drawdown

        # 设置资产偏好
        if parsed_req.preferred_assets:
            system_params["asset_preferences"] = {
                "preferred": parsed_req.preferred_assets,
                "excluded": parsed_req.excluded_assets,
            }

        # 设置行业偏好
        if parsed_req.target_sectors or parsed_req.excluded_sectors:
            system_params["sector_preferences"] = {
                "target": parsed_req.target_sectors,
                "excluded": parsed_req.excluded_sectors,
            }

        return system_params

    def validate_parameter_consistency(
        self, parameters: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """验证参数一致性

        Args:
            parameters: 系统参数

        Returns:
            (是否一致, 问题列表)
        """
        issues = []

        # 检查风险参数一致性
        risk_profile = parameters.get("risk_profile", {})
        if risk_profile.get("risk_tolerance") == "conservative":
            if risk_profile.get("max_drawdown", 0) > 0.2:
                issues.append("Conservative risk tolerance but high max drawdown")
            if risk_profile.get("leverage", 1) > 1.0:
                issues.append("Conservative risk tolerance but using leverage")

        # 检查策略组合权重
        strategy_mix = parameters.get("strategy_config", {}).get("strategy_mix", {})
        total_weight = sum(strategy_mix.values())
        if abs(total_weight - 1.0) > 0.01:
            issues.append(f"Strategy weights don't sum to 1.0: {total_weight}")

        # 检查执行参数
        execution = parameters.get("execution_settings", {})
        if (
            execution.get("urgency") == "high"
            and execution.get("order_type") == "limit"
        ):
            issues.append("High urgency but using limit orders may cause delays")

        is_consistent = len(issues) == 0
        return is_consistent, issues

    def _extract_amount(self, text: str) -> Optional[float]:
        """提取投资金额

        Args:
            text: 输入文本

        Returns:
            金额（元）
        """
        # 匹配各种金额格式
        patterns = [
            r"(\d+(?:\.\d+)?)\s*万",
            r"(\d+(?:\.\d+)?)\s*千",
            r"(\d+(?:\.\d+)?)\s*百",
            r"(\d+(?:\.\d+)?)\s*元",
            r"(\d+(?:\.\d+)?)\s*块",
            r"￥\s*(\d+(?:\.\d+)?)",
            r"\$\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                amount = float(match.group(1))
                if "万" in pattern:
                    amount *= 10000
                elif "千" in pattern:
                    amount *= 1000
                elif "百" in pattern:
                    amount *= 100
                elif "$" in pattern:
                    amount *= 7  # 简单汇率转换
                return amount

        return None

    def _extract_percentage(self, text: str, context: str = "") -> Optional[float]:
        """提取百分比

        Args:
            text: 输入文本
            context: 上下文关键词

        Returns:
            百分比值（如15%返回0.15）
        """
        # 在关键词附近查找百分比
        search_area = text
        if context:
            # 获取关键词前后的文本
            idx = text.lower().find(context.lower())
            if idx != -1:
                start = max(0, idx - 20)
                end = min(len(text), idx + len(context) + 20)
                search_area = text[start:end]

        patterns = [
            r"(\d+(?:\.\d+)?)\s*%",
            r"(\d+(?:\.\d+)?)\s*百分",
            r"百分之\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, search_area)
            if match:
                return float(match.group(1)) / 100

        return None

    def _extract_time_horizon(self, text: str) -> Optional[InvestmentHorizon]:
        """提取投资期限

        Args:
            text: 输入文本

        Returns:
            投资期限枚举
        """
        text_lower = text.lower()

        for horizon, keywords in self.HORIZON_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return horizon

        # 尝试提取具体时间
        time_patterns = [
            (r"(\d+)\s*年", "year"),
            (r"(\d+)\s*个月", "month"),
            (r"(\d+)\s*月", "month"),
            (r"(\d+)\s*周", "week"),
            (r"(\d+)\s*天", "day"),
        ]

        for pattern, unit in time_patterns:
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                if unit == "year":
                    if value < 1:
                        return InvestmentHorizon.SHORT_TERM
                    elif value <= 3:
                        return InvestmentHorizon.MEDIUM_TERM
                    elif value <= 5:
                        return InvestmentHorizon.LONG_TERM
                    else:
                        return InvestmentHorizon.VERY_LONG_TERM
                elif unit == "month":
                    if value <= 12:
                        return InvestmentHorizon.SHORT_TERM
                    elif value <= 36:
                        return InvestmentHorizon.MEDIUM_TERM
                    else:
                        return InvestmentHorizon.LONG_TERM

        return None

    def _calculate_keyword_relevance(self, text: str, keyword: str) -> float:
        """计算关键词相关性

        Args:
            text: 文本
            keyword: 关键词

        Returns:
            相关性分数（0-1）
        """
        # 简单的相关性计算
        # 实际应该使用更复杂的NLP方法
        if keyword not in text:
            return 0.0

        # 考虑关键词位置（越靠前越重要）
        position = text.find(keyword) / len(text)
        position_score = 1.0 - position * 0.3

        # 考虑关键词频率
        frequency = text.count(keyword)
        frequency_score = min(1.0, frequency * 0.3)

        return min(1.0, position_score * 0.7 + frequency_score * 0.3)

    def _get_risk_parameters(self, risk_tolerance: RiskTolerance) -> Dict[str, Any]:
        """获取风险参数

        Args:
            risk_tolerance: 风险承受能力

        Returns:
            风险参数字典
        """
        risk_params = {
            RiskTolerance.CONSERVATIVE: {
                "risk_tolerance": "conservative",
                "max_drawdown": 0.08,
                "position_limit": 0.05,
                "leverage": 1.0,
            },
            RiskTolerance.MODERATE: {
                "risk_tolerance": "moderate",
                "max_drawdown": 0.15,
                "position_limit": 0.1,
                "leverage": 1.0,
            },
            RiskTolerance.AGGRESSIVE: {
                "risk_tolerance": "aggressive",
                "max_drawdown": 0.25,
                "position_limit": 0.15,
                "leverage": 1.5,
            },
            RiskTolerance.VERY_AGGRESSIVE: {
                "risk_tolerance": "very_aggressive",
                "max_drawdown": 0.35,
                "position_limit": 0.2,
                "leverage": 2.0,
            },
        }

        return risk_params.get(risk_tolerance, risk_params[RiskTolerance.MODERATE])

    def _get_strategy_mix(self, risk_tolerance: RiskTolerance) -> Dict[str, float]:
        """获取策略组合

        Args:
            risk_tolerance: 风险承受能力

        Returns:
            策略权重字典
        """
        strategy_mixes = {
            RiskTolerance.CONSERVATIVE: {
                "value": 0.5,
                "mean_reversion": 0.3,
                "trend_following": 0.1,
                "momentum": 0.1,
            },
            RiskTolerance.MODERATE: {
                "value": 0.3,
                "mean_reversion": 0.3,
                "trend_following": 0.2,
                "momentum": 0.2,
            },
            RiskTolerance.AGGRESSIVE: {
                "momentum": 0.35,
                "trend_following": 0.35,
                "mean_reversion": 0.2,
                "value": 0.1,
            },
            RiskTolerance.VERY_AGGRESSIVE: {
                "momentum": 0.5,
                "trend_following": 0.3,
                "mean_reversion": 0.15,
                "value": 0.05,
            },
        }

        return strategy_mixes.get(
            risk_tolerance, strategy_mixes[RiskTolerance.MODERATE]
        )

    def _get_horizon_parameters(self, horizon: InvestmentHorizon) -> Dict[str, Any]:
        """获取期限相关参数

        Args:
            horizon: 投资期限

        Returns:
            期限参数字典
        """
        horizon_params = {
            InvestmentHorizon.SHORT_TERM: {
                "rebalance_frequency": "daily",
                "min_holding_period": 1,
                "optimization_horizon": 60,
            },
            InvestmentHorizon.MEDIUM_TERM: {
                "rebalance_frequency": "weekly",
                "min_holding_period": 5,
                "optimization_horizon": 252,
            },
            InvestmentHorizon.LONG_TERM: {
                "rebalance_frequency": "monthly",
                "min_holding_period": 20,
                "optimization_horizon": 504,
            },
            InvestmentHorizon.VERY_LONG_TERM: {
                "rebalance_frequency": "quarterly",
                "min_holding_period": 60,
                "optimization_horizon": 1260,
            },
        }

        return horizon_params.get(
            horizon, horizon_params[InvestmentHorizon.MEDIUM_TERM]
        )

    def _apply_constraint(
        self, system_params: Dict[str, Any], constraint: InvestmentConstraint
    ) -> None:
        """应用约束到系统参数

        Args:
            system_params: 系统参数字典
            constraint: 约束对象
        """
        if constraint.constraint_type == "leverage":
            system_params["risk_profile"]["leverage"] = constraint.constraint_value
        elif constraint.constraint_type == "liquidity":
            system_params["execution_settings"]["liquidity_requirement"] = (
                constraint.constraint_value
            )
        elif constraint.constraint_type == "esg":
            system_params["esg_filter"] = constraint.constraint_value
        elif constraint.constraint_type == "exclusion":
            if "exclusions" not in system_params:
                system_params["exclusions"] = []
            system_params["exclusions"].append(constraint.constraint_value)


# 模块级别函数
def parse_user_requirement(user_input: str) -> ParsedRequirement:
    """解析用户需求的便捷函数

    Args:
        user_input: 用户输入文本

    Returns:
        解析后的需求
    """
    parser = RequirementParser()

    parsed = ParsedRequirement(timestamp=datetime.now(), raw_input=user_input)

    # 提取各种信息
    parsed.investment_amount = parser._extract_amount(user_input)
    parsed.investment_horizon = parser._extract_time_horizon(user_input)

    risk_tolerance, risk_confidence = parser.extract_risk_preferences(user_input)
    parsed.risk_tolerance = risk_tolerance
    parsed.confidence_scores["risk_tolerance"] = risk_confidence

    goals, goals_confidence = parser.parse_investment_goals(user_input)
    parsed.investment_goals = goals
    parsed.confidence_scores["goals"] = goals_confidence

    parsed.constraints = parser.identify_constraints(user_input)

    # 检查是否需要澄清
    if not parsed.investment_amount:
        parsed.clarification_needed.append("investment_amount")
    if not parsed.risk_tolerance:
        parsed.clarification_needed.append("risk_tolerance")
    if not parsed.investment_horizon:
        parsed.clarification_needed.append("investment_horizon")

    return parsed
