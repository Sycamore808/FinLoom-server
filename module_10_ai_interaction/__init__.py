"""
AI交互模块初始化文件
"""

from .fin_r1_integration import FINR1Integration, process_investment_request
from .requirement_parser import (
    InvestmentConstraint,
    InvestmentGoal,
    InvestmentHorizon,
    ParsedRequirement,
    RequirementParser,
    RiskTolerance,
    parse_user_requirement
)

__all__ = [
    "FINR1Integration",
    "process_investment_request",
    "InvestmentConstraint",
    "InvestmentGoal",
    "InvestmentHorizon",
    "ParsedRequirement",
    "RequirementParser",
    "RiskTolerance",
    "parse_user_requirement"
]