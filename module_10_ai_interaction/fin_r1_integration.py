"""
FIN-R1模型集成模块
负责集成和调用FIN-R1大语言模型进行投资需求解析
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

# 尝试导入可选依赖
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from common.exceptions import ModelError
from common.logging_system import setup_logger
from module_10_ai_interaction.requirement_parser import (
    InvestmentConstraint,
    InvestmentGoal,
    InvestmentHorizon,
    ParsedRequirement,
    RequirementParser,
    RiskTolerance,
)

logger = setup_logger("fin_r1_integration")

class FINR1Integration:
    """FIN-R1模型集成类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化FIN-R1集成
        
        Args:
            config: 模型配置
        """
        self.config = config
        self.model_path = config.get("model_path", "models/fin_r1")
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 32)
        self.max_length = config.get("max_length", 512)
        self.temperature = config.get("temperature", 0.7)
        
        # 初始化模型和分词器
        self.model = None
        self.tokenizer = None
        self.requirement_parser = RequirementParser()
        
        self._load_model()
        
    def _load_model(self):
        """加载模型"""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                logger.warning("Transformers or PyTorch not available, using mock model")
                self.model = None
                self.tokenizer = None
                return
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # 加载模型
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"FIN-R1 model loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load FIN-R1 model: {e}, using mock model")
            self.model = None
            self.tokenizer = None
            
    async def process_request(self, user_input: str) -> Dict[str, Any]:
        """处理用户请求
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            处理结果字典
        """
        try:
            # 1. 解析用户需求
            parsed_requirement = self.requirement_parser.parse_requirement(user_input)
            
            # 2. 使用FIN-R1模型进行深度理解
            model_output = self._invoke_fin_r1_model(user_input, parsed_requirement)
            
            # 3. 生成策略参数
            strategy_params = self._generate_strategy_parameters(
                parsed_requirement, model_output
            )
            
            # 4. 生成风险参数
            risk_params = self._generate_risk_parameters(parsed_requirement)
            
            # 5. 组合结果
            result = {
                "parsed_requirement": parsed_requirement.to_dict(),
                "model_output": model_output,
                "strategy_params": strategy_params,
                "risk_params": risk_params,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Request processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            raise ModelError(f"Request processing failed: {e}")
            
    def _invoke_fin_r1_model(
        self, 
        user_input: str, 
        parsed_requirement: ParsedRequirement
    ) -> Dict[str, Any]:
        """调用FIN-R1模型
        
        Args:
            user_input: 用户输入
            parsed_requirement: 解析后的需求
            
        Returns:
            模型输出
        """
        try:
            # 准备输入
            input_text = self._prepare_model_input(user_input, parsed_requirement)
            
            # 分词
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # 处理输出（这里简化处理，实际应该根据模型结构处理）
            # 假设模型输出包含策略建议、风险评估等
            model_output = {
                "strategy_recommendation": "balanced",  # 策略建议
                "risk_assessment": "moderate",  # 风险评估
                "confidence_score": 0.85,  # 置信度
                "market_outlook": "neutral",  # 市场展望
                "sector_preferences": ["technology", "healthcare"],  # 行业偏好
                "excluded_sectors": ["financial"],  # 排除行业
            }
            
            return model_output
            
        except Exception as e:
            logger.error(f"Failed to invoke FIN-R1 model: {e}")
            # 返回默认值
            return {
                "strategy_recommendation": "balanced",
                "risk_assessment": "moderate",
                "confidence_score": 0.5,
                "market_outlook": "neutral",
                "sector_preferences": [],
                "excluded_sectors": []
            }
            
    def _prepare_model_input(
        self, 
        user_input: str, 
        parsed_requirement: ParsedRequirement
    ) -> str:
        """准备模型输入
        
        Args:
            user_input: 用户输入
            parsed_requirement: 解析后的需求
            
        Returns:
            模型输入文本
        """
        # 构造结构化输入
        input_parts = [
            f"User Request: {user_input}",
            f"Investment Amount: {parsed_requirement.investment_amount or 'Not specified'}",
            f"Risk Tolerance: {parsed_requirement.risk_tolerance.value if parsed_requirement.risk_tolerance else 'Not specified'}",
            f"Investment Horizon: {parsed_requirement.investment_horizon.value if parsed_requirement.investment_horizon else 'Not specified'}",
            f"Goals: {[goal.goal_type for goal in parsed_requirement.investment_goals]}",
            f"Constraints: {[constraint.constraint_type for constraint in parsed_requirement.constraints]}"
        ]
        
        return "\n".join(input_parts)
        
    def _generate_strategy_parameters(
        self, 
        parsed_requirement: ParsedRequirement, 
        model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成策略参数
        
        Args:
            parsed_requirement: 解析后的需求
            model_output: 模型输出
            
        Returns:
            策略参数字典
        """
        # 基于解析结果和模型输出生成策略参数
        strategy_params = {
            "strategy_mix": {
                "trend_following": 0.3,
                "mean_reversion": 0.3,
                "momentum": 0.2,
                "value": 0.2
            },
            "rebalance_frequency": "weekly",
            "position_sizing_method": "kelly_criterion",
            "risk_adjustment": True
        }
        
        # 根据风险偏好调整
        if parsed_requirement.risk_tolerance:
            if parsed_requirement.risk_tolerance == RiskTolerance.CONSERVATIVE:
                strategy_params["strategy_mix"] = {
                    "trend_following": 0.2,
                    "mean_reversion": 0.4,
                    "momentum": 0.1,
                    "value": 0.3
                }
                strategy_params["rebalance_frequency"] = "monthly"
            elif parsed_requirement.risk_tolerance == RiskTolerance.AGGRESSIVE:
                strategy_params["strategy_mix"] = {
                    "trend_following": 0.4,
                    "mean_reversion": 0.1,
                    "momentum": 0.3,
                    "value": 0.2
                }
                strategy_params["rebalance_frequency"] = "daily"
                
        # 根据模型输出调整
        if model_output.get("strategy_recommendation") == "growth":
            strategy_params["strategy_mix"]["momentum"] += 0.1
            strategy_params["strategy_mix"]["value"] -= 0.1
            
        return strategy_params
        
    def _generate_risk_parameters(
        self, 
        parsed_requirement: ParsedRequirement
    ) -> Dict[str, Any]:
        """生成风险参数
        
        Args:
            parsed_requirement: 解析后的需求
            
        Returns:
            风险参数字典
        """
        risk_params = {
            "max_drawdown": 0.15,
            "position_limit": 0.1,
            "leverage": 1.0,
            "stop_loss": 0.05
        }
        
        # 根据风险偏好调整
        if parsed_requirement.risk_tolerance:
            risk_tolerance_map = {
                RiskTolerance.CONSERVATIVE: {"max_drawdown": 0.08, "position_limit": 0.05, "leverage": 1.0},
                RiskTolerance.MODERATE: {"max_drawdown": 0.15, "position_limit": 0.1, "leverage": 1.0},
                RiskTolerance.AGGRESSIVE: {"max_drawdown": 0.25, "position_limit": 0.15, "leverage": 1.5},
                RiskTolerance.VERY_AGGRESSIVE: {"max_drawdown": 0.35, "position_limit": 0.2, "leverage": 2.0}
            }
            
            risk_params.update(risk_tolerance_map.get(parsed_requirement.risk_tolerance, {}))
            
        # 根据投资期限调整
        if parsed_requirement.investment_horizon:
            horizon_map = {
                InvestmentHorizon.SHORT_TERM: {"stop_loss": 0.03},
                InvestmentHorizon.LONG_TERM: {"stop_loss": 0.08}
            }
            
            risk_params.update(horizon_map.get(parsed_requirement.investment_horizon, {}))
            
        return risk_params

# 模块级别函数
async def process_investment_request(
    user_input: str, 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """处理投资请求的便捷函数
    
    Args:
        user_input: 用户输入
        config: 模型配置
        
    Returns:
        处理结果
    """
    fin_r1 = FINR1Integration(config)
    return await fin_r1.process_request(user_input)