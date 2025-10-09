"""
FIN-R1模型集成模块
负责集成和调用FIN-R1大语言模型进行投资需求解析
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

# 尝试导入可选依赖
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from common.exceptions import ModelError
from common.logging_system import setup_logger
from common.model_inference_visualizer import (
    ModelInferenceVisualizer,
    display_model_info,
)
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

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None
    ):
        """初始化FIN-R1集成

        Args:
            config: 模型配置字典（可选）
            config_path: 配置文件路径（可选）
        """
        # 加载配置
        if config is None:
            if config_path is None:
                config_path = os.path.join(
                    "module_10_ai_interaction", "config", "fin_r1_config.yaml"
                )
            config = self._load_config(config_path)

        # 确保config是字典
        if config is None:
            config = {}

        self.config = config
        model_config = config.get("model", {})

        self.model_path = model_config.get("model_path", ".Fin-R1")
        self.device = model_config.get("device", "cpu")
        self.batch_size = model_config.get("batch_size", 1)
        self.max_length = model_config.get("max_length", 2048)
        self.temperature = model_config.get("temperature", 0.7)
        self.top_p = model_config.get("top_p", 0.9)
        self.top_k = model_config.get("top_k", 50)
        self.do_sample = model_config.get("do_sample", True)
        self.repetition_penalty = model_config.get("repetition_penalty", 1.1)

        # 提示词模板
        self.prompts = config.get("prompts", {})

        # 性能配置
        performance_config = config.get("performance", {})
        self.timeout = performance_config.get("timeout", 30)
        self.max_retries = performance_config.get("max_retries", 3)

        # 初始化模型和分词器
        self.model = None
        self.tokenizer = None
        self.requirement_parser = RequirementParser()

        # 初始化可视化器
        self.visualizer = ModelInferenceVisualizer()

        self._load_model()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        try:
            if HAS_YAML:
                import os

                # 处理相对路径
                if not os.path.isabs(config_path):
                    base_dir = os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                    config_path = os.path.join(base_dir, config_path)

                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    # 确保返回字典而不是None（空文件会返回None）
                    return config if config is not None else {}
            else:
                logger.warning("PyYAML not available, using default config")
                return {}
        except Exception as e:
            logger.warning(
                f"Failed to load config from {config_path}: {e}, using default"
            )
            return {}

    def _load_model(self):
        """加载模型"""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                logger.warning(
                    "Transformers or PyTorch not available, using mock model"
                )
                self.model = None
                self.tokenizer = None
                return

            import os

            if not os.path.exists(self.model_path):
                logger.warning(
                    f"Model path {self.model_path} does not exist, using mock model"
                )
                self.model = None
                self.tokenizer = None
                return

            logger.info(f"Loading FIN-R1 model from {self.model_path}...")

            # 显示加载进度
            if self.visualizer and self.visualizer.console:
                self.visualizer.console.print(
                    "\n[bold cyan]🚀 Initializing FIN-R1 Model[/bold cyan]\n"
                )

            # 加载分词器
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, use_fast=False
            )

            # 设置pad_token（如果未设置）
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载模型（显示进度）
            def load_model_with_progress(progress=None, task=None):
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    dtype=torch.float32,  # 🔧 修复：使用 dtype 而不是 torch_dtype
                    low_cpu_mem_usage=True,  # 减少CPU内存占用
                )
                if progress and task:
                    progress.update(task, advance=50)

                model.to(self.device)
                model.eval()

                if progress and task:
                    progress.update(task, advance=50)
                return model

            self.model = (
                self.visualizer.simple_progress_bar(
                    "[cyan]📦 Loading model weights",
                    total=100,
                    callback=load_model_with_progress,
                )
                if self.visualizer
                else load_model_with_progress()
            )

            logger.info("✅ FIN-R1 model loaded successfully")

            # 显示模型信息
            display_model_info("FIN-R1", "7B")
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
                "timestamp": datetime.now().isoformat(),
            }

            logger.info("Request processed successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            raise ModelError(f"Request processing failed: {e}")

    def _invoke_fin_r1_model(
        self, user_input: str, parsed_requirement: ParsedRequirement
    ) -> Dict[str, Any]:
        """调用FIN-R1模型

        Args:
            user_input: 用户输入
            parsed_requirement: 解析后的需求

        Returns:
            模型输出
        """
        try:
            # 如果模型未加载，使用基于规则的方法
            if self.model is None or self.tokenizer is None:
                logger.info("Using rule-based fallback (model not loaded)")
                return self._rule_based_analysis(user_input, parsed_requirement)

            # 准备输入
            input_text = self._prepare_model_input(user_input, parsed_requirement)

            logger.info(f"Invoking FIN-R1 model with input length: {len(input_text)}")

            # 分词
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.max_length,
                padding=False,
                truncation=True,
            )

            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 🚀 使用流式生成 + 可视化进度
            logger.info("Starting model generation with visualization...")

            max_new_tokens = 50  # 增加到50以展示进度

            try:
                # 尝试使用流式生成（如果支持）
                from threading import Thread

                from transformers import TextIteratorStreamer

                # 🔧 设置 timeout 防止死锁！
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                    timeout=30.0,  # 🔑 关键：30秒超时，防止无限等待
                )

                # 在后台线程运行生成
                generation_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # 使用贪心解码（更快）
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

                # 🔑 必须先启动生成线程
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                # 🔑 立即开始消费 streamer，避免死锁
                logger.info("🚀 Starting streaming generation...")

                # 🎯 使用可视化器显示流式生成进度
                generated_text = self.visualizer.visualize_generation(
                    streamer, max_new_tokens=max_new_tokens, model_name="FIN-R1"
                )

                # 等待生成线程完成
                thread.join(timeout=60)  # 最多等待60秒

                if thread.is_alive():
                    logger.warning("⚠️ Generation thread timeout, may be incomplete")
                    raise TimeoutError("Model generation timeout")

            except Exception as stream_error:
                # Fallback: 非流式生成（带简单进度）
                logger.warning(
                    f"Streaming not available: {stream_error}, using standard generation"
                )

                if self.visualizer and self.visualizer.console:
                    self.visualizer.console.print(
                        "[yellow]⚡ Using fast non-streaming mode[/yellow]"
                    )

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )

                generated_text = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )

            logger.info("✅ Model generation completed")

            # 提取输入后的生成部分
            if input_text in generated_text:
                generated_text = generated_text.split(input_text)[-1].strip()

            logger.info(f"Model generated text: {generated_text[:200]}...")

            # ⚠️ 由于FIN-R1是通用模型，生成的文本可能不是结构化输出
            # 我们主要使用规则引擎的结果，模型输出作为辅助参考
            logger.info("⚠️ 注意：当前使用规则引擎+模型辅助的混合模式")

            # 解析输出（主要使用规则引擎）
            model_output = self._rule_based_analysis(user_input, parsed_requirement)
            model_output["fin_r1_generated_text"] = (
                generated_text  # 保存模型生成文本作为参考
            )

            return model_output

        except Exception as e:
            logger.error(f"Failed to invoke FIN-R1 model: {e}")
            # 使用基于规则的fallback
            return self._rule_based_analysis(user_input, parsed_requirement)

    def _rule_based_analysis(
        self, user_input: str, parsed_requirement: ParsedRequirement
    ) -> Dict[str, Any]:
        """基于规则的分析（fallback方法）

        Args:
            user_input: 用户输入
            parsed_requirement: 解析后的需求

        Returns:
            分析结果
        """
        # 基于风险偏好确定策略
        strategy_map = {
            RiskTolerance.CONSERVATIVE: "conservative",
            RiskTolerance.MODERATE: "balanced",
            RiskTolerance.AGGRESSIVE: "growth",
            RiskTolerance.VERY_AGGRESSIVE: "aggressive",
        }

        strategy = strategy_map.get(parsed_requirement.risk_tolerance, "balanced")

        # 风险评估
        risk_map = {
            RiskTolerance.CONSERVATIVE: "low",
            RiskTolerance.MODERATE: "moderate",
            RiskTolerance.AGGRESSIVE: "high",
            RiskTolerance.VERY_AGGRESSIVE: "very_high",
        }

        risk_assessment = risk_map.get(parsed_requirement.risk_tolerance, "moderate")

        # 行业偏好
        sector_preferences = []
        if any(
            "科技" in goal.description or "technology" in goal.description.lower()
            for goal in parsed_requirement.investment_goals
        ):
            sector_preferences.extend(["technology", "software"])
        if any(
            "医疗" in goal.description or "health" in goal.description.lower()
            for goal in parsed_requirement.investment_goals
        ):
            sector_preferences.extend(["healthcare", "biotech"])

        if not sector_preferences:
            sector_preferences = ["diversified"]

        return {
            "strategy_recommendation": strategy,
            "risk_assessment": risk_assessment,
            "confidence_score": 0.75,
            "market_outlook": "neutral",
            "sector_preferences": sector_preferences,
            "excluded_sectors": [],
            "analysis_method": "rule_based",
        }

    def _parse_model_output(
        self, generated_text: str, parsed_requirement: ParsedRequirement
    ) -> Dict[str, Any]:
        """解析模型输出

        Args:
            generated_text: 模型生成的文本
            parsed_requirement: 解析后的需求

        Returns:
            结构化输出
        """
        # 尝试解析JSON
        try:
            import re

            json_match = re.search(r"\{[^{}]*\}", generated_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        # 使用规则解析
        output = self._rule_based_analysis("", parsed_requirement)
        output["raw_output"] = generated_text
        output["analysis_method"] = "model_generation"

        # 尝试从文本中提取关键信息
        text_lower = generated_text.lower()

        # 策略识别
        if any(word in text_lower for word in ["保守", "conservative", "稳健"]):
            output["strategy_recommendation"] = "conservative"
        elif any(word in text_lower for word in ["激进", "aggressive", "进取"]):
            output["strategy_recommendation"] = "aggressive"
        elif any(word in text_lower for word in ["成长", "growth"]):
            output["strategy_recommendation"] = "growth"

        # 风险识别
        if any(word in text_lower for word in ["低风险", "low risk"]):
            output["risk_assessment"] = "low"
        elif any(word in text_lower for word in ["高风险", "high risk"]):
            output["risk_assessment"] = "high"

        return output

    def _prepare_model_input(
        self, user_input: str, parsed_requirement: ParsedRequirement
    ) -> str:
        """准备模型输入

        Args:
            user_input: 用户输入
            parsed_requirement: 解析后的需求

        Returns:
            模型输入文本
        """
        # 使用配置的提示词模板
        template = self.prompts.get("investment_analysis", "")

        if template:
            # 格式化模板
            input_text = template.format(
                user_input=user_input,
                investment_amount=parsed_requirement.investment_amount or "未指定",
                risk_tolerance=parsed_requirement.risk_tolerance.value
                if parsed_requirement.risk_tolerance
                else "未指定",
                investment_horizon=parsed_requirement.investment_horizon.value
                if parsed_requirement.investment_horizon
                else "未指定",
                investment_goals=", ".join(
                    [goal.goal_type for goal in parsed_requirement.investment_goals]
                )
                if parsed_requirement.investment_goals
                else "未指定",
                constraints=", ".join(
                    [
                        constraint.constraint_type
                        for constraint in parsed_requirement.constraints
                    ]
                )
                if parsed_requirement.constraints
                else "无",
            )
        else:
            # 默认格式
            input_parts = [
                f"用户需求：{user_input}",
                f"投资金额：{parsed_requirement.investment_amount or '未指定'}",
                f"风险偏好：{parsed_requirement.risk_tolerance.value if parsed_requirement.risk_tolerance else '未指定'}",
                f"投资期限：{parsed_requirement.investment_horizon.value if parsed_requirement.investment_horizon else '未指定'}",
                f"投资目标：{[goal.goal_type for goal in parsed_requirement.investment_goals]}",
                f"约束条件：{[constraint.constraint_type for constraint in parsed_requirement.constraints]}",
            ]
            input_text = "\n".join(input_parts)

        return input_text

    def _generate_strategy_parameters(
        self, parsed_requirement: ParsedRequirement, model_output: Dict[str, Any]
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
                "value": 0.2,
            },
            "rebalance_frequency": "weekly",
            "position_sizing_method": "kelly_criterion",
            "risk_adjustment": True,
        }

        # 根据风险偏好调整
        if parsed_requirement.risk_tolerance:
            if parsed_requirement.risk_tolerance == RiskTolerance.CONSERVATIVE:
                strategy_params["strategy_mix"] = {
                    "trend_following": 0.2,
                    "mean_reversion": 0.4,
                    "momentum": 0.1,
                    "value": 0.3,
                }
                strategy_params["rebalance_frequency"] = "monthly"
            elif parsed_requirement.risk_tolerance == RiskTolerance.AGGRESSIVE:
                strategy_params["strategy_mix"] = {
                    "trend_following": 0.4,
                    "mean_reversion": 0.1,
                    "momentum": 0.3,
                    "value": 0.2,
                }
                strategy_params["rebalance_frequency"] = "daily"

        # 根据模型输出调整
        if model_output.get("strategy_recommendation") == "growth":
            strategy_params["strategy_mix"]["momentum"] += 0.1
            strategy_params["strategy_mix"]["value"] -= 0.1

        return strategy_params

    def _generate_risk_parameters(
        self, parsed_requirement: ParsedRequirement
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
            "stop_loss": 0.05,
        }

        # 根据风险偏好调整
        if parsed_requirement.risk_tolerance:
            risk_tolerance_map = {
                RiskTolerance.CONSERVATIVE: {
                    "max_drawdown": 0.08,
                    "position_limit": 0.05,
                    "leverage": 1.0,
                    "stop_loss": 0.03,
                },
                RiskTolerance.MODERATE: {
                    "max_drawdown": 0.15,
                    "position_limit": 0.1,
                    "leverage": 1.0,
                    "stop_loss": 0.05,
                },
                RiskTolerance.AGGRESSIVE: {
                    "max_drawdown": 0.25,
                    "position_limit": 0.15,
                    "leverage": 1.5,
                    "stop_loss": 0.07,
                },
                RiskTolerance.VERY_AGGRESSIVE: {
                    "max_drawdown": 0.35,
                    "position_limit": 0.2,
                    "leverage": 2.0,
                    "stop_loss": 0.10,
                },
            }

            risk_params.update(
                risk_tolerance_map.get(parsed_requirement.risk_tolerance, {})
            )

        # 根据投资期限调整
        if parsed_requirement.investment_horizon:
            horizon_map = {
                InvestmentHorizon.SHORT_TERM: {"stop_loss": 0.03},
                InvestmentHorizon.LONG_TERM: {"stop_loss": 0.08},
            }

            risk_params.update(
                horizon_map.get(parsed_requirement.investment_horizon, {})
            )

        return risk_params


# 模块级别函数
async def process_investment_request(
    user_input: str, config: Dict[str, Any]
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
