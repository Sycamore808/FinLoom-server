"""
FIN-R1金融大语言模型集成模块
负责集成和管理FIN-R1模型，提供金融领域的专业AI能力
"""

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from common.exceptions import QuantSystemError
from common.logging_system import setup_logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

logger = setup_logger("fin_r1_integration")


@dataclass
class FINModelConfig:
    """FIN-R1模型配置"""

    model_name: str = "SUFE/FIN-R1-8B"
    model_path: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1
    do_sample: bool = True
    repetition_penalty: float = 1.2
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = True
    cache_dir: str = "./model_cache"


@dataclass
class GenerationResult:
    """生成结果数据结构"""

    text: str
    confidence_score: float
    tokens_used: int
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinancialContext:
    """金融上下文信息"""

    user_profile: Dict[str, Any]
    market_conditions: Dict[str, Any]
    portfolio_status: Dict[str, Any]
    risk_parameters: Dict[str, Any]
    historical_performance: Dict[str, Any]
    constraints: List[str]
    preferences: Dict[str, Any]


class FINR1Model:
    """FIN-R1模型封装类"""

    SYSTEM_PROMPT = """你是一个专业的金融投资顾问AI助手，基于上海财经大学开发的FIN-R1金融大语言模型。
你具备以下专业能力：
1. 深入理解金融市场和投资策略
2. 精准分析用户的投资需求和风险偏好
3. 提供专业的资产配置建议
4. 解释复杂的金融概念和策略逻辑
5. 实时分析市场动态并给出投资建议

请始终保持专业、客观、谨慎的态度，为用户提供有价值的投资指导。"""

    def __init__(self, config: Optional[FINModelConfig] = None):
        """初始化FIN-R1模型

        Args:
            config: 模型配置对象
        """
        self.config = config or FINModelConfig()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.generation_queue = Queue()
        self.is_initialized = False

    def initialize(self) -> bool:
        """初始化模型

        Returns:
            是否成功初始化
        """
        try:
            logger.info("Initializing FIN-R1 model...")

            # 设置设备
            if self.config.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.config.device = "cpu"
            self.device = torch.device(self.config.device)

            # 配置量化
            quantization_config = None
            if self.config.load_in_8bit or self.config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=self.config.load_in_8bit,
                    load_in_4bit=self.config.load_in_4bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path or self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True,
            )

            # 设置padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载模型
            model_kwargs = {
                "cache_dir": self.config.cache_dir,
                "torch_dtype": torch.float16
                if self.device.type == "cuda"
                else torch.float32,
                "device_map": "auto" if self.device.type == "cuda" else None,
                "trust_remote_code": True,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            if self.config.use_flash_attention and self.device.type == "cuda":
                model_kwargs["attn_implementation"] = "flash_attention_2"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path or self.config.model_name, **model_kwargs
            )

            # 设置为评估模式
            self.model.eval()

            self.is_initialized = True
            logger.info("FIN-R1 model initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize FIN-R1 model: {e}")
            raise QuantSystemError(f"Model initialization failed: {e}")

    def generate_response(
        self,
        prompt: str,
        context: Optional[FinancialContext] = None,
        stream: bool = False,
        max_new_tokens: int = 512,
    ) -> GenerationResult:
        """生成响应

        Args:
            prompt: 用户输入提示
            context: 金融上下文信息
            stream: 是否流式生成
            max_new_tokens: 最大生成token数

        Returns:
            生成结果
        """
        if not self.is_initialized:
            raise QuantSystemError("Model not initialized")

        start_time = datetime.now()

        # 构建完整提示
        full_prompt = self._build_prompt(prompt, context)

        # 编码输入
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length - max_new_tokens,
            padding=True,
        ).to(self.device)

        # 生成参数
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "num_beams": self.config.num_beams,
            "do_sample": self.config.do_sample,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if stream:
            return self._generate_stream(inputs, generation_kwargs, start_time)
        else:
            return self._generate_batch(inputs, generation_kwargs, start_time)

    def _generate_batch(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, Any],
        start_time: datetime,
    ) -> GenerationResult:
        """批量生成（非流式）

        Args:
            inputs: 编码后的输入
            generation_kwargs: 生成参数
            start_time: 开始时间

        Returns:
            生成结果
        """
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_kwargs,
            )

        # 解码输出
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # 计算置信度分数（简化版本）
        confidence_score = self._calculate_confidence(outputs)

        generation_time = (datetime.now() - start_time).total_seconds()

        return GenerationResult(
            text=generated_text,
            confidence_score=confidence_score,
            tokens_used=len(generated_ids),
            generation_time=generation_time,
            metadata={
                "model": self.config.model_name,
                "device": str(self.device),
                "temperature": self.config.temperature,
            },
        )

    def _generate_stream(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, Any],
        start_time: datetime,
    ) -> GenerationResult:
        """流式生成

        Args:
            inputs: 编码后的输入
            generation_kwargs: 生成参数
            start_time: 开始时间

        Returns:
            生成结果
        """
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        generation_kwargs["streamer"] = streamer

        # 在单独线程中运行生成
        generation_thread = threading.Thread(
            target=self.model.generate,
            kwargs={
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                **generation_kwargs,
            },
        )
        generation_thread.start()

        # 收集生成的文本
        generated_text = ""
        for text in streamer:
            generated_text += text
            # 可以在这里实现流式输出回调

        generation_thread.join()

        generation_time = (datetime.now() - start_time).total_seconds()

        return GenerationResult(
            text=generated_text,
            confidence_score=0.85,  # 流式生成使用默认置信度
            tokens_used=len(self.tokenizer.encode(generated_text)),
            generation_time=generation_time,
            metadata={
                "model": self.config.model_name,
                "device": str(self.device),
                "stream": True,
            },
        )

    def _build_prompt(
        self, user_input: str, context: Optional[FinancialContext]
    ) -> str:
        """构建完整的提示

        Args:
            user_input: 用户输入
            context: 金融上下文

        Returns:
            完整的提示文本
        """
        prompt_parts = [self.SYSTEM_PROMPT]

        if context:
            # 添加上下文信息
            context_prompt = self._format_context(context)
            prompt_parts.append(context_prompt)

        # 添加用户输入
        prompt_parts.append(f"\n用户问题：{user_input}\n\n助手回答：")

        return "\n\n".join(prompt_parts)

    def _format_context(self, context: FinancialContext) -> str:
        """格式化上下文信息

        Args:
            context: 金融上下文

        Returns:
            格式化的上下文文本
        """
        context_parts = ["当前投资环境和用户信息："]

        # 用户画像
        if context.user_profile:
            context_parts.append(
                f"用户画像：{json.dumps(context.user_profile, ensure_ascii=False)}"
            )

        # 市场状况
        if context.market_conditions:
            context_parts.append(
                f"市场状况：{json.dumps(context.market_conditions, ensure_ascii=False)}"
            )

        # 组合状态
        if context.portfolio_status:
            context_parts.append(
                f"当前组合：{json.dumps(context.portfolio_status, ensure_ascii=False)}"
            )

        # 风险参数
        if context.risk_parameters:
            context_parts.append(
                f"风险限制：{json.dumps(context.risk_parameters, ensure_ascii=False)}"
            )

        # 约束条件
        if context.constraints:
            context_parts.append(f"约束条件：{', '.join(context.constraints)}")

        return "\n".join(context_parts)

    def _calculate_confidence(self, outputs: torch.Tensor) -> float:
        """计算生成结果的置信度

        Args:
            outputs: 模型输出

        Returns:
            置信度分数
        """
        # 简化的置信度计算
        # 实际应该基于输出概率分布的熵等指标
        return 0.85

    def finetune_on_domain_data(
        self,
        training_data: List[Dict[str, str]],
        validation_data: Optional[List[Dict[str, str]]] = None,
        output_dir: str = "./finetuned_model",
    ) -> bool:
        """在领域数据上微调模型

        Args:
            training_data: 训练数据
            validation_data: 验证数据
            output_dir: 输出目录

        Returns:
            是否成功微调
        """
        if not self.is_initialized:
            raise QuantSystemError("Model not initialized")

        try:
            from torch.utils.data import Dataset
            from transformers import Trainer, TrainingArguments

            class FineTuneDataset(Dataset):
                def __init__(self, data, tokenizer, max_length=512):
                    self.data = data
                    self.tokenizer = tokenizer
                    self.max_length = max_length

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    item = self.data[idx]
                    text = f"{item['input']}\n{item['output']}"

                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt",
                    )

                    return {
                        "input_ids": encoding["input_ids"].squeeze(),
                        "attention_mask": encoding["attention_mask"].squeeze(),
                        "labels": encoding["input_ids"].squeeze(),
                    }

            # 创建数据集
            train_dataset = FineTuneDataset(training_data, self.tokenizer)
            val_dataset = (
                FineTuneDataset(validation_data, self.tokenizer)
                if validation_data
                else None
            )

            # 训练参数
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=10,
                save_steps=500,
                evaluation_strategy="steps" if val_dataset else "no",
                eval_steps=100 if val_dataset else None,
                save_total_limit=2,
                load_best_model_at_end=True if val_dataset else False,
                metric_for_best_model="eval_loss" if val_dataset else None,
                greater_is_better=False if val_dataset else None,
                fp16=self.device.type == "cuda",
                gradient_checkpointing=True,
                gradient_accumulation_steps=4,
            )

            # 创建训练器
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
            )

            # 开始训练
            logger.info("Starting fine-tuning...")
            trainer.train()

            # 保存模型
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return False

    def explain_strategy_rationale(
        self,
        strategy_name: str,
        parameters: Dict[str, Any],
        performance_metrics: Dict[str, float],
    ) -> str:
        """解释策略逻辑

        Args:
            strategy_name: 策略名称
            parameters: 策略参数
            performance_metrics: 性能指标

        Returns:
            策略解释文本
        """
        prompt = f"""请解释以下投资策略的逻辑和原理：

策略名称：{strategy_name}
策略参数：{json.dumps(parameters, ensure_ascii=False, indent=2)}
历史表现：{json.dumps(performance_metrics, ensure_ascii=False, indent=2)}

请从以下方面解释：
1. 策略的核心理念和投资逻辑
2. 适用的市场环境
3. 主要风险和收益特征
4. 参数设置的合理性
5. 历史表现分析"""

        result = self.generate_response(prompt, max_new_tokens=1024)
        return result.text


# 模块级别函数
def create_fin_r1_model(config_path: Optional[str] = None) -> FINR1Model:
    """创建FIN-R1模型实例

    Args:
        config_path: 配置文件路径

    Returns:
        FIN-R1模型实例
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config = FINModelConfig(**config_dict)
    else:
        config = FINModelConfig()

    model = FINR1Model(config)
    model.initialize()
    return model


def test_model_response(model: FINR1Model) -> bool:
    """测试模型响应

    Args:
        model: FIN-R1模型实例

    Returns:
        测试是否通过
    """
    test_prompt = "请简要介绍一下价值投资策略"

    try:
        result = model.generate_response(test_prompt, max_new_tokens=256)
        logger.info(f"Test response: {result.text[:100]}...")
        return len(result.text) > 0
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        return False
