"""
Transformer预测模型模块
实现时序Transformer用于金融预测
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.constants import DEFAULT_BATCH_SIZE, MAX_EPOCHS
from common.exceptions import ModelError
from common.logging_system import setup_logger

logger = setup_logger("transformer_model")


@dataclass
class TransformerConfig:
    """Transformer配置"""

    input_dim: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_length: int = 100
    dropout: float = 0.1
    learning_rate: float = 0.0001
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = MAX_EPOCHS
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    use_relative_position: bool = True


class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model: int, max_len: int = 5000):
        """初始化位置编码

        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 (seq_len, batch_size, d_model)

        Returns:
            位置编码后的张量
        """
        return x + self.pe[: x.size(0), :]


class RelativePositionEncoding(nn.Module):
    """相对位置编码"""

    def __init__(self, d_model: int, max_relative_position: int = 32):
        """初始化相对位置编码

        Args:
            d_model: 模型维度
            max_relative_position: 最大相对位置
        """
        super(RelativePositionEncoding, self).__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position

        # 相对位置嵌入
        self.relative_position_embeddings = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )

    def forward(self, seq_len: int) -> torch.Tensor:
        """生成相对位置编码

        Args:
            seq_len: 序列长度

        Returns:
            相对位置编码张量
        """
        positions = torch.arange(seq_len, dtype=torch.long)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)

        # 裁剪到最大相对位置
        relative_positions = torch.clamp(
            relative_positions, -self.max_relative_position, self.max_relative_position
        )

        # 偏移使索引为正
        relative_positions = relative_positions + self.max_relative_position

        return self.relative_position_embeddings(relative_positions)


class MultiHeadAttention(nn.Module):
    """多头注意力层"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """初始化多头注意力

        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout率
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            mask: 掩码张量

        Returns:
            输出张量和注意力权重
        """
        batch_size = query.size(0)
        seq_len = query.size(1)

        # 线性变换并重塑为多头
        Q = (
            self.W_q(query)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_k(key)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_v(value)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        context = torch.matmul(attention_weights, V)

        # 重塑并线性变换
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        output = self.W_o(context)

        return output, attention_weights


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """初始化Transformer块

        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: Dropout率
        """
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量
            mask: 掩码张量

        Returns:
            输出张量
        """
        # 自注意力
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TemporalTransformer(nn.Module):
    """时序Transformer模型"""

    def __init__(self, config: TransformerConfig):
        """初始化时序Transformer

        Args:
            config: Transformer配置
        """
        super(TemporalTransformer, self).__init__()
        self.config = config

        # 输入投影
        self.input_projection = nn.Linear(config.input_dim, config.d_model)

        # 位置编码
        if config.use_relative_position:
            self.pos_encoding = RelativePositionEncoding(config.d_model)
        else:
            self.pos_encoding = PositionalEncoding(
                config.d_model, config.max_seq_length
            )

        # Transformer层
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model, config.n_heads, config.d_ff, config.dropout
                )
                for _ in range(config.n_layers)
            ]
        )

        # 输出层
        self.output_projection = nn.Linear(config.d_model, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            mask: 掩码张量

        Returns:
            预测值和中间结果字典
        """
        batch_size, seq_len, _ = x.size()

        # 输入投影
        x = self.input_projection(x)

        # 添加位置编码
        if self.config.use_relative_position:
            rel_pos = self.pos_encoding(seq_len)
            # 这里简化处理，实际应用中需要在注意力计算中使用
            x = x + rel_pos.mean(dim=0).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)

        x = self.dropout(x)

        # 通过Transformer块
        intermediate_outputs = []
        for block in self.transformer_blocks:
            x = block(x, mask)
            intermediate_outputs.append(x)

        # 输出预测
        output = self.output_projection(x)

        return output, {
            "intermediate_outputs": intermediate_outputs,
            "final_representation": x,
        }


class TransformerPredictor:
    """Transformer预测器"""

    def __init__(self, config: TransformerConfig):
        """初始化预测器

        Args:
            config: Transformer配置
        """
        self.config = config
        self.model = TemporalTransformer(config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        self.scheduler = self._create_scheduler()
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def build_temporal_transformer(
        self, input_shape: Tuple[int, int]
    ) -> TemporalTransformer:
        """构建时序Transformer

        Args:
            input_shape: 输入形状 (序列长度, 特征维度)

        Returns:
            Transformer模型
        """
        seq_len, input_dim = input_shape

        config = TransformerConfig(input_dim=input_dim, max_seq_length=seq_len)

        self.config = config
        self.model = TemporalTransformer(config)

        logger.info(f"Built Transformer with {self._count_parameters()} parameters")
        return self.model

    def implement_cross_attention(
        self, query_features: torch.Tensor, key_value_features: torch.Tensor
    ) -> torch.Tensor:
        """实现交叉注意力

        Args:
            query_features: 查询特征
            key_value_features: 键值特征

        Returns:
            交叉注意力输出
        """
        # 使用模型的注意力层
        attention = MultiHeadAttention(
            self.config.d_model, self.config.n_heads, self.config.dropout
        )

        output, weights = attention(
            query_features, key_value_features, key_value_features
        )

        return output

    def apply_position_encoding(
        self, embeddings: torch.Tensor, method: str = "sinusoidal"
    ) -> torch.Tensor:
        """应用位置编码

        Args:
            embeddings: 嵌入张量
            method: 编码方法 ('sinusoidal', 'learned', 'relative')

        Returns:
            位置编码后的张量
        """
        if method == "sinusoidal":
            pe = PositionalEncoding(embeddings.size(-1))
            return pe(embeddings.transpose(0, 1)).transpose(0, 1)
        elif method == "learned":
            # 学习的位置编码
            seq_len = embeddings.size(1)
            pos_emb = nn.Embedding(seq_len, embeddings.size(-1))
            positions = torch.arange(seq_len)
            return embeddings + pos_emb(positions).unsqueeze(0)
        elif method == "relative":
            rpe = RelativePositionEncoding(embeddings.size(-1))
            rel_pos = rpe(embeddings.size(1))
            return embeddings + rel_pos.mean(dim=0).unsqueeze(0)
        else:
            raise ValueError(f"Unknown position encoding method: {method}")

    def generate_multihorizon_predictions(
        self, features: pd.DataFrame, horizons: List[int] = [1, 5, 10, 20]
    ) -> Dict[int, np.ndarray]:
        """生成多时间尺度预测

        Args:
            features: 特征DataFrame
            horizons: 预测时间尺度列表

        Returns:
            各时间尺度的预测结果
        """
        self.model.eval()
        predictions = {}

        X = torch.FloatTensor(features.values).unsqueeze(0)

        with torch.no_grad():
            for horizon in horizons:
                # 对每个时间尺度进行预测
                horizon_preds = []

                current_input = X
                for _ in range(horizon):
                    output, _ = self.model(current_input)
                    horizon_preds.append(output[:, -1, :])

                    # 更新输入（滑动窗口）
                    current_input = torch.cat(
                        [current_input[:, 1:, :], output[:, -1:, :]], dim=1
                    )

                predictions[horizon] = torch.cat(horizon_preds, dim=1).numpy()

        return predictions

    def calculate_prediction_uncertainty(
        self, features: pd.DataFrame, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算预测不确定性

        Args:
            features: 特征DataFrame
            n_samples: MC Dropout采样数

        Returns:
            预测均值和标准差
        """
        self.model.train()  # 启用Dropout

        X = torch.FloatTensor(features.values).unsqueeze(0)
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                output, _ = self.model(X)
                predictions.append(output.numpy())

        predictions = np.array(predictions)

        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)

        return mean_pred, std_pred

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """创建学习率调度器

        Returns:
            调度器
        """

        def lr_lambda(step):
            if step == 0:
                return 1
            arg1 = step**-0.5
            arg2 = step * (self.config.warmup_steps**-1.5)
            return self.config.d_model**-0.5 * min(arg1, arg2)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _count_parameters(self) -> int:
        """统计模型参数数量

        Returns:
            参数数量
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# 模块级别函数
def create_transformer_predictor(
    input_dim: int, seq_length: int = 100, config: Optional[TransformerConfig] = None
) -> TransformerPredictor:
    """创建Transformer预测器的便捷函数

    Args:
        input_dim: 输入维度
        seq_length: 序列长度
        config: Transformer配置

    Returns:
        Transformer预测器
    """
    if config is None:
        config = TransformerConfig(input_dim=input_dim, max_seq_length=seq_length)

    return TransformerPredictor(config)
