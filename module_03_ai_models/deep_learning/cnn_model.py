"""
CNN模型实现 - 用于财务时间序列的卷积神经网络

提供一维卷积神经网络用于捕获时间序列中的局部特征模式
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from common.exceptions import ModelError
from common.logging_system import setup_logger

from ..storage_management.ai_model_database import get_ai_model_database_manager

logger = setup_logger("cnn_model")


@dataclass
class CNNModelConfig:
    """CNN模型配置"""

    sequence_length: int = 20
    num_filters: List[int] = None  # [32, 64, 128]
    kernel_sizes: List[int] = None  # [3, 5, 7]
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    hidden_dim: int = 128
    output_dim: int = 1

    def __post_init__(self):
        if self.num_filters is None:
            self.num_filters = [32, 64, 128]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 5, 7]


@dataclass
class CNNPrediction:
    """CNN预测结果"""

    predictions: np.ndarray
    confidence: float
    features: np.ndarray
    timestamp: str


class CNNModel(nn.Module):
    """一维卷积神经网络模型"""

    def __init__(self, config: CNNModelConfig, input_dim: int):
        super(CNNModel, self).__init__()
        self.config = config
        self.input_dim = input_dim

        # 多个卷积分支
        self.conv_branches = nn.ModuleList()
        for i, (num_filters, kernel_size) in enumerate(
            zip(config.num_filters, config.kernel_sizes)
        ):
            branch = nn.Sequential(
                nn.Conv1d(
                    input_dim, num_filters, kernel_size, padding=kernel_size // 2
                ),
                nn.ReLU(),
                nn.BatchNorm1d(num_filters),
                nn.Dropout(config.dropout),
                nn.AdaptiveAvgPool1d(1),
            )
            self.conv_branches.append(branch)

        # 合并特征
        total_filters = sum(config.num_filters)
        self.fc_layers = nn.Sequential(
            nn.Linear(total_filters, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = x.transpose(1, 2)  # 转换为 (batch_size, input_dim, sequence_length)

        # 并行卷积分支
        branch_outputs = []
        for branch in self.conv_branches:
            branch_out = branch(x)  # (batch_size, num_filters, 1)
            branch_outputs.append(branch_out.squeeze(-1))  # (batch_size, num_filters)

        # 合并所有分支输出
        combined = torch.cat(branch_outputs, dim=1)  # (batch_size, total_filters)

        # 全连接层
        output = self.fc_layers(combined)
        return output


class CNNPredictor:
    """CNN预测器"""

    def __init__(self, config: CNNModelConfig):
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = None
        self.db_manager = get_ai_model_database_manager()
        self.trained = False

    def set_model_id(self, model_id: str):
        """设置模型ID"""
        self.model_id = model_id

    def prepare_data(
        self, data: pd.DataFrame, target_column: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        try:
            # 选择数值列
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            feature_data = data[numeric_columns].drop(columns=[target_column])
            target_data = data[target_column]

            # 标准化特征
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)

            # 创建序列
            X, y = [], []
            for i in range(len(scaled_features) - self.config.sequence_length + 1):
                X.append(scaled_features[i : i + self.config.sequence_length])
                y.append(target_data.iloc[i + self.config.sequence_length - 1])

            return np.array(X), np.array(y)

        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise ModelError(f"Data preparation failed: {e}")

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """训练CNN模型"""
        try:
            # 初始化模型
            input_dim = X.shape[2]
            self.model = CNNModel(self.config, input_dim).to(self.device)

            # 训练设置
            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.learning_rate
            )

            # 准备数据
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            # 训练循环
            self.model.train()
            train_losses = []

            for epoch in range(self.config.epochs):
                epoch_loss = 0
                num_batches = 0

                for i in range(0, len(X), self.config.batch_size):
                    batch_X = X_tensor[i : i + self.config.batch_size]
                    batch_y = y_tensor[i : i + self.config.batch_size]

                    optimizer.zero_grad()
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / num_batches
                train_losses.append(avg_loss)

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}/{self.config.epochs}, Loss: {avg_loss:.6f}"
                    )

            self.trained = True

            # 保存训练历史
            if self.model_id:
                for epoch, loss in enumerate(train_losses):
                    self.db_manager.save_training_history(self.model_id, epoch, loss)

            return {"train_loss": train_losses[-1], "epochs": len(train_losses)}

        except Exception as e:
            logger.error(f"CNN training failed: {e}")
            raise ModelError(f"Training failed: {e}")

    def predict(self, X: np.ndarray) -> CNNPrediction:
        """CNN预测"""
        try:
            if not self.trained or self.model is None:
                raise ModelError("Model not trained yet")

            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = self.model(X_tensor).squeeze()
                predictions = outputs.cpu().numpy()

                # 计算置信度（基于预测方差）
                confidence = 1.0 / (1.0 + np.std(predictions))

                return CNNPrediction(
                    predictions=predictions,
                    confidence=confidence,
                    features=X,
                    timestamp=datetime.now().isoformat(),
                )

        except Exception as e:
            logger.error(f"CNN prediction failed: {e}")
            raise ModelError(f"Prediction failed: {e}")

    def save_model(self, filepath: str) -> bool:
        """保存模型"""
        try:
            if self.model is None:
                return False

            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                    "model_id": self.model_id,
                },
                filepath,
            )

            logger.info(f"CNN model saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save CNN model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            # 重建模型
            if "input_dim" in checkpoint:
                input_dim = checkpoint["input_dim"]
            else:
                # 假设输入维度
                input_dim = 10

            self.model = CNNModel(checkpoint["config"], input_dim).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.config = checkpoint["config"]
            self.model_id = checkpoint.get("model_id")
            self.trained = True

            logger.info(f"CNN model loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load CNN model: {e}")
            return False


def create_cnn_predictor(
    sequence_length: int = 20,
    num_filters: List[int] = None,
    kernel_sizes: List[int] = None,
    learning_rate: float = 0.001,
) -> CNNPredictor:
    """创建CNN预测器的便捷函数"""
    config = CNNModelConfig(
        sequence_length=sequence_length,
        num_filters=num_filters or [32, 64, 128],
        kernel_sizes=kernel_sizes or [3, 5, 7],
        learning_rate=learning_rate,
    )
    return CNNPredictor(config)
