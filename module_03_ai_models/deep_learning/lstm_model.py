"""
LSTM模型模块
实现基于PyTorch的LSTM金融预测模型
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from common.exceptions import ModelError
from common.logging_system import setup_logger

from ..storage_management.ai_model_database import get_ai_model_database_manager

logger = setup_logger("lstm_model")


@dataclass
class LSTMModelConfig:
    """LSTM模型配置"""

    sequence_length: int = 60
    hidden_size: int = 50
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100


@dataclass
class LSTMPrediction:
    """LSTM预测结果"""

    predictions: np.ndarray
    confidence: float
    model_metrics: Dict[str, float]


class LSTMNet(nn.Module):
    """PyTorch LSTM网络"""

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # 只取最后一个时间步
        out = self.fc(out)
        return out


class LSTMModel:
    """LSTM模型类"""

    def __init__(self, config: LSTMModelConfig):
        """初始化LSTM模型

        Args:
            config: 模型配置
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = None
        self.db_manager = get_ai_model_database_manager()

    def prepare_data(
        self, data: pd.DataFrame, target_column: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备训练数据

        Args:
            data: 输入数据
            target_column: 目标列名

        Returns:
            (X, y) 训练数据张量
        """
        try:
            # 选择特征列（排除目标列和非数值列）
            # 只保留数值型列
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            feature_columns = [
                col
                for col in numeric_columns
                if col != target_column and col != "symbol"
            ]

            if len(feature_columns) == 0:
                raise ModelError("No numeric feature columns found")

            # 标准化特征数据
            feature_data = self.scaler.fit_transform(data[feature_columns].values)
            target_data = data[target_column].values

            X, y = [], []

            # 创建序列数据
            for i in range(self.config.sequence_length, len(feature_data)):
                X.append(feature_data[i - self.config.sequence_length : i])
                y.append(target_data[i])

            X = np.array(X)
            y = np.array(y)

            # 转换为PyTorch张量
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            # 初始化模型
            if self.model is None:
                input_size = X.shape[2]  # 特征维度
                self.model = LSTMNet(
                    input_size=input_size,
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    dropout=self.config.dropout,
                ).to(self.device)

                self.optimizer = optim.Adam(
                    self.model.parameters(), lr=self.config.learning_rate
                )

            return X_tensor, y_tensor

        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise ModelError(f"Data preparation failed: {e}")

    def train(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """训练模型

        Args:
            X: 输入特征张量
            y: 目标值张量

        Returns:
            训练指标
        """
        try:
            if self.model is None:
                raise ModelError("Model not initialized. Call prepare_data first.")

            logger.info(f"Training LSTM model with {len(X)} samples")

            # 分割训练和验证数据
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            train_losses = []
            val_losses = []

            self.model.train()

            for epoch in range(self.config.epochs):
                # 训练阶段
                epoch_train_loss = 0
                num_batches = 0

                for i in range(0, len(X_train), self.config.batch_size):
                    batch_X = X_train[i : i + self.config.batch_size]
                    batch_y = y_train[i : i + self.config.batch_size]

                    self.optimizer.zero_grad()

                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs.squeeze(), batch_y)

                    loss.backward()
                    self.optimizer.step()

                    epoch_train_loss += loss.item()
                    num_batches += 1

                avg_train_loss = epoch_train_loss / num_batches
                train_losses.append(avg_train_loss)

                # 验证阶段
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = self.criterion(val_outputs.squeeze(), y_val)
                    val_losses.append(val_loss.item())

                # 保存训练历史到数据库
                if self.model_id and epoch % 10 == 0:
                    self.db_manager.save_training_history(
                        model_id=self.model_id,
                        epoch=epoch,
                        train_loss=avg_train_loss,
                        val_loss=val_loss.item(),
                    )

                if epoch % 20 == 0:
                    logger.info(
                        f"Epoch {epoch}/{self.config.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )

                self.model.train()

            self.is_trained = True

            # 计算最终指标
            final_metrics = {
                "train_loss": train_losses[-1],
                "val_loss": val_losses[-1],
                "min_train_loss": min(train_losses),
                "min_val_loss": min(val_losses),
            }

            logger.info("LSTM model training completed")
            return final_metrics

        except Exception as e:
            logger.error(f"Failed to train LSTM model: {e}")
            raise ModelError(f"LSTM training failed: {e}")

    def predict(self, X: np.ndarray) -> LSTMPrediction:
        """进行预测

        Args:
            X: 输入特征 (numpy array)

        Returns:
            预测结果
        """
        try:
            if not self.is_trained or self.model is None:
                raise ModelError("Model not trained yet")

            self.model.eval()

            # 如果输入是原始特征，需要转换为序列格式
            if X.ndim == 2:  # (samples, features)
                # 标准化特征
                X_scaled = self.scaler.transform(X)

                # 创建序列（使用最后sequence_length个样本）
                if len(X_scaled) >= self.config.sequence_length:
                    X_seq = X_scaled[-self.config.sequence_length :].reshape(
                        1, self.config.sequence_length, -1
                    )
                else:
                    # 如果数据不足，用零填充
                    padded = np.zeros((self.config.sequence_length, X_scaled.shape[1]))
                    padded[-len(X_scaled) :] = X_scaled
                    X_seq = padded.reshape(1, self.config.sequence_length, -1)
            else:  # 已经是序列格式
                X_seq = X

            # 转换为张量
            X_tensor = torch.FloatTensor(X_seq).to(self.device)

            with torch.no_grad():
                predictions = self.model(X_tensor)
                predictions_np = predictions.cpu().numpy().flatten()

            # 计算置信度（基于预测的标准差）
            confidence = 0.8  # 简化的置信度计算

            metrics = {
                "prediction_std": np.std(predictions_np),
                "prediction_mean": np.mean(predictions_np),
                "num_predictions": len(predictions_np),
            }

            return LSTMPrediction(
                predictions=predictions_np, confidence=confidence, model_metrics=metrics
            )

        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise ModelError(f"LSTM prediction failed: {e}")

    def save_model(self, model_id: str) -> bool:
        """保存模型到数据库

        Args:
            model_id: 模型ID

        Returns:
            是否保存成功
        """
        try:
            if self.model is None:
                logger.error("No model to save")
                return False

            # 保存模型参数
            model_state = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
                if self.optimizer
                else None,
                "scaler_params": {
                    "scale_": self.scaler.scale_.tolist()
                    if hasattr(self.scaler, "scale_")
                    else None,
                    "min_": self.scaler.min_.tolist()
                    if hasattr(self.scaler, "min_")
                    else None,
                    "data_min_": self.scaler.data_min_.tolist()
                    if hasattr(self.scaler, "data_min_")
                    else None,
                    "data_max_": self.scaler.data_max_.tolist()
                    if hasattr(self.scaler, "data_max_")
                    else None,
                },
                "config": self.config.__dict__,
            }

            success = self.db_manager.save_model_parameters(model_id, model_state)

            if success:
                self.model_id = model_id
                logger.info(f"LSTM model saved to database with ID: {model_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, model_id: str) -> bool:
        """从数据库加载模型

        Args:
            model_id: 模型ID

        Returns:
            是否加载成功
        """
        try:
            # 从数据库加载模型参数
            model_state = self.db_manager.load_model_parameters(model_id)

            if model_state is None:
                logger.error(f"No model found with ID: {model_id}")
                return False

            # 恢复配置
            if "config" in model_state:
                config_dict = model_state["config"]
                self.config = LSTMModelConfig(**config_dict)

            # 重建模型结构（需要知道输入维度）
            # 这里假设有保存的配置信息
            if "model_state_dict" in model_state:
                # 从state_dict推断模型结构
                state_dict = model_state["model_state_dict"]
                lstm_weight = list(state_dict.values())[0]  # 第一个LSTM权重
                input_size = lstm_weight.shape[1]  # 推断输入维度

                self.model = LSTMNet(
                    input_size=input_size,
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    dropout=self.config.dropout,
                ).to(self.device)

                self.model.load_state_dict(state_dict)

                # 恢复优化器
                self.optimizer = optim.Adam(
                    self.model.parameters(), lr=self.config.learning_rate
                )

                if (
                    "optimizer_state_dict" in model_state
                    and model_state["optimizer_state_dict"]
                ):
                    self.optimizer.load_state_dict(model_state["optimizer_state_dict"])

            # 恢复缩放器参数
            if "scaler_params" in model_state:
                scaler_params = model_state["scaler_params"]
                if scaler_params["scale_"] is not None:
                    self.scaler.scale_ = np.array(scaler_params["scale_"])
                    self.scaler.min_ = np.array(scaler_params["min_"])
                    self.scaler.data_min_ = np.array(scaler_params["data_min_"])
                    self.scaler.data_max_ = np.array(scaler_params["data_max_"])

            self.is_trained = True
            self.model_id = model_id

            logger.info(f"LSTM model loaded from database: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def set_model_id(self, model_id: str):
        """设置模型ID"""
        self.model_id = model_id


# 便捷函数
def create_lstm_model(
    input_dim: int,
    sequence_length: int = 60,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 100,
) -> LSTMModel:
    """创建LSTM模型的便捷函数

    Args:
        input_dim: 输入特征维度
        sequence_length: 序列长度
        hidden_size: 隐藏层大小
        num_layers: LSTM层数
        epochs: 训练轮数

    Returns:
        LSTM模型实例
    """
    config = LSTMModelConfig(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=epochs,
    )

    return LSTMModel(config)

    def set_model_id(self, model_id: str):
        """设置模型ID"""
        self.model_id = model_id
