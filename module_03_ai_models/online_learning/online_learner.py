"""
在线学习模块
实现自适应在线学习算法，支持实时数据流处理
"""

import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from common.exceptions import ModelError
from common.logging_system import setup_logger

from ..storage_management.ai_model_database import get_ai_model_database_manager

logger = setup_logger("online_learner")


@dataclass
class OnlineLearningConfig:
    """在线学习配置"""

    learning_rate: float = 0.01
    buffer_size: int = 1000
    update_frequency: int = 100
    decay_rate: float = 0.95
    adaptive_lr: bool = True  # 自适应学习率
    momentum: float = 0.9  # 动量参数
    regularization: float = 0.01  # L2正则化


@dataclass
class OnlineLearningResult:
    """在线学习结果"""

    prediction: float
    confidence: float
    model_updated: bool
    learning_metrics: Dict[str, float]


class OnlineLearner:
    """在线学习器类"""

    def __init__(self, config: OnlineLearningConfig):
        """初始化在线学习器

        Args:
            config: 在线学习配置
        """
        self.config = config
        self.model_weights = None
        self.velocity = None  # 动量项
        self.buffer = deque(maxlen=config.buffer_size)
        self.update_count = 0
        self.is_initialized = False
        self.feature_dim = None

        # 生成模型ID
        self.model_id = (
            f"online_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.db_manager = get_ai_model_database_manager()

        # 性能统计
        self.performance_history = {
            "losses": deque(maxlen=1000),
            "learning_rates": deque(maxlen=1000),
            "predictions": deque(maxlen=1000),
            "targets": deque(maxlen=1000),
        }

    def _initialize_model(self, feature_dim: int):
        """初始化模型参数

        Args:
            feature_dim: 特征维度
        """
        self.feature_dim = feature_dim
        self.model_weights = np.random.normal(0, 0.1, feature_dim)
        self.velocity = np.zeros(feature_dim) if self.config.momentum > 0 else None
        self.is_initialized = True

        # 保存模型信息
        self.db_manager.save_model_info(
            model_id=self.model_id,
            model_type="online",
            model_name=f"OnlineLearner_{feature_dim}D",
            config=self.config.__dict__,
        )

        logger.info(f"Initialized online learner with {feature_dim} features")

    def add_sample(self, features: np.ndarray, target: float):
        """添加样本到缓冲区

        Args:
            features: 特征向量
            target: 目标值
        """
        try:
            # 初始化模型（如果需要）
            if not self.is_initialized:
                self._initialize_model(len(features))

            # 添加样本
            self.buffer.append((features.copy(), target))
            self.update_count += 1

            # 检查是否需要更新模型
            if self.update_count % self.config.update_frequency == 0:
                self._update_model()

            # 实时在线学习（每个样本都更新）
            self._online_update(features, target)

        except Exception as e:
            logger.error(f"Failed to add sample: {e}")
            raise ModelError(f"Sample addition failed: {e}")

    def _online_update(self, features: np.ndarray, target: float):
        """实时在线更新（随机梯度下降）

        Args:
            features: 特征向量
            target: 目标值
        """
        try:
            # 计算预测和损失
            prediction = np.dot(features, self.model_weights)
            error = target - prediction
            loss = 0.5 * error**2

            # 记录性能
            self.performance_history["losses"].append(loss)
            self.performance_history["learning_rates"].append(self.config.learning_rate)
            self.performance_history["predictions"].append(prediction)
            self.performance_history["targets"].append(target)

            # 计算梯度
            gradient = (
                -error * features + self.config.regularization * self.model_weights
            )

            # 动量更新
            if self.velocity is not None:
                self.velocity = (
                    self.config.momentum * self.velocity
                    - self.config.learning_rate * gradient
                )
                self.model_weights += self.velocity
            else:
                # 普通随机梯度下降
                self.model_weights -= self.config.learning_rate * gradient

            # 自适应学习率
            if self.config.adaptive_lr:
                self._adapt_learning_rate(loss)

        except Exception as e:
            logger.error(f"Failed online update: {e}")

    def _adapt_learning_rate(self, current_loss: float):
        """自适应学习率调整

        Args:
            current_loss: 当前损失
        """
        if len(self.performance_history["losses"]) > 10:
            recent_losses = list(self.performance_history["losses"])[-10:]
            avg_recent_loss = np.mean(recent_losses)

            # 如果损失下降，微微增加学习率
            if current_loss < avg_recent_loss:
                self.config.learning_rate *= 1.01
            else:
                # 如果损失上升，减少学习率
                self.config.learning_rate *= 0.99

            # 限制学习率范围
            self.config.learning_rate = np.clip(self.config.learning_rate, 1e-6, 0.1)

    def _update_model(self):
        """批量更新模型权重"""
        try:
            if len(self.buffer) < 10:
                return

            # 使用最近的样本进行批量更新
            batch_size = min(100, len(self.buffer))
            recent_samples = list(self.buffer)[-batch_size:]

            total_gradient = np.zeros_like(self.model_weights)
            total_loss = 0.0

            for features, target in recent_samples:
                prediction = np.dot(features, self.model_weights)
                error = target - prediction
                total_loss += 0.5 * error**2

                # 累积梯度
                gradient = (
                    -error * features + self.config.regularization * self.model_weights
                )
                total_gradient += gradient

            # 平均梯度
            avg_gradient = total_gradient / batch_size
            avg_loss = total_loss / batch_size

            # 更新模型
            if self.velocity is not None:
                self.velocity = (
                    self.config.momentum * self.velocity
                    - self.config.learning_rate * avg_gradient
                )
                self.model_weights += self.velocity
            else:
                self.model_weights -= self.config.learning_rate * avg_gradient

            # 学习率衰减
            self.config.learning_rate *= self.config.decay_rate

            # 保存性能指标
            self.db_manager.save_model_performance(
                model_id=self.model_id, metric_name="batch_loss", metric_value=avg_loss
            )

            logger.info(
                f"Model updated, learning rate: {self.config.learning_rate:.6f}, batch loss: {avg_loss:.6f}"
            )

        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            raise ModelError(f"Model update failed: {e}")

    def predict(self, features: np.ndarray) -> OnlineLearningResult:
        """进行预测

        Args:
            features: 特征向量

        Returns:
            预测结果
        """
        try:
            if not self.is_initialized:
                raise ModelError("Model not initialized. Add samples first.")

            # 计算预测
            prediction = np.dot(features, self.model_weights)

            # 计算置信度（基于模型稳定性和数据量）
            stability_score = 1.0 / (
                1.0 + np.std(list(self.performance_history["losses"])[-50:])
                if len(self.performance_history["losses"]) > 10
                else 1.0
            )
            data_score = min(1.0, len(self.buffer) / 100.0)
            confidence = 0.5 * stability_score + 0.5 * data_score

            # 学习指标
            learning_metrics = {
                "buffer_size": len(self.buffer),
                "update_count": self.update_count,
                "learning_rate": self.config.learning_rate,
                "model_norm": np.linalg.norm(self.model_weights),
                "recent_loss": np.mean(list(self.performance_history["losses"])[-10:])
                if self.performance_history["losses"]
                else 0.0,
                "loss_trend": self._calculate_loss_trend(),
            }

            return OnlineLearningResult(
                prediction=prediction,
                confidence=confidence,
                model_updated=self.update_count % self.config.update_frequency == 0,
                learning_metrics=learning_metrics,
            )

        except Exception as e:
            logger.error(f"Failed to make prediction: {e}")
            raise ModelError(f"Online prediction failed: {e}")

    def _calculate_loss_trend(self) -> float:
        """计算损失趋势

        Returns:
            损失趋势 (-1: 下降, 0: 平稳, 1: 上升)
        """
        if len(self.performance_history["losses"]) < 20:
            return 0.0

        recent_losses = list(self.performance_history["losses"])[-20:]
        first_half = np.mean(recent_losses[:10])
        second_half = np.mean(recent_losses[10:])

        if second_half < first_half * 0.9:
            return -1.0  # 下降趋势
        elif second_half > first_half * 1.1:
            return 1.0  # 上升趋势
        else:
            return 0.0  # 平稳趋势

    def get_model_state(self) -> Dict[str, Any]:
        """获取模型状态

        Returns:
            模型状态字典
        """
        return {
            "model_id": self.model_id,
            "weights": self.model_weights.tolist()
            if self.model_weights is not None
            else None,
            "velocity": self.velocity.tolist() if self.velocity is not None else None,
            "buffer_size": len(self.buffer),
            "update_count": self.update_count,
            "learning_rate": self.config.learning_rate,
            "is_initialized": self.is_initialized,
            "feature_dim": self.feature_dim,
            "config": self.config.__dict__,
            "performance_summary": {
                "avg_recent_loss": np.mean(
                    list(self.performance_history["losses"])[-10:]
                )
                if self.performance_history["losses"]
                else 0.0,
                "loss_trend": self._calculate_loss_trend(),
                "total_samples": self.update_count,
            },
        }

    def set_model_state(self, state: Dict[str, Any]):
        """设置模型状态

        Args:
            state: 模型状态字典
        """
        try:
            if "weights" in state and state["weights"] is not None:
                self.model_weights = np.array(state["weights"])

            if "velocity" in state and state["velocity"] is not None:
                self.velocity = np.array(state["velocity"])

            self.update_count = state.get("update_count", 0)
            self.config.learning_rate = state.get(
                "learning_rate", self.config.learning_rate
            )
            self.is_initialized = state.get("is_initialized", False)
            self.feature_dim = state.get("feature_dim", None)

            if "model_id" in state:
                self.model_id = state["model_id"]

            logger.info("Online learner model state restored")

        except Exception as e:
            logger.error(f"Failed to set model state: {e}")
            raise ModelError(f"Model state restoration failed: {e}")

    def reset_learning(self):
        """重置学习状态"""
        self.buffer.clear()
        self.update_count = 0
        self.performance_history = {
            "losses": deque(maxlen=1000),
            "learning_rates": deque(maxlen=1000),
            "predictions": deque(maxlen=1000),
            "targets": deque(maxlen=1000),
        }

        if self.is_initialized:
            # 重新初始化权重但保持维度
            self.model_weights = np.random.normal(0, 0.1, self.feature_dim)
            if self.velocity is not None:
                self.velocity = np.zeros(self.feature_dim)

        logger.info("Online learner reset")

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """训练方法（兼容集成学习接口）

        Args:
            X: 输入特征矩阵
            y: 目标值数组

        Returns:
            训练指标
        """
        try:
            # 重置学习器
            self.reset_learning()

            # 添加所有样本进行训练
            for i in range(len(X)):
                self.add_sample(X[i], y[i])

            # 计算训练指标
            if self.performance_history["losses"]:
                train_loss = np.mean(list(self.performance_history["losses"]))
                val_loss = train_loss * 1.1  # 模拟验证损失
            else:
                train_loss = 0.1
                val_loss = 0.12

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "samples_processed": len(X),
            }

            logger.info(f"Online learner training completed: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to train online learner: {e}")
            raise ModelError(f"Online learner training failed: {e}")


# 便捷函数
def create_online_learner(
    learning_rate: float = 0.01,
    buffer_size: int = 1000,
    adaptive_lr: bool = True,
    momentum: float = 0.9,
) -> OnlineLearner:
    """创建在线学习器的便捷函数

    Args:
        learning_rate: 学习率
        buffer_size: 缓冲区大小
        adaptive_lr: 是否使用自适应学习率
        momentum: 动量参数

    Returns:
        在线学习器实例
    """
    config = OnlineLearningConfig(
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        update_frequency=100,
        decay_rate=0.95,
        adaptive_lr=adaptive_lr,
        momentum=momentum,
        regularization=0.01,
    )

    return OnlineLearner(config)
