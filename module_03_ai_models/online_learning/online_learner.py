"""
在线学习模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from common.logging_system import setup_logger
from common.exceptions import ModelError

logger = setup_logger("online_learner")

@dataclass
class OnlineLearningConfig:
    """在线学习配置"""
    learning_rate: float = 0.01
    buffer_size: int = 1000
    update_frequency: int = 100
    decay_rate: float = 0.95

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
        self.model_weights = np.random.normal(0, 0.1, 10)  # 简化的模型权重
        self.buffer = []
        self.update_count = 0
        self.is_initialized = False
    
    def add_sample(self, features: np.ndarray, target: float):
        """添加样本到缓冲区
        
        Args:
            features: 特征向量
            target: 目标值
        """
        try:
            self.buffer.append((features, target))
            
            # 限制缓冲区大小
            if len(self.buffer) > self.config.buffer_size:
                self.buffer.pop(0)
            
            self.update_count += 1
            
            # 检查是否需要更新模型
            if self.update_count % self.config.update_frequency == 0:
                self._update_model()
                
        except Exception as e:
            logger.error(f"Failed to add sample: {e}")
            raise ModelError(f"Sample addition failed: {e}")
    
    def _update_model(self):
        """更新模型权重"""
        try:
            if len(self.buffer) < 10:
                return
            
            # 简化的在线学习更新
            for features, target in self.buffer[-100:]:  # 使用最近的100个样本
                prediction = np.dot(features, self.model_weights)
                error = target - prediction
                
                # 梯度下降更新
                gradient = -2 * error * features
                self.model_weights -= self.config.learning_rate * gradient
            
            # 学习率衰减
            self.config.learning_rate *= self.config.decay_rate
            
            logger.info(f"Model updated, learning rate: {self.config.learning_rate:.6f}")
            
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
            # 简化的预测
            prediction = np.dot(features, self.model_weights)
            
            # 计算置信度（基于模型稳定性）
            confidence = min(1.0, len(self.buffer) / 100.0)
            
            # 学习指标
            learning_metrics = {
                'buffer_size': len(self.buffer),
                'update_count': self.update_count,
                'learning_rate': self.config.learning_rate,
                'model_norm': np.linalg.norm(self.model_weights)
            }
            
            return OnlineLearningResult(
                prediction=prediction,
                confidence=confidence,
                model_updated=self.update_count % self.config.update_frequency == 0,
                learning_metrics=learning_metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to make prediction: {e}")
            raise ModelError(f"Online prediction failed: {e}")
    
    def get_model_state(self) -> Dict[str, Any]:
        """获取模型状态
        
        Returns:
            模型状态字典
        """
        return {
            'weights': self.model_weights.tolist(),
            'buffer_size': len(self.buffer),
            'update_count': self.update_count,
            'learning_rate': self.config.learning_rate,
            'is_initialized': self.is_initialized
        }
    
    def set_model_state(self, state: Dict[str, Any]):
        """设置模型状态
        
        Args:
            state: 模型状态字典
        """
        try:
            self.model_weights = np.array(state.get('weights', self.model_weights))
            self.update_count = state.get('update_count', 0)
            self.config.learning_rate = state.get('learning_rate', self.config.learning_rate)
            self.is_initialized = state.get('is_initialized', False)
            
            logger.info("Model state restored")
            
        except Exception as e:
            logger.error(f"Failed to set model state: {e}")
            raise ModelError(f"Model state restoration failed: {e}")
