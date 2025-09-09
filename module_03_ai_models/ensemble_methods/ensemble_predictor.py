"""
集成预测器模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from common.logging_system import setup_logger
from common.exceptions import ModelError

logger = setup_logger("ensemble_predictor")

@dataclass
class EnsembleConfig:
    """集成模型配置"""
    models: List[Dict[str, Any]]
    voting_strategy: str = "weighted"  # "weighted", "majority", "average"
    weights: Optional[List[float]] = None

@dataclass
class EnsemblePrediction:
    """集成预测结果"""
    predictions: np.ndarray
    confidence: float
    individual_predictions: Dict[str, np.ndarray]
    ensemble_metrics: Dict[str, float]

class EnsemblePredictor:
    """集成预测器类"""
    
    def __init__(self, config: EnsembleConfig):
        """初始化集成预测器
        
        Args:
            config: 集成配置
        """
        self.config = config
        self.models = {}
        self.is_trained = False
        
        # 初始化权重
        if self.config.weights is None:
            self.config.weights = [1.0 / len(self.config.models)] * len(self.config.models)
    
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """添加模型
        
        Args:
            name: 模型名称
            model: 模型实例
            weight: 模型权重
        """
        try:
            self.models[name] = model
            logger.info(f"Added model: {name} with weight: {weight}")
            
        except Exception as e:
            logger.error(f"Failed to add model {name}: {e}")
            raise ModelError(f"Model addition failed: {e}")
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """训练集成模型
        
        Args:
            X: 输入特征
            y: 目标值
            
        Returns:
            训练指标
        """
        try:
            logger.info(f"Training ensemble with {len(self.models)} models")
            
            training_metrics = {}
            
            # 训练每个模型
            for name, model in self.models.items():
                if hasattr(model, 'train'):
                    metrics = model.train(X, y)
                    training_metrics[f"{name}_train_loss"] = metrics.get('train_loss', 0.0)
                else:
                    logger.warning(f"Model {name} does not have train method")
            
            self.is_trained = True
            logger.info("Ensemble training completed")
            return training_metrics
            
        except Exception as e:
            logger.error(f"Failed to train ensemble: {e}")
            raise ModelError(f"Ensemble training failed: {e}")
    
    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """进行集成预测
        
        Args:
            X: 输入特征
            
        Returns:
            集成预测结果
        """
        try:
            if not self.is_trained:
                raise ModelError("Ensemble not trained yet")
            
            individual_predictions = {}
            predictions_list = []
            weights = []
            
            # 获取每个模型的预测
            for i, (name, model) in enumerate(self.models.items()):
                if hasattr(model, 'predict'):
                    pred_result = model.predict(X)
                    if hasattr(pred_result, 'predictions'):
                        pred = pred_result.predictions
                    else:
                        pred = pred_result
                    
                    individual_predictions[name] = pred
                    predictions_list.append(pred)
                    weights.append(self.config.weights[i])
                else:
                    logger.warning(f"Model {name} does not have predict method")
            
            if not predictions_list:
                raise ModelError("No valid predictions from models")
            
            # 集成预测
            if self.config.voting_strategy == "weighted":
                # 加权平均
                weights = np.array(weights)
                weights = weights / weights.sum()  # 归一化权重
                ensemble_pred = np.average(predictions_list, axis=0, weights=weights)
            elif self.config.voting_strategy == "average":
                # 简单平均
                ensemble_pred = np.mean(predictions_list, axis=0)
            elif self.config.voting_strategy == "majority":
                # 多数投票（适用于分类）
                ensemble_pred = np.round(np.mean(predictions_list, axis=0))
            else:
                raise ModelError(f"Unknown voting strategy: {self.config.voting_strategy}")
            
            # 计算置信度
            pred_std = np.std(predictions_list, axis=0)
            confidence = 1.0 / (1.0 + np.mean(pred_std))
            
            # 计算集成指标
            ensemble_metrics = {
                'prediction_std': np.std(ensemble_pred),
                'prediction_mean': np.mean(ensemble_pred),
                'model_agreement': 1.0 - np.mean(pred_std),
                'num_models': len(self.models)
            }
            
            return EnsemblePrediction(
                predictions=ensemble_pred,
                confidence=confidence,
                individual_predictions=individual_predictions,
                ensemble_metrics=ensemble_metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to make ensemble predictions: {e}")
            raise ModelError(f"Ensemble prediction failed: {e}")
    
    def get_model_importance(self) -> Dict[str, float]:
        """获取模型重要性
        
        Returns:
            模型重要性字典
        """
        try:
            importance = {}
            total_weight = sum(self.config.weights)
            
            for i, (name, _) in enumerate(self.models.items()):
                importance[name] = self.config.weights[i] / total_weight
            
            return importance
            
        except Exception as e:
            logger.error(f"Failed to get model importance: {e}")
            return {}
