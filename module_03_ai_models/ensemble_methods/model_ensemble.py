"""
模型集成器 - 提供多种集成学习策略

支持各种机器学习模型的集成，包括：
- 投票集成（Voting）
- 加权集成（Weighted）
- 堆叠集成（Stacking）
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from common.exceptions import ModelError
from common.logging_system import setup_logger

from ..storage_management.ai_model_database import get_ai_model_database_manager

logger = setup_logger("model_ensemble")


@dataclass
class ModelEnsembleConfig:
    """模型集成配置"""

    ensemble_method: str = "voting"  # voting, weighted, stacking
    cross_validation_folds: int = 5
    meta_learner: str = "linear"  # linear, rf
    weight_optimization: bool = True
    performance_threshold: float = 0.5


@dataclass
class EnsembleResult:
    """集成结果"""

    predictions: np.ndarray
    confidence: float
    model_weights: Dict[str, float]
    performance_metrics: Dict[str, float]
    ensemble_score: float


class ModelEnsemble:
    """高级模型集成器"""

    def __init__(self, config: ModelEnsembleConfig):
        self.config = config
        self.models = {}
        self.model_weights = {}
        self.meta_learner = None
        self.db_manager = get_ai_model_database_manager()
        self.ensemble_id = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.trained = False

    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """添加模型到集成中"""
        try:
            self.models[name] = model
            self.model_weights[name] = weight
            logger.info(f"Added model {name} to ensemble with weight {weight}")

        except Exception as e:
            logger.error(f"Failed to add model {name}: {e}")
            raise ModelError(f"Failed to add model: {e}")

    def _evaluate_model_performance(
        self, model: Any, X: np.ndarray, y: np.ndarray
    ) -> float:
        """评估单个模型性能"""
        try:
            if hasattr(model, "predict"):
                # 使用交叉验证评估
                if hasattr(model, "fit"):
                    scores = cross_val_score(
                        model,
                        X,
                        y,
                        cv=self.config.cross_validation_folds,
                        scoring="neg_mean_squared_error",
                    )
                    return -scores.mean()
                else:
                    # 对于已训练模型，直接预测
                    predictions = model.predict(X)
                    return mean_squared_error(y, predictions)
            else:
                logger.warning("Model does not have predict method")
                return float("inf")

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return float("inf")

    def _optimize_weights(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """优化模型权重"""
        try:
            if not self.config.weight_optimization:
                # 使用均等权重
                equal_weight = 1.0 / len(self.models)
                return {name: equal_weight for name in self.models.keys()}

            # 基于性能优化权重
            model_scores = {}
            for name, model in self.models.items():
                score = self._evaluate_model_performance(model, X, y)
                model_scores[name] = 1.0 / (1.0 + score)  # 转换为正向分数

            # 归一化权重
            total_score = sum(model_scores.values())
            if total_score > 0:
                optimized_weights = {
                    name: score / total_score for name, score in model_scores.items()
                }
            else:
                # 回退到均等权重
                equal_weight = 1.0 / len(self.models)
                optimized_weights = {name: equal_weight for name in self.models.keys()}

            logger.info(f"Optimized weights: {optimized_weights}")
            return optimized_weights

        except Exception as e:
            logger.error(f"Weight optimization failed: {e}")
            # 回退到均等权重
            equal_weight = 1.0 / len(self.models)
            return {name: equal_weight for name in self.models.keys()}

    def _train_meta_learner(self, meta_features: np.ndarray, y: np.ndarray):
        """训练元学习器（用于Stacking）"""
        try:
            if self.config.meta_learner == "linear":
                self.meta_learner = LinearRegression()
            elif self.config.meta_learner == "rf":
                self.meta_learner = RandomForestRegressor(
                    n_estimators=100, random_state=42
                )
            else:
                raise ModelError(f"Unknown meta learner: {self.config.meta_learner}")

            self.meta_learner.fit(meta_features, y)
            logger.info(
                f"Meta learner ({self.config.meta_learner}) trained successfully"
            )

        except Exception as e:
            logger.error(f"Meta learner training failed: {e}")
            raise ModelError(f"Meta learner training failed: {e}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """训练集成模型"""
        try:
            if len(self.models) < 2:
                raise ModelError("At least 2 models required for ensemble")

            training_metrics = {}

            # 优化权重
            self.model_weights = self._optimize_weights(X, y)

            if self.config.ensemble_method == "stacking":
                # 生成元特征
                meta_features = []
                for name, model in self.models.items():
                    if hasattr(model, "predict"):
                        try:
                            if hasattr(model, "fit"):
                                model.fit(X, y)

                            predictions = model.predict(X)
                            meta_features.append(predictions)
                            training_metrics[f"{name}_trained"] = True

                        except Exception as e:
                            logger.warning(
                                f"Model {name} training/prediction failed: {e}"
                            )
                            training_metrics[f"{name}_trained"] = False

                if meta_features:
                    meta_features = np.column_stack(meta_features)
                    self._train_meta_learner(meta_features, y)
                    training_metrics["meta_learner_trained"] = True
                else:
                    raise ModelError("No valid meta features generated")

            else:
                # 简单集成（voting/weighted）
                for name, model in self.models.items():
                    if hasattr(model, "fit"):
                        try:
                            model.fit(X, y)
                            training_metrics[f"{name}_trained"] = True
                        except Exception as e:
                            logger.warning(f"Model {name} training failed: {e}")
                            training_metrics[f"{name}_trained"] = False

            self.trained = True
            training_metrics["ensemble_method"] = self.config.ensemble_method
            training_metrics["num_models"] = len(self.models)

            logger.info(f"Ensemble training completed: {training_metrics}")
            return training_metrics

        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            raise ModelError(f"Ensemble training failed: {e}")

    def predict(self, X: np.ndarray) -> EnsembleResult:
        """集成预测"""
        try:
            if not self.trained:
                raise ModelError("Ensemble not trained yet")

            predictions_dict = {}
            predictions_list = []

            # 获取各模型预测
            for name, model in self.models.items():
                if hasattr(model, "predict"):
                    try:
                        pred = model.predict(X)
                        predictions_dict[name] = pred
                        predictions_list.append(pred)
                    except Exception as e:
                        logger.warning(f"Model {name} prediction failed: {e}")

            if not predictions_list:
                raise ModelError("No valid predictions from models")

            # 根据集成方法计算最终预测
            if self.config.ensemble_method == "voting":
                # 简单平均
                ensemble_pred = np.mean(predictions_list, axis=0)

            elif self.config.ensemble_method == "weighted":
                # 加权平均
                weights = np.array(
                    [
                        self.model_weights.get(name, 1.0)
                        for name in predictions_dict.keys()
                    ]
                )
                weights = weights / weights.sum()
                ensemble_pred = np.average(predictions_list, axis=0, weights=weights)

            elif self.config.ensemble_method == "stacking":
                # 使用元学习器
                if self.meta_learner is None:
                    raise ModelError("Meta learner not trained")

                meta_features = np.column_stack(predictions_list)
                ensemble_pred = self.meta_learner.predict(meta_features)

            else:
                raise ModelError(
                    f"Unknown ensemble method: {self.config.ensemble_method}"
                )

            # 计算置信度和性能指标
            pred_std = np.std(predictions_list, axis=0)
            confidence = 1.0 / (1.0 + np.mean(pred_std))

            performance_metrics = {
                "prediction_variance": np.var(ensemble_pred),
                "prediction_std": np.std(ensemble_pred),
                "model_agreement": 1.0 - np.mean(pred_std),
                "num_models": len(predictions_list),
            }

            ensemble_score = confidence * performance_metrics["model_agreement"]

            return EnsembleResult(
                predictions=ensemble_pred,
                confidence=confidence,
                model_weights=self.model_weights,
                performance_metrics=performance_metrics,
                ensemble_score=ensemble_score,
            )

        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise ModelError(f"Ensemble prediction failed: {e}")

    def get_model_importance(self) -> Dict[str, float]:
        """获取模型重要性"""
        try:
            if self.config.ensemble_method == "stacking" and self.meta_learner:
                if hasattr(self.meta_learner, "coef_"):
                    # 线性模型的系数
                    coefficients = np.abs(self.meta_learner.coef_)
                    total = coefficients.sum()

                    importance = {}
                    for i, name in enumerate(self.models.keys()):
                        if i < len(coefficients):
                            importance[name] = (
                                coefficients[i] / total if total > 0 else 0
                            )
                        else:
                            importance[name] = 0

                    return importance

                elif hasattr(self.meta_learner, "feature_importances_"):
                    # 树模型的特征重要性
                    importances = self.meta_learner.feature_importances_

                    importance = {}
                    for i, name in enumerate(self.models.keys()):
                        if i < len(importances):
                            importance[name] = importances[i]
                        else:
                            importance[name] = 0

                    return importance

            # 回退到权重
            return self.model_weights.copy()

        except Exception as e:
            logger.error(f"Failed to get model importance: {e}")
            return self.model_weights.copy()

    def save_ensemble(self, filepath: str) -> bool:
        """保存集成模型"""
        try:
            import pickle

            ensemble_data = {
                "config": self.config,
                "model_weights": self.model_weights,
                "meta_learner": self.meta_learner,
                "ensemble_id": self.ensemble_id,
                "trained": self.trained,
            }

            with open(filepath, "wb") as f:
                pickle.dump(ensemble_data, f)

            logger.info(f"Ensemble saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save ensemble: {e}")
            return False


def create_model_ensemble(
    ensemble_method: str = "weighted",
    weight_optimization: bool = True,
    meta_learner: str = "linear",
) -> ModelEnsemble:
    """创建模型集成器的便捷函数"""
    config = ModelEnsembleConfig(
        ensemble_method=ensemble_method,
        weight_optimization=weight_optimization,
        meta_learner=meta_learner,
    )
    return ModelEnsemble(config)
