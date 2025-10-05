"""
堆叠集成（Stacking）实现

提供高级堆叠集成功能：
- 多层堆叠
- 交叉验证堆叠
- 自动特征选择
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.svm import SVR

from common.exceptions import ModelError
from common.logging_system import setup_logger

from ..storage_management.ai_model_database import get_ai_model_database_manager

logger = setup_logger("stacking")


@dataclass
class StackingConfig:
    """堆叠集成配置"""

    cv_folds: int = 5
    random_state: int = 42
    meta_learner: str = "linear"  # linear, ridge, lasso, rf, gbr
    use_base_features: bool = True
    feature_selection: bool = False
    selection_threshold: float = 0.01


@dataclass
class StackingResult:
    """堆叠预测结果"""

    predictions: np.ndarray
    confidence: float
    base_predictions: Dict[str, np.ndarray]
    meta_features: np.ndarray
    feature_importance: Dict[str, float]


class StackingEnsemble:
    """堆叠集成学习器"""

    def __init__(self, config: StackingConfig):
        self.config = config
        self.base_models = {}
        self.meta_learner = None
        self.cv_predictions = {}
        self.feature_selector = None
        self.db_manager = get_ai_model_database_manager()
        self.ensemble_id = f"stacking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.trained = False

    def add_base_model(self, name: str, model: Any):
        """添加基学习器"""
        try:
            self.base_models[name] = model
            logger.info(f"Added base model: {name}")

        except Exception as e:
            logger.error(f"Failed to add base model {name}: {e}")
            raise ModelError(f"Failed to add base model: {e}")

    def _get_meta_learner(self):
        """获取元学习器"""
        try:
            if self.config.meta_learner == "linear":
                return LinearRegression()
            elif self.config.meta_learner == "ridge":
                return Ridge(alpha=1.0, random_state=self.config.random_state)
            elif self.config.meta_learner == "lasso":
                return Lasso(alpha=0.1, random_state=self.config.random_state)
            elif self.config.meta_learner == "rf":
                return RandomForestRegressor(
                    n_estimators=100, random_state=self.config.random_state
                )
            elif self.config.meta_learner == "gbr":
                return GradientBoostingRegressor(
                    n_estimators=100, random_state=self.config.random_state
                )
            elif self.config.meta_learner == "svr":
                return SVR(kernel="rbf")
            else:
                raise ModelError(f"Unknown meta learner: {self.config.meta_learner}")

        except Exception as e:
            logger.error(f"Failed to create meta learner: {e}")
            raise ModelError(f"Meta learner creation failed: {e}")

    def _generate_cv_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """生成交叉验证预测（避免过拟合）"""
        try:
            kf = KFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )

            meta_features = []

            for name, model in self.base_models.items():
                try:
                    # 使用交叉验证生成预测
                    cv_pred = cross_val_predict(model, X, y, cv=kf)
                    self.cv_predictions[name] = cv_pred
                    meta_features.append(cv_pred)

                    logger.info(f"Generated CV predictions for {name}")

                except Exception as e:
                    logger.warning(f"Failed to generate CV predictions for {name}: {e}")
                    # 使用简单预测作为备选
                    model.fit(X, y)
                    pred = model.predict(X)
                    self.cv_predictions[name] = pred
                    meta_features.append(pred)

            if not meta_features:
                raise ModelError("No valid meta features generated")

            meta_features_array = np.column_stack(meta_features)

            # 如果使用原始特征
            if self.config.use_base_features:
                meta_features_array = np.column_stack([X, meta_features_array])

            return meta_features_array

        except Exception as e:
            logger.error(f"CV prediction generation failed: {e}")
            raise ModelError(f"CV prediction generation failed: {e}")

    def _select_features(self, meta_features: np.ndarray, y: np.ndarray):
        """特征选择"""
        try:
            if not self.config.feature_selection:
                return meta_features

            from sklearn.feature_selection import SelectKBest, f_regression

            # 选择最重要的特征
            k = max(1, int(meta_features.shape[1] * 0.8))  # 保留80%的特征
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            selected_features = self.feature_selector.fit_transform(meta_features, y)

            logger.info(
                f"Feature selection: {meta_features.shape[1]} -> {selected_features.shape[1]}"
            )
            return selected_features

        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return meta_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """训练堆叠集成"""
        try:
            if len(self.base_models) < 2:
                raise ModelError("At least 2 base models required for stacking")

            training_metrics = {}

            # 1. 生成元特征
            logger.info("Generating meta features using cross-validation...")
            meta_features = self._generate_cv_predictions(X, y)
            training_metrics["meta_features_shape"] = meta_features.shape

            # 2. 特征选择
            if self.config.feature_selection:
                meta_features = self._select_features(meta_features, y)
                training_metrics["selected_features_shape"] = meta_features.shape

            # 3. 训练元学习器
            logger.info("Training meta learner...")
            self.meta_learner = self._get_meta_learner()
            self.meta_learner.fit(meta_features, y)

            # 4. 评估元学习器性能
            meta_pred = self.meta_learner.predict(meta_features)
            meta_mse = mean_squared_error(y, meta_pred)
            meta_r2 = r2_score(y, meta_pred)

            training_metrics.update(
                {
                    "meta_mse": meta_mse,
                    "meta_r2": meta_r2,
                    "num_base_models": len(self.base_models),
                    "cv_folds": self.config.cv_folds,
                    "meta_learner": self.config.meta_learner,
                }
            )

            # 5. 训练最终的基模型（用于预测阶段）
            for name, model in self.base_models.items():
                try:
                    model.fit(X, y)
                    training_metrics[f"{name}_trained"] = True
                except Exception as e:
                    logger.warning(f"Final training failed for {name}: {e}")
                    training_metrics[f"{name}_trained"] = False

            self.trained = True

            logger.info(f"Stacking training completed: {training_metrics}")
            return training_metrics

        except Exception as e:
            logger.error(f"Stacking training failed: {e}")
            raise ModelError(f"Stacking training failed: {e}")

    def predict(self, X: np.ndarray) -> StackingResult:
        """堆叠预测"""
        try:
            if not self.trained or self.meta_learner is None:
                raise ModelError("Stacking ensemble not trained yet")

            # 1. 获取基模型预测
            base_predictions = {}
            meta_features_list = []

            for name, model in self.base_models.items():
                try:
                    pred = model.predict(X)
                    base_predictions[name] = pred
                    meta_features_list.append(pred)
                except Exception as e:
                    logger.warning(f"Base model {name} prediction failed: {e}")

            if not meta_features_list:
                raise ModelError("No valid base predictions")

            # 2. 构建元特征
            meta_features = np.column_stack(meta_features_list)

            if self.config.use_base_features:
                meta_features = np.column_stack([X, meta_features])

            # 3. 特征选择（如果启用）
            if self.feature_selector is not None:
                meta_features = self.feature_selector.transform(meta_features)

            # 4. 元学习器预测
            final_predictions = self.meta_learner.predict(meta_features)

            # 5. 计算置信度
            base_pred_array = np.array(list(base_predictions.values()))
            pred_std = np.std(base_pred_array, axis=0)
            confidence = 1.0 / (1.0 + np.mean(pred_std))

            # 6. 获取特征重要性
            feature_importance = self._get_feature_importance()

            return StackingResult(
                predictions=final_predictions,
                confidence=confidence,
                base_predictions=base_predictions,
                meta_features=meta_features,
                feature_importance=feature_importance,
            )

        except Exception as e:
            logger.error(f"Stacking prediction failed: {e}")
            raise ModelError(f"Stacking prediction failed: {e}")

    def _get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        try:
            if hasattr(self.meta_learner, "feature_importances_"):
                # 树模型
                importances = self.meta_learner.feature_importances_
            elif hasattr(self.meta_learner, "coef_"):
                # 线性模型
                importances = np.abs(self.meta_learner.coef_)
            else:
                # 无法获取重要性
                return {}

            # 分配重要性给基模型
            feature_names = list(self.base_models.keys())

            if self.config.use_base_features:
                # 前面是原始特征，后面是基模型预测
                num_base_features = len(self.base_models)
                if len(importances) >= num_base_features:
                    model_importances = importances[-num_base_features:]
                else:
                    model_importances = importances[: len(feature_names)]
            else:
                model_importances = importances[: len(feature_names)]

            # 归一化
            total_importance = model_importances.sum()
            if total_importance > 0:
                normalized_importances = model_importances / total_importance
            else:
                normalized_importances = np.ones(len(feature_names)) / len(
                    feature_names
                )

            return dict(zip(feature_names, normalized_importances))

        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}

    def get_cv_performance(self) -> Dict[str, float]:
        """获取交叉验证性能"""
        try:
            performance = {}

            for name, cv_pred in self.cv_predictions.items():
                if len(cv_pred) > 0:
                    # 这里需要原始目标值来计算性能，简化处理
                    performance[f"{name}_cv_std"] = np.std(cv_pred)
                    performance[f"{name}_cv_mean"] = np.mean(cv_pred)

            return performance

        except Exception as e:
            logger.error(f"Failed to get CV performance: {e}")
            return {}

    def save_stacking(self, filepath: str) -> bool:
        """保存堆叠模型"""
        try:
            import pickle

            stacking_data = {
                "config": self.config,
                "base_models": self.base_models,
                "meta_learner": self.meta_learner,
                "feature_selector": self.feature_selector,
                "cv_predictions": self.cv_predictions,
                "ensemble_id": self.ensemble_id,
                "trained": self.trained,
            }

            with open(filepath, "wb") as f:
                pickle.dump(stacking_data, f)

            logger.info(f"Stacking ensemble saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save stacking ensemble: {e}")
            return False


def create_stacking_ensemble(
    meta_learner: str = "linear",
    cv_folds: int = 5,
    use_base_features: bool = True,
    feature_selection: bool = False,
) -> StackingEnsemble:
    """创建堆叠集成的便捷函数"""
    config = StackingConfig(
        meta_learner=meta_learner,
        cv_folds=cv_folds,
        use_base_features=use_base_features,
        feature_selection=feature_selection,
    )
    return StackingEnsemble(config)
