"""
自适应模型 - 提供概念漂移检测和模型自适应能力

支持：
- 概念漂移检测
- 模型参数自适应调整
- 性能监控和报警
"""

import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.exceptions import ModelError
from common.logging_system import setup_logger

from ..storage_management.ai_model_database import get_ai_model_database_manager

logger = setup_logger("adaptive_model")


@dataclass
class AdaptiveConfig:
    """自适应配置"""

    drift_detection_method: str = "ddm"  # ddm, eddm, adwin
    drift_threshold: float = 0.1
    warning_threshold: float = 0.05
    adaptation_rate: float = 0.1
    performance_window: int = 100
    min_samples_for_adaptation: int = 50
    adaptation_strategy: str = "incremental"  # incremental, replacement


@dataclass
class DriftDetectionResult:
    """概念漂移检测结果"""

    drift_detected: bool
    warning_detected: bool
    drift_score: float
    drift_type: str
    detection_time: str


class DriftDetector:
    """概念漂移检测器"""

    def __init__(self, method: str = "ddm", threshold: float = 0.1):
        self.method = method
        self.threshold = threshold
        self.error_rates = deque(maxlen=1000)
        self.drift_history = []

        # DDM参数
        self.ddm_min = float("inf")
        self.ddm_min_std = float("inf")

        # EDDM参数
        self.eddm_distances = deque(maxlen=100)
        self.eddm_mean = 0
        self.eddm_std = 0

    def add_error(self, error: float) -> DriftDetectionResult:
        """添加错误率并检测漂移"""
        try:
            self.error_rates.append(error)

            if self.method == "ddm":
                return self._detect_ddm()
            elif self.method == "eddm":
                return self._detect_eddm()
            elif self.method == "adwin":
                return self._detect_adwin()
            else:
                raise ModelError(f"Unknown drift detection method: {self.method}")

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                warning_detected=False,
                drift_score=0.0,
                drift_type="unknown",
                detection_time=datetime.now().isoformat(),
            )

    def _detect_ddm(self) -> DriftDetectionResult:
        """DDM (Drift Detection Method) 检测"""
        try:
            if len(self.error_rates) < 30:
                return DriftDetectionResult(
                    drift_detected=False,
                    warning_detected=False,
                    drift_score=0.0,
                    drift_type="ddm",
                    detection_time=datetime.now().isoformat(),
                )

            # 计算错误率和标准差
            mean_error = np.mean(list(self.error_rates))
            std_error = np.std(list(self.error_rates))

            # 更新最小值
            current_value = mean_error + 2 * std_error
            if current_value < self.ddm_min:
                self.ddm_min = current_value
                self.ddm_min_std = std_error

            # 检测警告和漂移
            warning_threshold = self.ddm_min + 2 * self.ddm_min_std
            drift_threshold = self.ddm_min + 3 * self.ddm_min_std

            drift_detected = current_value > drift_threshold
            warning_detected = current_value > warning_threshold

            drift_score = (
                (current_value - self.ddm_min) / (drift_threshold - self.ddm_min)
                if drift_threshold > self.ddm_min
                else 0
            )

            return DriftDetectionResult(
                drift_detected=drift_detected,
                warning_detected=warning_detected,
                drift_score=min(drift_score, 1.0),
                drift_type="ddm",
                detection_time=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"DDM detection failed: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                warning_detected=False,
                drift_score=0.0,
                drift_type="ddm",
                detection_time=datetime.now().isoformat(),
            )

    def _detect_eddm(self) -> DriftDetectionResult:
        """EDDM (Early Drift Detection Method) 检测"""
        try:
            if len(self.error_rates) < 2:
                return DriftDetectionResult(
                    drift_detected=False,
                    warning_detected=False,
                    drift_score=0.0,
                    drift_type="eddm",
                    detection_time=datetime.now().isoformat(),
                )

            # 计算错误间距离
            errors = list(self.error_rates)
            last_error_idx = len(errors) - 1
            for i in range(len(errors) - 2, -1, -1):
                if errors[i] > 0:
                    distance = last_error_idx - i
                    self.eddm_distances.append(distance)
                    break

            if len(self.eddm_distances) < 30:
                return DriftDetectionResult(
                    drift_detected=False,
                    warning_detected=False,
                    drift_score=0.0,
                    drift_type="eddm",
                    detection_time=datetime.now().isoformat(),
                )

            # 更新距离统计
            distances = list(self.eddm_distances)
            self.eddm_mean = np.mean(distances)
            self.eddm_std = np.std(distances)

            # 检测漂移
            current_distance = distances[-1]
            threshold = self.eddm_mean + 2 * self.eddm_std

            drift_detected = current_distance < threshold * 0.5
            warning_detected = current_distance < threshold * 0.7

            drift_score = 1.0 - (current_distance / threshold) if threshold > 0 else 0

            return DriftDetectionResult(
                drift_detected=drift_detected,
                warning_detected=warning_detected,
                drift_score=max(0, min(drift_score, 1.0)),
                drift_type="eddm",
                detection_time=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"EDDM detection failed: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                warning_detected=False,
                drift_score=0.0,
                drift_type="eddm",
                detection_time=datetime.now().isoformat(),
            )

    def _detect_adwin(self) -> DriftDetectionResult:
        """ADWIN (Adaptive Windowing) 检测"""
        try:
            if len(self.error_rates) < 50:
                return DriftDetectionResult(
                    drift_detected=False,
                    warning_detected=False,
                    drift_score=0.0,
                    drift_type="adwin",
                    detection_time=datetime.now().isoformat(),
                )

            # 简化的ADWIN实现
            window_size = min(100, len(self.error_rates))
            recent_errors = list(self.error_rates)[-window_size:]

            # 分割窗口
            split_point = window_size // 2
            left_window = recent_errors[:split_point]
            right_window = recent_errors[split_point:]

            # 计算均值差异
            left_mean = np.mean(left_window)
            right_mean = np.mean(right_window)

            mean_diff = abs(left_mean - right_mean)

            # 计算统计显著性
            combined_std = np.sqrt((np.var(left_window) + np.var(right_window)) / 2)
            significance = mean_diff / (combined_std + 1e-8)

            drift_detected = significance > 2.0
            warning_detected = significance > 1.5

            drift_score = min(significance / 3.0, 1.0)

            return DriftDetectionResult(
                drift_detected=drift_detected,
                warning_detected=warning_detected,
                drift_score=drift_score,
                drift_type="adwin",
                detection_time=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"ADWIN detection failed: {e}")
            return DriftDetectionResult(
                drift_detected=False,
                warning_detected=False,
                drift_score=0.0,
                drift_type="adwin",
                detection_time=datetime.now().isoformat(),
            )


class AdaptiveModel:
    """自适应模型"""

    def __init__(self, base_model: Any, config: AdaptiveConfig):
        self.base_model = base_model
        self.config = config
        self.drift_detector = DriftDetector(
            method=config.drift_detection_method, threshold=config.drift_threshold
        )

        self.performance_history = deque(maxlen=config.performance_window)
        self.model_versions = []
        self.current_version = 0
        self.adaptation_count = 0

        self.db_manager = get_ai_model_database_manager()
        self.model_id = f"adaptive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def update(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """更新模型并检测概念漂移"""
        try:
            # 进行预测
            if hasattr(self.base_model, "predict"):
                predictions = self.base_model.predict(X)

                # 计算错误率
                if len(predictions) == len(y):
                    errors = np.abs(predictions - y)
                    mean_error = np.mean(errors)
                    self.performance_history.append(mean_error)

                    # 检测概念漂移
                    drift_result = self.drift_detector.add_error(mean_error)

                    update_result = {
                        "drift_detected": drift_result.drift_detected,
                        "warning_detected": drift_result.warning_detected,
                        "drift_score": drift_result.drift_score,
                        "current_error": mean_error,
                        "adaptation_performed": False,
                    }

                    # 如果检测到漂移，执行适应
                    if drift_result.drift_detected:
                        adaptation_result = self._perform_adaptation(X, y)
                        update_result.update(adaptation_result)
                        update_result["adaptation_performed"] = True

                        logger.warning(
                            f"Concept drift detected, adaptation performed: {adaptation_result}"
                        )

                    elif drift_result.warning_detected:
                        logger.info("Drift warning detected, monitoring closely")

                    return update_result

            return {"error": "Model update failed"}

        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return {"error": str(e)}

    def _perform_adaptation(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """执行模型适应"""
        try:
            if len(X) < self.config.min_samples_for_adaptation:
                return {"adaptation_skipped": "Insufficient samples"}

            adaptation_result = {}

            if self.config.adaptation_strategy == "incremental":
                # 增量学习
                if hasattr(self.base_model, "partial_fit"):
                    self.base_model.partial_fit(X, y)
                    adaptation_result["method"] = "partial_fit"
                elif hasattr(self.base_model, "fit"):
                    # 使用recent data重新训练
                    recent_samples = min(len(X), 200)
                    self.base_model.fit(X[-recent_samples:], y[-recent_samples:])
                    adaptation_result["method"] = "incremental_retrain"

            elif self.config.adaptation_strategy == "replacement":
                # 完全重新训练
                if hasattr(self.base_model, "fit"):
                    self.base_model.fit(X, y)
                    adaptation_result["method"] = "full_retrain"

            # 保存模型版本
            self.model_versions.append(
                {
                    "version": self.current_version,
                    "timestamp": datetime.now().isoformat(),
                    "adaptation_method": adaptation_result.get("method", "unknown"),
                    "sample_size": len(X),
                }
            )

            self.current_version += 1
            self.adaptation_count += 1

            adaptation_result.update(
                {
                    "adaptation_count": self.adaptation_count,
                    "current_version": self.current_version,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            return adaptation_result

        except Exception as e:
            logger.error(f"Model adaptation failed: {e}")
            return {"adaptation_error": str(e)}

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """预测并返回元信息"""
        try:
            if hasattr(self.base_model, "predict"):
                predictions = self.base_model.predict(X)

                meta_info = {
                    "model_version": self.current_version,
                    "adaptation_count": self.adaptation_count,
                    "recent_performance": np.mean(list(self.performance_history))
                    if self.performance_history
                    else None,
                    "drift_detection_method": self.config.drift_detection_method,
                    "prediction_timestamp": datetime.now().isoformat(),
                }

                return predictions, meta_info
            else:
                raise ModelError("Base model does not support prediction")

        except Exception as e:
            logger.error(f"Adaptive prediction failed: {e}")
            raise ModelError(f"Prediction failed: {e}")

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """获取适应历史"""
        return self.model_versions.copy()

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        try:
            if not self.performance_history:
                return {}

            performance_data = list(self.performance_history)

            return {
                "current_performance": performance_data[-1],
                "average_performance": np.mean(performance_data),
                "performance_std": np.std(performance_data),
                "performance_trend": "improving"
                if len(performance_data) > 1
                and performance_data[-1] < performance_data[0]
                else "degrading",
                "adaptation_count": self.adaptation_count,
                "model_version": self.current_version,
                "window_size": len(performance_data),
            }

        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}


def create_adaptive_model(
    base_model: Any,
    drift_detection_method: str = "ddm",
    adaptation_strategy: str = "incremental",
    performance_window: int = 100,
) -> AdaptiveModel:
    """创建自适应模型的便捷函数"""
    config = AdaptiveConfig(
        drift_detection_method=drift_detection_method,
        adaptation_strategy=adaptation_strategy,
        performance_window=performance_window,
    )
    return AdaptiveModel(base_model, config)
