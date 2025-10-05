"""
在线更新器 - 提供实时模型更新功能

支持：
- 实时数据流处理
- 批量和增量更新
- 性能监控
- 自动重训练触发
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.exceptions import ModelError
from common.logging_system import setup_logger

from ..storage_management.ai_model_database import get_ai_model_database_manager

logger = setup_logger("online_updater")


@dataclass
class UpdateConfig:
    """更新配置"""

    update_frequency: int = 10  # 每多少个样本更新一次
    batch_size: int = 32
    performance_threshold: float = 0.1
    retrain_threshold: float = 0.2
    max_buffer_size: int = 1000
    auto_save: bool = True
    save_frequency: int = 100


@dataclass
class UpdateResult:
    """更新结果"""

    updated: bool
    performance_change: float
    samples_processed: int
    update_time: float
    retrain_triggered: bool
    timestamp: str


class OnlineUpdater:
    """在线更新器"""

    def __init__(self, model: Any, config: UpdateConfig):
        self.model = model
        self.config = config
        self.db_manager = get_ai_model_database_manager()

        # 数据缓冲区
        self.feature_buffer = deque(maxlen=config.max_buffer_size)
        self.target_buffer = deque(maxlen=config.max_buffer_size)

        # 性能监控
        self.performance_history = deque(maxlen=1000)
        self.last_performance = None

        # 更新统计
        self.update_count = 0
        self.samples_processed = 0
        self.retrain_count = 0

        # 线程控制
        self.update_thread = None
        self.stop_updating = False
        self.update_lock = threading.Lock()

        self.updater_id = f"updater_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def add_sample(self, features: np.ndarray, target: float):
        """添加新样本到缓冲区"""
        try:
            with self.update_lock:
                self.feature_buffer.append(features)
                self.target_buffer.append(target)
                self.samples_processed += 1

            # 检查是否需要更新
            if len(self.feature_buffer) >= self.config.update_frequency:
                self._trigger_update()

        except Exception as e:
            logger.error(f"Failed to add sample: {e}")

    def add_batch(self, features_batch: np.ndarray, targets_batch: np.ndarray):
        """批量添加样本"""
        try:
            with self.update_lock:
                for features, target in zip(features_batch, targets_batch):
                    self.feature_buffer.append(features)
                    self.target_buffer.append(target)

                self.samples_processed += len(features_batch)

            # 检查是否需要更新
            if len(self.feature_buffer) >= self.config.update_frequency:
                self._trigger_update()

        except Exception as e:
            logger.error(f"Failed to add batch: {e}")

    def _trigger_update(self) -> UpdateResult:
        """触发模型更新"""
        try:
            start_time = time.time()

            with self.update_lock:
                # 获取更新数据
                if len(self.feature_buffer) == 0:
                    return UpdateResult(
                        updated=False,
                        performance_change=0.0,
                        samples_processed=0,
                        update_time=0.0,
                        retrain_triggered=False,
                        timestamp=datetime.now().isoformat(),
                    )

                # 准备更新数据
                update_features = np.array(list(self.feature_buffer))
                update_targets = np.array(list(self.target_buffer))

                # 执行更新
                update_result = self._perform_update(update_features, update_targets)

                # 清空缓冲区
                self.feature_buffer.clear()
                self.target_buffer.clear()

            update_time = time.time() - start_time

            result = UpdateResult(
                updated=update_result.get("updated", False),
                performance_change=update_result.get("performance_change", 0.0),
                samples_processed=len(update_features),
                update_time=update_time,
                retrain_triggered=update_result.get("retrain_triggered", False),
                timestamp=datetime.now().isoformat(),
            )

            self.update_count += 1

            # 自动保存
            if (
                self.config.auto_save
                and self.update_count % self.config.save_frequency == 0
            ):
                self._auto_save()

            return result

        except Exception as e:
            logger.error(f"Update trigger failed: {e}")
            return UpdateResult(
                updated=False,
                performance_change=0.0,
                samples_processed=0,
                update_time=0.0,
                retrain_triggered=False,
                timestamp=datetime.now().isoformat(),
            )

    def _perform_update(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """执行实际的模型更新"""
        try:
            update_result = {
                "updated": False,
                "performance_change": 0.0,
                "retrain_triggered": False,
            }

            # 评估当前性能
            if hasattr(self.model, "predict"):
                current_predictions = self.model.predict(X)
                current_error = np.mean(np.abs(current_predictions - y))

                # 记录性能
                self.performance_history.append(current_error)

                # 计算性能变化
                if self.last_performance is not None:
                    performance_change = current_error - self.last_performance
                    update_result["performance_change"] = performance_change

                    # 检查是否需要重训练
                    if performance_change > self.config.retrain_threshold:
                        retrain_result = self._perform_retrain(X, y)
                        update_result["retrain_triggered"] = retrain_result

                        if retrain_result:
                            self.retrain_count += 1
                            logger.info(
                                f"Model retrained due to performance degradation: {performance_change:.4f}"
                            )

                self.last_performance = current_error

            # 执行增量更新
            if hasattr(self.model, "partial_fit"):
                # 支持增量学习的模型
                self.model.partial_fit(X, y)
                update_result["updated"] = True
                update_result["method"] = "partial_fit"

            elif hasattr(self.model, "fit") and not update_result.get(
                "retrain_triggered", False
            ):
                # 使用窗口数据重新训练
                window_size = min(len(X) * 3, 500)  # 使用3倍的数据窗口

                if len(self.performance_history) > 0:
                    # 简化的增量更新：只用新数据微调
                    self.model.fit(X, y)
                    update_result["updated"] = True
                    update_result["method"] = "incremental_fit"

            return update_result

        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return {"updated": False, "error": str(e)}

    def _perform_retrain(self, recent_X: np.ndarray, recent_y: np.ndarray) -> bool:
        """执行完整重训练"""
        try:
            if hasattr(self.model, "fit"):
                # 使用最近的数据重新训练
                self.model.fit(recent_X, recent_y)
                logger.info("Model retrained successfully")
                return True
            else:
                logger.warning("Model does not support retraining")
                return False

        except Exception as e:
            logger.error(f"Model retrain failed: {e}")
            return False

    def _auto_save(self):
        """自动保存模型状态"""
        try:
            if hasattr(self.model, "save"):
                filename = f"auto_save_{self.updater_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                success = self.model.save(filename)
                if success:
                    logger.info(f"Model auto-saved: {filename}")

            # 保存更新统计到数据库
            if hasattr(self, "db_manager"):
                self.db_manager.save_model_performance(
                    self.updater_id, "update_count", self.update_count
                )

                self.db_manager.save_model_performance(
                    self.updater_id, "samples_processed", self.samples_processed
                )

        except Exception as e:
            logger.error(f"Auto save failed: {e}")

    def start_background_updates(self, check_interval: float = 1.0):
        """启动后台更新线程"""
        try:
            if self.update_thread is not None and self.update_thread.is_alive():
                logger.warning("Background updates already running")
                return

            self.stop_updating = False
            self.update_thread = threading.Thread(
                target=self._background_update_loop, args=(check_interval,), daemon=True
            )
            self.update_thread.start()

            logger.info("Background updates started")

        except Exception as e:
            logger.error(f"Failed to start background updates: {e}")

    def stop_background_updates(self):
        """停止后台更新"""
        try:
            self.stop_updating = True
            if self.update_thread is not None:
                self.update_thread.join(timeout=5.0)

            logger.info("Background updates stopped")

        except Exception as e:
            logger.error(f"Failed to stop background updates: {e}")

    def _background_update_loop(self, check_interval: float):
        """后台更新循环"""
        while not self.stop_updating:
            try:
                # 检查是否有足够的样本进行更新
                if len(self.feature_buffer) >= self.config.update_frequency:
                    update_result = self._trigger_update()

                    if update_result.updated:
                        logger.info(f"Background update completed: {update_result}")

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"Background update error: {e}")
                time.sleep(check_interval)

    def get_update_statistics(self) -> Dict[str, Any]:
        """获取更新统计信息"""
        try:
            recent_performance = (
                list(self.performance_history)[-10:] if self.performance_history else []
            )

            return {
                "update_count": self.update_count,
                "samples_processed": self.samples_processed,
                "retrain_count": self.retrain_count,
                "buffer_size": len(self.feature_buffer),
                "current_performance": self.last_performance,
                "recent_performance_trend": np.mean(recent_performance)
                if recent_performance
                else None,
                "performance_variance": np.var(recent_performance)
                if recent_performance
                else None,
                "updater_id": self.updater_id,
                "background_updating": self.update_thread is not None
                and self.update_thread.is_alive(),
            }

        except Exception as e:
            logger.error(f"Failed to get update statistics: {e}")
            return {}

    def force_update(self) -> UpdateResult:
        """强制执行更新"""
        try:
            return self._trigger_update()
        except Exception as e:
            logger.error(f"Force update failed: {e}")
            return UpdateResult(
                updated=False,
                performance_change=0.0,
                samples_processed=0,
                update_time=0.0,
                retrain_triggered=False,
                timestamp=datetime.now().isoformat(),
            )

    def reset_buffers(self):
        """重置缓冲区"""
        try:
            with self.update_lock:
                self.feature_buffer.clear()
                self.target_buffer.clear()

            logger.info("Buffers reset")

        except Exception as e:
            logger.error(f"Failed to reset buffers: {e}")

    def set_update_callback(self, callback: Callable[[UpdateResult], None]):
        """设置更新回调函数"""
        self.update_callback = callback


def create_online_updater(
    model: Any,
    update_frequency: int = 10,
    performance_threshold: float = 0.1,
    auto_save: bool = True,
) -> OnlineUpdater:
    """创建在线更新器的便捷函数"""
    config = UpdateConfig(
        update_frequency=update_frequency,
        performance_threshold=performance_threshold,
        auto_save=auto_save,
    )
    return OnlineUpdater(model, config)
