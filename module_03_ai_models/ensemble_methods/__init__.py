"""
集成学习模块初始化文件

提供集成学习方法：
- 集成预测器
- 模型集成
- 堆叠集成（Stacking）
"""

from .ensemble_predictor import (
    EnsembleConfig,
    EnsemblePrediction,
    EnsemblePredictor,
)

__all__ = [
    "EnsembleConfig",
    "EnsemblePrediction",
    "EnsemblePredictor",
]
