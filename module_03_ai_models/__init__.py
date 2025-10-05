"""
AI模型模块初始化文件 - Module 03

Module 03 提供多范式机器学习模型架构：
- 深度学习模型（LSTM、Transformer）
- 集成学习方法
- 在线学习算法
- 强化学习智能体（RL、PPO）
- 统一的模型管理和数据库存储

Dataflow:
Module 01 (数据) -> Module 02 (特征) -> Module 03 (AI模型) -> Module 04/05/09 (应用)
"""

from .deep_learning.lstm_model import LSTMModel, LSTMModelConfig, LSTMPrediction
from .deep_learning.transformer_model import (
    TemporalTransformer,
    TransformerConfig,
    TransformerPredictor,
    create_transformer_predictor,
)
from .ensemble_methods.ensemble_predictor import (
    EnsembleConfig,
    EnsemblePrediction,
    EnsemblePredictor,
)
from .online_learning.online_learner import (
    OnlineLearner,
    OnlineLearningConfig,
    OnlineLearningResult,
)
from .reinforcement_learning.ppo_agent import (
    PPOAgent,
    PPOConfig,
    TradingEnvironment,
    create_ppo_agent,
)
from .reinforcement_learning.rl_agent import (
    Action,
    RLAction,
    RLAgent,
    RLConfig,
    RLState,
)

# 数据库管理器
from .storage_management.ai_model_database import get_ai_model_database_manager

# 便捷函数
from .utils import (
    create_lstm_predictor,
    create_online_learner,
    evaluate_model_performance,
    prepare_features_for_training,
    train_ensemble_model,
)

__all__ = [
    # 深度学习模型
    "LSTMModel",
    "LSTMModelConfig",
    "LSTMPrediction",
    "TransformerPredictor",
    "TransformerConfig",
    "TemporalTransformer",
    "create_transformer_predictor",
    # 集成学习
    "EnsemblePredictor",
    "EnsembleConfig",
    "EnsemblePrediction",
    # 在线学习
    "OnlineLearner",
    "OnlineLearningConfig",
    "OnlineLearningResult",
    # 强化学习
    "RLAgent",
    "RLConfig",
    "RLState",
    "RLAction",
    "Action",
    "PPOAgent",
    "PPOConfig",
    "TradingEnvironment",
    "create_ppo_agent",
    # 数据库管理
    "get_ai_model_database_manager",
    # 便捷函数
    "train_ensemble_model",
    "create_lstm_predictor",
    "create_online_learner",
    "prepare_features_for_training",
    "evaluate_model_performance",
]
