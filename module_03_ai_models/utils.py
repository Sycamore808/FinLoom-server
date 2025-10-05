"""
AI模型工具函数和便捷接口

提供常用的模型创建、训练、评估等便捷函数
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.exceptions import ModelError
from common.logging_system import setup_logger

# Module imports
from module_01_data_pipeline import AkshareDataCollector, get_database_manager
from module_02_feature_engineering import (
    TechnicalIndicators,
    calculate_technical_indicators,
    get_feature_database_manager,
)

# Local imports
from .deep_learning.lstm_model import LSTMModel, LSTMModelConfig
from .ensemble_methods.ensemble_predictor import EnsembleConfig, EnsemblePredictor
from .online_learning.online_learner import OnlineLearner, OnlineLearningConfig
from .storage_management.ai_model_database import get_ai_model_database_manager

logger = setup_logger("ai_models_utils")


def prepare_features_for_training(
    symbols: List[str], start_date: str, end_date: str, feature_types: List[str] = None
) -> pd.DataFrame:
    """从Module 01和02准备训练特征数据

    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        feature_types: 特征类型 ['technical', 'fundamental', 'macro']

    Returns:
        特征DataFrame
    """
    try:
        if feature_types is None:
            feature_types = ["technical"]

        # 获取数据管理器
        data_db = get_database_manager()
        feature_db = get_feature_database_manager()

        all_features = []

        for symbol in symbols:
            # 从Module 01获取基础数据
            price_data = data_db.get_stock_prices(symbol, start_date, end_date)
            if price_data.empty:
                logger.warning(f"No price data for {symbol}")
                continue

            symbol_features = price_data.copy()

            # 添加技术指标特征
            if "technical" in feature_types:
                try:
                    tech_indicators = feature_db.get_technical_indicators(
                        symbol, start_date, end_date
                    )
                    if not tech_indicators.empty:
                        symbol_features = pd.merge(
                            symbol_features,
                            tech_indicators,
                            left_index=True,
                            right_index=True,
                            how="left",
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to get technical indicators for {symbol}: {e}"
                    )

            # 添加时间序列特征
            if "time_series" in feature_types:
                try:
                    ts_features = feature_db.get_time_series_features(
                        symbol, start_date, end_date
                    )
                    if not ts_features.empty:
                        symbol_features = pd.merge(
                            symbol_features,
                            ts_features,
                            left_index=True,
                            right_index=True,
                            how="left",
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to get time series features for {symbol}: {e}"
                    )

            # 添加股票标识
            symbol_features["symbol"] = symbol
            all_features.append(symbol_features)

        if not all_features:
            raise ModelError("No feature data available")

        # 合并所有特征
        combined_features = pd.concat(all_features, ignore_index=True)

        # 清理和预处理
        combined_features = combined_features.dropna()

        logger.info(
            f"Prepared features: {combined_features.shape[0]} samples, {combined_features.shape[1]} features"
        )
        return combined_features

    except Exception as e:
        logger.error(f"Failed to prepare features: {e}")
        raise ModelError(f"Feature preparation failed: {e}")


def create_lstm_predictor(
    input_dim: int, sequence_length: int = 60, model_name: str = None
) -> LSTMModel:
    """创建LSTM预测器

    Args:
        input_dim: 输入特征维度
        sequence_length: 序列长度
        model_name: 模型名称

    Returns:
        LSTM模型实例
    """
    try:
        config = LSTMModelConfig(
            sequence_length=sequence_length,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
        )

        model = LSTMModel(config)

        # 保存模型信息到数据库
        if model_name:
            model_id = f"lstm_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ai_db = get_ai_model_database_manager()
            ai_db.save_model_info(
                model_id=model_id,
                model_type="lstm",
                model_name=model_name,
                config=config.__dict__,
            )

        logger.info(f"Created LSTM predictor with sequence_length={sequence_length}")
        return model

    except Exception as e:
        logger.error(f"Failed to create LSTM predictor: {e}")
        raise ModelError(f"LSTM creation failed: {e}")


def create_online_learner(
    learning_rate: float = 0.01, buffer_size: int = 1000, model_name: str = None
) -> OnlineLearner:
    """创建在线学习器

    Args:
        learning_rate: 学习率
        buffer_size: 缓冲区大小
        model_name: 模型名称

    Returns:
        在线学习器实例
    """
    try:
        config = OnlineLearningConfig(
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            update_frequency=100,
            decay_rate=0.95,
        )

        learner = OnlineLearner(config)

        # 保存模型信息到数据库
        if model_name:
            model_id = f"online_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ai_db = get_ai_model_database_manager()
            ai_db.save_model_info(
                model_id=model_id,
                model_type="online",
                model_name=model_name,
                config=config.__dict__,
            )

        logger.info(f"Created online learner with learning_rate={learning_rate}")
        return learner

    except Exception as e:
        logger.error(f"Failed to create online learner: {e}")
        raise ModelError(f"Online learner creation failed: {e}")


def train_ensemble_model(
    models: List[Dict[str, Any]],
    features: pd.DataFrame,
    target_column: str,
    model_name: str = None,
) -> EnsemblePredictor:
    """训练集成模型

    Args:
        models: 模型列表，每个包含model实例和weight
        features: 特征数据
        target_column: 目标列名
        model_name: 模型名称

    Returns:
        训练好的集成预测器
    """
    try:
        # 准备训练数据
        X = features.drop(columns=[target_column, "symbol"], errors="ignore").values
        y = features[target_column].values

        # 创建集成配置
        config = EnsembleConfig(
            models=models,
            voting_strategy="weighted",
            weights=[model.get("weight", 1.0) for model in models],
        )

        # 创建集成预测器
        ensemble = EnsemblePredictor(config)

        # 添加各个模型
        for model_info in models:
            ensemble.add_model(
                name=model_info["name"],
                model=model_info["model"],
                weight=model_info.get("weight", 1.0),
            )

        # 训练集成模型
        training_metrics = ensemble.train_ensemble(X, y)

        # 保存到数据库
        if model_name:
            model_id = (
                f"ensemble_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            ai_db = get_ai_model_database_manager()

            # 保存集成模型信息
            ai_db.save_model_info(
                model_id=model_id,
                model_type="ensemble",
                model_name=model_name,
                config=config.__dict__,
            )

            # 保存训练指标
            for metric_name, metric_value in training_metrics.items():
                ai_db.save_model_performance(model_id, metric_name, metric_value)

        logger.info(f"Trained ensemble model with {len(models)} components")
        return ensemble

    except Exception as e:
        logger.error(f"Failed to train ensemble model: {e}")
        raise ModelError(f"Ensemble training failed: {e}")


def evaluate_model_performance(
    model: Any, test_features: pd.DataFrame, target_column: str, model_id: str = None
) -> Dict[str, float]:
    """评估模型性能

    Args:
        model: 模型实例
        test_features: 测试特征数据
        target_column: 目标列名
        model_id: 模型ID（用于保存结果）

    Returns:
        性能指标字典
    """
    try:
        # 准备测试数据
        X_test = test_features.drop(
            columns=[target_column, "symbol"], errors="ignore"
        ).values
        y_test = test_features[target_column].values

        # 进行预测
        if hasattr(model, "predict"):
            predictions = model.predict(X_test)
            if hasattr(predictions, "predictions"):
                y_pred = predictions.predictions
                confidence = getattr(predictions, "confidence", 0.8)
            else:
                y_pred = predictions
                confidence = 0.8
        else:
            raise ModelError("Model does not have predict method")

        # 计算性能指标
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(mse)

        # 计算方向准确率（金融中重要）
        direction_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))

        # 计算相关系数
        correlation = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0

        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "direction_accuracy": direction_accuracy,
            "correlation": correlation,
            "mean_confidence": confidence,
        }

        # 保存到数据库
        if model_id:
            ai_db = get_ai_model_database_manager()
            for metric_name, metric_value in metrics.items():
                ai_db.save_model_performance(model_id, metric_name, metric_value)

        logger.info(
            f"Model evaluation completed: RMSE={rmse:.4f}, Direction Accuracy={direction_accuracy:.4f}"
        )
        return metrics

    except Exception as e:
        logger.error(f"Failed to evaluate model performance: {e}")
        raise ModelError(f"Model evaluation failed: {e}")


def create_features_from_raw_data(
    symbols: List[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """从原始数据创建特征（如果Module 02特征不存在）

    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        特征DataFrame
    """
    try:
        collector = AkshareDataCollector(rate_limit=0.5)
        calculator = TechnicalIndicators()

        all_features = []

        for symbol in symbols:
            # 获取原始数据
            raw_data = collector.fetch_stock_history(symbol, start_date, end_date)
            if raw_data.empty:
                continue

            # 计算技术指标
            indicators = calculator.calculate_all_indicators(raw_data)

            # 计算基础特征
            features = indicators.copy()
            features["returns"] = features["close"].pct_change()
            features["log_returns"] = np.log(
                features["close"] / features["close"].shift(1)
            )
            features["volatility"] = features["returns"].rolling(20).std()
            features["volume_ratio"] = (
                features["volume"] / features["volume"].rolling(20).mean()
            )

            # 添加股票标识
            features["symbol"] = symbol
            all_features.append(features)

        if not all_features:
            raise ModelError("No data available for feature creation")

        # 合并和清理
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_features = combined_features.dropna()

        logger.info(f"Created features from raw data: {combined_features.shape}")
        return combined_features

    except Exception as e:
        logger.error(f"Failed to create features from raw data: {e}")
        raise ModelError(f"Feature creation failed: {e}")


def batch_predict_stocks(
    model: Any, symbols: List[str], prediction_date: str, model_id: str = None
) -> Dict[str, Dict[str, float]]:
    """批量预测多只股票

    Args:
        model: 训练好的模型
        symbols: 股票代码列表
        prediction_date: 预测日期
        model_id: 模型ID

    Returns:
        预测结果字典 {symbol: {prediction: value, confidence: value}}
    """
    try:
        predictions = {}

        # 准备特征数据
        features = prepare_features_for_training(
            symbols=symbols,
            start_date="2024-01-01",  # 使用足够的历史数据
            end_date=prediction_date,
        )

        for symbol in symbols:
            try:
                # 获取该股票的最新特征
                symbol_features = features[features["symbol"] == symbol].tail(
                    60
                )  # 使用最近60天

                if len(symbol_features) < 10:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue

                # 准备预测输入
                X = symbol_features.drop(columns=["symbol"], errors="ignore").values

                # 进行预测
                prediction_result = model.predict(X[-1:])  # 只用最新的数据点

                if hasattr(prediction_result, "predictions"):
                    pred_value = prediction_result.predictions[0]
                    confidence = getattr(prediction_result, "confidence", 0.8)
                else:
                    pred_value = (
                        prediction_result[0]
                        if hasattr(prediction_result, "__getitem__")
                        else prediction_result
                    )
                    confidence = 0.8

                predictions[symbol] = {
                    "prediction": float(pred_value),
                    "confidence": float(confidence),
                }

                # 保存预测结果到数据库
                if model_id:
                    ai_db = get_ai_model_database_manager()
                    ai_db.save_model_prediction(
                        model_id=model_id,
                        symbol=symbol,
                        prediction_date=prediction_date,
                        prediction_value=pred_value,
                        confidence=confidence,
                    )

            except Exception as e:
                logger.warning(f"Failed to predict {symbol}: {e}")
                continue

        logger.info(f"Batch prediction completed for {len(predictions)} stocks")
        return predictions

    except Exception as e:
        logger.error(f"Failed to batch predict stocks: {e}")
        raise ModelError(f"Batch prediction failed: {e}")


# 便捷的训练和预测工作流
def quick_lstm_training_workflow(
    symbols: List[str], target_column: str = "returns", model_name: str = "quick_lstm"
) -> Tuple[LSTMModel, Dict[str, float]]:
    """快速LSTM训练工作流

    Args:
        symbols: 股票代码列表
        target_column: 目标列名
        model_name: 模型名称

    Returns:
        (训练好的模型, 性能指标)
    """
    try:
        # 准备数据
        features = prepare_features_for_training(
            symbols=symbols,
            start_date="2023-01-01",
            end_date="2024-12-01",
            feature_types=["technical"],
        )

        if features.empty:
            # 如果没有特征数据，从原始数据创建
            features = create_features_from_raw_data(
                symbols, "2023-01-01", "2024-12-01"
            )

        # 分割训练和测试数据
        train_size = int(0.8 * len(features))
        train_features = features[:train_size]
        test_features = features[train_size:]

        # 创建LSTM模型
        input_dim = len(train_features.columns) - 2  # 减去symbol和target列
        model = create_lstm_predictor(input_dim=input_dim, model_name=model_name)

        # 准备训练数据
        X_train = train_features.drop(
            columns=[target_column, "symbol"], errors="ignore"
        ).values
        y_train = train_features[target_column].values

        X_train, y_train = model.prepare_data(
            data=train_features.drop(columns=["symbol"], errors="ignore"),
            target_column=target_column,
        )

        # 训练模型
        training_metrics = model.train(X_train, y_train)

        # 评估模型
        performance = evaluate_model_performance(model, test_features, target_column)

        logger.info(f"Quick LSTM training completed: {model_name}")
        return model, performance

    except Exception as e:
        logger.error(f"Failed quick LSTM training workflow: {e}")
        raise ModelError(f"Quick training workflow failed: {e}")
