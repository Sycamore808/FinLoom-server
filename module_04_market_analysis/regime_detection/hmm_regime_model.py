"""
隐马尔可夫模型市场状态检测
使用HMM模型识别市场状态转换
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
try:
    from hmmlearn import hmm
except ImportError:
    hmm = None
from sklearn.preprocessing import StandardScaler
from common.logging_system import setup_logger

logger = setup_logger("hmm_regime_model")


@dataclass
class HMMConfig:
    """HMM配置"""
    n_components: int = 4  # 状态数量
    covariance_type: str = "full"  # 协方差类型
    n_iter: int = 100  # 迭代次数
    random_state: int = 42
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = [
                "returns", "volatility", "volume", "trend", 
                "momentum", "correlation", "skewness", "kurtosis"
            ]


@dataclass
class HMMRegimeResult:
    """HMM状态检测结果"""
    regime: int
    probability: float
    regime_probabilities: List[float]
    transition_matrix: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    log_likelihood: float
    timestamp: datetime


class HMMRegimeModel:
    """HMM市场状态模型"""
    
    def __init__(self, config: Optional[HMMConfig] = None):
        """初始化HMM模型
        
        Args:
            config: HMM配置
        """
        self.config = config or HMMConfig()
        self.model: Optional[hmm.GaussianHMM] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # 状态标签映射
        self.regime_labels = {
            0: "bear_volatile",    # 熊市高波动
            1: "bear_quiet",       # 熊市低波动  
            2: "bull_quiet",       # 牛市低波动
            3: "bull_volatile"     # 牛市高波动
        }
        
        logger.info(f"HMMRegimeModel initialized with {self.config.n_components} states")
    
    def fit(self, data: pd.DataFrame) -> HMMRegimeResult:
        """训练HMM模型
        
        Args:
            data: 市场数据DataFrame
            
        Returns:
            训练结果
        """
        try:
            logger.info("Training HMM regime model...")
            
            # 准备特征数据
            features = self._prepare_features(data)
            
            # 标准化特征
            features_scaled = self.scaler.fit_transform(features)
            
            # 创建并训练HMM模型
            if hmm is None:
                raise ImportError("hmmlearn package is required for HMM functionality")
            
            self.model = hmm.GaussianHMM(
                n_components=self.config.n_components,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                random_state=self.config.random_state
            )
            
            self.model.fit(features_scaled)
            self.is_fitted = True
            
            # 预测最新状态
            latest_features = features_scaled[-1:].reshape(1, -1)
            regime = self.model.predict(latest_features)[0]
            probabilities = self.model.predict_proba(latest_features)[0]
            
            result = HMMRegimeResult(
                regime=regime,
                probability=probabilities[regime],
                regime_probabilities=probabilities.tolist(),
                transition_matrix=self.model.transmat_,
                means=self.model.means_,
                covariances=self.model.covars_,
                log_likelihood=self.model.score(features_scaled),
                timestamp=datetime.now()
            )
            
            logger.info(f"HMM model trained successfully. Latest regime: {regime}")
            return result
            
        except Exception as e:
            logger.error(f"HMM model training failed: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> HMMRegimeResult:
        """预测市场状态
        
        Args:
            data: 市场数据DataFrame
            
        Returns:
            预测结果
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # 准备特征数据
            features = self._prepare_features(data)
            
            # 标准化特征
            features_scaled = self.scaler.transform(features)
            
            # 预测状态
            regime = self.model.predict(features_scaled)[-1]
            probabilities = self.model.predict_proba(features_scaled)[-1]
            
            result = HMMRegimeResult(
                regime=regime,
                probability=probabilities[regime],
                regime_probabilities=probabilities.tolist(),
                transition_matrix=self.model.transmat_,
                means=self.model.means_,
                covariances=self.model.covars_,
                log_likelihood=self.model.score(features_scaled),
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            raise
    
    def predict_sequence(self, data: pd.DataFrame) -> List[int]:
        """预测状态序列
        
        Args:
            data: 市场数据DataFrame
            
        Returns:
            状态序列
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # 准备特征数据
            features = self._prepare_features(data)
            
            # 标准化特征
            features_scaled = self.scaler.transform(features)
            
            # 预测状态序列
            regimes = self.model.predict(features_scaled)
            
            return regimes.tolist()
            
        except Exception as e:
            logger.error(f"HMM sequence prediction failed: {e}")
            raise
    
    def get_regime_characteristics(self, regime: int) -> Dict[str, Any]:
        """获取状态特征
        
        Args:
            regime: 状态编号
            
        Returns:
            状态特征描述
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted")
        
        if regime >= self.config.n_components:
            raise ValueError(f"Invalid regime: {regime}")
        
        # 获取状态均值和协方差
        means = self.model.means_[regime]
        covariances = self.model.covars_[regime]
        
        # 反标准化均值
        means_original = self.scaler.inverse_transform(means.reshape(1, -1))[0]
        
        characteristics = {
            "regime_id": regime,
            "regime_label": self.regime_labels.get(regime, f"regime_{regime}"),
            "means": means_original.tolist(),
            "volatility": np.sqrt(np.diag(covariances)).tolist(),
            "feature_names": self.config.features
        }
        
        # 添加状态描述
        if regime == 0:  # bear_volatile
            characteristics["description"] = "熊市高波动状态：市场下跌且波动率较高"
        elif regime == 1:  # bear_quiet
            characteristics["description"] = "熊市低波动状态：市场下跌但波动率较低"
        elif regime == 2:  # bull_quiet
            characteristics["description"] = "牛市低波动状态：市场上涨且波动率较低"
        elif regime == 3:  # bull_volatile
            characteristics["description"] = "牛市高波动状态：市场上涨但波动率较高"
        
        return characteristics
    
    def get_transition_probabilities(self, current_regime: int) -> Dict[int, float]:
        """获取状态转换概率
        
        Args:
            current_regime: 当前状态
            
        Returns:
            到各状态的转换概率
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted")
        
        if current_regime >= self.config.n_components:
            raise ValueError(f"Invalid regime: {current_regime}")
        
        transition_probs = {}
        for i in range(self.config.n_components):
            transition_probs[i] = float(self.model.transmat_[current_regime, i])
        
        return transition_probs
    
    def get_regime_persistence(self, regimes: List[int]) -> Dict[int, float]:
        """计算状态持续性
        
        Args:
            regimes: 状态序列
            
        Returns:
            各状态的持续性指标
        """
        if not regimes:
            return {}
        
        persistence = {}
        for regime in range(self.config.n_components):
            # 计算该状态的持续时间分布
            durations = []
            current_duration = 0
            
            for r in regimes:
                if r == regime:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0
            
            # 处理最后一个状态
            if current_duration > 0:
                durations.append(current_duration)
            
            # 计算平均持续时间
            if durations:
                persistence[regime] = np.mean(durations)
            else:
                persistence[regime] = 0.0
        
        return persistence
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """准备特征数据
        
        Args:
            data: 原始数据
            
        Returns:
            特征矩阵
        """
        features = []
        
        for feature_name in self.config.features:
            if feature_name in data.columns:
                feature_data = data[feature_name].values
                # 处理缺失值
                feature_data = np.nan_to_num(feature_data, nan=0.0)
                features.append(feature_data)
            else:
                # 如果特征不存在，用零填充
                logger.warning(f"Feature {feature_name} not found in data")
                features.append(np.zeros(len(data)))
        
        return np.column_stack(features)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要
        
        Returns:
            模型摘要信息
        """
        if not self.is_fitted or self.model is None:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "n_components": self.config.n_components,
            "covariance_type": self.config.covariance_type,
            "features": self.config.features,
            "regime_labels": self.regime_labels,
            "log_likelihood": float(self.model.score(self.scaler.transform(
                self._prepare_features(pd.DataFrame())
            ))) if hasattr(self, 'scaler') else None
        }


# 便捷函数
def detect_market_regime_hmm(
    market_data: pd.DataFrame,
    n_components: int = 4
) -> Dict[str, Any]:
    """使用HMM检测市场状态的便捷函数
    
    Args:
        market_data: 市场数据
        n_components: 状态数量
        
    Returns:
        市场状态检测结果
    """
    config = HMMConfig(n_components=n_components)
    model = HMMRegimeModel(config)
    
    # 训练模型
    result = model.fit(market_data)
    
    # 获取状态特征
    characteristics = model.get_regime_characteristics(result.regime)
    
    # 获取转换概率
    transition_probs = model.get_transition_probabilities(result.regime)
    
    return {
        "current_regime": result.regime,
        "regime_probability": result.probability,
        "regime_probabilities": result.regime_probabilities,
        "regime_characteristics": characteristics,
        "transition_probabilities": transition_probs,
        "log_likelihood": result.log_likelihood,
        "model_summary": model.get_model_summary()
    }