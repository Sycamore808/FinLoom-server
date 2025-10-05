"""
波动率状态检测模块
检测市场波动率状态和转换
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from common.logging_system import setup_logger

logger = setup_logger("volatility_regime")


@dataclass
class VolatilityRegime:
    """波动率状态"""
    regime_type: str  # 'low', 'normal', 'high', 'extreme'
    volatility_level: float
    confidence: float
    characteristics: Dict[str, Any]
    timestamp: datetime


class VolatilityRegimeDetector:
    """波动率状态检测器"""
    
    def __init__(self):
        """初始化波动率检测器"""
        # 波动率阈值
        self.low_threshold = 0.15    # 15%
        self.normal_threshold = 0.25  # 25%
        self.high_threshold = 0.40    # 40%
        
        self.volatility_history: List[VolatilityRegime] = []
        logger.info("VolatilityRegimeDetector initialized")
    
    def detect_volatility_regime(
        self, 
        returns: pd.Series,
        window: int = 20
    ) -> VolatilityRegime:
        """检测波动率状态
        
        Args:
            returns: 收益率序列
            window: 计算窗口
            
        Returns:
            波动率状态
        """
        try:
            if len(returns) < window:
                return self._create_default_regime("数据不足")
            
            # 计算当前波动率
            current_vol = returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
            
            # 计算历史波动率
            historical_vol = returns.std() * np.sqrt(252)
            
            # 识别波动率状态
            if current_vol < self.low_threshold:
                regime_type = "low"
            elif current_vol < self.normal_threshold:
                regime_type = "normal"
            elif current_vol < self.high_threshold:
                regime_type = "high"
            else:
                regime_type = "extreme"
            
            # 计算置信度
            confidence = self._calculate_confidence(current_vol, historical_vol)
            
            # 提取特征
            characteristics = {
                "current_volatility": current_vol,
                "historical_volatility": historical_vol,
                "volatility_ratio": current_vol / historical_vol if historical_vol > 0 else 1.0,
                "window": window
            }
            
            regime = VolatilityRegime(
                regime_type=regime_type,
                volatility_level=current_vol,
                confidence=confidence,
                characteristics=characteristics,
                timestamp=datetime.now()
            )
            
            self.volatility_history.append(regime)
            logger.info(f"Volatility regime detected: {regime_type} ({current_vol:.3f})")
            return regime
            
        except Exception as e:
            logger.error(f"Volatility regime detection failed: {e}")
            return self._create_default_regime(f"检测失败: {e}")
    
    def _calculate_confidence(
        self, 
        current_vol: float, 
        historical_vol: float
    ) -> float:
        """计算置信度"""
        if historical_vol == 0:
            return 0.5
        
        # 基于当前波动率与历史波动率的比较
        vol_ratio = current_vol / historical_vol
        
        if 0.8 <= vol_ratio <= 1.2:
            return 0.9  # 接近历史水平
        elif 0.6 <= vol_ratio <= 1.4:
            return 0.7  # 略有偏离
        else:
            return 0.5  # 显著偏离
    
    def _create_default_regime(self, reason: str) -> VolatilityRegime:
        """创建默认波动率状态"""
        return VolatilityRegime(
            regime_type="normal",
            volatility_level=0.2,
            confidence=0.0,
            characteristics={"error": reason},
            timestamp=datetime.now()
        )