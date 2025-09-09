"""
趋势状态分析器模块
分析市场趋势状态和转换点
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from common.logging_system import setup_logger

logger = setup_logger("trend_regime_analyzer")


@dataclass
class TrendRegime:
    """趋势状态"""
    regime_type: str
    start_date: datetime
    duration_days: int
    strength: float
    confidence: float
    characteristics: Dict[str, Any]


class TrendRegimeAnalyzer:
    """趋势状态分析器"""
    
    def __init__(self):
        """初始化趋势分析器"""
        self.short_window = 20
        self.long_window = 50
        self.trend_threshold = 0.02
        self.trend_history: List[TrendRegime] = []
        logger.info("TrendRegimeAnalyzer initialized")
    
    def analyze_trend_regime(self, price_data: pd.Series) -> TrendRegime:
        """分析当前趋势状态"""
        try:
            if len(price_data) < self.long_window:
                return self._create_default_regime("数据不足")
            
            # 计算移动平均线
            ma_short = price_data.rolling(window=self.short_window).mean()
            ma_long = price_data.rolling(window=self.long_window).mean()
            
            # 计算趋势指标
            ma_ratio = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
            momentum = (price_data.iloc[-1] - price_data.iloc[-20]) / price_data.iloc[-20]
            
            # 识别趋势类型
            if ma_ratio > self.trend_threshold:
                regime_type = "uptrend"
            elif ma_ratio < -self.trend_threshold:
                regime_type = "downtrend"
            else:
                regime_type = "sideways"
            
            # 计算趋势强度
            strength = min(1.0, abs(ma_ratio) * 10)
            
            # 计算置信度
            confidence = 0.8 if abs(momentum) > 0.01 else 0.5
            
            regime = TrendRegime(
                regime_type=regime_type,
                start_date=datetime.now(),
                duration_days=1,
                strength=strength,
                confidence=confidence,
                characteristics={
                    "ma_ratio": ma_ratio,
                    "momentum": momentum,
                    "current_price": float(price_data.iloc[-1])
                }
            )
            
            self.trend_history.append(regime)
            logger.info(f"Trend regime identified: {regime_type}")
            return regime
            
        except Exception as e:
            logger.error(f"Trend regime analysis failed: {e}")
            return self._create_default_regime(f"分析失败: {e}")
    
    def _create_default_regime(self, reason: str) -> TrendRegime:
        """创建默认趋势状态"""
        return TrendRegime(
            regime_type="sideways",
            start_date=datetime.now(),
            duration_days=1,
            strength=0.0,
            confidence=0.0,
            characteristics={"error": reason}
        )