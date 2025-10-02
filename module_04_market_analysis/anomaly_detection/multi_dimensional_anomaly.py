"""
Multi-Dimensional Anomaly Detector
多维度异常检测器，综合价格、成交量、技术指标等多个维度进行异常检测
"""

import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from common.logging_system import setup_logger
from module_01_data_pipeline import AkshareDataCollector, get_database_manager

from ..storage_management import get_market_analysis_db
from .price_anomaly_detector import AnomalyDetection, PriceAnomalyDetector
from .volume_anomaly_detector import VolumeAnomalyDetector

logger = setup_logger("multi_dimensional_anomaly")


@dataclass
class MultiDimensionalAnomaly:
    """多维度异常结果"""

    symbol: str
    timestamp: datetime
    anomaly_dimensions: List[str]  # 异常维度列表
    overall_score: float
    dimension_scores: Dict[str, float]
    description: str
    feature_values: Dict[str, float]
    detection_method: str
    confidence: float
    severity: str  # low, medium, high, critical


class MultiDimensionalAnomalyDetector:
    """多维度异常检测器"""

    def __init__(self, data_collector: Optional[AkshareDataCollector] = None):
        """初始化多维度异常检测器

        Args:
            data_collector: 数据收集器
        """
        self.data_collector = data_collector or AkshareDataCollector(rate_limit=0.5)
        self.db_manager = get_database_manager()
        self.analysis_db = get_market_analysis_db()

        # 初始化单维度检测器
        self.price_detector = PriceAnomalyDetector(data_collector)
        self.volume_detector = VolumeAnomalyDetector(data_collector)

        # 机器学习模型
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # 保畅95%的方差
        self.isolation_forest = IsolationForest(
            contamination=0.1, random_state=42, n_estimators=100
        )
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)

        # 异常阈值
        self.severity_thresholds = {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.9,
        }

        logger.info("Initialized multi-dimensional anomaly detector")

    async def detect_multi_dimensional_anomalies(
        self, symbols: List[str], lookback_days: int = 60, methods: List[str] = None
    ) -> Dict[str, List[MultiDimensionalAnomaly]]:
        """检测多维度异常

        Args:
            symbols: 股票代码列表
            lookback_days: 回看天数
            methods: 检测方法列表 ['comprehensive', 'pca', 'clustering', 'correlation']

        Returns:
            异常检测结果字典
        """
        if methods is None:
            methods = ["comprehensive", "pca", "clustering", "correlation"]

        logger.info(
            f"Starting multi-dimensional anomaly detection for {len(symbols)} symbols using methods: {methods}"
        )

        results = {}

        for symbol in symbols:
            try:
                # 获取和预处理数据
                multi_dim_data = await self._prepare_multi_dimensional_data(
                    symbol, lookback_days
                )

                if multi_dim_data.empty:
                    logger.warning(f"No multi-dimensional data available for {symbol}")
                    results[symbol] = []
                    continue

                if len(multi_dim_data) < 20:
                    logger.warning(
                        f"Insufficient data for {symbol}: {len(multi_dim_data)} points"
                    )
                    results[symbol] = []
                    continue

                # 执行各种多维度异常检测方法
                symbol_anomalies = []

                if "comprehensive" in methods:
                    comp_anomalies = await self._detect_comprehensive_anomalies(
                        multi_dim_data, symbol
                    )
                    symbol_anomalies.extend(comp_anomalies)

                if "pca" in methods:
                    pca_anomalies = self._detect_pca_anomalies(multi_dim_data, symbol)
                    symbol_anomalies.extend(pca_anomalies)

                if "clustering" in methods:
                    cluster_anomalies = self._detect_clustering_anomalies(
                        multi_dim_data, symbol
                    )
                    symbol_anomalies.extend(cluster_anomalies)

                if "correlation" in methods:
                    corr_anomalies = self._detect_correlation_anomalies(
                        multi_dim_data, symbol
                    )
                    symbol_anomalies.extend(corr_anomalies)

                # 保存异常检测结果到数据库
                for anomaly in symbol_anomalies:
                    self._save_multi_dim_anomaly_to_db(anomaly)

                results[symbol] = symbol_anomalies

                logger.info(
                    f"Detected {len(symbol_anomalies)} multi-dimensional anomalies for {symbol}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to detect multi-dimensional anomalies for {symbol}: {e}"
                )
                results[symbol] = []

        return results

    def _determine_severity(self, score: float) -> str:
        """确定严重程度"""
        for severity, threshold in sorted(
            self.severity_thresholds.items(), key=lambda x: x[1], reverse=True
        ):
            if score >= threshold:
                return severity
        return "low"


# 便捷函数
async def detect_multi_dimensional_anomalies_batch(
    symbols: List[str], lookback_days: int = 60, methods: List[str] = None
) -> Dict[str, List[MultiDimensionalAnomaly]]:
    """批量检测多维度异常

    Args:
        symbols: 股票代码列表
        lookback_days: 回看天数
        methods: 检测方法列表

    Returns:
        异常检测结果字典
    """
    detector = MultiDimensionalAnomalyDetector()
    return await detector.detect_multi_dimensional_anomalies(
        symbols, lookback_days, methods
    )


def create_multi_dimensional_anomaly_detector(
    data_collector: Optional[AkshareDataCollector] = None,
) -> MultiDimensionalAnomalyDetector:
    """创建多维度异常检测器

    Args:
        data_collector: 数据收集器

    Returns:
        多维度异常检测器实例
    """
    return MultiDimensionalAnomalyDetector(data_collector)
