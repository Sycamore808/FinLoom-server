"""
Price Anomaly Detector
基于统计方法和机器学习的股价异常检测器
"""

import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from common.logging_system import setup_logger
from module_01_data_pipeline import AkshareDataCollector, get_database_manager

from ..storage_management import get_market_analysis_db

logger = setup_logger("price_anomaly_detector")


@dataclass
class AnomalyDetection:
    """异常检测结果"""

    symbol: str
    timestamp: datetime
    anomaly_type: str
    anomaly_score: float
    description: str
    data_point: Dict[str, Any]
    threshold_values: Dict[str, float]
    detection_method: str
    confidence: float


class PriceAnomalyDetector:
    """股价异常检测器"""

    def __init__(self, data_collector: Optional[AkshareDataCollector] = None):
        """初始化异常检测器

        Args:
            data_collector: 数据收集器
        """
        self.data_collector = data_collector or AkshareDataCollector(rate_limit=0.5)
        self.db_manager = get_database_manager()
        self.analysis_db = get_market_analysis_db()
        self.scaler = StandardScaler()

        # 异常检测参数
        self.z_score_threshold = 3.0
        self.iqr_multiplier = 1.5
        self.isolation_forest_contamination = 0.1

        logger.info("Initialized price anomaly detector")

    async def detect_price_anomalies(
        self, symbols: List[str], lookback_days: int = 60, methods: List[str] = None
    ) -> Dict[str, List[AnomalyDetection]]:
        """检测股价异常

        Args:
            symbols: 股票代码列表
            lookback_days: 回看天数
            methods: 检测方法列表 ['z_score', 'iqr', 'isolation_forest', 'statistical']

        Returns:
            异常检测结果字典
        """
        if methods is None:
            methods = ["z_score", "iqr", "isolation_forest", "statistical"]

        logger.info(
            f"Starting anomaly detection for {len(symbols)} symbols using methods: {methods}"
        )

        results = {}

        for symbol in symbols:
            try:
                # 获取历史数据
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (
                    datetime.now() - timedelta(days=lookback_days + 30)
                ).strftime("%Y%m%d")

                price_data = self.data_collector.fetch_stock_history(
                    symbol, start_date, end_date
                )

                if price_data.empty:
                    logger.warning(f"No price data available for {symbol}")
                    results[symbol] = []
                    continue

                # 清理和预处理数据
                price_data = self._preprocess_data(price_data)

                if len(price_data) < 20:  # 至少需要20个数据点
                    logger.warning(
                        f"Insufficient data for {symbol}: {len(price_data)} points"
                    )
                    results[symbol] = []
                    continue

                # 执行各种异常检测方法
                symbol_anomalies = []

                if "z_score" in methods:
                    z_score_anomalies = self._detect_z_score_anomalies(
                        price_data, symbol
                    )
                    symbol_anomalies.extend(z_score_anomalies)

                if "iqr" in methods:
                    iqr_anomalies = self._detect_iqr_anomalies(price_data, symbol)
                    symbol_anomalies.extend(iqr_anomalies)

                if "isolation_forest" in methods:
                    isolation_anomalies = self._detect_isolation_forest_anomalies(
                        price_data, symbol
                    )
                    symbol_anomalies.extend(isolation_anomalies)

                if "statistical" in methods:
                    statistical_anomalies = self._detect_statistical_anomalies(
                        price_data, symbol
                    )
                    symbol_anomalies.extend(statistical_anomalies)

                # 保存异常检测结果到数据库
                for anomaly in symbol_anomalies:
                    self._save_anomaly_to_db(anomaly)

                results[symbol] = symbol_anomalies

                logger.info(f"Detected {len(symbol_anomalies)} anomalies for {symbol}")

            except Exception as e:
                logger.error(f"Failed to detect anomalies for {symbol}: {e}")
                results[symbol] = []

        return results

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理价格数据

        Args:
            df: 原始价格数据

        Returns:
            预处理后的数据
        """
        # 确保有必要的列
        required_columns = ["date", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
                return pd.DataFrame()

        # 按日期排序
        df = df.sort_values("date").reset_index(drop=True)

        # 计算技术指标
        df["price_change"] = df["close"].pct_change()
        df["price_range"] = (df["high"] - df["low"]) / df["close"]
        df["volume_change"] = df["volume"].pct_change()
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # 计算移动平均和标准差
        df["ma_5"] = df["close"].rolling(window=5).mean()
        df["ma_20"] = df["close"].rolling(window=20).mean()
        df["std_20"] = df["close"].rolling(window=20).std()

        # 计算布林带
        df["bb_upper"] = df["ma_20"] + 2 * df["std_20"]
        df["bb_lower"] = df["ma_20"] - 2 * df["std_20"]

        # 删除NaN值
        df = df.dropna().reset_index(drop=True)

        return df

    def _detect_z_score_anomalies(
        self, df: pd.DataFrame, symbol: str
    ) -> List[AnomalyDetection]:
        """基于Z-Score的异常检测

        Args:
            df: 价格数据
            symbol: 股票代码

        Returns:
            异常检测结果列表
        """
        anomalies = []

        # 对不同指标进行Z-Score检测
        indicators = {
            "price_change": "价格变化",
            "volume_change": "成交量变化",
            "price_range": "价格波动幅度",
            "gap": "跳空幅度",
        }

        for indicator, desc in indicators.items():
            if indicator not in df.columns:
                continue

            # 计算Z-Score
            values = df[indicator].dropna()
            if len(values) < 10:
                continue

            z_scores = np.abs(stats.zscore(values))

            # 找出异常值
            anomaly_indices = np.where(z_scores > self.z_score_threshold)[0]

            for idx in anomaly_indices:
                if idx < len(df):
                    row = df.iloc[idx]

                    anomaly = AnomalyDetection(
                        symbol=symbol,
                        timestamp=row["date"],
                        anomaly_type=f"{indicator}_z_score",
                        anomaly_score=float(z_scores[idx]),
                        description=f"{desc}异常：Z-Score = {z_scores[idx]:.2f}",
                        data_point={
                            "date": row["date"].isoformat()
                            if hasattr(row["date"], "isoformat")
                            else str(row["date"]),
                            "close": float(row["close"]),
                            "volume": int(row["volume"]),
                            indicator: float(row[indicator]),
                            "z_score": float(z_scores[idx]),
                        },
                        threshold_values={
                            "z_score_threshold": self.z_score_threshold,
                            "mean": float(values.mean()),
                            "std": float(values.std()),
                        },
                        detection_method="z_score",
                        confidence=min(float(z_scores[idx]) / 5.0, 1.0),  # 归一化置信度
                    )

                    anomalies.append(anomaly)

        logger.info(f"Z-Score detection found {len(anomalies)} anomalies for {symbol}")
        return anomalies

    def _detect_iqr_anomalies(
        self, df: pd.DataFrame, symbol: str
    ) -> List[AnomalyDetection]:
        """基于IQR的异常检测

        Args:
            df: 价格数据
            symbol: 股票代码

        Returns:
            异常检测结果列表
        """
        anomalies = []

        indicators = {
            "price_change": "价格变化",
            "volume_change": "成交量变化",
            "price_range": "价格波动幅度",
        }

        for indicator, desc in indicators.items():
            if indicator not in df.columns:
                continue

            values = df[indicator].dropna()
            if len(values) < 10:
                continue

            # 计算IQR
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr

            # 找出异常值
            outliers = values[(values < lower_bound) | (values > upper_bound)]

            for idx in outliers.index:
                if idx < len(df):
                    row = df.iloc[idx]
                    value = outliers.loc[idx]

                    # 计算异常程度
                    if value < lower_bound:
                        anomaly_score = abs(value - lower_bound) / iqr
                        direction = "下方"
                    else:
                        anomaly_score = abs(value - upper_bound) / iqr
                        direction = "上方"

                    anomaly = AnomalyDetection(
                        symbol=symbol,
                        timestamp=row["date"],
                        anomaly_type=f"{indicator}_iqr",
                        anomaly_score=float(anomaly_score),
                        description=f"{desc}异常：值{value:.4f}位于IQR{direction}",
                        data_point={
                            "date": row["date"].isoformat()
                            if hasattr(row["date"], "isoformat")
                            else str(row["date"]),
                            "close": float(row["close"]),
                            "volume": int(row["volume"]),
                            indicator: float(value),
                            "direction": direction,
                        },
                        threshold_values={
                            "q1": float(q1),
                            "q3": float(q3),
                            "iqr": float(iqr),
                            "lower_bound": float(lower_bound),
                            "upper_bound": float(upper_bound),
                        },
                        detection_method="iqr",
                        confidence=min(float(anomaly_score) / 3.0, 1.0),
                    )

                    anomalies.append(anomaly)

        logger.info(f"IQR detection found {len(anomalies)} anomalies for {symbol}")
        return anomalies

    def _detect_isolation_forest_anomalies(
        self, df: pd.DataFrame, symbol: str
    ) -> List[AnomalyDetection]:
        """基于Isolation Forest的异常检测

        Args:
            df: 价格数据
            symbol: 股票代码

        Returns:
            异常检测结果列表
        """
        anomalies = []

        try:
            # 准备特征数据
            feature_columns = ["price_change", "volume_change", "price_range", "gap"]
            available_columns = [col for col in feature_columns if col in df.columns]

            if len(available_columns) < 2:
                logger.warning(
                    f"Insufficient features for isolation forest: {available_columns}"
                )
                return anomalies

            feature_data = df[available_columns].dropna()

            if len(feature_data) < 10:
                logger.warning(
                    f"Insufficient data points for isolation forest: {len(feature_data)}"
                )
                return anomalies

            # 标准化特征
            scaled_features = self.scaler.fit_transform(feature_data)

            # 训练Isolation Forest模型
            iso_forest = IsolationForest(
                contamination=self.isolation_forest_contamination,
                random_state=42,
                n_estimators=100,
            )

            # 预测异常
            predictions = iso_forest.fit_predict(scaled_features)
            anomaly_scores = iso_forest.decision_function(scaled_features)

            # 找出异常点
            anomaly_indices = np.where(predictions == -1)[0]

            for idx in anomaly_indices:
                if idx < len(df):
                    original_idx = feature_data.index[idx]
                    row = df.iloc[original_idx]

                    anomaly = AnomalyDetection(
                        symbol=symbol,
                        timestamp=row["date"],
                        anomaly_type="isolation_forest",
                        anomaly_score=float(abs(anomaly_scores[idx])),
                        description=f"机器学习异常检测：异常分数 = {anomaly_scores[idx]:.4f}",
                        data_point={
                            "date": row["date"].isoformat()
                            if hasattr(row["date"], "isoformat")
                            else str(row["date"]),
                            "close": float(row["close"]),
                            "volume": int(row["volume"]),
                            "anomaly_score": float(anomaly_scores[idx]),
                            "features": {
                                col: float(row[col]) for col in available_columns
                            },
                        },
                        threshold_values={
                            "contamination": self.isolation_forest_contamination,
                            "min_score": float(anomaly_scores.min()),
                            "max_score": float(anomaly_scores.max()),
                        },
                        detection_method="isolation_forest",
                        confidence=min(float(abs(anomaly_scores[idx])) / 0.5, 1.0),
                    )

                    anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Isolation forest detection failed for {symbol}: {e}")

        logger.info(
            f"Isolation Forest detection found {len(anomalies)} anomalies for {symbol}"
        )
        return anomalies

    def _detect_statistical_anomalies(
        self, df: pd.DataFrame, symbol: str
    ) -> List[AnomalyDetection]:
        """基于统计方法的异常检测

        Args:
            df: 价格数据
            symbol: 股票代码

        Returns:
            异常检测结果列表
        """
        anomalies = []

        try:
            # 布林带突破检测
            bollinger_anomalies = self._detect_bollinger_band_anomalies(df, symbol)
            anomalies.extend(bollinger_anomalies)

            # 价格跳空检测
            gap_anomalies = self._detect_gap_anomalies(df, symbol)
            anomalies.extend(gap_anomalies)

            # 成交量激增检测
            volume_spike_anomalies = self._detect_volume_spike_anomalies(df, symbol)
            anomalies.extend(volume_spike_anomalies)

        except Exception as e:
            logger.error(f"Statistical anomaly detection failed for {symbol}: {e}")

        logger.info(
            f"Statistical detection found {len(anomalies)} anomalies for {symbol}"
        )
        return anomalies

    def _detect_bollinger_band_anomalies(
        self, df: pd.DataFrame, symbol: str
    ) -> List[AnomalyDetection]:
        """检测布林带突破异常"""
        anomalies = []

        # 检测突破布林带的情况
        bb_breaks = df[(df["close"] > df["bb_upper"]) | (df["close"] < df["bb_lower"])]

        for _, row in bb_breaks.iterrows():
            if row["close"] > row["bb_upper"]:
                direction = "上轨"
                distance = (row["close"] - row["bb_upper"]) / row["bb_upper"]
            else:
                direction = "下轨"
                distance = (row["bb_lower"] - row["close"]) / row["bb_lower"]

            anomaly = AnomalyDetection(
                symbol=symbol,
                timestamp=row["date"],
                anomaly_type="bollinger_break",
                anomaly_score=float(abs(distance)),
                description=f"突破布林带{direction}：距离{abs(distance) * 100:.2f}%",
                data_point={
                    "date": row["date"].isoformat()
                    if hasattr(row["date"], "isoformat")
                    else str(row["date"]),
                    "close": float(row["close"]),
                    "bb_upper": float(row["bb_upper"]),
                    "bb_lower": float(row["bb_lower"]),
                    "direction": direction,
                    "distance_pct": float(abs(distance) * 100),
                },
                threshold_values={
                    "bb_upper": float(row["bb_upper"]),
                    "bb_lower": float(row["bb_lower"]),
                    "bb_width": float(row["bb_upper"] - row["bb_lower"]),
                },
                detection_method="bollinger_bands",
                confidence=min(float(abs(distance)) * 10, 1.0),
            )

            anomalies.append(anomaly)

        return anomalies

    def _detect_gap_anomalies(
        self, df: pd.DataFrame, symbol: str
    ) -> List[AnomalyDetection]:
        """检测价格跳空异常"""
        anomalies = []

        # 检测显著跳空（大于3%）
        significant_gaps = df[abs(df["gap"]) > 0.03]

        for _, row in significant_gaps.iterrows():
            gap_pct = row["gap"] * 100
            direction = "上" if gap_pct > 0 else "下"

            anomaly = AnomalyDetection(
                symbol=symbol,
                timestamp=row["date"],
                anomaly_type="price_gap",
                anomaly_score=float(abs(gap_pct)),
                description=f"价格跳空{direction}：{abs(gap_pct):.2f}%",
                data_point={
                    "date": row["date"].isoformat()
                    if hasattr(row["date"], "isoformat")
                    else str(row["date"]),
                    "open": float(row["open"]),
                    "close": float(row["close"]),
                    "prev_close": float(row["close"] - row["gap"] * row["close"])
                    if row["gap"] != 0
                    else float(row["close"]),
                    "gap_pct": float(gap_pct),
                    "direction": direction,
                },
                threshold_values={"gap_threshold": 3.0},
                detection_method="price_gap",
                confidence=min(float(abs(gap_pct)) / 10.0, 1.0),
            )

            anomalies.append(anomaly)

        return anomalies

    def _detect_volume_spike_anomalies(
        self, df: pd.DataFrame, symbol: str
    ) -> List[AnomalyDetection]:
        """检测成交量激增异常"""
        anomalies = []

        # 计算成交量移动平均
        df["volume_ma_20"] = df["volume"].rolling(window=20).mean()

        # 检测成交量激增（超过20日均量的3倍）
        volume_spikes = df[
            (df["volume"] > df["volume_ma_20"] * 3) & (df["volume_ma_20"].notna())
        ]

        for _, row in volume_spikes.iterrows():
            volume_ratio = row["volume"] / row["volume_ma_20"]

            anomaly = AnomalyDetection(
                symbol=symbol,
                timestamp=row["date"],
                anomaly_type="volume_spike",
                anomaly_score=float(volume_ratio),
                description=f"成交量激增：{volume_ratio:.2f}倍于20日均量",
                data_point={
                    "date": row["date"].isoformat()
                    if hasattr(row["date"], "isoformat")
                    else str(row["date"]),
                    "volume": int(row["volume"]),
                    "volume_ma_20": float(row["volume_ma_20"]),
                    "volume_ratio": float(volume_ratio),
                    "close": float(row["close"]),
                },
                threshold_values={
                    "volume_spike_threshold": 3.0,
                    "volume_ma_20": float(row["volume_ma_20"]),
                },
                detection_method="volume_spike",
                confidence=min(float(volume_ratio) / 10.0, 1.0),
            )

            anomalies.append(anomaly)

        return anomalies

    def _save_anomaly_to_db(self, anomaly: AnomalyDetection) -> bool:
        """保存异常检测结果到数据库

        Args:
            anomaly: 异常检测结果

        Returns:
            是否保存成功
        """
        try:
            anomaly_data = {
                "symbol": anomaly.symbol,
                "anomaly_type": anomaly.anomaly_type,
                "anomaly_score": anomaly.anomaly_score,
                "description": anomaly.description,
                "detection_method": anomaly.detection_method,
                "timestamp": anomaly.timestamp,
                "data_point": anomaly.data_point,
                "threshold_values": anomaly.threshold_values,
            }

            return self.analysis_db.save_anomaly_detection(anomaly_data)

        except Exception as e:
            logger.error(f"Failed to save anomaly to database: {e}")
            return False

    def get_recent_anomalies(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取最近的异常检测结果

        Args:
            symbol: 股票代码（可选）
            limit: 返回数量限制

        Returns:
            异常检测结果列表
        """
        return self.analysis_db.get_anomaly_detections(symbol, limit)

    def get_anomaly_statistics(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """获取异常统计信息

        Args:
            symbol: 股票代码
            days: 统计天数

        Returns:
            异常统计信息
        """
        try:
            # 获取指定时间范围内的异常
            anomalies = self.get_recent_anomalies(symbol, limit=1000)

            # 过滤指定天数内的异常
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_anomalies = [
                a
                for a in anomalies
                if datetime.fromisoformat(a["timestamp"].replace("Z", "+00:00"))
                > cutoff_date
            ]

            # 统计信息
            stats = {
                "symbol": symbol,
                "total_anomalies": len(recent_anomalies),
                "anomaly_types": {},
                "detection_methods": {},
                "avg_anomaly_score": 0.0,
                "days_analyzed": days,
            }

            if recent_anomalies:
                # 按类型统计
                for anomaly in recent_anomalies:
                    anomaly_type = anomaly["anomaly_type"]
                    method = anomaly["detection_method"]

                    stats["anomaly_types"][anomaly_type] = (
                        stats["anomaly_types"].get(anomaly_type, 0) + 1
                    )
                    stats["detection_methods"][method] = (
                        stats["detection_methods"].get(method, 0) + 1
                    )

                # 平均异常分数
                total_score = sum(a["anomaly_score"] for a in recent_anomalies)
                stats["avg_anomaly_score"] = total_score / len(recent_anomalies)

            return stats

        except Exception as e:
            logger.error(f"Failed to get anomaly statistics for {symbol}: {e}")


# 便捷函数
async def detect_price_anomalies_batch(
    symbols: List[str], lookback_days: int = 60, methods: List[str] = None
) -> Dict[str, List[AnomalyDetection]]:
    """批量检测股价异常

    Args:
        symbols: 股票代码列表
        lookback_days: 回看天数
        methods: 检测方法列表

    Returns:
        异常检测结果字典
    """
    detector = PriceAnomalyDetector()
    return await detector.detect_price_anomalies(symbols, lookback_days, methods)


def create_price_anomaly_detector(
    data_collector: Optional[AkshareDataCollector] = None,
) -> PriceAnomalyDetector:
    """创建股价异常检测器

    Args:
        data_collector: 数据收集器

    Returns:
        股价异常检测器实例
    """
    return PriceAnomalyDetector(data_collector)
