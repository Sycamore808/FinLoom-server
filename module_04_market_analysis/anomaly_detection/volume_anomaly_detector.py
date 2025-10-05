"""
Volume Anomaly Detector
成交量异常检测器
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
from .price_anomaly_detector import AnomalyDetection

logger = setup_logger("volume_anomaly_detector")


class VolumeAnomalyDetector:
    """成交量异常检测器"""

    def __init__(self, data_collector: Optional[AkshareDataCollector] = None):
        """初始化成交量异常检测器

        Args:
            data_collector: 数据收集器
        """
        self.data_collector = data_collector or AkshareDataCollector(rate_limit=0.5)
        self.db_manager = get_database_manager()
        self.analysis_db = get_market_analysis_db()
        self.scaler = StandardScaler()

        # 成交量异常检测参数
        self.volume_spike_threshold = 3.0  # 成交量激增阈值（倍数）
        self.volume_dry_threshold = 0.3  # 成交量萎缩阈值（倍数）
        self.z_score_threshold = 3.0

        logger.info("Initialized volume anomaly detector")

    async def detect_volume_anomalies(
        self, symbols: List[str], lookback_days: int = 60, methods: List[str] = None
    ) -> Dict[str, List[AnomalyDetection]]:
        """检测成交量异常

        Args:
            symbols: 股票代码列表
            lookback_days: 回看天数
            methods: 检测方法列表 ['spike', 'dry_up', 'statistical', 'pattern']

        Returns:
            异常检测结果字典
        """
        if methods is None:
            methods = ["spike", "dry_up", "statistical", "pattern"]

        logger.info(
            f"Starting volume anomaly detection for {len(symbols)} symbols using methods: {methods}"
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

                # 预处理数据
                volume_data = self._preprocess_volume_data(price_data)

                if len(volume_data) < 20:
                    logger.warning(
                        f"Insufficient data for {symbol}: {len(volume_data)} points"
                    )
                    results[symbol] = []
                    continue

                # 执行各种成交量异常检测方法
                symbol_anomalies = []

                if "spike" in methods:
                    spike_anomalies = self._detect_volume_spikes(volume_data, symbol)
                    symbol_anomalies.extend(spike_anomalies)

                if "dry_up" in methods:
                    dry_anomalies = self._detect_volume_dry_up(volume_data, symbol)
                    symbol_anomalies.extend(dry_anomalies)

                if "statistical" in methods:
                    statistical_anomalies = self._detect_statistical_volume_anomalies(
                        volume_data, symbol
                    )
                    symbol_anomalies.extend(statistical_anomalies)

                if "pattern" in methods:
                    pattern_anomalies = self._detect_volume_pattern_anomalies(
                        volume_data, symbol
                    )
                    symbol_anomalies.extend(pattern_anomalies)

                # 保存异常检测结果到数据库
                for anomaly in symbol_anomalies:
                    self._save_anomaly_to_db(anomaly)

                results[symbol] = symbol_anomalies

                logger.info(
                    f"Detected {len(symbol_anomalies)} volume anomalies for {symbol}"
                )

            except Exception as e:
                logger.error(f"Failed to detect volume anomalies for {symbol}: {e}")
                results[symbol] = []

        return results

    def _preprocess_volume_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理成交量数据

        Args:
            df: 原始价格数据

        Returns:
            预处理后的数据
        """
        # 确保有必要的列
        required_columns = ["date", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
                return pd.DataFrame()

        # 按日期排序
        df = df.sort_values("date").reset_index(drop=True)

        # 计算成交量相关指标
        df["volume_ma_5"] = df["volume"].rolling(window=5).mean()
        df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ma_60"] = df["volume"].rolling(window=60).mean()

        # 成交量变化率
        df["volume_change"] = df["volume"].pct_change()
        df["volume_ratio_ma20"] = df["volume"] / df["volume_ma_20"]
        df["volume_ratio_ma60"] = df["volume"] / df["volume_ma_60"]

        # 价量关系
        df["price_change"] = df["close"].pct_change()
        df["price_volume_correlation"] = (
            df["price_change"].rolling(window=20).corr(df["volume_change"])
        )

        # 成交量标准差
        df["volume_std_20"] = df["volume"].rolling(window=20).std()
        df["volume_zscore"] = (df["volume"] - df["volume_ma_20"]) / df["volume_std_20"]

        # 删除NaN值
        df = df.dropna().reset_index(drop=True)

        return df

    def _detect_volume_spikes(
        self, df: pd.DataFrame, symbol: str
    ) -> List[AnomalyDetection]:
        """检测成交量激增异常

        Args:
            df: 成交量数据
            symbol: 股票代码

        Returns:
            异常检测结果列表
        """
        anomalies = []

        # 检测成交量激增（超过20日均量的N倍）
        volume_spikes = df[
            (df["volume_ratio_ma20"] > self.volume_spike_threshold)
            & (df["volume_ma_20"].notna())
        ]

        for _, row in volume_spikes.iterrows():
            ratio = row["volume_ratio_ma20"]

            # 判断激增类型
            if ratio > 10:
                spike_level = "超级激增"
            elif ratio > 5:
                spike_level = "强烈激增"
            else:
                spike_level = "适度激增"

            anomaly = AnomalyDetection(
                symbol=symbol,
                timestamp=row["date"],
                anomaly_type="volume_spike",
                anomaly_score=float(ratio),
                description=f"成交量{spike_level}：{ratio:.2f}倍于20日均量",
                data_point={
                    "date": row["date"].isoformat()
                    if hasattr(row["date"], "isoformat")
                    else str(row["date"]),
                    "volume": int(row["volume"]),
                    "volume_ma_20": float(row["volume_ma_20"]),
                    "volume_ratio": float(ratio),
                    "close": float(row["close"]),
                    "price_change": float(row.get("price_change", 0)),
                    "spike_level": spike_level,
                },
                threshold_values={
                    "spike_threshold": self.volume_spike_threshold,
                    "volume_ma_20": float(row["volume_ma_20"]),
                },
                detection_method="volume_spike",
                confidence=min(float(ratio) / 10.0, 1.0),
            )

            anomalies.append(anomaly)

        logger.info(
            f"Volume spike detection found {len(anomalies)} anomalies for {symbol}"
        )
        return anomalies

    def _detect_volume_dry_up(
        self, df: pd.DataFrame, symbol: str
    ) -> List[AnomalyDetection]:
        """检测成交量萎缩异常

        Args:
            df: 成交量数据
            symbol: 股票代码

        Returns:
            异常检测结果列表
        """
        anomalies = []

        # 检测成交量萎缩（低于20日均量的N倍）
        volume_dry_ups = df[
            (df["volume_ratio_ma20"] < self.volume_dry_threshold)
            & (df["volume_ma_20"].notna())
        ]

        for _, row in volume_dry_ups.iterrows():
            ratio = row["volume_ratio_ma20"]

            # 判断萎缩程度
            if ratio < 0.1:
                dry_level = "极度萎缩"
            elif ratio < 0.2:
                dry_level = "严重萎缩"
            else:
                dry_level = "轻度萎缩"

            anomaly = AnomalyDetection(
                symbol=symbol,
                timestamp=row["date"],
                anomaly_type="volume_dry_up",
                anomaly_score=float(1.0 - ratio),  # 萎缩程度
                description=f"成交量{dry_level}：仅为20日均量的{ratio * 100:.1f}%",
                data_point={
                    "date": row["date"].isoformat()
                    if hasattr(row["date"], "isoformat")
                    else str(row["date"]),
                    "volume": int(row["volume"]),
                    "volume_ma_20": float(row["volume_ma_20"]),
                    "volume_ratio": float(ratio),
                    "close": float(row["close"]),
                    "price_change": float(row.get("price_change", 0)),
                    "dry_level": dry_level,
                },
                threshold_values={
                    "dry_threshold": self.volume_dry_threshold,
                    "volume_ma_20": float(row["volume_ma_20"]),
                },
                detection_method="volume_dry_up",
                confidence=float(1.0 - ratio),
            )

            anomalies.append(anomaly)

        logger.info(
            f"Volume dry-up detection found {len(anomalies)} anomalies for {symbol}"
        )
        return anomalies

    def _detect_statistical_volume_anomalies(
        self, df: pd.DataFrame, symbol: str
    ) -> List[AnomalyDetection]:
        """基于统计方法的成交量异常检测

        Args:
            df: 成交量数据
            symbol: 股票代码

        Returns:
            异常检测结果列表
        """
        anomalies = []

        try:
            # Z-Score异常检测
            z_score_anomalies = df[
                (abs(df["volume_zscore"]) > self.z_score_threshold)
                & (df["volume_zscore"].notna())
            ]

            for _, row in z_score_anomalies.iterrows():
                z_score = row["volume_zscore"]
                direction = "高于" if z_score > 0 else "低于"

                anomaly = AnomalyDetection(
                    symbol=symbol,
                    timestamp=row["date"],
                    anomaly_type="volume_zscore",
                    anomaly_score=float(abs(z_score)),
                    description=f"成交量统计异常：Z-Score = {z_score:.2f}，{direction}正常水平",
                    data_point={
                        "date": row["date"].isoformat()
                        if hasattr(row["date"], "isoformat")
                        else str(row["date"]),
                        "volume": int(row["volume"]),
                        "volume_ma_20": float(row["volume_ma_20"]),
                        "volume_std_20": float(row["volume_std_20"]),
                        "z_score": float(z_score),
                        "direction": direction,
                        "close": float(row["close"]),
                    },
                    threshold_values={
                        "z_score_threshold": self.z_score_threshold,
                        "volume_mean": float(row["volume_ma_20"]),
                        "volume_std": float(row["volume_std_20"]),
                    },
                    detection_method="volume_zscore",
                    confidence=min(float(abs(z_score)) / 5.0, 1.0),
                )

                anomalies.append(anomaly)

        except Exception as e:
            logger.error(
                f"Statistical volume anomaly detection failed for {symbol}: {e}"
            )

        logger.info(
            f"Statistical volume detection found {len(anomalies)} anomalies for {symbol}"
        )
        return anomalies

    def _detect_volume_pattern_anomalies(
        self, df: pd.DataFrame, symbol: str
    ) -> List[AnomalyDetection]:
        """检测成交量模式异常

        Args:
            df: 成交量数据
            symbol: 股票代码

        Returns:
            异常检测结果列表
        """
        anomalies = []

        try:
            # 价量背离检测
            price_volume_divergence = self._detect_price_volume_divergence(df, symbol)
            anomalies.extend(price_volume_divergence)

            # 连续异常成交量检测
            consecutive_anomalies = self._detect_consecutive_volume_anomalies(
                df, symbol
            )
            anomalies.extend(consecutive_anomalies)

        except Exception as e:
            logger.error(f"Volume pattern anomaly detection failed for {symbol}: {e}")

        logger.info(
            f"Volume pattern detection found {len(anomalies)} anomalies for {symbol}"
        )
        return anomalies

    def _detect_price_volume_divergence(
        self, df: pd.DataFrame, symbol: str
    ) -> List[AnomalyDetection]:
        """检测价量背离异常"""
        anomalies = []

        # 寻找价量背离点：价格上涨但成交量下降，或价格下跌但成交量上涨
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            price_change = current_row["price_change"]
            volume_change = current_row["volume_change"]

            # 价量背离条件
            is_divergence = False
            divergence_type = ""

            if price_change > 0.02 and volume_change < -0.2:  # 价涨量缩
                is_divergence = True
                divergence_type = "价涨量缩"
            elif price_change < -0.02 and volume_change > 0.2:  # 价跌量增
                is_divergence = True
                divergence_type = "价跌量增"

            if is_divergence:
                divergence_strength = abs(price_change) + abs(volume_change)

                anomaly = AnomalyDetection(
                    symbol=symbol,
                    timestamp=current_row["date"],
                    anomaly_type="price_volume_divergence",
                    anomaly_score=float(divergence_strength),
                    description=f"价量背离：{divergence_type}（价格变化{price_change * 100:.2f}%，成交量变化{volume_change * 100:.2f}%）",
                    data_point={
                        "date": current_row["date"].isoformat()
                        if hasattr(current_row["date"], "isoformat")
                        else str(current_row["date"]),
                        "close": float(current_row["close"]),
                        "volume": int(current_row["volume"]),
                        "price_change": float(price_change),
                        "volume_change": float(volume_change),
                        "divergence_type": divergence_type,
                        "divergence_strength": float(divergence_strength),
                    },
                    threshold_values={"price_threshold": 0.02, "volume_threshold": 0.2},
                    detection_method="price_volume_divergence",
                    confidence=min(float(divergence_strength), 1.0),
                )

                anomalies.append(anomaly)

        return anomalies

    def _detect_consecutive_volume_anomalies(
        self, df: pd.DataFrame, symbol: str
    ) -> List[AnomalyDetection]:
        """检测连续成交量异常"""
        anomalies = []

        # 寻找连续3天以上的成交量异常
        consecutive_count = 0
        start_idx = 0

        for i, row in df.iterrows():
            is_anomaly = (
                row["volume_ratio_ma20"] > self.volume_spike_threshold
                or row["volume_ratio_ma20"] < self.volume_dry_threshold
            )

            if is_anomaly:
                if consecutive_count == 0:
                    start_idx = i
                consecutive_count += 1
            else:
                if consecutive_count >= 3:  # 连续3天以上异常
                    end_row = df.iloc[i - 1]
                    start_row = df.iloc[start_idx]

                    avg_ratio = df.iloc[start_idx:i]["volume_ratio_ma20"].mean()

                    if avg_ratio > self.volume_spike_threshold:
                        anomaly_type = "连续成交量激增"
                        description = f"连续{consecutive_count}天成交量激增，平均{avg_ratio:.2f}倍于均量"
                    else:
                        anomaly_type = "连续成交量萎缩"
                        description = f"连续{consecutive_count}天成交量萎缩，平均仅为均量的{avg_ratio * 100:.1f}%"

                    anomaly = AnomalyDetection(
                        symbol=symbol,
                        timestamp=end_row["date"],
                        anomaly_type="consecutive_volume_anomaly",
                        anomaly_score=float(consecutive_count),
                        description=description,
                        data_point={
                            "date": end_row["date"].isoformat()
                            if hasattr(end_row["date"], "isoformat")
                            else str(end_row["date"]),
                            "start_date": start_row["date"].isoformat()
                            if hasattr(start_row["date"], "isoformat")
                            else str(start_row["date"]),
                            "consecutive_days": consecutive_count,
                            "avg_volume_ratio": float(avg_ratio),
                            "anomaly_type": anomaly_type,
                        },
                        threshold_values={
                            "min_consecutive_days": 3,
                            "spike_threshold": self.volume_spike_threshold,
                            "dry_threshold": self.volume_dry_threshold,
                        },
                        detection_method="consecutive_volume_anomaly",
                        confidence=min(float(consecutive_count) / 7.0, 1.0),
                    )

                    anomalies.append(anomaly)

                consecutive_count = 0

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
            logger.error(f"Failed to save volume anomaly to database: {e}")
            return False

    def get_volume_anomaly_summary(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """获取成交量异常汇总

        Args:
            symbol: 股票代码
            days: 统计天数

        Returns:
            成交量异常汇总信息
        """
        try:
            # 获取异常检测结果
            anomalies = self.analysis_db.get_anomaly_detections(symbol, limit=1000)

            # 过滤成交量相关异常
            volume_anomalies = [
                a for a in anomalies if "volume" in a["anomaly_type"].lower()
            ]

            # 过滤指定天数内的异常
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_anomalies = [
                a
                for a in volume_anomalies
                if datetime.fromisoformat(a["timestamp"].replace("Z", "+00:00"))
                > cutoff_date
            ]

            # 统计信息
            summary = {
                "symbol": symbol,
                "total_volume_anomalies": len(recent_anomalies),
                "spike_count": 0,
                "dry_up_count": 0,
                "divergence_count": 0,
                "consecutive_count": 0,
                "avg_anomaly_score": 0.0,
                "most_recent_anomaly": None,
                "days_analyzed": days,
            }

            if recent_anomalies:
                # 按类型统计
                for anomaly in recent_anomalies:
                    anomaly_type = anomaly["anomaly_type"]
                    if "spike" in anomaly_type:
                        summary["spike_count"] += 1
                    elif "dry" in anomaly_type:
                        summary["dry_up_count"] += 1
                    elif "divergence" in anomaly_type:
                        summary["divergence_count"] += 1
                    elif "consecutive" in anomaly_type:
                        summary["consecutive_count"] += 1

                # 平均异常分数
                total_score = sum(a["anomaly_score"] for a in recent_anomalies)
                summary["avg_anomaly_score"] = total_score / len(recent_anomalies)

                # 最近异常
                recent_anomalies.sort(key=lambda x: x["timestamp"], reverse=True)
                summary["most_recent_anomaly"] = recent_anomalies[0]

            return summary

        except Exception as e:
            logger.error(f"Failed to get volume anomaly summary for {symbol}: {e}")
            return {"error": str(e)}


# 便捷函数
async def detect_volume_anomalies_batch(
    symbols: List[str], lookback_days: int = 60, methods: List[str] = None
) -> Dict[str, List[AnomalyDetection]]:
    """批量检测成交量异常

    Args:
        symbols: 股票代码列表
        lookback_days: 回看天数
        methods: 检测方法列表

    Returns:
        异常检测结果字典
    """
    detector = VolumeAnomalyDetector()
    return await detector.detect_volume_anomalies(symbols, lookback_days, methods)


def create_volume_anomaly_detector(
    data_collector: Optional[AkshareDataCollector] = None,
) -> VolumeAnomalyDetector:
    """创建成交量异常检测器

    Args:
        data_collector: 数据收集器

    Returns:
        成交量异常检测器实例
    """
    return VolumeAnomalyDetector(data_collector)
