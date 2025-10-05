"""
数据验证器模块
负责数据质量检查和验证
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.constants import DATA_QUALITY_THRESHOLD
from common.exceptions import DataError
from common.logging_system import setup_logger

# 如果DATA_QUALITY_THRESHOLD不存在，使用默认值
try:
    from common.constants import DATA_QUALITY_THRESHOLD
except ImportError:
    DATA_QUALITY_THRESHOLD = 0.7  # 默认质量门限

logger = setup_logger("data_validator")


@dataclass
class ValidationResult:
    """验证结果"""

    is_valid: bool
    quality_score: float
    issues: List[str]
    statistics: Dict[str, Any]


@dataclass
class DataQualityMetrics:
    """数据质量指标"""

    completeness: float  # 完整性 0-1
    accuracy: float  # 准确性 0-1
    consistency: float  # 一致性 0-1
    timeliness: float  # 及时性 0-1
    uniqueness: float  # 唯一性 0-1

    @property
    def overall_score(self) -> float:
        """计算总体质量分数"""
        return (
            self.completeness * 0.3
            + self.accuracy * 0.25
            + self.consistency * 0.2
            + self.timeliness * 0.15
            + self.uniqueness * 0.1
        )


class DataValidator:
    """数据验证器"""

    # 验证规则
    VALIDATION_RULES = {
        "price_range": {"min": 0.01, "max": 100000},
        "volume_range": {"min": 0, "max": 1e12},
        "missing_threshold": 0.05,  # 5%缺失率阈值
        "outlier_std": 4,  # 4个标准差之外视为异常值
        "duplicate_threshold": 0.01,  # 1%重复率阈值
    }

    def __init__(self):
        """初始化数据验证器"""
        self.validation_cache: Dict[str, ValidationResult] = {}

    def validate_market_data(
        self, df: pd.DataFrame, symbol: Optional[str] = None
    ) -> ValidationResult:
        """验证市场数据

        Args:
            df: 市场数据DataFrame
            symbol: 股票代码（可选）

        Returns:
            验证结果
        """
        issues = []
        statistics = {}

        # 基本检查
        if df.empty:
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                issues=["Empty DataFrame"],
                statistics={},
            )

        # 检查必需列
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")

        # 计算质量指标
        metrics = self._calculate_quality_metrics(df)
        statistics["quality_metrics"] = metrics

        # 检查缺失值
        missing_check = self._check_missing_values(df)
        if missing_check["has_issues"]:
            issues.extend(missing_check["issues"])
        statistics["missing_values"] = missing_check["stats"]

        # 检查异常值
        outlier_check = self._check_outliers(df)
        if outlier_check["has_issues"]:
            issues.extend(outlier_check["issues"])
        statistics["outliers"] = outlier_check["stats"]

        # 检查数据一致性
        consistency_check = self._check_consistency(df)
        if consistency_check["has_issues"]:
            issues.extend(consistency_check["issues"])
        statistics["consistency"] = consistency_check["stats"]

        # 检查重复值
        duplicate_check = self._check_duplicates(df)
        if duplicate_check["has_issues"]:
            issues.extend(duplicate_check["issues"])
        statistics["duplicates"] = duplicate_check["stats"]

        # 计算总体质量分数
        quality_score = metrics.overall_score
        is_valid = quality_score >= DATA_QUALITY_THRESHOLD and len(issues) == 0

        result = ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            statistics=statistics,
        )

        # 缓存结果
        if symbol:
            self.validation_cache[symbol] = result

        logger.info(
            f"Validation completed. Score: {quality_score:.2f}, Valid: {is_valid}"
        )
        return result

    def validate_data_integrity(self, df: pd.DataFrame) -> bool:
        """验证数据完整性

        Args:
            df: 数据DataFrame

        Returns:
            是否完整
        """
        # 检查时间序列连续性
        if "timestamp" in df.columns or df.index.name == "timestamp":
            timestamps = df.index if df.index.name == "timestamp" else df["timestamp"]

            # 检查是否排序
            if not timestamps.is_monotonic_increasing:
                logger.warning("Timestamps are not sorted")
                return False

            # 检查间隔一致性
            diffs = pd.Series(timestamps).diff()
            if len(diffs.value_counts()) > 2:  # 允许有一个不同的间隔（首尾）
                logger.warning("Inconsistent time intervals detected")
                return False

        return True

    def detect_outliers_online(
        self, value: float, history: List[float], method: str = "zscore"
    ) -> bool:
        """在线异常值检测

        Args:
            value: 待检测值
            history: 历史值列表
            method: 检测方法 ('zscore', 'iqr', 'isolation')

        Returns:
            是否为异常值
        """
        if len(history) < 30:  # 需要足够的历史数据
            return False

        history_array = np.array(history)

        if method == "zscore":
            mean = np.mean(history_array)
            std = np.std(history_array)
            if std == 0:
                return False
            z_score = abs((value - mean) / std)
            return z_score > self.VALIDATION_RULES["outlier_std"]

        elif method == "iqr":
            q1 = np.percentile(history_array, 25)
            q3 = np.percentile(history_array, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return value < lower_bound or value > upper_bound

        else:
            raise ValueError(f"Unknown method: {method}")

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> DataQualityMetrics:
        """计算数据质量指标

        Args:
            df: 数据DataFrame

        Returns:
            质量指标对象
        """
        # 完整性：非空值比例
        completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))

        # 准确性：检查价格关系
        accuracy = 1.0
        if all(col in df.columns for col in ["high", "low", "close"]):
            invalid_prices = (
                (df["high"] < df["low"])
                | (df["close"] > df["high"])
                | (df["close"] < df["low"])
            )
            accuracy = 1 - (invalid_prices.sum() / len(df))

        # 一致性：检查数据格式和范围
        consistency = 1.0
        if "volume" in df.columns:
            invalid_volume = df["volume"] < 0
            consistency = 1 - (invalid_volume.sum() / len(df))

        # 及时性：检查最新数据时间
        timeliness = 1.0
        if "timestamp" in df.columns or df.index.name == "timestamp":
            timestamps = df.index if df.index.name == "timestamp" else df["timestamp"]
            latest = pd.to_datetime(timestamps).max()
            age_hours = (datetime.now() - latest).total_seconds() / 3600
            if age_hours < 1:
                timeliness = 1.0
            elif age_hours < 24:
                timeliness = 0.8
            else:
                timeliness = max(0.5, 1 - age_hours / 168)  # 一周后降到0.5

        # 唯一性：检查重复记录
        uniqueness = len(df.drop_duplicates()) / len(df) if len(df) > 0 else 1.0

        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            uniqueness=uniqueness,
        )

    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查缺失值

        Args:
            df: 数据DataFrame

        Returns:
            检查结果
        """
        result = {"has_issues": False, "issues": [], "stats": {}}

        missing_counts = df.isnull().sum()
        missing_pct = missing_counts / len(df)

        result["stats"] = {
            "total_missing": int(missing_counts.sum()),
            "missing_by_column": missing_counts.to_dict(),
            "missing_pct_by_column": missing_pct.to_dict(),
        }

        # 检查是否超过阈值
        for col, pct in missing_pct.items():
            if pct > self.VALIDATION_RULES["missing_threshold"]:
                result["has_issues"] = True
                result["issues"].append(f"Column '{col}' has {pct:.1%} missing values")

        return result

    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查异常值

        Args:
            df: 数据DataFrame

        Returns:
            检查结果
        """
        result = {"has_issues": False, "issues": [], "stats": {}}

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}

        for col in numeric_columns:
            if col in ["open", "high", "low", "close", "volume"]:
                # 使用Z-score方法
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > self.VALIDATION_RULES["outlier_std"]
                outlier_counts[col] = int(outliers.sum())

                if outlier_counts[col] > 0:
                    result["has_issues"] = True
                    result["issues"].append(
                        f"Column '{col}' has {outlier_counts[col]} outliers"
                    )

        result["stats"] = {
            "outlier_counts": outlier_counts,
            "total_outliers": sum(outlier_counts.values()),
        }

        return result

    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查数据一致性

        Args:
            df: 数据DataFrame

        Returns:
            检查结果
        """
        result = {"has_issues": False, "issues": [], "stats": {}}

        # 检查OHLC关系
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            invalid_high_low = df["high"] < df["low"]
            invalid_close_high = df["close"] > df["high"]
            invalid_close_low = df["close"] < df["low"]
            invalid_open_high = df["open"] > df["high"]
            invalid_open_low = df["open"] < df["low"]

            total_invalid = (
                invalid_high_low.sum()
                + invalid_close_high.sum()
                + invalid_close_low.sum()
                + invalid_open_high.sum()
                + invalid_open_low.sum()
            )

            if total_invalid > 0:
                result["has_issues"] = True
                result["issues"].append(
                    f"Found {total_invalid} OHLC consistency violations"
                )

            result["stats"]["ohlc_violations"] = int(total_invalid)

        # 检查价格范围
        if "close" in df.columns:
            price_range = self.VALIDATION_RULES["price_range"]
            invalid_prices = (df["close"] < price_range["min"]) | (
                df["close"] > price_range["max"]
            )

            if invalid_prices.any():
                result["has_issues"] = True
                result["issues"].append(
                    f"Found {invalid_prices.sum()} prices outside valid range"
                )

            result["stats"]["invalid_prices"] = int(invalid_prices.sum())

        return result

    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查重复值

        Args:
            df: 数据DataFrame

        Returns:
            检查结果
        """
        result = {"has_issues": False, "issues": [], "stats": {}}

        duplicates = df.duplicated()
        duplicate_count = duplicates.sum()
        duplicate_pct = duplicate_count / len(df) if len(df) > 0 else 0

        result["stats"] = {
            "duplicate_count": int(duplicate_count),
            "duplicate_percentage": duplicate_pct,
        }

        if duplicate_pct > self.VALIDATION_RULES["duplicate_threshold"]:
            result["has_issues"] = True
            result["issues"].append(f"Found {duplicate_pct:.1%} duplicate records")

        return result


# 模块级别函数
def validate_dataframe(df: pd.DataFrame, data_type: str = "market") -> ValidationResult:
    """验证DataFrame的便捷函数

    Args:
        df: 待验证的DataFrame
        data_type: 数据类型

    Returns:
        验证结果
    """
    validator = DataValidator()

    if data_type == "market":
        return validator.validate_market_data(df)
    else:
        # 可以扩展其他数据类型的验证
        return validator.validate_market_data(df)


def ensure_data_quality(
    df: pd.DataFrame, min_quality_score: float = DATA_QUALITY_THRESHOLD
) -> pd.DataFrame:
    """确保数据质量

    Args:
        df: 输入DataFrame
        min_quality_score: 最小质量分数

    Returns:
        清理后的DataFrame

    Raises:
        DataError: 数据质量不满足要求
    """
    result = validate_dataframe(df)

    if result.quality_score < min_quality_score:
        raise DataError(
            f"Data quality score {result.quality_score:.2f} below threshold {min_quality_score}. "
            f"Issues: {result.issues}"
        )

    # 返回原始数据（可以在这里添加清理逻辑）
    return df
