"""
Correlation Analyzer
相关性分析器，用于分析股票之间的相关性关系
"""

import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from common.exceptions import DataError
from common.logging_system import setup_logger
from module_01_data_pipeline import AkshareDataCollector, get_database_manager

from ..storage_management import get_market_analysis_db

logger = setup_logger("correlation_analyzer")


@dataclass
class CorrelationResult:
    """相关性分析结果"""

    analysis_id: str
    symbols: List[str]
    correlation_matrix: pd.DataFrame
    correlation_type: str  # pearson, spearman, kendall
    time_window: int
    analysis_date: datetime
    insights: List[str]
    significant_pairs: List[Tuple[str, str, float]]  # (symbol1, symbol2, correlation)
    clustering_results: Optional[Dict[str, Any]] = None
    network_metrics: Optional[Dict[str, Any]] = None
    # Keep original fields for compatibility
    significant_correlations: Optional[List[Dict[str, Any]]] = None
    average_correlation: Optional[float] = None
    max_correlation: Optional[float] = None
    min_correlation: Optional[float] = None
    analysis_timestamp: Optional[pd.Timestamp] = None


class CorrelationAnalyzer:
    """相关性分析器"""

    def __init__(
        self,
        data_collector: Optional[AkshareDataCollector] = None,
        significance_threshold: float = 0.7,
    ):
        """初始化相关性分析器

        Args:
            data_collector: 数据收集器
            significance_threshold: 显著性阈值
        """
        self.data_collector = data_collector or AkshareDataCollector(rate_limit=0.5)
        self.db_manager = get_database_manager()
        self.analysis_db = get_market_analysis_db()
        self.scaler = StandardScaler()

        # 相关性分析参数
        self.significance_threshold = significance_threshold
        self.strong_correlation_threshold = 0.7  # 强相关阈值
        self.moderate_correlation_threshold = 0.5  # 中等相关阈值

        logger.info("Initialized correlation analyzer")

    async def analyze_correlations(
        self,
        symbols: List[str],
        lookback_days: int = 252,  # 默认一年
        correlation_types: List[str] = None,
        data_frequency: str = "daily",
    ) -> Dict[str, CorrelationResult]:
        """分析股票间相关性

        Args:
            symbols: 股票代码列表
            lookback_days: 回看天数
            correlation_types: 相关性类型列表 ['pearson', 'spearman', 'kendall']
            data_frequency: 数据频率 ('daily', 'weekly', 'monthly')

        Returns:
            相关性分析结果字典
        """
        if correlation_types is None:
            correlation_types = ["pearson", "spearman"]

        logger.info(
            f"Starting correlation analysis for {len(symbols)} symbols using {correlation_types}"
        )

        results = {}

        try:
            # 获取数据
            price_data = await self._get_price_data(symbols, lookback_days)

            if price_data.empty:
                logger.warning("No price data available for correlation analysis")
                return results

            # 按频率重采样
            if data_frequency != "daily":
                price_data = self._resample_data(price_data, data_frequency)

            # 计算收益率
            returns_data = self._calculate_returns(price_data)

            if returns_data.empty:
                logger.warning("No valid returns data for correlation analysis")
                return results

            # 执行各种相关性分析
            for corr_type in correlation_types:
                try:
                    result = self._perform_correlation_analysis(
                        returns_data, symbols, corr_type, lookback_days
                    )

                    if result:
                        results[corr_type] = result

                        # 保存结果到数据库
                        self._save_correlation_result(result)

                        logger.info(f"Completed {corr_type} correlation analysis")

                except Exception as e:
                    logger.error(
                        f"Failed to perform {corr_type} correlation analysis: {e}"
                    )
                    continue

        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")

        return results

    def calculate_correlation_matrix(
        self, data: pd.DataFrame, method: str = "pearson"
    ) -> pd.DataFrame:
        """计算相关性矩阵

        Args:
            data: 价格数据 (股票 x 时间)
            method: 相关性计算方法 ('pearson', 'spearman', 'kendall')

        Returns:
            相关性矩阵
        """
        try:
            # 计算收益率
            returns = data.pct_change().dropna()

            # 计算相关性矩阵
            correlation_matrix = returns.corr(method=method)

            logger.info(
                f"Calculated {method} correlation matrix for {len(correlation_matrix)} assets"
            )
            return correlation_matrix

        except Exception as e:
            logger.error(f"Failed to calculate correlation matrix: {e}")
            raise DataError(f"Correlation matrix calculation failed: {e}")

    def comprehensive_analysis(
        self, data: pd.DataFrame, sector_mapping: Optional[Dict[str, str]] = None
    ) -> CorrelationResult:
        """综合分析

        Args:
            data: 价格数据
            sector_mapping: 行业映射

        Returns:
            综合分析结果
        """
        try:
            # 计算相关性矩阵
            correlation_matrix = self.calculate_correlation_matrix(data)

            # 找出高相关性对
            significant_correlations = self.find_highly_correlated_pairs(
                correlation_matrix
            )

            # 计算统计指标
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            correlations = upper_triangle.stack().dropna()

            # Convert to expected format
            significant_pairs = [
                (pair["asset1"], pair["asset2"], pair["correlation"])
                for pair in significant_correlations
            ]

            result = CorrelationResult(
                analysis_id=f"corr_{uuid.uuid4().hex[:8]}",
                symbols=list(correlation_matrix.columns),
                correlation_matrix=correlation_matrix,
                correlation_type="pearson",
                time_window=len(data),
                analysis_date=datetime.now(),
                insights=[f"Average correlation: {correlations.mean():.3f}"],
                significant_pairs=significant_pairs,
                # Legacy fields for compatibility
                significant_correlations=significant_correlations,
                average_correlation=correlations.mean(),
                max_correlation=correlations.max(),
                min_correlation=correlations.min(),
                analysis_timestamp=pd.Timestamp.now(),
            )

            logger.info("Completed comprehensive correlation analysis")
            return result

        except Exception as e:
            logger.error(f"Failed to perform comprehensive analysis: {e}")
            raise DataError(f"Comprehensive correlation analysis failed: {e}")

    def find_highly_correlated_pairs(
        self, correlation_matrix: pd.DataFrame, threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """找出高相关性股票对

        Args:
            correlation_matrix: 相关性矩阵
            threshold: 相关性阈值

        Returns:
            高相关性股票对列表
        """
        if threshold is None:
            threshold = self.significance_threshold

        try:
            highly_correlated = []

            # 获取上三角矩阵（避免重复）
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )

            # 找出高相关性对
            for i in range(len(upper_triangle.columns)):
                for j in range(i + 1, len(upper_triangle.columns)):
                    corr_value = upper_triangle.iloc[i, j]

                    if not pd.isna(corr_value) and abs(corr_value) >= threshold:
                        asset1 = upper_triangle.columns[i]
                        asset2 = upper_triangle.columns[j]

                        highly_correlated.append(
                            {
                                "asset1": asset1,
                                "asset2": asset2,
                                "correlation": corr_value,
                                "abs_correlation": abs(corr_value),
                                "type": "positive" if corr_value > 0 else "negative",
                            }
                        )

            # 按绝对相关性排序
            highly_correlated.sort(key=lambda x: x["abs_correlation"], reverse=True)

            logger.info(
                f"Found {len(highly_correlated)} highly correlated pairs (threshold: {threshold})"
            )
            return highly_correlated

        except Exception as e:
            logger.error(f"Failed to find highly correlated pairs: {e}")
            raise DataError(f"High correlation analysis failed: {e}")

    async def _get_price_data(
        self, symbols: List[str], lookback_days: int
    ) -> pd.DataFrame:
        """获取价格数据"""
        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=lookback_days + 30)).strftime(
                "%Y%m%d"
            )

            all_price_data = []

            for symbol in symbols:
                try:
                    price_data = self.data_collector.fetch_stock_history(
                        symbol, start_date, end_date
                    )

                    if not price_data.empty:
                        # 只保留需要的列
                        price_data = price_data[["date", "close"]].copy()
                        price_data.columns = ["date", symbol]
                        price_data["date"] = pd.to_datetime(price_data["date"])
                        price_data.set_index("date", inplace=True)
                        all_price_data.append(price_data)

                except Exception as e:
                    logger.warning(f"Failed to get price data for {symbol}: {e}")
                    continue

            if not all_price_data:
                return pd.DataFrame()

            # 合并数据
            combined_data = pd.concat(all_price_data, axis=1, join="inner")
            combined_data = combined_data.dropna()

            # 只保留最近的lookback_days天数据
            if len(combined_data) > lookback_days:
                combined_data = combined_data.tail(lookback_days)

            logger.info(
                f"Retrieved price data for {len(combined_data.columns)} symbols over {len(combined_data)} days"
            )
            return combined_data

        except Exception as e:
            logger.error(f"Failed to get price data: {e}")
            return pd.DataFrame()

    def _calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """计算收益率"""
        try:
            returns = price_data.pct_change().dropna()
            logger.info(
                f"Calculated returns for {len(returns.columns)} symbols over {len(returns)} periods"
            )
            return returns
        except Exception as e:
            logger.error(f"Failed to calculate returns: {e}")
            return pd.DataFrame()

    def _perform_correlation_analysis(
        self,
        returns_data: pd.DataFrame,
        symbols: List[str],
        correlation_type: str,
        time_window: int,
    ) -> Optional[CorrelationResult]:
        """执行相关性分析"""
        try:
            # 计算相关性矩阵
            if correlation_type == "pearson":
                corr_matrix = returns_data.corr(method="pearson")
            elif correlation_type == "spearman":
                corr_matrix = returns_data.corr(method="spearman")
            elif correlation_type == "kendall":
                corr_matrix = returns_data.corr(method="kendall")
            else:
                raise ValueError(f"Unsupported correlation type: {correlation_type}")

            # 识别显著相关性
            significant_pairs = self._identify_significant_correlations(
                corr_matrix, correlation_type
            )

            # 生成洞察
            insights = self._generate_correlation_insights(
                corr_matrix, significant_pairs
            )

            result = CorrelationResult(
                analysis_id=f"corr_{uuid.uuid4().hex[:8]}",
                symbols=list(corr_matrix.columns),
                correlation_matrix=corr_matrix,
                correlation_type=correlation_type,
                time_window=time_window,
                analysis_date=datetime.now(),
                insights=insights,
                significant_pairs=significant_pairs,
            )

            return result

        except Exception as e:
            logger.error(f"Failed to perform correlation analysis: {e}")
            return None

    def _identify_significant_correlations(
        self, corr_matrix: pd.DataFrame, correlation_type: str
    ) -> List[Tuple[str, str, float]]:
        """识别显著相关性"""
        significant_pairs = []

        try:
            # 遍历相关性矩阵的上三角
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    symbol1 = corr_matrix.columns[i]
                    symbol2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]

                    # 检查是否显著相关
                    if abs(corr_value) >= self.moderate_correlation_threshold:
                        significant_pairs.append((symbol1, symbol2, corr_value))

            # 按相关性强度排序
            significant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

            return significant_pairs

        except Exception as e:
            logger.error(f"Failed to identify significant correlations: {e}")
            return []

    def _generate_correlation_insights(
        self, corr_matrix: pd.DataFrame, significant_pairs: List[Tuple[str, str, float]]
    ) -> List[str]:
        """生成相关性洞察"""
        insights = []

        try:
            # 整体相关性统计
            corr_values = corr_matrix.values
            upper_triangle = np.triu(corr_values, k=1)
            non_zero_corrs = upper_triangle[upper_triangle != 0]

            if len(non_zero_corrs) > 0:
                avg_correlation = np.mean(np.abs(non_zero_corrs))
                insights.append(f"平均相关性系数: {avg_correlation:.3f}")

            # 强相关性对
            strong_pairs = [
                pair
                for pair in significant_pairs
                if abs(pair[2]) >= self.strong_correlation_threshold
            ]
            if strong_pairs:
                insights.append(f"检测到 {len(strong_pairs)} 对强相关股票")

            return insights
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return []

    def _save_correlation_result(self, result: CorrelationResult) -> bool:
        """保存相关性分析结果到数据库"""
        try:
            correlation_data = {
                "analysis_id": result.analysis_id,
                "symbols": result.symbols,
                "correlation_matrix": result.correlation_matrix.to_dict(),
                "correlation_type": result.correlation_type,
                "time_window": result.time_window,
                "analysis_date": result.analysis_date,
                "insights": result.insights,
            }

            return self.analysis_db.save_correlation_analysis(correlation_data)

        except Exception as e:
            logger.error(f"Failed to save correlation result: {e}")
            return False


# 便捷函数
async def analyze_market_correlation(
    symbols: List[str], lookback_days: int = 252, correlation_types: List[str] = None
) -> Dict[str, CorrelationResult]:
    """分析市场相关性

    Args:
        symbols: 股票代码列表
        lookback_days: 回看天数
        correlation_types: 相关性类型列表

    Returns:
        相关性分析结果字典
    """
    analyzer = CorrelationAnalyzer()
    return await analyzer.analyze_correlations(
        symbols, lookback_days, correlation_types
    )


def create_correlation_analyzer(
    data_collector: Optional[AkshareDataCollector] = None,
) -> CorrelationAnalyzer:
    """创建相关性分析器

    Args:
        data_collector: 数据收集器

    Returns:
        相关性分析器实例
    """
    return CorrelationAnalyzer(data_collector)
