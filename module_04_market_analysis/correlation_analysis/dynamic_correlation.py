"""
动态相关性分析器模块
分析资产间的动态相关性和领先滞后关系
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from common.constants import TRADING_DAYS_PER_YEAR
from common.exceptions import DataError
from common.logging_system import setup_logger
from scipy import signal, stats
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

logger = setup_logger("dynamic_correlation")


@dataclass
class CorrelationConfig:
    """相关性分析配置"""

    window_size: int = 60
    min_periods: int = 30
    correlation_method: str = "pearson"  # 'pearson', 'spearman', 'kendall'
    shrinkage_method: str = "ledoit_wolf"  # 'ledoit_wolf', 'empirical'
    lead_lag_max: int = 10
    significance_level: float = 0.05
    rolling_window: bool = True
    exponential_weighting: bool = False
    halflife: int = 30


@dataclass
class CorrelationResult:
    """相关性分析结果"""

    timestamp: datetime
    correlation_matrix: pd.DataFrame
    stability_score: float
    significant_pairs: List[Tuple[str, str, float]]
    network_metrics: Dict[str, float]
    regime_correlation: Dict[str, pd.DataFrame]


@dataclass
class LeadLagRelationship:
    """领先滞后关系"""

    leader: str
    follower: str
    lag: int
    correlation: float
    p_value: float
    confidence: float
    causality_score: float


class DynamicCorrelationAnalyzer:
    """动态相关性分析器"""

    def __init__(self, config: Optional[CorrelationConfig] = None):
        """初始化动态相关性分析器

        Args:
            config: 相关性分析配置
        """
        self.config = config or CorrelationConfig()
        self.correlation_history: List[pd.DataFrame] = []
        self.lead_lag_relationships: List[LeadLagRelationship] = []
        self.correlation_network: Optional[nx.Graph] = None

    def calculate_dynamic_correlation(
        self, returns_df: pd.DataFrame, method: Optional[str] = None
    ) -> pd.DataFrame:
        """计算动态相关性矩阵

        Args:
            returns_df: 收益率DataFrame (index=日期, columns=资产)
            method: 相关性方法

        Returns:
            动态相关性DataFrame
        """
        method = method or self.config.correlation_method

        if self.config.rolling_window:
            # 滚动窗口相关性
            correlation_series = self._calculate_rolling_correlation(returns_df, method)
        else:
            # 扩展窗口相关性
            correlation_series = self._calculate_expanding_correlation(
                returns_df, method
            )

        return correlation_series

    def analyze_cross_asset_correlation(
        self, asset_returns: Dict[str, pd.DataFrame], asset_classes: Dict[str, str]
    ) -> CorrelationResult:
        """分析跨资产相关性

        Args:
            asset_returns: 资产类别到收益率DataFrame的映射
            asset_classes: 资产到类别的映射

        Returns:
            相关性分析结果
        """
        logger.info("Analyzing cross-asset correlations...")

        # 合并所有资产收益率
        all_returns = pd.DataFrame()
        for asset_class, returns_df in asset_returns.items():
            for col in returns_df.columns:
                all_returns[f"{asset_class}_{col}"] = returns_df[col]

        # 计算相关性矩阵
        if self.config.shrinkage_method == "ledoit_wolf":
            corr_matrix = self._calculate_shrunk_correlation(all_returns)
        else:
            corr_matrix = all_returns.corr(method=self.config.correlation_method)

        # 计算稳定性分数
        stability_score = self._calculate_correlation_stability(all_returns)

        # 识别显著相关对
        significant_pairs = self._identify_significant_correlations(
            corr_matrix, all_returns
        )

        # 构建相关性网络
        network_metrics = self._build_correlation_network(corr_matrix)

        # 分析不同市场状态下的相关性
        regime_correlation = self._analyze_regime_correlations(
            all_returns, asset_classes
        )

        result = CorrelationResult(
            timestamp=datetime.now(),
            correlation_matrix=corr_matrix,
            stability_score=stability_score,
            significant_pairs=significant_pairs,
            network_metrics=network_metrics,
            regime_correlation=regime_correlation,
        )

        return result

    def detect_lead_lag_relationships(
        self, returns_df: pd.DataFrame, max_lag: Optional[int] = None
    ) -> List[LeadLagRelationship]:
        """检测领先滞后关系

        Args:
            returns_df: 收益率DataFrame
            max_lag: 最大滞后期数

        Returns:
            领先滞后关系列表
        """
        max_lag = max_lag or self.config.lead_lag_max
        relationships = []

        columns = returns_df.columns
        for i, leader in enumerate(columns):
            for j, follower in enumerate(columns):
                if i >= j:  # 避免重复检测
                    continue

                # 计算交叉相关
                cross_corr = self._calculate_cross_correlation(
                    returns_df[leader], returns_df[follower], max_lag
                )

                # 找到最大相关性的滞后
                best_lag = np.argmax(np.abs(cross_corr)) - max_lag
                best_corr = cross_corr[best_lag + max_lag]

                # 计算显著性
                p_value = self._calculate_correlation_pvalue(best_corr, len(returns_df))

                if p_value < self.config.significance_level:
                    # 计算Granger因果性
                    causality_score = self._test_granger_causality(
                        returns_df[leader], returns_df[follower], abs(best_lag)
                    )

                    relationship = LeadLagRelationship(
                        leader=leader if best_lag < 0 else follower,
                        follower=follower if best_lag < 0 else leader,
                        lag=abs(best_lag),
                        correlation=best_corr,
                        p_value=p_value,
                        confidence=1 - p_value,
                        causality_score=causality_score,
                    )

                    relationships.append(relationship)

        # 按相关性强度排序
        relationships.sort(key=lambda x: abs(x.correlation), reverse=True)
        self.lead_lag_relationships = relationships

        logger.info(f"Detected {len(relationships)} significant lead-lag relationships")
        return relationships

    def calculate_correlation_network_centrality(
        self, correlation_matrix: pd.DataFrame, threshold: float = 0.3
    ) -> pd.DataFrame:
        """计算相关性网络中心性

        Args:
            correlation_matrix: 相关性矩阵
            threshold: 相关性阈值

        Returns:
            中心性度量DataFrame
        """
        # 构建网络
        G = nx.Graph()

        # 添加节点
        for asset in correlation_matrix.columns:
            G.add_node(asset)

        # 添加边（基于相关性）
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    G.add_edge(
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        weight=abs(corr),
                    )

        # 计算中心性度量
        centrality_measures = pd.DataFrame(index=correlation_matrix.columns)

        centrality_measures["degree_centrality"] = pd.Series(nx.degree_centrality(G))
        centrality_measures["betweenness_centrality"] = pd.Series(
            nx.betweenness_centrality(G, weight="weight")
        )
        centrality_measures["closeness_centrality"] = pd.Series(
            nx.closeness_centrality(G, distance="weight")
        )
        centrality_measures["eigenvector_centrality"] = pd.Series(
            nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
        )

        # 计算聚类系数
        centrality_measures["clustering_coefficient"] = pd.Series(
            nx.clustering(G, weight="weight")
        )

        return centrality_measures

    def _calculate_rolling_correlation(
        self, returns_df: pd.DataFrame, method: str
    ) -> pd.DataFrame:
        """计算滚动相关性

        Args:
            returns_df: 收益率DataFrame
            method: 相关性方法

        Returns:
            滚动相关性DataFrame
        """
        window = self.config.window_size
        min_periods = self.config.min_periods

        if self.config.exponential_weighting:
            # 指数加权相关性
            corr_matrix = returns_df.ewm(
                halflife=self.config.halflife, min_periods=min_periods
            ).corr(method=method)
        else:
            # 简单滚动相关性
            corr_matrix = returns_df.rolling(
                window=window, min_periods=min_periods
            ).corr(method=method)

        # 提取最新的相关性矩阵
        latest_date = returns_df.index[-1]
        latest_corr = corr_matrix.loc[latest_date]

        return latest_corr

    def _calculate_expanding_correlation(
        self, returns_df: pd.DataFrame, method: str
    ) -> pd.DataFrame:
        """计算扩展窗口相关性

        Args:
            returns_df: 收益率DataFrame
            method: 相关性方法

        Returns:
            扩展相关性DataFrame
        """
        corr_matrix = returns_df.expanding(min_periods=self.config.min_periods).corr(
            method=method
        )

        # 提取最新的相关性矩阵
        latest_date = returns_df.index[-1]
        latest_corr = corr_matrix.loc[latest_date]

        return latest_corr

    def _calculate_shrunk_correlation(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """计算收缩相关性矩阵

        Args:
            returns_df: 收益率DataFrame

        Returns:
            收缩后的相关性矩阵
        """
        # 使用Ledoit-Wolf收缩估计
        lw = LedoitWolf()
        lw.fit(returns_df.dropna())

        # 转换协方差矩阵为相关性矩阵
        cov_matrix = lw.covariance_
        std_devs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

        return pd.DataFrame(
            corr_matrix, index=returns_df.columns, columns=returns_df.columns
        )

    def _calculate_correlation_stability(self, returns_df: pd.DataFrame) -> float:
        """计算相关性稳定性

        Args:
            returns_df: 收益率DataFrame

        Returns:
            稳定性分数
        """
        # 计算滚动相关性的标准差
        rolling_corr = returns_df.rolling(window=self.config.window_size).corr()

        # 计算相关性的时间序列标准差
        stability_scores = []

        for col1 in returns_df.columns:
            for col2 in returns_df.columns:
                if col1 < col2:
                    pair_corr = rolling_corr.xs(col1, level=1)[col2]
                    if not pair_corr.isna().all():
                        stability = 1 / (1 + pair_corr.std())
                        stability_scores.append(stability)

        return np.mean(stability_scores) if stability_scores else 0.0

    def _identify_significant_correlations(
        self, corr_matrix: pd.DataFrame, returns_df: pd.DataFrame
    ) -> List[Tuple[str, str, float]]:
        """识别显著相关对

        Args:
            corr_matrix: 相关性矩阵
            returns_df: 收益率DataFrame

        Returns:
            显著相关对列表
        """
        significant_pairs = []
        n = len(returns_df)

        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                corr = corr_matrix.iloc[i, j]

                # 计算p值
                p_value = self._calculate_correlation_pvalue(corr, n)

                if p_value < self.config.significance_level:
                    significant_pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], corr)
                    )

        # 按相关性绝对值排序
        significant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return significant_pairs

    def _build_correlation_network(self, corr_matrix: pd.DataFrame) -> Dict[str, float]:
        """构建相关性网络

        Args:
            corr_matrix: 相关性矩阵

        Returns:
            网络度量字典
        """
        # 创建网络
        G = nx.Graph()

        # 添加节点和边
        for i in range(len(corr_matrix)):
            G.add_node(corr_matrix.columns[i])
            for j in range(i + 1, len(corr_matrix)):
                weight = abs(corr_matrix.iloc[i, j])
                if weight > 0.1:  # 只添加有意义的边
                    G.add_edge(
                        corr_matrix.columns[i], corr_matrix.columns[j], weight=weight
                    )

        self.correlation_network = G

        # 计算网络度量
        metrics = {}

        if G.number_of_nodes() > 0:
            metrics["num_nodes"] = G.number_of_nodes()
            metrics["num_edges"] = G.number_of_edges()
            metrics["density"] = nx.density(G)
            metrics["avg_clustering"] = nx.average_clustering(G, weight="weight")

            if nx.is_connected(G):
                metrics["avg_path_length"] = nx.average_shortest_path_length(G)
            else:
                metrics["num_components"] = nx.number_connected_components(G)

        return metrics

    def _analyze_regime_correlations(
        self, returns_df: pd.DataFrame, asset_classes: Dict[str, str]
    ) -> Dict[str, pd.DataFrame]:
        """分析不同市场状态下的相关性

        Args:
            returns_df: 收益率DataFrame
            asset_classes: 资产类别映射

        Returns:
            状态相关性字典
        """
        regime_correlations = {}

        # 简单的市场状态划分（基于市场收益率）
        market_return = returns_df.mean(axis=1)

        # 划分状态
        bull_mask = market_return > market_return.quantile(0.67)
        bear_mask = market_return < market_return.quantile(0.33)
        normal_mask = ~(bull_mask | bear_mask)

        # 计算不同状态下的相关性
        if bull_mask.any():
            regime_correlations["bull"] = returns_df[bull_mask].corr()

        if bear_mask.any():
            regime_correlations["bear"] = returns_df[bear_mask].corr()

        if normal_mask.any():
            regime_correlations["normal"] = returns_df[normal_mask].corr()

        return regime_correlations

    def _calculate_cross_correlation(
        self, series1: pd.Series, series2: pd.Series, max_lag: int
    ) -> np.ndarray:
        """计算交叉相关

        Args:
            series1: 第一个序列
            series2: 第二个序列
            max_lag: 最大滞后

        Returns:
            交叉相关数组
        """
        # 标准化序列
        s1 = (series1 - series1.mean()) / series1.std()
        s2 = (series2 - series2.mean()) / series2.std()

        # 计算交叉相关
        cross_corr = signal.correlate(s1, s2, mode="full", method="auto")

        # 归一化
        cross_corr = cross_corr / len(s1)

        # 提取相关滞后范围
        mid = len(cross_corr) // 2
        return cross_corr[mid - max_lag : mid + max_lag + 1]

    def _calculate_correlation_pvalue(self, correlation: float, n: int) -> float:
        """计算相关系数的p值

        Args:
            correlation: 相关系数
            n: 样本数

        Returns:
            p值
        """
        # Fisher z变换
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))

        # 标准误差
        se = 1 / np.sqrt(n - 3)

        # z统计量
        z_stat = z / se

        # 双尾p值
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return p_value

    def _test_granger_causality(
        self, series1: pd.Series, series2: pd.Series, lag: int
    ) -> float:
        """测试Granger因果性

        Args:
            series1: 潜在原因序列
            series2: 潜在结果序列
            lag: 滞后期数

        Returns:
            因果性分数
        """
        # 简化的Granger因果检验
        # 实际应用中应使用statsmodels的grangercausalitytests

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        # 准备数据
        X_restricted = []
        X_unrestricted = []
        y = []

        for i in range(lag, len(series2)):
            # 限制模型：只用series2的滞后值
            X_restricted.append(series2.iloc[i - lag : i].values)

            # 非限制模型：用series1和series2的滞后值
            X_unrestricted.append(
                np.concatenate(
                    [series1.iloc[i - lag : i].values, series2.iloc[i - lag : i].values]
                )
            )

            y.append(series2.iloc[i])

        X_restricted = np.array(X_restricted)
        X_unrestricted = np.array(X_unrestricted)
        y = np.array(y)

        # 拟合模型
        model_restricted = LinearRegression()
        model_unrestricted = LinearRegression()

        model_restricted.fit(X_restricted, y)
        model_unrestricted.fit(X_unrestricted, y)

        # 计算R²
        r2_restricted = r2_score(y, model_restricted.predict(X_restricted))
        r2_unrestricted = r2_score(y, model_unrestricted.predict(X_unrestricted))

        # 因果性分数（R²的改进）
        causality_score = max(0, r2_unrestricted - r2_restricted)

        return causality_score


# 模块级别函数
def analyze_correlation_dynamics(
    returns_data: pd.DataFrame, config: Optional[CorrelationConfig] = None
) -> Dict[str, Any]:
    """分析相关性动态的便捷函数

    Args:
        returns_data: 收益率数据
        config: 相关性配置

    Returns:
        相关性分析结果字典
    """
    analyzer = DynamicCorrelationAnalyzer(config)

    # 计算动态相关性
    dynamic_corr = analyzer.calculate_dynamic_correlation(returns_data)

    # 检测领先滞后关系
    lead_lag = analyzer.detect_lead_lag_relationships(returns_data)

    # 计算网络中心性
    centrality = analyzer.calculate_correlation_network_centrality(dynamic_corr)

    return {
        "correlation_matrix": dynamic_corr,
        "lead_lag_relationships": [
            {
                "leader": rel.leader,
                "follower": rel.follower,
                "lag": rel.lag,
                "correlation": rel.correlation,
            }
            for rel in lead_lag
        ],
        "centrality_measures": centrality.to_dict(),
    }
