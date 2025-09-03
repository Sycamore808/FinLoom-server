"""
系统性风险分析器模块
分析和量化系统性风险
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.exceptions import ModelError
from common.logging_system import setup_logger
from scipy import linalg, stats

logger = setup_logger("systemic_risk_analyzer")


class SystemicRiskMeasure(Enum):
    """系统性风险度量枚举"""

    COVAR = "covar"
    MARGINAL_ES = "marginal_expected_shortfall"
    SRISK = "srisk"
    DIP = "distress_insurance_premium"
    NETWORK_CENTRALITY = "network_centrality"
    GRANGER_CAUSALITY = "granger_causality"


@dataclass
class SystemicRiskConfig:
    """系统性风险配置"""

    risk_measures: List[SystemicRiskMeasure] = None
    confidence_level: float = 0.95
    stress_level: float = 0.40  # 40%市场下跌
    correlation_threshold: float = 0.7
    network_density_threshold: float = 0.3
    use_dynamic_analysis: bool = True
    rolling_window: int = 252
    min_observations: int = 100


@dataclass
class NetworkMetrics:
    """网络度量"""

    degree_centrality: pd.Series
    betweenness_centrality: pd.Series
    eigenvector_centrality: pd.Series
    clustering_coefficient: float
    network_density: float
    average_path_length: float
    modularity: float


@dataclass
class SystemicRiskResult:
    """系统性风险分析结果"""

    risk_measures: Dict[str, float]
    risk_contributions: pd.Series
    systemic_importance: pd.Series
    network_metrics: NetworkMetrics
    contagion_matrix: pd.DataFrame
    vulnerability_scores: pd.Series
    early_warning_signals: Dict[str, float]
    risk_decomposition: pd.DataFrame
    time_series_analysis: pd.DataFrame


class SystemicRiskAnalyzer:
    """系统性风险分析器类"""

    def __init__(self, config: Optional[SystemicRiskConfig] = None):
        """初始化系统性风险分析器

        Args:
            config: 系统性风险配置
        """
        self.config = config or SystemicRiskConfig()
        if self.config.risk_measures is None:
            self.config.risk_measures = list(SystemicRiskMeasure)
        self.analysis_cache: Dict[str, Any] = {}

    def analyze_systemic_risk(
        self,
        returns_data: pd.DataFrame,
        market_returns: pd.Series,
        balance_sheet_data: Optional[pd.DataFrame] = None,
    ) -> SystemicRiskResult:
        """分析系统性风险

        Args:
            returns_data: 资产收益率数据
            market_returns: 市场收益率
            balance_sheet_data: 资产负债表数据（可选）

        Returns:
            系统性风险分析结果
        """
        logger.info("Starting systemic risk analysis")

        # 计算风险度量
        risk_measures = {}

        if SystemicRiskMeasure.COVAR in self.config.risk_measures:
            covar = self._calculate_covar(returns_data, market_returns)
            risk_measures["covar"] = covar.mean()

        if SystemicRiskMeasure.MARGINAL_ES in self.config.risk_measures:
            mes = self._calculate_marginal_es(returns_data, market_returns)
            risk_measures["marginal_es"] = mes.mean()

        if (
            SystemicRiskMeasure.SRISK in self.config.risk_measures
            and balance_sheet_data is not None
        ):
            srisk = self._calculate_srisk(
                returns_data, market_returns, balance_sheet_data
            )
            risk_measures["srisk"] = srisk.mean()

        if SystemicRiskMeasure.DIP in self.config.risk_measures:
            dip = self._calculate_distress_insurance_premium(returns_data)
            risk_measures["dip"] = dip

        # 计算风险贡献
        risk_contributions = self._calculate_risk_contributions(
            returns_data, market_returns
        )

        # 计算系统重要性
        systemic_importance = self._calculate_systemic_importance(
            returns_data, risk_contributions
        )

        # 网络分析
        network_metrics = self._analyze_network_structure(returns_data)

        # 传染矩阵
        contagion_matrix = self._calculate_contagion_matrix(returns_data)

        # 脆弱性评分
        vulnerability_scores = self._calculate_vulnerability_scores(
            returns_data, contagion_matrix
        )

        # 早期预警信号
        early_warning_signals = self._detect_early_warning_signals(returns_data)

        # 风险分解
        risk_decomposition = self._decompose_systemic_risk(returns_data, market_returns)

        # 时间序列分析
        if self.config.use_dynamic_analysis:
            time_series_analysis = self._perform_time_series_analysis(
                returns_data, market_returns
            )
        else:
            time_series_analysis = pd.DataFrame()

        result = SystemicRiskResult(
            risk_measures=risk_measures,
            risk_contributions=risk_contributions,
            systemic_importance=systemic_importance,
            network_metrics=network_metrics,
            contagion_matrix=contagion_matrix,
            vulnerability_scores=vulnerability_scores,
            early_warning_signals=early_warning_signals,
            risk_decomposition=risk_decomposition,
            time_series_analysis=time_series_analysis,
        )

        logger.info("Systemic risk analysis completed")

        return result

    def calculate_interconnectedness(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """计算互联性矩阵

        Args:
            returns_data: 资产收益率数据

        Returns:
            互联性矩阵
        """
        logger.info("Calculating interconnectedness matrix")

        # 动态条件相关性 (DCC)
        correlation_matrix = self._calculate_dynamic_correlation(returns_data)

        # 阈值处理
        interconnectedness = correlation_matrix.copy()
        interconnectedness[
            abs(interconnectedness) < self.config.correlation_threshold
        ] = 0

        return interconnectedness

    def identify_systemically_important_institutions(
        self, returns_data: pd.DataFrame, market_cap_data: Optional[pd.Series] = None
    ) -> pd.Series:
        """识别系统重要性机构

        Args:
            returns_data: 资产收益率数据
            market_cap_data: 市值数据（可选）

        Returns:
            系统重要性评分
        """
        logger.info("Identifying systemically important institutions")

        scores = pd.Series(index=returns_data.columns, dtype=float)

        # 1. 规模指标
        if market_cap_data is not None:
            size_score = market_cap_data / market_cap_data.sum()
        else:
            size_score = pd.Series(
                1 / len(returns_data.columns), index=returns_data.columns
            )

        # 2. 互联性指标
        correlation_matrix = returns_data.corr()
        interconnectedness_score = correlation_matrix.mean()

        # 3. 可替代性指标
        uniqueness = 1 - correlation_matrix.mean()
        substitutability_score = uniqueness / uniqueness.sum()

        # 4. 复杂性指标（使用收益率的峰度作为代理）
        complexity_score = returns_data.kurtosis()
        complexity_score = complexity_score / complexity_score.sum()

        # 综合评分
        scores = (
            0.3 * size_score
            + 0.3 * interconnectedness_score
            + 0.2 * substitutability_score
            + 0.2 * complexity_score
        )

        return scores

    def stress_test_contagion(
        self, returns_data: pd.DataFrame, initial_shock: Dict[str, float]
    ) -> pd.DataFrame:
        """压力测试传染效应

        Args:
            returns_data: 资产收益率数据
            initial_shock: 初始冲击

        Returns:
            传染路径DataFrame
        """
        logger.info("Running contagion stress test")

        # 计算传染矩阵
        contagion_matrix = self._calculate_contagion_matrix(returns_data)

        # 初始化冲击向量
        n_assets = len(returns_data.columns)
        shock_vector = pd.Series(0.0, index=returns_data.columns)

        for asset, shock in initial_shock.items():
            if asset in shock_vector.index:
                shock_vector[asset] = shock

        # 传播冲击（多轮）
        contagion_rounds = []
        current_shock = shock_vector.copy()

        for round_num in range(10):  # 最多10轮传染
            # 传播冲击
            next_shock = contagion_matrix @ current_shock

            # 记录
            contagion_rounds.append(
                {
                    "round": round_num + 1,
                    "total_impact": next_shock.sum(),
                    "max_impact": next_shock.max(),
                    "affected_assets": (next_shock != 0).sum(),
                }
            )

            # 检查收敛
            if np.allclose(next_shock, current_shock, rtol=1e-5):
                break

            current_shock = next_shock

        return pd.DataFrame(contagion_rounds)

    def _calculate_covar(
        self, returns_data: pd.DataFrame, market_returns: pd.Series
    ) -> pd.Series:
        """计算CoVaR

        Args:
            returns_data: 资产收益率数据
            market_returns: 市场收益率

        Returns:
            CoVaR序列
        """
        covar_values = pd.Series(index=returns_data.columns, dtype=float)

        for asset in returns_data.columns:
            asset_returns = returns_data[asset]

            # 计算条件VaR
            quantile = 1 - self.config.confidence_level

            # 当资产处于困境时的市场VaR
            distress_threshold = asset_returns.quantile(quantile)
            distressed_periods = asset_returns <= distress_threshold

            if distressed_periods.sum() > 0:
                conditional_market_returns = market_returns[distressed_periods]
                covar = conditional_market_returns.quantile(quantile)
            else:
                covar = market_returns.quantile(quantile)

            # 无条件VaR
            unconditional_var = market_returns.quantile(quantile)

            # Delta CoVaR
            delta_covar = covar - unconditional_var
            covar_values[asset] = delta_covar

        return covar_values

    def _calculate_marginal_es(
        self, returns_data: pd.DataFrame, market_returns: pd.Series
    ) -> pd.Series:
        """计算边际期望短缺

        Args:
            returns_data: 资产收益率数据
            market_returns: 市场收益率

        Returns:
            MES序列
        """
        mes_values = pd.Series(index=returns_data.columns, dtype=float)

        # 市场困境阈值
        market_threshold = market_returns.quantile(1 - self.config.confidence_level)
        distress_periods = market_returns <= market_threshold

        for asset in returns_data.columns:
            if distress_periods.sum() > 0:
                # 市场困境时的资产收益
                conditional_returns = returns_data[asset][distress_periods]
                mes = conditional_returns.mean()
            else:
                mes = returns_data[asset].quantile(1 - self.config.confidence_level)

            mes_values[asset] = mes

        return mes_values

    def _calculate_srisk(
        self,
        returns_data: pd.DataFrame,
        market_returns: pd.Series,
        balance_sheet_data: pd.DataFrame,
    ) -> pd.Series:
        """计算SRISK

        Args:
            returns_data: 资产收益率数据
            market_returns: 市场收益率
            balance_sheet_data: 资产负债表数据

        Returns:
            SRISK序列
        """
        srisk_values = pd.Series(index=returns_data.columns, dtype=float)

        # 目标资本比率
        k = 0.08  # 8%

        for asset in returns_data.columns:
            if asset in balance_sheet_data.index:
                # 获取资产负债表数据
                equity = (
                    balance_sheet_data.loc[asset, "equity"]
                    if "equity" in balance_sheet_data.columns
                    else 1e6
                )
                debt = (
                    balance_sheet_data.loc[asset, "debt"]
                    if "debt" in balance_sheet_data.columns
                    else 1e6
                )

                # 计算LRMES (长期边际期望短缺)
                mes = self._calculate_marginal_es(
                    returns_data[[asset]], market_returns
                )[asset]

                # 近似LRMES (6个月)
                lrmes = 1 - np.exp(18 * mes)

                # 计算SRISK
                srisk = equity * lrmes - k * (equity * (1 - lrmes) - debt)
                srisk_values[asset] = max(0, srisk)
            else:
                srisk_values[asset] = 0

        return srisk_values

    def _calculate_distress_insurance_premium(
        self, returns_data: pd.DataFrame
    ) -> float:
        """计算困境保险费

        Args:
            returns_data: 资产收益率数据

        Returns:
            DIP值
        """
        # 计算联合违约概率
        threshold = returns_data.quantile(1 - self.config.confidence_level)

        # 模拟联合分布
        n_simulations = 10000
        simulated_returns = np.random.multivariate_normal(
            returns_data.mean(), returns_data.cov(), n_simulations
        )

        # 计算违约指标
        defaults = (simulated_returns < threshold.values).all(axis=1)
        joint_default_prob = defaults.mean()

        # 预期损失
        expected_loss = joint_default_prob * self.config.stress_level

        # DIP (简化计算)
        dip = expected_loss * len(returns_data.columns)

        return dip

    def _calculate_risk_contributions(
        self, returns_data: pd.DataFrame, market_returns: pd.Series
    ) -> pd.Series:
        """计算风险贡献

        Args:
            returns_data: 资产收益率数据
            market_returns: 市场收益率

        Returns:
            风险贡献序列
        """
        contributions = pd.Series(index=returns_data.columns, dtype=float)

        # 计算每个资产对市场风险的贡献
        market_variance = market_returns.var()

        for asset in returns_data.columns:
            # 协方差贡献
            covariance = returns_data[asset].cov(market_returns)
            contribution = covariance / market_variance
            contributions[asset] = contribution

        return contributions

    def _calculate_systemic_importance(
        self, returns_data: pd.DataFrame, risk_contributions: pd.Series
    ) -> pd.Series:
        """计算系统重要性

        Args:
            returns_data: 资产收益率数据
            risk_contributions: 风险贡献

        Returns:
            系统重要性评分
        """
        # 多维度评分

        # 1. 风险贡献
        risk_score = risk_contributions / risk_contributions.sum()

        # 2. 网络中心性
        correlation_matrix = returns_data.corr()
        centrality_score = correlation_matrix.mean()
        centrality_score = centrality_score / centrality_score.sum()

        # 3. 尾部依赖
        tail_dependence = pd.Series(index=returns_data.columns, dtype=float)
        for col in returns_data.columns:
            # 计算与其他资产的尾部相关性
            tail_corr = 0
            for other_col in returns_data.columns:
                if col != other_col:
                    # 下尾相关性
                    threshold = 0.1
                    lower_tail = (
                        returns_data[col] < returns_data[col].quantile(threshold)
                    ) & (
                        returns_data[other_col]
                        < returns_data[other_col].quantile(threshold)
                    )
                    tail_corr += lower_tail.mean()
            tail_dependence[col] = tail_corr

        tail_score = tail_dependence / tail_dependence.sum()

        # 综合评分
        importance = 0.4 * risk_score + 0.3 * centrality_score + 0.3 * tail_score

        return importance

    def _analyze_network_structure(self, returns_data: pd.DataFrame) -> NetworkMetrics:
        """分析网络结构

        Args:
            returns_data: 资产收益率数据

        Returns:
            网络度量
        """
        # 构建相关性网络
        correlation_matrix = returns_data.corr()

        # 阈值处理创建邻接矩阵
        adjacency = (
            abs(correlation_matrix) > self.config.correlation_threshold
        ).astype(int)
        np.fill_diagonal(adjacency.values, 0)

        n_nodes = len(returns_data.columns)

        # 度中心性
        degree_centrality = pd.Series(
            adjacency.sum() / (n_nodes - 1), index=returns_data.columns
        )

        # 介数中心性（简化计算）
        betweenness_centrality = pd.Series(index=returns_data.columns, dtype=float)
        for node in returns_data.columns:
            # 简化：使用度作为代理
            betweenness_centrality[node] = degree_centrality[node]

        # 特征向量中心性
        eigenvalues, eigenvectors = linalg.eig(adjacency.values)
        largest_eigenvalue_index = np.argmax(eigenvalues.real)
        eigenvector_centrality = pd.Series(
            np.abs(eigenvectors[:, largest_eigenvalue_index].real),
            index=returns_data.columns,
        )
        eigenvector_centrality = eigenvector_centrality / eigenvector_centrality.sum()

        # 聚类系数
        clustering_coefficient = 0
        for node in range(n_nodes):
            neighbors = np.where(adjacency.iloc[node] == 1)[0]
            if len(neighbors) > 1:
                # 计算邻居之间的连接
                neighbor_connections = 0
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if adjacency.iloc[neighbors[i], neighbors[j]] == 1:
                            neighbor_connections += 1
                possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
                if possible_connections > 0:
                    clustering_coefficient += (
                        neighbor_connections / possible_connections
                    )
        clustering_coefficient /= n_nodes

        # 网络密度
        possible_edges = n_nodes * (n_nodes - 1) / 2
        actual_edges = adjacency.sum().sum() / 2
        network_density = actual_edges / possible_edges if possible_edges > 0 else 0

        # 平均路径长度（简化：使用平均度）
        average_path_length = 1 / (degree_centrality.mean() + 1e-10)

        # 模块度（简化计算）
        modularity = 1 - network_density  # 简化：密度越低，模块度越高

        return NetworkMetrics(
            degree_centrality=degree_centrality,
            betweenness_centrality=betweenness_centrality,
            eigenvector_centrality=eigenvector_centrality,
            clustering_coefficient=clustering_coefficient,
            network_density=network_density,
            average_path_length=average_path_length,
            modularity=modularity,
        )

    def _calculate_contagion_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """计算传染矩阵

        Args:
            returns_data: 资产收益率数据

        Returns:
            传染矩阵
        """
        n_assets = len(returns_data.columns)
        contagion_matrix = pd.DataFrame(
            np.zeros((n_assets, n_assets)),
            index=returns_data.columns,
            columns=returns_data.columns,
        )

        # 计算条件相关性
        for i, asset_i in enumerate(returns_data.columns):
            for j, asset_j in enumerate(returns_data.columns):
                if i != j:
                    # 计算asset_i困境时对asset_j的影响
                    threshold_i = returns_data[asset_i].quantile(0.1)
                    distress_i = returns_data[asset_i] <= threshold_i

                    if distress_i.sum() > 0:
                        # 困境时期的相关性
                        distress_corr = (
                            returns_data.loc[distress_i, [asset_i, asset_j]]
                            .corr()
                            .iloc[0, 1]
                        )
                        # 正常时期的相关性
                        normal_corr = returns_data[[asset_i, asset_j]].corr().iloc[0, 1]

                        # 传染强度
                        contagion = abs(distress_corr) - abs(normal_corr)
                        contagion_matrix.iloc[i, j] = max(0, contagion)

        return contagion_matrix

    def _calculate_vulnerability_scores(
        self, returns_data: pd.DataFrame, contagion_matrix: pd.DataFrame
    ) -> pd.Series:
        """计算脆弱性评分

        Args:
            returns_data: 资产收益率数据
            contagion_matrix: 传染矩阵

        Returns:
            脆弱性评分序列
        """
        vulnerability = pd.Series(index=returns_data.columns, dtype=float)

        for asset in returns_data.columns:
            # 1. 对传染的敏感性（入度）
            contagion_sensitivity = contagion_matrix[asset].sum()

            # 2. 波动率
            volatility = returns_data[asset].std()

            # 3. 下行风险
            downside_risk = returns_data[asset][returns_data[asset] < 0].std()

            # 4. 尾部风险
            var_95 = returns_data[asset].quantile(0.05)

            # 综合脆弱性评分
            vulnerability[asset] = (
                0.3 * contagion_sensitivity
                + 0.3 * volatility
                + 0.2 * downside_risk
                + 0.2 * abs(var_95)
            )

        # 标准化到[0, 1]
        if vulnerability.max() > vulnerability.min():
            vulnerability = (vulnerability - vulnerability.min()) / (
                vulnerability.max() - vulnerability.min()
            )

        return vulnerability

    def _detect_early_warning_signals(
        self, returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """检测早期预警信号

        Args:
            returns_data: 资产收益率数据

        Returns:
            早期预警信号字典
        """
        signals = {}

        # 1. 自相关增加（临界减速）
        autocorrelation = returns_data.apply(lambda x: x.autocorr())
        signals["autocorrelation"] = autocorrelation.mean()

        # 2. 方差增加
        rolling_variance = returns_data.rolling(window=20).var()
        variance_trend = rolling_variance.mean(axis=1).diff().mean()
        signals["variance_trend"] = variance_trend

        # 3. 偏度变化
        skewness = returns_data.skew()
        signals["skewness"] = skewness.mean()

        # 4. 网络密度增加
        correlation_matrix = returns_data.corr()
        high_corr = (
            (abs(correlation_matrix) > self.config.correlation_threshold).sum().sum()
        )
        total_pairs = len(returns_data.columns) * (len(returns_data.columns) - 1)
        signals["network_density"] = high_corr / total_pairs if total_pairs > 0 else 0

        # 5. 尾部依赖增加
        tail_correlation = 0
        for col1 in returns_data.columns:
            for col2 in returns_data.columns:
                if col1 != col2:
                    # 计算尾部相关性
                    threshold = 0.1
                    tail_events = (
                        returns_data[col1] < returns_data[col1].quantile(threshold)
                    ) & (returns_data[col2] < returns_data[col2].quantile(threshold))
                    tail_correlation += tail_events.mean()
        signals["tail_dependence"] = tail_correlation / (
            total_pairs if total_pairs > 0 else 1
        )

        return signals

    def _decompose_systemic_risk(
        self, returns_data: pd.DataFrame, market_returns: pd.Series
    ) -> pd.DataFrame:
        """分解系统性风险

        Args:
            returns_data: 资产收益率数据
            market_returns: 市场收益率

        Returns:
            风险分解DataFrame
        """
        decomposition = pd.DataFrame(index=returns_data.columns)

        # 1. 市场风险成分
        market_betas = pd.Series(index=returns_data.columns, dtype=float)
        for asset in returns_data.columns:
            covariance = returns_data[asset].cov(market_returns)
            market_variance = market_returns.var()
            market_betas[asset] = (
                covariance / market_variance if market_variance > 0 else 0
            )
        decomposition["market_risk"] = market_betas

        # 2. 特质风险成分
        idiosyncratic_risk = pd.Series(index=returns_data.columns, dtype=float)
        for asset in returns_data.columns:
            # 市场模型残差
            predicted = market_betas[asset] * market_returns
            residuals = returns_data[asset] - predicted
            idiosyncratic_risk[asset] = residuals.std()
        decomposition["idiosyncratic_risk"] = idiosyncratic_risk

        # 3. 传染风险成分
        contagion_risk = self._calculate_contagion_matrix(returns_data).mean()
        decomposition["contagion_risk"] = contagion_risk

        # 4. 尾部风险成分
        tail_risk = pd.Series(index=returns_data.columns, dtype=float)
        for asset in returns_data.columns:
            # 使用CVaR作为尾部风险度量
            var_95 = returns_data[asset].quantile(0.05)
            cvar_95 = returns_data[asset][returns_data[asset] <= var_95].mean()
            tail_risk[asset] = abs(cvar_95)
        decomposition["tail_risk"] = tail_risk

        # 标准化
        for col in decomposition.columns:
            if decomposition[col].std() > 0:
                decomposition[col] = decomposition[col] / decomposition[col].std()

        return decomposition

    def _perform_time_series_analysis(
        self, returns_data: pd.DataFrame, market_returns: pd.Series
    ) -> pd.DataFrame:
        """执行时间序列分析

        Args:
            returns_data: 资产收益率数据
            market_returns: 市场收益率

        Returns:
            时间序列分析结果
        """
        window = self.config.rolling_window

        if len(returns_data) < window:
            window = max(20, len(returns_data) // 2)

        # 滚动计算系统性风险指标
        rolling_results = []

        for i in range(window, len(returns_data)):
            window_data = returns_data.iloc[i - window : i]
            window_market = (
                market_returns.iloc[i - window : i] if len(market_returns) > i else None
            )

            # 计算窗口内的指标
            result = {
                "date": returns_data.index[i] if hasattr(returns_data, "index") else i,
                "correlation": window_data.corr().values.mean(),
                "volatility": window_data.std().mean(),
                "tail_dependence": self._calculate_tail_dependence(window_data),
            }

            rolling_results.append(result)

        return pd.DataFrame(rolling_results)

    def _calculate_dynamic_correlation(
        self, returns_data: pd.DataFrame
    ) -> pd.DataFrame:
        """计算动态相关性

        Args:
            returns_data: 资产收益率数据

        Returns:
            动态相关性矩阵
        """
        # 使用指数加权计算动态相关性
        ewm_cov = returns_data.ewm(span=self.config.rolling_window).cov()

        # 获取最新的协方差矩阵
        if isinstance(ewm_cov.index, pd.MultiIndex):
            latest_cov = ewm_cov.xs(returns_data.index[-1], level=0)
        else:
            latest_cov = returns_data.cov()

        # 转换为相关性矩阵
        std_devs = np.sqrt(np.diag(latest_cov))
        correlation_matrix = latest_cov / np.outer(std_devs, std_devs)

        return pd.DataFrame(
            correlation_matrix, index=returns_data.columns, columns=returns_data.columns
        )

    def _calculate_tail_dependence(self, returns_data: pd.DataFrame) -> float:
        """计算尾部依赖度

        Args:
            returns_data: 资产收益率数据

        Returns:
            平均尾部依赖度
        """
        threshold = 0.1
        tail_dependence = 0
        n_pairs = 0

        for col1 in returns_data.columns:
            for col2 in returns_data.columns:
                if col1 < col2:  # 避免重复计算
                    # 下尾依赖
                    lower_tail = (
                        returns_data[col1] < returns_data[col1].quantile(threshold)
                    ) & (returns_data[col2] < returns_data[col2].quantile(threshold))
                    tail_dependence += lower_tail.mean()
                    n_pairs += 1

        return tail_dependence / n_pairs if n_pairs > 0 else 0


# 模块级别函数
def analyze_systemic_risk(
    returns_data: pd.DataFrame,
    market_returns: pd.Series,
    config: Optional[SystemicRiskConfig] = None,
) -> Dict[str, Any]:
    """分析系统性风险的便捷函数

    Args:
        returns_data: 资产收益率数据
        market_returns: 市场收益率
        config: 配置

    Returns:
        系统性风险分析结果字典
    """
    analyzer = SystemicRiskAnalyzer(config)
    result = analyzer.analyze_systemic_risk(returns_data, market_returns)

    return {
        "risk_measures": result.risk_measures,
        "systemic_importance": result.systemic_importance.to_dict(),
        "network_density": result.network_metrics.network_density,
        "early_warning_signals": result.early_warning_signals,
        "max_vulnerability": result.vulnerability_scores.max(),
    }
