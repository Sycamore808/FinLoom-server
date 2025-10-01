#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票图构建器模块
用于构建股票关系图网络
"""

import warnings

warnings.filterwarnings("ignore")

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None
    cosine_similarity = None

logger = logging.getLogger(__name__)


@dataclass
class GraphConfig:
    """图构建配置"""

    correlation_threshold: float = 0.5  # 相关性阈值
    min_correlation_periods: int = 60  # 最小相关性计算期数
    similarity_method: str = "correlation"  # 相似性方法
    edge_weight_method: str = "correlation"  # 边权重方法
    max_edges_per_node: int = 20  # 每个节点最大边数
    remove_self_loops: bool = True  # 是否移除自环
    directed: bool = False  # 是否为有向图


class StockGraphBuilder:
    """股票图构建器"""

    def __init__(self, config: Optional[GraphConfig] = None):
        """初始化股票图构建器

        Args:
            config: 图构建配置
        """
        self.config = config or GraphConfig()

        if nx is None:
            logger.warning(
                "NetworkX not available. Graph functionality will be limited."
            )

        self.graph = None
        self.node_features = {}
        self.edge_features = {}

    def build_correlation_graph(
        self, price_data: pd.DataFrame, return_data: Optional[pd.DataFrame] = None
    ) -> Optional[Any]:
        """构建相关性图

        Args:
            price_data: 价格数据 (日期 x 股票代码)
            return_data: 收益率数据 (可选)

        Returns:
            NetworkX图对象
        """
        try:
            if nx is None:
                logger.error("NetworkX not available")
                return None

            logger.info("Building correlation graph...")

            # 使用收益率数据，如果没有提供则计算
            if return_data is None:
                return_data = price_data.pct_change().dropna()

            # 计算相关性矩阵
            correlation_matrix = return_data.corr()

            # 创建图
            self.graph = nx.Graph() if not self.config.directed else nx.DiGraph()

            # 添加节点
            stocks = correlation_matrix.columns
            self.graph.add_nodes_from(stocks)

            # 添加边
            for i, stock1 in enumerate(stocks):
                for j, stock2 in enumerate(stocks):
                    if i != j or not self.config.remove_self_loops:
                        correlation = correlation_matrix.loc[stock1, stock2]

                        # 检查相关性阈值
                        if abs(correlation) >= self.config.correlation_threshold:
                            weight = abs(correlation)
                            self.graph.add_edge(
                                stock1, stock2, weight=weight, correlation=correlation
                            )

            # 限制每个节点的边数
            self._limit_edges_per_node()

            logger.info(
                f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
            )
            return self.graph

        except Exception as e:
            logger.error(f"Failed to build correlation graph: {e}")
            return None

    def build_similarity_graph(self, feature_data: pd.DataFrame) -> Optional[Any]:
        """构建特征相似性图

        Args:
            feature_data: 特征数据 (股票代码 x 特征)

        Returns:
            NetworkX图对象
        """
        try:
            if nx is None or cosine_similarity is None:
                logger.error("Required libraries not available")
                return None

            logger.info("Building similarity graph...")

            # 标准化特征
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_data.fillna(0))

            # 计算相似性矩阵
            if self.config.similarity_method == "cosine":
                similarity_matrix = cosine_similarity(normalized_features)
            elif self.config.similarity_method == "correlation":
                similarity_matrix = np.corrcoef(normalized_features)
            else:
                raise ValueError(
                    f"Unknown similarity method: {self.config.similarity_method}"
                )

            # 创建图
            self.graph = nx.Graph() if not self.config.directed else nx.DiGraph()

            # 添加节点
            stocks = feature_data.index
            self.graph.add_nodes_from(stocks)

            # 添加边
            for i, stock1 in enumerate(stocks):
                for j, stock2 in enumerate(stocks):
                    if i != j or not self.config.remove_self_loops:
                        similarity = similarity_matrix[i, j]

                        # 检查相似性阈值
                        if abs(similarity) >= self.config.correlation_threshold:
                            weight = abs(similarity)
                            self.graph.add_edge(
                                stock1, stock2, weight=weight, similarity=similarity
                            )

            # 限制每个节点的边数
            self._limit_edges_per_node()

            logger.info(
                f"Similarity graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
            )
            return self.graph

        except Exception as e:
            logger.error(f"Failed to build similarity graph: {e}")
            return None

    def build_sector_graph(self, sector_data: pd.DataFrame) -> Optional[Any]:
        """构建行业图

        Args:
            sector_data: 行业数据 (股票代码 -> 行业)

        Returns:
            NetworkX图对象
        """
        try:
            if nx is None:
                logger.error("NetworkX not available")
                return None

            logger.info("Building sector graph...")

            # 创建图
            self.graph = nx.Graph() if not self.config.directed else nx.DiGraph()

            # 按行业分组
            sector_groups = sector_data.groupby("sector")

            # 添加节点和行业内连接
            for sector, group in sector_groups:
                stocks = group.index.tolist()

                # 添加节点
                for stock in stocks:
                    self.graph.add_node(stock, sector=sector)

                # 行业内股票互相连接
                for i, stock1 in enumerate(stocks):
                    for j, stock2 in enumerate(stocks[i + 1 :], i + 1):
                        self.graph.add_edge(
                            stock1,
                            stock2,
                            relationship="sector",
                            sector=sector,
                            weight=1.0,
                        )

            logger.info(
                f"Sector graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
            )
            return self.graph

        except Exception as e:
            logger.error(f"Failed to build sector graph: {e}")
            return None

    def build_market_cap_graph(self, market_cap_data: pd.DataFrame) -> Optional[Any]:
        """构建市值图

        Args:
            market_cap_data: 市值数据

        Returns:
            NetworkX图对象
        """
        try:
            if nx is None:
                logger.error("NetworkX not available")
                return None

            logger.info("Building market cap graph...")

            # 创建图
            self.graph = nx.Graph() if not self.config.directed else nx.DiGraph()

            # 按市值分层
            market_cap_values = market_cap_data["market_cap"].values
            market_cap_quantiles = pd.qcut(
                market_cap_values,
                q=5,
                labels=["small", "small_mid", "mid", "mid_large", "large"],
            )

            # 添加节点
            for stock, cap_level in zip(market_cap_data.index, market_cap_quantiles):
                self.graph.add_node(stock, market_cap_level=cap_level)

            # 同一市值层级内的股票连接
            for cap_level in ["small", "small_mid", "mid", "mid_large", "large"]:
                level_stocks = market_cap_data.index[
                    market_cap_quantiles == cap_level
                ].tolist()

                for i, stock1 in enumerate(level_stocks):
                    for j, stock2 in enumerate(level_stocks[i + 1 :], i + 1):
                        self.graph.add_edge(
                            stock1,
                            stock2,
                            relationship="market_cap",
                            cap_level=cap_level,
                            weight=1.0,
                        )

            logger.info(
                f"Market cap graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
            )
            return self.graph

        except Exception as e:
            logger.error(f"Failed to build market cap graph: {e}")
            return None

    def add_node_features(self, feature_data: pd.DataFrame, feature_prefix: str = ""):
        """添加节点特征

        Args:
            feature_data: 特征数据
            feature_prefix: 特征前缀
        """
        try:
            if self.graph is None:
                logger.warning("Graph not built yet")
                return

            for stock in feature_data.index:
                if stock in self.graph.nodes:
                    for feature_name, feature_value in feature_data.loc[stock].items():
                        attr_name = (
                            f"{feature_prefix}{feature_name}"
                            if feature_prefix
                            else feature_name
                        )
                        self.graph.nodes[stock][attr_name] = feature_value

            logger.info(f"Added node features with prefix '{feature_prefix}'")

        except Exception as e:
            logger.error(f"Failed to add node features: {e}")

    def add_edge_features(self, edge_feature_func: callable):
        """添加边特征

        Args:
            edge_feature_func: 边特征计算函数
        """
        try:
            if self.graph is None:
                logger.warning("Graph not built yet")
                return

            for edge in self.graph.edges():
                stock1, stock2 = edge
                features = edge_feature_func(stock1, stock2)

                for feature_name, feature_value in features.items():
                    self.graph.edges[edge][feature_name] = feature_value

            logger.info("Added edge features")

        except Exception as e:
            logger.error(f"Failed to add edge features: {e}")

    def get_adjacency_matrix(self) -> Optional[pd.DataFrame]:
        """获取邻接矩阵

        Returns:
            邻接矩阵
        """
        try:
            if self.graph is None or nx is None:
                return None

            # 获取邻接矩阵
            adj_matrix = nx.adjacency_matrix(self.graph, weight="weight")

            # 转换为DataFrame
            nodes = list(self.graph.nodes())
            adj_df = pd.DataFrame(adj_matrix.toarray(), index=nodes, columns=nodes)

            return adj_df

        except Exception as e:
            logger.error(f"Failed to get adjacency matrix: {e}")
            return None

    def get_degree_centrality(self) -> Optional[pd.Series]:
        """获取度中心性

        Returns:
            度中心性序列
        """
        try:
            if self.graph is None or nx is None:
                return None

            centrality = nx.degree_centrality(self.graph)
            return pd.Series(centrality)

        except Exception as e:
            logger.error(f"Failed to calculate degree centrality: {e}")
            return None

    def get_betweenness_centrality(self) -> Optional[pd.Series]:
        """获取介数中心性

        Returns:
            介数中心性序列
        """
        try:
            if self.graph is None or nx is None:
                return None

            centrality = nx.betweenness_centrality(self.graph)
            return pd.Series(centrality)

        except Exception as e:
            logger.error(f"Failed to calculate betweenness centrality: {e}")
            return None

    def get_closeness_centrality(self) -> Optional[pd.Series]:
        """获取紧密中心性

        Returns:
            紧密中心性序列
        """
        try:
            if self.graph is None or nx is None:
                return None

            centrality = nx.closeness_centrality(self.graph)
            return pd.Series(centrality)

        except Exception as e:
            logger.error(f"Failed to calculate closeness centrality: {e}")
            return None

    def get_eigenvector_centrality(self) -> Optional[pd.Series]:
        """获取特征向量中心性

        Returns:
            特征向量中心性序列
        """
        try:
            if self.graph is None or nx is None:
                return None

            centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
            return pd.Series(centrality)

        except Exception as e:
            logger.error(f"Failed to calculate eigenvector centrality: {e}")
            return None

    def get_pagerank(self) -> Optional[pd.Series]:
        """获取PageRank值

        Returns:
            PageRank序列
        """
        try:
            if self.graph is None or nx is None:
                return None

            pagerank = nx.pagerank(self.graph)
            return pd.Series(pagerank)

        except Exception as e:
            logger.error(f"Failed to calculate PageRank: {e}")
            return None

    def get_clustering_coefficient(self) -> Optional[pd.Series]:
        """获取聚集系数

        Returns:
            聚集系数序列
        """
        try:
            if self.graph is None or nx is None:
                return None

            clustering = nx.clustering(self.graph)
            return pd.Series(clustering)

        except Exception as e:
            logger.error(f"Failed to calculate clustering coefficient: {e}")
            return None

    def _limit_edges_per_node(self):
        """限制每个节点的边数"""
        try:
            if self.graph is None:
                return

            for node in list(self.graph.nodes()):
                # 获取节点的所有边
                edges = list(self.graph.edges(node, data=True))

                if len(edges) > self.config.max_edges_per_node:
                    # 按权重排序，保留权重最大的边
                    edges.sort(key=lambda x: x[2].get("weight", 0), reverse=True)

                    # 移除多余的边
                    edges_to_remove = edges[self.config.max_edges_per_node :]
                    for edge in edges_to_remove:
                        self.graph.remove_edge(edge[0], edge[1])

        except Exception as e:
            logger.error(f"Failed to limit edges per node: {e}")

    def save_graph(self, filepath: str):
        """保存图到文件

        Args:
            filepath: 文件路径
        """
        try:
            if self.graph is None or nx is None:
                logger.warning("No graph to save")
                return

            if filepath.endswith(".gexf"):
                nx.write_gexf(self.graph, filepath)
            elif filepath.endswith(".gml"):
                nx.write_gml(self.graph, filepath)
            elif filepath.endswith(".graphml"):
                nx.write_graphml(self.graph, filepath)
            else:
                # 默认使用pickle
                nx.write_gpickle(self.graph, filepath)

            logger.info(f"Graph saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def load_graph(self, filepath: str):
        """从文件加载图

        Args:
            filepath: 文件路径
        """
        try:
            if nx is None:
                logger.error("NetworkX not available")
                return

            if filepath.endswith(".gexf"):
                self.graph = nx.read_gexf(filepath)
            elif filepath.endswith(".gml"):
                self.graph = nx.read_gml(filepath)
            elif filepath.endswith(".graphml"):
                self.graph = nx.read_graphml(filepath)
            else:
                # 默认使用pickle
                self.graph = nx.read_gpickle(filepath)

            logger.info(f"Graph loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load graph: {e}")


# 便捷函数
def build_stock_correlation_graph(
    price_data: pd.DataFrame, correlation_threshold: float = 0.5
) -> Optional[Any]:
    """构建股票相关性图的便捷函数

    Args:
        price_data: 价格数据
        correlation_threshold: 相关性阈值

    Returns:
        NetworkX图对象
    """
    config = GraphConfig(correlation_threshold=correlation_threshold)
    builder = StockGraphBuilder(config)
    return builder.build_correlation_graph(price_data)
