#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
社区检测模块
用于检测股票网络中的社区结构
"""

import warnings

warnings.filterwarnings("ignore")

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx
    from networkx.algorithms import community
except ImportError:
    nx = None
    community = None

try:
    from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
    from sklearn.preprocessing import StandardScaler
except ImportError:
    KMeans = None
    SpectralClustering = None
    DBSCAN = None
    StandardScaler = None

logger = logging.getLogger(__name__)


@dataclass
class CommunityConfig:
    """社区检测配置"""

    method: str = "louvain"  # 检测方法
    resolution: float = 1.0  # 分辨率参数
    n_clusters: int = 5  # 聚类数量
    random_state: int = 42  # 随机种子
    min_community_size: int = 3  # 最小社区大小
    max_communities: int = 20  # 最大社区数量


class CommunityDetection:
    """社区检测器"""

    def __init__(self, config: Optional[CommunityConfig] = None):
        """初始化社区检测器

        Args:
            config: 社区检测配置
        """
        self.config = config or CommunityConfig()
        self.communities = None
        self.community_stats = {}

        if nx is None:
            logger.warning(
                "NetworkX not available. Community detection functionality will be limited."
            )

    def detect_communities(
        self, graph: Any, method: Optional[str] = None
    ) -> Optional[Dict[Any, int]]:
        """检测社区

        Args:
            graph: NetworkX图对象
            method: 检测方法

        Returns:
            节点到社区的映射
        """
        try:
            if nx is None:
                logger.error("NetworkX not available")
                return None

            method = method or self.config.method
            logger.info(f"Detecting communities using {method} method...")

            if method == "louvain":
                return self._louvain_communities(graph)
            elif method == "leiden":
                return self._leiden_communities(graph)
            elif method == "greedy_modularity":
                return self._greedy_modularity_communities(graph)
            elif method == "spectral":
                return self._spectral_communities(graph)
            elif method == "label_propagation":
                return self._label_propagation_communities(graph)
            elif method == "infomap":
                return self._infomap_communities(graph)
            elif method == "walktrap":
                return self._walktrap_communities(graph)
            else:
                raise ValueError(f"Unknown community detection method: {method}")

        except Exception as e:
            logger.error(f"Failed to detect communities: {e}")
            return None

    def _louvain_communities(self, graph: Any) -> Dict[Any, int]:
        """Louvain算法检测社区"""
        try:
            if community is None:
                logger.error("NetworkX community module not available")
                return {}

            # 使用Louvain算法
            communities = community.louvain_communities(
                graph,
                resolution=self.config.resolution,
                random_state=self.config.random_state,
            )

            # 转换为节点到社区的映射
            node_to_community = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_community[node] = i

            self.communities = node_to_community
            self._calculate_community_stats(graph, communities)

            logger.info(
                f"Detected {len(communities)} communities using Louvain algorithm"
            )
            return node_to_community

        except Exception as e:
            logger.error(f"Failed to run Louvain algorithm: {e}")
            return {}

    def _leiden_communities(self, graph: Any) -> Dict[Any, int]:
        """Leiden算法检测社区"""
        try:
            # NetworkX中可能没有Leiden算法，使用Louvain作为替代
            logger.warning("Using Louvain algorithm as Leiden alternative")
            return self._louvain_communities(graph)

        except Exception as e:
            logger.error(f"Failed to run Leiden algorithm: {e}")
            return {}

    def _greedy_modularity_communities(self, graph: Any) -> Dict[Any, int]:
        """贪心模块度算法检测社区"""
        try:
            if community is None:
                logger.error("NetworkX community module not available")
                return {}

            communities = community.greedy_modularity_communities(graph)

            # 转换为节点到社区的映射
            node_to_community = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_community[node] = i

            self.communities = node_to_community
            self._calculate_community_stats(graph, communities)

            logger.info(
                f"Detected {len(communities)} communities using greedy modularity"
            )
            return node_to_community

        except Exception as e:
            logger.error(f"Failed to run greedy modularity algorithm: {e}")
            return {}

    def _spectral_communities(self, graph: Any) -> Dict[Any, int]:
        """谱聚类检测社区"""
        try:
            if SpectralClustering is None or nx is None:
                logger.error("Required libraries not available")
                return {}

            # 获取邻接矩阵
            adj_matrix = nx.adjacency_matrix(graph)

            # 谱聚类
            spectral = SpectralClustering(
                n_clusters=self.config.n_clusters,
                random_state=self.config.random_state,
                affinity="precomputed",
            )

            labels = spectral.fit_predict(adj_matrix)

            # 转换为节点到社区的映射
            nodes = list(graph.nodes())
            node_to_community = {node: int(label) for node, label in zip(nodes, labels)}

            self.communities = node_to_community

            logger.info(
                f"Detected {self.config.n_clusters} communities using spectral clustering"
            )
            return node_to_community

        except Exception as e:
            logger.error(f"Failed to run spectral clustering: {e}")
            return {}

    def _label_propagation_communities(self, graph: Any) -> Dict[Any, int]:
        """标签传播算法检测社区"""
        try:
            if community is None:
                logger.error("NetworkX community module not available")
                return {}

            communities = community.label_propagation_communities(graph)

            # 转换为节点到社区的映射
            node_to_community = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_community[node] = i

            self.communities = node_to_community
            self._calculate_community_stats(graph, communities)

            logger.info(
                f"Detected {len(communities)} communities using label propagation"
            )
            return node_to_community

        except Exception as e:
            logger.error(f"Failed to run label propagation algorithm: {e}")
            return {}

    def _infomap_communities(self, graph: Any) -> Dict[Any, int]:
        """Infomap算法检测社区"""
        try:
            # NetworkX可能没有Infomap，使用Louvain作为替代
            logger.warning("Using Louvain algorithm as Infomap alternative")
            return self._louvain_communities(graph)

        except Exception as e:
            logger.error(f"Failed to run Infomap algorithm: {e}")
            return {}

    def _walktrap_communities(self, graph: Any) -> Dict[Any, int]:
        """Walktrap算法检测社区"""
        try:
            # NetworkX可能没有Walktrap，使用Louvain作为替代
            logger.warning("Using Louvain algorithm as Walktrap alternative")
            return self._louvain_communities(graph)

        except Exception as e:
            logger.error(f"Failed to run Walktrap algorithm: {e}")
            return {}

    def detect_hierarchical_communities(
        self, graph: Any
    ) -> Optional[List[Dict[Any, int]]]:
        """检测层次化社区

        Args:
            graph: NetworkX图对象

        Returns:
            不同层次的社区结构
        """
        try:
            if nx is None:
                logger.error("NetworkX not available")
                return None

            logger.info("Detecting hierarchical communities...")

            hierarchical_communities = []

            # 使用不同的分辨率参数
            resolutions = [0.5, 1.0, 1.5, 2.0]

            for resolution in resolutions:
                if community is not None:
                    communities = community.louvain_communities(
                        graph,
                        resolution=resolution,
                        random_state=self.config.random_state,
                    )

                    # 转换为节点到社区的映射
                    node_to_community = {}
                    for i, comm in enumerate(communities):
                        for node in comm:
                            node_to_community[node] = i

                    hierarchical_communities.append(node_to_community)

            logger.info(f"Detected {len(hierarchical_communities)} hierarchical levels")
            return hierarchical_communities

        except Exception as e:
            logger.error(f"Failed to detect hierarchical communities: {e}")
            return None

    def calculate_modularity(self, graph: Any, communities: Dict[Any, int]) -> float:
        """计算模块度

        Args:
            graph: NetworkX图对象
            communities: 社区分配

        Returns:
            模块度值
        """
        try:
            if nx is None:
                return 0.0

            # 转换社区格式
            community_sets = {}
            for node, comm_id in communities.items():
                if comm_id not in community_sets:
                    community_sets[comm_id] = set()
                community_sets[comm_id].add(node)

            community_list = list(community_sets.values())

            # 计算模块度
            modularity = nx.algorithms.community.modularity(graph, community_list)

            return modularity

        except Exception as e:
            logger.error(f"Failed to calculate modularity: {e}")
            return 0.0

    def get_community_statistics(
        self, graph: Any, communities: Dict[Any, int]
    ) -> Dict[str, Any]:
        """获取社区统计信息

        Args:
            graph: NetworkX图对象
            communities: 社区分配

        Returns:
            社区统计信息
        """
        try:
            if nx is None:
                return {}

            stats = {}

            # 社区数量
            num_communities = len(set(communities.values()))
            stats["num_communities"] = num_communities

            # 各社区大小
            community_sizes = {}
            for node, comm_id in communities.items():
                community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1

            stats["community_sizes"] = community_sizes
            stats["avg_community_size"] = np.mean(list(community_sizes.values()))
            stats["std_community_size"] = np.std(list(community_sizes.values()))
            stats["min_community_size"] = min(community_sizes.values())
            stats["max_community_size"] = max(community_sizes.values())

            # 模块度
            stats["modularity"] = self.calculate_modularity(graph, communities)

            # 社区内/外边数
            internal_edges = 0
            external_edges = 0

            for edge in graph.edges():
                node1, node2 = edge
                if communities.get(node1) == communities.get(node2):
                    internal_edges += 1
                else:
                    external_edges += 1

            stats["internal_edges"] = internal_edges
            stats["external_edges"] = external_edges
            stats["internal_edge_ratio"] = (
                internal_edges / (internal_edges + external_edges)
                if (internal_edges + external_edges) > 0
                else 0
            )

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate community statistics: {e}")
            return {}

    def _calculate_community_stats(self, graph: Any, communities: List[set]):
        """计算社区统计信息"""
        try:
            # 转换社区格式
            node_to_community = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_community[node] = i

            self.community_stats = self.get_community_statistics(
                graph, node_to_community
            )

        except Exception as e:
            logger.error(f"Failed to calculate community stats: {e}")

    def get_community_subgraphs(
        self, graph: Any, communities: Dict[Any, int]
    ) -> Dict[int, Any]:
        """获取各社区的子图

        Args:
            graph: NetworkX图对象
            communities: 社区分配

        Returns:
            社区ID到子图的映射
        """
        try:
            if nx is None:
                return {}

            subgraphs = {}

            # 按社区分组节点
            community_nodes = {}
            for node, comm_id in communities.items():
                if comm_id not in community_nodes:
                    community_nodes[comm_id] = []
                community_nodes[comm_id].append(node)

            # 为每个社区创建子图
            for comm_id, nodes in community_nodes.items():
                subgraph = graph.subgraph(nodes).copy()
                subgraphs[comm_id] = subgraph

            return subgraphs

        except Exception as e:
            logger.error(f"Failed to get community subgraphs: {e}")
            return {}

    def compare_communities(
        self, communities1: Dict[Any, int], communities2: Dict[Any, int]
    ) -> Dict[str, float]:
        """比较两个社区结构

        Args:
            communities1: 第一个社区结构
            communities2: 第二个社区结构

        Returns:
            比较指标
        """
        try:
            from sklearn.metrics import (
                adjusted_rand_score,
                normalized_mutual_info_score,
            )

            # 获取共同节点
            common_nodes = set(communities1.keys()) & set(communities2.keys())

            if not common_nodes:
                return {"adjusted_rand_score": 0.0, "normalized_mutual_info": 0.0}

            # 提取标签
            labels1 = [communities1[node] for node in common_nodes]
            labels2 = [communities2[node] for node in common_nodes]

            # 计算比较指标
            ari = adjusted_rand_score(labels1, labels2)
            nmi = normalized_mutual_info_score(labels1, labels2)

            return {
                "adjusted_rand_score": ari,
                "normalized_mutual_info": nmi,
                "num_common_nodes": len(common_nodes),
            }

        except Exception as e:
            logger.error(f"Failed to compare communities: {e}")
            return {"adjusted_rand_score": 0.0, "normalized_mutual_info": 0.0}

    def get_community_features(
        self, graph: Any, communities: Dict[Any, int]
    ) -> pd.DataFrame:
        """获取社区特征

        Args:
            graph: NetworkX图对象
            communities: 社区分配

        Returns:
            节点的社区特征
        """
        try:
            if nx is None:
                return pd.DataFrame()

            features = []

            for node in graph.nodes():
                comm_id = communities.get(node, -1)

                # 社区内度数
                internal_degree = 0
                external_degree = 0

                for neighbor in graph.neighbors(node):
                    if communities.get(neighbor, -1) == comm_id:
                        internal_degree += 1
                    else:
                        external_degree += 1

                # 社区特征
                feature = {
                    "node": node,
                    "community_id": comm_id,
                    "internal_degree": internal_degree,
                    "external_degree": external_degree,
                    "total_degree": internal_degree + external_degree,
                    "internal_degree_ratio": internal_degree
                    / (internal_degree + external_degree)
                    if (internal_degree + external_degree) > 0
                    else 0,
                }

                features.append(feature)

            return pd.DataFrame(features).set_index("node")

        except Exception as e:
            logger.error(f"Failed to get community features: {e}")
            return pd.DataFrame()


# 便捷函数
def detect_stock_communities(
    graph: Any, method: str = "louvain"
) -> Optional[Dict[Any, int]]:
    """检测股票社区的便捷函数

    Args:
        graph: NetworkX图对象
        method: 检测方法

    Returns:
        节点到社区的映射
    """
    config = CommunityConfig(method=method)
    detector = CommunityDetection(config)
    return detector.detect_communities(graph, method)
