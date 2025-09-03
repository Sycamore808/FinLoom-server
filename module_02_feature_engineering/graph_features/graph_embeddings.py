"""
图嵌入特征提取器模块
基于图神经网络提取股票关联特征
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.exceptions import ModelError
from common.logging_system import setup_logger
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GCNConv, GraphSAGE

logger = setup_logger("graph_embeddings")


@dataclass
class GraphConfig:
    """图配置"""

    embedding_dim: int = 64
    hidden_dims: List[int] = None
    num_heads: int = 4
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    gnn_type: str = "GAT"  # 'GCN', 'GAT', 'GraphSAGE'

    def __post_init__(self):
        """初始化后处理"""
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


@dataclass
class GraphEmbedding:
    """图嵌入结果"""

    node_embeddings: np.ndarray
    edge_index: np.ndarray
    node_features: np.ndarray
    graph_features: Dict[str, float]
    metadata: Dict[str, Any]


class GraphNeuralNetwork(nn.Module):
    """图神经网络模型"""

    def __init__(self, config: GraphConfig, input_dim: int):
        """初始化图神经网络

        Args:
            config: 图配置
            input_dim: 输入特征维度
        """
        super(GraphNeuralNetwork, self).__init__()
        self.config = config

        # 构建网络层
        layers = []
        prev_dim = input_dim

        for hidden_dim in config.hidden_dims:
            if config.gnn_type == "GCN":
                layers.append(GCNConv(prev_dim, hidden_dim))
            elif config.gnn_type == "GAT":
                layers.append(
                    GATConv(
                        prev_dim,
                        hidden_dim // config.num_heads,
                        heads=config.num_heads,
                        dropout=config.dropout,
                    )
                )
                hidden_dim = hidden_dim  # GAT输出已经是hidden_dim
            elif config.gnn_type == "GraphSAGE":
                layers.append(GraphSAGE(prev_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown GNN type: {config.gnn_type}")
            prev_dim = hidden_dim

        self.layers = nn.ModuleList(layers)

        # 输出层
        self.output_layer = nn.Linear(prev_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 节点特征 (num_nodes, input_dim)
            edge_index: 边索引 (2, num_edges)

        Returns:
            节点嵌入 (num_nodes, embedding_dim)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        x = self.output_layer(x)
        return x


class GraphEmbeddingExtractor:
    """图嵌入特征提取器"""

    def __init__(self, config: Optional[GraphConfig] = None):
        """初始化图嵌入提取器

        Args:
            config: 图配置
        """
        self.config = config or GraphConfig()
        self.model: Optional[GraphNeuralNetwork] = None
        self.graph: Optional[nx.Graph] = None
        self.node_mapping: Dict[str, int] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}

    def build_stock_correlation_graph(
        self,
        returns_df: pd.DataFrame,
        threshold: float = 0.3,
        method: str = "correlation",
    ) -> nx.Graph:
        """构建股票关联图

        Args:
            returns_df: 收益率DataFrame (index=日期, columns=股票代码)
            threshold: 相关性阈值
            method: 构建方法 ('correlation', 'partial_correlation', 'mi')

        Returns:
            股票关联图
        """
        logger.info(f"Building stock graph with {method} method")

        # 计算相关性矩阵
        if method == "correlation":
            corr_matrix = returns_df.corr()
        elif method == "partial_correlation":
            corr_matrix = self._calculate_partial_correlation(returns_df)
        elif method == "mi":  # Mutual Information
            corr_matrix = self._calculate_mutual_information(returns_df)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 创建图
        G = nx.Graph()

        # 添加节点
        for i, stock in enumerate(returns_df.columns):
            G.add_node(stock, index=i)
            self.node_mapping[stock] = i

        # 添加边
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                weight = abs(corr_matrix.iloc[i, j])
                if weight > threshold:
                    G.add_edge(
                        returns_df.columns[i], returns_df.columns[j], weight=weight
                    )

        self.graph = G
        logger.info(
            f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )
        return G

    def extract_graph_embeddings(
        self, node_features: pd.DataFrame, train_ratio: float = 0.8
    ) -> GraphEmbedding:
        """提取图嵌入

        Args:
            node_features: 节点特征DataFrame (index=股票代码, columns=特征)
            train_ratio: 训练集比例

        Returns:
            图嵌入结果
        """
        if self.graph is None:
            raise ModelError(
                "Graph not built. Call build_stock_correlation_graph first"
            )

        # 准备数据
        edge_index = self._get_edge_index()
        x = self._prepare_node_features(node_features)

        # 初始化模型
        if self.model is None:
            self.model = GraphNeuralNetwork(self.config, input_dim=x.shape[1])

        # 训练模型
        self._train_model(x, edge_index, train_ratio)

        # 提取嵌入
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(x, edge_index).numpy()

        # 计算图特征
        graph_features = self._calculate_graph_features()

        result = GraphEmbedding(
            node_embeddings=embeddings,
            edge_index=edge_index.numpy(),
            node_features=x.numpy(),
            graph_features=graph_features,
            metadata={
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "embedding_dim": self.config.embedding_dim,
            },
        )

        # 缓存嵌入
        for stock, idx in self.node_mapping.items():
            self.embeddings_cache[stock] = embeddings[idx]

        return result

    def detect_community_structures(self, method: str = "louvain") -> Dict[str, int]:
        """检测社区结构

        Args:
            method: 社区检测方法 ('louvain', 'label_propagation', 'girvan_newman')

        Returns:
            节点到社区的映射
        """
        if self.graph is None:
            raise ModelError("Graph not built")

        if method == "louvain":
            import community

            communities = community.best_partition(self.graph)
        elif method == "label_propagation":
            communities_generator = (
                nx.algorithms.community.label_propagation_communities(self.graph)
            )
            communities = {}
            for i, comm in enumerate(communities_generator):
                for node in comm:
                    communities[node] = i
        elif method == "girvan_newman":
            communities_generator = nx.algorithms.community.girvan_newman(self.graph)
            top_level_communities = next(communities_generator)
            communities = {}
            for i, comm in enumerate(top_level_communities):
                for node in comm:
                    communities[node] = i
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Detected {len(set(communities.values()))} communities")
        return communities

    def calculate_centrality_measures(self) -> pd.DataFrame:
        """计算中心性度量

        Returns:
            中心性度量DataFrame
        """
        if self.graph is None:
            raise ModelError("Graph not built")

        centrality_measures = pd.DataFrame()

        # 度中心性
        centrality_measures["degree_centrality"] = pd.Series(
            nx.degree_centrality(self.graph)
        )

        # 接近中心性
        centrality_measures["closeness_centrality"] = pd.Series(
            nx.closeness_centrality(self.graph)
        )

        # 介数中心性
        centrality_measures["betweenness_centrality"] = pd.Series(
            nx.betweenness_centrality(self.graph)
        )

        # 特征向量中心性
        try:
            centrality_measures["eigenvector_centrality"] = pd.Series(
                nx.eigenvector_centrality(self.graph, max_iter=1000)
            )
        except:
            centrality_measures["eigenvector_centrality"] = 0.0

        # PageRank
        centrality_measures["pagerank"] = pd.Series(nx.pagerank(self.graph))

        return centrality_measures

    def propagate_graph_signals(
        self, initial_signals: pd.Series, num_iterations: int = 3, damping: float = 0.85
    ) -> pd.Series:
        """图信号传播

        Args:
            initial_signals: 初始信号Series (index=股票代码)
            num_iterations: 迭代次数
            damping: 阻尼系数

        Returns:
            传播后的信号
        """
        if self.graph is None:
            raise ModelError("Graph not built")

        # 初始化信号
        signals = initial_signals.copy()

        # 获取邻接矩阵
        adj_matrix = nx.adjacency_matrix(self.graph).todense()

        # 归一化邻接矩阵
        row_sums = adj_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        norm_adj_matrix = adj_matrix / row_sums

        # 信号传播
        for _ in range(num_iterations):
            # 将信号转换为向量
            signal_vector = np.array(
                [signals.get(node, 0.0) for node in self.graph.nodes()]
            )

            # 传播
            propagated = (
                damping * norm_adj_matrix.dot(signal_vector)
                + (1 - damping) * signal_vector
            )

            # 更新信号
            for i, node in enumerate(self.graph.nodes()):
                signals[node] = propagated[i]

        return signals

    def _calculate_partial_correlation(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """计算偏相关系数矩阵

        Args:
            returns_df: 收益率DataFrame

        Returns:
            偏相关系数矩阵
        """
        from sklearn.covariance import GraphicalLassoCV

        # 使用Graphical Lasso估计精度矩阵
        estimator = GraphicalLassoCV(cv=5)
        estimator.fit(returns_df)

        # 精度矩阵转换为偏相关系数
        precision = estimator.precision_
        diag = np.diag(precision)
        partial_corr = -precision / np.sqrt(np.outer(diag, diag))
        np.fill_diagonal(partial_corr, 1.0)

        return pd.DataFrame(
            partial_corr, index=returns_df.columns, columns=returns_df.columns
        )

    def _calculate_mutual_information(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """计算互信息矩阵

        Args:
            returns_df: 收益率DataFrame

        Returns:
            互信息矩阵
        """
        from sklearn.feature_selection import mutual_info_regression

        n_stocks = len(returns_df.columns)
        mi_matrix = np.zeros((n_stocks, n_stocks))

        for i in range(n_stocks):
            for j in range(i, n_stocks):
                if i == j:
                    mi_matrix[i, j] = 1.0
                else:
                    mi = mutual_info_regression(
                        returns_df.iloc[:, i].values.reshape(-1, 1),
                        returns_df.iloc[:, j].values,
                        random_state=42,
                    )[0]
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi

        # 归一化到[0, 1]
        mi_matrix = mi_matrix / mi_matrix.max()

        return pd.DataFrame(
            mi_matrix, index=returns_df.columns, columns=returns_df.columns
        )

    def _get_edge_index(self) -> torch.Tensor:
        """获取边索引张量

        Returns:
            边索引张量
        """
        edges = list(self.graph.edges())
        edge_index = []

        for u, v in edges:
            i = self.node_mapping[u]
            j = self.node_mapping[v]
            edge_index.append([i, j])
            edge_index.append([j, i])  # 无向图，添加反向边

        return torch.LongTensor(edge_index).t()

    def _prepare_node_features(self, node_features: pd.DataFrame) -> torch.Tensor:
        """准备节点特征张量

        Args:
            node_features: 节点特征DataFrame

        Returns:
            特征张量
        """
        # 确保节点顺序一致
        features = []
        for node in self.graph.nodes():
            if node in node_features.index:
                features.append(node_features.loc[node].values)
            else:
                # 使用零向量填充缺失的特征
                features.append(np.zeros(node_features.shape[1]))

        return torch.FloatTensor(np.array(features))

    def _calculate_graph_features(self) -> Dict[str, float]:
        """计算图级别特征

        Returns:
            图特征字典
        """
        features = {}

        # 基本统计
        features["num_nodes"] = self.graph.number_of_nodes()
        features["num_edges"] = self.graph.number_of_edges()
        features["density"] = nx.density(self.graph)

        # 连通性
        features["num_components"] = nx.number_connected_components(self.graph)
        features["largest_component_size"] = len(
            max(nx.connected_components(self.graph), key=len)
        )

        # 聚类系数
        features["avg_clustering"] = nx.average_clustering(self.graph)
        features["transitivity"] = nx.transitivity(self.graph)

        # 路径长度
        if nx.is_connected(self.graph):
            features["avg_shortest_path"] = nx.average_shortest_path_length(self.graph)
            features["diameter"] = nx.diameter(self.graph)
        else:
            features["avg_shortest_path"] = -1
            features["diameter"] = -1

        # 度统计
        degrees = [d for n, d in self.graph.degree()]
        features["avg_degree"] = np.mean(degrees)
        features["max_degree"] = np.max(degrees)
        features["min_degree"] = np.min(degrees)
        features["std_degree"] = np.std(degrees)

        return features

    def _train_model(
        self, x: torch.Tensor, edge_index: torch.Tensor, train_ratio: float
    ) -> None:
        """训练图神经网络模型

        Args:
            x: 节点特征
            edge_index: 边索引
            train_ratio: 训练集比例
        """
        # 创建数据对象
        data = Data(x=x, edge_index=edge_index)

        # 划分训练集和验证集
        num_nodes = x.size(0)
        num_train = int(num_nodes * train_ratio)

        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:num_train]] = True
        val_mask = ~train_mask

        # 优化器
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # 训练循环
        self.model.train()
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()

            # 前向传播
            embeddings = self.model(data.x, data.edge_index)

            # 使用重构损失
            pos_edge_index = data.edge_index[:, ::2]  # 正样本边

            # 负样本采样
            neg_edge_index = self._negative_sampling(
                edge_index, num_nodes, pos_edge_index.size(1)
            )

            # 计算损失
            loss = self._link_prediction_loss(
                embeddings, pos_edge_index, neg_edge_index
            )

            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss={loss.item():.4f}")

    def _negative_sampling(
        self, edge_index: torch.Tensor, num_nodes: int, num_neg_samples: int
    ) -> torch.Tensor:
        """负采样

        Args:
            edge_index: 边索引
            num_nodes: 节点数
            num_neg_samples: 负样本数

        Returns:
            负样本边索引
        """
        neg_edges = []
        edge_set = set(map(tuple, edge_index.t().numpy()))

        while len(neg_edges) < num_neg_samples:
            i = np.random.randint(0, num_nodes)
            j = np.random.randint(0, num_nodes)

            if i != j and (i, j) not in edge_set and (j, i) not in edge_set:
                neg_edges.append([i, j])

        return torch.LongTensor(neg_edges).t()

    def _link_prediction_loss(
        self,
        embeddings: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """链接预测损失

        Args:
            embeddings: 节点嵌入
            pos_edge_index: 正样本边
            neg_edge_index: 负样本边

        Returns:
            损失值
        """
        # 正样本得分
        pos_src = embeddings[pos_edge_index[0]]
        pos_dst = embeddings[pos_edge_index[1]]
        pos_scores = (pos_src * pos_dst).sum(dim=1)

        # 负样本得分
        neg_src = embeddings[neg_edge_index[0]]
        neg_dst = embeddings[neg_edge_index[1]]
        neg_scores = (neg_src * neg_dst).sum(dim=1)

        # 二元交叉熵损失
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones(pos_scores.size(0))
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros(neg_scores.size(0))
        )

        return pos_loss + neg_loss


# 模块级别函数
def extract_graph_features(
    returns_df: pd.DataFrame,
    node_features: pd.DataFrame,
    config: Optional[GraphConfig] = None,
) -> GraphEmbedding:
    """提取图特征的便捷函数

    Args:
        returns_df: 收益率DataFrame
        node_features: 节点特征DataFrame
        config: 图配置

    Returns:
        图嵌入结果
    """
    extractor = GraphEmbeddingExtractor(config)
    extractor.build_stock_correlation_graph(returns_df)
    return extractor.extract_graph_embeddings(node_features)
