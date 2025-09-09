"""
图特征分析器模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from common.logging_system import setup_logger
from common.exceptions import DataError

logger = setup_logger("graph_analyzer")

@dataclass
class GraphFeature:
    """图特征"""
    name: str
    values: Dict[str, float]
    description: str

class GraphAnalyzer:
    """图特征分析器类"""
    
    def __init__(self):
        """初始化图分析器"""
        pass
    
    def build_correlation_graph(self, returns_matrix: pd.DataFrame, threshold: float = 0.3) -> Dict[str, List[str]]:
        """构建相关性图
        
        Args:
            returns_matrix: 收益率矩阵
            threshold: 相关性阈值
            
        Returns:
            图的邻接表表示
        """
        try:
            # 计算相关性矩阵
            corr_matrix = returns_matrix.corr()
            
            # 构建图
            graph = {}
            for asset in corr_matrix.columns:
                graph[asset] = []
                for other_asset in corr_matrix.columns:
                    if asset != other_asset and abs(corr_matrix.loc[asset, other_asset]) > threshold:
                        graph[asset].append(other_asset)
            
            logger.info(f"Built correlation graph with {len(graph)} nodes")
            return graph
            
        except Exception as e:
            logger.error(f"Failed to build correlation graph: {e}")
            raise DataError(f"Graph construction failed: {e}")
    
    def calculate_centrality_measures(self, graph: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """计算中心性指标
        
        Args:
            graph: 图的邻接表表示
            
        Returns:
            中心性指标字典
        """
        try:
            centrality_measures = {}
            
            for node in graph:
                # 度中心性
                degree_centrality = len(graph[node])
                
                # 简化的接近中心性
                closeness_centrality = 1.0 / (degree_centrality + 1)
                
                # 简化的介数中心性
                betweenness_centrality = degree_centrality * 0.1
                
                centrality_measures[node] = {
                    'degree': degree_centrality,
                    'closeness': closeness_centrality,
                    'betweenness': betweenness_centrality
                }
            
            return centrality_measures
            
        except Exception as e:
            logger.error(f"Failed to calculate centrality measures: {e}")
            raise DataError(f"Centrality calculation failed: {e}")
    
    def extract_graph_features(self, returns_matrix: pd.DataFrame) -> Dict[str, GraphFeature]:
        """提取图特征
        
        Args:
            returns_matrix: 收益率矩阵
            
        Returns:
            图特征字典
        """
        try:
            features = {}
            
            # 构建相关性图
            graph = self.build_correlation_graph(returns_matrix)
            
            # 计算中心性指标
            centrality_measures = self.calculate_centrality_measures(graph)
            
            # 创建图特征
            for asset in returns_matrix.columns:
                if asset in centrality_measures:
                    features[f'graph_centrality_{asset}'] = GraphFeature(
                        name=f'graph_centrality_{asset}',
                        values=centrality_measures[asset],
                        description=f'Graph centrality measures for {asset}'
                    )
            
            logger.info(f"Extracted {len(features)} graph features")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract graph features: {e}")
            raise DataError(f"Graph feature extraction failed: {e}")
