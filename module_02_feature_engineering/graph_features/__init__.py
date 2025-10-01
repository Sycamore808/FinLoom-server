#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图特征模块初始化文件
"""

from .community_detection import (
    CommunityConfig,
    CommunityDetection,
    detect_stock_communities,
)
from .graph_analyzer import GraphAnalyzer
from .graph_analyzer import GraphConfig as AnalyzerConfig
from .stock_graph_builder import (
    GraphConfig,
    StockGraphBuilder,
    build_stock_correlation_graph,
)

# from .graph_embeddings import GraphEmbeddings, EmbeddingConfig  # Temporarily disabled due to PyTorch dependency

__all__ = [
    "GraphAnalyzer",
    "AnalyzerConfig",
    "StockGraphBuilder",
    "GraphConfig",
    "build_stock_correlation_graph",
    "CommunityDetection",
    "CommunityConfig",
    "detect_stock_communities",
    # 'GraphEmbeddings',
    # 'EmbeddingConfig'
]
