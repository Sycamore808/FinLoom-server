"""
特征工程模块初始化文件
"""

from .factor_discovery.factor_analyzer import FactorAnalyzer
from .factor_discovery.neural_factor_discovery import (
    NeuralFactorDiscovery,
    discover_factors,
)
from .feature_extraction.technical_indicators import (
    TechnicalIndicators,
    calculate_technical_indicators,
)
from .graph_features.graph_analyzer import GraphAnalyzer
from .storage_management import (
    FeatureCacheManager,
    FeatureDatabaseManager,
    get_feature_database_manager,
)
from .temporal_features.time_series_features import TimeSeriesFeatures

# 可选的图嵌入功能 (暂时禁用由于PyTorch依赖问题)
# try:
#     from .graph_features.graph_embeddings import (
#         GraphEmbeddingExtractor,
#     )
#     GRAPH_EMBEDDINGS_AVAILABLE = True
# except ImportError:
GRAPH_EMBEDDINGS_AVAILABLE = False
GraphEmbeddingExtractor = None


class FeatureEngineeringPipeline:
    """特征工程主流水线"""

    def __init__(self):
        """初始化特征工程流水线"""
        self.technical_indicators = TechnicalIndicators()
        self.factor_analyzer = FactorAnalyzer()
        self.time_series_features = TimeSeriesFeatures()
        self.graph_analyzer = GraphAnalyzer()
        self.db_manager = get_feature_database_manager()

    def process_features(self, data):
        """处理特征"""
        return {
            "technical": self.technical_indicators.calculate_all_indicators(data),
            "time_series": self.time_series_features.extract_features(data),
            "graph": self.graph_analyzer.analyze_graph_features(data),
        }


__all__ = [
    "TechnicalIndicators",
    "calculate_technical_indicators",
    "FactorAnalyzer",
    "NeuralFactorDiscovery",
    "discover_factors",
    "TimeSeriesFeatures",
    "GraphAnalyzer",
    "FeatureDatabaseManager",
    "get_feature_database_manager",
    "FeatureCacheManager",
    "FeatureEngineeringPipeline",
    "GRAPH_EMBEDDINGS_AVAILABLE",
]

# 添加可用的图嵌入组件
if GRAPH_EMBEDDINGS_AVAILABLE:
    __all__.extend(["GraphEmbeddingExtractor"])
