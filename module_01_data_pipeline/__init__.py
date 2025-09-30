"""
数据管道模块初始化文件

主要功能：
1. 数据采集 - 支持多市场数据获取（A股、港股、美股、加密货币）
2. 数据处理 - 数据清洗、验证和实时处理
3. 数据存储 - 统一的数据库管理和缓存
4. 流处理 - 实时数据流处理和信号生成
"""

# 数据采集模块
from .data_acquisition.akshare_collector import (
    AkshareDataCollector,
    create_akshare_collector,
    fetch_stock_data_batch,
)
from .data_acquisition.market_data_collector import (
    MarketDataCollector,
    collect_market_data,
    collect_realtime_data,
)

# 尝试导入可选模块
try:
    from .data_acquisition.alternative_data_collector import AlternativeDataCollector
except ImportError:
    AlternativeDataCollector = None

try:
    from .data_acquisition.fundamental_collector import (
        FundamentalCollector as FundamentalDataCollector,
    )
except ImportError:
    FundamentalDataCollector = None

# 数据处理模块
from .data_processing.data_cleaner import (
    DataCleaner,
    create_data_cleaner,
    quick_clean_data,
)
from .data_processing.data_validator import (
    DataQualityMetrics,
    DataValidator,
    ValidationResult,
    ensure_data_quality,
    validate_dataframe,
)
from .data_processing.real_time_processor import (
    MarketSignal,
    PortfolioMetrics,
    RealTimeProcessor,
)

# 尝试导入可选模块
try:
    from .data_processing.data_transformer import DataTransformer
except ImportError:
    DataTransformer = None

# 存储管理模块
from .storage_management.database_manager import (
    DatabaseManager,
    create_database_manager,
    get_database_manager,
)

# 尝试导入可选模块
try:
    from .storage_management.cache_manager import CacheManager
except ImportError:
    CacheManager = None

try:
    from .storage_management.file_storage import FileStorageManager
except ImportError:
    FileStorageManager = None

# 流处理模块
# 尝试导入可选模块
try:
    from .stream_processing.kafka_handler import KafkaHandler
except ImportError:
    KafkaHandler = None

try:
    from .stream_processing.stream_processor import StreamProcessor
except ImportError:
    StreamProcessor = None

__all__ = [
    # 数据采集
    "AkshareDataCollector",
    "create_akshare_collector",
    "fetch_stock_data_batch",
    "MarketDataCollector",
    "collect_market_data",
    "collect_realtime_data",
    # 数据处理
    "DataCleaner",
    "create_data_cleaner",
    "quick_clean_data",
    "DataValidator",
    "ValidationResult",
    "DataQualityMetrics",
    "validate_dataframe",
    "ensure_data_quality",
    "RealTimeProcessor",
    "MarketSignal",
    "PortfolioMetrics",
    # 存储管理
    "DatabaseManager",
    "get_database_manager",
    "create_database_manager",
]

# 添加可选模块到导出列表
if AlternativeDataCollector is not None:
    __all__.append("AlternativeDataCollector")
if FundamentalDataCollector is not None:
    __all__.append("FundamentalDataCollector")
if DataTransformer is not None:
    __all__.append("DataTransformer")
if CacheManager is not None:
    __all__.append("CacheManager")
if FileStorageManager is not None:
    __all__.append("FileStorageManager")
if KafkaHandler is not None:
    __all__.append("KafkaHandler")
if StreamProcessor is not None:
    __all__.append("StreamProcessor")

# 版本信息
__version__ = "1.0.0"
__author__ = "FinLoom Team"
__description__ = "Financial data pipeline module for real-time and batch processing"
