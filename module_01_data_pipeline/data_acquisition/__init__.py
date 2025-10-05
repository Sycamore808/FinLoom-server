"""
数据采集模块初始化文件

主要采集器:
- AkshareDataCollector: 专门用于中国股票市场数据采集，使用akshare库

- ChineseAlternativeDataCollector: 中国市场另类数据采集（宏观、情绪、新闻等）
- ChineseFundamentalCollector: 中国上市公司财务数据采集
"""

from .akshare_collector import AkshareDataCollector
from .alternative_data_collector import ChineseAlternativeDataCollector
from .fundamental_collector import ChineseFundamentalCollector

__all__ = [
    "AkshareDataCollector",
    "ChineseAlternativeDataCollector",
    "ChineseFundamentalCollector",
]
