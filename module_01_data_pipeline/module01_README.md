# Module 01 - 数据管道模块

## 概述

数据管道模块是 FinLoom 量化交易系统的核心组件，负责金融数据的采集、处理、验证和存储。该模块支持多市场数据获取，包括中国A股、港股、美股和加密货币等。

## 主要功能

### 1. 数据采集 (Data Acquisition)
- **AkshareDataCollector**: 专门用于获取中国股票数据
- **MarketDataCollector**: 通用市场数据采集器，支持多市场
- **AlternativeDataCollector**: 另类数据采集
- **FundamentalDataCollector**: 基本面数据采集

### 2. 数据处理 (Data Processing)
- **DataCleaner**: 数据清洗和预处理
- **DataValidator**: 数据质量验证
- **RealTimeProcessor**: 实时数据处理和信号生成
- **DataTransformer**: 数据转换

### 3. 存储管理 (Storage Management)
- **DatabaseManager**: 统一数据库管理
- **CacheManager**: 内存缓存管理
- **FileStorageManager**: 文件存储管理

### 4. 流处理 (Stream Processing)
- **KafkaHandler**: Kafka消息队列处理
- **StreamProcessor**: 实时数据流处理

## 快速开始

### 基本用法

```python
import asyncio
from datetime import datetime, timedelta
from module_01_data_pipeline import (
    AkshareDataCollector,
    DataCleaner,
    DatabaseManager,
    get_database_manager
)

# 1. 创建数据收集器
collector = AkshareDataCollector(rate_limit=0.5)

# 2. 获取股票数据
symbols = ["000001", "600000", "000858"]
start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
end_date = datetime.now().strftime("%Y%m%d")

# 获取历史数据
for symbol in symbols:
    data = collector.fetch_stock_history(symbol, start_date, end_date)
    print(f"{symbol}: {len(data)} records")

# 3. 数据清洗
cleaner = DataCleaner(fill_method="interpolate")
cleaned_data = cleaner.clean_market_data(data, symbol)

# 4. 存储到数据库
db_manager = get_database_manager()
success = db_manager.save_stock_prices(symbol, cleaned_data)
```

### 异步数据收集

```python
from module_01_data_pipeline import collect_market_data, collect_realtime_data

async def async_data_collection():
    symbols = ["000001", "600036", "000858"]
    
    # 异步获取历史数据
    historical_data = await collect_market_data(symbols, lookback_days=30)
    
    # 异步获取实时数据
    realtime_data = await collect_realtime_data(symbols)
    
    return historical_data, realtime_data

# 运行异步函数
historical, realtime = asyncio.run(async_data_collection())
```

### 实时数据处理

```python
from module_01_data_pipeline import RealTimeProcessor

# 创建实时处理器
processor = RealTimeProcessor(config={})

# 添加信号回调
def signal_callback(symbol, signals):
    for signal in signals:
        print(f"信号: {symbol} - {signal.signal_type} (强度: {signal.strength:.2f})")

processor.add_signal_callback(signal_callback)

# 更新市场数据并生成信号
processor.update_market_data(symbol, market_data)
signals = processor.generate_signals(symbol)
```

## API 参考

### AkshareDataCollector

专门用于采集中国股票数据的收集器。

#### 构造函数
```python
AkshareDataCollector(rate_limit: float = 0.1)
```

#### 主要方法

**fetch_stock_list(market: str = "A股") -> pd.DataFrame**
- 获取股票列表
- 支持 A股、港股、美股

**fetch_stock_history(symbol: str, start_date: str, end_date: str, period: str = "daily", adjust: str = "qfq") -> pd.DataFrame**
- 获取股票历史数据
- 自动进行数据标准化

**fetch_realtime_data(symbols: List[str]) -> Dict[str, Dict[str, Any]]**
- 获取实时行情数据
- 支持批量查询

**get_stock_basic_info(symbol: str) -> Dict[str, Any]**
- 获取股票基本信息
- 包括行业、地区等信息

**fetch_financial_data(symbol: str, report_type: str = "资产负债表") -> pd.DataFrame**
- 获取财务数据
- 支持资产负债表、利润表、现金流量表

#### 示例
```python
collector = AkshareDataCollector(rate_limit=0.5)

# 获取股票列表
stock_list = collector.fetch_stock_list("A股")

# 获取历史数据
data = collector.fetch_stock_history("000001", "20240101", "20241231")

# 获取实时数据
realtime = collector.fetch_realtime_data(["000001", "600000"])

# 获取基本信息
info = collector.get_stock_basic_info("000001")
```

### MarketDataCollector

通用市场数据采集器，支持多市场数据获取。

#### 构造函数
```python
MarketDataCollector()
```

#### 主要方法

**fetch_historical_data(symbol: str, start_date: datetime, end_date: datetime, interval: str = "1d", market: str = "US") -> pd.DataFrame**
- 获取历史数据
- 自动检测市场类型

**fetch_realtime_data(symbols: List[str], market: str = "US") -> Dict[str, MarketData]**
- 获取实时数据
- 返回 MarketData 对象

#### 示例
```python
collector = MarketDataCollector()
await collector.initialize()

# 获取A股数据
cn_data = collector.fetch_historical_data(
    "000001", 
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    market="CN"
)

# 获取美股数据
us_data = collector.fetch_historical_data(
    "AAPL",
    start_date=datetime(2024, 1, 1), 
    end_date=datetime(2024, 12, 31),
    market="US"
)

await collector.cleanup()
```

### DataCleaner

数据清洗和预处理工具。

#### 构造函数
```python
DataCleaner(
    fill_method: str = "forward",
    outlier_method: str = "iqr", 
    outlier_threshold: float = 3.0
)
```

#### 主要方法

**clean_market_data(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame**
- 清洗市场数据
- 处理缺失值、异常值、重复数据

**detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, Any]**
- 检测数据质量问题
- 返回详细的质量报告

#### 示例
```python
cleaner = DataCleaner(
    fill_method="interpolate",
    outlier_method="iqr",
    outlier_threshold=3.0
)

# 清洗数据
cleaned_data = cleaner.clean_market_data(raw_data, "000001")

# 检测质量问题
quality_report = cleaner.detect_data_quality_issues(cleaned_data)
print(f"质量评分: {quality_report['quality_score']:.2f}")
```

### DataValidator

数据质量验证工具。

#### 构造函数
```python
DataValidator()
```

#### 主要方法

**validate_market_data(df: pd.DataFrame, symbol: Optional[str] = None) -> ValidationResult**
- 验证市场数据质量
- 返回验证结果和详细统计

#### 示例
```python
validator = DataValidator()

# 验证数据
result = validator.validate_market_data(data, "000001")

print(f"验证通过: {result.is_valid}")
print(f"质量分数: {result.quality_score:.2f}")
if result.issues:
    print(f"问题: {result.issues}")
```

### DatabaseManager

统一数据库管理工具。

#### 构造函数
```python
DatabaseManager(db_path: str = "data/finloom.db")
```

#### 主要方法

**save_stock_prices(symbol: str, df: pd.DataFrame) -> bool**
- 保存股票价格数据

**get_stock_prices(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame**
- 获取股票价格数据

**save_stock_info(symbol: str, name: str, **kwargs)**
- 保存股票基本信息

**get_database_stats() -> Dict[str, Any]**
- 获取数据库统计信息

#### 示例
```python
# 获取全局数据库管理器
db_manager = get_database_manager()

# 保存数据
success = db_manager.save_stock_prices("000001", price_data)

# 查询数据
data = db_manager.get_stock_prices("000001", "2024-01-01", "2024-12-31")

# 获取统计信息
stats = db_manager.get_database_stats()
print(f"数据库大小: {stats['database_size_mb']:.2f} MB")
```

### RealTimeProcessor

实时数据处理和信号生成器。

#### 构造函数
```python
RealTimeProcessor(config: Dict)
```

#### 主要方法

**update_market_data(symbol: str, data: pd.DataFrame)**
- 更新市场数据

**generate_signals(symbol: str) -> List[MarketSignal]**
- 生成交易信号

**add_signal_callback(callback: Callable)**
- 添加信号回调函数

#### 示例
```python
processor = RealTimeProcessor(config={})

# 添加信号回调
def signal_handler(symbol, signals):
    for signal in signals:
        print(f"{symbol}: {signal.signal_type} @{signal.price:.2f}")

processor.add_signal_callback(signal_handler)

# 更新数据并生成信号
processor.update_market_data("000001", market_data)
signals = processor.generate_signals("000001")
```

## 便捷函数

### collect_market_data
异步批量收集历史数据
```python
data = await collect_market_data(
    symbols=["000001", "600000"],
    lookback_days=30,
    market="AUTO"  # 自动检测市场
)
```

### collect_realtime_data
异步批量收集实时数据
```python
realtime_data = await collect_realtime_data(
    symbols=["000001", "600000"],
    market="AUTO"
)
```

### quick_clean_data
快速清洗数据
```python
cleaned_data = quick_clean_data(raw_data, symbol="000001")
```

### validate_dataframe
快速验证数据
```python
result = validate_dataframe(data, data_type="market")
```

## 配置说明

### 环境变量
- `FINLOOM_DB_PATH`: 数据库文件路径
- `FINLOOM_CACHE_SIZE`: 缓存大小限制
- `FINLOOM_LOG_LEVEL`: 日志级别

### 数据库配置
默认使用 SQLite 数据库，文件位于 `data/finloom.db`。可以通过以下方式自定义：

```python
db_manager = create_database_manager("custom/path/database.db")
```

## 错误处理

模块使用自定义异常类型：

- `DataError`: 数据相关错误
- `ValidationError`: 验证错误
- `ConfigError`: 配置错误

```python
from common.exceptions import DataError

try:
    data = collector.fetch_stock_history("INVALID", start_date, end_date)
except DataError as e:
    print(f"数据获取失败: {e}")
```

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
collector = AkshareDataCollector(rate_limit=1.0)  # 降低请求频率
```