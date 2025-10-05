# Module 11 - 可视化模块 API文档

## 模块概述

可视化模块(module_11_visualization)负责FinLoom量化交易系统的所有数据可视化功能，包括图表生成、交互式仪表板、报告构建和数据导出。

## 目录结构

```
module_11_visualization/
├── __init__.py                    # 模块导出
├── chart_generator.py             # 图表生成器
├── dashboard_manager.py           # 仪表板管理器
├── interactive_visualizer.py      # 交互式可视化
├── report_builder.py              # 报告生成器（输出JSON/CSV/Excel）
├── template_engine.py             # 模板引擎
├── export_manager.py              # 导出管理器
└── database_manager.py            # 数据库管理器
```

## API 快速开始

### 1. 导入模块

```python
from module_11_visualization import (
    ChartGenerator,
    DashboardManager,
    ReportBuilder,
    InteractiveVisualizer,
    ExportManager,
    get_visualization_database_manager
)
```

### 2. 基础图表生成

```python
import pandas as pd
from datetime import datetime, timedelta

# 初始化图表生成器
chart_gen = ChartGenerator(default_theme="dark")

# 准备数据（从其他模块获取）
from module_01_data_pipeline.storage_management import get_database_manager
db_manager = get_database_manager()
stock_data = db_manager.get_stock_prices(
    symbol='000001', 
    start_date='2024-01-01', 
    end_date='2024-12-31'
)

# 生成K线图
candlestick = chart_gen.generate_candlestick_chart(
    data=stock_data,
    volume_subplot=True
)

# 保存图表
chart_gen.save_chart(candlestick, "candlestick.html", format="html")
```

### 3. 交互式图表

```python
from module_11_visualization import InteractiveVisualizer

# 创建交互式可视化器
visualizer = InteractiveVisualizer()

# 创建交互式K线图（带技术指标）
interactive_chart = visualizer.create_interactive_candlestick(
    df=stock_data,
    symbol='000001',
    indicators=['rsi', 'macd'],  # 添加技术指标
    overlays=['ma', 'bollinger'],  # 添加叠加指标
    volume=True,
    annotations=[
        {'x': '2024-06-15', 'y': 15.5, 'text': '买入信号'}
    ]
)

# 导出为HTML
visualizer.export_interactive_html(
    interactive_chart, 
    'interactive_chart.html'
)
```

### 4. 绩效分析报告（JSON数据格式）

```python
from module_11_visualization import ReportBuilder, PerformanceMetrics, ReportConfig

# 获取回测数据（从module 09）
from module_09_backtesting import get_backtest_database_manager
backtest_db = get_backtest_database_manager()
backtest_result = backtest_db.get_backtest_result("backtest_20241201")

# 构建绩效指标
metrics = PerformanceMetrics(
    total_return=backtest_result['total_return'],
    annualized_return=backtest_result['annualized_return'],
    volatility=backtest_result['volatility'],
    sharpe_ratio=backtest_result['sharpe_ratio'],
    sortino_ratio=backtest_result['sortino_ratio'],
    max_drawdown=backtest_result['max_drawdown'],
    win_rate=backtest_result['win_rate'],
    profit_factor=backtest_result['profit_factor'],
    avg_win=backtest_result['avg_win'],
    avg_loss=backtest_result['avg_loss'],
    best_trade=backtest_result['best_trade'],
    worst_trade=backtest_result['worst_trade'],
    total_trades=backtest_result['total_trades'],
    winning_trades=backtest_result['winning_trades'],
    losing_trades=backtest_result['losing_trades']
)

# 生成报告（返回字典，自动保存到JSON文件和SQLite数据库）
report_builder = ReportBuilder()
result = report_builder.generate_performance_report(
    performance_data=backtest_result,
    metrics=metrics,
    config=ReportConfig(report_type='performance', output_format='json')
)

# 查看保存位置
print(f"报告ID: {result['database_report_id']}")
for location in result['saved_to']:
    print(f"✓ {location}")

# 从数据库读取报告
vis_db = get_visualization_database_manager()
report = vis_db.get_report(result['database_report_id'])
print(f"报告数据: {report['content']}")  # JSON格式的完整数据
```

### 5. 创建仪表板

```python
from module_11_visualization import DashboardManager, DashboardConfig

# 配置仪表板
config = DashboardConfig(
    title="投资组合实时监控",
    refresh_interval_ms=5000,
    layout='grid',
    theme='dark',
    components=['portfolio', 'positions', 'performance']
)

# 创建仪表板管理器
dashboard_mgr = DashboardManager(config)

# 获取持仓数据（从其他模块）
from module_05_risk_management import get_risk_database_manager
risk_db = get_risk_database_manager()
portfolio_risk = risk_db.get_portfolio_risk_history(
    portfolio_id='main_portfolio',
    start_date=datetime.now() - timedelta(days=30)
)

# 创建投资组合仪表板
dashboard_mgr.create_portfolio_dashboard(
    portfolio_data={'total_value': 1000000, 'daily_pnl': 5000},
    positions=[],  # 持仓列表
    signals=[]     # 信号列表
)

# 运行仪表板服务器
dashboard_mgr.run_server(host='127.0.0.1', port=8050)
```

## 核心API详解

### ChartGenerator - 图表生成器

#### 初始化

```python
chart_gen = ChartGenerator(default_theme="dark")
```

参数：
- `default_theme`: 默认主题 ("dark", "light", "presentation", "ggplot", "seaborn")

#### generate_candlestick_chart() - 生成K线图

```python
fig = chart_gen.generate_candlestick_chart(
    data=df,                  # pd.DataFrame 包含 open, high, low, close, volume
    indicators=None,          # Dict[str, pd.Series] 技术指标
    volume_subplot=True,      # bool 是否显示成交量子图
    config=None              # Optional[ChartConfig] 图表配置
)
```

**数据格式要求：**
```python
# DataFrame必须包含以下列：
df.columns = ['open', 'high', 'low', 'close', 'volume']
df.index = DatetimeIndex  # 日期索引
```

**返回：** `plotly.graph_objects.Figure`

#### generate_performance_chart() - 生成绩效曲线

```python
fig = chart_gen.generate_performance_chart(
    returns=returns_series,    # pd.Series 收益率序列
    benchmark=None,            # Optional[pd.Series] 基准收益率
    config=None               # Optional[ChartConfig]
)
```

**数据来源示例：**
```python
# 从回测结果获取收益率
from module_09_backtesting import get_backtest_database_manager
backtest_db = get_backtest_database_manager()
backtest_result = backtest_db.get_backtest_result("backtest_id")
returns = pd.Series(backtest_result['daily_returns'])
```

#### generate_heatmap() - 生成热力图

```python
fig = chart_gen.generate_heatmap(
    data=correlation_matrix,   # pd.DataFrame 相关性矩阵
    config=None               # Optional[ChartConfig]
)
```

**数据来源示例：**
```python
# 从市场分析模块获取相关性矩阵
from module_04_market_analysis.correlation_analysis import CorrelationCalculator
corr_calc = CorrelationCalculator()
correlation_matrix = corr_calc.calculate_correlation(returns_df)
```

#### generate_portfolio_composition() - 生成组合构成图

```python
fig = chart_gen.generate_portfolio_composition(
    positions={'AAPL': 0.3, 'GOOGL': 0.25, 'MSFT': 0.45},  # Dict 持仓权重
    config=None
)
```

#### generate_drawdown_chart() - 生成回撤图

```python
fig = chart_gen.generate_drawdown_chart(
    returns=returns_series,    # pd.Series 收益率序列
    config=None
)
```

### InteractiveVisualizer - 交互式可视化器

#### 初始化

```python
from module_11_visualization import InteractiveConfig, InteractiveVisualizer

config = InteractiveConfig(
    enable_zoom=True,
    enable_pan=True,
    enable_hover=True,
    enable_crosshair=True,
    enable_rangeselector=True,
    default_height=600,
    default_width=1200,
    theme='dark'
)

visualizer = InteractiveVisualizer(config)
```

#### create_interactive_candlestick() - 创建交互式K线图

```python
fig = visualizer.create_interactive_candlestick(
    df=stock_data,              # pd.DataFrame OHLCV数据
    symbol='000001',            # str 股票代码
    indicators=['rsi'],         # Optional[List[str]] 技术指标
    overlays=['ma'],            # Optional[List[str]] 叠加指标
    volume=True,                # bool 显示成交量
    annotations=None            # Optional[List[Dict]] 注释列表
)
```

**支持的indicators:**
- `'rsi'` - 相对强弱指数
- `'macd'` - MACD指标
- `'stochastic'` - 随机指标

**支持的overlays:**
- `'ma'` - 移动平均线 (20, 50, 200日)
- `'ema'` - 指数移动平均 (12, 26日)
- `'bollinger'` - 布林带

#### create_heatmap_matrix() - 创建热力图矩阵

```python
fig = visualizer.create_heatmap_matrix(
    data=correlation_df,        # pd.DataFrame 数据矩阵
    title='相关性矩阵',
    color_scale='RdBu',         # str 颜色方案
    show_values=True,           # bool 显示数值
    clustering=False            # bool 是否进行层次聚类
)
```

#### create_network_graph() - 创建网络图

```python
fig = visualizer.create_network_graph(
    nodes=[
        {'id': 'AAPL', 'label': 'Apple'},
        {'id': 'GOOGL', 'label': 'Google'}
    ],
    edges=[
        {'source': 'AAPL', 'target': 'GOOGL', 'weight': 0.75}
    ],
    title='股票关联网络',
    layout_type='spring'        # 'spring', 'circular', 'random', 'grid'
)
```

**应用场景：** 用于显示股票间的相关性网络、行业关联图等

#### create_animated_timeline() - 创建动画时间线

```python
fig = visualizer.create_animated_timeline(
    df=timeseries_data,         # pd.DataFrame 时序数据
    date_column='date',         # str 日期列名
    value_columns=['value1', 'value2'],  # List[str] 数值列
    title='资产增长动画',
    animation_speed=100         # int 动画速度(毫秒)
)
```

### DashboardManager - 仪表板管理器

#### 初始化

```python
from module_11_visualization import DashboardConfig, DashboardManager

config = DashboardConfig(
    title="实时监控仪表板",
    refresh_interval_ms=5000,   # int 刷新间隔(毫秒)
    layout='grid',              # str 布局类型 'grid', 'tabs', 'single'
    theme='dark',               # str 主题 'dark', 'light', 'auto'
    components=['portfolio', 'risk'],
    auto_refresh=True
)

dashboard_mgr = DashboardManager(config)
```

#### create_portfolio_dashboard() - 创建投资组合仪表板

```python
dashboard_app = dashboard_mgr.create_portfolio_dashboard(
    portfolio_data={
        'total_value': 1000000.0,
        'daily_pnl': 5000.0,
        'daily_return': 0.005,
        'ytd_return': 0.15
    },
    positions=positions_list,   # List[Position] 持仓列表
    signals=signals_list        # List[Signal] 信号列表
)
```

**数据来源：**

```python
# 从风险管理模块获取组合数据
from module_05_risk_management import PortfolioRiskAnalyzer
from module_05_risk_management.database_manager import get_risk_database_manager

risk_db = get_risk_database_manager()
portfolio_data = risk_db.get_portfolio_risk_history('portfolio_1')
```

#### 运行仪表板服务器

```python
dashboard_mgr.run_server(
    host='127.0.0.1',
    port=8050,
    debug=False
)
```

访问：`http://127.0.0.1:8050`

### ReportBuilder - 报告生成器 ⚠️ **新版API - 输出JSON/CSV/Excel + SQLite**

#### 初始化

```python
report_builder = ReportBuilder()
# 初始化时会自动显示输出目录和数据库路径
# 输出:
#   - 数据输出目录: /path/to/reports
#   - SQLite数据库: data/module11_visualization.db
```

#### generate_daily_report() - 生成日报（默认JSON + SQLite）

**新版API返回字典而不是HTML字符串！**

```python
from datetime import datetime
from module_11_visualization import ReportConfig

# 默认生成JSON格式并保存到SQLite
result = report_builder.generate_daily_report(
    date=datetime.now(),
    portfolio_data={
        'total_value': 1000000,
        'daily_pnl': 5000,
        'daily_return': 0.005,
        'ytd_return': 0.15,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.08,
        'win_rate': 0.65,
        'profit_factor': 2.1
    },
    positions=positions_list,   # List[Position]
    signals=signals_list,       # List[Signal]
    market_data=market_df,      # pd.DataFrame
    config=ReportConfig(report_type="daily", output_format="json")  # 可选，默认JSON
)

# 返回结果格式
print(result)
# {
#     "success": True,
#     "report_data": {...},  # 完整的报告数据
#     "saved_to": [
#         "SQLite数据库: data/module11_visualization.db",
#         "JSON文件: /path/to/reports/daily_report_20241201_143025.json"
#     ],
#     "database_report_id": "daily_20241201_143025",
#     "file_path": "/path/to/reports/daily_report_20241201_143025.json",
#     "errors": []
# }

# 查看保存位置
for location in result['saved_to']:
    print(f"✓ {location}")
```

**生成CSV格式报告：**

```python
csv_result = report_builder.generate_daily_report(
    date=datetime.now(),
    portfolio_data=portfolio_data,
    positions=positions_list,
    signals=signals_list,
    market_data=market_df,
    config=ReportConfig(report_type="daily", output_format="csv")
)
# 输出: CSV文件 + SQLite数据库
```

**生成Excel格式报告：**

```python
excel_result = report_builder.generate_daily_report(
    date=datetime.now(),
    portfolio_data=portfolio_data,
    positions=positions_list,
    signals=signals_list,
    market_data=market_df,
    config=ReportConfig(report_type="daily", output_format="excel")
)
# 输出: Excel文件（多sheet）+ SQLite数据库
```

#### generate_performance_report() - 生成绩效报告（默认JSON + SQLite）

```python
from module_11_visualization import PerformanceMetrics, ReportConfig

metrics = PerformanceMetrics(
    total_return=0.25,
    annualized_return=0.20,
    volatility=0.15,
    sharpe_ratio=1.33,
    sortino_ratio=1.8,
    max_drawdown=-0.12,
    win_rate=0.60,
    profit_factor=2.0,
    avg_win=500,
    avg_loss=-250,
    best_trade=2000,
    worst_trade=-800,
    total_trades=100,
    winning_trades=60,
    losing_trades=40
)

# 返回字典而不是HTML
result = report_builder.generate_performance_report(
    performance_data={
        'returns': {'daily': 0.005, 'monthly': 0.02, 'yearly': 0.25},
        'risk': {'volatility': 0.15, 'var_95': -0.05}
    },
    metrics=metrics,
    config=ReportConfig(report_type="performance", output_format="json")
)

# 查看结果
print(f"报告ID: {result['database_report_id']}")
print(f"保存位置:")
for location in result['saved_to']:
    print(f"  - {location}")
```

**报告数据结构（JSON格式）：**

```python
# result['report_data'] 包含以下结构：
{
    "metadata": {
        "report_date": "2024-12-01",
        "generation_time": "2024-12-01 14:30:25",
        "report_type": "Daily Report"
    },
    "summary": {
        "date": "2024-12-01",
        "total_value": 1000000.0,
        "daily_pnl": 5000.0,
        "daily_return": 0.005,
        "ytd_return": 0.15
    },
    "positions": [
        {
            "symbol": "000001",
            "quantity": 1000,
            "avg_cost": 10.5,
            "current_price": 11.2,
            "market_value": 11200,
            "unrealized_pnl": 700
        }
    ],
    "signals": [...],
    "market_overview": {...},
    "performance": {...}
}
```

#### 从数据库读取报告

```python
vis_db = get_visualization_database_manager()

# 读取特定报告
report = vis_db.get_report("daily_20241201_143025")
if report:
    print(f"报告标题: {report['title']}")
    print(f"报告类型: {report['report_type']}")
    print(f"报告日期: {report['report_date']}")
    print(f"报告数据: {report['content']}")  # JSON格式的完整数据

# 列出所有日报
daily_reports = vis_db.list_reports(report_type="daily", limit=10)
for r in daily_reports:
    print(f"{r['report_date']}: {r['title']}")
```

#### ReportConfig 配置参数

```python
from module_11_visualization import ReportConfig

config = ReportConfig(
    report_type="daily",           # 报告类型: daily/weekly/monthly/performance
    output_format="json",          # 输出格式: json/csv/excel/sqlite（默认json）
    output_path=None,              # 自定义输出路径（可选）
    save_to_database=True,         # 是否保存到SQLite（默认True）
    include_charts=False,          # 是否包含图表（纯数据模式默认False）
    include_tables=True,           # 是否包含表格
    include_summary=True,          # 是否包含概要
)
```

#### 快速生成报告

```python
from module_11_visualization.report_builder import generate_quick_report

# 快速生成自定义报告
data = {
    "summary": {"total_value": 1000000, "daily_pnl": 5000},
    "performance": {"sharpe_ratio": 1.5, "max_drawdown": -0.08}
}

result = generate_quick_report(
    data=data,
    report_type="custom",
    output_format="json"  # json/csv/excel
)

print(f"保存到: {result['saved_to']}")
```

#### ⚠️ 重要变更说明

**旧版API（已弃用）：**
```python
# ❌ 不再推荐
report_html = report_builder.generate_daily_report(...)  # 返回HTML字符串
```

**新版API（推荐使用）：**
```python
# ✅ 推荐
result = report_builder.generate_daily_report(...)  # 返回字典，包含保存信息
print(result['saved_to'])  # 明确告知保存位置
```

### ExportManager - 导出管理器

#### 初始化

```python
from module_11_visualization import ExportManager

export_mgr = ExportManager(default_output_dir='exports')
```

#### export_dataframe() - 导出DataFrame

```python
from module_11_visualization import ExportConfig

config = ExportConfig(
    export_format='excel',
    output_path='exports/portfolio_data.xlsx',
    compression=None,
    include_index=False
)

result = export_mgr.export_dataframe(
    df=portfolio_df,
    filename='portfolio_data.xlsx',
    format='excel',
    config=config
)

if result.success:
    print(f"导出成功: {result.file_path}")
    print(f"文件大小: {result.file_size} bytes")
```

**支持的格式：**
- `'csv'` - CSV文件
- `'excel'` - Excel文件
- `'json'` - JSON文件
- `'parquet'` - Parquet文件
- `'hdf5'` - HDF5文件
- `'html'` - HTML文件

#### export_multiple() - 导出多个数据集

```python
data_dict = {
    '持仓明细': positions_df,
        '交易记录': trades_df,
        '绩效指标': metrics_df
}

result = export_mgr.export_multiple(
    data_dict=data_dict,
    base_filename='portfolio_analysis',
    format='excel',
    create_archive=False
)
```

#### export_chart_to_image() - 导出图表为图片

```python
result = export_mgr.export_chart_to_image(
    chart=fig,
    filename='chart.png',
    format='png',
    width=1200,
    height=600,
    scale=2
)
```

### VisualizationDatabaseManager - 数据库管理器

#### 获取管理器实例

```python
from module_11_visualization import get_visualization_database_manager

vis_db = get_visualization_database_manager()
```

#### 保存图表

```python
vis_db.save_chart(
    chart_id='chart_20241201_001',
    chart_type='candlestick',
    title='平安银行K线图',
    data_source='module_01',
    config={'symbol': '000001', 'period': 'daily'},
    html_content=fig.to_html()
)
```

#### 获取图表

```python
chart = vis_db.get_chart('chart_20241201_001')
if chart:
    print(f"图表标题: {chart['title']}")
    print(f"创建时间: {chart['created_at']}")
```

#### 列出图表

```python
charts = vis_db.list_charts(chart_type='candlestick', limit=10)
for chart in charts:
    print(f"{chart['chart_id']}: {chart['title']}")
```

#### 保存仪表板

```python
vis_db.save_dashboard(
    dashboard_id='portfolio_dashboard_001',
    dashboard_type='portfolio',
    title='投资组合监控',
    layout_config={'rows': 4, 'cols': 12},
    components=[
        {'id': 'metric1', 'type': 'metric', 'position': (0, 0)},
        {'id': 'chart1', 'type': 'chart', 'position': (1, 0)}
    ]
)
```

#### 保存报告

```python
vis_db.save_report(
    report_id=f"daily_{datetime.now().strftime('%Y%m%d')}",
    report_type='daily',
    title='每日投资报告',
    report_date=datetime.now(),
    content_html=report_html,
    metadata={'author': 'system', 'version': '1.0'}
)
```

#### 列出报告

```python
reports = vis_db.list_reports(
    report_type='daily',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    limit=10
)
```

#### 缓存管理

```python
# 设置缓存
vis_db.set_cache(
    cache_key='portfolio_performance',
    cache_type='chart_data',
    data={'returns': [0.01, 0.02, -0.01]},
    expires_in_seconds=3600  # 1小时后过期
)

# 获取缓存
cached_data = vis_db.get_cache('portfolio_performance')
if cached_data:
    print("使用缓存数据")

# 清理过期缓存
deleted_count = vis_db.clear_expired_cache()
```

## 与其他模块的集成

### 从Module 01获取数据

```python
from module_01_data_pipeline.storage_management import get_database_manager

db_manager = get_database_manager()
stock_data = db_manager.get_stock_prices(
    symbol='000001',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# 生成K线图
chart_gen = ChartGenerator()
fig = chart_gen.generate_candlestick_chart(stock_data)
```

### 从Module 04获取市场分析数据

```python
from module_04_market_analysis.correlation_analysis import CorrelationCalculator

corr_calc = CorrelationCalculator()
correlation_matrix = corr_calc.calculate_correlation(returns_df)

# 生成热力图
fig = chart_gen.generate_heatmap(correlation_matrix)
```

### 从Module 05获取风险数据

```python
from module_05_risk_management import get_risk_database_manager

risk_db = get_risk_database_manager()
portfolio_risk = risk_db.get_portfolio_risk_history(
    portfolio_id='main_portfolio'
)

# 生成风险仪表板
dashboard_mgr.create_risk_dashboard(portfolio_risk)
```

### 从Module 09获取回测结果

```python
from module_09_backtesting import get_backtest_database_manager

backtest_db = get_backtest_database_manager()
backtest_result = backtest_db.get_backtest_result('backtest_001')

# 生成绩效报告
report_builder.generate_performance_report(
    performance_data=backtest_result,
    metrics=PerformanceMetrics(...)
)
```

## 完整示例：生成综合分析报告

```python
from datetime import datetime, timedelta
from module_11_visualization import (
    ChartGenerator,
    ReportBuilder,
    InteractiveVisualizer,
    get_visualization_database_manager
)
from module_01_data_pipeline.storage_management import get_database_manager
from module_09_backtesting import get_backtest_database_manager

# 1. 获取数据
data_db = get_database_manager()
backtest_db = get_backtest_database_manager()

stock_data = data_db.get_stock_prices('000001', '2024-01-01', '2024-12-31')
backtest_result = backtest_db.get_backtest_result('strategy_001')

# 2. 生成图表
chart_gen = ChartGenerator()
visualizer = InteractiveVisualizer()

# K线图
candlestick = visualizer.create_interactive_candlestick(
    df=stock_data,
    symbol='000001',
    indicators=['rsi', 'macd'],
    overlays=['ma', 'bollinger'],
    volume=True
)

# 绩效曲线
returns = pd.Series(backtest_result['daily_returns'])
performance_chart = chart_gen.generate_performance_chart(returns)

# 回撤分析
drawdown_chart = chart_gen.generate_drawdown_chart(returns)

# 3. 构建报告
report_builder = ReportBuilder()

# 添加自定义章节
from module_11_visualization import ReportSection

sections = [
    ReportSection(
        section_id='strategy_overview',
        title='策略概览',
        content_type='text',
        content='本报告分析了动量策略在2024年的表现...',
        order=1
    ),
    ReportSection(
        section_id='price_chart',
        title='价格走势',
        content_type='chart',
        content=candlestick.to_html(),
        order=2
    ),
    ReportSection(
        section_id='performance',
        title='绩效分析',
        content_type='chart',
        content=performance_chart.to_html(),
        order=3
    ),
    ReportSection(
        section_id='drawdown',
        title='回撤分析',
        content_type='chart',
        content=drawdown_chart.to_html(),
        order=4
    )
]

from module_11_visualization import ReportConfig

report_config = ReportConfig(
    report_type='custom',
    template_name='daily_report',
    output_format='html'
)

report_html = report_builder.create_custom_report(
    title='策略综合分析报告',
    sections=sections,
    config=report_config
)

# 4. 保存报告到数据库
vis_db = get_visualization_database_manager()
vis_db.save_report(
    report_id=f"analysis_{datetime.now().strftime('%Y%m%d')}",
    report_type='custom',
    title='策略综合分析报告',
    report_date=datetime.now(),
    content_html=report_html,
    file_path='reports/analysis_20241201.html'
)

print("报告生成完成！")
```

## 配置说明

### 图表主题配置

配置文件：`config/chart_themes.yaml`

```yaml
themes:
  dark:
    template: plotly_dark
    background_color: "#111111"
    color_scheme: ["#636EFA", "#00CC96", "#EF553B"]
```

使用：
```python
chart_gen = ChartGenerator(default_theme="dark")
```

### 仪表板布局配置

配置文件：`config/dashboard_layouts.yaml`

定义了预设的仪表板布局：
- `portfolio` - 投资组合仪表板
- `strategy` - 策略分析仪表板
- `risk` - 风险监控仪表板
- `realtime` - 实时监控仪表板

### 报告模板配置

配置文件：`config/report_templates.yaml`

定义了报告结构和章节：
- `daily_report` - 日报
- `weekly_report` - 周报
- `monthly_report` - 月报
- `backtest_report` - 回测报告
- `performance_report` - 绩效报告
- `risk_report` - 风险报告

## 数据库表结构

模块使用SQLite数据库存储可视化相关数据：

### charts表 - 图表数据
```sql
CREATE TABLE charts (
    id INTEGER PRIMARY KEY,
    chart_id TEXT UNIQUE,
    chart_type TEXT,
    title TEXT,
    data_source TEXT,
    config_json TEXT,
    html_content TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
```

### dashboards表 - 仪表板数据
```sql
CREATE TABLE dashboards (
    id INTEGER PRIMARY KEY,
    dashboard_id TEXT UNIQUE,
    dashboard_type TEXT,
    title TEXT,
    layout_config_json TEXT,
    components_json TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
```

### reports表 - 报告数据
```sql
CREATE TABLE reports (
    id INTEGER PRIMARY KEY,
    report_id TEXT UNIQUE,
    report_type TEXT,
    title TEXT,
    report_date DATE,
    content_html TEXT,
    content_json TEXT,
    metadata_json TEXT,
    file_path TEXT,
    created_at TIMESTAMP
)
```

### export_history表 - 导出历史
```sql
CREATE TABLE export_history (
    id INTEGER PRIMARY KEY,
    export_id TEXT UNIQUE,
    export_type TEXT,
    source_type TEXT,
    source_id TEXT,
    file_path TEXT,
    file_size INTEGER,
    export_format TEXT,
    metadata_json TEXT,
    created_at TIMESTAMP
)
```

### visualization_cache表 - 缓存数据
```sql
CREATE TABLE visualization_cache (
    id INTEGER PRIMARY KEY,
    cache_key TEXT UNIQUE,
    cache_type TEXT,
    data_json TEXT,
    expires_at TIMESTAMP,
    created_at TIMESTAMP
)
```

## 常见问题

### Q: 如何自定义图表样式？

A: 通过ChartConfig对象自定义：

```python
from module_11_visualization import ChartConfig

config = ChartConfig(
    chart_type='candlestick',
    title='自定义K线图',
    theme='dark',
    width=1400,
    height=700,
    show_legend=True,
    show_grid=True
)

fig = chart_gen.generate_candlestick_chart(data, config=config)
```

### Q: 如何实现实时数据更新的仪表板？

A: 使用DashboardManager的自动刷新功能：

```python
config = DashboardConfig(
    title="实时监控",
    refresh_interval_ms=1000,  # 1秒刷新一次
    auto_refresh=True
)

dashboard_mgr = DashboardManager(config)
dashboard_mgr.start_auto_refresh()
```

### Q: 导出的PDF质量如何优化？

A: 使用高DPI设置：

```python
report_builder.export_to_pdf(
    html_content=report_html,
    output_path='report.pdf',
    options={'dpi': 300, 'quality': 95}
)
```

### Q: 如何批量导出图表？

A: 使用export_chart_collection：

```python
from module_11_visualization import export_chart_collection

charts = {
    'candlestick': candlestick_fig,
    'performance': performance_fig,
    'drawdown': drawdown_fig
}

export_chart_collection(
    charts=charts,
    output_dir='charts_output',
    format='html'
)
```

## 性能优化建议

1. **使用缓存机制**
   ```python
   # 缓存计算结果
   vis_db.set_cache('expensive_calculation', 'chart_data', data, 3600)
   ```

2. **批量操作**
   ```python
   # 一次性导出多个数据集
   export_mgr.export_multiple(data_dict, 'batch_export', 'excel')
   ```

3. **异步加载**
   ```python
   # 仪表板异步更新组件
   dashboard_mgr.update_component_data('chart1', new_data)
   ```

## 测试

运行模块测试：

```bash
cd /Users/victor/Desktop/25fininnov/FinLoom-server
conda activate study
python tests/module11_visualization_test.py
```

## 版本信息

- **版本**: 1.0.0
- **作者**: FinLoom Team
- **最后更新**: 2024-12-01

## 依赖模块

本模块依赖以下其他模块的数据：
- `module_01_data_pipeline` - 市场数据
- `module_04_market_analysis` - 市场分析结果
- `module_05_risk_management` - 风险指标
- `module_09_backtesting` - 回测结果

## 技术栈

- **图表库**: Plotly, Matplotlib
- **仪表板**: Dash, Dash Bootstrap Components
- **模板引擎**: Jinja2
- **数据处理**: Pandas, NumPy
- **导出**: ReportLab (PDF), openpyxl (Excel), PyArrow (Parquet)
- **数据库**: SQLite3

---

**FinLoom量化投资系统 - 让数据可视化更简单**