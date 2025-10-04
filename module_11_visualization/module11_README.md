# Module 11 - 可视化模块

## 概述

可视化模块是 FinLoom 量化交易系统的数据展示引擎，提供丰富的图表生成、交互式仪表板、报告构建和数据导出功能。该模块从所有其他模块获取数据，通过专业的金融图表展示投资分析结果。

## 主要功能

### 1. 图表生成 (Chart Generation)
- **ChartGenerator**: 图表生成器
- **CandlestickChart**: K线图
- **TimeSeriesChart**: 时序图
- **HeatmapChart**: 热力图
- **CorrelationMatrix**: 相关性矩阵
- **PerformanceChart**: 绩效曲线

### 2. 仪表板 (Dashboard)
- **DashboardManager**: 仪表板管理器
- **RealTimeDashboard**: 实时监控仪表板
- **PortfolioDashboard**: 投资组合仪表板
- **StrategyDashboard**: 策略分析仪表板
- **RiskDashboard**: 风险监控仪表板

### 3. 报告生成 (Report Builder)
- **ReportBuilder**: 报告构建器
- **DailyReportGenerator**: 日报生成
- **BacktestReportGenerator**: 回测报告
- **PerformanceReportGenerator**: 绩效报告
- **RiskReportGenerator**: 风险报告

### 4. 交互式可视化 (Interactive Visualization)
- **InteractiveVisualizer**: 交互式可视化
- **ZoomableChart**: 可缩放图表
- **DynamicFilter**: 动态过滤器
- **DrillDownAnalysis**: 下钻分析

### 5. 导出管理 (Export Manager)
- **ExportManager**: 导出管理器
- **PDFExporter**: PDF导出
- **ExcelExporter**: Excel导出
- **ImageExporter**: 图片导出

## 快速开始

### 基础使用示例

```python
from module_11_visualization import (
    ChartGenerator,
    DashboardManager,
    ReportBuilder,
    InteractiveVisualizer,
    ExportManager
)

# 1. 创建图表生成器
chart_gen = ChartGenerator()

# 2. K线图
from module_01_data_pipeline import get_database_manager

db_manager = get_database_manager()
stock_data = db_manager.get_stock_prices('000001', '2024-01-01', '2024-12-01')

candlestick = chart_gen.create_candlestick(
    data=stock_data,
    symbol='000001',
    title='平安银行K线图',
    add_volume=True,
    add_ma=[5, 10, 20, 60]
)

# 3. 绩效曲线
from module_09_backtesting import BacktestEngine

backtest_result = backtest.run_backtest(strategy, symbols)

performance_chart = chart_gen.create_performance_chart(
    equity_curve=backtest_result.equity_curve,
    benchmark=benchmark_data,
    title='策略绩效对比',
    show_drawdown=True
)

# 4. 相关性热力图
from module_04_market_analysis import CorrelationAnalyzer

correlation_analyzer = CorrelationAnalyzer()
correlation_matrix = correlation_analyzer.calculate_correlation(returns_df)

heatmap = chart_gen.create_heatmap(
    data=correlation_matrix,
    title='股票相关性矩阵',
    cmap='RdYlGn',
    annot=True
)

# 5. 创建仪表板
dashboard_mgr = DashboardManager()

dashboard = dashboard_mgr.create_dashboard(
    dashboard_type='portfolio',
    title='投资组合监控仪表板'
)

# 添加图表到仪表板
dashboard.add_chart(candlestick, position=(0, 0), size=(2, 2))
dashboard.add_chart(performance_chart, position=(0, 2), size=(2, 2))
dashboard.add_chart(heatmap, position=(2, 0), size=(2, 2))

# 添加指标卡片
dashboard.add_metric_card(
    title='总收益率',
    value=f"{backtest_result.total_return:.2%}",
    change='+5.2%',
    position=(2, 2)
)

dashboard.add_metric_card(
    title='夏普比率',
    value=f"{backtest_result.sharpe_ratio:.2f}",
    position=(2, 3)
)

# 6. 生成报告
report_builder = ReportBuilder()

report = report_builder.create_report(
    report_type='backtest',
    title='策略回测报告',
    data={
        'strategy': strategy,
        'backtest_result': backtest_result,
        'charts': [candlestick, performance_chart, heatmap]
    }
)

# 7. 导出报告
export_mgr = ExportManager()

# 导出为PDF
pdf_path = export_mgr.export_to_pdf(
    report=report,
    filename='backtest_report_20241201.pdf'
)

# 导出为HTML
html_path = export_mgr.export_to_html(
    report=report,
    filename='backtest_report_20241201.html',
    interactive=True
)

print(f"✅ 报告已生成:")
print(f"  PDF: {pdf_path}")
print(f"  HTML: {html_path}")

# 8. 实时仪表板
realtime_dashboard = dashboard_mgr.create_realtime_dashboard()

# 启动实时更新
await realtime_dashboard.start_streaming(
    symbols=['000001', '600036', '000858'],
    update_interval=5  # 5秒更新一次
)

print("\n✅ 可视化系统运行中！")
```

## API 参考

### ChartGenerator

图表生成器。

#### 主要方法

**create_candlestick(data: pd.DataFrame, symbol: str, title: str = None, **kwargs) -> Chart**
- 创建K线图
- 支持成交量、均线等叠加

**create_performance_chart(equity_curve: pd.Series, benchmark: pd.Series = None, **kwargs) -> Chart**
- 创建绩效曲线图
- 支持基准对比、回撤显示

**create_heatmap(data: pd.DataFrame, title: str = None, **kwargs) -> Chart**
- 创建热力图
- 用于相关性、因子暴露等

**create_bar_chart(data: pd.Series, title: str = None, **kwargs) -> Chart**
- 创建柱状图
- 用于收益分布、持仓占比等

**create_pie_chart(data: pd.Series, title: str = None, **kwargs) -> Chart**
- 创建饼图
- 用于资产配置、行业分布等

**create_scatter_plot(x: pd.Series, y: pd.Series, title: str = None, **kwargs) -> Chart**
- 创建散点图
- 用于因子分析、风险收益分布等

#### 使用示例
```python
chart_gen = ChartGenerator()

# K线图
candlestick = chart_gen.create_candlestick(
    data=stock_data,
    symbol='000001',
    add_volume=True,
    add_ma=[5, 20, 60]
)

# 绩效图
performance = chart_gen.create_performance_chart(
    equity_curve=portfolio_value,
    benchmark=hs300_index,
    show_drawdown=True
)
```

### DashboardManager

仪表板管理器。

#### 主要方法

**create_dashboard(dashboard_type: str, title: str = None, **kwargs) -> Dashboard**
- 创建仪表板
- dashboard_type: 'portfolio', 'strategy', 'risk', 'realtime'

**create_realtime_dashboard(**kwargs) -> RealtimeDashboard**
- 创建实时监控仪表板
- 支持WebSocket实时更新

**save_dashboard(dashboard: Dashboard, filepath: str) -> bool**
- 保存仪表板配置

**load_dashboard(filepath: str) -> Dashboard**
- 加载仪表板配置

#### Dashboard对象方法

**add_chart(chart: Chart, position: Tuple[int, int], size: Tuple[int, int] = None) -> None**
- 添加图表到仪表板

**add_metric_card(title: str, value: Any, position: Tuple[int, int], **kwargs) -> None**
- 添加指标卡片

**add_table(data: pd.DataFrame, position: Tuple[int, int], **kwargs) -> None**
- 添加数据表格

**render() -> str**
- 渲染仪表板为HTML

#### 使用示例
```python
dashboard_mgr = DashboardManager()

# 创建投资组合仪表板
dashboard = dashboard_mgr.create_dashboard('portfolio', '我的投资组合')

# 添加组件
dashboard.add_chart(performance_chart, position=(0, 0), size=(2, 2))
dashboard.add_metric_card('总资产', '1,234,567.89元', position=(0, 2))
dashboard.add_table(holdings_df, position=(2, 0))

# 渲染
html = dashboard.render()
```

### ReportBuilder

报告构建器。

#### 主要方法

**create_report(report_type: str, title: str, data: Dict, template: str = None) -> Report**
- 创建报告
- report_type: 'daily', 'weekly', 'monthly', 'backtest', 'performance', 'risk'

**add_section(report: Report, section_type: str, content: Any) -> None**
- 添加报告章节
- section_type: 'summary', 'charts', 'tables', 'analysis', 'recommendations'

**generate_executive_summary(data: Dict) -> str**
- 生成执行摘要

**generate_detailed_analysis(data: Dict) -> str**
- 生成详细分析

#### 使用示例
```python
report_builder = ReportBuilder()

# 创建回测报告
report = report_builder.create_report(
    report_type='backtest',
    title='动量策略回测报告',
    data={
        'strategy_name': 'momentum_strategy',
        'backtest_result': result,
        'risk_metrics': risk_metrics
    }
)

# 添加章节
report_builder.add_section(report, 'summary', executive_summary)
report_builder.add_section(report, 'charts', [perf_chart, drawdown_chart])
report_builder.add_section(report, 'analysis', detailed_analysis)
```

### InteractiveVisualizer

交互式可视化。

#### 主要方法

**create_interactive_chart(data: pd.DataFrame, chart_type: str, **kwargs) -> InteractiveChart**
- 创建交互式图表
- 支持缩放、悬停、下钻等交互

**add_crosshair(chart: InteractiveChart) -> None**
- 添加十字光标

**add_range_selector(chart: InteractiveChart, ranges: List[str]) -> None**
- 添加时间范围选择器
- ranges: ['1d', '5d', '1m', '3m', '6m', '1y', 'ytd', 'all']

**add_annotation(chart: InteractiveChart, x: Any, y: Any, text: str) -> None**
- 添加标注

#### 使用示例
```python
interactive_viz = InteractiveVisualizer()

# 创建交互式K线图
chart = interactive_viz.create_interactive_chart(
    data=stock_data,
    chart_type='candlestick'
)

# 添加交互功能
interactive_viz.add_crosshair(chart)
interactive_viz.add_range_selector(chart, ranges=['1m', '3m', '6m', '1y'])

# 添加买卖标注
interactive_viz.add_annotation(chart, x='2024-06-15', y=15.5, text='买入')
```

### ExportManager

导出管理器。

#### 主要方法

**export_to_pdf(report: Report, filename: str, **kwargs) -> str**
- 导出为PDF
- 返回文件路径

**export_to_html(report: Report, filename: str, interactive: bool = True) -> str**
- 导出为HTML
- interactive=True时包含交互功能

**export_to_excel(data: Dict[str, pd.DataFrame], filename: str) -> str**
- 导出为Excel
- 支持多工作表

**export_chart_to_image(chart: Chart, filename: str, format: str = 'png', **kwargs) -> str**
- 导出图表为图片
- format: 'png', 'jpg', 'svg'

#### 使用示例
```python
export_mgr = ExportManager()

# 导出PDF
pdf_path = export_mgr.export_to_pdf(
    report=report,
    filename='report.pdf',
    page_size='A4',
    orientation='portrait'
)

# 导出Excel
excel_path = export_mgr.export_to_excel(
    data={
        '持仓明细': holdings_df,
        '交易记录': trades_df,
        '绩效指标': metrics_df
    },
    filename='portfolio_data.xlsx'
)
```

## 图表类型详解

### 金融图表
- **K线图**: 显示开高低收和成交量
- **分时图**: 日内分钟级价格走势
- **成交量图**: 成交量柱状图
- **技术指标图**: MACD、RSI、布林带等

### 绩效图表
- **收益曲线**: 累计收益率曲线
- **回撤曲线**: 最大回撤走势
- **月度收益热力图**: 月度收益分布
- **收益分布直方图**: 收益率分布

### 风险图表
- **VaR分布图**: 风险价值分布
- **风险贡献图**: 各资产风险贡献
- **相关性热力图**: 资产相关性矩阵
- **波动率锥**: 历史波动率分布

### 组合图表
- **持仓饼图**: 资产配置占比
- **行业分布图**: 行业暴露分布
- **因子暴露图**: 风格因子暴露
- **归因分析图**: 收益归因分解

### 交易图表
- **订单执行图**: 订单价格时间分布
- **滑点分析图**: 滑点统计分布
- **成交明细图**: 成交价量时间序列

## 仪表板模板

### 投资组合仪表板
```python
portfolio_dashboard = dashboard_mgr.create_dashboard('portfolio')

# 布局
# Row 1: 关键指标卡片
# Row 2: 收益曲线 + 资产配置饼图
# Row 3: 持仓明细表 + 最近交易
# Row 4: 风险指标 + 绩效分析
```

### 策略仪表板
```python
strategy_dashboard = dashboard_mgr.create_dashboard('strategy')

# 布局
# Row 1: 策略信息 + 回测指标
# Row 2: 收益对比图 + 回撤图
# Row 3: 信号统计 + 胜率分析
# Row 4: 参数敏感性 + 稳健性测试
```

### 风险仪表板
```python
risk_dashboard = dashboard_mgr.create_dashboard('risk')

# 布局
# Row 1: 风险预警 + VaR/CVaR
# Row 2: 相关性矩阵 + 风险贡献
# Row 3: 压力测试结果 + 情景分析
# Row 4: 风险敞口 + 限额使用率
```

## REST API 端点

```
# 图表生成
POST /api/v1/visualization/chart/create
GET /api/v1/visualization/chart/{chart_id}

# 仪表板
POST /api/v1/visualization/dashboard/create
GET /api/v1/visualization/dashboard/{dashboard_id}
PUT /api/v1/visualization/dashboard/{dashboard_id}

# 报告
POST /api/v1/visualization/report/generate
GET /api/v1/visualization/report/{report_id}
GET /api/v1/visualization/report/{report_id}/download

# 导出
POST /api/v1/visualization/export/pdf
POST /api/v1/visualization/export/excel
POST /api/v1/visualization/export/image
```

## WebSocket实时推送

```javascript
// 实时仪表板更新
const ws = new WebSocket('ws://localhost:8000/ws/visualization/realtime');

ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    
    // 更新图表
    if (update.type === 'chart_update') {
        updateChart(update.chart_id, update.data);
    }
    
    // 更新指标
    if (update.type === 'metric_update') {
        updateMetric(update.metric_id, update.value);
    }
};
```

## 与其他模块集成

### 完整的可视化流程
```python
from module_11_visualization import ChartGenerator, DashboardManager, ReportBuilder

# 1. 从Module 01获取数据
from module_01_data_pipeline import get_database_manager
data = get_database_manager().get_stock_prices('000001', start, end)

# 2. 从Module 02获取技术指标
from module_02_feature_engineering import TechnicalIndicators
indicators = TechnicalIndicators().calculate_all_indicators(data)

# 3. 从Module 03获取AI预测
from module_03_ai_models import LSTMModel
predictions = LSTMModel.load_model('predictor').predict(features)

# 4. 从Module 04获取市场分析
from module_04_market_analysis import get_sentiment_analyzer
sentiment = await get_sentiment_analyzer().analyze_stock_sentiment(['000001'])

# 5. 从Module 05获取风险指标
from module_05_risk_management import PortfolioRiskAnalyzer
risk = PortfolioRiskAnalyzer(config).analyze_portfolio_risk(portfolio, returns)

# 6. 从Module 09获取回测结果
from module_09_backtesting import BacktestEngine
backtest = BacktestEngine(config).run_backtest(strategy, symbols)

# 7. 生成综合可视化
chart_gen = ChartGenerator()
dashboard_mgr = DashboardManager()

# 创建图表
price_chart = chart_gen.create_candlestick(data, '000001')
indicator_chart = chart_gen.create_line_chart(indicators[['rsi', 'macd']])
prediction_chart = chart_gen.create_prediction_chart(predictions)
sentiment_chart = chart_gen.create_sentiment_gauge(sentiment)
risk_chart = chart_gen.create_risk_dashboard(risk)
backtest_chart = chart_gen.create_performance_chart(backtest.equity_curve)

# 创建综合仪表板
dashboard = dashboard_mgr.create_dashboard('comprehensive')
dashboard.add_chart(price_chart, position=(0, 0))
dashboard.add_chart(indicator_chart, position=(0, 1))
dashboard.add_chart(prediction_chart, position=(1, 0))
dashboard.add_chart(sentiment_chart, position=(1, 1))
dashboard.add_chart(risk_chart, position=(2, 0))
dashboard.add_chart(backtest_chart, position=(2, 1))

# 渲染和导出
html = dashboard.render()
export_mgr.export_to_pdf(dashboard, 'comprehensive_report.pdf')
```

## 测试和示例

### 运行测试
```bash
cd /Users/victor/Desktop/25fininnov/FinLoom-server
python tests/module11_visualization_test.py
```

## 配置说明

### 环境变量
- `MODULE11_DB_PATH`: 可视化数据库路径
- `MODULE11_OUTPUT_DIR`: 输出目录
- `MODULE11_TEMPLATE_DIR`: 模板目录

### 配置文件
```yaml
# config/visualization_config.yaml
charts:
  default_theme: 'professional'
  color_scheme: 'finloom'
  font_family: 'Arial, sans-serif'
  figure_size: [12, 8]

dashboard:
  refresh_interval: 5
  max_charts_per_dashboard: 20
  enable_realtime: true

export:
  pdf_dpi: 300
  image_format: 'png'
  excel_engine: 'openpyxl'

templates:
  report_template: 'templates/report_template.html'
  dashboard_template: 'templates/dashboard_template.html'
```

## 性能基准

| 操作 | 处理时间 | 输出大小 |
|------|----------|----------|
| 生成K线图 | ~100ms | ~100KB |
| 生成仪表板 | ~500ms | ~500KB |
| 生成PDF报告 | ~3s | ~2MB |
| 导出Excel | ~1s | ~500KB |
| 实时更新 | ~50ms | - |

## 总结

Module 11 提供了专业的金融数据可视化能力：

### 功能完整性 ✅
- ✓ 丰富的图表类型（K线、绩效、风险等）
- ✓ 交互式仪表板
- ✓ 自动报告生成
- ✓ 多格式导出（PDF、Excel、HTML）

### 集成能力 ✅
- ✓ 从所有模块获取数据
- ✓ REST API和WebSocket支持
- ✓ 实时数据更新
- ✓ 模板化报告系统

### 实用性 ✅
- ✓ 专业的金融图表
- ✓ 灵活的布局系统
- ✓ 美观的视觉设计
- ✓ 便捷的导出功能

**结论**: Module 11 将复杂的数据转化为直观的可视化展示，是用户理解和决策的重要工具。

---

**FinLoom量化投资系统 - 模块文档编写完成！**

现在所有11个核心模块（Module 00-11）的功能定义和API文档已经全部完成，形成了一个完整、专业、符合量化投资系统需求的架构体系。

