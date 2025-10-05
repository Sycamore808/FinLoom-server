# Module 09 - 回测验证模块

## 概述

回测验证模块是 FinLoom 量化交易系统的策略验证引擎,提供完整的历史回测、性能分析、风险归因和报告生成功能。本模块使用真实数据,所有结果保存到 SQLite 数据库。

## 核心功能

- **回测引擎**: 事件驱动回测,支持多策略、多标的
- **性能分析**: 完整的收益和风险指标计算
- **交易模拟**: 高保真交易执行模拟,包括滑点和佣金
- **市场冲击**: 多种市场冲击模型
- **风险归因**: 投资组合风险来源分解
- **Walk-Forward分析**: 滚动窗口优化和样本外验证
- **回测验证**: 过拟合检测和稳健性测试
- **报告生成**: HTML、PDF、Excel 格式报告
- **数据持久化**: 所有回测结果保存到 SQLite

---

## 文件结构

```
module_09_backtesting/
├── __init__.py                      # 模块导出
├── backtest_engine.py               # 回测引擎
├── performance_analyzer.py          # 性能分析
├── transaction_simulator.py         # 交易模拟
├── market_impact_model.py          # 市场冲击模型
├── risk_attribution.py             # 风险归因
├── walk_forward_analyzer.py        # Walk-forward分析
├── validation_tools.py             # 验证工具
├── report_generator.py             # 报告生成
├── database_manager.py             # 数据库管理
└── module09_README.md              # 本文档
```

---

## API 文档

### 1. 回测引擎 (BacktestEngine)

#### 初始化

```python
from module_09_backtesting import BacktestEngine, BacktestConfig
from datetime import datetime

# 创建回测配置
config = BacktestConfig(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_capital=1000000.0,
    commission_rate=0.0003,          # 佣金率 0.03%
    slippage_bps=5.0,                # 滑点 5 基点
    benchmark_symbol='000300',        # 基准(沪深300)
    save_to_db=True,                 # 保存到数据库
    strategy_name="MA Crossover"     # 策略名称
)

# 创建回测引擎
engine = BacktestEngine(config)
```

#### 加载数据

```python
from module_01_data_pipeline import AkshareDataCollector

# 获取市场数据
collector = AkshareDataCollector()
symbols = ["000001", "600036", "000858"]

market_data = {}
for symbol in symbols:
    df = collector.fetch_stock_history(symbol, "20230101", "20241231")
    market_data[symbol] = df

# 加载到回测引擎
engine.load_market_data(symbols, market_data)
```

#### 定义策略

```python
from common.data_structures import Signal

def ma_crossover_strategy(current_data, positions, capital):
    """双均线交叉策略"""
    signals = []
    
    for symbol, data in current_data.items():
        # 这里简化处理,实际需要访问历史数据计算均线
        # 示例: 如果有足够资金,就买入
        if capital > data['close'] * 100:
            signal = Signal(
                signal_id=f"sig_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                symbol=symbol,
                signal_type="BUY",
                price=data['close'],
                quantity=100,
                confidence=0.8,
                timestamp=datetime.now(),
                strategy_name="MA_CROSSOVER",
                metadata={}
            )
            signals.append(signal)
    
    return signals

# 设置策略
engine.set_strategy(ma_crossover_strategy)
```

#### 运行回测

```python
# 运行回测
result = engine.run()

# 查看结果
print(f"回测ID: {engine.backtest_id}")
print(f"初始资金: {result.initial_capital:,.2f}")
print(f"最终资金: {result.final_capital:,.2f}")
print(f"总收益率: {result.total_return:.2%}")
print(f"年化收益率: {result.annualized_return:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.3f}")
print(f"最大回撤: {result.max_drawdown:.2%}")
print(f"总交易次数: {result.total_trades}")
print(f"胜率: {result.win_rate:.2%}")
```

#### 便捷函数

```python
from module_09_backtesting import create_backtest_engine

# 使用字典创建
engine = create_backtest_engine({
    'start_date': datetime(2023, 1, 1),
    'end_date': datetime(2024, 12, 31),
    'initial_capital': 1000000.0,
    'commission_rate': 0.0003,
    'save_to_db': True
})
```

---

### 2. 性能分析 (PerformanceAnalyzer)

#### 基础分析

```python
from module_09_backtesting import PerformanceAnalyzer
import pandas as pd

# 创建分析器
analyzer = PerformanceAnalyzer()

# 从回测结果提取收益率序列
returns = result.equity_curve['equity'].pct_change().dropna()

# 执行分析
performance_report = analyzer.analyze(
    returns=returns,
    benchmark_returns=None,  # 可选: 基准收益率
    factor_returns=None      # 可选: 因子收益率
)

# 查看汇总统计
stats = performance_report.summary_stats
print(f"总收益率: {stats['total_return']:.2%}")
print(f"年化收益率: {stats['annual_return']:.2%}")
print(f"波动率: {stats['volatility']:.2%}")
print(f"夏普比率: {stats['sharpe_ratio']:.3f}")
print(f"索提诺比率: {stats['sortino_ratio']:.3f}")
print(f"卡尔玛比率: {stats['calmar_ratio']:.3f}")
print(f"最大回撤: {stats['max_drawdown']:.2%}")
print(f"偏度: {stats['skewness']:.3f}")
print(f"峰度: {stats['kurtosis']:.3f}")
print(f"VaR(95%): {stats['var_95']:.4f}")
print(f"CVaR(95%): {stats['cvar_95']:.4f}")
```

#### 与基准比较

```python
# 获取基准数据
benchmark_data = collector.fetch_stock_history("000300", "20230101", "20241231")
benchmark_returns = benchmark_data['close'].pct_change().dropna()

# 带基准的分析
performance_report = analyzer.analyze(
    returns=returns,
    benchmark_returns=benchmark_returns
)

# 查看基准比较结果
benchmark_comp = performance_report.benchmark_comparison
print(f"超额收益: {benchmark_comp['total_excess_return']:.2%}")
print(f"跟踪误差: {benchmark_comp['tracking_error']:.2%}")
print(f"信息比率: {benchmark_comp['information_ratio']:.3f}")
print(f"Beta: {benchmark_comp['beta']:.3f}")
print(f"Alpha: {benchmark_comp['alpha']:.2%}")
print(f"相关系数: {benchmark_comp['correlation']:.3f}")
print(f"上行捕获率: {benchmark_comp['up_capture_ratio']:.2%}")
print(f"下行捕获率: {benchmark_comp['down_capture_ratio']:.2%}")
```

#### 便捷函数

```python
from module_09_backtesting import analyze_performance, compare_strategies

# 快速分析
report = analyze_performance(returns, benchmark_returns)

# 多策略比较
strategies = {
    'Strategy A': returns_a,
    'Strategy B': returns_b,
    'Strategy C': returns_c
}
comparison_df = compare_strategies(strategies)
print(comparison_df)
```

---

### 3. 交易模拟 (TransactionSimulator)

#### 基础使用

```python
from module_09_backtesting import TransactionSimulator

# 创建交易模拟器
simulator = TransactionSimulator(config={
    'latency_ms': 10,                    # 延迟 10ms
    'tick_size': 0.01,                   # 最小价格变动
    'lot_size': 100,                     # 最小交易单位
    'max_participation_rate': 0.1,       # 最大市场参与率
    'permanent_impact': 0.1,             # 永久冲击系数
    'temporary_impact': 0.05             # 临时冲击系数
})

# 从历史数据初始化
simulator.initialize_from_historical_data(
    market_data=price_data,
    volume_data=volume_data
)

# 模拟订单簿
order_book = simulator.simulate_order_book(
    symbol="000001",
    timestamp=datetime.now(),
    mid_price=15.50,
    volume=1000000,
    volatility=0.02
)

# 查看订单簿
print(f"最佳买价: {order_book.best_bid}")
print(f"最佳卖价: {order_book.best_ask}")
print(f"价差: {order_book.spread}")
print(f"中间价: {order_book.mid_price}")
```

#### 模拟交易执行

```python
# 模拟市价单
result = simulator.execute_market_order(
    symbol="000001",
    side="BUY",
    quantity=1000,
    order_book=order_book
)

# 查看执行结果
print(f"请求数量: {result.requested_quantity}")
print(f"成交数量: {result.filled_quantity}")
print(f"平均成交价: {result.average_price:.2f}")
print(f"滑点: {result.slippage:.4f}")
print(f"市场冲击: {result.market_impact:.4f}")
print(f"交易成本: {result.transaction_cost:.2f}")
print(f"成交率: {result.fill_rate:.2%}")
```

---

### 4. 市场冲击模型 (MarketImpactModel)

#### 线性冲击模型

```python
from module_09_backtesting import LinearImpactModel, ImpactParameters

# 创建线性冲击模型
model = LinearImpactModel()

# 定义参数
params = ImpactParameters(
    permanent_impact=0.1,
    temporary_impact=0.05,
    decay_rate=0.5,
    volatility=0.02,
    daily_volume=5000000,
    spread=0.01
)

# 估算冲击
estimate = model.estimate_impact(
    order_size=10000,
    parameters=params
)

print(f"总冲击: {estimate.total_impact:.2f} bps")
print(f"永久冲击: {estimate.permanent_component:.2f} bps")
print(f"临时冲击: {estimate.temporary_component:.2f} bps")
print(f"执行成本: {estimate.execution_cost:.4%}")
print(f"置信区间: {estimate.confidence_interval}")
```

#### 平方根冲击模型

```python
from module_09_backtesting import SquareRootImpactModel

# 平方根模型(更适合大订单)
model = SquareRootImpactModel()
estimate = model.estimate_impact(order_size=100000, parameters=params)

print(f"总冲击: {estimate.total_impact:.2f} bps")
```

#### 模型校准

```python
# 使用历史交易数据校准模型
historical_trades = pd.DataFrame({
    'volume': [1000000, 1200000, 800000],
    'price': [15.0, 15.5, 14.8],
    'bid': [14.99, 15.48, 14.78],
    'ask': [15.01, 15.52, 14.82]
})

# 校准参数
calibrated_params = model.calibrate(historical_trades)
print(f"校准后的参数: {calibrated_params.to_dict()}")
```

---

### 5. 风险归因 (RiskAttributor)

#### 投资组合风险分解

```python
from module_09_backtesting import RiskAttributor
import numpy as np

# 创建风险归因分析器
attributor = RiskAttributor()

# 准备数据
portfolio_returns = pd.DataFrame({
    '000001': returns_1,
    '600036': returns_2,
    '000858': returns_3
})

weights = np.array([0.4, 0.3, 0.3])

# 执行风险归因
attribution_report = attributor.attribute_risk(
    portfolio_returns=portfolio_returns,
    weights=weights,
    factor_returns=None  # 可选
)

# 查看结果
print(f"总风险: {attribution_report.total_risk:.2%}")

# 风险分解
decomp = attribution_report.risk_decomposition
print(f"组合风险: {decomp['portfolio_risk']:.2%}")
print(f"加权平均风险: {decomp['weighted_average_risk']:.2%}")
print(f"分散化比率: {decomp['diversification_ratio']:.3f}")
print(f"分散化收益: {decomp['diversification_benefit']:.2%}")

# 边际贡献
marginal = attribution_report.marginal_contributions
for asset, contribution in marginal.items():
    if not asset.endswith('_pct'):
        print(f"{asset} 边际风险贡献: {contribution:.4f}")
```

---

### 6. Walk-Forward 分析 (WalkForwardAnalyzer)

#### 滚动窗口回测

```python
from module_09_backtesting import WalkForwardAnalyzer, WalkForwardConfig

# 创建配置
wf_config = WalkForwardConfig(
    total_periods=500,      # 总期数(交易日)
    train_periods=252,      # 训练期: 1年
    test_periods=63,        # 测试期: 3个月
    step_periods=21,        # 步进: 1个月
    optimization_metric='sharpe_ratio',
    anchored=False,         # 非锚定(滚动窗口)
    parallel=True,          # 并行执行
    max_workers=4
)

# 创建分析器
wf_analyzer = WalkForwardAnalyzer(wf_config)

# 定义优化函数
def optimize_strategy(train_data, param_ranges):
    """在训练数据上优化参数"""
    # 参数搜索逻辑
    best_params = {
        'ma_short': 10,
        'ma_long': 30
    }
    return best_params

# 定义回测函数
def backtest_strategy(test_data, params):
    """使用给定参数在测试数据上回测"""
    # 回测逻辑
    metrics = {
        'sharpe_ratio': 1.5,
        'total_return': 0.15,
        'max_drawdown': -0.08
    }
    return metrics

# 设置函数
wf_analyzer.set_optimization_function(optimize_strategy)
wf_analyzer.set_backtest_function(backtest_strategy)

# 运行分析
param_ranges = {
    'ma_short': range(5, 20),
    'ma_long': range(20, 60)
}

wf_result = wf_analyzer.run(
    data=combined_data,
    parameter_ranges=param_ranges
)

# 查看结果
print(f"窗口数量: {len(wf_result.windows)}")
print(f"总体指标: {wf_result.aggregate_metrics}")
print(f"参数稳定性: {wf_result.parameter_stability}")
print(f"性能衰减: {wf_result.performance_decay:.2%}")
print(f"稳健性评分: {wf_result.robustness_score:.3f}")
```

---

### 7. 回测验证 (BacktestValidator)

#### 验证回测质量

```python
from module_09_backtesting import BacktestValidator

# 创建验证器
validator = BacktestValidator()

# 执行验证
validation_report = validator.validate(
    backtest_results=result,
    market_data=market_data,
    strategy_func=ma_crossover_strategy
)

# 过拟合测试
overfitting = validation_report.overfitting_tests
print(f"样本内夏普比率: {overfitting['in_sample_sharpe']:.3f}")
print(f"样本外夏普比率: {overfitting['out_sample_sharpe']:.3f}")
print(f"夏普衰减: {overfitting['sharpe_decay']:.2%}")
print(f"过拟合评分: {overfitting['overfitting_score']:.3f}")

# 稳定性测试
stability = validation_report.stability_tests
print(f"夏普稳定性: {stability['sharpe_stability']}")

# 稳健性测试
robustness = validation_report.robustness_tests
print(f"参数敏感性: {robustness.get('parameter_sensitivity', {})}")

# 统计显著性
stats_tests = validation_report.statistical_tests
print(f"T统计量: {stats_tests.get('t_statistic', 0):.3f}")
print(f"P值: {stats_tests.get('p_value', 1):.4f}")
```

---

### 8. 报告生成 (BacktestReportGenerator)

#### 生成完整报告

```python
from module_09_backtesting import BacktestReportGenerator, ReportConfig

# 创建报告配置
report_config = ReportConfig(
    title="量化策略回测报告",
    author="FinLoom 系统",
    include_charts=True,
    include_tables=True,
    chart_theme='plotly_white',
    language='zh',
    formats=['html', 'excel'],  # 生成 HTML 和 Excel
    output_dir='reports'
)

# 创建报告生成器
report_gen = BacktestReportGenerator(report_config)

# 生成报告
report_files = report_gen.generate_report(
    backtest_result=result,
    performance_report=performance_report,
    risk_report=attribution_report,
    validation_report=validation_report
)

# 查看生成的文件
for format_type, filepath in report_files.items():
    print(f"{format_type} 报告: {filepath}")
```

---

### 9. 数据库管理 (BacktestDatabaseManager)

#### 保存和查询回测结果

```python
from module_09_backtesting import get_backtest_database_manager

# 获取数据库管理器
db = get_backtest_database_manager()

# 列出所有回测
backtests_df = db.list_backtests(limit=50)
print(backtests_df)

# 获取特定回测结果
backtest_id = "backtest_20241205_143022_a1b2c3d4"
result_dict = db.get_backtest_result(backtest_id)
print(result_dict)

# 获取交易记录
trades_df = db.get_trades(backtest_id)
print(f"交易数量: {len(trades_df)}")
print(trades_df.head())

# 获取权益曲线
equity_df = db.get_equity_curve(backtest_id)
print(equity_df.head())

# 获取性能指标
metrics = db.get_performance_metrics(backtest_id)
print(metrics)

# 数据库统计
stats = db.get_statistics()
print(f"总回测数: {stats['total_backtests']}")
print(f"总交易数: {stats['total_trades']}")
print(f"平均夏普: {stats['avg_sharpe_ratio']:.3f}")
print(f"数据库大小: {stats['database_size_mb']:.2f} MB")
```

#### 手动保存数据

```python
# 保存自定义回测结果
db.save_backtest_result(
    backtest_id="custom_backtest_001",
    result=result,
    metadata={
        'strategy_name': 'Custom Strategy',
        'note': '自定义回测'
    }
)

# 保存交易记录
trades = [
    {
        'symbol': '000001',
        'action': 'BUY',
        'quantity': 1000,
        'price': 15.50,
        'value': 15500,
        'commission': 4.65,
        'date': datetime(2024, 1, 15)
    }
]
db.save_trades("custom_backtest_001", trades)

# 保存自定义性能指标
custom_metrics = {
    'custom_metric_1': 0.85,
    'custom_metric_2': 1.23
}
db.save_performance_metrics("custom_backtest_001", custom_metrics, category='custom')
```

---

## 完整使用示例

```python
from datetime import datetime
from module_01_data_pipeline import AkshareDataCollector
from module_09_backtesting import (
    BacktestEngine,
    BacktestConfig,
    PerformanceAnalyzer,
    BacktestReportGenerator,
    ReportConfig,
    get_backtest_database_manager
)
from common.data_structures import Signal

# ===== 1. 准备数据 (Module 01) =====
collector = AkshareDataCollector()
symbols = ["000001", "600036", "000858"]

market_data = {}
for symbol in symbols:
    df = collector.fetch_stock_history(symbol, "20230101", "20241231")
    market_data[symbol] = df
    print(f"加载 {symbol}: {len(df)} 条记录")

# ===== 2. 配置回测 =====
config = BacktestConfig(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_capital=1000000.0,
    commission_rate=0.0003,
    slippage_bps=5.0,
    save_to_db=True,
    strategy_name="双均线交叉策略"
)

# ===== 3. 创建回测引擎 =====
engine = BacktestEngine(config)
engine.load_market_data(symbols, market_data)

# ===== 4. 定义策略 =====
def simple_strategy(current_data, positions, capital):
    """简单策略示例"""
    signals = []
    # 策略逻辑...
    return signals

engine.set_strategy(simple_strategy)

# ===== 5. 运行回测 =====
print("\n开始回测...")
result = engine.run()

print(f"\n回测完成!")
print(f"回测ID: {engine.backtest_id}")
print(f"总收益率: {result.total_return:.2%}")
print(f"年化收益率: {result.annualized_return:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.3f}")
print(f"最大回撤: {result.max_drawdown:.2%}")
print(f"总交易: {result.total_trades} 笔")

# ===== 6. 性能分析 =====
analyzer = PerformanceAnalyzer()
returns = result.equity_curve['equity'].pct_change().dropna()
performance_report = analyzer.analyze(returns)

print(f"\n详细性能指标:")
stats = performance_report.summary_stats
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")

# ===== 7. 生成报告 =====
report_config = ReportConfig(
    title=f"{config.strategy_name}回测报告",
    formats=['html', 'excel'],
    output_dir='reports'
)

report_gen = BacktestReportGenerator(report_config)
report_files = report_gen.generate_report(
    backtest_result=result,
    performance_report=performance_report
)

print(f"\n报告已生成:")
for fmt, path in report_files.items():
    print(f"  {fmt}: {path}")

# ===== 8. 从数据库查询 =====
db = get_backtest_database_manager()

# 列出所有回测
print(f"\n历史回测记录:")
all_backtests = db.list_backtests(limit=10)
print(all_backtests[['backtest_id', 'strategy_name', 'total_return', 'sharpe_ratio']])

# 获取统计信息
stats = db.get_statistics()
print(f"\n数据库统计:")
print(f"  总回测数: {stats['total_backtests']}")
print(f"  总交易数: {stats['total_trades']}")
print(f"  平均夏普: {stats['avg_sharpe_ratio']:.3f}")
```

---

## 与其他模块集成

### 与 Module 01 (数据管道) 集成

```python
from module_01_data_pipeline import AkshareDataCollector, get_database_manager

# 从 Module 01 获取实时数据
collector = AkshareDataCollector()
data = collector.fetch_stock_history("000001", "20230101", "20241231")

# 或从 Module 01 数据库获取
db_manager = get_database_manager()
data = db_manager.get_stock_prices("000001", "2023-01-01", "2024-12-31")

# 用于回测
engine.load_market_data(["000001"], {"000001": data})
```

### 与 Module 02 (特征工程) 集成

```python
from module_02_feature_engineering import TechnicalIndicatorCalculator

# 计算技术指标
calculator = TechnicalIndicatorCalculator()
features = calculator.calculate_all_indicators(data)

# 在策略中使用特征
def feature_based_strategy(current_data, positions, capital):
    signals = []
    for symbol, data in current_data.items():
        # 使用技术指标特征
        if 'rsi' in data and data['rsi'] < 30:
            # 超卖,买入信号
            signals.append(create_buy_signal(symbol, data))
    return signals
```

### 与 Module 03 (AI模型) 集成

```python
from module_03_ai_models import LSTMModel

# 加载训练好的模型
model = LSTMModel.load_model('path/to/model.pth')

# 在策略中使用AI预测
def ai_strategy(current_data, positions, capital):
    signals = []
    for symbol, data in current_data.items():
        # 使用AI模型预测
        prediction = model.predict(prepare_features(data))
        if prediction > 0.7:  # 高置信度
            signals.append(create_buy_signal(symbol, data))
    return signals
```

### 与 Module 05 (风险管理) 集成

```python
from module_05_risk_management import PortfolioRiskAnalyzer

# 创建风险分析器
risk_analyzer = PortfolioRiskAnalyzer()

# 在策略中加入风险控制
def risk_aware_strategy(current_data, positions, capital):
    # 计算当前风险
    portfolio_risk = risk_analyzer.calculate_portfolio_risk(positions)
    
    # 如果风险过高,停止交易
    if portfolio_risk['var_95'] > 0.05:
        return []
    
    # 否则生成信号
    signals = generate_signals(current_data)
    return signals
```

---

## 测试

### 运行测试

```bash
# 切换到项目目录
cd /Users/victor/Desktop/25fininnov/FinLoom-server

# 激活conda环境
conda activate study

# 运行测试
python tests/module09_backtesting_test.py
```

### 测试说明

- 所有测试使用 Module 01 获取的真实市场数据
- 不使用任何模拟数据
- 测试涵盖回测引擎、性能分析、数据库存储等完整流程
- 测试结果保存到 SQLite 数据库

---

## 配置说明

### 回测配置

```python
BacktestConfig(
    start_date=datetime(2023, 1, 1),     # 回测开始日期
    end_date=datetime(2024, 12, 31),     # 回测结束日期
    initial_capital=1000000.0,            # 初始资金
    commission_rate=0.0003,               # 佣金率(双边)
    slippage_bps=5.0,                     # 滑点(基点)
    benchmark_symbol='000300',            # 基准指数
    rebalance_frequency='daily',          # 再平衡频率
    save_to_db=True,                      # 保存到数据库
    strategy_name="策略名称"               # 策略名称
)
```

### 数据库配置

默认数据库路径: `data/module09_backtest.db`

自定义数据库路径:
```python
from module_09_backtesting import BacktestDatabaseManager

db = BacktestDatabaseManager(db_path="custom/path/backtest.db")
```

---

## 注意事项

1. **数据来源**: 使用 Module 01 获取真实市场数据,不使用模拟数据
2. **数据持久化**: 所有回测结果自动保存到 SQLite 数据库
3. **策略函数**: 策略函数需要返回 Signal 对象列表
4. **时间序列**: 确保市场数据按时间排序
5. **内存管理**: 大量回测时注意内存使用
6. **并行执行**: Walk-forward 分析支持并行,注意线程安全
7. **测试环境**: 测试需在 conda 的 `study` 环境中运行

---

## 常见问题

**Q: 如何查看历史回测记录?**
```python
db = get_backtest_database_manager()
backtests = db.list_backtests(limit=100)
```

**Q: 如何删除旧的回测记录?**
```python
db.delete_backtest(backtest_id)
```

**Q: 支持哪些性能指标?**

A: 包括总收益率、年化收益率、波动率、夏普比率、索提诺比率、卡尔玛比率、最大回撤、VaR、CVaR、偏度、峰度、信息比率等30+指标。

**Q: 如何导出回测结果?**

A: 使用 `BacktestReportGenerator` 生成 HTML、PDF 或 Excel 格式报告。

**Q: 数据库文件在哪里?**

A: 默认位置是 `data/module09_backtest.db`。

**Q: 如何集成自定义策略?**

A: 定义一个返回 Signal 列表的函数,然后使用 `engine.set_strategy(your_strategy)` 设置即可。