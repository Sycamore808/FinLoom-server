# Module 09 - 回测模块

## 概述

回测模块是 FinLoom 量化交易系统的策略验证引擎，提供完整的历史回测、性能分析、风险归因和策略优化功能。该模块支持事件驱动回测、滚动窗口分析，并生成详细的绩效报告。

## 主要功能

### 1. 回测引擎 (Backtest Engine)
- **BacktestEngine**: 事件驱动回测引擎
- **VectorizedBacktest**: 向量化快速回测
- **WalkForwardAnalyzer**: 滚动窗口分析
- **MonteCarloBacktest**: 蒙特卡洛模拟

### 2. 性能分析 (Performance Analysis)
- **PerformanceAnalyzer**: 综合绩效分析
- **RiskMetrics**: 风险指标计算
- **BenchmarkComparison**: 基准比较
- **AttributionAnalysis**: 归因分析

### 3. 交易模拟 (Transaction Simulation)
- **TransactionSimulator**: 交易成本模拟
- **MarketImpactModel**: 市场冲击模型
- **SlippageModel**: 滑点模型
- **CommissionModel**: 佣金计算模型

### 4. 报告生成 (Report Generation)
- **ReportGenerator**: 回测报告生成器
- **VisualReportBuilder**: 可视化报告
- **PerformanceDashboard**: 绩效仪表板

## 快速开始

### 基础使用示例

```python
from module_09_backtesting import BacktestEngine, BacktestConfig
from module_01_data_pipeline import get_database_manager
from datetime import datetime

# 1. 配置回测参数
backtest_config = BacktestConfig(
    start_date='2023-01-01',
    end_date='2024-12-01',
    initial_capital=1000000,
    commission_rate=0.0003,
    slippage_model='percentage',
    slippage_value=0.001,
    position_sizing='equal_weight'
)

# 2. 创建回测引擎
backtest_engine = BacktestEngine(backtest_config)

# 3. 定义策略
from module_03_ai_models import LSTMModel

def momentum_strategy(data, context):
    """动量策略示例"""
    # 获取预测
    lstm_model = context.get('model')
    predictions = lstm_model.predict(data)
    
    # 生成信号
    signals = {}
    for symbol, pred in predictions.items():
        if pred > 0.6:
            signals[symbol] = 'BUY'
        elif pred < -0.6:
            signals[symbol] = 'SELL'
        else:
            signals[symbol] = 'HOLD'
    
    return signals

# 4. 运行回测
results = backtest_engine.run_backtest(
    strategy=momentum_strategy,
    symbols=['000001', '600036', '000858'],
    context={'model': LSTMModel.load_model('momentum_predictor')}
)

# 5. 性能分析
from module_09_backtesting import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
performance = analyzer.analyze(results)

print(f"回测结果:")
print(f"  总收益率: {performance['total_return']:.2%}")
print(f"  年化收益率: {performance['annual_return']:.2%}")
print(f"  最大回撤: {performance['max_drawdown']:.2%}")
print(f"  夏普比率: {performance['sharpe_ratio']:.3f}")
print(f"  索提诺比率: {performance['sortino_ratio']:.3f}")
print(f"  卡尔玛比率: {performance['calmar_ratio']:.3f}")
print(f"  胜率: {performance['win_rate']:.2%}")
print(f"  盈亏比: {performance['profit_loss_ratio']:.2f}")

# 6. 生成报告
from module_09_backtesting import ReportGenerator

report_gen = ReportGenerator()
report = report_gen.generate_report(
    results=results,
    performance=performance,
    output_format='html'
)

print(f"\n回测报告已生成: {report['filepath']}")
```

## API 参考

### BacktestEngine

#### 构造函数
```python
BacktestEngine(config: BacktestConfig)
```

#### 主要方法

**run_backtest(strategy: Callable, symbols: List[str], context: Dict = None) -> BacktestResult**
- 运行回测
- 返回BacktestResult对象

**run_walk_forward(strategy: Callable, symbols: List[str], train_period: int = 252, test_period: int = 63) -> List[BacktestResult]**
- 滚动窗口回测
- 避免前视偏差

**run_monte_carlo(strategy: Callable, symbols: List[str], n_simulations: int = 1000) -> MonteCarloResult**
- 蒙特卡洛模拟
- 评估策略稳健性

### PerformanceAnalyzer

#### 主要方法

**analyze(results: BacktestResult) -> Dict[str, float]**
- 综合性能分析
- 返回各项指标

**calculate_returns(equity_curve: pd.Series) -> Dict[str, float]**
- 计算收益指标

**calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]**
- 计算风险指标

**compare_to_benchmark(results: BacktestResult, benchmark: pd.Series) -> Dict[str, Any]**
- 与基准比较

## 与其他模块集成

### 与 Module 01-07 集成
```python
# 完整的回测流程
from module_01_data_pipeline import AkshareDataCollector
from module_02_feature_engineering import TechnicalIndicators
from module_03_ai_models import LSTMModel
from module_05_risk_management import PortfolioRiskAnalyzer
from module_07_optimization import StrategyOptimizer
from module_09_backtesting import BacktestEngine

# 1. 数据准备
collector = AkshareDataCollector()
calculator = TechnicalIndicators()

data = {}
for symbol in symbols:
    prices = collector.fetch_stock_history(symbol, start_date, end_date)
    features = calculator.calculate_all_indicators(prices)
    data[symbol] = features

# 2. 策略定义
lstm_model = LSTMModel.load_model('strategy_model')

def ai_strategy(data, context):
    predictions = lstm_model.predict(data)
    
    # 风险检查
    risk_analyzer = context['risk_analyzer']
    risk = risk_analyzer.analyze_portfolio_risk(portfolio, returns)
    
    if risk['var_95'] > 0.05:
        return {}  # 风险过高，不交易
    
    return generate_signals(predictions)

# 3. 参数优化
strategy_optimizer = StrategyOptimizer(config)
best_params = strategy_optimizer.optimize_strategy(
    strategy_function=ai_strategy,
    data=data,
    param_space=param_space
)

# 4. 回测验证
backtest = BacktestEngine(config)
results = backtest.run_backtest(ai_strategy, symbols, context)
```

## 测试和示例

### 运行测试
```bash
cd /Users/victor/Desktop/25fininnov/FinLoom-server
python tests/module09_backtesting_test.py
```

## 配置说明

### 环境变量
- `MODULE09_DB_PATH`: 回测数据库路径
- `MODULE09_REPORT_DIR`: 报告输出目录

### 配置文件
```yaml
# config/backtest_config.yaml
backtest:
  initial_capital: 1000000
  commission_rate: 0.0003
  slippage_model: 'percentage'
  slippage_value: 0.001
  
performance:
  risk_free_rate: 0.03
  benchmark: '000300'  # 沪深300
  
reporting:
  output_format: 'html'
  include_charts: true
```

## 性能基准

| 操作 | 数据规模 | 处理时间 |
|------|----------|----------|
| 向量化回测 | 3股票×2年 | ~2s |
| 事件驱动回测 | 3股票×2年 | ~10s |
| 滚动窗口回测 | 3股票×2年 | ~30s |
| 蒙特卡洛模拟 | 1000次 | ~60s |

## 总结

Module 09 提供了专业级的回测验证能力：

### 功能完整性 ✅
- ✓ 事件驱动和向量化回测
- ✓ 滚动窗口分析
- ✓ 完整的性能指标
- ✓ 详细的回测报告

### 集成能力 ✅
- ✓ 与所有模块无缝集成
- ✓ 支持AI模型策略回测
- ✓ 集成风险管理
- ✓ 支持策略优化

**结论**: Module 09 是策略开发和验证的重要工具，确保策略在实盘前得到充分验证。

