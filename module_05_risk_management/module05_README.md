# Module 05 - 风险管理模块 API 文档

## 📋 模块概述

风险管理模块提供全面的风险评估、仓位管理、止损策略和投资组合优化功能。

**核心功能：**
- 风险分析：投资组合风险评估、VaR/CVaR计算
- 仓位管理：凯利准则、风险平价、动态仓位调整  
- 投资组合优化：均值方差优化、风险预算、有效前沿
- 止损策略：自适应止损、追踪止损
- 压力测试：蒙特卡洛模拟、场景分析
- 数据持久化：风险数据库管理

---

## 🚀 快速开始

### 导入模块

```python
from module_05_risk_management import (
    # 风险分析
    PortfolioRiskAnalyzer, RiskConfig,
    VaRCalculator, VaRConfig,
    RiskExposureAnalyzer, ExposureConfig,
    
    # 仓位管理
    KellyCriterion, KellyResult,
    DynamicPositionSizer, PositionSizingConfig, PositionSizingMethod,
    RiskParity, RiskParityConfig,
    
    # 投资组合优化
    PortfolioWeightOptimizer, OptimizationConfig,
    OptimizationMethod, OptimizationObjective,
    MeanVarianceOptimizer, MVOConfig,
    
    # 止损策略
    StopLossManager, StopLossConfig,
    AdaptiveStopLoss, TrailingStop,
    
    # 压力测试
    MonteCarloSimulator, SimulationConfig,
    ScenarioGenerator, ScenarioConfig,
    
    # 数据库
    get_risk_database_manager,
)
```

---

## 📊 风险分析 API

### PortfolioRiskAnalyzer

投资组合综合风险评估。

#### 初始化

```python
from module_05_risk_management import PortfolioRiskAnalyzer, RiskConfig

config = RiskConfig(
    confidence_level=0.95,          # 置信水平
    time_horizon=1,                 # 持有期（天）
    calculation_method='historical', # 'historical', 'parametric', 'monte_carlo'
    rolling_window=252,             # 滚动窗口
    annualization_factor=252        # 年化因子
)

analyzer = PortfolioRiskAnalyzer(config)
```

#### 主要方法

**analyze_portfolio_risk(portfolio, returns)**
```python
# 准备数据
portfolio = {
    '000001': {'weight': 0.3, 'shares': 1000, 'cost': 15.5},
    '600036': {'weight': 0.4, 'shares': 800, 'cost': 45.2},
}

# returns 是 pandas DataFrame，列为股票代码，值为收益率
risk_metrics = analyzer.analyze_portfolio_risk(
    portfolio=portfolio,
    returns=returns_df
)

# 返回结果
print(f"VaR (95%): {risk_metrics['var_95']}")
print(f"CVaR (95%): {risk_metrics['cvar_95']}")
print(f"最大回撤: {risk_metrics['max_drawdown']}")
print(f"夏普比率: {risk_metrics['sharpe_ratio']}")
print(f"波动率: {risk_metrics['volatility']}")
```

**calculate_var(returns, confidence_level)**
```python
# 计算单只股票或投资组合的VaR
var_95 = analyzer.calculate_var(
    returns=portfolio_returns,
    confidence_level=0.95
)
```

**calculate_cvar(returns, confidence_level)**
```python
# 计算条件VaR（尾部风险）
cvar_95 = analyzer.calculate_cvar(
    returns=portfolio_returns,
    confidence_level=0.95
)
```

---

### VaRCalculator

专业的风险价值计算工具。

#### 初始化

```python
from module_05_risk_management import VaRCalculator, VaRConfig

config = VaRConfig(
    confidence_level=0.95,
    time_horizon=1,
    method='historical'  # 'historical', 'parametric', 'monte_carlo'
)

var_calc = VaRCalculator(config)
```

#### 主要方法

**historical_var(returns)**
```python
# 历史模拟法
var = var_calc.historical_var(returns)
```

**parametric_var(returns)**
```python
# 参数法（假设正态分布）
var = var_calc.parametric_var(returns)
```

**monte_carlo_var(returns, n_simulations)**
```python
# 蒙特卡洛模拟法
var = var_calc.monte_carlo_var(returns, n_simulations=10000)
```

**conditional_var(returns)**
```python
# 计算CVaR
cvar = var_calc.conditional_var(returns)
```

**calculate_portfolio_var(returns_df, weights)**
```python
# 计算投资组合VaR
import numpy as np

weights = np.array([0.3, 0.4, 0.3])
result = var_calc.calculate_portfolio_var(returns_df, weights)
print(f"组合VaR: {result['var']}")
print(f"组合CVaR: {result['cvar']}")
```

---

### RiskExposureAnalyzer

风险敞口分析。

#### 初始化

```python
from module_05_risk_management import RiskExposureAnalyzer, ExposureConfig

config = ExposureConfig(
    max_single_stock=0.30,      # 单股最大30%
    max_sector=0.50,            # 单行业最大50%
    max_correlation=0.70        # 最大相关性
)

exposure_analyzer = RiskExposureAnalyzer(config)
```

#### 主要方法

**analyze_exposure(portfolio, returns_data, sector_mapping)**
```python
sector_mapping = {
    '000001': '金融',
    '600036': '金融',
    '000858': '消费'
}

exposure = exposure_analyzer.analyze_exposure(
    portfolio=portfolio,
    returns_data=returns_df,
    sector_mapping=sector_mapping
)

print(f"总敞口: {exposure.total_exposure}")
print(f"单股最大: {exposure.single_stock_max}")
print(f"行业集中度: {exposure.sector_concentration}")
print(f"违规项: {exposure.violations}")
```

---

## 💰 仓位管理 API

### KellyCriterion

凯利公式最优仓位计算。

#### 初始化

```python
from module_05_risk_management import KellyCriterion

kelly = KellyCriterion(
    max_kelly_fraction=0.25,    # 最大凯利分数
    min_kelly_fraction=0.01     # 最小凯利分数
)
```

#### 主要方法

**calculate_kelly_fraction(returns)**
```python
# 根据历史收益率计算凯利分数
result = kelly.calculate_kelly_fraction(returns)

print(f"凯利分数: {result.kelly_fraction}")
print(f"推荐仓位: {result.recommended_position}")
print(f"胜率: {result.win_rate}")
print(f"平均盈利: {result.avg_win}")
print(f"平均亏损: {result.avg_loss}")
```

**calculate_position_size(account_value, signal_strength, volatility, returns)**
```python
# 计算实际仓位大小
position_size = kelly.calculate_position_size(
    account_value=100000,
    signal_strength=0.8,
    volatility=0.02,
    returns=stock_returns
)

print(f"建议投入金额: {position_size}")
```

**optimize_portfolio_kelly(returns_df)**
```python
# 多资产凯利优化
optimal_weights = kelly.optimize_portfolio_kelly(returns_df)

for symbol, weight in optimal_weights.items():
    print(f"{symbol}: {weight:.2%}")
```

---

### DynamicPositionSizer

动态仓位管理器（支持6种方法）。

#### 初始化

```python
from module_05_risk_management import (
    DynamicPositionSizer,
    PositionSizingConfig,
    PositionSizingMethod
)

config = PositionSizingConfig(
    max_position_size=0.20,         # 最大仓位20%
    min_position_size=0.01,         # 最小仓位1%
    target_volatility=0.15,         # 目标波动率15%
    risk_per_trade=0.02,            # 单笔风险2%
    max_total_exposure=0.95,        # 最大总敞口95%
    correlation_threshold=0.7,      # 相关性阈值
    concentration_limit=0.30        # 集中度限制30%
)

sizer = DynamicPositionSizer(config)
```

#### 主要方法

**calculate_position_size(symbol, current_price, account_value, signal_strength, confidence, historical_returns, method)**
```python
# 计算单个资产的仓位
result = sizer.calculate_position_size(
    symbol='000001',
    current_price=15.5,
    account_value=100000,
    signal_strength=0.8,        # 信号强度 0-1
    confidence=0.75,            # 置信度 0-1
    historical_returns=returns,
    method=PositionSizingMethod.ADAPTIVE  # 或其他方法
)

print(f"推荐仓位: {result.recommended_size:.2%}")
print(f"推荐股数: {result.recommended_shares}")
print(f"仓位价值: {result.position_value}")
print(f"风险金额: {result.risk_amount}")
print(f"市场状态: {result.market_regime}")
```

**支持的方法：**
```python
# 6种仓位计算方法
PositionSizingMethod.ADAPTIVE           # 自适应（综合多因素）
PositionSizingMethod.KELLY              # 凯利准则
PositionSizingMethod.VOLATILITY_TARGET  # 目标波动率
PositionSizingMethod.RISK_PARITY        # 风险平价
PositionSizingMethod.CONFIDENCE_WEIGHTED # 置信度加权
PositionSizingMethod.FIXED              # 固定比例
```

**calculate_multi_position_allocation(signals, account_value, current_prices, returns_data)**
```python
# 多仓位联合配置
signals = {
    '000001': {'strength': 0.9, 'confidence': 0.85},
    '600036': {'strength': 0.8, 'confidence': 0.75},
    '000858': {'strength': 0.7, 'confidence': 0.80},
}

current_prices = {
    '000001': 15.5,
    '600036': 45.2,
    '000858': 180.0
}

results = sizer.calculate_multi_position_allocation(
    signals=signals,
    account_value=100000,
    current_prices=current_prices,
    returns_data=returns_df
)

for symbol, result in results.items():
    print(f"{symbol}: 仓位={result.recommended_size:.2%}, 股数={result.recommended_shares}")
```

**get_position_statistics()**
```python
# 获取仓位统计信息
stats = sizer.get_position_statistics()
print(f"总计算次数: {stats['total_calculations']}")
print(f"平均仓位: {stats['avg_position_size']:.2%}")
```

---

### RiskParity

风险平价仓位分配。

#### 初始化

```python
from module_05_risk_management import RiskParity, RiskParityConfig

config = RiskParityConfig(
    target_risk=0.10,           # 目标风险10%
    min_weight=0.05,            # 最小权重5%
    max_weight=0.40,            # 最大权重40%
    risk_aversion=1.0           # 风险厌恶系数
)

risk_parity = RiskParity(config)
```

#### 主要方法

**apply_risk_parity_allocation(returns_df)**
```python
# 计算风险平价权重
result = risk_parity.apply_risk_parity_allocation(returns_df)

print("权重分配:")
for asset, weight in zip(result.asset_names, result.weights):
    print(f"  {asset}: {weight:.2%}")

print(f"\n组合波动率: {result.portfolio_volatility:.2%}")
print(f"有效资产数: {result.effective_n_assets:.1f}")
```

---

## 📈 投资组合优化 API

### PortfolioWeightOptimizer

投资组合权重优化器（支持10种方法）。

#### 初始化

```python
from module_05_risk_management import (
    PortfolioWeightOptimizer,
    OptimizationConfig,
    OptimizationMethod,
    OptimizationObjective
)

config = OptimizationConfig(
    min_weight=0.05,                # 最小权重5%
    max_weight=0.35,                # 最大权重35%
    target_return=0.15,             # 目标收益15%
    target_volatility=0.15,         # 目标波动率15%
    risk_free_rate=0.03,            # 无风险利率3%
    max_leverage=1.0,               # 最大杠杆
    allow_short=False               # 是否允许做空
)

optimizer = PortfolioWeightOptimizer(config)
```

#### 主要方法

**optimize(returns_data, method, objective)**
```python
# 优化投资组合权重
result = optimizer.optimize(
    returns_data=returns_df,
    method=OptimizationMethod.MAX_SHARPE,
    objective=OptimizationObjective.MAXIMIZE_SHARPE
)

print("最优权重:")
for asset, weight in result.weights.items():
    if weight > 0.01:
        print(f"  {asset}: {weight:.2%}")

print(f"\n预期收益率: {result.expected_return:.2%}")
print(f"波动率: {result.volatility:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
print(f"有效资产数: {result.effective_n:.1f}")
```

**支持的优化方法：**
```python
OptimizationMethod.MEAN_VARIANCE      # 均值方差优化
OptimizationMethod.MIN_VARIANCE       # 最小方差
OptimizationMethod.MAX_SHARPE         # 最大夏普比率
OptimizationMethod.MAX_RETURN         # 最大收益
OptimizationMethod.RISK_PARITY        # 风险平价
OptimizationMethod.EQUAL_WEIGHT       # 等权重
OptimizationMethod.INVERSE_VOLATILITY # 反波动率
OptimizationMethod.MAX_DIVERSIFICATION # 最大分散化
OptimizationMethod.MIN_CORRELATION    # 最小相关性
OptimizationMethod.ENSEMBLE           # 集成方法
```

**优化目标：**
```python
OptimizationObjective.MAXIMIZE_RETURN   # 最大化收益
OptimizationObjective.MINIMIZE_RISK     # 最小化风险
OptimizationObjective.MAXIMIZE_SHARPE   # 最大化夏普比率
```

**optimize_with_constraints(returns_data, sector_mapping, sector_limits, method)**
```python
# 带行业约束的优化
sector_mapping = {
    '000001': '金融',
    '600036': '金融',
    '000858': '消费'
}

sector_limits = {
    '金融': (0.2, 0.5),  # 金融股占20%-50%
    '消费': (0.1, 0.3),  # 消费股占10%-30%
}

result = optimizer.optimize_with_constraints(
    returns_data=returns_df,
    sector_mapping=sector_mapping,
    sector_limits=sector_limits,
    method=OptimizationMethod.MAX_SHARPE
)
```

**backtest_optimization(returns_data, rebalance_frequency, method)**
```python
# 回测优化策略
backtest_results = optimizer.backtest_optimization(
    returns_data=returns_df,
    rebalance_frequency='monthly',  # 'daily', 'weekly', 'monthly', 'quarterly'
    method=OptimizationMethod.MAX_SHARPE
)

print(f"总收益: {backtest_results['total_return']:.2f}%")
print(f"年化收益: {backtest_results['annualized_return']:.2f}%")
print(f"年化波动率: {backtest_results['annualized_volatility']:.2f}%")
print(f"夏普比率: {backtest_results['sharpe_ratio']:.2f}")
print(f"最大回撤: {backtest_results['max_drawdown']:.2f}%")
```

**generate_efficient_frontier(returns_data, n_portfolios)**
```python
# 生成有效前沿
frontier = optimizer.generate_efficient_frontier(
    returns_data=returns_df,
    n_portfolios=50
)

# frontier 是包含 n_portfolios 个组合的列表
for portfolio in frontier:
    print(f"收益: {portfolio['return']:.2%}, 风险: {portfolio['risk']:.2%}")
```

---

### MeanVarianceOptimizer

经典均值方差优化。

#### 初始化

```python
from module_05_risk_management import MeanVarianceOptimizer, MVOConfig

config = MVOConfig(
    min_weight=0.0,
    max_weight=1.0,
    risk_aversion=1.0,
    target_return=None,
    target_volatility=None
)

optimizer = MeanVarianceOptimizer(config)
```

#### 主要方法

**optimize(expected_returns, cov_matrix, objective)**
```python
# 输入预期收益和协方差矩阵
expected_returns = returns_df.mean() * 252  # 年化
cov_matrix = returns_df.cov() * 252

result = optimizer.optimize(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    objective=OptimizationObjective.MAXIMIZE_SHARPE
)
```

**calculate_efficient_frontier(expected_returns, cov_matrix, n_points)**
```python
# 计算有效前沿
frontier = optimizer.calculate_efficient_frontier(
    expected_returns=expected_returns,
    cov_matrix=cov_matrix,
    n_points=50
)
```

---

## 🛑 止损策略 API

### StopLossManager

基础止损管理器。

#### 初始化

```python
from module_05_risk_management import StopLossManager, StopLossConfig

config = StopLossConfig(
    method='atr',               # 'fixed', 'percent', 'atr'
    atr_multiplier=2.0,         # ATR倍数
    max_loss_percent=0.05,      # 最大亏损5%
    trailing_stop=True          # 启用移动止损
)

stop_manager = StopLossManager(config)
```

#### 主要方法

**calculate_stop_loss(entry_price, current_price, atr, position_type)**
```python
# 计算止损价格
result = stop_manager.calculate_stop_loss(
    entry_price=15.5,
    current_price=16.8,
    atr=0.5,                    # 来自技术指标
    position_type='long'        # 'long' 或 'short'
)

print(f"止损价: {result.stop_price}")
print(f"最大损失: {result.max_loss}")
print(f"最大损失百分比: {result.max_loss_percent:.2%}")
```

**update_trailing_stop(current_price, current_stop, highest_price)**
```python
# 更新移动止损
new_stop = stop_manager.update_trailing_stop(
    current_price=17.5,
    current_stop=15.0,
    highest_price=18.0
)
```

**check_stop_triggered(current_price, stop_price, position_type)**
```python
# 检查是否触发止损
if stop_manager.check_stop_triggered(current_price, stop_price, 'long'):
    print("止损触发！")
```

---

### AdaptiveStopLoss

自适应止损（根据波动率和市场状态动态调整）。

#### 初始化

```python
from module_05_risk_management import AdaptiveStopLoss, StopLossConfig

config = StopLossConfig(
    method='atr',
    atr_multiplier=2.0,
    max_loss_percent=0.05,
    trailing_stop=True,
    time_stop_days=30
)

adaptive_stop = AdaptiveStopLoss(config)
```

#### 主要方法

**calculate_stop(entry_price, current_price, volatility, holding_period, unrealized_profit)**
```python
# 计算自适应止损
result = adaptive_stop.calculate_stop(
    entry_price=15.5,
    current_price=16.8,
    volatility=0.025,           # 当前波动率
    holding_period=10,          # 持有天数
    unrealized_profit=0.08      # 未实现收益8%
)

print(f"止损价: {result.stop_price}")
print(f"止损类型: {result.stop_type}")
```

---

### TrailingStop

追踪止损。

#### 初始化

```python
from module_05_risk_management import TrailingStop, TrailingStopConfig

config = TrailingStopConfig(
    initial_stop_pct=0.05,      # 初始止损5%
    trailing_pct=0.03,          # 追踪距离3%
    activation_profit=0.10,     # 激活利润10%
    use_atr=True,               # 使用ATR
    atr_multiplier=2.0
)

trailing = TrailingStop(config)
```

#### 主要方法

**initialize(entry_price, position_type, atr)**
```python
# 初始化追踪止损
state = trailing.initialize(
    entry_price=15.5,
    position_type='long',
    atr=0.5
)
```

**update(current_price, atr)**
```python
# 更新追踪止损
update = trailing.update(
    current_price=17.5,
    atr=0.5
)

print(f"新止损价: {update.new_stop_price}")
print(f"止损触发: {update.stop_triggered}")
print(f"利润保护: {update.profit_locked:.2%}")
```

---

## 🎯 压力测试 API

### MonteCarloSimulator

蒙特卡洛模拟。

#### 初始化

```python
from module_05_risk_management import (
    MonteCarloSimulator,
    SimulationConfig,
    DistributionType
)

config = SimulationConfig(
    n_simulations=10000,        # 模拟次数
    time_horizon=252,           # 时间跨度（交易日）
    confidence_levels=[0.95, 0.99],
    distribution=DistributionType.NORMAL,  # 或 STUDENT_T, HISTORICAL
    random_seed=42
)

simulator = MonteCarloSimulator(config)
```

#### 主要方法

**simulate_portfolio(initial_value, expected_returns, cov_matrix)**
```python
# 模拟投资组合未来走势
result = simulator.simulate_portfolio(
    initial_value=100000,
    expected_returns=returns_df.mean(),
    cov_matrix=returns_df.cov()
)

print(f"预期终值: {result.expected_terminal_value}")
print(f"VaR (95%): {result.var_95}")
print(f"CVaR (95%): {result.cvar_95}")
print(f"最大损失概率: {result.probability_of_loss:.2%}")
```

**simulate_paths(portfolio, returns_data)**
```python
# 生成模拟路径
paths = simulator.simulate_paths(
    portfolio=portfolio,
    returns_data=returns_df
)

# paths 是包含所有模拟路径的列表
for i, path in enumerate(paths[:5]):  # 查看前5条路径
    print(f"路径 {i+1} 终值: {path.terminal_value}")
```

---

### ScenarioGenerator

压力情景生成器。

#### 初始化

```python
from module_05_risk_management import (
    ScenarioGenerator,
    ScenarioConfig,
    ScenarioType
)

config = ScenarioConfig(
    scenario_types=[
        ScenarioType.MARKET_CRASH,
        ScenarioType.VOLATILITY_SPIKE,
        ScenarioType.CORRELATION_BREAKDOWN
    ],
    severity_level=0.95,        # 严重程度
    historical_lookback=252     # 历史回溯期
)

generator = ScenarioGenerator(config)
```

#### 主要方法

**generate_scenarios(returns_data)**
```python
# 生成压力情景
scenario_set = generator.generate_scenarios(returns_data=returns_df)

for scenario in scenario_set.scenarios:
    print(f"情景: {scenario.name}")
    print(f"  预期损失: {scenario.expected_loss:.2%}")
    print(f"  最大损失: {scenario.max_loss:.2%}")
    print(f"  发生概率: {scenario.probability:.2%}")
```

**apply_scenario(portfolio, scenario, returns_data)**
```python
# 应用情景到投资组合
result = generator.apply_scenario(
    portfolio=portfolio,
    scenario=scenario,
    returns_data=returns_df
)

print(f"情景下损失: {result['loss']}")
print(f"新组合价值: {result['new_value']}")
```

---

## 💾 数据库管理 API

### RiskDatabaseManager

风险数据持久化。

#### 获取实例

```python
from module_05_risk_management import get_risk_database_manager

risk_db = get_risk_database_manager()
```

#### 主要方法

**保存风险数据**

```python
from datetime import datetime

# 保存投资组合风险
risk_db.save_portfolio_risk(
    portfolio_id='main_portfolio',
    risk_metrics=risk_metrics,
    timestamp=datetime.now()
)

# 保存止损记录
risk_db.save_stop_loss(
    symbol='000001',
    entry_price=15.5,
    stop_price=14.7,
    max_loss=800,
    max_loss_percent=0.05,
    stop_type='atr',
    reason='ATR-based stop loss',
    timestamp=datetime.now()
)

# 保存敞口分析
risk_db.save_exposure_analysis(
    portfolio_id='main_portfolio',
    exposure=exposure_result,
    timestamp=datetime.now()
)

# 保存压力测试结果
risk_db.save_stress_test_result(
    portfolio_id='main_portfolio',
    scenario_name='market_crash',
    result=stress_result,
    timestamp=datetime.now()
)
```

**查询风险数据**

```python
# 查询风险历史
risk_history = risk_db.get_portfolio_risk_history(
    portfolio_id='main_portfolio',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# 查询止损历史
stop_loss_history = risk_db.get_stop_loss_history(
    symbol='000001',
    start_date='2024-01-01'
)

# 查询当前敞口
current_exposure = risk_db.get_current_exposure(
    portfolio_id='main_portfolio'
)

# 查询压力测试历史
stress_history = risk_db.get_stress_test_history(
    portfolio_id='main_portfolio'
)
```

**数据库统计**

```python
# 获取统计信息
stats = risk_db.get_database_stats()
print(f"数据库大小: {stats['database_size_mb']:.2f} MB")
print(f"风险记录数: {stats['total_risk_records']}")
print(f"止损记录数: {stats['total_stop_loss_records']}")

# 清理旧数据
risk_db.cleanup_old_data(days_to_keep=365)
```

---

## 🔗 与其他模块集成示例

### 与 Module 01 (数据管道) 集成

```python
from module_01_data_pipeline import AkshareDataCollector, get_database_manager
from module_05_risk_management import PortfolioRiskAnalyzer, RiskConfig

# 获取数据
collector = AkshareDataCollector()
db_manager = get_database_manager()

symbols = ['000001', '600036', '000858']
returns_data = {}

for symbol in symbols:
    prices = db_manager.get_stock_prices(symbol, '2023-01-01', '2024-12-01')
    returns_data[symbol] = prices['close'].pct_change().dropna()

returns_df = pd.DataFrame(returns_data)

# 风险分析
analyzer = PortfolioRiskAnalyzer(RiskConfig())
risk_metrics = analyzer.analyze_portfolio_risk(portfolio, returns_df)
```

### 与 Module 02 (特征工程) 集成

```python
from module_02_feature_engineering import TechnicalIndicators
from module_05_risk_management import StopLossManager, StopLossConfig

# 计算技术指标
calculator = TechnicalIndicators()
atr = calculator.calculate_atr(prices['high'], prices['low'], prices['close'], window=14)

# 基于ATR设置止损
stop_manager = StopLossManager(StopLossConfig(method='atr', atr_multiplier=2.0))
stop_loss = stop_manager.calculate_stop_loss(
    entry_price=15.5,
    current_price=16.8,
    atr=atr.iloc[-1],
    position_type='long'
)
```

### 与 Module 03 (AI模型) 集成

```python
from module_03_ai_models import LSTMModel
from module_05_risk_management import DynamicPositionSizer, PositionSizingConfig

# 获取AI预测信号
lstm_model = LSTMModel.load_model("price_predictor")
prediction = lstm_model.predict(latest_features)

signal_strength = prediction['confidence']
confidence = prediction['probability']

# 根据AI信号调整仓位
sizer = DynamicPositionSizer(PositionSizingConfig())
result = sizer.calculate_position_size(
    symbol='000001',
    current_price=15.5,
    account_value=100000,
    signal_strength=signal_strength,
    confidence=confidence,
    historical_returns=returns
)
```

### 与 Module 08 (执行) 集成

```python
from module_08_execution import OrderManager
from module_05_risk_management import PortfolioRiskAnalyzer, RiskConfig

# 下单前风险检查
def place_order_with_risk_check(symbol, quantity, price, portfolio, returns_df):
    # 模拟新订单后的投资组合
    simulated_portfolio = portfolio.copy()
    simulated_portfolio[symbol] = {
        'weight': calculate_new_weight(quantity, price),
        'shares': quantity,
        'cost': price
    }
    
    # 风险评估
    analyzer = PortfolioRiskAnalyzer(RiskConfig())
    risk_metrics = analyzer.analyze_portfolio_risk(simulated_portfolio, returns_df)
    
    # 检查风险限额
    if risk_metrics['var_95'] > MAX_VAR_LIMIT:
        print(f"订单被拒绝: VaR超限 ({risk_metrics['var_95']:.2%} > {MAX_VAR_LIMIT:.2%})")
        return False
    
    # 通过风险检查，提交订单
    order_manager = OrderManager()
    order = order_manager.create_order(symbol, quantity, price)
    return order_manager.submit_order(order)
```

---

## 🔧 便捷函数

模块提供了简化版便捷函数：

```python
from module_05_risk_management import (
    calculate_portfolio_var,
    calculate_kelly_position,
    calculate_risk_parity_weights,
    calculate_dynamic_position,
    calculate_adaptive_stop,
    analyze_exposure,
    run_monte_carlo_simulation,
    generate_stress_scenarios,
    optimize_portfolio
)

# 快速计算VaR
var = calculate_portfolio_var(returns_df, weights, confidence_level=0.95)

# 快速凯利仓位
kelly_weights = calculate_kelly_position(returns_df)

# 快速风险平价
rp_weights = calculate_risk_parity_weights(returns_df)

# 快速动态仓位
position = calculate_dynamic_position(
    symbol='000001',
    current_price=15.5,
    account_value=100000,
    signal_strength=0.8,
    confidence=0.75,
    historical_returns=returns
)

# 快速自适应止损
stop = calculate_adaptive_stop(
    entry_price=15.5,
    current_price=16.8,
    volatility=0.025
)

# 快速敞口分析
exposure = analyze_exposure(portfolio, returns_df)

# 快速蒙特卡洛
mc_result = run_monte_carlo_simulation(
    portfolio=portfolio,
    returns_data=returns_df,
    n_simulations=10000
)

# 快速压力情景
scenarios = generate_stress_scenarios(returns_df)

# 快速组合优化
optimal_weights = optimize_portfolio(
    returns_df,
    method='max_sharpe'
)
```

---

## 📝 完整工作流示例

```python
from datetime import datetime, timedelta
import pandas as pd
from module_01_data_pipeline import AkshareDataCollector, get_database_manager
from module_02_feature_engineering import TechnicalIndicators
from module_05_risk_management import (
    PortfolioRiskAnalyzer, RiskConfig,
    DynamicPositionSizer, PositionSizingConfig,
    PortfolioWeightOptimizer, OptimizationConfig, OptimizationMethod,
    StopLossManager, StopLossConfig,
    get_risk_database_manager
)

# 1. 获取数据
collector = AkshareDataCollector()
symbols = ['000001', '600036', '000858']
returns_data = {}

for symbol in symbols:
    prices = collector.fetch_stock_history(
        symbol,
        (datetime.now() - timedelta(days=365)).strftime('%Y%m%d'),
        datetime.now().strftime('%Y%m%d')
    )
    returns_data[symbol] = prices['close'].pct_change().dropna()

returns_df = pd.DataFrame(returns_data).dropna()

# 2. 投资组合优化
optimizer = PortfolioWeightOptimizer(OptimizationConfig(
    min_weight=0.10,
    max_weight=0.40
))

opt_result = optimizer.optimize(
    returns_data=returns_df,
    method=OptimizationMethod.MAX_SHARPE
)

print("最优权重:")
for asset, weight in opt_result.weights.items():
    print(f"  {asset}: {weight:.2%}")

# 3. 计算具体仓位
account_value = 100000
sizer = DynamicPositionSizer(PositionSizingConfig())

signals = {
    symbol: {'strength': 0.8, 'confidence': 0.75}
    for symbol in symbols
}

current_prices = {
    '000001': 15.5,
    '600036': 45.2,
    '000858': 180.0
}

positions = sizer.calculate_multi_position_allocation(
    signals=signals,
    account_value=account_value,
    current_prices=current_prices,
    returns_data=returns_df
)

# 4. 设置止损
tech_calc = TechnicalIndicators()
stop_manager = StopLossManager(StopLossConfig(method='atr', atr_multiplier=2.0))

portfolio = {}
for symbol, result in positions.items():
    prices = collector.fetch_stock_history(symbol, '20240101', '20241231')
    atr = tech_calc.calculate_atr(prices['high'], prices['low'], prices['close'])
    
    stop_loss = stop_manager.calculate_stop_loss(
        entry_price=current_prices[symbol],
        current_price=current_prices[symbol],
        atr=atr.iloc[-1],
        position_type='long'
    )
    
    portfolio[symbol] = {
        'weight': result.recommended_size,
        'shares': result.recommended_shares,
        'cost': current_prices[symbol],
        'stop_price': stop_loss.stop_price
    }

# 5. 风险评估
analyzer = PortfolioRiskAnalyzer(RiskConfig())
risk_metrics = analyzer.analyze_portfolio_risk(portfolio, returns_df)

print(f"\n投资组合风险:")
print(f"  VaR (95%): {risk_metrics['var_95']:.2%}")
print(f"  夏普比率: {risk_metrics['sharpe_ratio']:.2f}")
print(f"  最大回撤: {risk_metrics['max_drawdown']:.2%}")

# 6. 保存到数据库
risk_db = get_risk_database_manager()
risk_db.save_portfolio_risk('main_portfolio', risk_metrics, datetime.now())

for symbol, pos in portfolio.items():
    risk_db.save_stop_loss(
        symbol=symbol,
        entry_price=pos['cost'],
        stop_price=pos['stop_price'],
        max_loss=(pos['cost'] - pos['stop_price']) * pos['shares'],
        max_loss_percent=(pos['cost'] - pos['stop_price']) / pos['cost'],
        stop_type='atr',
        reason='Initial setup',
        timestamp=datetime.now()
    )

print("\n✅ 风险管理流程完成!")
```

---

## 🧪 测试

运行测试：

```bash
# 测试核心风险分析功能
python tests/module05_test_risk_analysis.py

# 测试仓位优化功能
python tests/module05_test_position_optimization.py
```

---

## ⚙️ 配置

### 环境变量
- `MODULE05_DB_PATH`: 风险数据库路径（默认: `data/module05_risk_management.db`）
- `MODULE05_RISK_FREE_RATE`: 无风险利率（默认: `0.03`）
- `MODULE05_LOG_LEVEL`: 日志级别（默认: `INFO`）

### 数据库位置
默认使用 SQLite 数据库，位于 `data/module05_risk_management.db`

---

## 📚 参考资料

### 类型定义汇总

```python
# 枚举类型
PositionSizingMethod: ADAPTIVE, KELLY, VOLATILITY_TARGET, RISK_PARITY, CONFIDENCE_WEIGHTED, FIXED
OptimizationMethod: MEAN_VARIANCE, MIN_VARIANCE, MAX_SHARPE, MAX_RETURN, RISK_PARITY, EQUAL_WEIGHT, INVERSE_VOLATILITY, MAX_DIVERSIFICATION, MIN_CORRELATION, ENSEMBLE
OptimizationObjective: MAXIMIZE_RETURN, MINIMIZE_RISK, MAXIMIZE_SHARPE
MarketRegime: BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY, LOW_VOLATILITY
ScenarioType: MARKET_CRASH, VOLATILITY_SPIKE, CORRELATION_BREAKDOWN, INTEREST_RATE_SHOCK
DistributionType: NORMAL, STUDENT_T, HISTORICAL

# 配置类
RiskConfig, VaRConfig, ExposureConfig
PositionSizingConfig, RiskParityConfig
OptimizationConfig, MVOConfig
StopLossConfig, TrailingStopConfig
SimulationConfig, ScenarioConfig

# 结果类
KellyResult, RiskParityResult, PositionSizingResult
OptimizationResult, ExposureResult
StopLossResult, TrailingStopUpdate
MonteCarloResult, ScenarioSet
```