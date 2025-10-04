# Module 07 - 优化模块

## 概述

优化模块是 FinLoom 量化交易系统的智能优化引擎，负责投资组合优化、策略参数优化、多目标优化和超参数调优。该模块集成了经典优化算法和现代机器学习优化方法，为系统提供全方位的优化解决方案。

## 主要功能

### 1. 投资组合优化 (Portfolio Optimization)
- **MarkowitzOptimizer**: 马科维茨均值-方差优化
- **BlackLittermanOptimizer**: Black-Litterman模型
- **RiskParityOptimizer**: 风险平价优化
- **HierarchicalRiskParity**: 层次风险平价(HRP)
- **CVaROptimizer**: 条件风险价值优化

### 2. 策略优化 (Strategy Optimization)
- **StrategyOptimizer**: 策略参数优化
- **WalkForwardOptimizer**: 滚动窗口优化
- **GeneticAlgorithmOptimizer**: 遗传算法优化
- **ParticleSwarmOptimizer**: 粒子群优化

### 3. 超参数调优 (Hyperparameter Tuning)
- **BayesianOptimizer**: 贝叶斯优化
- **OptunaOptimizer**: Optuna超参数调优
- **GridSearchOptimizer**: 网格搜索
- **RandomSearchOptimizer**: 随机搜索

### 4. 多目标优化 (Multi-Objective Optimization)
- **NSGAOptimizer**: NSGA-II算法
- **MOEADOptimizer**: MOEA/D算法
- **ParetoFrontAnalyzer**: 帕累托前沿分析

### 5. 约束优化 (Constrained Optimization)
- **ConstraintManager**: 约束条件管理
- **PenaltyMethodOptimizer**: 惩罚函数法
- **BarrierMethodOptimizer**: 障碍函数法

## 快速开始

### 环境配置

```python
# 导入 Module 07 组件
from module_07_optimization import (
    MarkowitzOptimizer,
    BlackLittermanOptimizer,
    RiskParityOptimizer,
    StrategyOptimizer,
    BayesianOptimizer,
    NSGAOptimizer,
    OptimizationManager,
    get_optimization_database_manager
)

# 导入其他模块
from module_01_data_pipeline import get_database_manager
from module_02_feature_engineering import TechnicalIndicators
from module_05_risk_management import PortfolioRiskAnalyzer
```

### 基础使用示例

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. 投资组合优化 - 马科维茨模型
from module_07_optimization import MarkowitzOptimizer, MarkowitzConfig

# 配置优化参数
markowitz_config = MarkowitzConfig(
    objective='max_sharpe',      # 'max_sharpe', 'min_variance', 'max_return'
    risk_free_rate=0.03,         # 无风险利率
    target_return=None,          # 目标收益率
    max_position=0.30,           # 单股最大仓位
    min_position=0.05,           # 单股最小仓位
    allow_short=False            # 是否允许做空
)

# 创建优化器
markowitz = MarkowitzOptimizer(markowitz_config)

# 准备数据 - 获取股票收益率
symbols = ['000001', '600036', '000858', '000002', '600519']
db_manager = get_database_manager()

returns_data = {}
for symbol in symbols:
    prices = db_manager.get_stock_prices(symbol, 
                                        start_date='2023-01-01',
                                        end_date='2024-12-01')
    returns_data[symbol] = prices['close'].pct_change().dropna()

returns_df = pd.DataFrame(returns_data)

# 执行优化
optimal_weights = markowitz.optimize(returns_df)

print("马科维茨最优权重:")
for symbol, weight in optimal_weights.items():
    print(f"  {symbol}: {weight:.2%}")

# 获取优化结果
optimization_result = markowitz.get_optimization_result()
print(f"\n优化结果:")
print(f"  期望收益: {optimization_result['expected_return']:.2%}")
print(f"  预期波动率: {optimization_result['volatility']:.2%}")
print(f"  夏普比率: {optimization_result['sharpe_ratio']:.3f}")

# 2. Black-Litterman模型
from module_07_optimization import BlackLittermanOptimizer, BLConfig

# 配置Black-Litterman参数
bl_config = BLConfig(
    tau=0.05,                    # 不确定性参数
    risk_aversion=2.5,           # 风险厌恶系数
    market_cap_weights=None,     # 市值权重（可选）
    confidence_level=0.90        # 观点置信度
)

bl_optimizer = BlackLittermanOptimizer(bl_config)

# 定义市场观点
views = {
    '000001': {'type': 'absolute', 'return': 0.15, 'confidence': 0.80},  # 看好平安银行
    '600519': {'type': 'absolute', 'return': 0.20, 'confidence': 0.90},  # 看好贵州茅台
    'relative': [
        {'stocks': ['600036', '000001'], 'return': 0.05, 'confidence': 0.70}  # 招行比平安好5%
    ]
}

# 执行Black-Litterman优化
bl_weights = bl_optimizer.optimize_with_views(returns_df, views)

print("\nBlack-Litterman最优权重:")
for symbol, weight in bl_weights.items():
    print(f"  {symbol}: {weight:.2%}")

# 3. 风险平价优化
from module_07_optimization import RiskParityOptimizer, RPConfig

rp_config = RPConfig(
    risk_measure='volatility',   # 'volatility', 'cvar', 'drawdown'
    rebalance_frequency='monthly',
    transaction_cost=0.001       # 交易成本0.1%
)

rp_optimizer = RiskParityOptimizer(rp_config)

# 执行风险平价优化
rp_weights = rp_optimizer.optimize(returns_df)

print("\n风险平价最优权重:")
for symbol, weight in rp_weights.items():
    print(f"  {symbol}: {weight:.2%}")

# 计算风险贡献
risk_contribution = rp_optimizer.calculate_risk_contribution(rp_weights, returns_df)
print(f"\n风险贡献:")
for symbol, contribution in risk_contribution.items():
    print(f"  {symbol}: {contribution:.2%}")

# 4. 策略参数优化
from module_07_optimization import StrategyOptimizer, StrategyOptConfig

# 定义策略
def moving_average_strategy(data, short_window, long_window, stop_loss):
    """简单移动平均策略"""
    from module_02_feature_engineering import TechnicalIndicators
    
    calculator = TechnicalIndicators()
    data['sma_short'] = calculator.calculate_sma(data['close'], short_window)
    data['sma_long'] = calculator.calculate_sma(data['close'], long_window)
    
    # 生成信号
    data['signal'] = 0
    data.loc[data['sma_short'] > data['sma_long'], 'signal'] = 1
    data.loc[data['sma_short'] < data['sma_long'], 'signal'] = -1
    
    # 计算收益
    data['returns'] = data['close'].pct_change()
    data['strategy_returns'] = data['signal'].shift(1) * data['returns']
    
    # 应用止损
    cumulative_returns = (1 + data['strategy_returns']).cumprod()
    max_returns = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - max_returns) / max_returns
    
    data.loc[drawdown < -stop_loss, 'strategy_returns'] = 0
    
    return data['strategy_returns'].sum()

# 配置策略优化
strategy_config = StrategyOptConfig(
    optimization_method='bayesian',  # 'bayesian', 'genetic', 'grid'
    objective='sharpe',               # 'sharpe', 'return', 'sortino'
    n_iterations=100,
    cv_folds=5                        # 交叉验证折数
)

strategy_optimizer = StrategyOptimizer(strategy_config)

# 定义参数空间
param_space = {
    'short_window': (5, 30),
    'long_window': (30, 120),
    'stop_loss': (0.05, 0.20)
}

# 准备策略数据
stock_data = db_manager.get_stock_prices('000001', 
                                         start_date='2022-01-01',
                                         end_date='2024-12-01')

# 执行优化
best_params = strategy_optimizer.optimize_strategy(
    strategy_function=moving_average_strategy,
    data=stock_data,
    param_space=param_space
)

print("\n策略最优参数:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

optimization_history = strategy_optimizer.get_optimization_history()
print(f"\n最优夏普比率: {optimization_history['best_score']:.3f}")

# 5. 贝叶斯超参数优化
from module_07_optimization import BayesianOptimizer, BayesianConfig

# 用于AI模型超参数优化
bayesian_config = BayesianConfig(
    n_iterations=50,
    init_points=10,
    acquisition_function='ucb',      # 'ucb', 'ei', 'poi'
    kappa=2.5,                       # UCB参数
    random_state=42
)

bayesian_opt = BayesianOptimizer(bayesian_config)

# 定义目标函数（例如：优化LSTM模型）
def lstm_objective(hidden_size, num_layers, dropout, learning_rate):
    """LSTM模型性能评估函数"""
    from module_03_ai_models import LSTMModel, LSTMModelConfig
    
    # 创建配置
    config = LSTMModelConfig(
        sequence_length=60,
        hidden_size=int(hidden_size),
        num_layers=int(num_layers),
        dropout=dropout,
        learning_rate=learning_rate,
        epochs=10
    )
    
    # 训练模型并返回性能指标
    model = LSTMModel(config)
    # ... 训练逻辑 ...
    
    # 返回负损失（最大化问题）
    return -validation_loss

# 定义参数边界
param_bounds = {
    'hidden_size': (32, 128),
    'num_layers': (1, 3),
    'dropout': (0.1, 0.5),
    'learning_rate': (0.0001, 0.01)
}

# 执行贝叶斯优化
best_params = bayesian_opt.optimize(
    objective_function=lstm_objective,
    param_bounds=param_bounds
)

print("\n贝叶斯优化最优参数:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# 6. 多目标优化 - NSGA-II
from module_07_optimization import NSGAOptimizer, NSGAConfig

nsga_config = NSGAConfig(
    population_size=100,
    n_generations=50,
    crossover_prob=0.9,
    mutation_prob=0.1,
    objectives=['return', 'risk', 'drawdown']  # 多个目标
)

nsga_optimizer = NSGAOptimizer(nsga_config)

# 定义多目标函数
def portfolio_multi_objective(weights, returns_df):
    """投资组合多目标评估"""
    portfolio_returns = (returns_df * weights).sum(axis=1)
    
    # 目标1: 最大化收益
    total_return = portfolio_returns.mean() * 252
    
    # 目标2: 最小化风险
    volatility = portfolio_returns.std() * np.sqrt(252)
    
    # 目标3: 最小化最大回撤
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'return': total_return,
        'risk': -volatility,           # 负号因为要最小化
        'drawdown': -max_drawdown      # 负号因为要最小化
    }

# 执行多目标优化
pareto_front = nsga_optimizer.optimize(
    objective_function=portfolio_multi_objective,
    data=returns_df,
    n_variables=len(symbols)
)

print(f"\n找到 {len(pareto_front)} 个帕累托最优解")
print("帕累托前沿部分解:")
for i, solution in enumerate(pareto_front[:5]):
    print(f"  解{i+1}:")
    print(f"    收益: {solution['objectives']['return']:.2%}")
    print(f"    风险: {-solution['objectives']['risk']:.2%}")
    print(f"    最大回撤: {-solution['objectives']['drawdown']:.2%}")

# 7. 优化结果管理
optimization_db = get_optimization_database_manager()

# 保存马科维茨优化结果
optimization_db.save_portfolio_optimization(
    strategy_name='markowitz_max_sharpe',
    weights=optimal_weights,
    metrics=optimization_result,
    timestamp=datetime.now()
)

# 保存策略优化结果
optimization_db.save_strategy_optimization(
    strategy_name='moving_average',
    parameters=best_params,
    performance=optimization_history,
    timestamp=datetime.now()
)

# 保存超参数优化结果
optimization_db.save_hyperparameter_optimization(
    model_type='lstm',
    parameters=best_params,
    score=bayesian_opt.get_best_score(),
    timestamp=datetime.now()
)

print("\n✅ 优化任务完成！")
```

## API 参考

### MarkowitzOptimizer

马科维茨均值-方差优化器。

#### 构造函数
```python
MarkowitzOptimizer(config: MarkowitzConfig)
```

#### 配置参数 (MarkowitzConfig)
```python
@dataclass
class MarkowitzConfig:
    objective: str = 'max_sharpe'        # 'max_sharpe', 'min_variance', 'max_return'
    risk_free_rate: float = 0.03         # 无风险利率
    target_return: float = None          # 目标收益率
    target_risk: float = None            # 目标风险
    max_position: float = 1.0            # 单股最大仓位
    min_position: float = 0.0            # 单股最小仓位
    allow_short: bool = False            # 是否允许做空
    max_leverage: float = 1.0            # 最大杠杆
```

#### 主要方法

**optimize(returns: pd.DataFrame) -> Dict[str, float]**
- 执行投资组合优化
- 返回最优权重字典

**calculate_efficient_frontier(returns: pd.DataFrame, n_points: int = 100) -> pd.DataFrame**
- 计算有效前沿
- 返回收益-风险曲线

**get_optimization_result() -> Dict[str, Any]**
- 获取详细优化结果
- 包括期望收益、波动率、夏普比率等

**optimize_with_constraints(returns: pd.DataFrame, constraints: List[Dict]) -> Dict[str, float]**
- 带约束条件的优化
- 支持线性和非线性约束

#### 使用示例
```python
optimizer = MarkowitzOptimizer(config)

# 基本优化
weights = optimizer.optimize(returns_df)

# 计算有效前沿
frontier = optimizer.calculate_efficient_frontier(returns_df)

# 带约束优化
constraints = [
    {'type': 'sector', 'sector': '银行', 'max_weight': 0.40},
    {'type': 'group', 'stocks': ['000001', '600036'], 'max_weight': 0.50}
]
weights = optimizer.optimize_with_constraints(returns_df, constraints)
```

### BlackLittermanOptimizer

Black-Litterman模型优化器。

#### 构造函数
```python
BlackLittermanOptimizer(config: BLConfig)
```

#### 配置参数 (BLConfig)
```python
@dataclass
class BLConfig:
    tau: float = 0.05                    # 不确定性参数
    risk_aversion: float = 2.5           # 风险厌恶系数
    market_cap_weights: Dict[str, float] = None  # 市值权重
    confidence_level: float = 0.90       # 默认置信度
```

#### 主要方法

**optimize_with_views(returns: pd.DataFrame, views: Dict) -> Dict[str, float]**
- 基于投资者观点优化
- views格式见使用示例

**calculate_implied_returns(returns: pd.DataFrame, market_weights: Dict[str, float]) -> pd.Series**
- 计算隐含收益率
- 基于市场均衡

**combine_views_and_market(market_returns: pd.Series, views: Dict) -> pd.Series**
- 结合市场观点和投资者观点
- 返回综合期望收益

#### 使用示例
```python
bl_optimizer = BlackLittermanOptimizer(config)

# 定义观点
views = {
    '000001': {'type': 'absolute', 'return': 0.15, 'confidence': 0.80},
    'relative': [
        {'stocks': ['600036', '000001'], 'return': 0.05, 'confidence': 0.70}
    ]
}

# 优化
weights = bl_optimizer.optimize_with_views(returns_df, views)
```

### RiskParityOptimizer

风险平价优化器。

#### 构造函数
```python
RiskParityOptimizer(config: RPConfig)
```

#### 配置参数 (RPConfig)
```python
@dataclass
class RPConfig:
    risk_measure: str = 'volatility'     # 'volatility', 'cvar', 'drawdown'
    rebalance_frequency: str = 'monthly' # 'daily', 'weekly', 'monthly', 'quarterly'
    transaction_cost: float = 0.001      # 交易成本
    risk_budget: Dict[str, float] = None # 自定义风险预算
```

#### 主要方法

**optimize(returns: pd.DataFrame) -> Dict[str, float]**
- 风险平价优化
- 使每个资产贡献相同风险

**calculate_risk_contribution(weights: Dict, returns: pd.DataFrame) -> Dict[str, float]**
- 计算风险贡献
- 返回各资产风险贡献比例

**optimize_with_risk_budget(returns: pd.DataFrame, risk_budget: Dict[str, float]) -> Dict[str, float]**
- 基于风险预算优化
- 可指定各资产风险贡献目标

#### 使用示例
```python
rp_optimizer = RiskParityOptimizer(config)

# 标准风险平价
weights = rp_optimizer.optimize(returns_df)

# 自定义风险预算
risk_budget = {'000001': 0.20, '600036': 0.30, '000858': 0.25, '000002': 0.15, '600519': 0.10}
weights = rp_optimizer.optimize_with_risk_budget(returns_df, risk_budget)
```

### StrategyOptimizer

策略参数优化器。

#### 构造函数
```python
StrategyOptimizer(config: StrategyOptConfig)
```

#### 配置参数 (StrategyOptConfig)
```python
@dataclass
class StrategyOptConfig:
    optimization_method: str = 'bayesian'  # 'bayesian', 'genetic', 'grid', 'random'
    objective: str = 'sharpe'               # 'sharpe', 'return', 'sortino', 'calmar'
    n_iterations: int = 100
    cv_folds: int = 5                       # 交叉验证折数
    test_size: float = 0.2                  # 测试集比例
    scoring_period: int = 252               # 评分周期
```

#### 主要方法

**optimize_strategy(strategy_function: Callable, data: pd.DataFrame, param_space: Dict) -> Dict[str, Any]**
- 优化策略参数
- 返回最优参数

**walk_forward_optimization(strategy_function: Callable, data: pd.DataFrame, param_space: Dict, window_size: int = 252) -> List[Dict]**
- 滚动窗口优化
- 避免过拟合

**get_optimization_history() -> Dict[str, Any]**
- 获取优化历史
- 包括所有尝试的参数和得分

**evaluate_strategy(strategy_function: Callable, data: pd.DataFrame, params: Dict) -> Dict[str, float]**
- 评估策略性能
- 返回多个性能指标

#### 使用示例
```python
strategy_optimizer = StrategyOptimizer(config)

# 优化策略
best_params = strategy_optimizer.optimize_strategy(
    strategy_function=my_strategy,
    data=stock_data,
    param_space={'param1': (10, 50), 'param2': (0.01, 0.1)}
)

# 滚动窗口优化
walk_forward_results = strategy_optimizer.walk_forward_optimization(
    strategy_function=my_strategy,
    data=stock_data,
    param_space=param_space,
    window_size=252
)
```

### BayesianOptimizer

贝叶斯优化器。

#### 构造函数
```python
BayesianOptimizer(config: BayesianConfig)
```

#### 配置参数 (BayesianConfig)
```python
@dataclass
class BayesianConfig:
    n_iterations: int = 50
    init_points: int = 10
    acquisition_function: str = 'ucb'      # 'ucb', 'ei', 'poi'
    kappa: float = 2.5                     # UCB探索参数
    xi: float = 0.01                       # EI探索参数
    random_state: int = None
```

#### 主要方法

**optimize(objective_function: Callable, param_bounds: Dict) -> Dict[str, float]**
- 执行贝叶斯优化
- 返回最优参数

**suggest_next_point() -> Dict[str, float]**
- 建议下一个尝试点
- 基于采集函数

**get_best_score() -> float**
- 获取最佳得分

**get_optimization_trace() -> pd.DataFrame**
- 获取优化轨迹
- 可视化优化过程

#### 使用示例
```python
bayesian_opt = BayesianOptimizer(config)

best_params = bayesian_opt.optimize(
    objective_function=my_objective,
    param_bounds={'x': (-5, 5), 'y': (0, 10)}
)
```

### NSGAOptimizer

NSGA-II多目标优化器。

#### 构造函数
```python
NSGAOptimizer(config: NSGAConfig)
```

#### 配置参数 (NSGAConfig)
```python
@dataclass
class NSGAConfig:
    population_size: int = 100
    n_generations: int = 50
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    objectives: List[str] = None           # 目标列表
    constraints: List[Dict] = None         # 约束条件
```

#### 主要方法

**optimize(objective_function: Callable, data: Any, n_variables: int) -> List[Dict]**
- 执行多目标优化
- 返回帕累托最优解集

**get_pareto_front() -> pd.DataFrame**
- 获取帕累托前沿
- 返回所有非支配解

**select_solution(pareto_front: List[Dict], preferences: Dict[str, float]) -> Dict**
- 根据偏好选择解
- 从帕累托前沿中选择

**visualize_pareto_front(pareto_front: List[Dict]) -> None**
- 可视化帕累托前沿
- 2D或3D图表

#### 使用示例
```python
nsga_optimizer = NSGAOptimizer(config)

pareto_front = nsga_optimizer.optimize(
    objective_function=multi_objective_func,
    data=returns_df,
    n_variables=5
)

# 根据偏好选择解
preferences = {'return': 0.5, 'risk': 0.3, 'drawdown': 0.2}
selected = nsga_optimizer.select_solution(pareto_front, preferences)
```

## 优化管理器

### OptimizationManager

统一优化任务管理。

#### 构造函数
```python
OptimizationManager(workspace_dir: str = "./optimization_workspace")
```

#### 主要方法

**create_optimization_task(task_name: str, optimizer_type: str, config: Any) -> str**
- 创建优化任务
- 返回task_id

**run_optimization_task(task_id: str) -> Dict[str, Any]**
- 运行优化任务
- 返回优化结果

**get_task_status(task_id: str) -> str**
- 获取任务状态
- 'pending', 'running', 'completed', 'failed'

**get_task_result(task_id: str) -> Dict[str, Any]**
- 获取任务结果

**schedule_periodic_optimization(task_name: str, schedule: str) -> str**
- 定时优化
- 支持cron表达式

**compare_optimization_results(task_ids: List[str]) -> pd.DataFrame**
- 比较多个优化结果

#### 使用示例
```python
opt_manager = OptimizationManager()

# 创建任务
task_id = opt_manager.create_optimization_task(
    task_name='portfolio_opt_daily',
    optimizer_type='markowitz',
    config=markowitz_config
)

# 运行任务
result = opt_manager.run_optimization_task(task_id)

# 定时优化
opt_manager.schedule_periodic_optimization(
    task_name='daily_rebalance',
    schedule='0 15 * * *'  # 每天15:00
)
```

## 数据库管理

### OptimizationDatabaseManager

优化数据专用数据库管理。

#### 使用方法
```python
from module_07_optimization import get_optimization_database_manager

opt_db = get_optimization_database_manager()
```

#### 主要方法

**保存优化结果**
- `save_portfolio_optimization(strategy_name: str, weights: Dict, metrics: Dict, timestamp: datetime) -> bool`
- `save_strategy_optimization(strategy_name: str, parameters: Dict, performance: Dict, timestamp: datetime) -> bool`
- `save_hyperparameter_optimization(model_type: str, parameters: Dict, score: float, timestamp: datetime) -> bool`
- `save_multi_objective_result(task_name: str, pareto_front: List[Dict], timestamp: datetime) -> bool`

**查询优化结果**
- `get_portfolio_optimization_history(strategy_name: str, start_date: datetime = None) -> List[Dict]`
- `get_strategy_optimization_history(strategy_name: str, limit: int = 100) -> List[Dict]`
- `get_best_hyperparameters(model_type: str) -> Dict`
- `get_pareto_solutions(task_name: str, timestamp: datetime = None) -> List[Dict]`

**统计和分析**
- `get_optimization_statistics() -> Dict[str, Any]`
- `compare_strategies(strategy_names: List[str]) -> pd.DataFrame`
- `get_database_stats() -> Dict[str, Any]`

#### 使用示例
```python
opt_db = get_optimization_database_manager()

# 保存优化结果
opt_db.save_portfolio_optimization(
    strategy_name='markowitz_max_sharpe',
    weights={'000001': 0.3, '600036': 0.4, '000858': 0.3},
    metrics={'sharpe': 1.85, 'return': 0.15, 'volatility': 0.12},
    timestamp=datetime.now()
)

# 查询历史
history = opt_db.get_portfolio_optimization_history('markowitz_max_sharpe')

# 比较策略
comparison = opt_db.compare_strategies(['markowitz', 'risk_parity', 'black_litterman'])
```

## 与其他模块集成

### 与 Module 02 (特征工程) 集成
```python
# 使用特征进行策略优化
from module_02_feature_engineering import TechnicalIndicators, FactorAnalyzer
from module_07_optimization import StrategyOptimizer

calculator = TechnicalIndicators()
factor_analyzer = FactorAnalyzer()

def factor_strategy(data, rsi_period, macd_fast, macd_slow):
    # 计算技术指标
    rsi = calculator.calculate_rsi(data['close'], rsi_period)
    macd_data = calculator.calculate_macd(data['close'], macd_fast, macd_slow)
    
    # 生成信号并评估
    # ...
    return performance_metric

# 优化因子参数
optimizer = StrategyOptimizer(config)
best_params = optimizer.optimize_strategy(
    strategy_function=factor_strategy,
    data=stock_data,
    param_space={'rsi_period': (10, 20), 'macd_fast': (8, 15), 'macd_slow': (20, 30)}
)
```

### 与 Module 03 (AI模型) 集成
```python
# 优化AI模型超参数
from module_03_ai_models import LSTMModel, LSTMModelConfig
from module_07_optimization import BayesianOptimizer

def train_lstm_with_params(hidden_size, num_layers, dropout, learning_rate):
    config = LSTMModelConfig(
        hidden_size=int(hidden_size),
        num_layers=int(num_layers),
        dropout=dropout,
        learning_rate=learning_rate
    )
    
    model = LSTMModel(config)
    metrics = model.train(X_train, y_train)
    
    # 返回验证集性能
    val_loss = model.evaluate(X_val, y_val)
    return -val_loss  # 最大化负损失

bayesian_opt = BayesianOptimizer(config)
best_params = bayesian_opt.optimize(
    objective_function=train_lstm_with_params,
    param_bounds={'hidden_size': (32, 128), 'num_layers': (1, 3), 
                  'dropout': (0.1, 0.5), 'learning_rate': (0.0001, 0.01)}
)
```

### 与 Module 05 (风险管理) 集成
```python
# 结合风险约束的投资组合优化
from module_05_risk_management import VaRCalculator, RiskLimitManager
from module_07_optimization import MarkowitzOptimizer

risk_calculator = VaRCalculator()
limit_manager = RiskLimitManager(config)

def portfolio_with_risk_constraints(weights, returns_df):
    # 计算投资组合收益
    portfolio_returns = (returns_df * weights).sum(axis=1)
    
    # 风险约束
    var_95 = risk_calculator.historical_var(portfolio_returns)
    
    # 检查限额
    if var_95 > 0.05:  # VaR不超过5%
        return -np.inf
    
    # 计算夏普比率
    sharpe = portfolio_returns.mean() / portfolio_returns.std()
    return sharpe

# 优化
markowitz = MarkowitzOptimizer(config)
optimal_weights = markowitz.optimize_with_constraints(
    returns_df,
    constraints=[{'type': 'risk', 'var_limit': 0.05}]
)
```

## 便捷函数

```python
# 快速投资组合优化
from module_07_optimization import quick_portfolio_optimize

weights = quick_portfolio_optimize(returns_df, method='markowitz', objective='max_sharpe')

# 快速策略参数优化
from module_07_optimization import quick_strategy_optimize

best_params = quick_strategy_optimize(
    strategy=my_strategy,
    data=stock_data,
    param_space=param_space,
    method='bayesian'
)

# 快速有效前沿计算
from module_07_optimization import calculate_efficient_frontier

frontier = calculate_efficient_frontier(returns_df, n_points=100)

# 快速超参数调优
from module_07_optimization import quick_hyperparameter_tune

best_params = quick_hyperparameter_tune(
    model_class=LSTMModel,
    X_train=X_train,
    y_train=y_train,
    param_space=param_space
)
```

## 测试和示例

### 运行完整测试
```bash
cd /Users/victor/Desktop/25fininnov/FinLoom-server
python tests/module07_optimization_test.py
```

### 测试覆盖内容
- 马科维茨优化测试
- Black-Litterman优化测试
- 风险平价优化测试
- 策略参数优化测试
- 贝叶斯优化测试
- 多目标优化测试
- 有效前沿计算测试
- 数据库操作测试
- 与其他模块集成测试

## 配置说明

### 环境变量
- `MODULE07_DB_PATH`: 优化数据库路径
- `MODULE07_WORKSPACE_DIR`: 优化工作空间目录
- `MODULE07_LOG_LEVEL`: 日志级别

### 优化配置文件
```yaml
# config/optimization_config.yaml
portfolio_optimization:
  default_method: 'markowitz'
  risk_free_rate: 0.03
  max_single_position: 0.30
  rebalance_frequency: 'monthly'

strategy_optimization:
  default_method: 'bayesian'
  n_iterations: 100
  cv_folds: 5

hyperparameter_tuning:
  default_method: 'bayesian'
  n_iterations: 50
  init_points: 10
```

## 性能基准

| 操作 | 数据规模 | 处理时间 | 内存使用 |
|------|----------|----------|----------|
| 马科维茨优化 | 50股票 | ~500ms | ~20MB |
| 有效前沿计算 | 50股票×100点 | ~5s | ~50MB |
| 贝叶斯优化 | 50次迭代 | ~30s | ~100MB |
| NSGA-II优化 | 100个体×50代 | ~60s | ~200MB |

## 总结

Module 07 优化模块提供了全面的优化解决方案：

### 功能完整性 ✅
- ✓ 经典投资组合优化（马科维茨、Black-Litterman、风险平价）
- ✓ 策略参数优化（贝叶斯、遗传算法、网格搜索）
- ✓ 超参数调优（Bayesian、Optuna、随机搜索）
- ✓ 多目标优化（NSGA-II、帕累托前沿）
- ✓ 约束优化（线性、非线性约束）

### 集成能力 ✅
- ✓ 与Module 02特征工程无缝集成
- ✓ 与Module 03 AI模型超参数优化集成
- ✓ 与Module 05风险管理约束集成
- ✓ 专用数据库存储优化历史

### 实用性 ✅
- ✓ 多种优化算法可选
- ✓ 滚动窗口优化避免过拟合
- ✓ 定时优化任务调度
- ✓ 完善的结果比较和分析

**结论**: Module 07 提供了专业级的优化工具箱，支持从简单的参数调优到复杂的多目标优化，是量化策略开发的强大助手。

