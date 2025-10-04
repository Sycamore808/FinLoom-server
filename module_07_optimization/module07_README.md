# Module 07 - 优化模块 API文档

## 模块概述

Module 07 提供全面的优化解决方案，包括超参数优化、策略参数优化、多目标优化和资源优化。所有数据保存到 `data/module07_optimization.db`。

## API 导入

```python
from module_07_optimization import (
    # 基础
    Parameter, OptimizationResult,
    # 超参数优化
    BayesianOptimizer, GridSearchOptimizer, OptunaOptimizer, RandomSearchOptimizer,
    # 多目标优化
    NSGAOptimizer, ParetoFrontier, PortfolioObjectives,
    # 策略优化
    StrategyOptimizer, PerformanceEvaluator, get_strategy_space,
    # 资源优化
    ComputeOptimizer, CostOptimizer, MemoryOptimizer,
    # 数据库
    get_optimization_database_manager,
    # 管理器
    get_optimization_manager
)
```

## 1. 超参数优化 API

### 1.1 贝叶斯优化

```python
from module_07_optimization import BayesianOptimizer, Parameter

# 定义参数空间
param_space = [
    Parameter(name="learning_rate", param_type="float", low=0.0001, high=0.01, log_scale=True),
    Parameter(name="hidden_size", param_type="int", low=32, high=128),
    Parameter(name="dropout", param_type="float", low=0.1, high=0.5)
]

# 定义目标函数
def objective(params):
    # 使用参数训练模型并返回性能指标
    lr = params["learning_rate"]
    hidden = params["hidden_size"]
    dropout = params["dropout"]
    # 训练并评估...
    return validation_loss

# 创建优化器
optimizer = BayesianOptimizer(
    parameter_space=param_space,
    objective_function=objective,
    maximize=False,  # 最小化损失
    n_trials=50,
    n_initial_points=10,
    acquisition_function="ei"  # 'ei', 'pi', 'ucb'
)

# 执行优化
result = optimizer.optimize()

print(f"最佳参数: {result.best_parameters}")
print(f"最佳值: {result.best_value}")
```

### 1.2 网格搜索

```python
from module_07_optimization import GridSearchOptimizer

optimizer = GridSearchOptimizer(
    parameter_space=param_space,
    objective_function=objective,
    maximize=False,
    n_grid_points=10  # 每个参数的网格点数
)

result = optimizer.optimize()
```

### 1.3 Optuna优化

```python
from module_07_optimization import OptunaOptimizer

optimizer = OptunaOptimizer(
    parameter_space=param_space,
    objective_function=objective,
    maximize=False,
    n_trials=100,
    sampler="tpe",  # 'tpe', 'cmaes', 'random'
    pruner="median"  # 'median', 'hyperband', None
)

result = optimizer.optimize()

# 获取参数重要性
importance = optimizer.get_parameter_importance()
print(f"参数重要性: {importance}")
```

### 1.4 随机搜索

```python
from module_07_optimization import RandomSearchOptimizer

optimizer = RandomSearchOptimizer(
    parameter_space=param_space,
    objective_function=objective,
    maximize=False,
    n_trials=50
)

result = optimizer.optimize()
```

## 2. 策略优化 API

### 2.1 策略参数优化

```python
from module_07_optimization import StrategyOptimizer, get_strategy_space
from module_01_data_pipeline import AkshareDataCollector

# 获取市场数据
collector = AkshareDataCollector()
market_data = collector.fetch_stock_history("000001", "20230101", "20241201")

# 定义策略类（需要有generate_signal方法）
class MyStrategy:
    def __init__(self, short_window, long_window, stop_loss):
        self.short_window = short_window
        self.long_window = long_window
        self.stop_loss = stop_loss
    
    def generate_signal(self, history):
        # 生成交易信号
        # 返回 Signal 对象或 None
        pass

# 获取预定义的参数空间
param_space = get_strategy_space("ma_crossover")
# 或自定义参数空间
# param_space = [Parameter(...), ...]

# 创建优化器
optimizer = StrategyOptimizer(
    strategy_class=MyStrategy,
    market_data=market_data,
    optimization_metric="sharpe_ratio",  # 优化目标
    test_split=0.2,  # 测试集比例
    walk_forward_windows=5  # Walk Forward窗口数
)

# 执行优化
result = optimizer.optimize(
    parameter_space=param_space,
    n_trials=100,
    optimizer_type="optuna"
)

print(f"最佳参数: {result['best_parameters']}")
print(f"训练性能: {result['train_performance']}")
print(f"测试性能: {result['test_performance']}")
```

### 2.2 性能评估

```python
from module_07_optimization import PerformanceEvaluator
import pandas as pd

# 创建评估器
evaluator = PerformanceEvaluator(annual_trading_days=252)

# 评估策略（需要收益率序列）
returns = pd.Series([0.01, -0.005, 0.02, ...])  # 策略收益率
metrics = evaluator.evaluate(
    returns=returns,
    risk_free_rate=0.03
)

print(f"年化收益: {metrics.annual_return:.2%}")
print(f"夏普比率: {metrics.sharpe_ratio:.2f}")
print(f"最大回撤: {metrics.max_drawdown:.2%}")
print(f"胜率: {metrics.win_rate:.2%}")
```

## 3. 多目标优化 API

### 3.1 NSGA-II优化

```python
from module_07_optimization import NSGAOptimizer, Parameter
import numpy as np

# 定义多个目标函数
def objective1(params):
    # 最小化风险
    return calculate_risk(params)

def objective2(params):
    # 最大化收益（转为最小化）
    return -calculate_return(params)

def objective3(params):
    # 最小化回撤
    return calculate_drawdown(params)

# 创建优化器
optimizer = NSGAOptimizer(
    parameter_space=param_space,
    objective_functions=[objective1, objective2, objective3],
    population_size=100,
    n_generations=50,
    crossover_prob=0.9,
    mutation_prob=0.1
)

# 执行优化
result = optimizer.optimize()

# 获取Pareto前沿
pareto_front = result["pareto_front"]
print(f"找到 {len(pareto_front)} 个Pareto最优解")

# 分析Pareto前沿
from module_07_optimization import ParetoFrontier

frontier = ParetoFrontier(
    solutions=pareto_front,
    objective_names=["risk", "return", "drawdown"]
)

# 根据偏好选择解
preferences = {"risk": 0.3, "return": 0.5, "drawdown": 0.2}
selected = frontier.select_solution_by_preference(preferences)
print(f"选中的解: {selected}")
```

### 3.2 投资组合多目标优化

```python
from module_07_optimization import create_portfolio_objectives, NSGAOptimizer
import pandas as pd

# 准备收益率数据
returns_data = pd.DataFrame({
    "000001": [0.01, -0.005, ...],
    "600036": [0.02, 0.01, ...],
    # ...
})

# 创建目标函数
objective_functions = create_portfolio_objectives(
    returns_data=returns_data,
    objective_names=["return", "risk", "drawdown"],
    risk_free_rate=0.03
)

# 定义权重参数空间
param_space = [
    Parameter(name=f"weight_{symbol}", param_type="float", low=0.0, high=1.0)
    for symbol in returns_data.columns
]

# 优化
optimizer = NSGAOptimizer(
    parameter_space=param_space,
    objective_functions=objective_functions,
    population_size=100,
    n_generations=50
)

result = optimizer.optimize()
```

## 4. 资源优化 API

### 4.1 计算资源优化

```python
from module_07_optimization import (
    ComputeOptimizer, ComputeResource, ComputeTask
)

# 定义资源
resources = [
    ComputeResource(
        resource_id="cpu1",
        resource_type="CPU",
        capacity=100.0,
        cost_per_unit=0.1,
        availability=1.0
    ),
    ComputeResource(
        resource_id="gpu1",
        resource_type="GPU",
        capacity=50.0,
        cost_per_unit=0.5,
        availability=1.0
    )
]

# 定义任务
tasks = [
    ComputeTask(
        task_id="task1",
        task_type="training",
        resource_requirements={"CPU": 20, "GPU": 10},
        priority=2,
        estimated_duration=2.0
    ),
    # ...
]

# 创建优化器
optimizer = ComputeOptimizer(
    resources=resources,
    optimization_objective="min_cost"  # 'min_cost', 'min_time', 'max_utilization'
)

# 分配资源
allocation = optimizer.allocate_resources(tasks)
print(f"资源分配: {allocation}")

# 计算成本
total_cost = optimizer.calculate_total_cost(allocation, tasks)
print(f"总成本: {total_cost}")

# 计算利用率
utilization = optimizer.calculate_utilization(allocation)
print(f"利用率: {utilization}")
```

### 4.2 成本优化

```python
from module_07_optimization import CostOptimizer, CostComponent

# 定义成本组成
components = [
    CostComponent(
        component_id="data_storage",
        component_type="storage",
        fixed_cost=100.0,
        variable_cost=0.1,  # 每GB
        volume=1000.0  # 当前使用1000GB
    ),
    CostComponent(
        component_id="compute",
        component_type="compute",
        fixed_cost=500.0,
        variable_cost=1.0,  # 每小时
        volume=200.0  # 当前使用200小时
    )
]

# 创建优化器
optimizer = CostOptimizer(cost_components=components)

# 计算总成本
total_cost = optimizer.calculate_total_cost()
print(f"总成本: {total_cost}")

# 成本明细
breakdown = optimizer.calculate_cost_breakdown()
print(f"成本明细: {breakdown}")

# 预算约束下的优化
priorities = {"data_storage": 0.3, "compute": 0.7}
allocation = optimizer.optimize_volume_allocation(
    budget=1000.0,
    priorities=priorities
)
print(f"优化后的使用量: {allocation}")

# 成本削减建议
recommendations = optimizer.recommend_cost_reduction(target_reduction=200.0)
for rec in recommendations:
    print(f"组件: {rec['component_id']}, 建议削减: {rec['recommended_reduction']}")
```

### 4.3 内存优化

```python
from module_07_optimization import MemoryOptimizer
import pandas as pd

# 创建优化器
optimizer = MemoryOptimizer(memory_limit_mb=1000.0)

# 获取内存使用
current_usage = optimizer.get_memory_usage()
print(f"当前内存使用: {current_usage:.2f}MB")

# 内存分析
profile = optimizer.profile_memory(top_n=10)
print(f"总内存: {profile.total_memory_mb:.2f}MB")
print(f"对象数量: {profile.object_count}")
print(f"最大对象: {profile.largest_objects}")

# 优化DataFrame
df = pd.DataFrame(...)  # 大型DataFrame
optimized_df = optimizer.optimize_dataframe(df)

# 数据分块
chunks = optimizer.chunk_dataframe(df, chunk_size=10000)
for chunk in chunks:
    # 处理每个chunk
    pass

# 缓存管理
optimizer.set_cache("key1", some_data, size_limit_mb=100)
cached = optimizer.get_cache("key1")
optimizer.clear_cache()

# 优化建议
suggestions = optimizer.suggest_optimizations()
for suggestion in suggestions:
    print(suggestion)
```

## 5. 数据库 API

### 5.1 保存优化结果

```python
from module_07_optimization import get_optimization_database_manager
from datetime import datetime

db = get_optimization_database_manager()

# 保存优化任务
db.save_optimization_task(
    task_id="opt_task_001",
    task_name="LSTM超参数优化",
    optimizer_type="bayesian",
    config={"n_trials": 50},
    status="pending"
)

# 更新任务状态
db.update_task_status("opt_task_001", "completed", datetime.now())

# 保存优化结果
db.save_optimization_result(
    task_id="opt_task_001",
    best_parameters={"learning_rate": 0.001, "hidden_size": 64},
    best_value=0.025,
    n_trials=50,
    n_successful_trials=48,
    total_time_seconds=300.0,
    convergence_history=[0.1, 0.08, 0.05, 0.025]
)

# 保存策略优化结果
db.save_strategy_optimization(
    strategy_name="ma_crossover",
    parameters={"short_window": 10, "long_window": 50},
    train_performance={"sharpe_ratio": 1.5, "annual_return": 0.20},
    test_performance={"sharpe_ratio": 1.3, "annual_return": 0.18},
    symbol="000001"
)

# 保存投资组合优化
db.save_portfolio_optimization(
    optimization_name="max_sharpe",
    weights={"000001": 0.3, "600036": 0.4, "000858": 0.3},
    expected_return=0.15,
    volatility=0.12,
    sharpe_ratio=1.25
)

# 保存多目标优化结果
db.save_multi_objective_result(
    task_id="multi_obj_001",
    pareto_front=pareto_solutions,
    objective_names=["return", "risk", "drawdown"]
)
```

### 5.2 查询优化结果

```python
# 获取优化结果
result = db.get_optimization_result("opt_task_001")
print(f"最佳参数: {result['best_parameters']}")

# 获取策略优化历史
history = db.get_strategy_optimization_history("ma_crossover", limit=10)
for record in history:
    print(f"日期: {record['optimization_date']}, 参数: {record['parameters']}")

# 数据库统计
stats = db.get_database_stats()
print(f"总任务数: {stats['total_tasks']}")
print(f"总结果数: {stats['total_results']}")
print(f"数据库大小: {stats['database_size_mb']:.2f}MB")
```

## 6. 优化管理器 API

```python
from module_07_optimization import get_optimization_manager

manager = get_optimization_manager()

# 创建优化任务
task_id = manager.create_optimization_task(
    task_name="策略参数优化",
    optimizer_type="optuna",
    parameter_space=param_space,
    objective_function=objective,
    n_trials=100
)

# 运行优化
result = manager.run_optimization(
    task_id=task_id,
    optimizer_type="optuna",
    parameter_space=param_space,
    objective_function=objective,
    n_trials=100
)

# 加载结果
loaded_result = manager.load_result(task_id)

# 比较多个优化结果
comparison = manager.compare_results([task_id1, task_id2, task_id3])
print(comparison)

# 获取最佳参数
best_params = manager.get_best_parameters(task_id)

# 列出所有任务
tasks = manager.list_tasks()
for task in tasks:
    print(f"任务: {task['task_name']}, 状态: {task.get('has_result')}")

# 清理旧任务
cleaned = manager.clean_old_tasks(days=30)
print(f"清理了 {cleaned} 个旧任务")
```

## 7. 完整示例

### 示例1: LSTM超参数优化

```python
from module_07_optimization import BayesianOptimizer, Parameter, get_optimization_database_manager
from module_03_ai_models import LSTMModel
from module_01_data_pipeline import AkshareDataCollector
from module_02_feature_engineering import TechnicalIndicators

# 1. 准备数据
collector = AkshareDataCollector()
data = collector.fetch_stock_history("000001", "20230101", "20241201")

calculator = TechnicalIndicators()
features = calculator.calculate_all_indicators(data)

# 2. 定义目标函数
def train_lstm(params):
    model = LSTMModel(config={
        "sequence_length": 60,
        "hidden_size": params["hidden_size"],
        "num_layers": params["num_layers"],
        "dropout": params["dropout"],
        "learning_rate": params["learning_rate"]
    })
    X, y = model.prepare_data(features, "close")
    metrics = model.train(X, y)
    return metrics["val_loss"]

# 3. 定义参数空间
param_space = [
    Parameter("hidden_size", "int", low=32, high=128),
    Parameter("num_layers", "int", low=1, high=3),
    Parameter("dropout", "float", low=0.1, high=0.5),
    Parameter("learning_rate", "float", low=0.0001, high=0.01, log_scale=True)
]

# 4. 优化
optimizer = BayesianOptimizer(
    parameter_space=param_space,
    objective_function=train_lstm,
    maximize=False,
    n_trials=50
)

result = optimizer.optimize()

# 5. 保存到数据库
db = get_optimization_database_manager()
db.save_optimization_result(
    task_id="lstm_opt_001",
    best_parameters=result.best_parameters,
    best_value=result.best_value,
    n_trials=result.n_trials,
    n_successful_trials=result.n_successful_trials,
    total_time_seconds=result.total_time_seconds
)

print(f"最佳参数: {result.best_parameters}")
print(f"最佳验证损失: {result.best_value}")
```

### 示例2: 投资组合多目标优化

```python
from module_07_optimization import NSGAOptimizer, ParetoFrontier, create_portfolio_objectives, Parameter
from module_01_data_pipeline import AkshareDataCollector
import pandas as pd

# 1. 获取多只股票数据
symbols = ["000001", "600036", "000858", "600519"]
returns_data = pd.DataFrame()

collector = AkshareDataCollector()
for symbol in symbols:
    data = collector.fetch_stock_history(symbol, "20230101", "20241201")
    returns_data[symbol] = data["close"].pct_change().dropna()

returns_data = returns_data.dropna()

# 2. 创建多目标函数
objective_functions = create_portfolio_objectives(
    returns_data=returns_data,
    objective_names=["return", "risk", "drawdown"],
    risk_free_rate=0.03
)

# 3. 定义权重参数空间
param_space = [
    Parameter(f"weight_{symbol}", "float", low=0.0, high=1.0)
    for symbol in symbols
]

# 4. 优化
optimizer = NSGAOptimizer(
    parameter_space=param_space,
    objective_functions=objective_functions,
    population_size=100,
    n_generations=50
)

result = optimizer.optimize()

# 5. 分析Pareto前沿
frontier = ParetoFrontier(
    solutions=result["pareto_front"],
    objective_names=["return", "risk", "drawdown"]
)

# 根据偏好选择
preferences = {"return": 0.5, "risk": 0.3, "drawdown": 0.2}
selected = frontier.select_solution_by_preference(preferences)

print(f"选中的投资组合权重: {selected['parameters']}")
print(f"目标值: {selected['objectives']}")

# 6. 保存到数据库
db = get_optimization_database_manager()
db.save_multi_objective_result(
    task_id="portfolio_opt_001",
    pareto_front=result["pareto_front"],
    objective_names=["return", "risk", "drawdown"]
)
```

## 8. 预定义策略参数空间

```python
from module_07_optimization import get_strategy_space

# 可用的策略空间
strategies = [
    "ma_crossover",      # 移动平均交叉
    "rsi",               # RSI策略
    "bollinger_bands",   # 布林带
    "macd",              # MACD
    "mean_reversion",    # 均值回归
    "momentum"           # 动量策略
]

# 获取策略参数空间
for strategy in strategies:
    space = get_strategy_space(strategy)
    print(f"{strategy}: {len(space)} 个参数")
```

## 环境要求

运行测试需要激活 conda 环境：

```bash
conda activate study
cd /Users/victor/Desktop/25fininnov/FinLoom-server
python tests/module07_optimization_test.py
```

## 注意事项

1. 所有优化结果自动保存到 `data/module07_optimization.db`
2. 参数空间使用 `Parameter` 类定义，支持 float、int、categorical、bool 类型
3. 目标函数接收参数字典，返回单个数值（最小化或最大化）
4. 多目标优化返回 Pareto 前沿，单目标优化返回最佳解
5. 数据来自 Module 01，确保先准备好数据
6. 策略优化需要定义策略类，实现 `generate_signal` 方法
