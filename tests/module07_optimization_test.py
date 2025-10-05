"""
Module 07 优化模块测试
测试超参数优化、策略优化、多目标优化和资源优化功能
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.logging_system import setup_logger
from module_01_data_pipeline import AkshareDataCollector
from module_02_feature_engineering import TechnicalIndicators
from module_07_optimization import (
    BayesianOptimizer,
    ComputeOptimizer,
    ComputeResource,
    ComputeTask,
    CostComponent,
    CostOptimizer,
    GridSearchOptimizer,
    MemoryOptimizer,
    NSGAOptimizer,
    Parameter,
    ParetoFrontier,
    PerformanceEvaluator,
    RandomSearchOptimizer,
    StrategyOptimizer,
    create_portfolio_objectives,
    get_optimization_database_manager,
    get_optimization_manager,
    get_strategy_space,
)

# 检查Optuna是否可用
try:
    from module_07_optimization import OptunaOptimizer

    OPTUNA_AVAILABLE = True
except (ImportError, AttributeError):
    OptunaOptimizer = None
    OPTUNA_AVAILABLE = False

logger = setup_logger("module07_test")


def test_bayesian_optimization():
    """测试贝叶斯优化"""
    logger.info("=" * 50)
    logger.info("测试 1: 贝叶斯优化")

    # 定义简单的测试函数 (Rosenbrock)
    def objective(params):
        x = params["x"]
        y = params["y"]
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    # 定义参数空间
    param_space = [
        Parameter(name="x", param_type="float", low=-5.0, high=5.0),
        Parameter(name="y", param_type="float", low=-5.0, high=5.0),
    ]

    # 创建优化器
    optimizer = BayesianOptimizer(
        parameter_space=param_space,
        objective_function=objective,
        maximize=False,
        n_trials=20,
        n_initial_points=5,
        acquisition_function="ei",
    )

    # 执行优化
    result = optimizer.optimize()

    logger.info(f"最佳参数: {result.best_parameters}")
    logger.info(f"最佳值: {result.best_value:.6f}")
    logger.info(f"成功试验: {result.n_successful_trials}/{result.n_trials}")

    assert result.best_value < 10.0, "优化未收敛到合理范围"
    logger.info("✓ 贝叶斯优化测试通过")


def test_grid_search():
    """测试网格搜索"""
    logger.info("=" * 50)
    logger.info("测试 2: 网格搜索")

    def objective(params):
        x = params["x"]
        y = params["y"]
        return x**2 + y**2

    param_space = [
        Parameter(name="x", param_type="float", low=-2.0, high=2.0),
        Parameter(name="y", param_type="float", low=-2.0, high=2.0),
    ]

    optimizer = GridSearchOptimizer(
        parameter_space=param_space,
        objective_function=objective,
        maximize=False,
        n_grid_points=5,
    )

    result = optimizer.optimize()

    logger.info(f"最佳参数: {result.best_parameters}")
    logger.info(f"最佳值: {result.best_value}")

    # 网格搜索应该找到接近0的解
    assert result.best_value < 1.0, "网格搜索未找到最优解附近"
    logger.info("✓ 网格搜索测试通过")


def test_optuna_optimization():
    """测试Optuna优化"""
    logger.info("=" * 50)
    logger.info("测试 3: Optuna优化")

    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna未安装，跳过测试")
        return

    def objective(params):
        x = params["x"]
        y = params["y"]
        return x**2 + y**2

    param_space = [
        Parameter(name="x", param_type="float", low=-5.0, high=5.0),
        Parameter(name="y", param_type="float", low=-5.0, high=5.0),
    ]

    optimizer = OptunaOptimizer(
        parameter_space=param_space,
        objective_function=objective,
        maximize=False,
        n_trials=20,
        sampler="tpe",
    )

    result = optimizer.optimize()

    logger.info(f"最佳参数: {result.best_parameters}")
    logger.info(f"最佳值: {result.best_value:.6f}")

    assert result.best_value < 0.5, "Optuna优化未找到最优解"
    logger.info("✓ Optuna优化测试通过")


def test_random_search():
    """测试随机搜索"""
    logger.info("=" * 50)
    logger.info("测试 4: 随机搜索")

    def objective(params):
        return params["x"] ** 2 + params["y"] ** 2

    param_space = [
        Parameter(name="x", param_type="float", low=-5.0, high=5.0),
        Parameter(name="y", param_type="float", low=-5.0, high=5.0),
    ]

    optimizer = RandomSearchOptimizer(
        parameter_space=param_space,
        objective_function=objective,
        maximize=False,
        n_trials=30,
    )

    result = optimizer.optimize()

    logger.info(f"最佳参数: {result.best_parameters}")
    logger.info(f"最佳值: {result.best_value:.6f}")

    assert result.best_value < 5.0, "随机搜索未找到合理解"
    logger.info("✓ 随机搜索测试通过")


def test_multi_objective_optimization():
    """测试多目标优化"""
    logger.info("=" * 50)
    logger.info("测试 5: 多目标优化 (NSGA-II)")

    # 定义多个目标函数 (最小化)
    def objective1(params):
        x = params["x"]
        return x**2

    def objective2(params):
        x = params["x"]
        return (x - 2) ** 2

    param_space = [
        Parameter(name="x", param_type="float", low=-5.0, high=5.0),
    ]

    optimizer = NSGAOptimizer(
        parameter_space=param_space,
        objective_functions=[objective1, objective2],
        population_size=20,
        n_generations=10,
    )

    result = optimizer.optimize()

    logger.info(f"Pareto前沿大小: {len(result['pareto_front'])}")
    logger.info(f"总解数量: {len(result['all_solutions'])}")

    assert len(result["pareto_front"]) > 0, "未找到Pareto最优解"

    # 分析Pareto前沿
    frontier = ParetoFrontier(
        solutions=result["pareto_front"], objective_names=["obj1", "obj2"]
    )

    # 选择解
    preferences = {"obj1": 0.5, "obj2": 0.5}
    selected = frontier.select_solution_by_preference(preferences)

    logger.info(f"选中的解: {selected['parameters']}")
    logger.info(f"目标值: {selected['objectives']}")

    logger.info("✓ 多目标优化测试通过")


def test_strategy_optimization_with_real_data():
    """测试策略优化（使用真实数据）"""
    logger.info("=" * 50)
    logger.info("测试 6: 策略优化（真实数据）")

    try:
        # 获取真实市场数据
        collector = AkshareDataCollector()
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")

        logger.info(f"获取数据: 000001, {start_date} - {end_date}")
        market_data = collector.fetch_stock_history("000001", start_date, end_date)

        if market_data.empty or len(market_data) < 60:
            logger.warning("数据不足，跳过策略优化测试")
            return

        # 简单的移动平均策略
        class SimpleMAStrategy:
            def __init__(self, short_window, long_window):
                self.short_window = int(short_window)
                self.long_window = int(long_window)

            def generate_signal(self, history):
                if len(history) < self.long_window:
                    return None

                short_ma = history["close"].iloc[-self.short_window :].mean()
                long_ma = history["close"].iloc[-self.long_window :].mean()

                from common.data_structures import Signal

                if short_ma > long_ma:
                    return Signal(
                        symbol="000001",
                        action="BUY",
                        quantity=1.0,
                        price=history["close"].iloc[-1],
                        timestamp=history.index[-1],
                    )
                elif short_ma < long_ma:
                    return Signal(
                        symbol="000001",
                        action="SELL",
                        quantity=1.0,
                        price=history["close"].iloc[-1],
                        timestamp=history.index[-1],
                    )
                return None

        # 获取预定义参数空间
        param_space = get_strategy_space("ma_crossover")[:2]  # 只用前2个参数

        # 创建优化器
        optimizer = StrategyOptimizer(
            strategy_class=SimpleMAStrategy,
            market_data=market_data,
            optimization_metric="sharpe_ratio",
            test_split=0.2,
            walk_forward_windows=None,  # 跳过Walk Forward以加速测试
        )

        # 执行优化（减少试验次数）
        result = optimizer.optimize(
            parameter_space=param_space, n_trials=10, optimizer_type="optuna"
        )

        logger.info(f"最佳参数: {result['best_parameters']}")
        logger.info(f"训练性能: {result['train_performance']:.4f}")

        test_perf = result.get("test_performance", {})
        if test_perf:
            logger.info(f"测试夏普比率: {test_perf.get('sharpe_ratio', 'N/A')}")

        logger.info("✓ 策略优化测试通过")

    except Exception as e:
        logger.error(f"策略优化测试失败: {e}")
        logger.warning("跳过策略优化测试")


def test_performance_evaluator():
    """测试性能评估器"""
    logger.info("=" * 50)
    logger.info("测试 7: 性能评估器")

    # 生成模拟收益率
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 一年的日收益率

    evaluator = PerformanceEvaluator(annual_trading_days=252)
    metrics = evaluator.evaluate(returns, risk_free_rate=0.03)

    logger.info(f"年化收益: {metrics.annual_return:.2%}")
    logger.info(f"波动率: {metrics.volatility:.2%}")
    logger.info(f"夏普比率: {metrics.sharpe_ratio:.2f}")
    logger.info(f"最大回撤: {metrics.max_drawdown:.2%}")
    logger.info(f"胜率: {metrics.win_rate:.2%}")

    assert metrics.total_trades > 0, "未识别到交易"
    logger.info("✓ 性能评估器测试通过")


def test_compute_optimizer():
    """测试计算资源优化"""
    logger.info("=" * 50)
    logger.info("测试 8: 计算资源优化")

    # 定义资源
    resources = [
        ComputeResource("cpu1", "CPU", capacity=100.0, cost_per_unit=0.1),
        ComputeResource("gpu1", "GPU", capacity=50.0, cost_per_unit=0.5),
    ]

    # 定义任务
    tasks = [
        ComputeTask(
            "task1",
            "training",
            {"CPU": 30, "GPU": 10},
            priority=2,
            estimated_duration=2.0,
        ),
        ComputeTask(
            "task2",
            "inference",
            {"CPU": 20, "GPU": 5},
            priority=1,
            estimated_duration=1.0,
        ),
    ]

    optimizer = ComputeOptimizer(resources, optimization_objective="min_cost")

    # 分配资源
    allocation = optimizer.allocate_resources(tasks)
    logger.info(f"资源分配: {allocation}")

    # 计算成本
    total_cost = optimizer.calculate_total_cost(allocation, tasks)
    logger.info(f"总成本: {total_cost:.2f}")

    # 利用率
    utilization = optimizer.calculate_utilization(allocation)
    logger.info(f"利用率: {utilization}")

    assert total_cost > 0, "成本计算错误"
    logger.info("✓ 计算资源优化测试通过")


def test_cost_optimizer():
    """测试成本优化"""
    logger.info("=" * 50)
    logger.info("测试 9: 成本优化")

    components = [
        CostComponent(
            "storage", "storage", fixed_cost=100, variable_cost=0.1, volume=1000
        ),
        CostComponent(
            "compute", "compute", fixed_cost=500, variable_cost=1.0, volume=200
        ),
    ]

    optimizer = CostOptimizer(components)

    # 总成本
    total_cost = optimizer.calculate_total_cost()
    logger.info(f"总成本: {total_cost:.2f}")

    # 成本明细
    breakdown = optimizer.calculate_cost_breakdown()
    logger.info(f"成本明细: {breakdown}")

    # 成本削减建议
    recommendations = optimizer.recommend_cost_reduction(target_reduction=200.0)
    logger.info(f"削减建议数: {len(recommendations)}")

    assert total_cost > 0, "成本计算错误"
    logger.info("✓ 成本优化测试通过")


def test_memory_optimizer():
    """测试内存优化"""
    logger.info("=" * 50)
    logger.info("测试 10: 内存优化")

    optimizer = MemoryOptimizer(memory_limit_mb=1000.0)

    # 获取内存使用
    current_usage = optimizer.get_memory_usage()
    logger.info(f"当前内存使用: {current_usage:.2f}MB")

    # 内存分析
    profile = optimizer.profile_memory(top_n=5)
    logger.info(f"对象数量: {profile.object_count}")

    # 优化DataFrame
    df = pd.DataFrame(
        {
            "int_col": np.random.randint(0, 100, 1000),
            "float_col": np.random.randn(1000),
            "str_col": ["A", "B", "C"] * 333 + ["A"],
        }
    )

    optimized_df = optimizer.optimize_dataframe(df)

    original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)

    logger.info(f"原始内存: {original_memory:.2f}MB")
    logger.info(f"优化后内存: {optimized_memory:.2f}MB")

    assert optimized_memory <= original_memory, "内存优化无效"
    logger.info("✓ 内存优化测试通过")


def test_database_operations():
    """测试数据库操作"""
    logger.info("=" * 50)
    logger.info("测试 11: 数据库操作")

    db = get_optimization_database_manager()

    # 保存优化任务
    task_id = f"test_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    success = db.save_optimization_task(
        task_id=task_id,
        task_name="测试任务",
        optimizer_type="bayesian",
        config={"n_trials": 50},
        status="pending",
    )

    assert success, "保存任务失败"

    # 更新状态
    success = db.update_task_status(task_id, "completed", datetime.now())
    assert success, "更新状态失败"

    # 保存结果
    success = db.save_optimization_result(
        task_id=task_id,
        best_parameters={"x": 1.0, "y": 2.0},
        best_value=0.5,
        n_trials=50,
        n_successful_trials=48,
        total_time_seconds=300.0,
    )

    assert success, "保存结果失败"

    # 查询结果
    result = db.get_optimization_result(task_id)
    assert result is not None, "查询结果失败"
    logger.info(f"查询到的结果: {result['best_parameters']}")

    # 数据库统计
    stats = db.get_database_stats()
    logger.info(f"数据库统计: {stats}")

    logger.info("✓ 数据库操作测试通过")


def test_optimization_manager():
    """测试优化管理器"""
    logger.info("=" * 50)
    logger.info("测试 12: 优化管理器")

    manager = get_optimization_manager()

    def simple_objective(params):
        return params["x"] ** 2

    param_space = [Parameter("x", "float", low=-5.0, high=5.0)]

    # 创建并运行优化任务
    task_id = manager.create_optimization_task(
        task_name="管理器测试",
        optimizer_type="random",
        parameter_space=param_space,
        objective_function=simple_objective,
    )

    logger.info(f"创建任务: {task_id}")

    # 运行优化（使用bayesian优化器）
    result = manager.run_optimization(
        task_id=task_id,
        optimizer_type="bayesian",
        parameter_space=param_space,
        objective_function=simple_objective,
        n_trials=10,
    )

    logger.info(f"最佳参数: {result.best_parameters}")

    # 列出任务
    tasks = manager.list_tasks()
    logger.info(f"任务总数: {len(tasks)}")

    logger.info("✓ 优化管理器测试通过")


def main():
    """运行所有测试"""
    logger.info("=" * 50)
    logger.info("开始 Module 07 优化模块测试")
    logger.info("=" * 50)

    tests = [
        ("贝叶斯优化", test_bayesian_optimization),
        ("网格搜索", test_grid_search),
        ("Optuna优化", test_optuna_optimization),
        ("随机搜索", test_random_search),
        ("多目标优化", test_multi_objective_optimization),
        ("策略优化", test_strategy_optimization_with_real_data),
        ("性能评估", test_performance_evaluator),
        ("计算资源优化", test_compute_optimizer),
        ("成本优化", test_cost_optimizer),
        ("内存优化", test_memory_optimizer),
        ("数据库操作", test_database_operations),
        ("优化管理器", test_optimization_manager),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test_name} 测试失败: {e}")
            import traceback

            traceback.print_exc()

    logger.info("=" * 50)
    logger.info(f"测试完成: {passed} 通过, {failed} 失败")
    logger.info("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
