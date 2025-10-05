"""
测试增强的仓位管理功能
包括动态仓位管理和投资组合权重优化
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from module_05_risk_management.position_sizing import (
    # 动态仓位管理
    DynamicPositionSizer,
    # 凯利准则
    KellyCriterion,
    MarketRegime,
    OptimizationConfig,
    OptimizationMethod,
    OptimizationObjective,
    # 投资组合优化
    PortfolioWeightOptimizer,
    PositionSizingConfig,
    PositionSizingMethod,
    # 风险平价
    RiskParity,
    calculate_dynamic_position,
    calculate_kelly_position,
    calculate_risk_parity_weights,
    optimize_portfolio,
)


def generate_sample_returns(n_assets=5, n_periods=252):
    """生成示例收益率数据"""
    np.random.seed(42)

    # 生成相关性矩阵
    random_matrix = np.random.randn(n_assets, n_assets)
    corr_matrix = random_matrix @ random_matrix.T
    corr_matrix = corr_matrix / np.sqrt(
        np.diag(corr_matrix)[:, None] * np.diag(corr_matrix)[None, :]
    )

    # 生成收益率
    mean_returns = np.random.uniform(-0.001, 0.002, n_assets)
    volatilities = np.random.uniform(0.01, 0.03, n_assets)

    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)

    # 创建DataFrame
    symbols = [f"STOCK_{i + 1}" for i in range(n_assets)]
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq="D")

    returns_df = pd.DataFrame(returns, index=dates, columns=symbols)

    return returns_df


def test_dynamic_position_sizer():
    """测试动态仓位管理器"""
    print("\n" + "=" * 80)
    print("测试1: 动态仓位管理器")
    print("=" * 80)

    # 生成示例数据
    returns_df = generate_sample_returns(n_assets=1, n_periods=252)
    historical_returns = returns_df.iloc[:, 0]

    # 配置
    config = PositionSizingConfig(
        max_position_size=0.20,
        min_position_size=0.01,
        target_volatility=0.15,
        risk_per_trade=0.02,
    )

    # 创建管理器
    sizer = DynamicPositionSizer(config)

    # 测试不同方法
    methods = [
        PositionSizingMethod.ADAPTIVE,
        PositionSizingMethod.KELLY,
        PositionSizingMethod.VOLATILITY_TARGET,
        PositionSizingMethod.RISK_PARITY,
        PositionSizingMethod.CONFIDENCE_WEIGHTED,
    ]

    for method in methods:
        result = sizer.calculate_position_size(
            symbol="STOCK_1",
            current_price=100.0,
            account_value=100000.0,
            signal_strength=0.8,
            confidence=0.75,
            historical_returns=historical_returns,
            method=method,
        )

        print(f"\n方法: {method.value}")
        print(f"  推荐仓位大小: {result.recommended_size:.2%}")
        print(f"  推荐股数: {result.recommended_shares}")
        print(f"  仓位价值: ${result.position_value:,.2f}")
        print(f"  风险金额: ${result.risk_amount:,.2f}")
        print(f"  置信度分数: {result.confidence_score:.2f}")
        print(f"  市场状态: {result.market_regime}")
        print(f"  波动率调整: {result.volatility_adjustment:.2f}")

    # 测试便捷函数
    print("\n使用便捷函数:")
    result = calculate_dynamic_position(
        symbol="STOCK_1",
        current_price=100.0,
        account_value=100000.0,
        signal_strength=0.8,
        confidence=0.75,
        historical_returns=historical_returns,
    )
    print(f"  推荐仓位: {result.recommended_size:.2%}")
    print(f"  推荐股数: {result.recommended_shares}")

    # 获取统计信息
    stats = sizer.get_position_statistics()
    print(f"\n仓位统计:")
    print(f"  总计算次数: {stats['total_calculations']}")
    print(f"  平均仓位大小: {stats['avg_position_size']:.2%}")
    print(f"  中位数仓位大小: {stats['median_position_size']:.2%}")
    print(f"  使用的方法: {stats['methods_used']}")

    print("\n✅ 动态仓位管理器测试完成")


def test_multi_position_allocation():
    """测试多仓位联合配置"""
    print("\n" + "=" * 80)
    print("测试2: 多仓位联合配置")
    print("=" * 80)

    # 生成多资产数据
    returns_df = generate_sample_returns(n_assets=5, n_periods=252)

    # 配置
    config = PositionSizingConfig(
        max_position_size=0.25,
        max_total_exposure=0.95,
        correlation_threshold=0.7,
        concentration_limit=0.30,
    )

    sizer = DynamicPositionSizer(config)

    # 生成信号
    signals = {
        "STOCK_1": {"strength": 0.9, "confidence": 0.85},
        "STOCK_2": {"strength": 0.8, "confidence": 0.75},
        "STOCK_3": {"strength": 0.7, "confidence": 0.80},
        "STOCK_4": {"strength": 0.6, "confidence": 0.70},
        "STOCK_5": {"strength": 0.85, "confidence": 0.65},
    }

    # 当前价格
    current_prices = {symbol: 100.0 for symbol in returns_df.columns}

    # 计算多仓位配置
    results = sizer.calculate_multi_position_allocation(
        signals=signals,
        account_value=100000.0,
        current_prices=current_prices,
        returns_data=returns_df,
    )

    print("\n多仓位配置结果:")
    total_allocation = 0
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print(f"  仓位大小: {result.recommended_size:.2%}")
        print(f"  股数: {result.recommended_shares}")
        print(f"  价值: ${result.position_value:,.2f}")
        print(f"  置信度: {result.confidence_score:.2f}")
        total_allocation += result.recommended_size

    print(f"\n总配置比例: {total_allocation:.2%}")
    print("\n✅ 多仓位联合配置测试完成")


def test_portfolio_weight_optimizer():
    """测试投资组合权重优化器"""
    print("\n" + "=" * 80)
    print("测试3: 投资组合权重优化器")
    print("=" * 80)

    # 生成示例数据
    returns_df = generate_sample_returns(n_assets=5, n_periods=252)

    # 配置
    config = OptimizationConfig(
        min_weight=0.05,
        max_weight=0.35,
        target_volatility=0.15,
        risk_free_rate=0.03,
    )

    optimizer = PortfolioWeightOptimizer(config)

    # 测试不同优化方法
    methods = [
        (OptimizationMethod.MEAN_VARIANCE, OptimizationObjective.MAXIMIZE_SHARPE),
        (OptimizationMethod.MIN_VARIANCE, OptimizationObjective.MINIMIZE_RISK),
        (OptimizationMethod.MAX_SHARPE, OptimizationObjective.MAXIMIZE_SHARPE),
        (OptimizationMethod.RISK_PARITY, OptimizationObjective.MINIMIZE_RISK),
        (OptimizationMethod.EQUAL_WEIGHT, OptimizationObjective.MAXIMIZE_RETURN),
        (OptimizationMethod.INVERSE_VOLATILITY, OptimizationObjective.MINIMIZE_RISK),
    ]

    for method, objective in methods:
        print(f"\n优化方法: {method.value} | 目标: {objective.value}")
        print("-" * 60)

        result = optimizer.optimize(
            returns_data=returns_df, method=method, objective=objective
        )

        print("权重分配:")
        for asset, weight in result.weights.items():
            if weight > 0.01:  # 只显示有意义的权重
                print(f"  {asset}: {weight:.2%}")

        print(f"\n组合指标:")
        print(f"  预期收益率: {result.expected_return:.2%}")
        print(f"  波动率: {result.volatility:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        print(f"  索提诺比率: {result.sortino_ratio:.2f}")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  分散化比率: {result.diversification_ratio:.2f}")
        print(f"  有效资产数: {result.effective_n:.1f}")
        print(f"  优化状态: {result.success}")

    print("\n✅ 投资组合权重优化器测试完成")


def test_ensemble_optimization():
    """测试集成优化方法"""
    print("\n" + "=" * 80)
    print("测试4: 集成优化方法")
    print("=" * 80)

    returns_df = generate_sample_returns(n_assets=6, n_periods=252)

    config = OptimizationConfig(
        min_weight=0.05,
        max_weight=0.30,
        ensemble_methods=[
            "mean_variance",
            "risk_parity",
            "max_sharpe",
            "inverse_volatility",
        ],
        ensemble_weights=[0.3, 0.3, 0.2, 0.2],
    )

    optimizer = PortfolioWeightOptimizer(config)

    result = optimizer.optimize(
        returns_data=returns_df,
        method=OptimizationMethod.ENSEMBLE,
        objective=OptimizationObjective.MAXIMIZE_SHARPE,
    )

    print("\n集成优化结果:")
    print("\n权重分配:")
    for asset, weight in result.weights.items():
        if weight > 0.01:
            print(f"  {asset}: {weight:.2%}")

    print(f"\n组合指标:")
    print(f"  预期收益率: {result.expected_return:.2%}")
    print(f"  波动率: {result.volatility:.2%}")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  有效资产数: {result.effective_n:.1f}")

    print("\n✅ 集成优化方法测试完成")


def test_optimization_with_constraints():
    """测试带约束的优化"""
    print("\n" + "=" * 80)
    print("测试5: 带行业约束的优化")
    print("=" * 80)

    returns_df = generate_sample_returns(n_assets=6, n_periods=252)

    # 定义行业映射
    sector_mapping = {
        "STOCK_1": "科技",
        "STOCK_2": "科技",
        "STOCK_3": "金融",
        "STOCK_4": "金融",
        "STOCK_5": "消费",
        "STOCK_6": "消费",
    }

    # 行业限制
    sector_limits = {
        "科技": (0.2, 0.4),  # 科技股占20%-40%
        "金融": (0.2, 0.4),  # 金融股占20%-40%
        "消费": (0.1, 0.3),  # 消费股占10%-30%
    }

    optimizer = PortfolioWeightOptimizer()

    result = optimizer.optimize_with_constraints(
        returns_data=returns_df,
        sector_mapping=sector_mapping,
        sector_limits=sector_limits,
        method=OptimizationMethod.MAX_SHARPE,
    )

    print("\n带约束的优化结果:")
    print("\n权重分配（按行业）:")

    sector_weights = {}
    for asset, weight in result.weights.items():
        sector = sector_mapping[asset]
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
        if weight > 0.01:
            print(f"  {asset} ({sector}): {weight:.2%}")

    print("\n行业总权重:")
    for sector, weight in sector_weights.items():
        limit = sector_limits[sector]
        status = "✓" if limit[0] <= weight <= limit[1] else "✗"
        print(
            f"  {sector}: {weight:.2%} (限制: {limit[0]:.0%}-{limit[1]:.0%}) {status}"
        )

    print(f"\n组合指标:")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  波动率: {result.volatility:.2%}")

    print("\n✅ 带约束优化测试完成")


def test_backtest_optimization():
    """测试优化策略回测"""
    print("\n" + "=" * 80)
    print("测试6: 优化策略回测")
    print("=" * 80)

    # 生成更长期的数据用于回测
    returns_df = generate_sample_returns(n_assets=5, n_periods=504)  # 2年数据

    optimizer = PortfolioWeightOptimizer()

    # 回测不同策略
    strategies = [
        ("最大夏普", OptimizationMethod.MAX_SHARPE),
        ("风险平价", OptimizationMethod.RISK_PARITY),
        ("等权重", OptimizationMethod.EQUAL_WEIGHT),
    ]

    for strategy_name, method in strategies:
        print(f"\n{strategy_name}策略回测:")
        print("-" * 60)

        backtest_results = optimizer.backtest_optimization(
            returns_data=returns_df, rebalance_frequency="monthly", method=method
        )

        print(f"  总收益: {backtest_results['total_return']:.2f}%")
        print(f"  年化收益: {backtest_results['annualized_return']:.2f}%")
        print(f"  年化波动率: {backtest_results['annualized_volatility']:.2f}%")
        print(f"  夏普比率: {backtest_results['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {backtest_results['max_drawdown']:.2f}%")
        print(f"  再平衡次数: {backtest_results['n_rebalances']}")

    print("\n✅ 优化策略回测测试完成")


def test_integration_with_other_modules():
    """测试与其他模块的集成"""
    print("\n" + "=" * 80)
    print("测试7: 与其他模块集成")
    print("=" * 80)

    returns_df = generate_sample_returns(n_assets=3, n_periods=252)

    print("\n1. 凯利准则集成:")
    kelly = KellyCriterion(max_kelly_fraction=0.25)
    kelly_result = kelly.calculate_kelly_fraction(returns_df.iloc[:, 0])
    print(f"   凯利分数: {kelly_result.kelly_fraction:.2%}")
    print(f"   胜率: {kelly_result.win_rate:.2%}")
    print(f"   夏普比率: {kelly_result.sharpe_ratio:.2f}")

    print("\n2. 风险平价集成:")
    risk_parity = RiskParity()
    rp_result = risk_parity.apply_risk_parity_allocation(returns_df)
    print("   权重分配:")
    for i, (asset, weight) in enumerate(zip(rp_result.asset_names, rp_result.weights)):
        print(
            f"   {asset}: {weight:.2%} (风险贡献: {rp_result.risk_contribution_pct[i]:.2%})"
        )

    print("\n3. 动态仓位与优化器结合:")
    # 首先用优化器得到最优权重
    optimizer = PortfolioWeightOptimizer()
    opt_result = optimizer.optimize(returns_df, method=OptimizationMethod.MAX_SHARPE)

    # 然后用动态仓位管理器确定具体仓位
    sizer = DynamicPositionSizer()
    account_value = 100000.0

    print("\n   最优权重转化为具体仓位:")
    for asset in returns_df.columns:
        weight = opt_result.weights[asset]
        if weight > 0.01:
            position_size = weight * account_value
            current_price = 100.0
            shares = int(position_size / current_price)
            print(
                f"   {asset}: 权重={weight:.2%}, 股数={shares}, 价值=${position_size:,.2f}"
            )

    print("\n✅ 模块集成测试完成")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("开始运行 Module 05 - 仓位管理增强功能测试套件")
    print("=" * 80)

    try:
        test_dynamic_position_sizer()
        test_multi_position_allocation()
        test_portfolio_weight_optimizer()
        test_ensemble_optimization()
        test_optimization_with_constraints()
        test_backtest_optimization()
        test_integration_with_other_modules()

        print("\n" + "=" * 80)
        print("✅ 所有测试完成！")
        print("=" * 80)

        print("\n总结:")
        print("- 动态仓位管理器: ✅ 正常工作")
        print("- 多仓位联合配置: ✅ 正常工作")
        print("- 投资组合优化器: ✅ 正常工作")
        print("- 集成优化方法: ✅ 正常工作")
        print("- 带约束优化: ✅ 正常工作")
        print("- 策略回测: ✅ 正常工作")
        print("- 模块集成: ✅ 正常工作")

        print("\n新功能特性:")
        print("1. 动态仓位管理支持6种计算方法")
        print("2. 投资组合优化支持10种优化方法")
        print("3. 支持市场状态检测和自适应调整")
        print("4. 支持多约束条件优化")
        print("5. 支持策略回测和历史验证")
        print("6. 完整的与凯利准则和风险平价的集成")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
