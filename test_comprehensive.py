"""
综合测试脚本
测试FinLoom系统的所有核心功能
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_environment_modules():
    """测试环境模块"""
    print("=== 测试环境模块 ===")
    
    try:
        from module_00_environment.config_loader import ConfigLoader
        from module_00_environment.health_monitor import HealthMonitor
        
        # 测试配置加载器
        config_loader = ConfigLoader()
        system_config = config_loader.load_system_config()
        print(f"✓ 系统配置加载成功: {len(system_config)} 个配置项")
        
        # 测试健康监控器
        health_monitor = HealthMonitor()
        # 注意：check_all是异步方法，这里只测试创建
        print(f"✓ 健康监控器创建成功: 注册了 {len(health_monitor.components)} 个组件")
        
        return True
    except Exception as e:
        print(f"✗ 环境模块测试失败: {e}")
        return False

def test_data_pipeline():
    """测试数据管道模块"""
    print("\n=== 测试数据管道模块 ===")
    
    try:
        from module_01_data_pipeline.data_acquisition.akshare_collector import AkshareDataCollector
        from module_01_data_pipeline.storage_management.database_manager import DatabaseManager
        
        # 测试数据收集器
        collector = AkshareDataCollector()
        stock_list = collector.fetch_stock_list()
        print(f"✓ 数据收集器测试成功: 获取到 {len(stock_list)} 只股票")
        
        # 测试数据库管理器
        db_manager = DatabaseManager("data/test.db")
        
        # 创建测试数据
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        test_data = pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)) * 2,
            'high': 102 + np.random.randn(len(dates)) * 2,
            'low': 98 + np.random.randn(len(dates)) * 2,
            'close': 100 + np.random.randn(len(dates)) * 2,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # 测试数据存储
        success = db_manager.insert_market_data(test_data, "TEST001")
        print(f"✓ 数据库存储测试: {'成功' if success else '失败'}")
        
        # 测试数据检索
        retrieved_data = db_manager.get_market_data("TEST001")
        print(f"✓ 数据检索测试: 检索到 {len(retrieved_data)} 条记录")
        
        db_manager.close()
        return True
        
    except Exception as e:
        print(f"✗ 数据管道模块测试失败: {e}")
        return False

def test_feature_engineering():
    """测试特征工程模块"""
    print("\n=== 测试特征工程模块 ===")
    
    try:
        from module_02_feature_engineering.feature_extraction.technical_indicators import TechnicalIndicators
        
        # 创建测试数据
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        
        test_data = pd.DataFrame({
            'open': prices,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # 测试技术指标计算
        ti = TechnicalIndicators()
        indicators = ti.calculate_all_indicators(test_data)
        print(f"✓ 技术指标计算成功: 计算了 {len(indicators.columns)} 个指标")
        
        # 测试特定指标
        sma_5 = ti.calculate_sma(test_data['close'], 5)
        rsi = ti.calculate_rsi(test_data['close'], 14)
        print(f"✓ SMA和RSI计算成功: SMA长度={len(sma_5.dropna())}, RSI长度={len(rsi.dropna())}")
        
        return True
        
    except Exception as e:
        print(f"✗ 特征工程模块测试失败: {e}")
        return False

def test_risk_management():
    """测试风险管理模块"""
    print("\n=== 测试风险管理模块 ===")
    
    try:
        from module_05_risk_management.position_sizing.kelly_criterion import KellyCriterion
        
        # 创建测试收益率数据
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.02)  # 252个交易日
        
        # 测试凯利准则
        kelly = KellyCriterion()
        result = kelly.calculate_kelly_fraction(returns)
        print(f"✓ 凯利准则计算成功: 凯利分数={result.kelly_fraction:.3f}")
        print(f"✓ 建议仓位: {result.recommended_position:.1%}")
        
        return True
        
    except Exception as e:
        print(f"✗ 风险管理模块测试失败: {e}")
        return False

def test_execution_modules():
    """测试执行模块"""
    print("\n=== 测试执行模块 ===")
    
    try:
        from module_08_execution.signal_generator import SignalGenerator, generate_trading_signals
        from module_08_execution.order_manager import OrderManager, OrderStatus
        
        # 创建测试数据
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        
        test_data = pd.DataFrame({
            'open': prices,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # 测试信号生成
        signals = generate_trading_signals("TEST001", test_data, ["MA_CROSSOVER", "RSI"])
        print(f"✓ 信号生成成功: 生成了 {len(signals)} 个信号")
        
        # 测试订单管理
        order_manager = OrderManager()
        if signals:
            order = order_manager.create_order_from_signal(signals[0])
            print(f"✓ 订单创建成功: 订单ID={order.order_id}")
            
            # 测试订单提交
            success = order_manager.submit_order(order)
            print(f"✓ 订单提交: {'成功' if success else '失败'}")
            
            # 测试订单统计
            stats = order_manager.get_order_statistics()
            print(f"✓ 订单统计: 总订单数={stats['total_orders']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 执行模块测试失败: {e}")
        return False

def test_backtesting():
    """测试回测模块"""
    print("\n=== 测试回测模块 ===")
    
    try:
        from module_09_backtesting.backtest_engine import BacktestConfig, BacktestEngine
        
        # 创建回测配置
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 30),
            initial_capital=1000000
        )
        
        # 创建回测引擎
        engine = BacktestEngine(config)
        print("✓ 回测引擎创建成功")
        
        # 创建测试市场数据
        dates = pd.date_range(start=config.start_date, end=config.end_date, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        
        market_data = {
            "TEST001": pd.DataFrame({
                'open': prices,
                'high': prices + 2,
                'low': prices - 2,
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)
        }
        
        # 加载市场数据
        engine.load_market_data(["TEST001"], market_data)
        print("✓ 市场数据加载成功")
        
        # 设置简单策略
        def simple_strategy(current_data, positions, cash):
            return []  # 空策略
        
        engine.set_strategy(simple_strategy)
        
        # 运行回测
        result = engine.run()
        print(f"✓ 回测完成: 最终资金={result.final_capital:,.2f}")
        print(f"✓ 总收益率: {result.total_return:.2%}")
        
        return True
        
    except Exception as e:
        print(f"✗ 回测模块测试失败: {e}")
        return False

def test_ai_interaction():
    """测试AI交互模块"""
    print("\n=== 测试AI交互模块 ===")
    
    try:
        from module_10_ai_interaction.requirement_parser import parse_user_requirement
        from module_10_ai_interaction.fin_r1_integration import FINR1Integration
        
        # 测试需求解析
        user_input = "我想投资50万，希望获得稳健收益，可以接受中等风险，投资期限2年"
        parsed = parse_user_requirement(user_input)
        print(f"✓ 需求解析成功: 投资金额={parsed.investment_amount}")
        print(f"✓ 风险偏好: {parsed.risk_tolerance}")
        
        # 测试FIN-R1集成
        from module_00_environment.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        model_config = config_loader.load_model_config()
        fin_r1 = FINR1Integration(model_config)
        import asyncio
        response = asyncio.run(fin_r1.process_request(user_input))
        print(f"✓ FIN-R1处理成功: 响应长度={len(str(response))}")
        
        return True
        
    except Exception as e:
        print(f"✗ AI交互模块测试失败: {e}")
        return False

def test_market_analysis():
    """测试市场分析模块"""
    print("\n=== 测试市场分析模块 ===")
    
    try:
        from module_04_market_analysis.correlation_analysis.correlation_analyzer import analyze_market_correlation
        
        # 创建测试多股票数据
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        np.random.seed(42)
        
        price_data = pd.DataFrame({
            'STOCK1': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'STOCK2': 50 + np.cumsum(np.random.randn(len(dates)) * 0.3),
            'STOCK3': 200 + np.cumsum(np.random.randn(len(dates)) * 0.7)
        }, index=dates)
        
        # 测试相关性分析
        result = analyze_market_correlation(price_data)
        print(f"✓ 相关性分析成功: 平均相关性={result.average_correlation:.3f}")
        print(f"✓ 显著相关对: {len(result.significant_correlations)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 市场分析模块测试失败: {e}")
        return False

def test_monitoring():
    """测试监控模块"""
    print("\n=== 测试监控模块 ===")
    
    try:
        from module_06_monitoring_alerting.real_time_monitoring.performance_monitor import get_performance_monitor
        
        # 获取性能监控器
        monitor = get_performance_monitor(monitoring_interval=1)
        print("✓ 性能监控器创建成功")
        
        # 添加告警规则
        from module_06_monitoring_alerting.real_time_monitoring.performance_monitor import AlertRule
        cpu_rule = AlertRule(
            name="high_cpu",
            metric_type="system",
            metric_name="cpu_percent",
            operator=">",
            threshold=80.0
        )
        monitor.add_alert_rule(cpu_rule)
        print("✓ 告警规则添加成功")
        
        # 启动监控
        monitor.start_monitoring()
        print("✓ 监控启动成功")
        
        # 等待一段时间收集数据
        time.sleep(3)
        
        # 获取最新指标
        system_metrics = monitor.get_latest_system_metrics()
        if system_metrics:
            print(f"✓ 系统指标收集成功: CPU={system_metrics.cpu_percent:.1f}%, 内存={system_metrics.memory_percent:.1f}%")
        
        # 获取指标摘要
        summary = monitor.get_metrics_summary(minutes=1)
        print(f"✓ 指标摘要: 系统指标数={summary['system_metrics_count']}")
        
        # 停止监控
        monitor.stop_monitoring()
        print("✓ 监控停止成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 监控模块测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("FinLoom 系统综合测试")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("环境模块", test_environment_modules),
        ("数据管道", test_data_pipeline),
        ("特征工程", test_feature_engineering),
        ("风险管理", test_risk_management),
        ("执行模块", test_execution_modules),
        ("回测模块", test_backtesting),
        ("AI交互", test_ai_interaction),
        ("市场分析", test_market_analysis),
        ("监控模块", test_monitoring)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}测试出现异常: {e}")
            test_results.append((test_name, False))
    
    # 输出测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要:")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:12} : {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 个模块测试通过")
    
    if passed == total:
        print("🎉 所有模块测试通过！系统运行正常。")
        return True
    else:
        print(f"⚠️  有 {total - passed} 个模块测试失败，需要检查。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
