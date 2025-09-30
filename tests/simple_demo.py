"""
简单演示脚本
展示FinLoom系统的基本功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def demo_data_pipeline():
    """演示数据管道功能"""
    print("=== 数据管道演示 ===")
    
    # 导入数据收集器
    from module_01_data_pipeline.data_acquisition.akshare_collector import AkshareDataCollector
    from module_01_data_pipeline.storage_management.database_manager import DatabaseManager
    
    # 创建数据收集器
    collector = AkshareDataCollector()
    print("✓ 数据收集器创建成功")
    
    # 创建数据库管理器
    db_manager = DatabaseManager("data/demo.db")
    print("✓ 数据库管理器创建成功")
    
    # 获取股票列表
    try:
        stock_list = collector.fetch_stock_list()
        print(f"✓ 获取到 {len(stock_list)} 只股票")
    except Exception as e:
        print(f"✗ 获取股票列表失败: {e}")
    
    # 创建模拟数据
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    np.random.seed(42)
    
    mock_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + 2,
        'low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - 2,
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # 存储到数据库
    db_manager.insert_market_data(mock_data, "000001")
    print("✓ 模拟数据存储成功")
    
    db_manager.close()
    print("✓ 数据管道演示完成\n")

def demo_feature_engineering():
    """演示特征工程功能"""
    print("=== 特征工程演示 ===")
    
    from module_02_feature_engineering.feature_extraction.technical_indicators import TechnicalIndicators
    
    # 创建技术指标计算器
    ti = TechnicalIndicators()
    print("✓ 技术指标计算器创建成功")
    
    # 创建模拟价格数据
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    mock_data = pd.DataFrame({
        'open': prices,
        'high': prices + 2,
        'low': prices - 2,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # 计算技术指标
    try:
        indicators = ti.calculate_all_indicators(mock_data)
        print(f"✓ 计算了 {len(indicators.columns)} 个技术指标")
        print(f"  指标包括: {list(indicators.columns)[:5]}...")
    except Exception as e:
        print(f"✗ 技术指标计算失败: {e}")
    
    print("✓ 特征工程演示完成\n")

def demo_risk_management():
    """演示风险管理功能"""
    print("=== 风险管理演示 ===")
    
    from module_05_risk_management.position_sizing.kelly_criterion import KellyCriterion
    
    # 创建凯利准则计算器
    kelly = KellyCriterion()
    print("✓ 凯利准则计算器创建成功")
    
    # 创建模拟收益率数据
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.02)  # 252个交易日
    
    # 计算凯利分数
    try:
        result = kelly.calculate_kelly_fraction(returns)
        print(f"✓ 凯利分数: {result.kelly_fraction:.3f}")
        print(f"  建议仓位: {result.recommended_position:.1%}")
    except Exception as e:
        print(f"✗ 凯利计算失败: {e}")
    
    print("✓ 风险管理演示完成\n")

def demo_signal_generation():
    """演示信号生成功能"""
    print("=== 信号生成演示 ===")
    
    from module_08_execution.signal_generator import generate_trading_signals
    
    # 创建模拟价格数据
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    mock_data = pd.DataFrame({
        'open': prices,
        'high': prices + 2,
        'low': prices - 2,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # 生成交易信号
    try:
        signals = generate_trading_signals("000001", mock_data, ["MA_CROSSOVER", "RSI"])
        print(f"✓ 生成了 {len(signals)} 个交易信号")
        
        if signals:
            signal = signals[0]
            print(f"  示例信号: {signal.action} {signal.symbol} @ {signal.price:.2f}")
            print(f"  置信度: {signal.confidence:.3f}")
    except Exception as e:
        print(f"✗ 信号生成失败: {e}")
    
    print("✓ 信号生成演示完成\n")

def demo_backtesting():
    """演示回测功能"""
    print("=== 回测演示 ===")
    
    from module_09_backtesting.backtest_engine import BacktestConfig, BacktestEngine
    
    # 创建回测配置
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        initial_capital=1000000
    )
    print("✓ 回测配置创建成功")
    
    # 创建回测引擎
    engine = BacktestEngine(config)
    print("✓ 回测引擎创建成功")
    
    # 创建模拟市场数据
    dates = pd.date_range(start=config.start_date, end=config.end_date, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    market_data = {
        "000001": pd.DataFrame({
            'open': prices,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
    }
    
    # 加载市场数据
    engine.load_market_data(["000001"], market_data)
    print("✓ 市场数据加载成功")
    
    # 设置简单策略
    def simple_strategy(current_data, positions, cash):
        return []  # 空策略，不产生信号
    
    engine.set_strategy(simple_strategy)
    print("✓ 策略设置成功")
    
    # 运行回测
    try:
        result = engine.run()
        print(f"✓ 回测完成")
        print(f"  初始资金: {result.initial_capital:,.2f}")
        print(f"  最终资金: {result.final_capital:,.2f}")
        print(f"  总收益率: {result.total_return:.2%}")
    except Exception as e:
        print(f"✗ 回测失败: {e}")
    
    print("✓ 回测演示完成\n")

def demo_ai_interaction():
    """演示AI交互功能"""
    print("=== AI交互演示 ===")
    
    from module_10_ai_interaction.requirement_parser import parse_user_requirement
    
    # 解析用户需求
    user_input = "我想投资100万，希望获得稳健的收益，可以接受一定的波动，投资期限3年"
    
    try:
        parsed = parse_user_requirement(user_input)
        print("✓ 用户需求解析成功")
        print(f"  投资金额: {parsed.investment_amount}")
        print(f"  风险偏好: {parsed.risk_tolerance}")
        print(f"  投资期限: {parsed.investment_horizon}")
    except Exception as e:
        print(f"✗ 需求解析失败: {e}")
    
    print("✓ AI交互演示完成\n")

def demo_correlation_analysis():
    """演示相关性分析功能"""
    print("=== 相关性分析演示 ===")
    
    from module_04_market_analysis.correlation_analysis.correlation_analyzer import analyze_market_correlation
    
    # 创建模拟多股票价格数据
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    np.random.seed(42)
    
    price_data = pd.DataFrame({
        '000001': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        '000002': 50 + np.cumsum(np.random.randn(len(dates)) * 0.3),
        '000003': 200 + np.cumsum(np.random.randn(len(dates)) * 0.7)
    }, index=dates)
    
    # 分析相关性
    try:
        result = analyze_market_correlation(price_data)
        print("✓ 相关性分析完成")
        print(f"  平均相关性: {result.average_correlation:.3f}")
        print(f"  显著相关对: {len(result.significant_correlations)}")
    except Exception as e:
        print(f"✗ 相关性分析失败: {e}")
    
    print("✓ 相关性分析演示完成\n")

def main():
    """主函数"""
    print("FinLoom 系统功能演示")
    print("=" * 50)
    
    try:
        demo_data_pipeline()
        demo_feature_engineering()
        demo_risk_management()
        demo_signal_generation()
        demo_backtesting()
        demo_ai_interaction()
        demo_correlation_analysis()
        
        print("=" * 50)
        print("🎉 所有演示完成！系统运行正常。")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")

if __name__ == "__main__":
    main()
