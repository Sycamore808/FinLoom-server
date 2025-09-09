"""
系统测试脚本
"""

import sys
import os
import traceback

def test_imports():
    """测试所有模块导入"""
    print("Testing module imports...")
    
    try:
        # 测试通用模块
        from common.logging_system import setup_logger
        from common.data_structures import MarketData, Signal, Position
        from common.exceptions import QuantSystemError, DataError, ModelError, ExecutionError
        print("✓ Common modules imported successfully")
        
        # 测试环境模块
        from module_00_environment.config_loader import ConfigLoader
        from module_00_environment.dependency_installer import auto_install_dependencies
        from module_00_environment.env_checker import run_environment_check
        print("✓ Environment modules imported successfully")
        
        # 测试数据管道模块
        from module_01_data_pipeline.data_acquisition.akshare_collector import AkshareDataCollector
        print("✓ Data pipeline modules imported successfully")
        
        # 测试特征工程模块
        from module_02_feature_engineering.feature_extraction.technical_indicators import TechnicalIndicators
        print("✓ Feature engineering modules imported successfully")
        
        # 测试风险管理模块
        from module_05_risk_management.position_sizing.kelly_criterion import KellyCriterion
        print("✓ Risk management modules imported successfully")
        
        # 测试回测模块
        from module_09_backtesting.backtest_engine import BacktestEngine, BacktestConfig
        print("✓ Backtesting modules imported successfully")
        
        # 测试AI交互模块
        from module_10_ai_interaction.fin_r1_integration import FINR1Integration
        from module_10_ai_interaction.requirement_parser import RequirementParser
        print("✓ AI interaction modules imported successfully")
        
        print("\n🎉 All modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\nTesting basic functionality...")
    
    try:
        # 测试日志系统
        from common.logging_system import setup_logger
        logger = setup_logger("test")
        logger.info("Logging system test successful")
        print("✓ Logging system working")
        
        # 测试配置加载
        from module_00_environment.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        print("✓ Config loader created successfully")
        
        # 测试数据收集器
        from module_01_data_pipeline.data_acquisition.akshare_collector import AkshareDataCollector
        collector = AkshareDataCollector()
        print("✓ Data collector created successfully")
        
        # 测试技术指标
        from module_02_feature_engineering.feature_extraction.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        print("✓ Technical indicators created successfully")
        
        # 测试凯利准则
        from module_05_risk_management.position_sizing.kelly_criterion import KellyCriterion
        kelly = KellyCriterion()
        print("✓ Kelly criterion created successfully")
        
        # 测试回测引擎
        from module_09_backtesting.backtest_engine import BacktestEngine, BacktestConfig
        from datetime import datetime, timedelta
        config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now(),
            initial_capital=1000000
        )
        engine = BacktestEngine(config)
        print("✓ Backtest engine created successfully")
        
        # 测试需求解析器
        from module_10_ai_interaction.requirement_parser import RequirementParser
        parser = RequirementParser()
        print("✓ Requirement parser created successfully")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_main_app():
    """测试主应用"""
    print("\nTesting main application...")
    
    try:
        # 导入主应用
        from main import app, FinLoomEngine
        print("✓ Main application imported successfully")
        
        # 创建引擎实例
        engine = FinLoomEngine()
        print("✓ FinLoom engine created successfully")
        
        print("\n🎉 Main application test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Main application test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("FinLoom System Test")
    print("=" * 50)
    
    # 测试导入
    import_success = test_imports()
    
    if import_success:
        # 测试基本功能
        functionality_success = test_basic_functionality()
        
        if functionality_success:
            # 测试主应用
            app_success = test_main_app()
            
            if app_success:
                print("\n" + "=" * 50)
                print("🎉 ALL TESTS PASSED! System is ready to run.")
                print("You can now run: python main.py")
                return True
    
    print("\n" + "=" * 50)
    print("❌ Some tests failed. Please check the errors above.")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
