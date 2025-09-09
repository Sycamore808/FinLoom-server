#!/usr/bin/env python3
"""
FinLoom 数据收集示例
演示如何使用FinLoom系统收集和处理市场数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module_01_data_pipeline.data_acquisition.akshare_collector import AkshareDataCollector
from module_02_feature_engineering.feature_extraction.technical_indicators import TechnicalIndicators
from module_01_data_pipeline.storage_management.database_manager import get_database_manager
import pandas as pd
from datetime import datetime, timedelta

def main():
    """主函数 - 演示数据收集和处理"""
    print("FinLoom 数据收集示例")
    print("=" * 50)
    
    # 1. 创建数据收集器
    print("1. 初始化数据收集器...")
    collector = AkshareDataCollector()
    
    # 2. 收集股票数据
    print("2. 收集股票数据...")
    symbols = ["000001", "000002", "600000"]  # 示例股票代码
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")
    
    all_data = {}
    for symbol in symbols:
        print(f"   收集 {symbol} 的数据...")
        try:
            data = collector.get_stock_data(symbol, start_date, end_date)
            if not data.empty:
                all_data[symbol] = data
                print(f"   成功收集 {len(data)} 条记录")
            else:
                print(f"   未获取到数据")
        except Exception as e:
            print(f"   收集失败: {e}")
    
    print()
    
    # 3. 计算技术指标
    print("3. 计算技术指标...")
    calculator = TechnicalIndicators()
    
    for symbol, data in all_data.items():
        print(f"   计算 {symbol} 的技术指标...")
        try:
            # 确保数据格式正确
            if 'close' in data.columns:
                indicators = calculator.calculate_all_indicators(data)
                print(f"   成功计算 {len(indicators.columns)} 个指标")
                
                # 显示一些指标
                latest_data = indicators.iloc[-1]
                print(f"   RSI: {latest_data.get('rsi', 'N/A'):.2f}")
                print(f"   MACD: {latest_data.get('macd', 'N/A'):.4f}")
            else:
                print(f"   数据格式不正确，跳过技术指标计算")
        except Exception as e:
            print(f"   计算失败: {e}")
    
    print()
    
    # 4. 存储到数据库
    print("4. 存储数据到数据库...")
    try:
        db_manager = get_database_manager()
        
        for symbol, data in all_data.items():
            print(f"   存储 {symbol} 的数据...")
            success = db_manager.insert_market_data(data, symbol)
            if success:
                print(f"   存储成功")
            else:
                print(f"   存储失败")
        
        # 查询验证
        print("   验证数据存储...")
        test_data = db_manager.get_market_data(symbols[0])
        if not test_data.empty:
            print(f"   验证成功，查询到 {len(test_data)} 条记录")
        else:
            print(f"   验证失败，未查询到数据")
            
    except Exception as e:
        print(f"   数据库操作失败: {e}")
    
    print()
    print("数据收集示例完成！")

if __name__ == "__main__":
    main()
