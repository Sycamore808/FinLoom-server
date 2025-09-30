#!/usr/bin/env python3
"""
FinLoom 回测示例
演示如何使用FinLoom系统进行策略回测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module_09_backtesting.backtest_engine import BacktestEngine, BacktestConfig
from module_01_data_pipeline.data_acquisition.akshare_collector import AkshareDataCollector
from module_02_feature_engineering.feature_extraction.technical_indicators import TechnicalIndicators
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def simple_moving_average_strategy(data: pd.DataFrame) -> str:
    """简单的移动平均策略
    
    Args:
        data: 市场数据
        
    Returns:
        交易信号: 'BUY', 'SELL', 'HOLD'
    """
    try:
        if len(data) < 20:
            return 'HOLD'
        
        # 计算5日和20日移动平均
        sma_5 = data['close'].rolling(5).mean().iloc[-1]
        sma_20 = data['close'].rolling(20).mean().iloc[-1]
        
        # 简单策略：5日均线上穿20日均线买入，下穿卖出
        if sma_5 > sma_20:
            return 'BUY'
        elif sma_5 < sma_20:
            return 'SELL'
        else:
            return 'HOLD'
            
    except Exception:
        return 'HOLD'

def rsi_strategy(data: pd.DataFrame) -> str:
    """RSI策略
    
    Args:
        data: 市场数据
        
    Returns:
        交易信号: 'BUY', 'SELL', 'HOLD'
    """
    try:
        if len(data) < 14:
            return 'HOLD'
        
        # 计算RSI
        calculator = TechnicalIndicators()
        rsi = calculator.calculate_rsi(data['close'])
        
        if rsi.iloc[-1] < 30:  # 超卖
            return 'BUY'
        elif rsi.iloc[-1] > 70:  # 超买
            return 'SELL'
        else:
            return 'HOLD'
            
    except Exception:
        return 'HOLD'

def main():
    """主函数 - 演示回测功能"""
    print("FinLoom 回测示例")
    print("=" * 50)
    
    # 1. 收集数据
    print("1. 收集回测数据...")
    collector = AkshareDataCollector()
    
    symbol = "000001"  # 平安银行
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")
    
    try:
        data = collector.get_stock_data(symbol, start_date, end_date)
        if data.empty:
            print("   未获取到数据，退出")
            return
        
        print(f"   成功收集 {len(data)} 条记录")
        
        # 准备数据
        market_data = {symbol: data}
        
    except Exception as e:
        print(f"   数据收集失败: {e}")
        return
    
    print()
    
    # 2. 配置回测参数
    print("2. 配置回测参数...")
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=300),
        end_date=datetime.now(),
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_bps=5.0
    )
    
    print(f"   初始资金: {config.initial_capital:,.0f}")
    print(f"   佣金费率: {config.commission_rate:.3f}")
    print(f"   滑点: {config.slippage_bps} bps")
    
    print()
    
    # 3. 运行移动平均策略回测
    print("3. 运行移动平均策略回测...")
    try:
        engine = BacktestEngine(config)
        engine.load_market_data([symbol], market_data)
        engine.set_strategy(simple_moving_average_strategy)
        
        result = engine.run()
        
        print(f"   总收益率: {result.total_return:.2%}")
        print(f"   年化收益率: {result.annualized_return:.2%}")
        print(f"   夏普比率: {result.sharpe_ratio:.2f}")
        print(f"   最大回撤: {result.max_drawdown:.2%}")
        print(f"   胜率: {result.win_rate:.2%}")
        print(f"   总交易次数: {result.total_trades}")
        
    except Exception as e:
        print(f"   移动平均策略回测失败: {e}")
    
    print()
    
    # 4. 运行RSI策略回测
    print("4. 运行RSI策略回测...")
    try:
        engine2 = BacktestEngine(config)
        engine2.load_market_data([symbol], market_data)
        engine2.set_strategy(rsi_strategy)
        
        result2 = engine2.run()
        
        print(f"   总收益率: {result2.total_return:.2%}")
        print(f"   年化收益率: {result2.annualized_return:.2%}")
        print(f"   夏普比率: {result2.sharpe_ratio:.2f}")
        print(f"   最大回撤: {result2.max_drawdown:.2%}")
        print(f"   胜率: {result2.win_rate:.2%}")
        print(f"   总交易次数: {result2.total_trades}")
        
    except Exception as e:
        print(f"   RSI策略回测失败: {e}")
    
    print()
    print("回测示例完成！")

if __name__ == "__main__":
    main()
