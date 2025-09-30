#!/usr/bin/env python3
"""
FinLoom 数据收集示例
演示如何使用FinLoom系统收集和处理市场数据

主要功能：
1. 数据收集 - 从多个源获取中国股票数据
2. 数据清洗 - 数据质量检查和清洗
3. 技术指标 - 计算常用技术指标
4. 数据存储 - 保存到数据库
5. 实时处理 - 演示实时数据处理和信号生成
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# 导入FinLoom模块
from module_01_data_pipeline import (
    AkshareDataCollector,
    DatabaseManager,
    DataCleaner,
    DataValidator,
    MarketDataCollector,
    RealTimeProcessor,
    get_database_manager,
    quick_clean_data,
    validate_dataframe,
)

# 尝试导入技术指标模块
try:
    from module_02_feature_engineering.feature_extraction.technical_indicators import (
        TechnicalIndicators,
    )

    HAS_TECHNICAL_INDICATORS = True
except ImportError:
    HAS_TECHNICAL_INDICATORS = False
    print("Warning: Technical indicators module not available")


def main():
    """主函数 - 演示数据收集和处理"""
    print("FinLoom 数据收集示例")
    print("=" * 50)

    # 测试用股票代码（中国市场）
    symbols = [
        "000001",
        "000002",
        "600000",
        "600036",
        "000858",
    ]  # 平安银行、万科A、浦发银行、招商银行、五粮液
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")  # 最近3个月
    end_date = datetime.now().strftime("%Y%m%d")

    # 1. 初始化组件
    print("1. 初始化数据收集组件...")
    collector = AkshareDataCollector(rate_limit=0.5)  # 0.5秒间隔
    cleaner = DataCleaner(fill_method="interpolate", outlier_method="iqr")
    validator = DataValidator()

    if HAS_TECHNICAL_INDICATORS:
        tech_indicators = TechnicalIndicators()

    db_manager = get_database_manager()

    print(f"   目标股票: {', '.join(symbols)}")
    print(f"   日期范围: {start_date} - {end_date}")
    print()

    # 2. 数据收集
    print("2. 开始数据收集...")
    all_data = {}
    all_cleaned_data = {}

    for i, symbol in enumerate(symbols, 1):
        print(f"   ({i}/{len(symbols)}) 收集 {symbol} 的数据...")

        try:
            # 获取基本信息
            basic_info = collector.get_stock_basic_info(symbol)
            if basic_info:
                print(f"       股票名称: {basic_info.get('name', '未知')}")
                print(f"       所属行业: {basic_info.get('industry', '未知')}")

            # 获取历史数据
            data = collector.fetch_stock_history(symbol, start_date, end_date)

            if not data.empty:
                all_data[symbol] = data
                print(f"       收集成功: {len(data)} 条记录")
                print(f"       日期范围: {data['date'].min()} 至 {data['date'].max()}")
                print(
                    f"       价格范围: {data['close'].min():.2f} - {data['close'].max():.2f}"
                )
            else:
                print(f"       未获取到数据")

        except Exception as e:
            print(f"       收集失败: {e}")

    print(f"   数据收集完成，成功获取 {len(all_data)} 只股票数据\n")

    # 3. 数据清洗和验证
    print("3. 数据清洗和验证...")

    for symbol, data in all_data.items():
        print(f"   处理 {symbol}:")

        # 数据验证
        validation_result = validator.validate_market_data(data, symbol)
        print(f"       数据质量评分: {validation_result.quality_score:.2f}")
        validation_status = "通过" if validation_result.is_valid else "失败"
        print(f"       验证结果: {validation_status}")

        if validation_result.issues:
            print(f"       问题: {', '.join(validation_result.issues[:3])}")

        # 数据清洗
        cleaned_data = cleaner.clean_market_data(data, symbol)
        all_cleaned_data[symbol] = cleaned_data

        # 数据质量报告
        quality_report = cleaner.detect_data_quality_issues(cleaned_data)
        print(f"       清洗后质量评分: {quality_report['quality_score']:.2f}")
        print(f"       记录数量: {quality_report['total_records']}")

    print()

    # 4. 技术指标计算
    if HAS_TECHNICAL_INDICATORS:
        print("4. 计算技术指标...")

        for symbol, data in all_cleaned_data.items():
            if len(data) >= 50:  # 需要足够的数据
                print(f"   计算 {symbol} 的技术指标...")

                try:
                    # 计算所有指标
                    indicators = tech_indicators.calculate_all_indicators(data)

                    # 显示最新指标值
                    latest = indicators.iloc[-1]
                    rsi_val = latest.get("rsi", "N/A")
                    macd_val = latest.get("macd", "N/A")
                    sma20_val = latest.get("sma_20", "N/A")
                    bb_upper_val = latest.get("bb_upper", "N/A")
                    bb_lower_val = latest.get("bb_lower", "N/A")

                    if isinstance(rsi_val, (int, float)):
                        print(f"       RSI: {rsi_val:.2f}")
                    else:
                        print(f"       RSI: {rsi_val}")

                    if isinstance(macd_val, (int, float)):
                        print(f"       MACD: {macd_val:.4f}")
                    else:
                        print(f"       MACD: {macd_val}")

                    if isinstance(sma20_val, (int, float)):
                        print(f"       SMA20: {sma20_val:.2f}")
                    else:
                        print(f"       SMA20: {sma20_val}")

                    if isinstance(bb_upper_val, (int, float)):
                        print(f"       布林上轨: {bb_upper_val:.2f}")
                    else:
                        print(f"       布林上轨: {bb_upper_val}")

                    if isinstance(bb_lower_val, (int, float)):
                        print(f"       布林下轨: {bb_lower_val:.2f}")
                    else:
                        print(f"       布林下轨: {bb_lower_val}")

                    # 保存技术指标到数据中
                    all_cleaned_data[symbol] = indicators

                except Exception as e:
                    print(f"       指标计算失败: {e}")
            else:
                print(f"   {symbol} 数据不足，跳过技术指标计算")

        print()
    else:
        print("4. 技术指标模块不可用，跳过指标计算\n")

    # 5. 数据存储
    print("5. 保存数据到数据库...")

    for symbol, data in all_cleaned_data.items():
        try:
            # 保存股票基本信息
            basic_info = collector.get_stock_basic_info(symbol)
            if basic_info:
                db_manager.save_stock_info(
                    symbol=symbol,
                    name=basic_info.get("name", ""),
                    sector=basic_info.get("industry", ""),
                    industry=basic_info.get("area", ""),
                )

            # 保存价格数据
            success = db_manager.save_stock_prices(symbol, data)

            if success:
                print(f"   {symbol}: 保存成功 ({len(data)} 条记录)")
            else:
                print(f"   {symbol}: 保存失败")

            # 如果有技术指标，也保存
            if HAS_TECHNICAL_INDICATORS and "rsi" in data.columns:
                db_manager.save_technical_indicators(symbol, data)
                print(f"   {symbol}: 技术指标保存成功")

        except Exception as e:
            print(f"   {symbol}: 保存失败 - {e}")

    # 验证数据存储
    print("\n   验证数据存储...")
    try:
        # 获取数据库统计信息
        stats = db_manager.get_database_stats()
        print(f"   数据库统计:")
        print(f"       股票价格记录: {stats.get('stock_prices_count', 0)}")
        print(f"       独特股票数: {stats.get('unique_symbols', 0)}")
        print(f"       数据库大小: {stats.get('database_size_mb', 0):.2f} MB")

        # 获取一只股票的数据验证
        if symbols:
            test_symbol = symbols[0]
            test_data = db_manager.get_stock_prices(test_symbol)
            if not test_data.empty:
                print(
                    f"       验证成功: 查询到 {test_symbol} 的 {len(test_data)} 条记录"
                )
            else:
                print(f"       验证失败: 未查询到 {test_symbol} 的数据")

    except Exception as e:
        print(f"   数据库验证失败: {e}")

    print()

    # 6. 实时数据处理演示
    print("6. 实时数据处理演示...")

    try:
        # 创建实时处理器
        processor = RealTimeProcessor(config={})

        # 添加信号回调
        def signal_callback(symbol, signals):
            for signal in signals:
                print(
                    f"       信号: {symbol} - {signal.signal_type} (强度: {signal.strength:.2f}, 置信度: {signal.confidence:.2f})"
                )

        processor.add_signal_callback(signal_callback)

        # 模拟实时数据更新和信号生成
        for symbol, data in list(all_cleaned_data.items())[:2]:  # 只演示前2只股票
            if len(data) >= 50:
                print(f"   模拟 {symbol} 的实时数据处理...")

                # 更新市场数据
                processor.update_market_data(symbol, data)

                # 生成信号
                signals = processor.generate_signals(symbol)

                print(f"       生成了 {len(signals)} 个信号")

                # 显示最新信号
                recent_signals = processor.get_latest_signals(symbol, limit=3)
                for signal in recent_signals[-3:]:
                    print(
                        f"         - {signal.signal_type} @{signal.price:.2f} (强度:{signal.strength:.2f})"
                    )

    except Exception as e:
        print(f"   实时处理演示失败: {e}")

    print()

    # 7. 实时数据收集演示
    print("7. 实时数据收集演示...")

    try:
        # 使用便捷函数收集实时数据
        print("   获取部分股票的实时数据...")
        realtime_symbols = symbols[:3]  # 只获取前3只

        realtime_data = collector.fetch_realtime_data(realtime_symbols)

        for symbol, data in realtime_data.items():
            print(
                f"       {symbol}: {data.get('name', '')} - 现价: {data.get('price', 0):.2f}, 涨跌幅: {data.get('change', 0):.2f}%"
            )

    except Exception as e:
        print(f"   实时数据获取失败: {e}")

    print()
    print("数据收集示例完成！")
    print("=" * 50)


async def async_data_collection_demo():
    """
    异步数据收集演示
    """
    print("\n异步数据收集演示")
    print("-" * 30)

    try:
        from module_01_data_pipeline import collect_market_data, collect_realtime_data

        symbols = ["000001", "600036", "000858"]

        # 异步收集历史数据
        print("异步收集历史数据...")
        historical_data = await collect_market_data(symbols, lookback_days=30)
        print(f"成功收集 {len(historical_data)} 条历史记录")

        # 异步收集实时数据
        print("异步收集实时数据...")
        realtime_data = await collect_realtime_data(symbols)
        print(f"成功收集 {len(realtime_data)} 只股票的实时数据")

        for symbol, data in realtime_data.items():
            print(f"  {symbol}: 价格 {data.close:.2f}")

    except Exception as e:
        print(f"异步演示失败: {e}")


if __name__ == "__main__":
    # 运行主要演示
    main()

    # 运行异步演示
    try:
        import asyncio

        asyncio.run(async_data_collection_demo())
    except Exception as e:
        print(f"异步演示跳过: {e}")
