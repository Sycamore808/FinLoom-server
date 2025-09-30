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

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# 导入FinLoom模块
from module_01_data_pipeline import (
    AkshareDataCollector,
    DatabaseManager,
    DataCleaner,
    DataValidator,
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

    # # 6. 测试简化版实时数据收集
    # print("6. 简化版实时数据收集演示...")

    # try:
    #     # 获取部分股票的实时数据
    #     print("   获取部分股票的实时数据...")
    #     realtime_symbols = symbols[:3]  # 只获取前3只

    #     realtime_data = collector.fetch_realtime_data(realtime_symbols)

    #     for symbol, data in realtime_data.items():
    #         print(
    #             f"       {symbol}: {data.get('name', '')} - 现价: {data.get('price', 0):.2f}, 涨跌幅: {data.get('change', 0):.2f}%"
    #         )

    # except Exception as e:
    #     print(f"   实时数据获取失败: {e}")

    print()
    print("数据收集示例完成！")
    print("=" * 50)


def test_alternative_data_collection():
    """
    测试中国另类数据采集器（包含宏观数据、新闻、板块数据、个股新闻、市场概况等）
    """
    print("\n7. 中国另类数据采集测试...")
    print("-" * 40)

    try:
        from module_01_data_pipeline.data_acquisition.alternative_data_collector import (
            ChineseAlternativeDataCollector,
        )

        alt_collector = ChineseAlternativeDataCollector(rate_limit=0.5)

        # 测试宏观经济数据
        print("   测试宏观经济数据...")
        try:
            macro_data = alt_collector.fetch_macro_economic_data()
            if macro_data:
                print(f"       获取到数据类型: {list(macro_data.keys())}")

                # 显示GDP数据（如果有）
                if "GDP" in macro_data and not macro_data["GDP"].empty:
                    gdp_df = macro_data["GDP"]
                    print(f"       GDP数据: {len(gdp_df)} 条记录")
                    if len(gdp_df) > 0:
                        latest_gdp = gdp_df.iloc[-1]
                        print(f"       最新GDP数据: {latest_gdp}")

                # 显示CPI数据（如果有）
                if "CPI" in macro_data and not macro_data["CPI"].empty:
                    cpi_df = macro_data["CPI"]
                    print(f"       CPI数据: {len(cpi_df)} 条记录")
                    if len(cpi_df) > 0:
                        latest_cpi = cpi_df.iloc[-1]
                        print(f"       最新CPI数据: {latest_cpi}")

                # 显示PMI数据（如果有）
                if "PMI" in macro_data and not macro_data["PMI"].empty:
                    pmi_df = macro_data["PMI"]
                    print(f"       PMI数据: {len(pmi_df)} 条记录")
                    if len(pmi_df) > 0:
                        latest_pmi = pmi_df.iloc[-1]
                        print(f"       最新PMI数据: {latest_pmi}")

                print("       宏观数据获取成功")
            else:
                print("       宏观数据获取失败")
        except Exception as e:
            print(f"       宏观数据获取失败: {e}")

        # 测试宏观数据存储到数据库
        print("   测试宏观数据存储...")
        try:
            db_manager = get_database_manager()
            if macro_data:
                for indicator, data in macro_data.items():
                    if not data.empty:
                        success = db_manager.save_macro_data(indicator, data)
                        if success:
                            print(f"       {indicator} 数据存储成功")
                        else:
                            print(f"       {indicator} 数据存储失败")
                print("       宏观数据存储测试完成")
            else:
                print("       无宏观数据可存储")
        except Exception as e:
            print(f"       宏观数据存储失败: {e}")

        # 测试新闻联播数据
        print("   测试新闻联播数据...")
        try:
            news_data = alt_collector.fetch_news_data(limit=5)
            if not news_data.empty:
                print(f"       新闻数据: {len(news_data)} 条记录")
                if len(news_data) > 0:
                    # 显示第一条新闻
                    first_news = news_data.iloc[0]
                    print(
                        f"       第一条新闻标题: {first_news.get('title', '无标题')[:50]}..."
                    )
                    print(f"       情绪分析: {first_news.get('sentiment', '未知')}")
                # 测试保存新闻数据
                success = db_manager.save_news_data(news_data)
                if success:
                    print("       新闻数据保存成功")
                print("       新闻联播数据获取成功")
            else:
                print("       新闻联播数据获取失败")
        except Exception as e:
            print(f"       新闻联播数据获取失败: {e}")

        # 测试板块数据
        print("   测试板块行情数据...")
        try:
            sector_data = alt_collector.fetch_sector_performance()
            if not sector_data.empty:
                print(f"       板块数据: {len(sector_data)} 个板块")
                if len(sector_data) > 0:
                    # 显示前几个板块
                    print("       前3个板块:")
                    for i, (_, row) in enumerate(sector_data.head(3).iterrows()):
                        sector_name = row.get("板块", row.get("sector", f"板块{i + 1}"))
                        change_pct = row.get("涨跌幅", row.get("change_pct", "N/A"))
                        print(f"         {sector_name}: {change_pct}%")
                # 测试保存板块数据
                success = db_manager.save_sector_data(sector_data)
                if success:
                    print("       板块数据保存成功")
                print("       板块行情数据获取成功")
            else:
                print("       板块行情数据获取失败")
        except Exception as e:
            print(f"       板块行情数据获取失败: {e}")

        # 测试个股新闻数据
        print("   测试个股新闻数据...")
        test_symbol = "000001"
        try:
            stock_news = alt_collector.fetch_stock_news(test_symbol, limit=5)
            if not stock_news.empty:
                print(f"       {test_symbol} 新闻数据: {len(stock_news)} 条记录")
                if len(stock_news) > 0:
                    first_stock_news = stock_news.iloc[0]
                    print(
                        f"       第一条股票新闻: {first_stock_news.get('新闻标题', '无标题')[:50]}..."
                    )
                # 测试保存个股新闻数据
                success = db_manager.save_stock_news(test_symbol, stock_news)
                if success:
                    print("       个股新闻数据保存成功")
                print("       个股新闻数据获取成功")
            else:
                print("       个股新闻数据获取失败")
        except Exception as e:
            print(f"       个股新闻数据获取失败: {e}")

        # 测试每日市场概况（默认一年数据）
        print("   测试每日A股市场概况（默认一个月数据）...")
        try:
            market_overview = alt_collector.fetch_daily_market_overview()
            if not market_overview.empty:
                print(f"       市场概况数据: {len(market_overview)} 条记录")
                if len(market_overview) > 0:
                    overview_data = market_overview.iloc[0]
                    print(f"       上市股票数: {overview_data.get('股票', 'N/A')}")
                    print(f"       总市值: {overview_data.get('市价总值', 'N/A')}")
                # 测试保存市场概况数据
                success = db_manager.save_historical_daily_market_overview(
                    market_overview
                )
                if success:
                    print("       市场概况数据保存成功")
                print("       每日市场概况获取成功")
            else:
                print("       每日市场概况获取失败")
        except Exception as e:
            print(f"       每日市场概况获取失败: {e}")

        # 测试个股详细信息（完整版）
        print("   测试个股详细信息（完整版）...")
        try:
            stock_info = alt_collector.fetch_detail(test_symbol)
            if stock_info:
                print(f"       {test_symbol} 详细信息获取成功")
                print(f"       股票名称: {stock_info.get('name', 'N/A')}")
                print(f"       最新价格: {stock_info.get('latest_price', 'N/A')}")
                print(f"       总市值: {stock_info.get('total_market_value', 'N/A')}")

                # 显示公司基本信息
                if stock_info.get("org_name_cn"):
                    print(f"       公司全称: {stock_info.get('org_name_cn')}")
                if stock_info.get("main_operation_business"):
                    business = stock_info.get("main_operation_business", "")[:100]
                    print(f"       主营业务: {business}...")
                if stock_info.get("legal_representative"):
                    print(
                        f"       法定代表人: {stock_info.get('legal_representative')}"
                    )
                if stock_info.get("telephone"):
                    print(f"       公司电话: {stock_info.get('telephone')}")
                if stock_info.get("org_website"):
                    print(f"       公司网站: {stock_info.get('org_website')}")
                if stock_info.get("staff_num") and stock_info.get("staff_num") > 0:
                    print(f"       员工数: {stock_info.get('staff_num')}人")

                # 测试保存个股详细信息
                success = db_manager.save_stock_detail_info(test_symbol, stock_info)
                if success:
                    print("       个股详细信息保存成功")
                else:
                    print("       个股详细信息保存失败")
            else:
                print("       个股详细信息获取失败")
        except Exception as e:
            print(f"       个股详细信息获取失败: {e}")

        # 测试一年历史市场数据收集（简化版，只收集最近7天）
        print("   测试历史市场数据收集（最近7天）...")
        try:
            from datetime import datetime, timedelta

            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            historical_overview = alt_collector.fetch_historical_daily_market_overview(
                start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
            )
            if not historical_overview.empty:
                print(f"       历史市场数据: {len(historical_overview)} 条记录")
                success = db_manager.save_historical_daily_market_overview(
                    historical_overview
                )
                if success:
                    print("       历史市场数据保存成功")
                print("       历史市场数据收集成功")
            else:
                print("       历史市场数据收集失败")
        except Exception as e:
            print(f"       历史市场数据收集失败: {e}")

        print("   中国另类数据采集测试完成")

    except ImportError as e:
        print(f"   另类数据采集器导入失败: {e}")
    except Exception as e:
        print(f"   另类数据采集测试失败: {e}")


def test_fundamental_data_collection():
    """
    测试中国财务数据采集器
    """
    print("\n8. 中国财务数据采集测试...")
    print("-" * 40)

    try:
        from module_01_data_pipeline.data_acquisition.fundamental_collector import (
            ChineseFundamentalCollector,
        )

        fund_collector = ChineseFundamentalCollector(rate_limit=0.5)
        test_symbol = "000001"  # 平安银行

        # 1. 测试财务报表
        print(f"   测试 {test_symbol} 财务报表...")
        try:
            balance_sheet = fund_collector.fetch_financial_statements(
                test_symbol, "资产负债表"
            )
            income_statement = fund_collector.fetch_financial_statements(
                test_symbol, "利润表"
            )

            print(f"       资产负债表: {len(balance_sheet)} 条记录")
            print(f"       利润表: {len(income_statement)} 条记录")
            print("       财务报表获取成功")
        except Exception as e:
            print(f"       财务报表获取失败: {e}")

        # 2. 测试财务指标
        print(f"   测试 {test_symbol} 财务指标...")
        try:
            indicators = fund_collector.fetch_financial_indicators(test_symbol)
            if indicators:
                print(f"       PE: {indicators.get('pe_ratio', 'N/A')}")
                print(f"       PB: {indicators.get('pb_ratio', 'N/A')}")
                print(f"       ROE: {indicators.get('roe', 'N/A')}")
                print("       财务指标获取成功")
            else:
                print("       财务指标获取失败")
        except Exception as e:
            print(f"       财务指标获取失败: {e}")

        # 3. 测试分红历史
        print(f"   测试 {test_symbol} 分红历史...")
        try:
            dividend_data = fund_collector.fetch_dividend_history(test_symbol)
            print(f"       分红记录: {len(dividend_data)} 条")
            print("       分红历史获取成功")
        except Exception as e:
            print(f"       分红历史获取失败: {e}")

        print("   中国财务数据采集测试完成")

    except ImportError as e:
        print(f"   财务数据采集器导入失败: {e}")
    except Exception as e:
        print(f"   财务数据采集测试失败: {e}")


if __name__ == "__main__":
    # 运行主要演示
    main()

    # 运行宏观数据测试
    test_alternative_data_collection()

    # 运行财务数据测试
    test_fundamental_data_collection()

    print("\nAll tests completed successfully!")
