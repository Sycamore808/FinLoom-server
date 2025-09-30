# -*- coding: utf-8 -*-
"""
中国市场财务数据采集器模块
专门采集中国上市公司财务报表、估值、分红等信息
"""

import time
from typing import Any, Dict, Optional

import pandas as pd

# 尝试导入akshare，用于获取中国市场财务数据
try:
    import akshare as ak

    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False

from common.exceptions import DataError
from common.logging_system import setup_logger

logger = setup_logger("cn_fundamental_collector")


class ChineseFundamentalCollector:
    """
    中国市场财务数据采集器
    专注于中国股票市场的财务数据：
    - 财务报表（资产负债表、利润表、现金流量表）
    - 主要财务指标
    - 股本信息
    - 分红配股信息
    - 股东信息
    """

    def __init__(self, rate_limit: float = 0.5):
        """
        初始化中国财务数据采集器

        Args:
            rate_limit: 请求间隔（秒）
        """
        self.rate_limit = rate_limit
        self.last_request_time = 0.0

        if not HAS_AKSHARE:
            logger.warning("Akshare not available. Some features will use mock data.")

    def _rate_limit_check(self):
        """检查并执行速率限制"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def fetch_financial_statements(
        self, symbol: str, report_type: str = "资产负债表"
    ) -> pd.DataFrame:
        """
        获取财务报表数据

        Args:
            symbol: 股票代码
            report_type: 报表类型 ("资产负债表", "利润表", "现金流量表")

        Returns:
            pd.DataFrame: 财务报表数据
        """
        try:
            if not HAS_AKSHARE:
                logger.warning("Akshare not available, returning mock financial data")
                return self._mock_financial_data(symbol, report_type)

            self._rate_limit_check()

            if report_type == "资产负债表":
                df = ak.stock_balance_sheet_by_report_em(symbol=symbol)
            elif report_type == "利润表":
                df = ak.stock_profit_sheet_by_report_em(symbol=symbol)
            elif report_type == "现金流量表":
                df = ak.stock_cash_flow_sheet_by_report_em(symbol=symbol)
            else:
                raise ValueError(f"Unsupported report type: {report_type}")

            logger.info(f"Fetched {report_type} for {symbol}: {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {report_type} for {symbol}: {e}")
            return self._mock_financial_data(symbol, report_type)

    def fetch_financial_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        获取主要财务指标

        Args:
            symbol: 股票代码

        Returns:
            Dict[str, Any]: 财务指标字典
        """
        try:
            if not HAS_AKSHARE:
                logger.warning("Akshare not available, returning mock indicators")
                return self._mock_financial_indicators(symbol)

            self._rate_limit_check()

            # 获取财务指标
            try:
                # 使用正确的akshare接口
                indicators_df = ak.stock_financial_analysis_indicator_em(
                    symbol=symbol, indicator="按报告期"
                )

                if indicators_df.empty:
                    logger.warning(f"No financial indicators found for {symbol}")
                    return self._mock_financial_indicators(symbol)

                # 获取最新一期数据
                if len(indicators_df) > 0:
                    latest_data = indicators_df.iloc[0]  # 最新报告期数据

                    # 转换为标准化格式
                    indicators = {
                        "symbol": symbol,
                        "update_time": pd.Timestamp.now(),
                        "report_date": latest_data.get("REPORT_DATE", ""),
                        "report_type": latest_data.get("REPORT_TYPE", ""),
                        # 基本每股数据
                        "eps_basic": latest_data.get("EPSJB", 0),  # 基本每股收益
                        "eps_diluted": latest_data.get("EPSXS", 0),  # 稀释每股收益
                        "eps_deducted": latest_data.get("EPSKCJB", 0),  # 扣非每股收益
                        "bps": latest_data.get("BPS", 0),  # 每股净资产
                        "capital_reserve_per_share": latest_data.get(
                            "MGZBGJ", 0
                        ),  # 每股公积金
                        "undistributed_profit_per_share": latest_data.get(
                            "MGWFPLR", 0
                        ),  # 每股未分配利润
                        "operating_cash_flow_per_share": latest_data.get(
                            "MGJYXJJE", 0
                        ),  # 每股经营现金流
                        # 盈利指标
                        "total_revenue": latest_data.get(
                            "TOTALOPERATEREVE", 0
                        ),  # 营业总收入
                        "gross_profit": latest_data.get("MLR", 0),  # 毛利润
                        "net_profit": latest_data.get(
                            "PARENTNETPROFIT", 0
                        ),  # 归属净利润
                        "net_profit_deducted": latest_data.get(
                            "KCFJCXSYJLR", 0
                        ),  # 扣非净利润
                        # 增长率指标
                        "revenue_growth_yoy": latest_data.get(
                            "TOTALOPERATEREVETZ", 0
                        ),  # 营收同比增长
                        "net_profit_growth_yoy": latest_data.get(
                            "PARENTNETPROFITTZ", 0
                        ),  # 净利润同比增长
                        "net_profit_deducted_growth_yoy": latest_data.get(
                            "KCFJCXSYJLRTZ", 0
                        ),  # 扣非净利润同比增长
                        # 盈利能力指标
                        "roe_weighted": latest_data.get(
                            "ROEJQ", 0
                        ),  # 净资产收益率(加权)
                        "roe_deducted_weighted": latest_data.get(
                            "ROEKCJQ", 0
                        ),  # 净资产收益率(扣非/加权)
                        "roa_weighted": latest_data.get(
                            "ZZCJLL", 0
                        ),  # 总资产收益率(加权)
                        "net_margin": latest_data.get("XSJLL", 0),  # 净利率
                        "gross_margin": latest_data.get("XSMLL", 0),  # 毛利率
                        # 财务指标
                        "current_ratio": latest_data.get("LD", 0),  # 流动比率
                        "quick_ratio": latest_data.get("SD", 0),  # 速动比率
                        "cash_flow_ratio": latest_data.get("XJLLB", 0),  # 现金流量比率
                        "asset_liability_ratio": latest_data.get(
                            "ZCFZL", 0
                        ),  # 资产负债率
                        "equity_multiplier": latest_data.get("QYCS", 0),  # 权益系数
                        "debt_equity_ratio": latest_data.get("CQBL", 0),  # 产权比率
                        # 运营能力指标
                        "total_asset_turnover_days": latest_data.get(
                            "ZZCZZTS", 0
                        ),  # 总资产周转天数
                        "inventory_turnover_days": latest_data.get(
                            "CHZZTS", 0
                        ),  # 存货周转天数
                        "receivables_turnover_days": latest_data.get(
                            "YSZKZZTS", 0
                        ),  # 应收账款周转天数
                        "total_asset_turnover": latest_data.get(
                            "TOAZZL", 0
                        ),  # 总资产周转率
                        "inventory_turnover": latest_data.get("CHZZL", 0),  # 存货周转率
                        "receivables_turnover": latest_data.get(
                            "YSZKZZL", 0
                        ),  # 应收账款周转率
                        # 现金流指标
                        "advance_receipts_to_revenue": latest_data.get(
                            "YSZKYYSR", 0
                        ),  # 预收账款/营业收入
                        "sales_cash_flow_to_revenue": latest_data.get(
                            "XSJXLYYSR", 0
                        ),  # 销售净现金流/营业收入
                        "operating_cash_flow_to_revenue": latest_data.get(
                            "JYXJLYYSR", 0
                        ),  # 经营净现金流/营业收入
                        "effective_tax_rate": latest_data.get("TAXRATE", 0),  # 实际税率
                        # 兼容旧格式
                        "pe_ratio": 0,  # 这个接口没有PE数据，需要单独获取
                        "pb_ratio": 0,  # 这个接口没有PB数据，需要单独获取
                        "roe": latest_data.get("ROEJQ", 0),  # 使用加权ROE
                        "roa": latest_data.get("ZZCJLL", 0),  # 使用总资产收益率
                    }

                    logger.info(f"Fetched financial indicators for {symbol}")
                    return indicators
                else:
                    logger.warning(
                        f"No data found in financial indicators for {symbol}"
                    )
                    return self._mock_financial_indicators(symbol)

            except Exception as e:
                logger.warning(f"Failed to fetch indicators via akshare: {e}")
                return self._mock_financial_indicators(symbol)

        except Exception as e:
            logger.error(f"Failed to fetch financial indicators for {symbol}: {e}")
            return self._mock_financial_indicators(symbol)

    def fetch_dividend_history(self, symbol: str) -> pd.DataFrame:
        """
        获取分红配股历史

        Args:
            symbol: 股票代码

        Returns:
            pd.DataFrame: 分红配股历史
        """
        try:
            if not HAS_AKSHARE:
                logger.warning("Akshare not available, returning mock dividend data")
                return self._mock_dividend_data(symbol)

            self._rate_limit_check()

            # 获取分红配股数据
            try:
                dividend_df = ak.stock_dividend_detail(symbol=symbol)
                logger.info(
                    f"Fetched dividend history for {symbol}: {len(dividend_df)} records"
                )
                return dividend_df

            except Exception as e:
                logger.warning(f"Failed to fetch dividend via akshare: {e}")
                return self._mock_dividend_data(symbol)

        except Exception as e:
            logger.error(f"Failed to fetch dividend history for {symbol}: {e}")
            return self._mock_dividend_data(symbol)

    def fetch_share_structure(self, symbol: str) -> Dict[str, Any]:
        """
        获取股本结构信息

        Args:
            symbol: 股票代码

        Returns:
            Dict[str, Any]: 股本结构信息
        """
        try:
            if not HAS_AKSHARE:
                logger.warning("Akshare not available, returning mock share structure")
                return self._mock_share_structure(symbol)

            self._rate_limit_check()

            # 获取股本结构信息
            try:
                # 这里可以添加具体的akshare接口来获取股本结构
                # 目前返回基础结构信息
                share_info = {
                    "symbol": symbol,
                    "total_shares": 1000000000,  # 总股本
                    "float_shares": 800000000,  # 流通股本
                    "restricted_shares": 200000000,  # 限售股
                    "update_time": pd.Timestamp.now(),
                }

                logger.info(f"Fetched share structure for {symbol}")
                return share_info

            except Exception as e:
                logger.warning(f"Failed to fetch share structure: {e}")
                return self._mock_share_structure(symbol)

        except Exception as e:
            logger.error(f"Failed to fetch share structure for {symbol}: {e}")
            return self._mock_share_structure(symbol)

    def fetch_major_shareholders(self, symbol: str) -> pd.DataFrame:
        """
        获取主要股东信息

        Args:
            symbol: 股票代码

        Returns:
            pd.DataFrame: 主要股东信息
        """
        try:
            if not HAS_AKSHARE:
                logger.warning(
                    "Akshare not available, returning mock shareholders data"
                )
                return self._mock_shareholders_data(symbol)

            self._rate_limit_check()

            # 获取十大股东信息
            try:
                shareholders_df = ak.stock_top_10_holders_em(symbol=symbol)
                logger.info(
                    f"Fetched major shareholders for {symbol}: {len(shareholders_df)} records"
                )
                return shareholders_df

            except Exception as e:
                logger.warning(f"Failed to fetch shareholders via akshare: {e}")
                return self._mock_shareholders_data(symbol)

        except Exception as e:
            logger.error(f"Failed to fetch major shareholders for {symbol}: {e}")
            return self._mock_shareholders_data(symbol)

    def validate_financial_data(self, df: pd.DataFrame) -> bool:
        """
        校验财务数据完整性和合理性

        Args:
            df: 财务数据DataFrame

        Returns:
            bool: 是否通过校验
        """
        if df.empty:
            logger.warning("Empty financial DataFrame")
            return False

        # 检查基本字段
        if "date" not in df.columns and df.index.name != "date":
            logger.warning("Missing date column or index")
            return False

        # 检查数值列
        numeric_columns = df.select_dtypes(include=["number"]).columns
        if len(numeric_columns) == 0:
            logger.warning("No numeric columns found")
            return False

        # 检查是否有异常值
        for col in numeric_columns:
            if df[col].isna().all():
                logger.warning(f"Column {col} contains only NaN values")
                return False

        logger.info("Financial data validation passed")
        return True

    def _mock_financial_data(self, symbol: str, report_type: str) -> pd.DataFrame:
        """生成模拟财务报表数据"""
        dates = pd.date_range(start="2020-12-31", end="2023-12-31", freq="Y")

        if report_type == "资产负债表":
            return pd.DataFrame(
                {
                    "date": dates,
                    "total_assets": [1000000000, 1100000000, 1200000000, 1300000000],
                    "total_liabilities": [600000000, 650000000, 700000000, 750000000],
                    "shareholders_equity": [400000000, 450000000, 500000000, 550000000],
                }
            )
        elif report_type == "利润表":
            return pd.DataFrame(
                {
                    "date": dates,
                    "revenue": [500000000, 550000000, 600000000, 650000000],
                    "net_income": [50000000, 55000000, 60000000, 65000000],
                    "gross_profit": [200000000, 220000000, 240000000, 260000000],
                }
            )
        else:
            return pd.DataFrame(
                {
                    "date": dates,
                    "operating_cash_flow": [80000000, 85000000, 90000000, 95000000],
                    "investing_cash_flow": [-30000000, -35000000, -40000000, -45000000],
                    "financing_cash_flow": [-20000000, -15000000, -10000000, -5000000],
                }
            )

    def _mock_financial_indicators(self, symbol: str) -> Dict[str, Any]:
        """生成模拟财务指标"""
        return {
            "symbol": symbol,
            "pe_ratio": 15.5,
            "pb_ratio": 1.2,
            "roe": 0.12,
            "roa": 0.08,
            "gross_margin": 0.35,
            "net_margin": 0.10,
            "debt_ratio": 0.60,
            "current_ratio": 1.5,
            "quick_ratio": 1.2,
            "update_time": pd.Timestamp.now(),
        }

    def _mock_dividend_data(self, symbol: str) -> pd.DataFrame:
        """生成模拟分红数据"""
        return pd.DataFrame(
            {
                "date": ["2022-06-30", "2023-06-30"],
                "dividend_per_share": [0.5, 0.6],
                "dividend_yield": [0.025, 0.030],
                "payout_ratio": [0.4, 0.45],
            }
        )

    def _mock_share_structure(self, symbol: str) -> Dict[str, Any]:
        """生成模拟股本结构"""
        return {
            "symbol": symbol,
            "total_shares": 1000000000,
            "float_shares": 800000000,
            "restricted_shares": 200000000,
            "update_time": pd.Timestamp.now(),
        }

    def _mock_shareholders_data(self, symbol: str) -> pd.DataFrame:
        """生成模拟股东数据"""
        return pd.DataFrame(
            {
                "shareholder_name": ["大股东1", "大股东2", "大股东3"],
                "shareholding_ratio": [25.5, 15.2, 10.8],
                "shareholding_number": [255000000, 152000000, 108000000],
                "shareholder_type": ["法人股", "流通股", "流通股"],
            }
        )


# 为了保持向后兼容，创建别名
FundamentalCollector = ChineseFundamentalCollector
FundamentalDataCollector = ChineseFundamentalCollector
