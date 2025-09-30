# -*- coding: utf-8 -*-
"""
财务数据采集器模块
支持多种API采集上市公司财务报表、估值、分红等信息
"""

from typing import Any, Dict, Optional

import pandas as pd
import requests

from common.exceptions import DataError
from common.logging_system import setup_logger

logger = setup_logger("fundamental_collector")


class FundamentalCollector:
    """
    财务数据采集器，支持多种数据源（如新浪、Yahoo、Tushare、FinancialModelingPrep等）
    """

    def __init__(self, proxies: Optional[Dict[str, str]] = None):
        self.proxies = proxies

    def fetch_yahoo_financials(self, symbol: str) -> Dict[str, Any]:
        """
        获取Yahoo Finance财务摘要（伪实现）
        Args:
                symbol: 股票代码（如AAPL）
        Returns:
                Dict[str, Any]: 财务摘要
        Raises:
                DataError: 获取失败
        """
        try:
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=financialData,defaultKeyStatistics,price"
            resp = requests.get(url, proxies=self.proxies, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            summary = data["quoteSummary"]["result"][0]
            logger.info(f"Fetched Yahoo financials for {symbol}")
            return summary
        except Exception as e:
            logger.error(f"Failed to fetch Yahoo financials: {symbol}, error: {e}")
            raise DataError(f"Yahoo financials fetch failed: {e}")

    def fetch_fmp_financials(self, symbol: str, api_key: str) -> pd.DataFrame:
        """
        获取FinancialModelingPrep财报（Income Statement）
        Args:
                symbol: 股票代码
                api_key: FMP API密钥
        Returns:
                pd.DataFrame: 财报数据
        Raises:
                DataError: 获取失败
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?apikey={api_key}&limit=12"
            resp = requests.get(url, proxies=self.proxies, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list) or not data:
                raise DataError("No FMP data returned")
            df = pd.DataFrame(data)
            logger.info(f"Fetched FMP income statement for {symbol}, {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch FMP financials: {symbol}, error: {e}")
            raise DataError(f"FMP financials fetch failed: {e}")

    def fetch_dividend_history(self, symbol: str) -> pd.DataFrame:
        """
        获取分红历史（Yahoo Finance接口，伪实现）
        Args:
                symbol: 股票代码
        Returns:
                pd.DataFrame: 分红历史
        Raises:
                DataError: 获取失败
        """
        try:
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?events=div"
            resp = requests.get(url, proxies=self.proxies, timeout=10)
            resp.raise_for_status()
            df = pd.read_csv(pd.compat.StringIO(resp.text))
            logger.info(f"Fetched dividend history for {symbol}, {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch dividend history: {symbol}, error: {e}")
            raise DataError(f"Dividend history fetch failed: {e}")

    def validate_financials(self, df: pd.DataFrame) -> bool:
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
        # 检查常见字段
        required_cols = ["date", "revenue", "netIncome", "grossProfit"]
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
                return False
        # 检查数值合理性
        if (df["revenue"] < 0).any() or (df["netIncome"] < -1e9).any():
            logger.warning("Unreasonable values in financials")
            return False
        logger.info("Financials validation passed")
        return True
