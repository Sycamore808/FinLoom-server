"""
市场数据采集器模块
负责实时和历史市场数据的采集
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import numpy as np
import pandas as pd
import yfinance as yf
from common.constants import (
    DEFAULT_LOOKBACK_DAYS,
    MAX_SYMBOLS_PER_REQUEST,
    TIMEOUT_SECONDS,
)
from common.data_structures import MarketData
from common.exceptions import DataError
from common.logging_system import setup_logger

logger = setup_logger("market_data_collector")


@dataclass
class DataSource:
    """数据源配置"""

    name: str
    api_key: Optional[str]
    base_url: str
    rate_limit: int  # 每分钟请求数
    priority: int  # 优先级，数字越小优先级越高


class MarketDataCollector:
    """市场数据采集器"""

    # 数据源配置
    DATA_SOURCES = {
        "yahoo": DataSource(
            name="yahoo",
            api_key=None,
            base_url="https://query1.finance.yahoo.com",
            rate_limit=2000,
            priority=1,
        ),
        "polygon": DataSource(
            name="polygon",
            api_key=None,  # 需要设置
            base_url="https://api.polygon.io",
            rate_limit=500,
            priority=2,
        ),
        "alpha_vantage": DataSource(
            name="alpha_vantage",
            api_key=None,  # 需要设置
            base_url="https://www.alphavantage.co",
            rate_limit=500,
            priority=3,
        ),
    }

    def __init__(self):
        """初始化市场数据采集器"""
        self.active_subscriptions: Dict[str, List[str]] = {}
        self.data_cache: Dict[str, MarketData] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_streaming = False
        self.stream_callbacks: List[Callable[[MarketData], None]] = []

    async def initialize(self) -> None:
        """异步初始化

        必须在使用前调用此方法
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        logger.info("Market data collector initialized")

    async def cleanup(self) -> None:
        """清理资源"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("Market data collector cleaned up")

    def subscribe_market_data(self, symbols: List[str], frequency: str = "1m") -> bool:
        """订阅市场数据

        Args:
            symbols: 股票代码列表
            frequency: 数据频率 ('1m', '5m', '15m', '30m', '1h', '1d')

        Returns:
            是否成功订阅

        Raises:
            ValueError: 无效的频率或符号
        """
        valid_frequencies = ["1m", "5m", "15m", "30m", "1h", "1d"]
        if frequency not in valid_frequencies:
            raise ValueError(f"Invalid frequency: {frequency}")

        if len(symbols) > MAX_SYMBOLS_PER_REQUEST:
            raise ValueError(
                f"Too many symbols: {len(symbols)} > {MAX_SYMBOLS_PER_REQUEST}"
            )

        # 验证符号格式
        for symbol in symbols:
            if not self._validate_symbol(symbol):
                raise ValueError(f"Invalid symbol format: {symbol}")

        # 添加到订阅列表
        if frequency not in self.active_subscriptions:
            self.active_subscriptions[frequency] = []

        self.active_subscriptions[frequency].extend(symbols)
        self.active_subscriptions[frequency] = list(
            set(self.active_subscriptions[frequency])
        )

        logger.info(f"Subscribed to {len(symbols)} symbols at {frequency} frequency")
        return True

    async def fetch_realtime_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """获取实时数据

        Args:
            symbols: 股票代码列表

        Returns:
            符号到市场数据的映射

        Raises:
            DataError: 数据获取失败
        """
        if not self.session:
            await self.initialize()

        result = {}

        # 分批请求
        for i in range(0, len(symbols), 10):
            batch = symbols[i : i + 10]
            batch_data = await self._fetch_batch_realtime(batch)
            result.update(batch_data)

        return result

    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """获取历史数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔

        Returns:
            历史数据DataFrame

        Raises:
            DataError: 数据获取失败
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                raise DataError(f"No data found for {symbol}")

            # 标准化列名
            df.columns = [col.lower() for col in df.columns]
            df.index.name = "timestamp"

            # 添加额外字段
            df["symbol"] = symbol
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3

            logger.info(f"Fetched {len(df)} historical records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            raise DataError(f"Historical data fetch failed: {e}")

    def start_streaming(self, callback: Callable[[MarketData], None]) -> None:
        """启动数据流

        Args:
            callback: 数据回调函数
        """
        self.stream_callbacks.append(callback)
        if not self.is_streaming:
            self.is_streaming = True
            asyncio.create_task(self._streaming_loop())
            logger.info("Started market data streaming")

    def stop_streaming(self) -> None:
        """停止数据流"""
        self.is_streaming = False
        logger.info("Stopped market data streaming")

    async def _fetch_batch_realtime(self, symbols: List[str]) -> Dict[str, MarketData]:
        """批量获取实时数据

        Args:
            symbols: 股票代码列表

        Returns:
            数据字典
        """
        result = {}

        # 使用Yahoo Finance API
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                # 获取最新报价
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=info.get("regularMarketOpen", 0),
                    high=info.get("regularMarketDayHigh", 0),
                    low=info.get("regularMarketDayLow", 0),
                    close=info.get("regularMarketPrice", 0),
                    volume=info.get("regularMarketVolume", 0),
                    bid=info.get("bid", None),
                    ask=info.get("ask", None),
                    bid_size=info.get("bidSize", None),
                    ask_size=info.get("askSize", None),
                )

                result[symbol] = market_data
                self.data_cache[symbol] = market_data

            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")

        return result

    async def _streaming_loop(self) -> None:
        """数据流循环"""
        while self.is_streaming:
            try:
                # 获取所有订阅的符号
                all_symbols = set()
                for symbols in self.active_subscriptions.values():
                    all_symbols.update(symbols)

                if all_symbols:
                    # 获取数据
                    data = await self.fetch_realtime_data(list(all_symbols))

                    # 触发回调
                    for market_data in data.values():
                        for callback in self.stream_callbacks:
                            try:
                                callback(market_data)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

                # 根据最高频率决定休眠时间
                if "1m" in self.active_subscriptions:
                    await asyncio.sleep(60)
                elif "5m" in self.active_subscriptions:
                    await asyncio.sleep(300)
                else:
                    await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Streaming loop error: {e}")
                await asyncio.sleep(60)

    def _validate_symbol(self, symbol: str) -> bool:
        """验证股票代码格式

        Args:
            symbol: 股票代码

        Returns:
            是否有效
        """
        # 基本验证：1-5个大写字母
        if not symbol or len(symbol) > 5:
            return False
        return symbol.isalpha() and symbol.isupper()


# 模块级别函数
async def collect_market_data(
    symbols: List[str], lookback_days: int = DEFAULT_LOOKBACK_DAYS
) -> pd.DataFrame:
    """收集市场数据的便捷函数

    Args:
        symbols: 股票代码列表
        lookback_days: 回看天数

    Returns:
        市场数据DataFrame
    """
    collector = MarketDataCollector()
    await collector.initialize()

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        all_data = []
        for symbol in symbols:
            df = collector.fetch_historical_data(
                symbol=symbol, start_date=start_date, end_date=end_date
            )
            all_data.append(df)

        result = pd.concat(all_data, ignore_index=True)
        return result

    finally:
        await collector.cleanup()
