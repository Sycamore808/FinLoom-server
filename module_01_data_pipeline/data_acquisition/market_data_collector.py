"""
市场数据采集器模块
负责实时和历史市场数据的采集
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import aiohttp
import pandas as pd
import yfinance as yf

# 尝试导入依赖
try:
    import yfinance as yf

    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import akshare as ak

    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False

from common.constants import (
    DEFAULT_LOOKBACK_DAYS,
    MAX_SYMBOLS_PER_REQUEST,
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

    # 数据源配置，支持多市场
    DATA_SOURCES = {
        "yahoo": DataSource(
            name="yahoo",
            api_key=None,
            base_url="https://query1.finance.yahoo.com",
            rate_limit=2000,
            priority=1,
        ),
        "akshare": DataSource(
            name="akshare",
            api_key=None,
            base_url="",  # akshare不需要URL
            rate_limit=1000,
            priority=1,  # 对于中国市场优先级最高
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
        "binance": DataSource(
            name="binance",
            api_key=None,
            base_url="https://api.binance.com",
            rate_limit=1200,
            priority=4,
        ),
        "hkex": DataSource(
            name="hkex",
            api_key=None,
            base_url="https://www1.hkex.com.hk",
            rate_limit=100,
            priority=5,
        ),
    }

    def __init__(self):
        """初始化市场数据采集器"""
        self.active_subscriptions: Dict[str, List[str]] = {}
        self.data_cache: Dict[str, MarketData] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_streaming = False
        self.stream_callbacks: List[Callable[[MarketData], None]] = []

        # 检查依赖
        if not HAS_YFINANCE:
            logger.warning("yfinance not available. US market data will be limited.")
        if not HAS_AKSHARE:
            logger.warning(
                "akshare not available. Chinese market data will be limited."
            )

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

    async def fetch_realtime_data(
        self, symbols: List[str], market: str = "US"
    ) -> Dict[str, MarketData]:
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
            try:
                batch_data = await self._fetch_batch_realtime(batch, market=market)
                result.update(batch_data)
            except Exception as e:
                logger.error(f"Batch realtime fetch failed for {batch}: {e}")

        return result

    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        market: str = "US",
    ) -> pd.DataFrame:
        """获取历史数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔
            market: 市场类型 ("US", "CN", "HK")

        Returns:
            历史数据DataFrame

        Raises:
            DataError: 数据获取失败
        """
        try:
            if market == "CN" and HAS_AKSHARE:
                # 使用akshare获取A股数据
                start_str = start_date.strftime("%Y%m%d")
                end_str = end_date.strftime("%Y%m%d")

                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_str,
                    end_date=end_str,
                    adjust="qfq",
                )

                if df.empty:
                    raise DataError(f"No data found for {symbol}")

                # 标准化列名
                column_mapping = {
                    "日期": "timestamp",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                }
                df = df.rename(columns=column_mapping)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)

            elif market in ["US", "HK"] and HAS_YFINANCE:
                # 使用yfinance获取美股/港股数据
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)

                if df.empty:
                    raise DataError(f"No data found for {symbol}")

                # 标准化列名
                df.columns = [col.lower() for col in df.columns]
                df.index.name = "timestamp"

            else:
                # 回退到yfinance（如果可用）
                if HAS_YFINANCE:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(
                        start=start_date, end=end_date, interval=interval
                    )

                    if df.empty:
                        raise DataError(f"No data found for {symbol}")

                    # 标准化列名
                    df.columns = [col.lower() for col in df.columns]
                    df.index.name = "timestamp"
                else:
                    raise DataError("No suitable data source available")

            # 添加额外字段
            df["symbol"] = symbol
            if "high" in df.columns and "low" in df.columns and "close" in df.columns:
                df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3

            logger.info(f"Fetched {len(df)} historical records for {symbol} ({market})")
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

    async def _fetch_batch_realtime(
        self, symbols: List[str], market: str = "US"
    ) -> Dict[str, MarketData]:
        """批量获取实时数据

        Args:
            symbols: 股票代码列表
            market: 市场类型 ("US", "CN", "HK", "CRYPTO")

        Returns:
            数据字典
        """
        result = {}

        for symbol in symbols:
            try:
                if market == "CN" and HAS_AKSHARE:
                    # 使用akshare获取A股实时数据
                    realtime_data = ak.stock_zh_a_spot_em()
                    symbol_data = realtime_data[realtime_data["代码"] == symbol]

                    if not symbol_data.empty:
                        row = symbol_data.iloc[0]
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            open=float(row.get("今开", 0)),
                            high=float(row.get("最高", 0)),
                            low=float(row.get("最低", 0)),
                            close=float(row.get("最新价", 0)),
                            volume=int(row.get("成交量", 0)),
                            bid=None,
                            ask=None,
                            bid_size=None,
                            ask_size=None,
                        )
                    else:
                        continue

                elif market == "US" and HAS_YFINANCE:
                    # 使用yfinance获取美股数据
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
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
                elif market == "CRYPTO":
                    # 示例：Binance现货API
                    import requests

                    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        open=float(data.get("openPrice", 0)),
                        high=float(data.get("highPrice", 0)),
                        low=float(data.get("lowPrice", 0)),
                        close=float(data.get("lastPrice", 0)),
                        volume=int(float(data.get("volume", 0))),
                        bid=float(data.get("bidPrice", 0)),
                        ask=float(data.get("askPrice", 0)),
                        bid_size=int(float(data.get("bidQty", 0))),
                        ask_size=int(float(data.get("askQty", 0))),
                    )
                elif market == "HK":
                    # 港股数据，尝试使用yfinance（添加.HK后缀）
                    if HAS_YFINANCE:
                        hk_symbol = (
                            f"{symbol}.HK" if not symbol.endswith(".HK") else symbol
                        )
                        ticker = yf.Ticker(hk_symbol)
                        info = ticker.info
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            open=info.get("regularMarketOpen", 0),
                            high=info.get("regularMarketDayHigh", 0),
                            low=info.get("regularMarketDayLow", 0),
                            close=info.get("regularMarketPrice", 0),
                            volume=info.get("regularMarketVolume", 0),
                        )
                    else:
                        # 伪实现
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            open=0,
                            high=0,
                            low=0,
                            close=0,
                            volume=0,
                        )
                else:
                    raise DataError(f"Unsupported market: {market}")

                result[symbol] = market_data
                self.data_cache[symbol] = market_data
                logger.info(f"Fetched realtime data for {symbol} ({market})")
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol} ({market}): {e}")
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
        # 基本验证：不为空且不超过10个字符
        if not symbol or len(symbol) > 10:
            return False
        return True

    def _detect_market(self, symbol: str) -> str:
        """根据股票代码自动检测市场类型

        Args:
            symbol: 股票代码

        Returns:
            市场类型 ("CN", "US", "HK", "CRYPTO")
        """
        symbol = symbol.upper()

        # A股：6位数字
        if symbol.isdigit() and len(symbol) == 6:
            return "CN"

        # 港股：以.HK结尾或以0-9开头的4-5位数字
        if symbol.endswith(".HK") or (symbol.isdigit() and len(symbol) in [4, 5]):
            return "HK"

        # 加密货币：常见的加密货币代码
        crypto_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"]
        if symbol in crypto_symbols or "USDT" in symbol:
            return "CRYPTO"

        # 默认为美股
        return "US"


# 模块级别函数
async def collect_market_data(
    symbols: List[str], lookback_days: int = DEFAULT_LOOKBACK_DAYS, market: str = "AUTO"
) -> pd.DataFrame:
    """收集市场数据的便捷函数

    Args:
        symbols: 股票代码列表
        lookback_days: 回看天数
        market: 市场类型，"AUTO"为自动检测

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
            # 自动检测市场类型
            if market == "AUTO":
                detected_market = collector._detect_market(symbol)
            else:
                detected_market = market

            df = collector.fetch_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                market=detected_market,
            )
            all_data.append(df)

        result = pd.concat(all_data, ignore_index=True)
        return result

    finally:
        await collector.cleanup()


async def collect_realtime_data(
    symbols: List[str], market: str = "AUTO"
) -> Dict[str, MarketData]:
    """收集实时数据的便捷函数

    Args:
        symbols: 股票代码列表
        market: 市场类型，"AUTO"为自动检测

    Returns:
        实时数据字典
    """
    collector = MarketDataCollector()
    await collector.initialize()

    try:
        # 按市场分组
        market_groups = {}
        for symbol in symbols:
            if market == "AUTO":
                detected_market = collector._detect_market(symbol)
            else:
                detected_market = market

            if detected_market not in market_groups:
                market_groups[detected_market] = []
            market_groups[detected_market].append(symbol)

        # 分市场获取数据
        all_data = {}
        for market_type, market_symbols in market_groups.items():
            market_data = await collector.fetch_realtime_data(
                market_symbols, market_type
            )
            all_data.update(market_data)

        return all_data

    finally:
        await collector.cleanup()
