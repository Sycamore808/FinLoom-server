"""
Akshare数据收集器模块
负责从Akshare获取中国金融市场数据
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import time

# 尝试导入可选依赖
try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from common.data_structures import MarketData
from common.logging_system import setup_logger
from common.exceptions import DataError

logger = setup_logger("akshare_collector")

class AkshareDataCollector:
    """Akshare数据收集器类"""
    
    def __init__(self, rate_limit: float = 0.1):
        """初始化数据收集器
        
        Args:
            rate_limit: 请求间隔（秒）
        """
        self.rate_limit = rate_limit
        self.last_request_time = 0.0
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
            
    def _rate_limit_check(self):
        """检查并执行速率限制"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
        
    def fetch_stock_list(self, market: str = "A股") -> pd.DataFrame:
        """获取股票列表
        
        Args:
            market: 市场类型 ("A股", "港股", "美股")
            
        Returns:
            股票列表DataFrame
        """
        try:
            if not HAS_AKSHARE:
                logger.warning("Akshare not available, returning mock data")
                # 返回模拟数据
                return pd.DataFrame({
                    'code': ['000001', '000002', '000003'],
                    'name': ['平安银行', '万科A', '国农科技']
                })
            
            self._rate_limit_check()
            
            if market == "A股":
                # 获取A股股票列表
                stock_list = ak.stock_info_a_code_name()
                logger.info(f"Fetched {len(stock_list)} A-share stocks")
                return stock_list
            elif market == "港股":
                # 获取港股股票列表
                stock_list = ak.stock_hk_spot()
                logger.info(f"Fetched {len(stock_list)} Hong Kong stocks")
                return stock_list
            elif market == "美股":
                # 获取美股股票列表
                stock_list = ak.stock_us_spot_em()
                logger.info(f"Fetched {len(stock_list)} US stocks")
                return stock_list
            else:
                raise ValueError(f"Unsupported market: {market}")
                
        except Exception as e:
            logger.error(f"Failed to fetch stock list for {market}: {e}")
            raise DataError(f"Stock list fetch failed: {e}")
            
    def fetch_stock_history(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        period: str = "daily",
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """获取股票历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            period: 周期 ("daily", "weekly", "monthly")
            adjust: 复权类型 ("qfq", "hfq", "")
            
        Returns:
            历史数据DataFrame
        """
        try:
            if not HAS_AKSHARE:
                logger.warning("Akshare not available, returning empty DataFrame")
                return pd.DataFrame()
            
            self._rate_limit_check()
            
            # 转换日期格式
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            
            # 获取历史数据
            if period == "daily":
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )
            elif period == "weekly":
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="weekly",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )
            elif period == "monthly":
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="monthly",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )
            else:
                raise ValueError(f"Unsupported period: {period}")
                
            if df.empty:
                logger.warning(f"No data found for {symbol} from {start_date} to {end_date}")
                return df
                
            # 标准化列名
            df = self._standardize_columns(df)
            
            # 添加股票代码
            df['symbol'] = symbol
            
            logger.info(f"Fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch history for {symbol}: {e}")
            # 返回空DataFrame而不是抛出异常
            return pd.DataFrame()
            
    def fetch_realtime_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取实时数据
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            实时数据字典
        """
        try:
            self._rate_limit_check()
            
            # 获取实时行情
            realtime_data = ak.stock_zh_a_spot_em()
            
            # 筛选指定股票
            if symbols:
                realtime_data = realtime_data[realtime_data['代码'].isin(symbols)]
                
            # 转换为字典格式
            result = {}
            for _, row in realtime_data.iterrows():
                symbol = row['代码']
                result[symbol] = {
                    'symbol': symbol,
                    'name': row.get('名称', ''),
                    'price': row.get('最新价', 0.0),
                    'change': row.get('涨跌幅', 0.0),
                    'change_amount': row.get('涨跌额', 0.0),
                    'volume': row.get('成交量', 0),
                    'amount': row.get('成交额', 0.0),
                    'high': row.get('最高', 0.0),
                    'low': row.get('最低', 0.0),
                    'open': row.get('今开', 0.0),
                    'close': row.get('昨收', 0.0),
                    'timestamp': datetime.now()
                }
                
            logger.info(f"Fetched realtime data for {len(result)} stocks")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch realtime data: {e}")
            raise DataError(f"Realtime data fetch failed: {e}")
            
    def fetch_financial_data(self, symbol: str, report_type: str = "资产负债表") -> pd.DataFrame:
        """获取财务数据
        
        Args:
            symbol: 股票代码
            report_type: 报表类型 ("资产负债表", "利润表", "现金流量表")
            
        Returns:
            财务数据DataFrame
        """
        try:
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
            raise DataError(f"Financial data fetch failed for {symbol}: {e}")
            
    def fetch_industry_data(self, industry: str = "全部") -> pd.DataFrame:
        """获取行业数据
        
        Args:
            industry: 行业名称
            
        Returns:
            行业数据DataFrame
        """
        try:
            self._rate_limit_check()
            
            # 获取行业板块数据
            df = ak.stock_board_industry_cons_em()
            
            if industry != "全部":
                df = df[df['板块名称'].str.contains(industry, na=False)]
                
            logger.info(f"Fetched industry data for {industry}: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch industry data for {industry}: {e}")
            raise DataError(f"Industry data fetch failed: {e}")
            
    def fetch_macro_data(self, indicator: str = "GDP") -> pd.DataFrame:
        """获取宏观经济数据
        
        Args:
            indicator: 指标名称 ("GDP", "CPI", "PPI", "PMI")
            
        Returns:
            宏观经济数据DataFrame
        """
        try:
            self._rate_limit_check()
            
            if indicator == "GDP":
                df = ak.macro_china_gdp()
            elif indicator == "CPI":
                df = ak.macro_china_cpi()
            elif indicator == "PPI":
                df = ak.macro_china_ppi()
            elif indicator == "PMI":
                df = ak.macro_china_pmi()
            else:
                raise ValueError(f"Unsupported macro indicator: {indicator}")
                
            logger.info(f"Fetched {indicator} data: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {indicator} data: {e}")
            raise DataError(f"Macro data fetch failed: {e}")
            
    def fetch_news_data(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取新闻数据
        
        Args:
            symbol: 股票代码（可选）
            limit: 获取数量限制
            
        Returns:
            新闻数据列表
        """
        try:
            self._rate_limit_check()
            
            # 获取财经新闻
            df = ak.stock_news_em(symbol=symbol)
            
            # 限制数量
            if len(df) > limit:
                df = df.head(limit)
                
            # 转换为字典列表
            news_list = []
            for _, row in df.iterrows():
                news_list.append({
                    'title': row.get('新闻标题', ''),
                    'content': row.get('新闻内容', ''),
                    'publish_time': row.get('发布时间', ''),
                    'source': row.get('新闻来源', ''),
                    'url': row.get('新闻链接', ''),
                    'symbol': symbol
                })
                
            logger.info(f"Fetched {len(news_list)} news items")
            return news_list
            
        except Exception as e:
            logger.error(f"Failed to fetch news data: {e}")
            raise DataError(f"News data fetch failed: {e}")
            
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名
        
        Args:
            df: 原始DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        # 列名映射
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        }
        
        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # 确保日期列是datetime类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        # 确保数值列是float类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
        
    def convert_to_market_data(self, df: pd.DataFrame, symbol: str) -> List[MarketData]:
        """将DataFrame转换为MarketData对象列表
        
        Args:
            df: 数据DataFrame
            symbol: 股票代码
            
        Returns:
            MarketData对象列表
        """
        market_data_list = []
        
        for _, row in df.iterrows():
            try:
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=row.get('date', datetime.now()),
                    open=float(row.get('open', 0.0)),
                    high=float(row.get('high', 0.0)),
                    low=float(row.get('low', 0.0)),
                    close=float(row.get('close', 0.0)),
                    volume=int(row.get('volume', 0)),
                    vwap=float(row.get('amount', 0.0)) / max(int(row.get('volume', 1)), 1) if row.get('volume', 0) > 0 else None
                )
                market_data_list.append(market_data)
            except Exception as e:
                logger.warning(f"Failed to convert row to MarketData: {e}")
                continue
                
        return market_data_list
        
    async def fetch_multiple_stocks(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        max_concurrent: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """并发获取多只股票数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            max_concurrent: 最大并发数
            
        Returns:
            股票数据字典
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_single_stock(symbol: str) -> tuple:
            async with semaphore:
                try:
                    df = self.fetch_stock_history(symbol, start_date, end_date)
                    return symbol, df
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
                    return symbol, pd.DataFrame()
                    
        # 创建任务
        tasks = [fetch_single_stock(symbol) for symbol in symbols]
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 整理结果
        stock_data = {}
        for result in results:
            if isinstance(result, tuple):
                symbol, df = result
                stock_data[symbol] = df
            else:
                logger.error(f"Task failed with exception: {result}")
                
        logger.info(f"Fetched data for {len(stock_data)} stocks")
        return stock_data

# 便捷函数
def create_akshare_collector(rate_limit: float = 0.1) -> AkshareDataCollector:
    """创建Akshare数据收集器
    
    Args:
        rate_limit: 请求间隔
        
    Returns:
        数据收集器实例
    """
    return AkshareDataCollector(rate_limit=rate_limit)

async def fetch_stock_data_batch(
    symbols: List[str], 
    start_date: str, 
    end_date: str,
    rate_limit: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """批量获取股票数据的便捷函数
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        rate_limit: 请求间隔
        
    Returns:
        股票数据字典
    """
    async with AkshareDataCollector(rate_limit=rate_limit) as collector:
        return await collector.fetch_multiple_stocks(symbols, start_date, end_date)