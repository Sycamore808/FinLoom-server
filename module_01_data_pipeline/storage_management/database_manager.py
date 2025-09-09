"""
数据库管理器模块
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from common.logging_system import setup_logger
from common.exceptions import DataError

logger = setup_logger("database_manager")

class DatabaseManager:
    """数据库管理器类"""
    
    def __init__(self, db_path: str = "data/finloom.db"):
        """初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """初始化数据库表结构"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建股票基本信息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_info (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    pe_ratio REAL,
                    pb_ratio REAL,
                    dividend_yield REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # 创建股票价格数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    amount REAL,
                    pct_change REAL,
                    created_at TEXT NOT NULL,
                    UNIQUE(symbol, date)
                )
            ''')
            
            # 创建技术指标表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    sma_5 REAL,
                    sma_10 REAL,
                    sma_20 REAL,
                    sma_50 REAL,
                    ema_12 REAL,
                    ema_26 REAL,
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bb_upper REAL,
                    bb_middle REAL,
                    bb_lower REAL,
                    atr REAL,
                    stoch_k REAL,
                    stoch_d REAL,
                    created_at TEXT NOT NULL,
                    UNIQUE(symbol, date)
                )
            ''')
            
            # 创建交易信号表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # 创建回测结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    initial_capital REAL NOT NULL,
                    final_capital REAL NOT NULL,
                    total_return REAL NOT NULL,
                    annualized_return REAL NOT NULL,
                    volatility REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    equity_curve TEXT,
                    trades TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date ON stock_prices(symbol, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_date ON technical_indicators(symbol, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp ON trading_signals(timestamp)')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DataError(f"Database initialization failed: {e}")
    
    def save_stock_info(self, symbol: str, name: str, sector: str = None, 
                       industry: str = None, **kwargs):
        """保存股票基本信息
        
        Args:
            symbol: 股票代码
            name: 股票名称
            sector: 行业板块
            industry: 细分行业
            **kwargs: 其他信息
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT OR REPLACE INTO stock_info 
                (symbol, name, sector, industry, market_cap, pe_ratio, pb_ratio, 
                 dividend_yield, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, name, sector, industry,
                kwargs.get('market_cap'),
                kwargs.get('pe_ratio'),
                kwargs.get('pb_ratio'),
                kwargs.get('dividend_yield'),
                now, now
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved stock info for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to save stock info for {symbol}: {e}")
            raise DataError(f"Stock info save failed: {e}")
    
    def save_stock_prices(self, symbol: str, df: pd.DataFrame):
        """保存股票价格数据
        
        Args:
            symbol: 股票代码
            df: 价格数据DataFrame
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            for _, row in df.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO stock_prices 
                    (symbol, date, open, high, low, close, volume, amount, 
                     pct_change, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    row.get('open', 0.0),
                    row.get('high', 0.0),
                    row.get('low', 0.0),
                    row.get('close', 0.0),
                    row.get('volume', 0),
                    row.get('amount', 0.0),
                    row.get('pct_change', 0.0),
                    now
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(df)} price records for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to save stock prices for {symbol}: {e}")
            raise DataError(f"Stock prices save failed: {e}")
    
    def save_technical_indicators(self, symbol: str, df: pd.DataFrame):
        """保存技术指标数据
        
        Args:
            symbol: 股票代码
            df: 技术指标数据DataFrame
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            for _, row in df.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO technical_indicators 
                    (symbol, date, sma_5, sma_10, sma_20, sma_50, ema_12, ema_26,
                     rsi, macd, macd_signal, macd_histogram, bb_upper, bb_middle, 
                     bb_lower, atr, stoch_k, stoch_d, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    row.get('sma_5'),
                    row.get('sma_10'),
                    row.get('sma_20'),
                    row.get('sma_50'),
                    row.get('ema_12'),
                    row.get('ema_26'),
                    row.get('rsi'),
                    row.get('macd'),
                    row.get('macd_signal'),
                    row.get('macd_histogram'),
                    row.get('bb_upper'),
                    row.get('bb_middle'),
                    row.get('bb_lower'),
                    row.get('atr'),
                    row.get('stoch_k'),
                    row.get('stoch_d'),
                    now
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(df)} technical indicator records for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to save technical indicators for {symbol}: {e}")
            raise DataError(f"Technical indicators save failed: {e}")
    
    def save_trading_signal(self, signal_data: Dict[str, Any]):
        """保存交易信号
        
        Args:
            signal_data: 信号数据字典
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT OR REPLACE INTO trading_signals 
                (signal_id, symbol, strategy_name, signal_type, confidence, 
                 price, quantity, timestamp, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data['signal_id'],
                signal_data['symbol'],
                signal_data['strategy_name'],
                signal_data['signal_type'],
                signal_data['confidence'],
                signal_data['price'],
                signal_data['quantity'],
                signal_data['timestamp'],
                json.dumps(signal_data.get('metadata', {})),
                now
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved trading signal: {signal_data['signal_id']}")
            
        except Exception as e:
            logger.error(f"Failed to save trading signal: {e}")
            raise DataError(f"Trading signal save failed: {e}")
    
    def save_backtest_result(self, result_data: Dict[str, Any]):
        """保存回测结果
        
        Args:
            result_data: 回测结果数据字典
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO backtest_results 
                (strategy_name, symbol, start_date, end_date, initial_capital, 
                 final_capital, total_return, annualized_return, volatility, 
                 sharpe_ratio, max_drawdown, win_rate, profit_factor, total_trades,
                 equity_curve, trades, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result_data['strategy_name'],
                result_data['symbol'],
                result_data['start_date'],
                result_data['end_date'],
                result_data['initial_capital'],
                result_data['final_capital'],
                result_data['total_return'],
                result_data['annualized_return'],
                result_data['volatility'],
                result_data['sharpe_ratio'],
                result_data['max_drawdown'],
                result_data['win_rate'],
                result_data['profit_factor'],
                result_data['total_trades'],
                json.dumps(result_data.get('equity_curve', [])),
                json.dumps(result_data.get('trades', [])),
                now
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved backtest result for {result_data['symbol']}")
            
        except Exception as e:
            logger.error(f"Failed to save backtest result: {e}")
            raise DataError(f"Backtest result save failed: {e}")
    
    def get_stock_prices(self, symbol: str, start_date: str = None, 
                        end_date: str = None) -> pd.DataFrame:
        """获取股票价格数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            价格数据DataFrame
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM stock_prices WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            logger.info(f"Retrieved {len(df)} price records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get stock prices for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_technical_indicators(self, symbol: str, start_date: str = None, 
                               end_date: str = None) -> pd.DataFrame:
        """获取技术指标数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            技术指标数据DataFrame
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM technical_indicators WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            logger.info(f"Retrieved {len(df)} technical indicator records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get technical indicators for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_trading_signals(self, symbol: str = None, strategy_name: str = None,
                          start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """获取交易信号
        
        Args:
            symbol: 股票代码
            strategy_name: 策略名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易信号列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM trading_signals WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # 获取列名
            columns = [description[0] for description in cursor.description]
            
            # 转换为字典列表
            signals = []
            for row in rows:
                signal = dict(zip(columns, row))
                if signal['metadata']:
                    signal['metadata'] = json.loads(signal['metadata'])
                signals.append(signal)
            
            conn.close()
            logger.info(f"Retrieved {len(signals)} trading signals")
            return signals
            
        except Exception as e:
            logger.error(f"Failed to get trading signals: {e}")
            return []
    
    def get_backtest_results(self, symbol: str = None, strategy_name: str = None) -> List[Dict[str, Any]]:
        """获取回测结果
        
        Args:
            symbol: 股票代码
            strategy_name: 策略名称
            
        Returns:
            回测结果列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM backtest_results WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # 获取列名
            columns = [description[0] for description in cursor.description]
            
            # 转换为字典列表
            results = []
            for row in rows:
                result = dict(zip(columns, row))
                if result['equity_curve']:
                    result['equity_curve'] = json.loads(result['equity_curve'])
                if result['trades']:
                    result['trades'] = json.loads(result['trades'])
                results.append(result)
            
            conn.close()
            logger.info(f"Retrieved {len(results)} backtest results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get backtest results: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息
        
        Returns:
            数据库统计信息字典
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # 获取各表的记录数
            tables = ['stock_info', 'stock_prices', 'technical_indicators', 
                     'trading_signals', 'backtest_results']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[f"{table}_count"] = count
            
            # 获取数据库文件大小
            stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            
            # 获取最后更新时间
            cursor.execute("SELECT MAX(created_at) FROM stock_prices")
            last_update = cursor.fetchone()[0]
            stats['last_update'] = last_update
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

# 便捷函数
def create_database_manager(db_path: str = "data/finloom.db") -> DatabaseManager:
    """创建数据库管理器的便捷函数
    
    Args:
        db_path: 数据库文件路径
        
    Returns:
        数据库管理器实例
    """
    return DatabaseManager(db_path)

# 全局数据库管理器实例
_global_db_manager = None

def get_database_manager(db_path: str = "data/finloom.db") -> DatabaseManager:
    """获取全局数据库管理器实例（单例模式）
    
    Args:
        db_path: 数据库文件路径
        
    Returns:
        数据库管理器实例
    """
    global _global_db_manager
    if _global_db_manager is None:
        _global_db_manager = DatabaseManager(db_path)
    return _global_db_manager