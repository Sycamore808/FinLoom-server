"""
数据库管理模块
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np

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
        self.db_path = db_path
        self.connection = None
        
        # 确保数据目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
    def _init_database(self):
        """初始化数据库表结构"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            cursor = self.connection.cursor()
            
            # 创建市场数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    vwap REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            # 创建技术指标表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    indicator_name TEXT NOT NULL,
                    indicator_value REAL NOT NULL,
                    parameters TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, indicator_name)
                )
            ''')
            
            # 创建交易信号表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    confidence REAL NOT NULL,
                    strategy_name TEXT NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建持仓表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    current_price REAL NOT NULL,
                    market_value REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    open_time DATETIME NOT NULL,
                    last_update DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建回测结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id TEXT UNIQUE NOT NULL,
                    strategy_name TEXT NOT NULL,
                    start_date DATETIME NOT NULL,
                    end_date DATETIME NOT NULL,
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
                    performance_metrics TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_time ON technical_indicators(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_time ON trading_signals(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)')
            
            self.connection.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DataError(f"Database initialization failed: {e}")
    
    def insert_market_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """插入市场数据
        
        Args:
            data: 市场数据DataFrame
            symbol: 股票代码
            
        Returns:
            是否插入成功
        """
        try:
            cursor = self.connection.cursor()
            
            # 准备数据
            records = []
            for timestamp, row in data.iterrows():
                # 转换时间戳为字符串格式
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp)
                record = (
                    symbol,
                    timestamp_str,
                    row.get('open', 0.0),
                    row.get('high', 0.0),
                    row.get('low', 0.0),
                    row.get('close', 0.0),
                    row.get('volume', 0),
                    row.get('vwap', None)
                )
                records.append(record)
            
            # 批量插入
            cursor.executemany('''
                INSERT OR REPLACE INTO market_data 
                (symbol, timestamp, open, high, low, close, volume, vwap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            
            self.connection.commit()
            logger.info(f"Inserted {len(records)} market data records for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert market data for {symbol}: {e}")
            return False
    
    def get_market_data(
        self, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """获取市场数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            市场数据DataFrame
        """
        try:
            query = "SELECT * FROM market_data WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, self.connection, params=params, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Retrieved {len(df)} market data records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def insert_technical_indicators(
        self, 
        symbol: str, 
        indicators: Dict[str, pd.Series]
    ) -> bool:
        """插入技术指标数据
        
        Args:
            symbol: 股票代码
            indicators: 技术指标字典
            
        Returns:
            是否插入成功
        """
        try:
            cursor = self.connection.cursor()
            
            records = []
            for indicator_name, series in indicators.items():
                for timestamp, value in series.dropna().items():
                    record = (
                        symbol,
                        timestamp,
                        indicator_name,
                        float(value),
                        None  # parameters
                    )
                    records.append(record)
            
            # 批量插入
            cursor.executemany('''
                INSERT OR REPLACE INTO technical_indicators 
                (symbol, timestamp, indicator_name, indicator_value, parameters)
                VALUES (?, ?, ?, ?, ?)
            ''', records)
            
            self.connection.commit()
            logger.info(f"Inserted {len(records)} technical indicator records for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert technical indicators for {symbol}: {e}")
            return False
    
    def get_technical_indicators(
        self, 
        symbol: str, 
        indicator_names: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """获取技术指标数据
        
        Args:
            symbol: 股票代码
            indicator_names: 指标名称列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            技术指标DataFrame
        """
        try:
            query = "SELECT * FROM technical_indicators WHERE symbol = ?"
            params = [symbol]
            
            if indicator_names:
                placeholders = ','.join(['?' for _ in indicator_names])
                query += f" AND indicator_name IN ({placeholders})"
                params.extend(indicator_names)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp, indicator_name"
            
            df = pd.read_sql_query(query, self.connection, params=params, parse_dates=['timestamp'])
            
            if not df.empty:
                # 透视表，将指标名称作为列
                df = df.pivot_table(
                    index='timestamp', 
                    columns='indicator_name', 
                    values='indicator_value'
                )
            
            logger.info(f"Retrieved technical indicators for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get technical indicators for {symbol}: {e}")
            return pd.DataFrame()
    
    def insert_trading_signal(self, signal_data: Dict[str, Any]) -> bool:
        """插入交易信号
        
        Args:
            signal_data: 信号数据字典
            
        Returns:
            是否插入成功
        """
        try:
            cursor = self.connection.cursor()
            
            record = (
                signal_data['signal_id'],
                signal_data['symbol'],
                signal_data['timestamp'],
                signal_data['action'],
                signal_data['quantity'],
                signal_data['price'],
                signal_data['confidence'],
                signal_data['strategy_name'],
                str(signal_data.get('metadata', {}))
            )
            
            cursor.execute('''
                INSERT OR REPLACE INTO trading_signals 
                (signal_id, symbol, timestamp, action, quantity, price, confidence, strategy_name, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', record)
            
            self.connection.commit()
            logger.info(f"Inserted trading signal: {signal_data['signal_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert trading signal: {e}")
            return False
    
    def get_trading_signals(
        self, 
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """获取交易信号
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易信号DataFrame
        """
        try:
            query = "SELECT * FROM trading_signals WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, self.connection, params=params, parse_dates=['timestamp'])
            
            logger.info(f"Retrieved {len(df)} trading signals")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get trading signals: {e}")
            return pd.DataFrame()
    
    def insert_backtest_result(self, result_data: Dict[str, Any]) -> bool:
        """插入回测结果
        
        Args:
            result_data: 回测结果数据
            
        Returns:
            是否插入成功
        """
        try:
            cursor = self.connection.cursor()
            
            record = (
                result_data['backtest_id'],
                result_data['strategy_name'],
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
                str(result_data.get('performance_metrics', {}))
            )
            
            cursor.execute('''
                INSERT OR REPLACE INTO backtest_results 
                (backtest_id, strategy_name, start_date, end_date, initial_capital, 
                 final_capital, total_return, annualized_return, volatility, 
                 sharpe_ratio, max_drawdown, win_rate, profit_factor, total_trades, performance_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', record)
            
            self.connection.commit()
            logger.info(f"Inserted backtest result: {result_data['backtest_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert backtest result: {e}")
            return False
    
    def get_backtest_results(
        self, 
        strategy_name: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """获取回测结果
        
        Args:
            strategy_name: 策略名称
            limit: 限制数量
            
        Returns:
            回测结果DataFrame
        """
        try:
            query = "SELECT * FROM backtest_results WHERE 1=1"
            params = []
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            df = pd.read_sql_query(query, self.connection, params=params, parse_dates=['start_date', 'end_date', 'created_at'])
            
            logger.info(f"Retrieved {len(df)} backtest results")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get backtest results: {e}")
            return pd.DataFrame()
    
    def execute_query(self, query: str, params: Optional[List] = None) -> pd.DataFrame:
        """执行自定义查询
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果DataFrame
        """
        try:
            df = pd.read_sql_query(query, self.connection, params=params or [])
            logger.info(f"Executed query, returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return pd.DataFrame()
    
    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

# 全局数据库管理器实例
_global_db_manager: Optional[DatabaseManager] = None

def get_database_manager(db_path: str = "data/finloom.db") -> DatabaseManager:
    """获取全局数据库管理器实例
    
    Args:
        db_path: 数据库文件路径
        
    Returns:
        数据库管理器实例
    """
    global _global_db_manager
    if _global_db_manager is None:
        _global_db_manager = DatabaseManager(db_path)
    return _global_db_manager
