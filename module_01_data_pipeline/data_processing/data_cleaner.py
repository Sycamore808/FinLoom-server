"""
数据清理模块
负责清理和预处理市场数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings

from common.logging_system import setup_logger
from common.exceptions import DataError

logger = setup_logger("data_cleaner")

class DataCleaner:
    """数据清理器类"""
    
    def __init__(self, 
                 fill_method: str = "forward",
                 outlier_method: str = "iqr",
                 outlier_threshold: float = 3.0):
        """初始化数据清理器
        
        Args:
            fill_method: 缺失值填充方法 ("forward", "backward", "interpolate", "drop")
            outlier_method: 异常值检测方法 ("iqr", "zscore", "modified_zscore")
            outlier_threshold: 异常值阈值
        """
        self.fill_method = fill_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
    def clean_market_data(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """清理市场数据
        
        Args:
            df: 原始市场数据DataFrame
            symbol: 股票代码（用于日志）
            
        Returns:
            清理后的DataFrame
        """
        try:
            if df.empty:
                logger.warning(f"Empty DataFrame provided for {symbol}")
                return df
                
            original_length = len(df)
            logger.info(f"Starting data cleaning for {symbol}: {original_length} records")
            
            # 1. 标准化列名
            df = self._standardize_columns(df)
            
            # 2. 数据类型转换
            df = self._convert_data_types(df)
            
            # 3. 处理重复数据
            df = self._remove_duplicates(df)
            
            # 4. 处理异常值
            df = self._handle_outliers(df)
            
            # 5. 处理缺失值
            df = self._handle_missing_values(df)
            
            # 6. 数据验证
            df = self._validate_data(df)
            
            final_length = len(df)
            logger.info(f"Data cleaning completed for {symbol}: {original_length} -> {final_length} records")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to clean data for {symbol}: {e}")
            raise DataError(f"Data cleaning failed: {e}")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        # 列名映射
        column_mapping = {
            '日期': 'date',
            '时间': 'time',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover',
            '市盈率': 'pe_ratio',
            '市净率': 'pb_ratio',
            '市值': 'market_cap'
        }
        
        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # 确保日期列存在
        if 'date' not in df.columns and 'time' in df.columns:
            df['date'] = df['time']
        elif 'date' not in df.columns and df.index.name == 'date':
            df = df.reset_index()
            
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型"""
        # 转换日期列
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        # 转换数值列
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 
                          'amplitude', 'pct_change', 'change', 'turnover', 
                          'pe_ratio', 'pb_ratio', 'market_cap']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """移除重复数据"""
        if 'date' in df.columns:
            # 基于日期去重
            df = df.drop_duplicates(subset=['date'], keep='last')
        else:
            # 基于索引去重
            df = df[~df.index.duplicated(keep='last')]
            
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            if col not in df.columns:
                continue
                
            if self.outlier_method == "iqr":
                df = self._remove_outliers_iqr(df, col)
            elif self.outlier_method == "zscore":
                df = self._remove_outliers_zscore(df, col)
            elif self.outlier_method == "modified_zscore":
                df = self._remove_outliers_modified_zscore(df, col)
                
        return df
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """使用IQR方法移除异常值"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        # 标记异常值
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            logger.warning(f"Found {outlier_count} outliers in {column} using IQR method")
            # 可以选择删除或替换异常值
            df.loc[outlier_mask, column] = np.nan
            
        return df
    
    def _remove_outliers_zscore(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """使用Z-score方法移除异常值"""
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outlier_mask = z_scores > self.outlier_threshold
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            logger.warning(f"Found {outlier_count} outliers in {column} using Z-score method")
            df.loc[outlier_mask, column] = np.nan
            
        return df
    
    def _remove_outliers_modified_zscore(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """使用修正Z-score方法移除异常值"""
        median = df[column].median()
        mad = np.median(np.abs(df[column] - median))
        
        if mad == 0:
            return df
            
        modified_z_scores = 0.6745 * (df[column] - median) / mad
        outlier_mask = np.abs(modified_z_scores) > self.outlier_threshold
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            logger.warning(f"Found {outlier_count} outliers in {column} using modified Z-score method")
            df.loc[outlier_mask, column] = np.nan
            
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        if self.fill_method == "drop":
            df = df.dropna()
        elif self.fill_method == "forward":
            df = df.fillna(method='ffill')
        elif self.fill_method == "backward":
            df = df.fillna(method='bfill')
        elif self.fill_method == "interpolate":
            # 对价格数据进行线性插值
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    df[col] = df[col].interpolate(method='linear')
            
            # 对成交量数据使用前向填充
            volume_columns = ['volume', 'amount']
            for col in volume_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill')
                    
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据验证"""
        # 检查价格数据的逻辑性
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # 最高价应该大于等于开盘价和收盘价
            invalid_high = (df['high'] < df['open']) | (df['high'] < df['close'])
            if invalid_high.any():
                logger.warning(f"Found {invalid_high.sum()} records with invalid high price")
                df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close']].max(axis=1)
            
            # 最低价应该小于等于开盘价和收盘价
            invalid_low = (df['low'] > df['open']) | (df['low'] > df['close'])
            if invalid_low.any():
                logger.warning(f"Found {invalid_low.sum()} records with invalid low price")
                df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close']].min(axis=1)
        
        # 检查负值
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_columns:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    logger.warning(f"Found {negative_count} negative values in {col}")
                    df.loc[df[col] < 0, col] = np.nan
                    
        return df
    
    def detect_data_quality_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检测数据质量问题
        
        Args:
            df: 数据DataFrame
            
        Returns:
            数据质量报告
        """
        report = {
            "total_records": len(df),
            "missing_values": {},
            "duplicates": 0,
            "outliers": {},
            "data_types": {},
            "date_range": {},
            "quality_score": 0.0
        }
        
        # 检查缺失值
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df) * 100
            report["missing_values"][col] = {
                "count": missing_count,
                "percentage": missing_pct
            }
        
        # 检查重复值
        if 'date' in df.columns:
            report["duplicates"] = df.duplicated(subset=['date']).sum()
        else:
            report["duplicates"] = df.duplicated().sum()
        
        # 检查异常值
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                report["outliers"][col] = outliers
        
        # 数据类型
        for col in df.columns:
            report["data_types"][col] = str(df[col].dtype)
        
        # 日期范围
        if 'date' in df.columns:
            report["date_range"] = {
                "start": df['date'].min(),
                "end": df['date'].max(),
                "span_days": (df['date'].max() - df['date'].min()).days
            }
        
        # 计算质量分数
        quality_score = 100.0
        
        # 缺失值扣分
        for col, info in report["missing_values"].items():
            if info["percentage"] > 10:
                quality_score -= info["percentage"] * 0.5
        
        # 重复值扣分
        if report["duplicates"] > 0:
            quality_score -= (report["duplicates"] / report["total_records"]) * 100
        
        # 异常值扣分
        total_outliers = sum(report["outliers"].values())
        if total_outliers > 0:
            quality_score -= (total_outliers / report["total_records"]) * 50
        
        report["quality_score"] = max(0.0, quality_score)
        
        return report
    
    def clean_multiple_symbols(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """批量清理多个股票的数据
        
        Args:
            data_dict: 股票数据字典
            
        Returns:
            清理后的数据字典
        """
        cleaned_data = {}
        
        for symbol, df in data_dict.items():
            try:
                cleaned_df = self.clean_market_data(df, symbol)
                cleaned_data[symbol] = cleaned_df
            except Exception as e:
                logger.error(f"Failed to clean data for {symbol}: {e}")
                # 保留原始数据
                cleaned_data[symbol] = df
                
        return cleaned_data

# 便捷函数
def create_data_cleaner(fill_method: str = "forward", 
                       outlier_method: str = "iqr",
                       outlier_threshold: float = 3.0) -> DataCleaner:
    """创建数据清理器
    
    Args:
        fill_method: 缺失值填充方法
        outlier_method: 异常值检测方法
        outlier_threshold: 异常值阈值
        
    Returns:
        数据清理器实例
    """
    return DataCleaner(fill_method, outlier_method, outlier_threshold)

def quick_clean_data(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """快速清理数据的便捷函数
    
    Args:
        df: 原始数据DataFrame
        symbol: 股票代码
        
    Returns:
        清理后的DataFrame
    """
    cleaner = DataCleaner()
    return cleaner.clean_market_data(df, symbol)
