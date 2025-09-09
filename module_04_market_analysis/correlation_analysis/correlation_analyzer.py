"""
相关性分析模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from common.logging_system import setup_logger
from common.exceptions import DataError

logger = setup_logger("correlation_analyzer")

@dataclass
class CorrelationResult:
    """相关性分析结果"""
    correlation_matrix: pd.DataFrame
    significant_correlations: List[Dict[str, Any]]
    average_correlation: float
    max_correlation: float
    min_correlation: float
    analysis_timestamp: pd.Timestamp

class CorrelationAnalyzer:
    """相关性分析器类"""
    
    def __init__(self, significance_threshold: float = 0.7):
        """初始化相关性分析器
        
        Args:
            significance_threshold: 显著性阈值
        """
        self.significance_threshold = significance_threshold
        
    def calculate_correlation_matrix(
        self, 
        data: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """计算相关性矩阵
        
        Args:
            data: 价格数据 (股票 x 时间)
            method: 相关性计算方法 ('pearson', 'spearman', 'kendall')
            
        Returns:
            相关性矩阵
        """
        try:
            # 计算收益率
            returns = data.pct_change().dropna()
            
            # 计算相关性矩阵
            correlation_matrix = returns.corr(method=method)
            
            logger.info(f"Calculated {method} correlation matrix for {len(correlation_matrix)} assets")
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Failed to calculate correlation matrix: {e}")
            raise DataError(f"Correlation matrix calculation failed: {e}")
    
    def find_highly_correlated_pairs(
        self, 
        correlation_matrix: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """找出高相关性股票对
        
        Args:
            correlation_matrix: 相关性矩阵
            threshold: 相关性阈值
            
        Returns:
            高相关性股票对列表
        """
        if threshold is None:
            threshold = self.significance_threshold
            
        try:
            highly_correlated = []
            
            # 获取上三角矩阵（避免重复）
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # 找出高相关性对
            for i in range(len(upper_triangle.columns)):
                for j in range(i+1, len(upper_triangle.columns)):
                    corr_value = upper_triangle.iloc[i, j]
                    
                    if not pd.isna(corr_value) and abs(corr_value) >= threshold:
                        asset1 = upper_triangle.columns[i]
                        asset2 = upper_triangle.columns[j]
                        
                        highly_correlated.append({
                            'asset1': asset1,
                            'asset2': asset2,
                            'correlation': corr_value,
                            'abs_correlation': abs(corr_value),
                            'type': 'positive' if corr_value > 0 else 'negative'
                        })
            
            # 按绝对相关性排序
            highly_correlated.sort(key=lambda x: x['abs_correlation'], reverse=True)
            
            logger.info(f"Found {len(highly_correlated)} highly correlated pairs (threshold: {threshold})")
            return highly_correlated
            
        except Exception as e:
            logger.error(f"Failed to find highly correlated pairs: {e}")
            raise DataError(f"High correlation analysis failed: {e}")
    
    def analyze_correlation_stability(
        self, 
        data: pd.DataFrame,
        window_size: int = 252,
        step_size: int = 21
    ) -> pd.DataFrame:
        """分析相关性稳定性
        
        Args:
            data: 价格数据
            window_size: 滚动窗口大小
            step_size: 步长
            
        Returns:
            相关性稳定性分析结果
        """
        try:
            returns = data.pct_change().dropna()
            
            # 滚动计算相关性
            rolling_correlations = []
            timestamps = []
            
            for i in range(0, len(returns) - window_size + 1, step_size):
                window_data = returns.iloc[i:i + window_size]
                corr_matrix = window_data.corr()
                
                # 计算平均相关性
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                avg_correlation = upper_triangle.stack().mean()
                
                rolling_correlations.append(avg_correlation)
                timestamps.append(returns.index[i + window_size - 1])
            
            # 创建结果DataFrame
            stability_df = pd.DataFrame({
                'timestamp': timestamps,
                'avg_correlation': rolling_correlations
            })
            stability_df.set_index('timestamp', inplace=True)
            
            # 计算稳定性指标
            stability_df['correlation_std'] = stability_df['avg_correlation'].rolling(10).std()
            stability_df['correlation_trend'] = stability_df['avg_correlation'].rolling(10).mean()
            
            logger.info(f"Analyzed correlation stability over {len(stability_df)} periods")
            return stability_df
            
        except Exception as e:
            logger.error(f"Failed to analyze correlation stability: {e}")
            raise DataError(f"Correlation stability analysis failed: {e}")
    
    def detect_correlation_breakdown(
        self, 
        data: pd.DataFrame,
        threshold: float = 0.3,
        window_size: int = 60
    ) -> List[Dict[str, Any]]:
        """检测相关性破裂
        
        Args:
            data: 价格数据
            threshold: 破裂阈值
            window_size: 检测窗口大小
            
        Returns:
            相关性破裂事件列表
        """
        try:
            returns = data.pct_change().dropna()
            breakdown_events = []
            
            # 计算长期平均相关性
            long_term_corr = returns.corr()
            
            # 滚动检测
            for i in range(window_size, len(returns)):
                window_data = returns.iloc[i-window_size:i]
                window_corr = window_data.corr()
                
                # 比较相关性变化
                for asset1 in long_term_corr.columns:
                    for asset2 in long_term_corr.columns:
                        if asset1 >= asset2:  # 避免重复
                            continue
                            
                        long_term_value = long_term_corr.loc[asset1, asset2]
                        window_value = window_corr.loc[asset1, asset2]
                        
                        # 检测显著变化
                        if not pd.isna(long_term_value) and not pd.isna(window_value):
                            change = abs(window_value - long_term_value)
                            
                            if change > threshold:
                                breakdown_events.append({
                                    'timestamp': returns.index[i],
                                    'asset1': asset1,
                                    'asset2': asset2,
                                    'long_term_correlation': long_term_value,
                                    'window_correlation': window_value,
                                    'change': change,
                                    'type': 'breakdown' if change > 0 else 'recovery'
                                })
            
            logger.info(f"Detected {len(breakdown_events)} correlation breakdown events")
            return breakdown_events
            
        except Exception as e:
            logger.error(f"Failed to detect correlation breakdown: {e}")
            raise DataError(f"Correlation breakdown detection failed: {e}")
    
    def calculate_portfolio_correlation(
        self, 
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> float:
        """计算投资组合内部相关性
        
        Args:
            returns: 收益率数据
            weights: 权重字典
            
        Returns:
            投资组合相关性
        """
        try:
            # 确保权重和为1
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
                weights = {k: v/total_weight for k, v in weights.items()}
            
            # 计算投资组合收益率
            portfolio_returns = pd.Series(0.0, index=returns.index)
            for asset, weight in weights.items():
                if asset in returns.columns:
                    portfolio_returns += weight * returns[asset]
            
            # 计算与各资产的相关性
            correlations = []
            for asset in weights.keys():
                if asset in returns.columns:
                    corr = portfolio_returns.corr(returns[asset])
                    if not pd.isna(corr):
                        correlations.append(corr)
            
            avg_correlation = np.mean(correlations) if correlations else 0.0
            
            logger.info(f"Calculated portfolio correlation: {avg_correlation:.3f}")
            return avg_correlation
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio correlation: {e}")
            raise DataError(f"Portfolio correlation calculation failed: {e}")
    
    def analyze_sector_correlation(
        self, 
        data: pd.DataFrame,
        sector_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """分析行业相关性
        
        Args:
            data: 价格数据
            sector_mapping: 股票到行业的映射
            
        Returns:
            行业相关性分析结果
        """
        try:
            returns = data.pct_change().dropna()
            
            # 按行业分组
            sectors = {}
            for asset in returns.columns:
                if asset in sector_mapping:
                    sector = sector_mapping[asset]
                    if sector not in sectors:
                        sectors[sector] = []
                    sectors[sector].append(asset)
            
            # 计算行业内部和行业间相关性
            intra_sector_correlations = {}
            inter_sector_correlations = {}
            
            for sector, assets in sectors.items():
                if len(assets) > 1:
                    sector_returns = returns[assets]
                    corr_matrix = sector_returns.corr()
                    
                    # 行业内部平均相关性
                    upper_triangle = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                    )
                    intra_corr = upper_triangle.stack().mean()
                    intra_sector_correlations[sector] = intra_corr
            
            # 计算行业间相关性
            sector_returns = {}
            for sector, assets in sectors.items():
                if len(assets) > 0:
                    sector_returns[sector] = returns[assets].mean(axis=1)
            
            if len(sector_returns) > 1:
                sector_df = pd.DataFrame(sector_returns)
                inter_corr_matrix = sector_df.corr()
                
                upper_triangle = inter_corr_matrix.where(
                    np.triu(np.ones(inter_corr_matrix.shape), k=1).astype(bool)
                )
                inter_sector_correlations = upper_triangle.stack().to_dict()
            
            result = {
                'intra_sector_correlations': intra_sector_correlations,
                'inter_sector_correlations': inter_sector_correlations,
                'sector_count': len(sectors),
                'total_assets': len(returns.columns)
            }
            
            logger.info(f"Analyzed sector correlations for {len(sectors)} sectors")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze sector correlation: {e}")
            raise DataError(f"Sector correlation analysis failed: {e}")
    
    def comprehensive_analysis(
        self, 
        data: pd.DataFrame,
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> CorrelationResult:
        """综合分析
        
        Args:
            data: 价格数据
            sector_mapping: 行业映射
            
        Returns:
            综合分析结果
        """
        try:
            # 计算相关性矩阵
            correlation_matrix = self.calculate_correlation_matrix(data)
            
            # 找出高相关性对
            significant_correlations = self.find_highly_correlated_pairs(correlation_matrix)
            
            # 计算统计指标
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            correlations = upper_triangle.stack().dropna()
            
            result = CorrelationResult(
                correlation_matrix=correlation_matrix,
                significant_correlations=significant_correlations,
                average_correlation=correlations.mean(),
                max_correlation=correlations.max(),
                min_correlation=correlations.min(),
                analysis_timestamp=pd.Timestamp.now()
            )
            
            logger.info("Completed comprehensive correlation analysis")
            return result
            
        except Exception as e:
            logger.error(f"Failed to perform comprehensive analysis: {e}")
            raise DataError(f"Comprehensive correlation analysis failed: {e}")

# 便捷函数
def analyze_market_correlation(
    data: pd.DataFrame,
    threshold: float = 0.7
) -> CorrelationResult:
    """分析市场相关性的便捷函数
    
    Args:
        data: 价格数据
        threshold: 相关性阈值
        
    Returns:
        相关性分析结果
    """
    analyzer = CorrelationAnalyzer(threshold)
    return analyzer.comprehensive_analysis(data)
