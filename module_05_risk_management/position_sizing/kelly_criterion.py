"""
凯利准则仓位管理模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from common.logging_system import setup_logger
from common.exceptions import QuantSystemError

logger = setup_logger("kelly_criterion")

@dataclass
class KellyResult:
    """凯利准则计算结果"""
    kelly_fraction: float
    recommended_position: float
    confidence: float
    win_rate: float
    avg_win: float
    avg_loss: float
    sharpe_ratio: float

class KellyCriterion:
    """凯利准则仓位管理器"""
    
    def __init__(self, max_kelly_fraction: float = 0.25, min_kelly_fraction: float = 0.01):
        """初始化凯利准则计算器
        
        Args:
            max_kelly_fraction: 最大凯利分数
            min_kelly_fraction: 最小凯利分数
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.min_kelly_fraction = min_kelly_fraction
        
    def calculate_kelly_fraction(self, returns: pd.Series) -> KellyResult:
        """计算凯利分数
        
        Args:
            returns: 收益率序列
            
        Returns:
            凯利准则计算结果
        """
        try:
            # 移除NaN值
            returns = returns.dropna()
            
            if len(returns) < 10:
                logger.warning("Insufficient data for Kelly calculation")
                return KellyResult(
                    kelly_fraction=0.0,
                    recommended_position=0.0,
                    confidence=0.0,
                    win_rate=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    sharpe_ratio=0.0
                )
            
            # 计算胜率和平均盈亏
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            
            win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
            
            # 计算凯利分数
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            else:
                kelly_fraction = 0.0
            
            # 限制凯利分数范围
            kelly_fraction = max(self.min_kelly_fraction, 
                               min(self.max_kelly_fraction, kelly_fraction))
            
            # 计算夏普比率
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # 计算置信度
            confidence = min(1.0, len(returns) / 100)  # 基于样本数量的置信度
            
            return KellyResult(
                kelly_fraction=kelly_fraction,
                recommended_position=kelly_fraction,
                confidence=confidence,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                sharpe_ratio=sharpe_ratio
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate Kelly fraction: {e}")
            raise QuantSystemError(f"Kelly calculation failed: {e}")
    
    def calculate_position_size(
        self, 
        account_value: float, 
        signal_strength: float, 
        volatility: float,
        returns: Optional[pd.Series] = None
    ) -> float:
        """计算仓位大小
        
        Args:
            account_value: 账户价值
            signal_strength: 信号强度 (0-1)
            volatility: 波动率
            returns: 历史收益率序列
            
        Returns:
            建议仓位大小
        """
        try:
            # 基础凯利分数
            if returns is not None:
                kelly_result = self.calculate_kelly_fraction(returns)
                base_kelly = kelly_result.kelly_fraction
            else:
                base_kelly = 0.1  # 默认凯利分数
            
            # 根据信号强度调整
            adjusted_kelly = base_kelly * signal_strength
            
            # 根据波动率调整
            volatility_adjustment = max(0.1, 1.0 - volatility)
            adjusted_kelly *= volatility_adjustment
            
            # 限制在合理范围内
            adjusted_kelly = max(self.min_kelly_fraction, 
                               min(self.max_kelly_fraction, adjusted_kelly))
            
            # 计算仓位大小
            position_size = account_value * adjusted_kelly
            
            logger.info(f"Calculated position size: {position_size:.2f} (Kelly: {adjusted_kelly:.3f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            raise QuantSystemError(f"Position sizing failed: {e}")
    
    def optimize_portfolio_kelly(self, returns_matrix: pd.DataFrame) -> Dict[str, float]:
        """使用凯利准则优化投资组合
        
        Args:
            returns_matrix: 收益率矩阵 (资产 x 时间)
            
        Returns:
            优化后的权重字典
        """
        try:
            weights = {}
            
            for asset in returns_matrix.columns:
                asset_returns = returns_matrix[asset]
                kelly_result = self.calculate_kelly_fraction(asset_returns)
                weights[asset] = kelly_result.kelly_fraction
            
            # 归一化权重
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            logger.info(f"Optimized portfolio weights: {weights}")
            return weights
            
        except Exception as e:
            logger.error(f"Failed to optimize portfolio: {e}")
            raise QuantSystemError(f"Portfolio optimization failed: {e}")
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """计算风险指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            风险指标字典
        """
        try:
            returns = returns.dropna()
            
            if len(returns) < 2:
                return {
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'var_95': 0.0,
                    'cvar_95': 0.0
                }
            
            # 波动率
            volatility = returns.std() * np.sqrt(252)
            
            # 夏普比率
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # 最大回撤
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # VaR和CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            
            return {
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
            raise QuantSystemError(f"Risk metrics calculation failed: {e}")

# 便捷函数
def calculate_kelly_position(
    account_value: float,
    returns: pd.Series,
    signal_strength: float = 1.0
) -> float:
    """计算凯利仓位的便捷函数
    
    Args:
        account_value: 账户价值
        returns: 历史收益率
        signal_strength: 信号强度
        
    Returns:
        建议仓位大小
    """
    kelly = KellyCriterion()
    return kelly.calculate_position_size(account_value, signal_strength, 0.2, returns)