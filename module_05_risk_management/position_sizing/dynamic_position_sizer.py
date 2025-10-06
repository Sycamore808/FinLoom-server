"""
动态仓位管理器模块
根据市场条件、风险水平、信号强度等因素动态调整仓位大小
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("dynamic_position_sizer")


class MarketRegime(Enum):
    """市场状态枚举"""

    BULL = "bull"  # 牛市
    BEAR = "bear"  # 熊市
    SIDEWAYS = "sideways"  # 震荡市
    HIGH_VOLATILITY = "high_volatility"  # 高波动
    LOW_VOLATILITY = "low_volatility"  # 低波动


class PositionSizingMethod(Enum):
    """仓位计算方法枚举"""

    FIXED = "fixed"  # 固定仓位
    KELLY = "kelly"  # 凯利准则
    VOLATILITY_TARGET = "volatility_target"  # 目标波动率
    RISK_PARITY = "risk_parity"  # 风险平价
    ADAPTIVE = "adaptive"  # 自适应
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # 置信度加权


@dataclass
class PositionSizingConfig:
    """动态仓位配置"""

    max_position_size: float = 0.20  # 单个持仓最大比例
    min_position_size: float = 0.01  # 单个持仓最小比例
    max_total_exposure: float = 1.0  # 最大总敞口
    target_volatility: float = 0.15  # 目标波动率
    risk_per_trade: float = 0.02  # 单笔交易风险
    max_leverage: float = 1.5  # 最大杠杆
    volatility_lookback: int = 60  # 波动率回看期

    # Kelly准则参数
    kelly_fraction: float = 0.25  # Kelly分数上限
    kelly_confidence: float = 0.7  # Kelly置信度

    # 市场状态调整
    bull_market_multiplier: float = 1.2  # 牛市乘数
    bear_market_multiplier: float = 0.6  # 熊市乘数
    high_vol_multiplier: float = 0.7  # 高波动乘数

    # 风险控制
    max_drawdown_threshold: float = 0.15  # 最大回撤阈值
    correlation_threshold: float = 0.7  # 相关性阈值
    concentration_limit: float = 0.30  # 集中度限制

    # 调整频率
    rebalance_frequency: str = "daily"  # 再平衡频率
    min_position_change: float = 0.05  # 最小变动阈值


@dataclass
class PositionSizingResult:
    """仓位计算结果"""

    symbol: str
    recommended_size: float  # 推荐仓位大小（资金比例）
    recommended_shares: int  # 推荐股数
    position_value: float  # 仓位价值
    risk_amount: float  # 风险金额
    confidence_score: float  # 置信度分数
    method_used: str  # 使用的方法
    market_regime: str  # 市场状态
    volatility_adjustment: float  # 波动率调整因子
    signal_strength: float  # 信号强度
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DynamicPositionSizer:
    """动态仓位管理器

    根据多种因素动态计算最优仓位大小：
    - 市场波动率
    - 信号强度和置信度
    - 市场状态（牛市/熊市/震荡）
    - 投资组合当前风险
    - 相关性和集中度
    """

    def __init__(self, config: Optional[PositionSizingConfig] = None):
        """初始化动态仓位管理器

        Args:
            config: 仓位配置
        """
        self.config = config or PositionSizingConfig()
        self.position_history: List[PositionSizingResult] = []
        logger.info(f"Initialized DynamicPositionSizer with config: {self.config}")

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        account_value: float,
        signal_strength: float,
        confidence: float,
        historical_returns: pd.Series,
        portfolio_positions: Optional[Dict[str, float]] = None,
        market_data: Optional[pd.DataFrame] = None,
        method: PositionSizingMethod = PositionSizingMethod.ADAPTIVE,
    ) -> PositionSizingResult:
        """计算动态仓位大小

        Args:
            symbol: 股票代码
            current_price: 当前价格
            account_value: 账户价值
            signal_strength: 信号强度 (0-1)
            confidence: 信号置信度 (0-1)
            historical_returns: 历史收益率序列
            portfolio_positions: 当前持仓字典 {symbol: position_value}
            market_data: 市场数据DataFrame
            method: 仓位计算方法

        Returns:
            仓位计算结果
        """
        try:
            logger.info(f"Calculating position size for {symbol} using {method.value}")

            # 检测市场状态
            market_regime = self._detect_market_regime(historical_returns, market_data)

            # 计算波动率
            volatility = self._calculate_volatility(historical_returns)

            # 计算波动率调整因子
            vol_adjustment = self._calculate_volatility_adjustment(volatility)

            # 根据方法计算基础仓位
            if method == PositionSizingMethod.ADAPTIVE:
                base_size = self._adaptive_position_sizing(
                    signal_strength,
                    confidence,
                    volatility,
                    historical_returns,
                    market_regime,
                )
            elif method == PositionSizingMethod.KELLY:
                base_size = self._kelly_position_sizing(historical_returns, confidence)
            elif method == PositionSizingMethod.VOLATILITY_TARGET:
                base_size = self._volatility_target_sizing(volatility, account_value)
            elif method == PositionSizingMethod.RISK_PARITY:
                base_size = self._risk_parity_sizing(volatility)
            elif method == PositionSizingMethod.CONFIDENCE_WEIGHTED:
                base_size = self._confidence_weighted_sizing(
                    signal_strength, confidence
                )
            else:
                base_size = self.config.max_position_size * 0.5

            # 应用市场状态调整
            adjusted_size = self._apply_market_regime_adjustment(
                base_size, market_regime
            )

            # 应用波动率调整
            adjusted_size *= vol_adjustment

            # 应用信号强度调整
            adjusted_size *= signal_strength

            # 检查投资组合约束
            if portfolio_positions:
                adjusted_size = self._apply_portfolio_constraints(
                    adjusted_size, symbol, portfolio_positions, account_value
                )

            # 应用边界限制
            final_size = self._apply_limits(adjusted_size)

            # 计算股数和价值
            position_value = account_value * final_size
            recommended_shares = int(position_value / current_price)
            actual_position_value = recommended_shares * current_price
            actual_size = actual_position_value / account_value

            # 计算风险金额
            risk_amount = actual_position_value * self.config.risk_per_trade

            # 计算综合置信度分数
            confidence_score = self._calculate_confidence_score(
                signal_strength, confidence, volatility, market_regime
            )

            result = PositionSizingResult(
                symbol=symbol,
                recommended_size=actual_size,
                recommended_shares=recommended_shares,
                position_value=actual_position_value,
                risk_amount=risk_amount,
                confidence_score=confidence_score,
                method_used=method.value,
                market_regime=market_regime.value,
                volatility_adjustment=vol_adjustment,
                signal_strength=signal_strength,
                metadata={
                    "base_size": base_size,
                    "adjusted_size": adjusted_size,
                    "volatility": volatility,
                    "current_price": current_price,
                    "account_value": account_value,
                },
            )

            self.position_history.append(result)
            logger.info(
                f"Position size calculated: {symbol} - {recommended_shares} shares "
                f"({actual_size:.2%} of portfolio), confidence: {confidence_score:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to calculate position size for {symbol}: {e}")
            raise QuantSystemError(f"Position sizing failed: {e}")

    def calculate_multi_position_allocation(
        self,
        signals: Dict[str, Dict[str, float]],
        account_value: float,
        current_prices: Dict[str, float],
        returns_data: pd.DataFrame,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, PositionSizingResult]:
        """计算多个持仓的联合配置

        Args:
            signals: 信号字典 {symbol: {'strength': float, 'confidence': float}}
            account_value: 账户价值
            current_prices: 当前价格字典
            returns_data: 收益率数据DataFrame
            current_positions: 当前持仓

        Returns:
            仓位配置结果字典
        """
        try:
            logger.info(
                f"Calculating multi-position allocation for {len(signals)} symbols"
            )

            results = {}
            total_allocated = 0.0
            current_positions = current_positions or {}

            # 按信号强度和置信度排序
            sorted_symbols = sorted(
                signals.keys(),
                key=lambda x: signals[x]["strength"] * signals[x]["confidence"],
                reverse=True,
            )

            # 计算相关性矩阵
            correlation_matrix = returns_data.corr()

            for symbol in sorted_symbols:
                signal = signals[symbol]

                # 检查是否超过总敞口限制
                if total_allocated >= self.config.max_total_exposure:
                    logger.warning(f"Reached max total exposure, skipping {symbol}")
                    continue

                # 获取历史收益率
                if symbol not in returns_data.columns:
                    logger.warning(f"No historical data for {symbol}, skipping")
                    continue

                historical_returns = returns_data[symbol].dropna()

                if len(historical_returns) < 30:
                    logger.warning(f"Insufficient data for {symbol}, skipping")
                    continue

                # 计算单个仓位
                result = self.calculate_position_size(
                    symbol=symbol,
                    current_price=current_prices[symbol],
                    account_value=account_value,
                    signal_strength=signal["strength"],
                    confidence=signal["confidence"],
                    historical_returns=historical_returns,
                    portfolio_positions=current_positions,
                    method=PositionSizingMethod.ADAPTIVE,
                )

                # 检查相关性约束
                adjusted_result = self._adjust_for_correlation(
                    result, symbol, results, correlation_matrix
                )

                results[symbol] = adjusted_result
                total_allocated += adjusted_result.recommended_size

                # 更新临时持仓用于后续计算
                current_positions[symbol] = adjusted_result.position_value

            # 归一化权重（如果超过最大敞口）
            if total_allocated > self.config.max_total_exposure:
                scale_factor = self.config.max_total_exposure / total_allocated
                for symbol in results:
                    results[symbol].recommended_size *= scale_factor
                    results[symbol].recommended_shares = int(
                        results[symbol].recommended_size
                        * account_value
                        / current_prices[symbol]
                    )
                    results[symbol].position_value = (
                        results[symbol].recommended_shares * current_prices[symbol]
                    )

            logger.info(
                f"Multi-position allocation completed: {len(results)} positions, "
                f"total allocation: {total_allocated:.2%}"
            )

            return results

        except Exception as e:
            logger.error(f"Failed to calculate multi-position allocation: {e}")
            raise QuantSystemError(f"Multi-position allocation failed: {e}")

    def adjust_position_for_drawdown(
        self, current_position: float, current_drawdown: float, max_drawdown: float
    ) -> float:
        """根据回撤调整仓位

        Args:
            current_position: 当前仓位
            current_drawdown: 当前回撤
            max_drawdown: 最大回撤

        Returns:
            调整后的仓位
        """
        if current_drawdown >= self.config.max_drawdown_threshold:
            # 回撤过大，减少仓位
            reduction_factor = max(0.3, 1 - (current_drawdown / max_drawdown))
            adjusted_position = current_position * reduction_factor
            logger.warning(
                f"Drawdown threshold exceeded ({current_drawdown:.2%}), "
                f"reducing position by {(1 - reduction_factor) * 100:.1f}%"
            )
            return adjusted_position

        return current_position

    def _detect_market_regime(
        self, returns: pd.Series, market_data: Optional[pd.DataFrame] = None
    ) -> MarketRegime:
        """检测市场状态

        Args:
            returns: 收益率序列
            market_data: 市场数据

        Returns:
            市场状态
        """
        try:
            # 计算趋势（短期和长期均线）
            if len(returns) < 50:
                return MarketRegime.SIDEWAYS

            cumulative = (1 + returns).cumprod()
            sma_20 = cumulative.rolling(20).mean()
            sma_50 = cumulative.rolling(50).mean()

            # 计算波动率
            volatility = returns.std() * np.sqrt(252)
            historical_vol = returns.rolling(60).std().mean() * np.sqrt(252)

            # 判断趋势
            if sma_20.iloc[-1] > sma_50.iloc[-1] * 1.05:
                trend = "bull"
            elif sma_20.iloc[-1] < sma_50.iloc[-1] * 0.95:
                trend = "bear"
            else:
                trend = "sideways"

            # 判断波动率状态
            if volatility > historical_vol * 1.5:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < historical_vol * 0.5:
                return MarketRegime.LOW_VOLATILITY

            # 综合判断
            if trend == "bull":
                return MarketRegime.BULL
            elif trend == "bear":
                return MarketRegime.BEAR
            else:
                return MarketRegime.SIDEWAYS

        except Exception as e:
            logger.error(f"Failed to detect market regime: {e}")
            return MarketRegime.SIDEWAYS

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """计算波动率

        Args:
            returns: 收益率序列

        Returns:
            年化波动率
        """
        if len(returns) < 2:
            return 0.2  # 默认波动率

        # 使用指数加权计算波动率
        ewm_vol = returns.ewm(span=self.config.volatility_lookback).std()
        return ewm_vol.iloc[-1] * np.sqrt(252)

    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """计算波动率调整因子

        Args:
            volatility: 波动率

        Returns:
            调整因子
        """
        # 目标波动率 / 实际波动率
        if volatility <= 0:
            return 1.0

        adjustment = self.config.target_volatility / volatility

        # 限制调整范围
        return np.clip(adjustment, 0.3, 2.0)

    def _adaptive_position_sizing(
        self,
        signal_strength: float,
        confidence: float,
        volatility: float,
        returns: pd.Series,
        market_regime: MarketRegime,
    ) -> float:
        """自适应仓位计算

        综合多种因素的自适应方法
        """
        # 基础权重
        base_weight = 0.1

        # Kelly分数
        kelly_weight = self._kelly_position_sizing(returns, confidence)

        # 波动率目标权重
        vol_weight = self._volatility_target_sizing(volatility, 1.0)

        # 加权组合
        adaptive_weight = 0.3 * base_weight + 0.4 * kelly_weight + 0.3 * vol_weight

        # 应用信号强度和置信度
        adaptive_weight *= signal_strength * confidence

        return adaptive_weight

    def _kelly_position_sizing(self, returns: pd.Series, confidence: float) -> float:
        """Kelly准则仓位计算"""
        try:
            if len(returns) < 10:
                return 0.05

            # 计算胜率和平均盈亏
            wins = returns[returns > 0]
            losses = returns[returns < 0]

            if len(returns) == 0:
                return 0.05

            win_rate = len(wins) / len(returns)
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-6

            # Kelly公式
            if avg_loss > 0:
                kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            else:
                kelly = 0.0

            # 应用置信度和上限
            kelly = kelly * confidence * self.config.kelly_confidence
            kelly = np.clip(kelly, 0, self.config.kelly_fraction)

            return kelly

        except Exception as e:
            logger.error(f"Kelly calculation failed: {e}")
            return 0.05

    def _volatility_target_sizing(
        self, volatility: float, account_value: float
    ) -> float:
        """目标波动率仓位计算"""
        if volatility <= 0:
            return 0.1

        # 目标波动率 / 资产波动率
        position_size = self.config.target_volatility / volatility

        return np.clip(
            position_size, self.config.min_position_size, self.config.max_position_size
        )

    def _risk_parity_sizing(self, volatility: float) -> float:
        """风险平价仓位计算"""
        if volatility <= 0:
            return 0.1

        # 逆波动率权重
        inv_vol_weight = 1 / volatility

        # 标准化到合理范围
        normalized_weight = inv_vol_weight * self.config.target_volatility

        return np.clip(
            normalized_weight,
            self.config.min_position_size,
            self.config.max_position_size,
        )

    def _confidence_weighted_sizing(
        self, signal_strength: float, confidence: float
    ) -> float:
        """置信度加权仓位计算"""
        # 基础权重乘以信号强度和置信度的几何平均
        combined_score = np.sqrt(signal_strength * confidence)
        position_size = self.config.max_position_size * combined_score

        return np.clip(
            position_size, self.config.min_position_size, self.config.max_position_size
        )

    def _apply_market_regime_adjustment(
        self, position_size: float, market_regime: MarketRegime
    ) -> float:
        """应用市场状态调整"""
        if market_regime == MarketRegime.BULL:
            return position_size * self.config.bull_market_multiplier
        elif market_regime == MarketRegime.BEAR:
            return position_size * self.config.bear_market_multiplier
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            return position_size * self.config.high_vol_multiplier
        else:
            return position_size

    def _apply_portfolio_constraints(
        self,
        position_size: float,
        symbol: str,
        portfolio_positions: Dict[str, float],
        account_value: float,
    ) -> float:
        """应用投资组合约束"""
        # 计算当前总敞口
        current_exposure = sum(portfolio_positions.values()) / account_value

        # 检查集中度
        if symbol in portfolio_positions:
            current_position = portfolio_positions[symbol] / account_value
            if current_position > self.config.concentration_limit:
                logger.warning(f"Position {symbol} exceeds concentration limit")
                position_size = min(
                    position_size, self.config.concentration_limit - current_position
                )

        # 检查总敞口限制
        if current_exposure + position_size > self.config.max_total_exposure:
            available = self.config.max_total_exposure - current_exposure
            position_size = min(position_size, available)
            logger.info(
                f"Adjusted position size to respect max total exposure: {position_size:.2%}"
            )

        return max(0, position_size)

    def _apply_limits(self, position_size: float) -> float:
        """应用边界限制"""
        return np.clip(
            position_size, self.config.min_position_size, self.config.max_position_size
        )

    def _calculate_confidence_score(
        self,
        signal_strength: float,
        confidence: float,
        volatility: float,
        market_regime: MarketRegime,
    ) -> float:
        """计算综合置信度分数"""
        # 基础置信度
        base_confidence = (signal_strength + confidence) / 2

        # 波动率惩罚
        vol_penalty = 1.0 if volatility < 0.3 else max(0.5, 1.0 - (volatility - 0.3))

        # 市场状态调整
        regime_adjustment = {
            MarketRegime.BULL: 1.1,
            MarketRegime.BEAR: 0.8,
            MarketRegime.SIDEWAYS: 0.9,
            MarketRegime.HIGH_VOLATILITY: 0.7,
            MarketRegime.LOW_VOLATILITY: 1.0,
        }.get(market_regime, 1.0)

        confidence_score = base_confidence * vol_penalty * regime_adjustment

        return np.clip(confidence_score, 0.0, 1.0)

    def _adjust_for_correlation(
        self,
        result: PositionSizingResult,
        symbol: str,
        existing_results: Dict[str, PositionSizingResult],
        correlation_matrix: pd.DataFrame,
    ) -> PositionSizingResult:
        """根据相关性调整仓位"""
        if not existing_results or symbol not in correlation_matrix.columns:
            return result

        try:
            # 检查与现有持仓的相关性
            high_corr_penalty = 1.0

            for existing_symbol in existing_results.keys():
                if existing_symbol in correlation_matrix.columns:
                    corr = abs(correlation_matrix.loc[symbol, existing_symbol])

                    if corr > self.config.correlation_threshold:
                        # 相关性过高，降低仓位
                        penalty = 1.0 - (corr - self.config.correlation_threshold) * 2
                        high_corr_penalty = min(high_corr_penalty, max(0.5, penalty))
                        logger.info(
                            f"High correlation detected between {symbol} and {existing_symbol} "
                            f"({corr:.2f}), applying penalty {penalty:.2f}"
                        )

            # 应用惩罚
            if high_corr_penalty < 1.0:
                result.recommended_size *= high_corr_penalty
                result.recommended_shares = int(
                    result.recommended_size
                    * result.metadata["account_value"]
                    / result.metadata["current_price"]
                )
                result.position_value = (
                    result.recommended_shares * result.metadata["current_price"]
                )
                result.metadata["correlation_penalty"] = high_corr_penalty

            return result

        except Exception as e:
            logger.error(f"Failed to adjust for correlation: {e}")
            return result

    def get_position_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[PositionSizingResult]:
        """获取仓位历史记录

        Args:
            symbol: 股票代码（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            仓位记录列表
        """
        filtered = self.position_history

        if symbol:
            filtered = [r for r in filtered if r.symbol == symbol]

        if start_date:
            filtered = [r for r in filtered if r.timestamp >= start_date]

        if end_date:
            filtered = [r for r in filtered if r.timestamp <= end_date]

        return filtered

    def get_position_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """获取仓位统计信息

        Args:
            symbol: 股票代码（可选）

        Returns:
            统计信息字典
        """
        history = self.get_position_history(symbol=symbol)

        if not history:
            return {}

        sizes = [r.recommended_size for r in history]
        confidence_scores = [r.confidence_score for r in history]

        return {
            "total_calculations": len(history),
            "avg_position_size": np.mean(sizes),
            "median_position_size": np.median(sizes),
            "std_position_size": np.std(sizes),
            "min_position_size": np.min(sizes),
            "max_position_size": np.max(sizes),
            "avg_confidence": np.mean(confidence_scores),
            "methods_used": {r.method_used for r in history},
            "market_regimes": {r.market_regime for r in history},
        }


# 便捷函数
def calculate_dynamic_position(
    symbol: str,
    current_price: float,
    account_value: float,
    signal_strength: float,
    confidence: float,
    historical_returns: pd.Series,
    config: Optional[PositionSizingConfig] = None,
) -> PositionSizingResult:
    """计算动态仓位的便捷函数

    Args:
        symbol: 股票代码
        current_price: 当前价格
        account_value: 账户价值
        signal_strength: 信号强度
        confidence: 置信度
        historical_returns: 历史收益率
        config: 仓位配置

    Returns:
        仓位计算结果
    """
    sizer = DynamicPositionSizer(config)
    return sizer.calculate_position_size(
        symbol=symbol,
        current_price=current_price,
        account_value=account_value,
        signal_strength=signal_strength,
        confidence=confidence,
        historical_returns=historical_returns,
    )
