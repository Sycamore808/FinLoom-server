"""
信号过滤器模块
负责过滤和优化交易信号
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from common.data_structures import Signal
from common.exceptions import ExecutionError
from common.logging_system import setup_logger
from module_08_execution.signal_generator import EnhancedSignal, SignalType

logger = setup_logger("signal_filter")


@dataclass
class FilterConfig:
    """过滤器配置"""

    min_signal_strength: float = 0.5
    max_position_size: float = 0.30  # 最大单股仓位比例
    enable_risk_filter: bool = True
    enable_liquidity_filter: bool = True
    min_volume_ratio: float = 1.0  # 最小成交量比率
    blacklist: List[str] = None
    whitelist: List[str] = None
    max_signals_per_day: int = 20
    min_signal_interval_seconds: int = 300  # 同一标的最小信号间隔
    max_correlation: float = 0.8  # 最大相关性
    enable_diversification: bool = True
    max_sector_concentration: float = 0.4  # 最大行业集中度

    def __post_init__(self):
        if self.blacklist is None:
            self.blacklist = []
        if self.whitelist is None:
            self.whitelist = []


@dataclass
class FilterResult:
    """过滤结果"""

    passed: bool
    reason: str
    score: float
    metadata: Dict[str, Any]


class SignalFilter:
    """信号过滤器类"""

    def __init__(self, config: FilterConfig = None):
        """初始化信号过滤器

        Args:
            config: 过滤器配置
        """
        self.config = config or FilterConfig()
        self.signal_history: List[EnhancedSignal] = []
        self.recent_signals: Dict[str, datetime] = {}  # symbol -> last signal time

    def filter_signals(
        self,
        signals: List[EnhancedSignal],
        current_portfolio: Optional[Dict[str, Any]] = None,
        market_data: Optional[pd.DataFrame] = None,
    ) -> List[EnhancedSignal]:
        """过滤信号列表

        Args:
            signals: 待过滤信号列表
            current_portfolio: 当前持仓信息
            market_data: 市场数据

        Returns:
            过滤后的信号列表
        """
        if not signals:
            return []

        filtered_signals = []

        for signal in signals:
            try:
                # 执行各项过滤检查
                result = self._check_signal(signal, current_portfolio, market_data)

                if result.passed:
                    filtered_signals.append(signal)
                    logger.debug(
                        f"Signal {signal.signal_id} passed filter "
                        f"(score: {result.score:.3f})"
                    )
                else:
                    logger.debug(f"Signal {signal.signal_id} rejected: {result.reason}")

            except Exception as e:
                logger.error(f"Error filtering signal {signal.signal_id}: {e}")
                continue

        # 应用信号数量限制
        if len(filtered_signals) > self.config.max_signals_per_day:
            # 按优先级和置信度排序
            filtered_signals.sort(
                key=lambda s: (s.priority.value, s.confidence), reverse=True
            )
            filtered_signals = filtered_signals[: self.config.max_signals_per_day]

        logger.info(
            f"Filtered {len(signals)} signals to {len(filtered_signals)} "
            f"high-quality signals"
        )

        return filtered_signals

    def _check_signal(
        self,
        signal: EnhancedSignal,
        current_portfolio: Optional[Dict[str, Any]],
        market_data: Optional[pd.DataFrame],
    ) -> FilterResult:
        """检查单个信号

        Args:
            signal: 信号对象
            current_portfolio: 当前持仓
            market_data: 市场数据

        Returns:
            过滤结果
        """
        metadata = {}

        # 1. 基本强度检查
        if signal.confidence < self.config.min_signal_strength:
            return FilterResult(
                passed=False,
                reason=f"Confidence {signal.confidence:.2f} below threshold",
                score=signal.confidence,
                metadata=metadata,
            )

        # 2. 黑白名单检查
        if self.config.blacklist and signal.symbol in self.config.blacklist:
            return FilterResult(
                passed=False,
                reason=f"Symbol {signal.symbol} in blacklist",
                score=0.0,
                metadata=metadata,
            )

        if self.config.whitelist and signal.symbol not in self.config.whitelist:
            return FilterResult(
                passed=False,
                reason=f"Symbol {signal.symbol} not in whitelist",
                score=0.0,
                metadata=metadata,
            )

        # 3. 时间间隔检查
        if signal.symbol in self.recent_signals:
            last_time = self.recent_signals[signal.symbol]
            time_diff = (datetime.now() - last_time).total_seconds()

            if time_diff < self.config.min_signal_interval_seconds:
                return FilterResult(
                    passed=False,
                    reason=f"Too soon since last signal ({time_diff:.0f}s)",
                    score=signal.confidence * 0.5,
                    metadata=metadata,
                )

        # 4. 持仓集中度检查
        if self.config.enable_risk_filter and current_portfolio:
            position_check = self._check_position_limits(signal, current_portfolio)
            if not position_check.passed:
                return position_check
            metadata.update(position_check.metadata)

        # 5. 流动性检查
        if self.config.enable_liquidity_filter and market_data is not None:
            liquidity_check = self._check_liquidity(signal, market_data)
            if not liquidity_check.passed:
                return liquidity_check
            metadata.update(liquidity_check.metadata)

        # 6. 风险评分检查
        if signal.risk_score and signal.risk_score > 0.9:
            return FilterResult(
                passed=False,
                reason=f"Risk score too high: {signal.risk_score:.2f}",
                score=signal.confidence * 0.3,
                metadata=metadata,
            )

        # 计算综合评分
        score = self._calculate_signal_score(signal, metadata)
        metadata["final_score"] = score

        # 更新信号历史
        self.recent_signals[signal.symbol] = datetime.now()
        self.signal_history.append(signal)

        return FilterResult(
            passed=True, reason="All checks passed", score=score, metadata=metadata
        )

    def _check_position_limits(
        self, signal: EnhancedSignal, portfolio: Dict[str, Any]
    ) -> FilterResult:
        """检查持仓限制

        Args:
            signal: 信号对象
            portfolio: 持仓信息

        Returns:
            过滤结果
        """
        metadata = {}

        # 获取当前持仓
        current_positions = portfolio.get("positions", {})
        total_value = portfolio.get("total_value", 0)

        if total_value == 0:
            return FilterResult(
                passed=True,
                reason="No position limits (empty portfolio)",
                score=1.0,
                metadata=metadata,
            )

        # 检查单股持仓比例
        if signal.symbol in current_positions:
            position = current_positions[signal.symbol]
            position_value = position.get("market_value", 0)
            position_ratio = position_value / total_value

            metadata["current_position_ratio"] = position_ratio

            # 如果是买入信号且已持仓过多
            if signal.signal_type == SignalType.BUY:
                if position_ratio > self.config.max_position_size:
                    return FilterResult(
                        passed=False,
                        reason=f"Position ratio {position_ratio:.2%} exceeds limit",
                        score=0.5,
                        metadata=metadata,
                    )

        return FilterResult(
            passed=True, reason="Position limits OK", score=1.0, metadata=metadata
        )

    def _check_liquidity(
        self, signal: EnhancedSignal, market_data: pd.DataFrame
    ) -> FilterResult:
        """检查流动性

        Args:
            signal: 信号对象
            market_data: 市场数据

        Returns:
            过滤结果
        """
        metadata = {}

        # 获取该股票的市场数据
        symbol_data = market_data[market_data["symbol"] == signal.symbol]

        if symbol_data.empty:
            logger.warning(f"No market data for {signal.symbol}")
            return FilterResult(
                passed=True,  # 无数据时通过
                reason="No market data available",
                score=0.8,
                metadata=metadata,
            )

        # 计算平均成交量
        avg_volume = symbol_data["volume"].mean()
        recent_volume = symbol_data["volume"].iloc[-1] if len(symbol_data) > 0 else 0

        # 成交量比率
        if avg_volume > 0:
            volume_ratio = recent_volume / avg_volume
            metadata["volume_ratio"] = volume_ratio

            if volume_ratio < self.config.min_volume_ratio:
                return FilterResult(
                    passed=False,
                    reason=f"Volume ratio {volume_ratio:.2f} too low",
                    score=0.6,
                    metadata=metadata,
                )

        # 检查成交额
        if "amount" in symbol_data.columns:
            recent_amount = (
                symbol_data["amount"].iloc[-1] if len(symbol_data) > 0 else 0
            )
            metadata["recent_amount"] = recent_amount

            # 成交额过低（如低于100万）
            if recent_amount < 1_000_000:
                return FilterResult(
                    passed=False,
                    reason=f"Trading amount {recent_amount:,.0f} too low",
                    score=0.5,
                    metadata=metadata,
                )

        return FilterResult(
            passed=True, reason="Liquidity OK", score=1.0, metadata=metadata
        )

    def _calculate_signal_score(
        self, signal: EnhancedSignal, metadata: Dict[str, Any]
    ) -> float:
        """计算信号综合评分

        Args:
            signal: 信号对象
            metadata: 元数据

        Returns:
            综合评分 (0-1)
        """
        # 基础评分来自信号置信度
        score = signal.confidence

        # 根据优先级调整
        priority_multiplier = {
            1: 0.8,  # LOW
            5: 1.0,  # NORMAL
            8: 1.15,  # HIGH
            10: 1.3,  # CRITICAL
        }
        score *= priority_multiplier.get(signal.priority.value, 1.0)

        # 根据预期收益调整
        if signal.expected_return:
            return_factor = min(1.2, 1.0 + signal.expected_return / 10)
            score *= return_factor

        # 根据风险调整
        if signal.risk_score:
            risk_factor = max(0.7, 1.0 - signal.risk_score * 0.3)
            score *= risk_factor

        # 根据流动性调整
        if "volume_ratio" in metadata:
            volume_factor = min(1.1, metadata["volume_ratio"])
            score *= volume_factor

        # 限制在0-1范围内
        score = max(0.0, min(1.0, score))

        return score

    def filter_by_strength(
        self, signals: List[EnhancedSignal], min_strength: Optional[float] = None
    ) -> List[EnhancedSignal]:
        """按信号强度过滤

        Args:
            signals: 信号列表
            min_strength: 最小强度（默认使用配置）

        Returns:
            过滤后的信号列表
        """
        threshold = min_strength or self.config.min_signal_strength

        filtered = [signal for signal in signals if signal.confidence >= threshold]

        logger.info(
            f"Filtered {len(signals)} signals to {len(filtered)} "
            f"by strength >= {threshold:.2f}"
        )

        return filtered

    def filter_by_risk(
        self, signals: List[EnhancedSignal], max_risk: float = 0.8
    ) -> List[EnhancedSignal]:
        """按风险评分过滤

        Args:
            signals: 信号列表
            max_risk: 最大风险评分

        Returns:
            过滤后的信号列表
        """
        filtered = [
            signal for signal in signals if (signal.risk_score or 0.5) <= max_risk
        ]

        logger.info(
            f"Filtered {len(signals)} signals to {len(filtered)} "
            f"by risk <= {max_risk:.2f}"
        )

        return filtered

    def remove_correlated_signals(
        self,
        signals: List[EnhancedSignal],
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> List[EnhancedSignal]:
        """移除高度相关的信号

        Args:
            signals: 信号列表
            correlation_matrix: 相关性矩阵

        Returns:
            去相关后的信号列表
        """
        if not self.config.enable_diversification:
            return signals

        if len(signals) <= 1:
            return signals

        # 如果没有提供相关性矩阵，跳过
        if correlation_matrix is None:
            logger.warning(
                "No correlation matrix provided, skipping correlation filter"
            )
            return signals

        # 按优先级和置信度排序
        sorted_signals = sorted(
            signals, key=lambda s: (s.priority.value, s.confidence), reverse=True
        )

        selected = [sorted_signals[0]]

        for signal in sorted_signals[1:]:
            # 检查与已选信号的相关性
            is_correlated = False

            for selected_signal in selected:
                try:
                    if (
                        signal.symbol in correlation_matrix.index
                        and selected_signal.symbol in correlation_matrix.columns
                    ):
                        corr = abs(
                            correlation_matrix.loc[
                                signal.symbol, selected_signal.symbol
                            ]
                        )

                        if corr > self.config.max_correlation:
                            is_correlated = True
                            break
                except (KeyError, AttributeError):
                    continue

            if not is_correlated:
                selected.append(signal)

        logger.info(f"Removed {len(signals) - len(selected)} correlated signals")

        return selected

    def get_filter_statistics(self) -> Dict[str, Any]:
        """获取过滤器统计信息

        Returns:
            统计信息字典
        """
        total_signals = len(self.signal_history)

        if total_signals == 0:
            return {
                "total_signals": 0,
                "by_type": {},
                "by_symbol": {},
                "avg_confidence": 0.0,
                "avg_risk_score": 0.0,
            }

        # 按类型统计
        by_type = {}
        for signal in self.signal_history:
            signal_type = signal.signal_type.value
            by_type[signal_type] = by_type.get(signal_type, 0) + 1

        # 按标的统计
        by_symbol = {}
        for signal in self.signal_history:
            by_symbol[signal.symbol] = by_symbol.get(signal.symbol, 0) + 1

        # 平均置信度
        avg_confidence = np.mean([s.confidence for s in self.signal_history])

        # 平均风险评分
        risk_scores = [s.risk_score for s in self.signal_history if s.risk_score]
        avg_risk_score = np.mean(risk_scores) if risk_scores else 0.0

        return {
            "total_signals": total_signals,
            "by_type": by_type,
            "by_symbol": by_symbol,
            "avg_confidence": avg_confidence,
            "avg_risk_score": avg_risk_score,
            "unique_symbols": len(by_symbol),
        }

    def reset(self) -> None:
        """重置过滤器状态"""
        self.signal_history.clear()
        self.recent_signals.clear()
        logger.info("Signal filter reset")


# 便捷函数
def create_signal_filter(config: Optional[FilterConfig] = None) -> SignalFilter:
    """创建信号过滤器

    Args:
        config: 过滤器配置

    Returns:
        信号过滤器实例
    """
    return SignalFilter(config)


def quick_filter_signals(
    signals: List[EnhancedSignal], min_confidence: float = 0.6, max_risk: float = 0.8
) -> List[EnhancedSignal]:
    """快速过滤信号

    Args:
        signals: 信号列表
        min_confidence: 最小置信度
        max_risk: 最大风险

    Returns:
        过滤后的信号列表
    """
    config = FilterConfig(min_signal_strength=min_confidence, enable_risk_filter=True)

    filter_instance = SignalFilter(config)
    filtered = filter_instance.filter_by_strength(signals)
    filtered = filter_instance.filter_by_risk(filtered, max_risk)

    return filtered
