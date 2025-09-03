"""
信号生成器模块
负责从策略输出生成标准化交易信号
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from common.constants import MAX_POSITION_PCT, MIN_POSITION_SIZE
from common.data_structures import MarketData, Signal
from common.exceptions import ExecutionError
from common.logging_system import setup_logger

logger = setup_logger("signal_generator")


class SignalType(Enum):
    """信号类型枚举"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class SignalPriority(Enum):
    """信号优先级枚举"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class EnhancedSignal(Signal):
    """增强信号数据结构"""

    signal_type: SignalType = SignalType.LIMIT
    priority: SignalPriority = SignalPriority.NORMAL
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    expire_time: Optional[datetime] = None
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    routing_preference: str = "SMART"

    def validate(self) -> bool:
        """验证信号有效性

        Returns:
            是否有效
        """
        # 基本验证
        if self.quantity <= 0:
            return False
        if self.confidence < 0 or self.confidence > 1:
            return False
        if self.action not in ["BUY", "SELL", "HOLD"]:
            return False

        # 价格验证
        if self.signal_type == SignalType.LIMIT and self.limit_price is None:
            return False
        if (
            self.signal_type in [SignalType.STOP, SignalType.STOP_LIMIT]
            and self.stop_price is None
        ):
            return False

        return True


@dataclass
class SignalAggregation:
    """信号聚合结果"""

    symbol: str
    net_action: str  # 最终动作
    net_quantity: int  # 净数量
    avg_confidence: float
    signal_count: int
    individual_signals: List[EnhancedSignal]
    aggregation_method: str
    timestamp: datetime


class SignalGenerator:
    """信号生成器类"""

    def __init__(self, config: Dict[str, Any]):
        """初始化信号生成器

        Args:
            config: 配置字典
        """
        self.config = config
        self.min_confidence = config.get("min_confidence", 0.6)
        self.position_limits = config.get("position_limits", {})
        self.signal_buffer: List[EnhancedSignal] = []
        self.aggregation_window_seconds = config.get("aggregation_window", 5)
        self.conflict_resolution_method = config.get(
            "conflict_resolution", "weighted_average"
        )

    def generate_trading_signals(
        self,
        predictions: Dict[str, np.ndarray],
        market_data: Dict[str, MarketData],
        current_positions: Dict[str, float],
        risk_limits: Dict[str, float],
    ) -> List[EnhancedSignal]:
        """生成交易信号

        Args:
            predictions: 模型预测结果
            market_data: 当前市场数据
            current_positions: 当前持仓
            risk_limits: 风险限制

        Returns:
            生成的信号列表
        """
        signals = []

        for symbol, pred_array in predictions.items():
            if symbol not in market_data:
                logger.warning(
                    f"No market data for {symbol}, skipping signal generation"
                )
                continue

            # 解析预测结果
            signal_strength = float(pred_array[0]) if len(pred_array) > 0 else 0
            confidence = float(pred_array[1]) if len(pred_array) > 1 else 0.5

            # 检查置信度阈值
            if confidence < self.min_confidence:
                continue

            # 确定动作
            action = self._determine_action(
                signal_strength,
                current_positions.get(symbol, 0),
                risk_limits.get(symbol, float("inf")),
            )

            if action == "HOLD":
                continue

            # 计算数量
            quantity = self._calculate_position_size(
                symbol,
                action,
                signal_strength,
                confidence,
                market_data[symbol],
                current_positions.get(symbol, 0),
                risk_limits.get(symbol, float("inf")),
            )

            if quantity < MIN_POSITION_SIZE:
                continue

            # 确定价格和类型
            signal_type, price_params = self._determine_order_params(
                action, market_data[symbol], signal_strength, confidence
            )

            # 创建信号
            signal = EnhancedSignal(
                signal_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=market_data[symbol].close,
                confidence=confidence,
                strategy_name=self.config.get("strategy_name", "default"),
                metadata={
                    "signal_strength": signal_strength,
                    "market_price": market_data[symbol].close,
                    "bid": market_data[symbol].bid,
                    "ask": market_data[symbol].ask,
                    "spread": market_data[symbol].ask - market_data[symbol].bid
                    if market_data[symbol].bid and market_data[symbol].ask
                    else 0,
                },
                signal_type=signal_type,
                **price_params,
            )

            if signal.validate():
                signals.append(signal)
                logger.info(
                    f"Generated signal: {signal.symbol} {signal.action} {signal.quantity} @ {signal.price}"
                )
            else:
                logger.warning(f"Invalid signal generated for {symbol}")

        return signals

    def combine_multi_strategy_signals(
        self,
        strategy_signals: Dict[str, List[EnhancedSignal]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[EnhancedSignal]:
        """组合多策略信号

        Args:
            strategy_signals: 策略名到信号列表的映射
            weights: 策略权重

        Returns:
            组合后的信号列表
        """
        if not strategy_signals:
            return []

        # 默认等权重
        if weights is None:
            weights = {name: 1.0 / len(strategy_signals) for name in strategy_signals}

        # 按symbol分组
        symbol_signals: Dict[str, List[Tuple[str, EnhancedSignal]]] = {}
        for strategy_name, signals in strategy_signals.items():
            for signal in signals:
                if signal.symbol not in symbol_signals:
                    symbol_signals[signal.symbol] = []
                symbol_signals[signal.symbol].append((strategy_name, signal))

        # 聚合每个symbol的信号
        combined_signals = []
        for symbol, signal_pairs in symbol_signals.items():
            aggregated = self._aggregate_signals(symbol, signal_pairs, weights)
            if aggregated:
                combined_signals.append(aggregated)

        return combined_signals

    def filter_signal_quality(
        self, signals: List[EnhancedSignal], quality_metrics: Dict[str, float]
    ) -> List[EnhancedSignal]:
        """过滤信号质量

        Args:
            signals: 原始信号列表
            quality_metrics: 质量指标

        Returns:
            过滤后的信号列表
        """
        filtered_signals = []

        for signal in signals:
            # 检查信号质量
            if self._check_signal_quality(signal, quality_metrics):
                filtered_signals.append(signal)
            else:
                logger.debug(f"Signal filtered out: {signal.symbol} {signal.action}")

        return filtered_signals

    def calculate_signal_confidence(
        self,
        prediction_scores: List[float],
        historical_accuracy: float,
        market_conditions: Dict[str, float],
    ) -> float:
        """计算信号置信度

        Args:
            prediction_scores: 预测分数列表
            historical_accuracy: 历史准确率
            market_conditions: 市场条件

        Returns:
            综合置信度
        """
        # 基础置信度（预测分数均值）
        base_confidence = np.mean(prediction_scores) if prediction_scores else 0.5

        # 历史准确率调整
        accuracy_weight = 0.3
        confidence = (
            base_confidence * (1 - accuracy_weight)
            + historical_accuracy * accuracy_weight
        )

        # 市场条件调整
        volatility = market_conditions.get("volatility", 1.0)
        if volatility > 2.0:  # 高波动环境
            confidence *= 0.8

        trend_strength = market_conditions.get("trend_strength", 0.5)
        confidence *= 0.7 + 0.6 * trend_strength  # 趋势越强，置信度越高

        return max(0.0, min(1.0, confidence))

    def prioritize_execution_queue(
        self, signals: List[EnhancedSignal]
    ) -> List[EnhancedSignal]:
        """优先级排序执行队列

        Args:
            signals: 信号列表

        Returns:
            排序后的信号列表
        """
        # 计算每个信号的综合得分
        signal_scores = []
        for signal in signals:
            score = self._calculate_priority_score(signal)
            signal_scores.append((signal, score))

        # 按得分排序
        signal_scores.sort(key=lambda x: x[1], reverse=True)

        # 设置优先级
        sorted_signals = []
        for i, (signal, score) in enumerate(signal_scores):
            if i < len(signal_scores) * 0.1:  # 前10%
                signal.priority = SignalPriority.CRITICAL
            elif i < len(signal_scores) * 0.3:  # 前30%
                signal.priority = SignalPriority.HIGH
            elif i < len(signal_scores) * 0.6:  # 前60%
                signal.priority = SignalPriority.NORMAL
            else:
                signal.priority = SignalPriority.LOW

            sorted_signals.append(signal)

        return sorted_signals

    def _determine_action(
        self, signal_strength: float, current_position: float, risk_limit: float
    ) -> str:
        """确定交易动作

        Args:
            signal_strength: 信号强度 (-1到1)
            current_position: 当前持仓
            risk_limit: 风险限制

        Returns:
            动作: BUY, SELL, 或 HOLD
        """
        # 强买入信号
        if signal_strength > 0.3:
            if current_position < risk_limit:
                return "BUY"
            else:
                return "HOLD"  # 已达风险限制

        # 强卖出信号
        elif signal_strength < -0.3:
            if current_position > 0:
                return "SELL"
            else:
                return "HOLD"  # 无持仓可卖

        # 弱信号
        else:
            return "HOLD"

    def _calculate_position_size(
        self,
        symbol: str,
        action: str,
        signal_strength: float,
        confidence: float,
        market_data: MarketData,
        current_position: float,
        risk_limit: float,
    ) -> int:
        """计算仓位大小

        Args:
            symbol: 标的代码
            action: 交易动作
            signal_strength: 信号强度
            confidence: 置信度
            market_data: 市场数据
            current_position: 当前持仓
            risk_limit: 风险限制

        Returns:
            仓位数量
        """
        # 基础仓位计算
        base_size = self.config.get("base_position_size", 1000)

        # 根据信号强度调整
        strength_multiplier = abs(signal_strength) * 2  # 0到2倍

        # 根据置信度调整
        confidence_multiplier = 0.5 + confidence  # 0.5到1.5倍

        # 计算目标仓位
        target_size = base_size * strength_multiplier * confidence_multiplier

        # 应用风险限制
        if action == "BUY":
            max_additional = risk_limit - current_position
            target_size = min(target_size, max_additional)
        elif action == "SELL":
            target_size = min(target_size, current_position)

        # 应用最大仓位百分比限制
        if symbol in self.position_limits:
            target_size = min(target_size, self.position_limits[symbol])

        # 取整到100股
        return int(target_size // 100) * 100

    def _determine_order_params(
        self,
        action: str,
        market_data: MarketData,
        signal_strength: float,
        confidence: float,
    ) -> Tuple[SignalType, Dict[str, Any]]:
        """确定订单参数

        Args:
            action: 交易动作
            market_data: 市场数据
            signal_strength: 信号强度
            confidence: 置信度

        Returns:
            (信号类型, 价格参数字典)
        """
        params = {}

        # 高置信度使用市价单
        if confidence > 0.85:
            return SignalType.MARKET, params

        # 中等置信度使用限价单
        spread = (
            market_data.ask - market_data.bid
            if market_data.ask and market_data.bid
            else 0.01
        )

        if action == "BUY":
            # 买入时，限价略高于买价
            params["limit_price"] = (
                market_data.bid + spread * 0.2 if market_data.bid else market_data.close
            )
        else:
            # 卖出时，限价略低于卖价
            params["limit_price"] = (
                market_data.ask - spread * 0.2 if market_data.ask else market_data.close
            )

        # 设置止损
        if abs(signal_strength) < 0.5:  # 信号不够强，设置止损
            stop_distance = market_data.close * 0.02  # 2%止损
            if action == "BUY":
                params["stop_price"] = market_data.close - stop_distance
            else:
                params["stop_price"] = market_data.close + stop_distance
            return SignalType.STOP_LIMIT, params

        return SignalType.LIMIT, params

    def _aggregate_signals(
        self,
        symbol: str,
        signal_pairs: List[Tuple[str, EnhancedSignal]],
        weights: Dict[str, float],
    ) -> Optional[EnhancedSignal]:
        """聚合同一标的的多个信号

        Args:
            symbol: 标的代码
            signal_pairs: (策略名, 信号)列表
            weights: 策略权重

        Returns:
            聚合后的信号
        """
        if not signal_pairs:
            return None

        # 分离买入和卖出信号
        buy_signals = [(name, sig) for name, sig in signal_pairs if sig.action == "BUY"]
        sell_signals = [
            (name, sig) for name, sig in signal_pairs if sig.action == "SELL"
        ]

        # 计算加权信号强度
        buy_strength = sum(weights[name] * sig.confidence for name, sig in buy_signals)
        sell_strength = sum(
            weights[name] * sig.confidence for name, sig in sell_signals
        )

        # 确定最终动作
        if buy_strength > sell_strength and buy_strength > 0.5:
            action = "BUY"
            relevant_signals = buy_signals
            net_confidence = buy_strength
        elif sell_strength > buy_strength and sell_strength > 0.5:
            action = "SELL"
            relevant_signals = sell_signals
            net_confidence = sell_strength
        else:
            return None  # 信号冲突或太弱

        # 计算加权平均数量
        total_weight = sum(weights[name] for name, _ in relevant_signals)
        weighted_quantity = (
            sum(weights[name] * sig.quantity for name, sig in relevant_signals)
            / total_weight
            if total_weight > 0
            else 0
        )

        # 创建聚合信号
        aggregated_signal = EnhancedSignal(
            signal_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            symbol=symbol,
            action=action,
            quantity=int(weighted_quantity),
            price=relevant_signals[0][1].price,  # 使用第一个信号的价格
            confidence=min(1.0, net_confidence),
            strategy_name="aggregated",
            metadata={
                "aggregation_method": self.conflict_resolution_method,
                "source_strategies": [name for name, _ in signal_pairs],
                "signal_count": len(signal_pairs),
            },
        )

        return aggregated_signal

    def _check_signal_quality(
        self, signal: EnhancedSignal, quality_metrics: Dict[str, float]
    ) -> bool:
        """检查信号质量

        Args:
            signal: 信号对象
            quality_metrics: 质量指标

        Returns:
            是否通过质量检查
        """
        # 检查最小置信度
        min_confidence = quality_metrics.get("min_confidence", 0.5)
        if signal.confidence < min_confidence:
            return False

        # 检查信号年龄
        max_age_seconds = quality_metrics.get("max_signal_age", 30)
        signal_age = (datetime.now() - signal.timestamp).total_seconds()
        if signal_age > max_age_seconds:
            return False

        # 检查价格偏离
        if "market_price" in signal.metadata:
            market_price = signal.metadata["market_price"]
            price_deviation = abs(signal.price - market_price) / market_price
            max_deviation = quality_metrics.get("max_price_deviation", 0.01)
            if price_deviation > max_deviation:
                return False

        return True

    def _calculate_priority_score(self, signal: EnhancedSignal) -> float:
        """计算信号优先级得分

        Args:
            signal: 信号对象

        Returns:
            优先级得分
        """
        score = 0.0

        # 置信度贡献 (0-40分)
        score += signal.confidence * 40

        # 信号强度贡献 (0-30分)
        if "signal_strength" in signal.metadata:
            score += abs(signal.metadata["signal_strength"]) * 30

        # 市价单优先级更高 (0-20分)
        if signal.signal_type == SignalType.MARKET:
            score += 20
        elif signal.signal_type == SignalType.LIMIT:
            score += 10

        # 大单优先级更高 (0-10分)
        normalized_quantity = min(signal.quantity / 10000, 1.0)
        score += normalized_quantity * 10

        return score


# 模块级别函数
def create_signal_generator(config: Dict[str, Any]) -> SignalGenerator:
    """创建信号生成器实例

    Args:
        config: 配置字典

    Returns:
        信号生成器实例
    """
    return SignalGenerator(config)
