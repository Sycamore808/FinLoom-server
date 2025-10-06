"""
自适应止损策略模块
实现动态调整的止损机制
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.constants import TRADING_DAYS_PER_YEAR
from common.exceptions import ModelError
from common.logging_system import setup_logger

logger = setup_logger("adaptive_stop_loss")


class StopLossType(Enum):
    """止损类型枚举"""

    FIXED = "fixed"
    TRAILING = "trailing"
    VOLATILITY_BASED = "volatility_based"
    TIME_BASED = "time_based"
    CHANDELIER = "chandelier"
    PARABOLIC_SAR = "parabolic_sar"
    DYNAMIC = "dynamic"


@dataclass
class StopLossConfig:
    """止损配置"""

    default_stop_pct: float = 0.02  # 默认止损百分比
    trailing_stop_pct: float = 0.03  # 追踪止损百分比
    atr_multiplier: float = 2.0  # ATR倍数
    volatility_window: int = 20  # 波动率窗口
    sar_acceleration: float = 0.02  # SAR加速因子
    sar_maximum: float = 0.2  # SAR最大值
    time_stop_days: int = 30  # 时间止损天数
    profit_target_ratio: float = 3.0  # 盈利目标倍数
    use_adaptive: bool = True  # 使用自适应止损
    regime_adjustment: bool = True  # 根据市场状态调整


@dataclass
class StopLossOrder:
    """止损订单"""

    order_id: str
    symbol: str
    position_size: float
    entry_price: float
    stop_price: float
    stop_type: StopLossType
    created_time: datetime
    last_update: datetime
    triggered: bool
    trigger_price: Optional[float]
    metrics: Dict[str, float]


@dataclass
class StopLossResult:
    """止损结果"""

    symbol: str
    entry_price: float
    stop_price: float
    current_price: float
    stop_distance: float
    stop_distance_pct: float
    should_stop: bool
    stop_type: StopLossType
    confidence: float
    risk_metrics: Dict[str, float]


class AdaptiveStopLoss:
    """自适应止损策略"""

    def __init__(self, config: Optional[StopLossConfig] = None):
        """初始化自适应止损

        Args:
            config: 止损配置
        """
        self.config = config or StopLossConfig()
        self.active_stops: Dict[str, StopLossOrder] = {}
        self.stop_history: List[StopLossOrder] = []
        self.market_regime: str = "normal"  # normal, volatile, trending

    def set_adaptive_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        market_data: pd.DataFrame,
        position_type: str = "long",
    ) -> StopLossResult:
        """设置自适应止损

        Args:
            symbol: 标的代码
            entry_price: 入场价格
            current_price: 当前价格
            market_data: 市场数据
            position_type: 仓位类型 ("long" 或 "short")

        Returns:
            止损结果
        """
        logger.info(f"Setting adaptive stop loss for {symbol}")

        # 计算各种止损价格
        stop_prices = {}

        # 1. 固定止损
        stop_prices["fixed"] = self._calculate_fixed_stop(entry_price, position_type)

        # 2. 追踪止损
        stop_prices["trailing"] = self._calculate_trailing_stop(
            entry_price, current_price, position_type
        )

        # 3. 波动率止损
        if "close" in market_data.columns:
            stop_prices["volatility"] = self._calculate_volatility_stop(
                current_price, market_data["close"], position_type
            )

        # 4. ATR止损
        if all(col in market_data.columns for col in ["high", "low", "close"]):
            stop_prices["atr"] = self._calculate_atr_stop(
                current_price, market_data, position_type
            )

        # 5. Chandelier止损
        if all(col in market_data.columns for col in ["high", "low", "close"]):
            stop_prices["chandelier"] = self._calculate_chandelier_stop(
                market_data, position_type
            )

        # 选择最优止损
        if self.config.use_adaptive:
            optimal_stop, stop_type = self._select_optimal_stop(
                stop_prices, current_price, position_type
            )
        else:
            optimal_stop = stop_prices.get("fixed", entry_price * 0.98)
            stop_type = StopLossType.FIXED

        # 根据市场状态调整
        if self.config.regime_adjustment:
            optimal_stop = self._adjust_for_regime(
                optimal_stop, current_price, position_type
            )

        # 计算止损距离
        if position_type == "long":
            stop_distance = current_price - optimal_stop
            should_stop = current_price <= optimal_stop
        else:
            stop_distance = optimal_stop - current_price
            should_stop = current_price >= optimal_stop

        stop_distance_pct = abs(stop_distance / current_price)

        # 计算风险指标
        risk_metrics = self._calculate_risk_metrics(
            entry_price, current_price, optimal_stop, position_type
        )

        result = StopLossResult(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=optimal_stop,
            current_price=current_price,
            stop_distance=stop_distance,
            stop_distance_pct=stop_distance_pct,
            should_stop=should_stop,
            stop_type=stop_type,
            confidence=self._calculate_stop_confidence(stop_prices, optimal_stop),
            risk_metrics=risk_metrics,
        )

        # 更新活跃止损订单
        self._update_active_stop(symbol, entry_price, optimal_stop, stop_type)

        return result

    def implement_trailing_stop(
        self,
        symbol: str,
        entry_price: float,
        highest_price: float,
        current_price: float,
        trailing_pct: Optional[float] = None,
    ) -> Tuple[float, bool]:
        """实现追踪止损

        Args:
            symbol: 标的代码
            entry_price: 入场价格
            highest_price: 历史最高价
            current_price: 当前价格
            trailing_pct: 追踪百分比

        Returns:
            (止损价格, 是否触发)
        """
        trailing_pct = trailing_pct or self.config.trailing_stop_pct

        # 计算追踪止损价
        stop_price = highest_price * (1 - trailing_pct)

        # 确保止损价不低于入场价的固定止损
        min_stop = entry_price * (1 - self.config.default_stop_pct)
        stop_price = max(stop_price, min_stop)

        # 检查是否触发
        triggered = current_price <= stop_price

        logger.info(
            f"Trailing stop for {symbol}: {stop_price:.2f}, Triggered: {triggered}"
        )

        return stop_price, triggered

    def calculate_volatility_based_stops(
        self, price_series: pd.Series, current_price: float, method: str = "std"
    ) -> Dict[str, float]:
        """计算基于波动率的止损

        Args:
            price_series: 价格序列
            current_price: 当前价格
            method: 方法 ("std", "atr", "percentage")

        Returns:
            止损价格字典
        """
        stops = {}

        if method in ["std", "all"]:
            # 标准差止损
            returns = price_series.pct_change()
            volatility = returns.std()
            stops["std_stop"] = current_price * (1 - 2 * volatility)

        if method in ["atr", "all"]:
            # ATR止损
            if len(price_series) > 14:
                high = price_series.rolling(window=1).max()
                low = price_series.rolling(window=1).min()
                close = price_series

                tr = pd.concat(
                    [high - low, abs(high - close.shift()), abs(low - close.shift())],
                    axis=1,
                ).max(axis=1)

                atr = tr.rolling(window=14).mean().iloc[-1]
                stops["atr_stop"] = current_price - self.config.atr_multiplier * atr

        if method in ["percentage", "all"]:
            # 百分比范围止损
            recent_range = price_series.tail(20)
            range_pct = (recent_range.max() - recent_range.min()) / recent_range.mean()
            stops["range_stop"] = current_price * (1 - range_pct)

        return stops

    def monitor_stop_triggers(
        self, current_prices: Dict[str, float]
    ) -> List[StopLossOrder]:
        """监控止损触发

        Args:
            current_prices: 当前价格字典

        Returns:
            触发的止损订单列表
        """
        triggered_orders = []

        for symbol, stop_order in self.active_stops.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]

                # 检查是否触发
                if current_price <= stop_order.stop_price:
                    stop_order.triggered = True
                    stop_order.trigger_price = current_price
                    stop_order.last_update = datetime.now()

                    triggered_orders.append(stop_order)

                    logger.warning(
                        f"Stop loss triggered for {symbol} at {current_price:.2f} "
                        f"(stop: {stop_order.stop_price:.2f})"
                    )

        # 移除触发的订单
        for order in triggered_orders:
            del self.active_stops[order.symbol]
            self.stop_history.append(order)

        return triggered_orders

    def execute_emergency_liquidation(
        self, positions: Dict[str, float], reason: str = "Risk limit breach"
    ) -> Dict[str, Dict[str, Any]]:
        """执行紧急平仓

        Args:
            positions: 持仓字典
            reason: 平仓原因

        Returns:
            平仓执行结果
        """
        logger.critical(f"Executing emergency liquidation: {reason}")

        liquidation_results = {}

        for symbol, position_size in positions.items():
            liquidation_results[symbol] = {
                "position_size": position_size,
                "liquidation_time": datetime.now(),
                "reason": reason,
                "status": "pending",
            }

            # 创建紧急止损订单
            if symbol in self.active_stops:
                # 更新现有止损为市价单
                self.active_stops[symbol].stop_price = 0  # 市价
                self.active_stops[symbol].stop_type = StopLossType.FIXED
                self.active_stops[symbol].metrics["emergency"] = True
            else:
                # 创建新的紧急止损
                emergency_stop = StopLossOrder(
                    order_id=f"emergency_{symbol}_{datetime.now().timestamp()}",
                    symbol=symbol,
                    position_size=position_size,
                    entry_price=0,  # Unknown
                    stop_price=0,  # Market order
                    stop_type=StopLossType.FIXED,
                    created_time=datetime.now(),
                    last_update=datetime.now(),
                    triggered=True,
                    trigger_price=None,
                    metrics={"emergency": True, "reason": reason},
                )
                self.active_stops[symbol] = emergency_stop

            liquidation_results[symbol]["status"] = "submitted"

        return liquidation_results

    def _calculate_fixed_stop(self, entry_price: float, position_type: str) -> float:
        """计算固定止损

        Args:
            entry_price: 入场价格
            position_type: 仓位类型

        Returns:
            止损价格
        """
        if position_type == "long":
            return entry_price * (1 - self.config.default_stop_pct)
        else:
            return entry_price * (1 + self.config.default_stop_pct)

    def _calculate_trailing_stop(
        self, entry_price: float, current_price: float, position_type: str
    ) -> float:
        """计算追踪止损

        Args:
            entry_price: 入场价格
            current_price: 当前价格
            position_type: 仓位类型

        Returns:
            止损价格
        """
        if position_type == "long":
            # 使用当前价格或入场价格中的较高者
            reference_price = max(current_price, entry_price)
            return reference_price * (1 - self.config.trailing_stop_pct)
        else:
            # 使用当前价格或入场价格中的较低者
            reference_price = min(current_price, entry_price)
            return reference_price * (1 + self.config.trailing_stop_pct)

    def _calculate_volatility_stop(
        self, current_price: float, price_series: pd.Series, position_type: str
    ) -> float:
        """计算波动率止损

        Args:
            current_price: 当前价格
            price_series: 价格序列
            position_type: 仓位类型

        Returns:
            止损价格
        """
        returns = price_series.pct_change()
        volatility = (
            returns.rolling(window=self.config.volatility_window).std().iloc[-1]
        )

        if position_type == "long":
            return current_price * (1 - 2 * volatility)
        else:
            return current_price * (1 + 2 * volatility)

    def _calculate_atr_stop(
        self, current_price: float, market_data: pd.DataFrame, position_type: str
    ) -> float:
        """计算ATR止损

        Args:
            current_price: 当前价格
            market_data: 市场数据
            position_type: 仓位类型

        Returns:
            止损价格
        """
        # 计算ATR
        high = market_data["high"]
        low = market_data["low"]
        close = market_data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]

        if position_type == "long":
            return current_price - self.config.atr_multiplier * atr
        else:
            return current_price + self.config.atr_multiplier * atr

    def _calculate_chandelier_stop(
        self, market_data: pd.DataFrame, position_type: str, period: int = 22
    ) -> float:
        """计算Chandelier止损

        Args:
            market_data: 市场数据
            position_type: 仓位类型
            period: 周期

        Returns:
            止损价格
        """
        high = market_data["high"]
        low = market_data["low"]
        close = market_data["close"]

        # 计算ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        if position_type == "long":
            # 长仓：最高价 - ATR * 倍数
            highest = high.rolling(window=period).max().iloc[-1]
            return highest - self.config.atr_multiplier * atr
        else:
            # 短仓：最低价 + ATR * 倍数
            lowest = low.rolling(window=period).min().iloc[-1]
            return lowest + self.config.atr_multiplier * atr

    def _select_optimal_stop(
        self, stop_prices: Dict[str, float], current_price: float, position_type: str
    ) -> Tuple[float, StopLossType]:
        """选择最优止损

        Args:
            stop_prices: 各种止损价格
            current_price: 当前价格
            position_type: 仓位类型

        Returns:
            (最优止损价格, 止损类型)
        """
        # 根据市场状态选择
        if self.market_regime == "volatile":
            # 波动市场使用更宽的止损
            if "atr" in stop_prices:
                return stop_prices["atr"], StopLossType.VOLATILITY_BASED
            elif "volatility" in stop_prices:
                return stop_prices["volatility"], StopLossType.VOLATILITY_BASED

        elif self.market_regime == "trending":
            # 趋势市场使用追踪止损
            if "trailing" in stop_prices:
                return stop_prices["trailing"], StopLossType.TRAILING
            elif "chandelier" in stop_prices:
                return stop_prices["chandelier"], StopLossType.CHANDELIER

        # 默认选择最保守的（最接近当前价格的）
        if position_type == "long":
            # 长仓选择最高的止损价
            optimal_stop = max(stop_prices.values())
        else:
            # 短仓选择最低的止损价
            optimal_stop = min(stop_prices.values())

        # 找到对应的类型
        for stop_type, price in stop_prices.items():
            if price == optimal_stop:
                return optimal_stop, self._map_stop_type(stop_type)

        return optimal_stop, StopLossType.DYNAMIC

    def _map_stop_type(self, type_str: str) -> StopLossType:
        """映射止损类型

        Args:
            type_str: 类型字符串

        Returns:
            止损类型枚举
        """
        mapping = {
            "fixed": StopLossType.FIXED,
            "trailing": StopLossType.TRAILING,
            "volatility": StopLossType.VOLATILITY_BASED,
            "atr": StopLossType.VOLATILITY_BASED,
            "chandelier": StopLossType.CHANDELIER,
        }
        return mapping.get(type_str, StopLossType.DYNAMIC)

    def _adjust_for_regime(
        self, stop_price: float, current_price: float, position_type: str
    ) -> float:
        """根据市场状态调整止损

        Args:
            stop_price: 原始止损价格
            current_price: 当前价格
            position_type: 仓位类型

        Returns:
            调整后的止损价格
        """
        adjustment_factor = 1.0

        if self.market_regime == "volatile":
            # 波动市场放宽止损
            adjustment_factor = 0.9
        elif self.market_regime == "trending":
            # 趋势市场收紧止损
            adjustment_factor = 1.1

        if position_type == "long":
            # 调整止损距离
            stop_distance = current_price - stop_price
            adjusted_distance = stop_distance / adjustment_factor
            return current_price - adjusted_distance
        else:
            stop_distance = stop_price - current_price
            adjusted_distance = stop_distance / adjustment_factor
            return current_price + adjusted_distance

    def _calculate_risk_metrics(
        self,
        entry_price: float,
        current_price: float,
        stop_price: float,
        position_type: str,
    ) -> Dict[str, float]:
        """计算风险指标

        Args:
            entry_price: 入场价格
            current_price: 当前价格
            stop_price: 止损价格
            position_type: 仓位类型

        Returns:
            风险指标字典
        """
        metrics = {}

        if position_type == "long":
            # 计算风险和收益
            risk = (entry_price - stop_price) / entry_price
            current_pnl = (current_price - entry_price) / entry_price
            max_loss = (stop_price - entry_price) / entry_price
        else:
            risk = (stop_price - entry_price) / entry_price
            current_pnl = (entry_price - current_price) / entry_price
            max_loss = (entry_price - stop_price) / entry_price

        metrics["risk_pct"] = abs(risk)
        metrics["current_pnl_pct"] = current_pnl
        metrics["max_loss_pct"] = max_loss

        # 风险回报比
        if risk != 0:
            potential_reward = self.config.profit_target_ratio * abs(risk)
            metrics["risk_reward_ratio"] = potential_reward / abs(risk)
        else:
            metrics["risk_reward_ratio"] = 0

        # 到止损的距离
        metrics["distance_to_stop_pct"] = (
            abs(current_price - stop_price) / current_price
        )

        return metrics

    def _calculate_stop_confidence(
        self, stop_prices: Dict[str, float], selected_stop: float
    ) -> float:
        """计算止损置信度

        Args:
            stop_prices: 各种止损价格
            selected_stop: 选定的止损价格

        Returns:
            置信度分数
        """
        if not stop_prices:
            return 0.5

        # 计算所有止损价格的一致性
        prices = list(stop_prices.values())
        mean_stop = np.mean(prices)
        std_stop = np.std(prices)

        if std_stop > 0:
            # 变异系数越小，一致性越高
            consistency = 1 / (1 + std_stop / mean_stop)
        else:
            consistency = 1.0

        return consistency

    def _update_active_stop(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        stop_type: StopLossType,
    ) -> None:
        """更新活跃止损订单

        Args:
            symbol: 标的代码
            entry_price: 入场价格
            stop_price: 止损价格
            stop_type: 止损类型
        """
        if symbol in self.active_stops:
            # 更新现有订单
            self.active_stops[symbol].stop_price = stop_price
            self.active_stops[symbol].stop_type = stop_type
            self.active_stops[symbol].last_update = datetime.now()
        else:
            # 创建新订单
            order = StopLossOrder(
                order_id=f"{symbol}_{datetime.now().timestamp()}",
                symbol=symbol,
                position_size=0,  # 需要外部设置
                entry_price=entry_price,
                stop_price=stop_price,
                stop_type=stop_type,
                created_time=datetime.now(),
                last_update=datetime.now(),
                triggered=False,
                trigger_price=None,
                metrics={},
            )
            self.active_stops[symbol] = order


# 模块级别函数
def calculate_adaptive_stop(
    symbol: str,
    entry_price: float,
    current_price: float,
    market_data: pd.DataFrame,
    config: Optional[StopLossConfig] = None,
) -> Dict[str, Any]:
    """计算自适应止损的便捷函数

    Args:
        symbol: 标的代码
        entry_price: 入场价格
        current_price: 当前价格
        market_data: 市场数据
        config: 止损配置

    Returns:
        止损信息字典
    """
    stop_loss = AdaptiveStopLoss(config)
    result = stop_loss.set_adaptive_stop_loss(
        symbol, entry_price, current_price, market_data
    )

    return {
        "stop_price": result.stop_price,
        "stop_distance_pct": result.stop_distance_pct,
        "should_stop": result.should_stop,
        "stop_type": result.stop_type.value,
        "confidence": result.confidence,
    }
