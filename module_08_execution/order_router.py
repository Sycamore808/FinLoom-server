"""
订单路由器模块
负责智能订单路由和执行策略选择
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from common.exceptions import ExecutionError
from common.logging_system import setup_logger
from module_08_execution.order_manager import Order, OrderType

logger = setup_logger("order_router")


class ExecutionVenue(Enum):
    """执行场所枚举"""

    EXCHANGE = "EXCHANGE"  # 交易所
    DARK_POOL = "DARK_POOL"  # 暗池
    BROKER_INTERNAL = "BROKER_INTERNAL"  # 券商内部
    SMART_ROUTER = "SMART_ROUTER"  # 智能路由


class ExecutionStrategy(Enum):
    """执行策略枚举"""

    AGGRESSIVE = "AGGRESSIVE"  # 激进执行
    PASSIVE = "PASSIVE"  # 被动执行
    MARKET_ON_CLOSE = "MARKET_ON_CLOSE"  # 收盘价
    TWAP = "TWAP"  # 时间加权平均价格
    VWAP = "VWAP"  # 成交量加权平均价格
    ICEBERG = "ICEBERG"  # 冰山订单
    IMPLEMENTATION_SHORTFALL = "IS"  # 实施缺口


@dataclass
class RoutingDecision:
    """路由决策"""

    order_id: str
    venue: ExecutionVenue
    strategy: ExecutionStrategy
    priority: int  # 0-10
    estimated_cost_bps: float  # 预估成本（基点）
    estimated_slippage_bps: float  # 预估滑点（基点）
    estimated_execution_time: int  # 预估执行时间（秒）
    reasoning: str  # 决策理由
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class RouterConfig:
    """路由器配置"""

    default_venue: ExecutionVenue = ExecutionVenue.EXCHANGE
    default_strategy: ExecutionStrategy = ExecutionStrategy.TWAP
    enable_smart_routing: bool = True
    cost_weight: float = 0.4  # 成本权重
    speed_weight: float = 0.3  # 速度权重
    quality_weight: float = 0.3  # 质量权重
    max_slippage_bps: float = 10.0  # 最大可接受滑点
    prefer_venue_for_large_orders: bool = True
    large_order_threshold: int = 10000  # 大单阈值


class OrderRouter:
    """订单路由器类"""

    def __init__(self, config: Optional[RouterConfig] = None):
        """初始化订单路由器

        Args:
            config: 路由器配置
        """
        self.config = config or RouterConfig()
        self.routing_history: List[RoutingDecision] = []
        self.venue_performance: Dict[ExecutionVenue, Dict[str, float]] = {}

        # 初始化场所性能指标
        for venue in ExecutionVenue:
            self.venue_performance[venue] = {
                "avg_cost_bps": 5.0,
                "avg_slippage_bps": 2.0,
                "avg_execution_time": 60.0,
                "fill_rate": 0.95,
                "reliability": 0.98,
            }

    def route_order(
        self,
        order: Order,
        market_conditions: Optional[Dict[str, Any]] = None,
        urgency: float = 0.5,
    ) -> RoutingDecision:
        """路由订单到最优执行场所

        Args:
            order: 订单对象
            market_conditions: 市场条件
            urgency: 紧急程度 (0-1)

        Returns:
            路由决策
        """
        try:
            logger.info(f"Routing order {order.order_id} for {order.symbol}")

            # 分析订单特征
            order_features = self._analyze_order(order)

            # 评估各个场所
            venue_scores = self._evaluate_venues(
                order, order_features, market_conditions, urgency
            )

            # 选择最优场所
            best_venue = max(venue_scores.items(), key=lambda x: x[1]["score"])
            venue = best_venue[0]
            venue_info = best_venue[1]

            # 选择执行策略
            strategy = self._select_execution_strategy(
                order, order_features, venue, urgency
            )

            # 创建路由决策
            decision = RoutingDecision(
                order_id=order.order_id,
                venue=venue,
                strategy=strategy,
                priority=self._calculate_priority(order, urgency),
                estimated_cost_bps=venue_info["estimated_cost_bps"],
                estimated_slippage_bps=venue_info["estimated_slippage_bps"],
                estimated_execution_time=venue_info["estimated_execution_time"],
                reasoning=venue_info["reasoning"],
                metadata={
                    "order_features": order_features,
                    "urgency": urgency,
                    "venue_score": venue_info["score"],
                },
                timestamp=datetime.now(),
            )

            # 记录路由历史
            self.routing_history.append(decision)

            logger.info(
                f"Routed order {order.order_id} to {venue.value} "
                f"using {strategy.value} strategy"
            )

            return decision

        except Exception as e:
            logger.error(f"Failed to route order {order.order_id}: {e}")
            raise ExecutionError(f"Order routing failed: {e}")

    def _analyze_order(self, order: Order) -> Dict[str, Any]:
        """分析订单特征

        Args:
            order: 订单对象

        Returns:
            订单特征字典
        """
        features = {
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "is_large_order": order.quantity > self.config.large_order_threshold,
            "has_limit_price": order.price is not None,
            "estimated_value": order.quantity * (order.price or 0),
        }

        # 分类订单大小
        if order.quantity < 1000:
            features["size_category"] = "SMALL"
        elif order.quantity < 10000:
            features["size_category"] = "MEDIUM"
        else:
            features["size_category"] = "LARGE"

        return features

    def _evaluate_venues(
        self,
        order: Order,
        order_features: Dict[str, Any],
        market_conditions: Optional[Dict[str, Any]],
        urgency: float,
    ) -> Dict[ExecutionVenue, Dict[str, Any]]:
        """评估各个执行场所

        Args:
            order: 订单对象
            order_features: 订单特征
            market_conditions: 市场条件
            urgency: 紧急程度

        Returns:
            场所评分字典
        """
        scores = {}

        for venue in ExecutionVenue:
            # 获取场所性能指标
            performance = self.venue_performance[venue]

            # 估算成本
            estimated_cost = self._estimate_venue_cost(
                venue, order_features, market_conditions
            )

            # 估算滑点
            estimated_slippage = self._estimate_venue_slippage(
                venue, order_features, market_conditions
            )

            # 估算执行时间
            estimated_time = self._estimate_execution_time(
                venue, order_features, urgency
            )

            # 计算综合评分
            score = self._calculate_venue_score(
                estimated_cost,
                estimated_slippage,
                estimated_time,
                performance["fill_rate"],
                urgency,
            )

            # 生成决策理由
            reasoning = self._generate_venue_reasoning(
                venue, score, estimated_cost, estimated_slippage, estimated_time
            )

            scores[venue] = {
                "score": score,
                "estimated_cost_bps": estimated_cost,
                "estimated_slippage_bps": estimated_slippage,
                "estimated_execution_time": estimated_time,
                "reasoning": reasoning,
            }

        return scores

    def _estimate_venue_cost(
        self,
        venue: ExecutionVenue,
        order_features: Dict[str, Any],
        market_conditions: Optional[Dict[str, Any]],
    ) -> float:
        """估算场所成本

        Args:
            venue: 执行场所
            order_features: 订单特征
            market_conditions: 市场条件

        Returns:
            预估成本（基点）
        """
        base_cost = self.venue_performance[venue]["avg_cost_bps"]

        # 根据订单大小调整
        if order_features["size_category"] == "LARGE":
            base_cost *= 1.2
        elif order_features["size_category"] == "SMALL":
            base_cost *= 0.9

        # 根据场所类型调整
        if venue == ExecutionVenue.DARK_POOL:
            base_cost *= 0.8  # 暗池通常成本较低
        elif venue == ExecutionVenue.BROKER_INTERNAL:
            base_cost *= 0.7  # 内部执行成本最低

        # 根据市场条件调整
        if market_conditions:
            volatility = market_conditions.get("volatility", 0.02)
            if volatility > 0.03:  # 高波动
                base_cost *= 1.3

        return base_cost

    def _estimate_venue_slippage(
        self,
        venue: ExecutionVenue,
        order_features: Dict[str, Any],
        market_conditions: Optional[Dict[str, Any]],
    ) -> float:
        """估算场所滑点

        Args:
            venue: 执行场所
            order_features: 订单特征
            market_conditions: 市场条件

        Returns:
            预估滑点（基点）
        """
        base_slippage = self.venue_performance[venue]["avg_slippage_bps"]

        # 根据订单大小调整
        if order_features["is_large_order"]:
            base_slippage *= 1.5

        # 根据订单类型调整
        if order_features["order_type"] == "MARKET":
            base_slippage *= 1.2

        # 根据场所类型调整
        if venue == ExecutionVenue.EXCHANGE:
            base_slippage *= 1.0  # 交易所标准滑点
        elif venue == ExecutionVenue.DARK_POOL:
            base_slippage *= 0.7  # 暗池滑点较低

        # 根据市场条件调整
        if market_conditions:
            spread = market_conditions.get("bid_ask_spread_bps", 5.0)
            base_slippage += spread * 0.5

        return base_slippage

    def _estimate_execution_time(
        self, venue: ExecutionVenue, order_features: Dict[str, Any], urgency: float
    ) -> int:
        """估算执行时间

        Args:
            venue: 执行场所
            order_features: 订单特征
            urgency: 紧急程度

        Returns:
            预估执行时间（秒）
        """
        base_time = self.venue_performance[venue]["avg_execution_time"]

        # 根据订单大小调整
        if order_features["size_category"] == "LARGE":
            base_time *= 2.0
        elif order_features["size_category"] == "SMALL":
            base_time *= 0.5

        # 根据紧急程度调整
        urgency_factor = 2.0 - urgency  # 高紧急度缩短时间
        base_time *= urgency_factor

        return int(base_time)

    def _calculate_venue_score(
        self,
        cost: float,
        slippage: float,
        exec_time: int,
        fill_rate: float,
        urgency: float,
    ) -> float:
        """计算场所综合评分

        Args:
            cost: 预估成本
            slippage: 预估滑点
            exec_time: 执行时间
            fill_rate: 成交率
            urgency: 紧急程度

        Returns:
            综合评分 (0-100)
        """
        # 成本评分 (越低越好)
        cost_score = max(0, 100 - cost * 5)

        # 滑点评分 (越低越好)
        slippage_score = max(0, 100 - slippage * 8)

        # 速度评分 (越快越好)
        speed_score = max(0, 100 - exec_time / 10)

        # 质量评分 (成交率越高越好)
        quality_score = fill_rate * 100

        # 根据紧急程度调整权重
        if urgency > 0.7:
            # 高紧急度：速度更重要
            speed_weight = 0.5
            cost_weight = 0.2
            quality_weight = 0.3
        else:
            # 低紧急度：成本更重要
            speed_weight = self.config.speed_weight
            cost_weight = self.config.cost_weight
            quality_weight = self.config.quality_weight

        # 计算加权评分
        total_score = (
            cost_score * cost_weight
            + slippage_score * cost_weight
            + speed_score * speed_weight
            + quality_score * quality_weight
        )

        return total_score

    def _select_execution_strategy(
        self,
        order: Order,
        order_features: Dict[str, Any],
        venue: ExecutionVenue,
        urgency: float,
    ) -> ExecutionStrategy:
        """选择执行策略

        Args:
            order: 订单对象
            order_features: 订单特征
            venue: 执行场所
            urgency: 紧急程度

        Returns:
            执行策略
        """
        # 高紧急度使用激进策略
        if urgency > 0.8:
            return ExecutionStrategy.AGGRESSIVE

        # 小单使用激进或被动策略
        if order_features["size_category"] == "SMALL":
            return (
                ExecutionStrategy.PASSIVE
                if urgency < 0.3
                else ExecutionStrategy.AGGRESSIVE
            )

        # 大单根据场所选择策略
        if order_features["size_category"] == "LARGE":
            if venue == ExecutionVenue.DARK_POOL:
                return ExecutionStrategy.ICEBERG
            else:
                return ExecutionStrategy.VWAP

        # 默认使用TWAP
        return ExecutionStrategy.TWAP

    def _calculate_priority(self, order: Order, urgency: float) -> int:
        """计算订单优先级

        Args:
            order: 订单对象
            urgency: 紧急程度

        Returns:
            优先级 (0-10)
        """
        # 基础优先级基于紧急程度
        priority = int(urgency * 10)

        # 市价单提高优先级
        if order.order_type == OrderType.MARKET:
            priority = min(10, priority + 2)

        return priority

    def _generate_venue_reasoning(
        self,
        venue: ExecutionVenue,
        score: float,
        cost: float,
        slippage: float,
        exec_time: int,
    ) -> str:
        """生成场所选择理由

        Args:
            venue: 执行场所
            score: 综合评分
            cost: 预估成本
            slippage: 预估滑点
            exec_time: 执行时间

        Returns:
            决策理由
        """
        reasoning = f"{venue.value}: score={score:.1f}, "
        reasoning += f"cost={cost:.2f}bps, "
        reasoning += f"slippage={slippage:.2f}bps, "
        reasoning += f"time={exec_time}s"

        return reasoning

    def update_venue_performance(
        self,
        venue: ExecutionVenue,
        actual_cost_bps: float,
        actual_slippage_bps: float,
        actual_execution_time: int,
        fill_rate: float,
    ) -> None:
        """更新场所性能指标

        Args:
            venue: 执行场所
            actual_cost_bps: 实际成本
            actual_slippage_bps: 实际滑点
            actual_execution_time: 实际执行时间
            fill_rate: 成交率
        """
        # 使用指数移动平均更新性能指标
        alpha = 0.3  # 平滑系数

        performance = self.venue_performance[venue]

        performance["avg_cost_bps"] = (
            alpha * actual_cost_bps + (1 - alpha) * performance["avg_cost_bps"]
        )

        performance["avg_slippage_bps"] = (
            alpha * actual_slippage_bps + (1 - alpha) * performance["avg_slippage_bps"]
        )

        performance["avg_execution_time"] = (
            alpha * actual_execution_time
            + (1 - alpha) * performance["avg_execution_time"]
        )

        performance["fill_rate"] = (
            alpha * fill_rate + (1 - alpha) * performance["fill_rate"]
        )

        logger.info(f"Updated performance metrics for {venue.value}")

    def get_routing_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息

        Returns:
            统计信息字典
        """
        if not self.routing_history:
            return {
                "total_routes": 0,
                "by_venue": {},
                "by_strategy": {},
                "avg_estimated_cost_bps": 0.0,
                "avg_estimated_slippage_bps": 0.0,
            }

        # 按场所统计
        by_venue = {}
        for decision in self.routing_history:
            venue = decision.venue.value
            by_venue[venue] = by_venue.get(venue, 0) + 1

        # 按策略统计
        by_strategy = {}
        for decision in self.routing_history:
            strategy = decision.strategy.value
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

        # 平均成本和滑点
        avg_cost = sum(d.estimated_cost_bps for d in self.routing_history) / len(
            self.routing_history
        )
        avg_slippage = sum(
            d.estimated_slippage_bps for d in self.routing_history
        ) / len(self.routing_history)

        return {
            "total_routes": len(self.routing_history),
            "by_venue": by_venue,
            "by_strategy": by_strategy,
            "avg_estimated_cost_bps": avg_cost,
            "avg_estimated_slippage_bps": avg_slippage,
            "venue_performance": {
                venue.value: metrics
                for venue, metrics in self.venue_performance.items()
            },
        }


# 全局路由器实例
_global_router: Optional[OrderRouter] = None


def get_order_router() -> OrderRouter:
    """获取全局订单路由器

    Returns:
        订单路由器实例
    """
    global _global_router
    if _global_router is None:
        _global_router = OrderRouter()
    return _global_router


def route_order_quick(order: Order, urgency: float = 0.5) -> RoutingDecision:
    """快速路由订单

    Args:
        order: 订单对象
        urgency: 紧急程度

    Returns:
        路由决策
    """
    router = get_order_router()
    return router.route_order(order, urgency=urgency)
