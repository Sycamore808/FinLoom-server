from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MarketData:
    """市场数据标准结构"""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketData":
        """从字典创建实例"""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class Signal:
    """交易信号标准结构"""

    signal_id: str
    timestamp: datetime
    symbol: str
    action: str  # 只能是 'BUY', 'SELL', 'HOLD'
    quantity: int
    price: float
    confidence: float  # 0.0 到 1.0
    strategy_name: str
    metadata: Dict[str, Any]

    def __post_init__(self):
        """初始化后验证"""
        if self.action not in ["BUY", "SELL", "HOLD"]:
            raise ValueError(f"Invalid action: {self.action}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1: {self.confidence}")
        if self.quantity < 0:
            raise ValueError(f"Quantity must be non-negative: {self.quantity}")


@dataclass
class Position:
    """持仓标准结构"""

    position_id: str
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    open_time: datetime
    last_update: datetime

    @property
    def return_pct(self) -> float:
        """计算收益率"""
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost

    @property
    def holding_period_days(self) -> int:
        """计算持仓天数"""
        return (self.last_update - self.open_time).days
