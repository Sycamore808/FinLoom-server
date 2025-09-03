"""
市场监控器模块
实时监控市场状态和异常情况
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from common.constants import TRADING_DAYS_PER_YEAR
from common.data_structures import MarketData
from common.exceptions import ModelError
from common.logging_system import setup_logger
from scipy import stats

logger = setup_logger("market_monitor")


class MarketRegime(Enum):
    """市场状态枚举"""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRASH = "crash"
    RALLY = "rally"


class MarketCondition(Enum):
    """市场条件枚举"""

    NORMAL = "normal"
    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    TRENDING = "trending"


@dataclass
class MarketMetrics:
    """市场指标"""

    timestamp: datetime
    regime: MarketRegime
    condition: MarketCondition
    volatility: float
    volatility_percentile: float
    liquidity_score: float
    breadth: float  # 市场广度
    momentum: float
    correlation_matrix: np.ndarray
    sector_performance: Dict[str, float]
    volume_ratio: float  # 成交量比率
    put_call_ratio: float
    vix_level: float
    term_structure: Dict[str, float]
    market_sentiment: float  # -1到1


@dataclass
class MarketAnomaly:
    """市场异常"""

    anomaly_id: str
    timestamp: datetime
    type: str  # 'price_spike', 'volume_surge', 'correlation_break', etc.
    severity: float  # 0-1
    affected_symbols: List[str]
    description: str
    metrics: Dict[str, float]
    expected_impact: str
    recommended_action: str


@dataclass
class MarketEvent:
    """市场事件"""

    event_id: str
    timestamp: datetime
    event_type: str  # 'circuit_breaker', 'halt', 'news', 'economic_data'
    impact_level: str  # 'low', 'medium', 'high', 'critical'
    affected_sectors: List[str]
    duration_estimate: Optional[int]  # 预计持续时间（分钟）
    metadata: Dict[str, Any]


class MarketMonitor:
    """市场监控器类"""

    # 阈值定义
    VOLATILITY_THRESHOLDS = {"low": 0.10, "normal": 0.20, "high": 0.35, "extreme": 0.50}

    LIQUIDITY_THRESHOLDS = {"poor": 0.3, "fair": 0.5, "good": 0.7, "excellent": 0.9}

    def __init__(
        self,
        symbols: List[str],
        benchmark_symbol: str = "SPY",
        lookback_period: int = 252,
    ):
        """初始化市场监控器

        Args:
            symbols: 监控的标的列表
            benchmark_symbol: 基准标的
            lookback_period: 历史数据回看期
        """
        self.symbols = symbols
        self.benchmark_symbol = benchmark_symbol
        self.lookback_period = lookback_period
        self.current_metrics: Optional[MarketMetrics] = None
        self.metrics_history: deque = deque(maxlen=lookback_period)
        self.anomaly_history: List[MarketAnomaly] = []
        self.event_history: List[MarketEvent] = []
        self.price_data: pd.DataFrame = pd.DataFrame()
        self.volume_data: pd.DataFrame = pd.DataFrame()
        self.monitoring_active = False
        self.anomaly_detectors: Dict[str, callable] = {}
        self._initialize_detectors()

    def update_market_data(self, market_data: List[MarketData]) -> None:
        """更新市场数据

        Args:
            market_data: 市场数据列表
        """
        # 转换为DataFrame
        data_dict = defaultdict(dict)
        volume_dict = defaultdict(dict)

        for data in market_data:
            data_dict[data.symbol][data.timestamp] = data.close
            volume_dict[data.symbol][data.timestamp] = data.volume

        # 更新价格数据
        new_prices = pd.DataFrame(data_dict)
        self.price_data = pd.concat([self.price_data, new_prices]).tail(
            self.lookback_period
        )

        # 更新成交量数据
        new_volumes = pd.DataFrame(volume_dict)
        self.volume_data = pd.concat([self.volume_data, new_volumes]).tail(
            self.lookback_period
        )

        # 计算市场指标
        self._calculate_market_metrics()

    def calculate_market_regime(self) -> MarketRegime:
        """计算市场状态

        Returns:
            市场状态
        """
        if self.price_data.empty:
            return MarketRegime.SIDEWAYS

        # 计算趋势
        returns = self.price_data.pct_change()
        recent_returns = returns.tail(20)

        # 计算移动平均
        sma_20 = self.price_data.rolling(20).mean()
        sma_50 = self.price_data.rolling(50).mean()

        # 获取基准数据
        if self.benchmark_symbol in self.price_data.columns:
            benchmark_price = self.price_data[self.benchmark_symbol].iloc[-1]
            benchmark_sma20 = sma_20[self.benchmark_symbol].iloc[-1]
            benchmark_sma50 = sma_50[self.benchmark_symbol].iloc[-1]

            # 判断趋势
            if benchmark_price > benchmark_sma20 > benchmark_sma50:
                trend = "up"
            elif benchmark_price < benchmark_sma20 < benchmark_sma50:
                trend = "down"
            else:
                trend = "sideways"

            # 计算波动率
            volatility = recent_returns[self.benchmark_symbol].std() * np.sqrt(252)

            # 计算累计收益
            cumulative_return = recent_returns[self.benchmark_symbol].sum()

            # 判断市场状态
            if trend == "down" and cumulative_return < -0.10:
                if volatility > self.VOLATILITY_THRESHOLDS["extreme"]:
                    return MarketRegime.CRASH
                else:
                    return MarketRegime.BEAR
            elif trend == "up" and cumulative_return > 0.10:
                if volatility > self.VOLATILITY_THRESHOLDS["high"]:
                    return MarketRegime.RALLY
                else:
                    return MarketRegime.BULL
            elif volatility > self.VOLATILITY_THRESHOLDS["high"]:
                return MarketRegime.VOLATILE
            else:
                return MarketRegime.SIDEWAYS
        else:
            return MarketRegime.SIDEWAYS

    def calculate_market_condition(self) -> MarketCondition:
        """计算市场条件

        Returns:
            市场条件
        """
        if self.price_data.empty:
            return MarketCondition.NORMAL

        # 计算RSI
        rsi_values = []
        for symbol in self.symbols[:10]:  # 取前10个标的作为样本
            if symbol in self.price_data.columns:
                rsi = self._calculate_rsi(self.price_data[symbol])
                if not np.isnan(rsi):
                    rsi_values.append(rsi)

        if rsi_values:
            avg_rsi = np.mean(rsi_values)

            if avg_rsi > 70:
                return MarketCondition.OVERBOUGHT
            elif avg_rsi < 30:
                return MarketCondition.OVERSOLD

        # 检查波动率
        returns = self.price_data.pct_change()
        recent_volatility = returns.tail(20).std().mean() * np.sqrt(252)

        if recent_volatility > self.VOLATILITY_THRESHOLDS["high"]:
            return MarketCondition.HIGH_VOLATILITY

        # 检查流动性
        if not self.volume_data.empty:
            recent_volume = self.volume_data.tail(5).mean()
            avg_volume = self.volume_data.mean()
            volume_ratio = (
                recent_volume.mean() / avg_volume.mean() if avg_volume.mean() > 0 else 1
            )

            if volume_ratio < 0.5:
                return MarketCondition.LOW_LIQUIDITY

        # 检查趋势
        if len(self.price_data) > 20:
            trend_strength = self._calculate_trend_strength()
            if trend_strength > 0.7:
                return MarketCondition.TRENDING

        return MarketCondition.NORMAL

    def detect_anomalies(self) -> List[MarketAnomaly]:
        """检测市场异常

        Returns:
            异常列表
        """
        anomalies = []

        for detector_name, detector_func in self.anomaly_detectors.items():
            try:
                detected = detector_func()
                if detected:
                    anomalies.extend(detected)
            except Exception as e:
                logger.error(f"Error in anomaly detector {detector_name}: {e}")

        # 记录异常
        self.anomaly_history.extend(anomalies)

        return anomalies

    def calculate_correlation_matrix(self) -> np.ndarray:
        """计算相关性矩阵

        Returns:
            相关性矩阵
        """
        if self.price_data.empty or len(self.price_data) < 20:
            return np.array([[]])

        returns = self.price_data.pct_change().dropna()

        if len(returns) > 0:
            return returns.corr().values
        else:
            return np.array([[]])

    def calculate_market_breadth(self) -> float:
        """计算市场广度

        Returns:
            市场广度（0-1）
        """
        if self.price_data.empty or len(self.price_data) < 2:
            return 0.5

        # 计算上涨股票比例
        recent_returns = self.price_data.pct_change().iloc[-1]
        advancing = (recent_returns > 0).sum()
        declining = (recent_returns < 0).sum()
        total = advancing + declining

        if total > 0:
            breadth = advancing / total
        else:
            breadth = 0.5

        return breadth

    def calculate_sector_performance(self) -> Dict[str, float]:
        """计算板块表现

        Returns:
            板块收益率字典
        """
        # 这里应该根据实际的板块分类来计算
        # 简化示例：随机分配板块
        sectors = ["Technology", "Finance", "Healthcare", "Energy", "Consumer"]
        sector_performance = {}

        if not self.price_data.empty and len(self.price_data) > 1:
            returns = self.price_data.pct_change().iloc[-1]

            for sector in sectors:
                # 简化：取部分股票作为板块代表
                sector_returns = returns.dropna().sample(min(5, len(returns)))
                sector_performance[sector] = sector_returns.mean()
        else:
            sector_performance = {sector: 0.0 for sector in sectors}

        return sector_performance

    async def start_monitoring(
        self, data_provider: callable, interval: int = 60
    ) -> None:
        """启动市场监控

        Args:
            data_provider: 数据提供函数
            interval: 监控间隔（秒）
        """
        logger.info("Starting market monitoring")
        self.monitoring_active = True

        while self.monitoring_active:
            try:
                # 获取最新数据
                market_data = await data_provider(self.symbols)

                # 更新数据
                self.update_market_data(market_data)

                # 检测异常
                anomalies = self.detect_anomalies()

                # 处理异常
                for anomaly in anomalies:
                    await self._handle_anomaly(anomaly)

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in market monitoring: {e}")
                await asyncio.sleep(interval)

    def stop_monitoring(self) -> None:
        """停止市场监控"""
        logger.info("Stopping market monitoring")
        self.monitoring_active = False

    def get_market_summary(self) -> Dict[str, Any]:
        """获取市场摘要

        Returns:
            市场摘要信息
        """
        if not self.current_metrics:
            return {}

        summary = {
            "timestamp": self.current_metrics.timestamp,
            "regime": self.current_metrics.regime.value,
            "condition": self.current_metrics.condition.value,
            "volatility": self.current_metrics.volatility,
            "liquidity_score": self.current_metrics.liquidity_score,
            "market_breadth": self.current_metrics.breadth,
            "momentum": self.current_metrics.momentum,
            "sentiment": self.current_metrics.market_sentiment,
            "top_sectors": self._get_top_sectors(),
            "recent_anomalies": self._get_recent_anomalies(5),
            "risk_level": self._calculate_risk_level(),
        }

        return summary

    def _initialize_detectors(self) -> None:
        """初始化异常检测器"""
        self.anomaly_detectors = {
            "price_spike": self._detect_price_spike,
            "volume_surge": self._detect_volume_surge,
            "correlation_break": self._detect_correlation_break,
            "liquidity_drop": self._detect_liquidity_drop,
            "volatility_jump": self._detect_volatility_jump,
        }

    def _calculate_market_metrics(self) -> None:
        """计算市场指标"""
        metrics = MarketMetrics(
            timestamp=datetime.now(),
            regime=self.calculate_market_regime(),
            condition=self.calculate_market_condition(),
            volatility=self._calculate_market_volatility(),
            volatility_percentile=self._calculate_volatility_percentile(),
            liquidity_score=self._calculate_liquidity_score(),
            breadth=self.calculate_market_breadth(),
            momentum=self._calculate_market_momentum(),
            correlation_matrix=self.calculate_correlation_matrix(),
            sector_performance=self.calculate_sector_performance(),
            volume_ratio=self._calculate_volume_ratio(),
            put_call_ratio=self._calculate_put_call_ratio(),
            vix_level=self._calculate_vix_proxy(),
            term_structure=self._calculate_term_structure(),
            market_sentiment=self._calculate_market_sentiment(),
        )

        self.current_metrics = metrics
        self.metrics_history.append(metrics)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI指标

        Args:
            prices: 价格序列
            period: 计算周期

        Returns:
            RSI值
        """
        if len(prices) < period + 1:
            return np.nan

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss if loss.iloc[-1] != 0 else 100
        rsi = 100 - (100 / (1 + rs.iloc[-1]))

        return rsi

    def _calculate_trend_strength(self) -> float:
        """计算趋势强度

        Returns:
            趋势强度（0-1）
        """
        if self.price_data.empty or len(self.price_data) < 20:
            return 0.0

        # 使用ADX或类似指标
        # 简化：使用移动平均的斜率
        sma = self.price_data.rolling(20).mean()
        if len(sma) > 1:
            trend_slopes = (sma.iloc[-1] - sma.iloc[-5]) / sma.iloc[-5]
            avg_slope = trend_slopes.mean()
            strength = min(abs(avg_slope) * 10, 1.0)
        else:
            strength = 0.0

        return strength

    def _calculate_market_volatility(self) -> float:
        """计算市场波动率

        Returns:
            年化波动率
        """
        if self.price_data.empty or len(self.price_data) < 2:
            return 0.0

        returns = self.price_data.pct_change().dropna()
        if len(returns) > 0:
            volatility = returns.std().mean() * np.sqrt(252)
        else:
            volatility = 0.0

        return volatility

    def _calculate_volatility_percentile(self) -> float:
        """计算波动率百分位

        Returns:
            波动率百分位（0-1）
        """
        if len(self.metrics_history) < 20:
            return 0.5

        historical_vols = [m.volatility for m in self.metrics_history]
        current_vol = self._calculate_market_volatility()

        percentile = stats.percentileofscore(historical_vols, current_vol) / 100

        return percentile

    def _calculate_liquidity_score(self) -> float:
        """计算流动性评分

        Returns:
            流动性评分（0-1）
        """
        if self.volume_data.empty:
            return 0.5

        # 计算成交量稳定性
        recent_volume = self.volume_data.tail(5).mean(axis=1)
        volume_stability = (
            1 - recent_volume.std() / recent_volume.mean()
            if recent_volume.mean() > 0
            else 0
        )

        # 计算相对成交量
        current_volume = (
            self.volume_data.iloc[-1].mean() if len(self.volume_data) > 0 else 0
        )
        avg_volume = self.volume_data.mean().mean()
        relative_volume = (
            min(current_volume / avg_volume if avg_volume > 0 else 0, 2) / 2
        )

        liquidity_score = (volume_stability + relative_volume) / 2

        return max(0, min(1, liquidity_score))

    def _calculate_market_momentum(self) -> float:
        """计算市场动量

        Returns:
            动量值（-1到1）
        """
        if self.price_data.empty or len(self.price_data) < 20:
            return 0.0

        # 计算不同周期的收益率
        returns_5d = (self.price_data.iloc[-1] / self.price_data.iloc[-5] - 1).mean()
        returns_20d = (self.price_data.iloc[-1] / self.price_data.iloc[-20] - 1).mean()

        # 组合动量
        momentum = returns_5d * 0.6 + returns_20d * 0.4

        # 标准化到-1到1
        momentum = max(-1, min(1, momentum * 10))

        return momentum

    def _calculate_volume_ratio(self) -> float:
        """计算成交量比率

        Returns:
            成交量比率
        """
        if self.volume_data.empty or len(self.volume_data) < 20:
            return 1.0

        recent_volume = self.volume_data.tail(5).mean().mean()
        avg_volume = self.volume_data.tail(20).mean().mean()

        if avg_volume > 0:
            return recent_volume / avg_volume
        else:
            return 1.0

    def _calculate_put_call_ratio(self) -> float:
        """计算看跌看涨比率

        Returns:
            看跌看涨比率
        """
        # 需要期权数据，这里返回默认值
        return 1.0

    def _calculate_vix_proxy(self) -> float:
        """计算VIX代理指标

        Returns:
            VIX代理值
        """
        # 使用历史波动率作为代理
        return self._calculate_market_volatility() * 100

    def _calculate_term_structure(self) -> Dict[str, float]:
        """计算期限结构

        Returns:
            期限结构字典
        """
        # 需要期货数据，这里返回简化版本
        return {"1M": 0.0, "3M": 0.0, "6M": 0.0, "12M": 0.0}

    def _calculate_market_sentiment(self) -> float:
        """计算市场情绪

        Returns:
            情绪值（-1到1）
        """
        # 综合多个指标
        breadth = self.calculate_market_breadth()
        momentum = self._calculate_market_momentum()

        # 简单加权
        sentiment = breadth * 0.5 + momentum * 0.5

        return max(-1, min(1, sentiment))

    def _detect_price_spike(self) -> List[MarketAnomaly]:
        """检测价格异常波动

        Returns:
            异常列表
        """
        anomalies = []

        if self.price_data.empty or len(self.price_data) < 2:
            return anomalies

        # 计算价格变化
        price_changes = self.price_data.pct_change().iloc[-1]

        for symbol in price_changes.index:
            change = price_changes[symbol]

            if abs(change) > 0.05:  # 5%阈值
                anomaly = MarketAnomaly(
                    anomaly_id=f"price_spike_{symbol}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    type="price_spike",
                    severity=min(abs(change) / 0.10, 1.0),
                    affected_symbols=[symbol],
                    description=f"{symbol} price changed {change:.2%}",
                    metrics={"price_change": change},
                    expected_impact="Potential volatility increase",
                    recommended_action="Review position sizing",
                )
                anomalies.append(anomaly)

        return anomalies

    def _detect_volume_surge(self) -> List[MarketAnomaly]:
        """检测成交量激增

        Returns:
            异常列表
        """
        anomalies = []

        if self.volume_data.empty or len(self.volume_data) < 20:
            return anomalies

        # 计算成交量比率
        current_volume = self.volume_data.iloc[-1]
        avg_volume = self.volume_data.tail(20).mean()

        for symbol in current_volume.index:
            if avg_volume[symbol] > 0:
                ratio = current_volume[symbol] / avg_volume[symbol]

                if ratio > 3:  # 3倍阈值
                    anomaly = MarketAnomaly(
                        anomaly_id=f"volume_surge_{symbol}_{datetime.now().timestamp()}",
                        timestamp=datetime.now(),
                        type="volume_surge",
                        severity=min((ratio - 3) / 3, 1.0),
                        affected_symbols=[symbol],
                        description=f"{symbol} volume surge {ratio:.1f}x average",
                        metrics={"volume_ratio": ratio},
                        expected_impact="Increased volatility expected",
                        recommended_action="Monitor for news or events",
                    )
                    anomalies.append(anomaly)

        return anomalies

    def _detect_correlation_break(self) -> List[MarketAnomaly]:
        """检测相关性破裂

        Returns:
            异常列表
        """
        anomalies = []

        if len(self.metrics_history) < 20:
            return anomalies

        # 比较当前相关性与历史相关性
        current_corr = self.calculate_correlation_matrix()

        if current_corr.size > 0:
            # 计算历史平均相关性
            historical_corrs = []
            for metrics in list(self.metrics_history)[-20:-1]:
                if metrics.correlation_matrix.size > 0:
                    historical_corrs.append(metrics.correlation_matrix)

            if historical_corrs:
                avg_corr = np.mean(historical_corrs, axis=0)
                corr_diff = np.abs(current_corr - avg_corr)

                # 检测显著变化
                if np.max(corr_diff) > 0.3:
                    anomaly = MarketAnomaly(
                        anomaly_id=f"correlation_break_{datetime.now().timestamp()}",
                        timestamp=datetime.now(),
                        type="correlation_break",
                        severity=min(np.max(corr_diff) / 0.5, 1.0),
                        affected_symbols=self.symbols[:10],  # 取前10个
                        description="Significant correlation structure change detected",
                        metrics={"max_correlation_change": float(np.max(corr_diff))},
                        expected_impact="Portfolio diversification may be affected",
                        recommended_action="Review portfolio correlation risk",
                    )
                    anomalies.append(anomaly)

        return anomalies

    def _detect_liquidity_drop(self) -> List[MarketAnomaly]:
        """检测流动性下降

        Returns:
            异常列表
        """
        anomalies = []

        liquidity_score = self._calculate_liquidity_score()

        if liquidity_score < self.LIQUIDITY_THRESHOLDS["poor"]:
            anomaly = MarketAnomaly(
                anomaly_id=f"liquidity_drop_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                type="liquidity_drop",
                severity=(self.LIQUIDITY_THRESHOLDS["poor"] - liquidity_score)
                / self.LIQUIDITY_THRESHOLDS["poor"],
                affected_symbols=self.symbols,
                description=f"Market liquidity dropped to {liquidity_score:.2f}",
                metrics={"liquidity_score": liquidity_score},
                expected_impact="Increased slippage and execution costs",
                recommended_action="Reduce position sizes or use limit orders",
            )
            anomalies.append(anomaly)

        return anomalies

    def _detect_volatility_jump(self) -> List[MarketAnomaly]:
        """检测波动率跳升

        Returns:
            异常列表
        """
        anomalies = []

        if len(self.metrics_history) < 2:
            return anomalies

        current_vol = self._calculate_market_volatility()
        prev_vol = (
            self.metrics_history[-2].volatility
            if len(self.metrics_history) > 1
            else current_vol
        )

        if prev_vol > 0:
            vol_change = (current_vol - prev_vol) / prev_vol

            if vol_change > 0.5:  # 50%增长
                anomaly = MarketAnomaly(
                    anomaly_id=f"volatility_jump_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    type="volatility_jump",
                    severity=min(vol_change, 1.0),
                    affected_symbols=self.symbols,
                    description=f"Volatility jumped {vol_change:.1%}",
                    metrics={
                        "volatility_change": vol_change,
                        "current_volatility": current_vol,
                    },
                    expected_impact="Increased risk across all positions",
                    recommended_action="Consider reducing leverage and position sizes",
                )
                anomalies.append(anomaly)

        return anomalies

    async def _handle_anomaly(self, anomaly: MarketAnomaly) -> None:
        """处理市场异常

        Args:
            anomaly: 异常对象
        """
        logger.warning(
            f"Market anomaly detected: {anomaly.type} - {anomaly.description}"
        )

        # 这里可以添加具体的异常处理逻辑
        # 例如发送通知、调整风控参数等

    def _get_top_sectors(self) -> List[Tuple[str, float]]:
        """获取表现最好的板块

        Returns:
            板块表现列表
        """
        if self.current_metrics and self.current_metrics.sector_performance:
            sorted_sectors = sorted(
                self.current_metrics.sector_performance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            return sorted_sectors[:3]
        else:
            return []

    def _get_recent_anomalies(self, n: int) -> List[Dict[str, Any]]:
        """获取最近的异常

        Args:
            n: 数量

        Returns:
            异常摘要列表
        """
        recent = (
            self.anomaly_history[-n:]
            if len(self.anomaly_history) > n
            else self.anomaly_history
        )

        return [
            {
                "timestamp": a.timestamp,
                "type": a.type,
                "severity": a.severity,
                "description": a.description,
            }
            for a in recent
        ]

    def _calculate_risk_level(self) -> str:
        """计算风险级别

        Returns:
            风险级别
        """
        if not self.current_metrics:
            return "unknown"

        # 基于多个因素计算风险级别
        risk_score = 0

        # 波动率因素
        if self.current_metrics.volatility > self.VOLATILITY_THRESHOLDS["high"]:
            risk_score += 3
        elif self.current_metrics.volatility > self.VOLATILITY_THRESHOLDS["normal"]:
            risk_score += 1

        # 流动性因素
        if self.current_metrics.liquidity_score < self.LIQUIDITY_THRESHOLDS["poor"]:
            risk_score += 2
        elif self.current_metrics.liquidity_score < self.LIQUIDITY_THRESHOLDS["fair"]:
            risk_score += 1

        # 市场状态因素
        if self.current_metrics.regime in [MarketRegime.CRASH, MarketRegime.VOLATILE]:
            risk_score += 3
        elif self.current_metrics.regime == MarketRegime.BEAR:
            risk_score += 2

        # 转换为风险级别
        if risk_score >= 7:
            return "critical"
        elif risk_score >= 5:
            return "high"
        elif risk_score >= 3:
            return "medium"
        else:
            return "low"
