"""
交易环境 - 为强化学习提供模拟交易环境

提供：
- 市场状态建模
- 交易动作执行
- 奖励函数计算
- 风险管理
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.exceptions import ModelError
from common.logging_system import setup_logger

logger = setup_logger("trading_environment")


class TradingAction(Enum):
    """交易动作"""

    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class TradingState:
    """交易状态"""

    prices: np.ndarray  # 价格序列
    technical_indicators: np.ndarray  # 技术指标
    position: float  # 持仓
    cash: float  # 现金
    portfolio_value: float  # 组合价值
    step: int  # 当前步数


@dataclass
class EnvironmentConfig:
    """环境配置"""

    initial_cash: float = 100000.0
    transaction_cost: float = 0.001  # 交易费用
    max_position_size: float = 1.0  # 最大持仓比例
    lookback_window: int = 20  # 状态窗口大小
    reward_scaling: float = 1.0
    risk_penalty: float = 0.1
    use_risk_management: bool = True


class TradingEnvironment:
    """交易环境"""

    def __init__(self, data: pd.DataFrame, config: EnvironmentConfig):
        self.data = data
        self.config = config

        # 验证数据
        self._validate_data()

        # 状态变量
        self.current_step = 0
        self.max_steps = len(data) - config.lookback_window - 1

        # 账户状态
        self.initial_cash = config.initial_cash
        self.cash = config.initial_cash
        self.position = 0.0  # 股票数量
        self.portfolio_values = []

        # 交易历史
        self.trade_history = []
        self.action_history = []

        # 性能指标
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0

        logger.info(f"Trading environment initialized with {len(data)} data points")

    def _validate_data(self):
        """验证数据格式"""
        required_columns = ["close", "open", "high", "low", "volume"]
        missing_columns = [
            col for col in required_columns if col not in self.data.columns
        ]

        if missing_columns:
            raise ModelError(f"Missing required columns: {missing_columns}")

        if len(self.data) < self.config.lookback_window + 10:
            raise ModelError(
                f"Insufficient data: need at least {self.config.lookback_window + 10} rows"
            )

    def reset(self) -> np.ndarray:
        """重置环境"""
        try:
            self.current_step = 0
            self.cash = self.initial_cash
            self.position = 0.0
            self.portfolio_values = []
            self.trade_history = []
            self.action_history = []

            return self._get_state()

        except Exception as e:
            logger.error(f"Environment reset failed: {e}")
            return np.zeros(self._get_state_size())

    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        try:
            # 获取价格窗口
            start_idx = self.current_step
            end_idx = self.current_step + self.config.lookback_window

            if end_idx > len(self.data):
                # 填充不足的数据
                price_window = (
                    self.data["close"].iloc[-self.config.lookback_window :].values
                )
            else:
                price_window = self.data["close"].iloc[start_idx:end_idx].values

            # 标准化价格
            price_returns = np.diff(price_window) / price_window[:-1]
            price_returns = np.nan_to_num(price_returns)

            # 技术指标
            current_price = self.data["close"].iloc[
                self.current_step + self.config.lookback_window - 1
            ]

            # 简单技术指标
            sma_5 = (
                self.data["close"].iloc[start_idx:end_idx].rolling(5).mean().iloc[-1]
                if end_idx - start_idx >= 5
                else current_price
            )
            sma_20 = (
                self.data["close"].iloc[start_idx:end_idx].rolling(20).mean().iloc[-1]
                if end_idx - start_idx >= 20
                else current_price
            )

            rsi = self._calculate_rsi(self.data["close"].iloc[start_idx:end_idx])

            # 账户状态
            portfolio_value = self.cash + self.position * current_price
            position_ratio = (
                (self.position * current_price) / portfolio_value
                if portfolio_value > 0
                else 0
            )

            # 构建状态向量
            state = np.concatenate(
                [
                    price_returns,  # 价格收益率序列
                    [current_price / 1000.0],  # 标准化当前价格
                    [sma_5 / current_price],  # SMA5相对值
                    [sma_20 / current_price],  # SMA20相对值
                    [rsi / 100.0],  # RSI
                    [position_ratio],  # 持仓比例
                    [self.cash / self.initial_cash],  # 现金比例
                    [portfolio_value / self.initial_cash],  # 组合价值比例
                ]
            )

            return state.astype(np.float32)

        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return np.zeros(self._get_state_size())

    def _get_state_size(self) -> int:
        """获取状态向量大小"""
        return self.config.lookback_window - 1 + 7  # 价格收益率 + 7个指标

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0  # 默认中性值

            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0

        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return 50.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行一步"""
        try:
            # 验证动作
            if action not in [0, 1, 2]:
                action = 0  # 默认HOLD

            # 获取当前价格
            current_price = self.data["close"].iloc[
                self.current_step + self.config.lookback_window
            ]

            # 执行交易
            reward = self._execute_trade(action, current_price)

            # 更新状态
            self.current_step += 1
            done = self.current_step >= self.max_steps

            # 记录历史
            self.action_history.append(action)
            portfolio_value = self.cash + self.position * current_price
            self.portfolio_values.append(portfolio_value)

            # 获取新状态
            next_state = (
                self._get_state() if not done else np.zeros(self._get_state_size())
            )

            # 额外信息
            info = {
                "portfolio_value": portfolio_value,
                "cash": self.cash,
                "position": self.position,
                "price": current_price,
                "action": action,
                "step": self.current_step,
            }

            return next_state, reward, done, info

        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            return np.zeros(self._get_state_size()), 0.0, True, {}

    def _execute_trade(self, action: int, price: float) -> float:
        """执行交易并计算奖励"""
        try:
            old_portfolio_value = self.cash + self.position * price

            if action == TradingAction.BUY.value:
                # 买入
                max_shares = (self.cash * self.config.max_position_size) / (
                    price * (1 + self.config.transaction_cost)
                )
                shares_to_buy = max_shares * 0.5  # 每次买入一半可用资金

                if shares_to_buy > 0 and self.cash >= shares_to_buy * price * (
                    1 + self.config.transaction_cost
                ):
                    cost = shares_to_buy * price * (1 + self.config.transaction_cost)
                    self.cash -= cost
                    self.position += shares_to_buy

                    self.trade_history.append(
                        {
                            "step": self.current_step,
                            "action": "BUY",
                            "shares": shares_to_buy,
                            "price": price,
                            "cost": cost,
                        }
                    )

            elif action == TradingAction.SELL.value:
                # 卖出
                shares_to_sell = self.position * 0.5  # 每次卖出一半持仓

                if shares_to_sell > 0:
                    proceeds = (
                        shares_to_sell * price * (1 - self.config.transaction_cost)
                    )
                    self.cash += proceeds
                    self.position -= shares_to_sell

                    self.trade_history.append(
                        {
                            "step": self.current_step,
                            "action": "SELL",
                            "shares": shares_to_sell,
                            "price": price,
                            "proceeds": proceeds,
                        }
                    )

            # 计算奖励
            new_portfolio_value = self.cash + self.position * price
            portfolio_return = (
                (new_portfolio_value - old_portfolio_value) / old_portfolio_value
                if old_portfolio_value > 0
                else 0
            )

            # 基础奖励
            reward = portfolio_return * self.config.reward_scaling

            # 风险惩罚
            if self.config.use_risk_management:
                position_ratio = (
                    (self.position * price) / new_portfolio_value
                    if new_portfolio_value > 0
                    else 0
                )
                if position_ratio > self.config.max_position_size:
                    reward -= self.config.risk_penalty

            return reward

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return 0.0

    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        try:
            if not self.portfolio_values:
                return {}

            # 计算收益率
            portfolio_values = np.array(self.portfolio_values)
            total_return = (
                portfolio_values[-1] - self.initial_cash
            ) / self.initial_cash

            # 计算最大回撤
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown)

            # 计算夏普比率
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = (
                np.mean(returns) / np.std(returns) * np.sqrt(252)
                if np.std(returns) > 0
                else 0
            )

            # 胜率
            profitable_trades = sum(
                1 for trade in self.trade_history if trade["action"] == "SELL"
            )
            win_rate = (
                profitable_trades / len(self.trade_history) if self.trade_history else 0
            )

            return {
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "total_trades": len(self.trade_history),
                "final_portfolio_value": portfolio_values[-1],
                "final_cash": self.cash,
                "final_position": self.position,
            }

        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return {}

    def render(self, mode: str = "human") -> Optional[str]:
        """渲染环境状态"""
        try:
            if not self.portfolio_values:
                return "No data to render"

            current_price = self.data["close"].iloc[
                self.current_step + self.config.lookback_window - 1
            ]
            portfolio_value = self.cash + self.position * current_price

            status = f"""
Step: {self.current_step}/{self.max_steps}
Price: ${current_price:.2f}
Cash: ${self.cash:.2f}
Position: {self.position:.2f} shares
Portfolio Value: ${portfolio_value:.2f}
Total Return: {((portfolio_value - self.initial_cash) / self.initial_cash) * 100:.2f}%
Recent Action: {self.action_history[-1] if self.action_history else "N/A"}
            """.strip()

            if mode == "human":
                print(status)

            return status

        except Exception as e:
            logger.error(f"Render failed: {e}")
            return "Render error"


def create_trading_environment(
    data: pd.DataFrame,
    initial_cash: float = 100000.0,
    transaction_cost: float = 0.001,
    lookback_window: int = 20,
) -> TradingEnvironment:
    """创建交易环境的便捷函数"""
    config = EnvironmentConfig(
        initial_cash=initial_cash,
        transaction_cost=transaction_cost,
        lookback_window=lookback_window,
    )
    return TradingEnvironment(data, config)
