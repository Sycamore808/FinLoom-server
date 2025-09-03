"""
PPO强化学习智能体模块
实现用于交易决策的PPO算法
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.constants import MAX_POSITION_PCT, MIN_POSITION_SIZE
from common.exceptions import ModelError
from common.logging_system import setup_logger
from gym import spaces
from torch.distributions import Categorical, Normal

logger = setup_logger("ppo_agent")


@dataclass
class PPOConfig:
    """PPO配置"""

    state_dim: int
    action_dim: int
    hidden_dims: List[int] = None
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 2048
    use_cuda: bool = True
    continuous_action: bool = False

    def __post_init__(self):
        """初始化后处理"""
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


@dataclass
class Experience:
    """经验数据"""

    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class ActorCritic(nn.Module):
    """Actor-Critic网络"""

    def __init__(self, config: PPOConfig):
        """初始化Actor-Critic网络

        Args:
            config: PPO配置
        """
        super(ActorCritic, self).__init__()
        self.config = config

        # 构建共享层
        shared_layers = []
        input_dim = config.state_dim

        for hidden_dim in config.hidden_dims[:-1]:
            shared_layers.append(nn.Linear(input_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim

        self.shared = nn.Sequential(*shared_layers)

        # Actor头
        self.actor = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[-1], config.action_dim),
        )

        # Critic头
        self.critic = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[-1], 1),
        )

        # 如果是连续动作空间，需要log_std
        if config.continuous_action:
            self.log_std = nn.Parameter(torch.zeros(config.action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            state: 状态张量

        Returns:
            动作分布参数和状态值
        """
        shared_features = self.shared(state)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)

        return action_logits, value

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作

        Args:
            state: 状态张量
            deterministic: 是否确定性动作

        Returns:
            动作、对数概率、状态值
        """
        action_logits, value = self.forward(state)

        if self.config.continuous_action:
            # 连续动作空间
            mean = action_logits
            std = self.log_std.exp()
            dist = Normal(mean, std)

            if deterministic:
                action = mean
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1)

        else:
            # 离散动作空间
            dist = Categorical(logits=action_logits)

            if deterministic:
                action = action_logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作

        Args:
            states: 状态张量
            actions: 动作张量

        Returns:
            对数概率、状态值、熵
        """
        action_logits, values = self.forward(states)

        if self.config.continuous_action:
            mean = action_logits
            std = self.log_std.exp()
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            dist = Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

        return log_probs, values, entropy


class ExperienceBuffer:
    """经验缓冲区"""

    def __init__(self, capacity: int):
        """初始化经验缓冲区

        Args:
            capacity: 缓冲区容量
        """
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0

    def push(self, experience: Experience) -> None:
        """添加经验

        Args:
            experience: 经验数据
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Experience]:
        """采样经验

        Args:
            batch_size: 批次大小

        Returns:
            经验列表
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def get_all(self) -> List[Experience]:
        """获取所有经验

        Returns:
            所有经验
        """
        return self.buffer.copy()

    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer.clear()
        self.position = 0

    def __len__(self) -> int:
        """缓冲区大小"""
        return len(self.buffer)


class TradingEnvironment(gym.Env):
    """交易环境"""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        """初始化交易环境

        Args:
            data: 市场数据DataFrame
            initial_balance: 初始资金
            commission: 手续费率
            slippage: 滑点率
        """
        super(TradingEnvironment, self).__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage

        # 动作空间：0=持有, 1=买入, 2=卖出
        self.action_space = spaces.Discrete(3)

        # 状态空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._get_state_dim(),), dtype=np.float32
        )

        self.reset()

    def reset(self) -> np.ndarray:
        """重置环境

        Returns:
            初始状态
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.total_trades = 0
        self.total_profit = 0

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行一步

        Args:
            action: 动作

        Returns:
            下一状态、奖励、是否结束、信息字典
        """
        prev_value = self._get_portfolio_value()

        # 执行动作
        if action == 1:  # 买入
            self._execute_buy()
        elif action == 2:  # 卖出
            self._execute_sell()

        # 更新步数
        self.current_step += 1

        # 计算奖励
        current_value = self._get_portfolio_value()
        reward = (current_value - prev_value) / prev_value

        # 检查是否结束
        done = self.current_step >= len(self.data) - 1 or self.balance <= 0

        # 获取下一状态
        next_state = self._get_state()

        # 信息字典
        info = {
            "balance": self.balance,
            "position": self.position,
            "portfolio_value": current_value,
            "total_trades": self.total_trades,
            "total_profit": self.total_profit,
        }

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """获取当前状态

        Returns:
            状态数组
        """
        if self.current_step >= len(self.data):
            return np.zeros(self._get_state_dim())

        # 市场特征
        market_features = self.data.iloc[self.current_step].values

        # 持仓特征
        position_features = np.array(
            [
                self.position,
                self.balance / self.initial_balance,
                self._get_portfolio_value() / self.initial_balance,
            ]
        )

        return np.concatenate([market_features, position_features])

    def _get_state_dim(self) -> int:
        """获取状态维度

        Returns:
            状态维度
        """
        return len(self.data.columns) + 3

    def _get_portfolio_value(self) -> float:
        """获取组合价值

        Returns:
            组合总价值
        """
        if self.current_step >= len(self.data):
            return self.balance

        current_price = self.data.iloc[self.current_step]["close"]
        return self.balance + self.position * current_price

    def _execute_buy(self) -> None:
        """执行买入"""
        if self.current_step >= len(self.data):
            return

        current_price = self.data.iloc[self.current_step]["close"]

        # 计算可买数量
        max_shares = int(
            self.balance / (current_price * (1 + self.commission + self.slippage))
        )

        if max_shares > 0:
            # 执行买入
            cost = max_shares * current_price * (1 + self.commission + self.slippage)
            self.balance -= cost
            self.position += max_shares
            self.total_trades += 1

    def _execute_sell(self) -> None:
        """执行卖出"""
        if self.current_step >= len(self.data) or self.position <= 0:
            return

        current_price = self.data.iloc[self.current_step]["close"]

        # 执行卖出
        proceeds = self.position * current_price * (1 - self.commission - self.slippage)
        self.balance += proceeds
        self.total_profit += proceeds - self.position * current_price
        self.position = 0
        self.total_trades += 1


class PPOAgent:
    """PPO智能体"""

    def __init__(self, config: PPOConfig):
        """初始化PPO智能体

        Args:
            config: PPO配置
        """
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_cuda and torch.cuda.is_available() else "cpu"
        )

        # 创建网络
        self.actor_critic = ActorCritic(config).to(self.device)
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=config.learning_rate
        )

        # 经验缓冲区
        self.buffer = ExperienceBuffer(config.buffer_size)

        # 训练统计
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "value_losses": [],
            "policy_losses": [],
            "entropies": [],
        }

    def initialize_trading_environment(
        self, market_data: pd.DataFrame, **env_kwargs
    ) -> TradingEnvironment:
        """初始化交易环境

        Args:
            market_data: 市场数据
            **env_kwargs: 环境参数

        Returns:
            交易环境
        """
        env = TradingEnvironment(market_data, **env_kwargs)

        # 更新配置的状态和动作维度
        self.config.state_dim = env.observation_space.shape[0]
        self.config.action_dim = env.action_space.n

        # 重新创建网络
        self.actor_critic = ActorCritic(self.config).to(self.device)
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=self.config.learning_rate
        )

        logger.info(
            f"Initialized trading environment with state_dim={self.config.state_dim}, action_dim={self.config.action_dim}"
        )
        return env

    def design_reward_function(
        self,
        portfolio_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        trade_frequency: float,
    ) -> float:
        """设计奖励函数

        Args:
            portfolio_return: 组合收益率
            sharpe_ratio: 夏普比率
            max_drawdown: 最大回撤
            trade_frequency: 交易频率

        Returns:
            奖励值
        """
        # 多目标奖励函数
        reward = (
            portfolio_return * 1.0  # 收益率
            + sharpe_ratio * 0.3  # 风险调整收益
            + max_drawdown * (-0.5)  # 惩罚大回撤
            + trade_frequency * (-0.1)  # 惩罚过度交易
        )

        return reward

    def train_ppo_agent(self, env: TradingEnvironment, n_episodes: int = 1000) -> None:
        """训练PPO智能体

        Args:
            env: 交易环境
            n_episodes: 训练回合数
        """
        logger.info(f"Starting PPO training for {n_episodes} episodes")

        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                # 获取动作
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, log_prob, value = self.actor_critic.get_action(state_tensor)

                action = action.cpu().numpy()[0]

                # 执行动作
                next_state, reward, done, info = env.step(action)

                # 存储经验
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    log_prob=log_prob.cpu().numpy()[0],
                    value=value.cpu().numpy()[0],
                )
                self.buffer.push(experience)

                episode_reward += reward
                episode_length += 1
                state = next_state

                # 更新策略
                if len(self.buffer) >= self.config.buffer_size:
                    self._update_policy()
                    self.buffer.clear()

                if done:
                    break

            # 记录统计
            self.training_stats["episode_rewards"].append(episode_reward)
            self.training_stats["episode_lengths"].append(episode_length)

            if episode % 100 == 0:
                avg_reward = np.mean(self.training_stats["episode_rewards"][-100:])
                logger.info(f"Episode {episode}: Avg Reward={avg_reward:.4f}")

    def implement_curiosity_driven_exploration(
        self, state: np.ndarray, next_state: np.ndarray
    ) -> float:
        """实现好奇心驱动探索

        Args:
            state: 当前状态
            next_state: 下一状态

        Returns:
            内在奖励
        """
        # 使用预测误差作为内在奖励
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, value = self.actor_critic.forward(state_tensor)
            _, next_value = self.actor_critic.forward(next_state_tensor)

        # 预测误差
        prediction_error = torch.abs(next_value - value).item()

        # 内在奖励
        intrinsic_reward = 0.1 * prediction_error

        return intrinsic_reward

    def execute_learned_policy(
        self, state: np.ndarray, deterministic: bool = True
    ) -> int:
        """执行学习的策略

        Args:
            state: 当前状态
            deterministic: 是否确定性执行

        Returns:
            动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, _ = self.actor_critic.get_action(state_tensor, deterministic)

        return action.cpu().numpy()[0]

    def _update_policy(self) -> None:
        """更新策略"""
        experiences = self.buffer.get_all()

        # 准备数据
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        old_log_probs = torch.FloatTensor([e.log_prob for e in experiences]).to(
            self.device
        )
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)

        # 计算优势和回报
        with torch.no_grad():
            _, values, _ = self.actor_critic.evaluate_actions(states, actions)
            values = values.squeeze()

        advantages, returns = self._compute_gae(rewards, values, dones)

        # PPO更新
        for _ in range(self.config.n_epochs):
            # 随机采样批次
            indices = torch.randperm(len(experiences))

            for start in range(0, len(experiences), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 计算新的log概率、值和熵
                new_log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions
                )
                values = values.squeeze()

                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # 计算裁剪的替代损失
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # 值函数损失
                value_loss = F.mse_loss(values, batch_returns)

                # 熵奖励
                entropy_loss = -entropy.mean()

                # 总损失
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                # 记录统计
                self.training_stats["policy_losses"].append(policy_loss.item())
                self.training_stats["value_losses"].append(value_loss.item())
                self.training_stats["entropies"].append(entropy.mean().item())

    def _compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算广义优势估计(GAE)

        Args:
            rewards: 奖励张量
            values: 值张量
            dones: 结束标志张量

        Returns:
            优势和回报张量
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = (
                rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            )
            gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            )

            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns


# 模块级别函数
def create_ppo_agent(
    state_dim: int, action_dim: int, config: Optional[PPOConfig] = None
) -> PPOAgent:
    """创建PPO智能体的便捷函数

    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        config: PPO配置

    Returns:
        PPO智能体
    """
    if config is None:
        config = PPOConfig(state_dim=state_dim, action_dim=action_dim)

    return PPOAgent(config)
