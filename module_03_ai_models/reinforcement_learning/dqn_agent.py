"""
DQN (Deep Q-Network) 智能体实现

提供深度Q网络用于量化交易：
- 深度Q网络
- 经验回放
- 目标网络
- 双DQN算法
"""

import random
from collections import deque, namedtuple
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common.exceptions import ModelError
from common.logging_system import setup_logger

from ..storage_management.ai_model_database import get_ai_model_database_manager

logger = setup_logger("dqn_agent")

# 经验回放数据结构
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


@dataclass
class DQNConfig:
    """DQN配置"""

    state_dim: int = 10
    action_dim: int = 3  # BUY, SELL, HOLD
    hidden_dims: List[int] = None
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 1000
    memory_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    double_dqn: bool = True

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


class DQNNetwork(nn.Module):
    """DQN网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super(DQNNetwork, self).__init__()

        # 构建网络层
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ReplayMemory:
    """经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """保存一个转换"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        """随机采样批次"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """DQN智能体"""

    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q网络
        self.q_network = DQNNetwork(
            config.state_dim, config.action_dim, config.hidden_dims
        ).to(self.device)

        # 目标网络
        self.target_network = DQNNetwork(
            config.state_dim, config.action_dim, config.hidden_dims
        ).to(self.device)

        # 复制参数到目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 优化器
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=config.learning_rate
        )

        # 经验回放
        self.memory = ReplayMemory(config.memory_size)

        # 训练统计
        self.steps_done = 0
        self.episode_rewards = []
        self.losses = []

        self.db_manager = get_ai_model_database_manager()
        self.agent_id = f"dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作（epsilon-greedy策略）"""
        try:
            # 计算epsilon
            epsilon = self.config.epsilon_end + (
                self.config.epsilon_start - self.config.epsilon_end
            ) * np.exp(-1.0 * self.steps_done / self.config.epsilon_decay)

            self.steps_done += 1

            if training and random.random() < epsilon:
                # 随机动作
                return random.randrange(self.config.action_dim)
            else:
                # 贪婪动作
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    return q_values.max(1)[1].item()

        except Exception as e:
            logger.error(f"Action selection failed: {e}")
            return random.randrange(self.config.action_dim)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ):
        """存储经验"""
        try:
            self.memory.push(state, action, next_state, reward, done)
        except Exception as e:
            logger.error(f"Failed to store transition: {e}")

    def train_step(self) -> Optional[float]:
        """执行一步训练"""
        try:
            if len(self.memory) < self.config.batch_size:
                return None

            # 采样批次
            transitions = self.memory.sample(self.config.batch_size)
            batch = Transition(*zip(*transitions))

            # 转换为张量
            state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
            action_batch = torch.LongTensor(batch.action).to(self.device)
            reward_batch = torch.FloatTensor(batch.reward).to(self.device)
            done_batch = torch.BoolTensor(batch.done).to(self.device)

            # 处理next_state（可能为None）
            non_final_mask = ~done_batch
            non_final_next_states = torch.FloatTensor(
                np.array([s for s, d in zip(batch.next_state, batch.done) if not d])
            ).to(self.device)

            # 当前Q值
            state_action_values = self.q_network(state_batch).gather(
                1, action_batch.unsqueeze(1)
            )

            # 下一状态的Q值
            next_state_values = torch.zeros(self.config.batch_size).to(self.device)

            if len(non_final_next_states) > 0:
                if self.config.double_dqn:
                    # Double DQN
                    next_actions = self.q_network(non_final_next_states).max(1)[1]
                    next_state_values[non_final_mask] = (
                        self.target_network(non_final_next_states)
                        .gather(1, next_actions.unsqueeze(1))
                        .squeeze()
                    )
                else:
                    # 标准DQN
                    next_state_values[non_final_mask] = self.target_network(
                        non_final_next_states
                    ).max(1)[0]

            # 期望Q值
            expected_state_action_values = (
                next_state_values * self.config.gamma
            ) + reward_batch

            # 计算损失
            loss = F.mse_loss(
                state_action_values.squeeze(), expected_state_action_values
            )

            # 优化步骤
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()

            self.losses.append(loss.item())

            # 更新目标网络
            if self.steps_done % self.config.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            return loss.item()

        except Exception as e:
            logger.error(f"Training step failed: {e}")
            return None

    def train_episode(self, env, max_steps: int = 1000) -> Dict[str, float]:
        """训练一个回合"""
        try:
            state = env.reset()
            episode_reward = 0
            step_count = 0
            losses = []

            for step in range(max_steps):
                # 选择动作
                action = self.select_action(state, training=True)

                # 执行动作
                next_state, reward, done, _ = env.step(action)

                # 存储经验
                self.store_transition(state, action, next_state, reward, done)

                # 训练
                loss = self.train_step()
                if loss is not None:
                    losses.append(loss)

                episode_reward += reward
                step_count += 1

                if done:
                    break

                state = next_state

            self.episode_rewards.append(episode_reward)

            return {
                "episode_reward": episode_reward,
                "steps": step_count,
                "average_loss": np.mean(losses) if losses else 0.0,
                "epsilon": self.config.epsilon_end
                + (self.config.epsilon_start - self.config.epsilon_end)
                * np.exp(-1.0 * self.steps_done / self.config.epsilon_decay),
            }

        except Exception as e:
            logger.error(f"Episode training failed: {e}")
            return {
                "episode_reward": 0.0,
                "steps": 0,
                "average_loss": 0.0,
                "epsilon": 0.0,
            }

    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """评估智能体性能"""
        try:
            eval_rewards = []

            for episode in range(num_episodes):
                state = env.reset()
                episode_reward = 0

                for step in range(1000):  # 最大步数
                    action = self.select_action(state, training=False)
                    next_state, reward, done, _ = env.step(action)

                    episode_reward += reward

                    if done:
                        break

                    state = next_state

                eval_rewards.append(episode_reward)

            return {
                "average_reward": np.mean(eval_rewards),
                "std_reward": np.std(eval_rewards),
                "min_reward": np.min(eval_rewards),
                "max_reward": np.max(eval_rewards),
                "num_episodes": num_episodes,
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "average_reward": 0.0,
                "std_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "num_episodes": 0,
            }

    def save_model(self, filepath: str) -> bool:
        """保存模型"""
        try:
            torch.save(
                {
                    "q_network_state_dict": self.q_network.state_dict(),
                    "target_network_state_dict": self.target_network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config,
                    "steps_done": self.steps_done,
                    "episode_rewards": self.episode_rewards,
                    "agent_id": self.agent_id,
                },
                filepath,
            )

            logger.info(f"DQN model saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save DQN model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.config = checkpoint["config"]
            self.steps_done = checkpoint["steps_done"]
            self.episode_rewards = checkpoint["episode_rewards"]
            self.agent_id = checkpoint.get("agent_id", self.agent_id)

            logger.info(f"DQN model loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load DQN model: {e}")
            return False

    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        try:
            recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else []
            recent_losses = self.losses[-100:] if self.losses else []

            return {
                "steps_done": self.steps_done,
                "episodes_trained": len(self.episode_rewards),
                "average_reward": np.mean(recent_rewards) if recent_rewards else 0.0,
                "reward_std": np.std(recent_rewards) if recent_rewards else 0.0,
                "average_loss": np.mean(recent_losses) if recent_losses else 0.0,
                "memory_size": len(self.memory),
                "current_epsilon": self.config.epsilon_end
                + (self.config.epsilon_start - self.config.epsilon_end)
                * np.exp(-1.0 * self.steps_done / self.config.epsilon_decay),
                "agent_id": self.agent_id,
            }

        except Exception as e:
            logger.error(f"Failed to get training stats: {e}")
            return {}


def create_dqn_agent(
    state_dim: int,
    action_dim: int = 3,
    hidden_dims: List[int] = None,
    learning_rate: float = 0.001,
    double_dqn: bool = True,
) -> DQNAgent:
    """创建DQN智能体的便捷函数"""
    config = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims or [128, 64, 32],
        learning_rate=learning_rate,
        double_dqn=double_dqn,
    )
    return DQNAgent(config)
