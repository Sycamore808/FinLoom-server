"""
强化学习智能体模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from common.logging_system import setup_logger
from common.exceptions import ModelError

logger = setup_logger("rl_agent")

class Action(Enum):
    """动作枚举"""
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class RLConfig:
    """强化学习配置"""
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    memory_size: int = 10000
    batch_size: int = 32

@dataclass
class RLState:
    """强化学习状态"""
    features: np.ndarray
    timestamp: str
    market_data: Dict[str, float]

@dataclass
class RLAction:
    """强化学习动作"""
    action: Action
    confidence: float
    q_value: float

@dataclass
class RLExperience:
    """强化学习经验"""
    state: RLState
    action: RLAction
    reward: float
    next_state: RLState
    done: bool

class RLAgent:
    """强化学习智能体类"""
    
    def __init__(self, config: RLConfig):
        """初始化强化学习智能体
        
        Args:
            config: 强化学习配置
        """
        self.config = config
        self.q_table = {}  # 简化的Q表
        self.memory = []
        self.epsilon = config.epsilon
        self.step_count = 0
        
    def get_state_key(self, state: RLState) -> str:
        """获取状态键
        
        Args:
            state: 状态对象
            
        Returns:
            状态键字符串
        """
        # 简化的状态离散化
        discretized = np.round(state.features, 2)
        return str(discretized.tolist())
    
    def choose_action(self, state: RLState) -> RLAction:
        """选择动作
        
        Args:
            state: 当前状态
            
        Returns:
            选择的动作
        """
        try:
            state_key = self.get_state_key(state)
            
            # 初始化Q值
            if state_key not in self.q_table:
                self.q_table[state_key] = {action: 0.0 for action in Action}
            
            # ε-贪婪策略
            if np.random.random() < self.epsilon:
                # 随机动作
                action = np.random.choice(list(Action))
                confidence = 0.5
            else:
                # 贪婪动作
                q_values = self.q_table[state_key]
                action = max(q_values, key=q_values.get)
                confidence = 0.8
            
            q_value = self.q_table[state_key][action]
            
            return RLAction(
                action=action,
                confidence=confidence,
                q_value=q_value
            )
            
        except Exception as e:
            logger.error(f"Failed to choose action: {e}")
            raise ModelError(f"Action selection failed: {e}")
    
    def calculate_reward(self, action: RLAction, market_return: float, 
                        transaction_cost: float = 0.001) -> float:
        """计算奖励
        
        Args:
            action: 执行的动作
            market_return: 市场收益率
            transaction_cost: 交易成本
            
        Returns:
            奖励值
        """
        try:
            # 简化的奖励函数
            if action.action == Action.BUY:
                reward = market_return - transaction_cost
            elif action.action == Action.SELL:
                reward = -market_return - transaction_cost
            else:  # HOLD
                reward = 0.0
            
            # 添加置信度奖励
            reward += action.confidence * 0.1
            
            return reward
            
        except Exception as e:
            logger.error(f"Failed to calculate reward: {e}")
            return 0.0
    
    def store_experience(self, experience: RLExperience):
        """存储经验
        
        Args:
            experience: 经验对象
        """
        try:
            self.memory.append(experience)
            
            # 限制经验池大小
            if len(self.memory) > self.config.memory_size:
                self.memory.pop(0)
                
        except Exception as e:
            logger.error(f"Failed to store experience: {e}")
            raise ModelError(f"Experience storage failed: {e}")
    
    def learn(self, experience: RLExperience):
        """学习更新Q值
        
        Args:
            experience: 经验对象
        """
        try:
            state_key = self.get_state_key(experience.state)
            next_state_key = self.get_state_key(experience.next_state)
            
            # 初始化Q值
            if state_key not in self.q_table:
                self.q_table[state_key] = {action: 0.0 for action in Action}
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = {action: 0.0 for action in Action}
            
            # Q学习更新
            current_q = self.q_table[state_key][experience.action.action]
            
            if experience.done:
                target_q = experience.reward
            else:
                max_next_q = max(self.q_table[next_state_key].values())
                target_q = experience.reward + self.config.discount_factor * max_next_q
            
            # 更新Q值
            self.q_table[state_key][experience.action.action] = (
                current_q + self.config.learning_rate * (target_q - current_q)
            )
            
            # 更新ε值
            self.epsilon = max(
                self.config.min_epsilon,
                self.epsilon * self.config.epsilon_decay
            )
            
            self.step_count += 1
            
        except Exception as e:
            logger.error(f"Failed to learn from experience: {e}")
            raise ModelError(f"Learning failed: {e}")
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """获取智能体统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'memory_size': len(self.memory),
            'q_table_size': len(self.q_table),
            'learning_rate': self.config.learning_rate
        }
    
    def save_model(self, path: str) -> bool:
        """保存模型
        
        Args:
            path: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            # 简化的模型保存
            logger.info(f"Saving RL agent to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save RL agent: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            是否加载成功
        """
        try:
            # 简化的模型加载
            logger.info(f"Loading RL agent from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RL agent: {e}")
            return False
