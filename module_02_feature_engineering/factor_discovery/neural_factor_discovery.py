"""
神经因子发现器模块
使用神经网络自动发现和创建投资因子
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from common.constants import DEFAULT_BATCH_SIZE, EARLY_STOPPING_PATIENCE, MAX_EPOCHS
from common.exceptions import ModelError
from common.logging_system import setup_logger
from torch.utils.data import DataLoader, Dataset

logger = setup_logger("neural_factor_discovery")


@dataclass
class FactorConfig:
    """因子配置"""

    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = MAX_EPOCHS
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    use_attention: bool = True
    use_residual: bool = True


@dataclass
class DiscoveredFactor:
    """发现的因子"""

    factor_id: str
    name: str
    formula: str  # 符号化表示
    importance_score: float
    ic_score: float  # Information Coefficient
    ir_score: float  # Information Ratio
    stability_score: float
    weights: np.ndarray
    metadata: Dict[str, Any]


class FactorDataset(Dataset):
    """因子数据集"""

    def __init__(
        self, features: np.ndarray, returns: np.ndarray, sequence_length: int = 20
    ):
        """初始化数据集

        Args:
            features: 特征数组 (样本数, 时间步, 特征数)
            returns: 收益率数组 (样本数,)
            sequence_length: 序列长度
        """
        self.features = torch.FloatTensor(features)
        self.returns = torch.FloatTensor(returns)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        """数据集长度"""
        return len(self.returns)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取数据项"""
        return self.features[idx], self.returns[idx]


class AttentionLayer(nn.Module):
    """注意力层"""

    def __init__(self, input_dim: int, hidden_dim: int):
        """初始化注意力层

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏维度
        """
        super(AttentionLayer, self).__init__()
        self.query_layer = nn.Linear(input_dim, hidden_dim)
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(input_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)

        Returns:
            输出张量和注意力权重
        """
        batch_size, seq_len, _ = x.size()

        # 计算Q, K, V
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)

        # 应用注意力权重
        context = torch.matmul(attention_weights, V)

        return context, attention_weights


class NeuralFactorNetwork(nn.Module):
    """神经因子网络"""

    def __init__(self, config: FactorConfig):
        """初始化网络

        Args:
            config: 因子配置
        """
        super(NeuralFactorNetwork, self).__init__()
        self.config = config

        # 构建编码器层
        encoder_layers = []
        input_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.Dropout(config.dropout_rate))
            input_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # 注意力层
        if config.use_attention:
            self.attention = AttentionLayer(
                config.hidden_dims[-1], config.hidden_dims[-1]
            )
        else:
            self.attention = None

        # 输出层
        self.output_layer = nn.Linear(config.hidden_dims[-1], config.output_dim)

        # 残差连接
        if config.use_residual and config.input_dim == config.output_dim:
            self.residual = nn.Identity()
        else:
            self.residual = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播

        Args:
            x: 输入张量

        Returns:
            输出张量和注意力权重（如果使用注意力）
        """
        # 编码
        encoded = self.encoder(x)

        # 注意力机制
        attention_weights = None
        if self.attention is not None:
            # 重塑为序列格式
            batch_size = encoded.size(0)
            encoded = encoded.unsqueeze(1)  # (batch, 1, features)
            attended, attention_weights = self.attention(encoded)
            encoded = attended.squeeze(1)

        # 输出
        output = self.output_layer(encoded)

        # 残差连接
        if self.residual is not None:
            output = output + self.residual(x)

        return output, attention_weights


class NeuralFactorDiscovery:
    """神经因子发现器"""

    def __init__(self, config: FactorConfig):
        """初始化因子发现器

        Args:
            config: 因子配置
        """
        self.config = config
        self.model = NeuralFactorNetwork(config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        self.discovered_factors: List[DiscoveredFactor] = []
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "ic": [],
            "ir": [],
        }

    def discover_neural_factors(
        self, features: pd.DataFrame, returns: pd.Series, validation_split: float = 0.2
    ) -> List[DiscoveredFactor]:
        """发现神经因子

        Args:
            features: 特征DataFrame
            returns: 收益率Series
            validation_split: 验证集比例

        Returns:
            发现的因子列表
        """
        logger.info("Starting neural factor discovery...")

        # 准备数据
        X = features.values
        y = returns.values

        # 分割训练集和验证集
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 创建数据加载器
        train_dataset = FactorDataset(X_train, y_train)
        val_dataset = FactorDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        # 训练模型
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            # 训练阶段
            train_loss = self._train_epoch(train_loader)

            # 验证阶段
            val_loss, ic, ir = self._validate_epoch(val_loader)

            # 记录历史
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["ic"].append(ic)
            self.training_history["ir"].append(ir)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_best_model()
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, IC={ic:.4f}, IR={ir:.4f}"
                )

        # 提取因子
        self._load_best_model()
        factors = self._extract_factors(features)

        logger.info(f"Discovered {len(factors)} neural factors")
        return factors

    def extract_attention_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """提取注意力特征

        Args:
            features: 输入特征

        Returns:
            注意力加权特征
        """
        if not self.config.use_attention:
            raise ModelError("Model was not configured with attention mechanism")

        self.model.eval()
        X = torch.FloatTensor(features.values)

        with torch.no_grad():
            _, attention_weights = self.model(X)

        if attention_weights is not None:
            # 应用注意力权重到原始特征
            weighted_features = features.values * attention_weights.numpy().squeeze()
            result_df = pd.DataFrame(
                weighted_features,
                index=features.index,
                columns=[f"att_{col}" for col in features.columns],
            )
            return result_df
        else:
            return features

    def generate_interaction_features(
        self, features: pd.DataFrame, max_interactions: int = 10
    ) -> pd.DataFrame:
        """生成交互特征

        Args:
            features: 输入特征
            max_interactions: 最大交互数

        Returns:
            交互特征DataFrame
        """
        interaction_features = []
        feature_names = []

        # 获取模型的中间层激活
        self.model.eval()
        X = torch.FloatTensor(features.values)

        # 注册钩子获取中间层输出
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()

            return hook

        # 为每个编码器层注册钩子
        for i, layer in enumerate(self.model.encoder):
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(get_activation(f"layer_{i}"))

        # 前向传播
        with torch.no_grad():
            _ = self.model(X)

        # 提取交互特征
        for name, activation in activations.items():
            if len(interaction_features) >= max_interactions:
                break

            # 计算特征间的交互
            act_np = activation.numpy()

            # 二次交互
            for i in range(min(5, act_np.shape[1])):
                for j in range(i + 1, min(5, act_np.shape[1])):
                    interaction = act_np[:, i] * act_np[:, j]
                    interaction_features.append(interaction)
                    feature_names.append(f"{name}_interact_{i}_{j}")

                    if len(interaction_features) >= max_interactions:
                        break

        # 创建DataFrame
        if interaction_features:
            result_df = pd.DataFrame(
                np.column_stack(interaction_features),
                index=features.index,
                columns=feature_names[: len(interaction_features)],
            )
            return result_df
        else:
            return pd.DataFrame(index=features.index)

    def evaluate_factor_effectiveness(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        periods: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        """评估因子有效性

        Args:
            factor_values: 因子值
            forward_returns: 前向收益率
            periods: 评估周期列表

        Returns:
            评估指标字典
        """
        metrics = {}

        # 计算不同周期的IC
        for period in periods:
            if period <= len(forward_returns):
                # 计算信息系数(IC)
                ic = factor_values.corr(forward_returns.shift(-period))
                metrics[f"ic_{period}d"] = ic

                # 计算IC的稳定性
                rolling_ic = factor_values.rolling(window=period).corr(
                    forward_returns.rolling(window=period).mean()
                )
                metrics[f"ic_std_{period}d"] = rolling_ic.std()

                # 计算信息比率(IR)
                if rolling_ic.std() > 0:
                    ir = rolling_ic.mean() / rolling_ic.std()
                    metrics[f"ir_{period}d"] = ir
                else:
                    metrics[f"ir_{period}d"] = 0.0

        # 计算因子收益
        factor_quantiles = pd.qcut(factor_values, q=5, labels=False)

        top_quintile_return = forward_returns[factor_quantiles == 4].mean()
        bottom_quintile_return = forward_returns[factor_quantiles == 0].mean()

        metrics["long_short_return"] = top_quintile_return - bottom_quintile_return
        metrics["monotonicity"] = self._calculate_monotonicity(
            factor_quantiles, forward_returns
        )

        return metrics

    def optimize_factor_combination(
        self,
        factors: List[pd.Series],
        target_returns: pd.Series,
        method: str = "linear",
    ) -> Dict[str, float]:
        """优化因子组合

        Args:
            factors: 因子列表
            target_returns: 目标收益率
            method: 优化方法 ('linear', 'nonlinear')

        Returns:
            因子权重字典
        """
        factor_matrix = pd.concat(factors, axis=1)

        if method == "linear":
            # 使用线性回归优化权重
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(factor_matrix, target_returns)

            weights = dict(zip(factor_matrix.columns, model.coef_))

        elif method == "nonlinear":
            # 使用神经网络优化权重
            X = factor_matrix.values
            y = target_returns.values

            # 简单的神经网络优化
            optimizer_net = nn.Sequential(
                nn.Linear(X.shape[1], 32), nn.ReLU(), nn.Linear(32, 1)
            )

            optimizer = optim.Adam(optimizer_net.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y).unsqueeze(1)

            for _ in range(100):
                optimizer.zero_grad()
                output = optimizer_net(X_tensor)
                loss = criterion(output, y_tensor)
                loss.backward()
                optimizer.step()

            # 提取第一层权重作为因子权重
            first_layer_weights = optimizer_net[0].weight.data.numpy()
            weights = dict(zip(factor_matrix.columns, first_layer_weights.mean(axis=0)))

        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # 归一化权重
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0

        for batch_features, batch_returns in train_loader:
            self.optimizer.zero_grad()

            predictions, _ = self.model(batch_features)
            loss = self.criterion(predictions.squeeze(), batch_returns)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """验证一个epoch

        Args:
            val_loader: 验证数据加载器

        Returns:
            验证损失、IC、IR
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_features, batch_returns in val_loader:
                predictions, _ = self.model(batch_features)
                loss = self.criterion(predictions.squeeze(), batch_returns)

                total_loss += loss.item()
                all_predictions.extend(predictions.squeeze().numpy())
                all_targets.extend(batch_returns.numpy())

        avg_loss = total_loss / len(val_loader)

        # 计算IC和IR
        predictions_series = pd.Series(all_predictions)
        targets_series = pd.Series(all_targets)

        ic = predictions_series.corr(targets_series)

        # 计算滚动IC的IR
        window_size = min(20, len(predictions_series) // 5)
        if window_size > 0:
            rolling_ic = predictions_series.rolling(window=window_size).corr(
                targets_series.rolling(window=window_size).mean()
            )
            ir = rolling_ic.mean() / rolling_ic.std() if rolling_ic.std() > 0 else 0.0
        else:
            ir = 0.0

        return avg_loss, ic, ir

    def _extract_factors(self, features: pd.DataFrame) -> List[DiscoveredFactor]:
        """提取因子

        Args:
            features: 特征DataFrame

        Returns:
            发现的因子列表
        """
        factors = []

        # 获取模型权重
        for name, param in self.model.named_parameters():
            if "weight" in name and param.requires_grad:
                weights = param.data.numpy()

                # 创建因子
                factor = DiscoveredFactor(
                    factor_id=f"neural_factor_{len(factors)}",
                    name=f"NeuralFactor_{name}",
                    formula=self._generate_formula(weights),
                    importance_score=np.abs(weights).mean(),
                    ic_score=self.training_history["ic"][-1]
                    if self.training_history["ic"]
                    else 0.0,
                    ir_score=self.training_history["ir"][-1]
                    if self.training_history["ir"]
                    else 0.0,
                    stability_score=self._calculate_stability(weights),
                    weights=weights,
                    metadata={
                        "layer_name": name,
                        "weight_shape": weights.shape,
                        "training_epochs": len(self.training_history["train_loss"]),
                    },
                )
                factors.append(factor)

        self.discovered_factors = factors
        return factors

    def _generate_formula(self, weights: np.ndarray) -> str:
        """生成因子公式的符号表示

        Args:
            weights: 权重数组

        Returns:
            公式字符串
        """
        # 简化的公式生成
        if weights.ndim == 1:
            terms = []
            for i, w in enumerate(weights[:5]):  # 只显示前5个
                if abs(w) > 0.01:
                    terms.append(f"{w:.3f}*X{i}")
            if len(weights) > 5:
                terms.append("...")
            return " + ".join(terms)
        else:
            return f"Matrix[{weights.shape}]"

    def _calculate_stability(self, weights: np.ndarray) -> float:
        """计算权重稳定性

        Args:
            weights: 权重数组

        Returns:
            稳定性分数
        """
        # 计算权重的变异系数的倒数作为稳定性
        if weights.std() > 0:
            cv = weights.std() / (abs(weights.mean()) + 1e-8)
            stability = 1.0 / (1.0 + cv)
        else:
            stability = 1.0
        return stability

    def _calculate_monotonicity(
        self, factor_quantiles: pd.Series, returns: pd.Series
    ) -> float:
        """计算因子单调性

        Args:
            factor_quantiles: 因子分位数
            returns: 收益率

        Returns:
            单调性分数
        """
        # 计算每个分位数的平均收益
        quantile_returns = []
        for q in range(5):
            q_returns = returns[factor_quantiles == q].mean()
            quantile_returns.append(q_returns)

        # 检查单调性
        monotonic_count = 0
        for i in range(len(quantile_returns) - 1):
            if quantile_returns[i + 1] > quantile_returns[i]:
                monotonic_count += 1

        monotonicity = monotonic_count / (len(quantile_returns) - 1)
        return monotonicity

    def _save_best_model(self) -> None:
        """保存最佳模型"""
        torch.save(self.model.state_dict(), "best_neural_factor_model.pth")

    def _load_best_model(self) -> None:
        """加载最佳模型"""
        self.model.load_state_dict(torch.load("best_neural_factor_model.pth"))


# 模块级别函数
def discover_factors(
    features: pd.DataFrame, returns: pd.Series, config: Optional[FactorConfig] = None
) -> List[DiscoveredFactor]:
    """发现因子的便捷函数

    Args:
        features: 特征DataFrame
        returns: 收益率Series
        config: 因子配置（可选）

    Returns:
        发现的因子列表
    """
    if config is None:
        config = FactorConfig(
            input_dim=features.shape[1], hidden_dims=[128, 64, 32], output_dim=1
        )

    discoverer = NeuralFactorDiscovery(config)
    return discoverer.discover_neural_factors(features, returns)
