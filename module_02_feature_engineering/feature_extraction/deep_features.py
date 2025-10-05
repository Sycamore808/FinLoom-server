"""
深度特征提取模块
使用深度学习方法提取高层次特征
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# 尝试导入深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # 创建占位符类
    class nn:
        class Module:
            pass

        class LSTM:
            pass

        class Linear:
            pass

        class Dropout:
            pass

    torch = None

from common.exceptions import DataError, ModelError
from common.logging_system import setup_logger

logger = setup_logger("deep_features")


@dataclass
class DeepFeatureConfig:
    """深度特征配置"""

    sequence_length: int = 20
    hidden_size: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.2
    feature_dim: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 32


class SequenceDataset(Dataset):
    """序列数据集"""

    def __init__(self, data: np.ndarray, sequence_length: int):
        """初始化序列数据集

        Args:
            data: 输入数据 (n_samples, n_features)
            sequence_length: 序列长度
        """
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        if not TORCH_AVAILABLE:
            raise ModelError("PyTorch is not available")

        sequence = self.data[idx : idx + self.sequence_length]
        return torch.FloatTensor(sequence)


class AutoEncoder(nn.Module):
    """自编码器用于特征提取"""

    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int):
        """初始化自编码器

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            feature_dim: 特征维度
        """
        if not TORCH_AVAILABLE:
            raise ModelError("PyTorch is not available")

        super(AutoEncoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x):
        """仅编码"""
        return self.encoder(x)


class LSTMAutoEncoder(nn.Module):
    """LSTM自编码器"""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        feature_dim: int,
        dropout_rate: float = 0.2,
    ):
        """初始化LSTM自编码器

        Args:
            input_dim: 输入特征维度
            hidden_size: LSTM隐藏状态大小
            num_layers: LSTM层数
            feature_dim: 特征维度
            dropout_rate: Dropout率
        """
        if not TORCH_AVAILABLE:
            raise ModelError("PyTorch is not available")

        super(LSTMAutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_dim = feature_dim

        # 编码器LSTM
        self.encoder_lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        # 特征提取层
        self.feature_layer = nn.Linear(hidden_size, feature_dim)

        # 解码器LSTM
        self.decoder_lstm = nn.LSTM(
            feature_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        # 输出层
        self.output_layer = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入序列 (batch_size, seq_len, input_dim)

        Returns:
            features: 提取的特征 (batch_size, feature_dim)
            reconstructed: 重构的序列 (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()

        # 编码
        encoded, (h_n, c_n) = self.encoder_lstm(x)

        # 使用最后时刻的隐藏状态提取特征
        features = self.feature_layer(h_n[-1])  # (batch_size, feature_dim)

        # 解码
        # 将特征扩展为序列
        decoder_input = features.unsqueeze(1).repeat(
            1, seq_len, 1
        )  # (batch_size, seq_len, feature_dim)
        decoded, _ = self.decoder_lstm(decoder_input)
        reconstructed = self.output_layer(decoded)

        return features, reconstructed

    def encode(self, x):
        """仅编码获取特征

        Args:
            x: 输入序列

        Returns:
            features: 提取的特征
        """
        encoded, (h_n, c_n) = self.encoder_lstm(x)
        features = self.feature_layer(h_n[-1])
        return features


class DeepFeatureExtractor:
    """深度特征提取器"""

    def __init__(self, config: Optional[DeepFeatureConfig] = None):
        """初始化深度特征提取器

        Args:
            config: 深度特征配置
        """
        self.config = config or DeepFeatureConfig()
        self.autoencoder = None
        self.lstm_autoencoder = None
        self.is_fitted = False

        if not TORCH_AVAILABLE:
            logger.warning(
                "PyTorch is not available. Deep feature extraction will be limited."
            )

    def fit_autoencoder(self, data: pd.DataFrame) -> bool:
        """训练自编码器

        Args:
            data: 训练数据

        Returns:
            是否训练成功
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping autoencoder training")
            return False

        try:
            # 数据预处理
            data_array = data.values
            data_normalized = (data_array - data_array.mean(axis=0)) / (
                data_array.std(axis=0) + 1e-8
            )

            # 创建模型
            input_dim = data_normalized.shape[1]
            self.autoencoder = AutoEncoder(
                input_dim, self.config.hidden_size, self.config.feature_dim
            )

            # 创建数据加载器
            dataset = torch.FloatTensor(data_normalized)
            dataloader = DataLoader(
                dataset, batch_size=self.config.batch_size, shuffle=True
            )

            # 训练
            optimizer = torch.optim.Adam(
                self.autoencoder.parameters(), lr=self.config.learning_rate
            )
            criterion = nn.MSELoss()

            self.autoencoder.train()
            for epoch in range(self.config.epochs):
                total_loss = 0
                for batch in dataloader:
                    optimizer.zero_grad()
                    features, reconstructed = self.autoencoder(batch)
                    loss = criterion(reconstructed, batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if epoch % 10 == 0:
                    avg_loss = total_loss / len(dataloader)
                    logger.info(f"AutoEncoder Epoch {epoch}: Loss = {avg_loss:.6f}")

            self.is_fitted = True
            logger.info("AutoEncoder training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to train autoencoder: {e}")
            return False

    def fit_lstm_autoencoder(self, data: pd.DataFrame) -> bool:
        """训练LSTM自编码器

        Args:
            data: 训练数据

        Returns:
            是否训练成功
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping LSTM autoencoder training")
            return False

        try:
            # 数据预处理
            data_array = data.values
            data_normalized = (data_array - data_array.mean(axis=0)) / (
                data_array.std(axis=0) + 1e-8
            )

            # 创建序列数据
            dataset = SequenceDataset(data_normalized, self.config.sequence_length)
            dataloader = DataLoader(
                dataset, batch_size=self.config.batch_size, shuffle=True
            )

            # 创建模型
            input_dim = data_normalized.shape[1]
            self.lstm_autoencoder = LSTMAutoEncoder(
                input_dim,
                self.config.hidden_size,
                self.config.num_layers,
                self.config.feature_dim,
                self.config.dropout_rate,
            )

            # 训练
            optimizer = torch.optim.Adam(
                self.lstm_autoencoder.parameters(), lr=self.config.learning_rate
            )
            criterion = nn.MSELoss()

            self.lstm_autoencoder.train()
            for epoch in range(self.config.epochs):
                total_loss = 0
                for batch in dataloader:
                    optimizer.zero_grad()
                    features, reconstructed = self.lstm_autoencoder(batch)
                    loss = criterion(reconstructed, batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if epoch % 10 == 0:
                    avg_loss = total_loss / len(dataloader)
                    logger.info(
                        f"LSTM AutoEncoder Epoch {epoch}: Loss = {avg_loss:.6f}"
                    )

            self.is_fitted = True
            logger.info("LSTM AutoEncoder training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to train LSTM autoencoder: {e}")
            return False

    def extract_autoencoder_features(self, data: pd.DataFrame) -> np.ndarray:
        """使用自编码器提取特征

        Args:
            data: 输入数据

        Returns:
            提取的特征
        """
        if not TORCH_AVAILABLE or self.autoencoder is None:
            logger.warning("AutoEncoder not available or not trained")
            return np.array([])

        try:
            # 数据预处理
            data_array = data.values
            data_normalized = (data_array - data_array.mean(axis=0)) / (
                data_array.std(axis=0) + 1e-8
            )

            # 提取特征
            self.autoencoder.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(data_normalized)
                features = self.autoencoder.encode(input_tensor)
                return features.numpy()

        except Exception as e:
            logger.error(f"Failed to extract autoencoder features: {e}")
            return np.array([])

    def extract_lstm_features(self, data: pd.DataFrame) -> np.ndarray:
        """使用LSTM自编码器提取特征

        Args:
            data: 输入数据

        Returns:
            提取的特征
        """
        if not TORCH_AVAILABLE or self.lstm_autoencoder is None:
            logger.warning("LSTM AutoEncoder not available or not trained")
            return np.array([])

        try:
            # 数据预处理
            data_array = data.values
            data_normalized = (data_array - data_array.mean(axis=0)) / (
                data_array.std(axis=0) + 1e-8
            )

            # 创建序列
            if len(data_normalized) < self.config.sequence_length:
                logger.warning(
                    f"Data length {len(data_normalized)} < sequence length {self.config.sequence_length}"
                )
                return np.array([])

            sequences = []
            for i in range(len(data_normalized) - self.config.sequence_length + 1):
                sequences.append(data_normalized[i : i + self.config.sequence_length])

            # 提取特征
            self.lstm_autoencoder.eval()
            features_list = []

            with torch.no_grad():
                for seq in sequences:
                    input_tensor = torch.FloatTensor(seq).unsqueeze(
                        0
                    )  # Add batch dimension
                    features = self.lstm_autoencoder.encode(input_tensor)
                    features_list.append(features.squeeze().numpy())

            return np.array(features_list)

        except Exception as e:
            logger.error(f"Failed to extract LSTM features: {e}")
            return np.array([])

    def extract_statistical_deep_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """提取基于统计的深度特征

        Args:
            data: 输入数据

        Returns:
            统计深度特征字典
        """
        try:
            features = {}

            # 多层感知机特征 (不需要PyTorch)
            data_array = data.values

            # 非线性变换特征
            features.update(self._extract_nonlinear_features(data_array))

            # 交互特征
            features.update(self._extract_interaction_features(data_array))

            # 高阶统计特征
            features.update(self._extract_higher_order_features(data_array))

            logger.info(f"Extracted {len(features)} statistical deep features")
            return features

        except Exception as e:
            logger.error(f"Failed to extract statistical deep features: {e}")
            return {}

    def _extract_nonlinear_features(self, data: np.ndarray) -> Dict[str, float]:
        """提取非线性特征"""
        features = {}

        try:
            # 对每个特征应用非线性变换
            for i in range(min(data.shape[1], 10)):  # 限制特征数量
                col_data = data[:, i]
                col_data_clean = col_data[~np.isnan(col_data)]

                if len(col_data_clean) > 0:
                    # 非线性变换
                    features[f"tanh_mean_{i}"] = np.tanh(col_data_clean).mean()
                    features[f"sigmoid_mean_{i}"] = (
                        1 / (1 + np.exp(-col_data_clean))
                    ).mean()
                    features[f"relu_mean_{i}"] = np.maximum(0, col_data_clean).mean()

                    # 分段线性特征
                    features[f"positive_ratio_{i}"] = (col_data_clean > 0).mean()
                    features[f"extreme_ratio_{i}"] = (
                        np.abs(col_data_clean) > 2 * np.std(col_data_clean)
                    ).mean()

            return features

        except Exception as e:
            logger.error(f"Failed to extract nonlinear features: {e}")
            return {}

    def _extract_interaction_features(self, data: np.ndarray) -> Dict[str, float]:
        """提取交互特征"""
        features = {}

        try:
            # 特征间交互
            n_features = min(data.shape[1], 5)  # 限制特征数量

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    col1 = data[:, i]
                    col2 = data[:, j]

                    # 清理NaN
                    valid_mask = ~(np.isnan(col1) | np.isnan(col2))
                    if valid_mask.sum() > 0:
                        col1_clean = col1[valid_mask]
                        col2_clean = col2[valid_mask]

                        # 乘积特征
                        features[f"product_{i}_{j}"] = (col1_clean * col2_clean).mean()

                        # 比率特征
                        if np.abs(col2_clean).mean() > 1e-8:
                            features[f"ratio_{i}_{j}"] = (
                                col1_clean / col2_clean
                            ).mean()

                        # 差值特征
                        features[f"diff_{i}_{j}"] = (col1_clean - col2_clean).mean()

            return features

        except Exception as e:
            logger.error(f"Failed to extract interaction features: {e}")
            return {}

    def _extract_higher_order_features(self, data: np.ndarray) -> Dict[str, float]:
        """提取高阶特征"""
        features = {}

        try:
            # 对每个特征计算高阶统计量
            for i in range(min(data.shape[1], 10)):
                col_data = data[:, i]
                col_data_clean = col_data[~np.isnan(col_data)]

                if len(col_data_clean) > 0:
                    # 高阶矩
                    if len(col_data_clean) > 1:
                        mean_val = np.mean(col_data_clean)
                        std_val = np.std(col_data_clean)

                        if std_val > 0:
                            # 标准化数据
                            normalized = (col_data_clean - mean_val) / std_val

                            # 3阶和4阶矩
                            features[f"moment_3_{i}"] = np.mean(normalized**3)
                            features[f"moment_4_{i}"] = np.mean(normalized**4)

                            # L-矩特征
                            features[f"l_moment_2_{i}"] = np.mean(np.abs(normalized))
                            features[f"l_moment_3_{i}"] = np.mean(
                                normalized * np.abs(normalized)
                            )

            return features

        except Exception as e:
            logger.error(f"Failed to extract higher order features: {e}")
            return {}

    def extract_all_deep_features(
        self, data: pd.DataFrame, train_models: bool = True
    ) -> Dict[str, Union[float, np.ndarray]]:
        """提取所有深度特征

        Args:
            data: 输入数据
            train_models: 是否训练深度学习模型

        Returns:
            所有深度特征
        """
        try:
            all_features = {}

            # 统计深度特征（总是可用）
            statistical_features = self.extract_statistical_deep_features(data)
            all_features.update(statistical_features)

            if TORCH_AVAILABLE and train_models:
                # 训练并提取自编码器特征
                if self.fit_autoencoder(data):
                    autoencoder_features = self.extract_autoencoder_features(data)
                    if autoencoder_features.size > 0:
                        # 转换为特征字典
                        for i, feature_val in enumerate(
                            autoencoder_features.mean(axis=0)
                        ):
                            all_features[f"autoencoder_feature_{i}"] = float(
                                feature_val
                            )

                # 训练并提取LSTM特征
                if len(
                    data
                ) >= self.config.sequence_length and self.fit_lstm_autoencoder(data):
                    lstm_features = self.extract_lstm_features(data)
                    if lstm_features.size > 0:
                        # 转换为特征字典
                        for i, feature_val in enumerate(lstm_features.mean(axis=0)):
                            all_features[f"lstm_feature_{i}"] = float(feature_val)

            logger.info(f"Extracted {len(all_features)} total deep features")
            return all_features

        except Exception as e:
            logger.error(f"Failed to extract all deep features: {e}")
            raise DataError(f"Deep feature extraction failed: {e}")


# 便捷函数
def extract_deep_features(
    data: pd.DataFrame,
    config: Optional[DeepFeatureConfig] = None,
    train_models: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """提取深度特征的便捷函数

    Args:
        data: 输入数据
        config: 深度特征配置
        train_models: 是否训练深度学习模型

    Returns:
        深度特征字典
    """
    extractor = DeepFeatureExtractor(config)
    return extractor.extract_all_deep_features(data, train_models)
