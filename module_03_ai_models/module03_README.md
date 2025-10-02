# Module 03 - AI Models

## Overview

Module 03 provides comprehensive AI modeling capabilities for the FinLoom quantitative trading system. It integrates with Module 01 (data pipeline) and Module 02 (feature engineering) to deliver machine learning predictions and trading decisions through multiple learning paradigms including deep learning, ensemble methods, online learning, and reinforcement learning.

## Core Components

### 1. Deep Learning Models
- **LSTMModel**: PyTorch-based LSTM for financial time series prediction
- **TransformerPredictor**: Temporal transformer with attention mechanisms
- **CNNPredictor**: 1D CNN for pattern recognition in financial data
- GPU acceleration support and automatic data preprocessing

### 2. Ensemble Methods
- **EnsemblePredictor**: Multi-model ensemble with intelligent weighting
- **ModelEnsemble**: Advanced ensemble with voting, weighted, and stacking strategies
- **StackingEnsemble**: Cross-validation based stacking with meta-learners
- Performance-based weight optimization

### 3. Online Learning
- **OnlineLearner**: Adaptive online learning with concept drift detection
- **AdaptiveModel**: Model adaptation with drift detection (DDM, EDDM, ADWIN)
- **OnlineUpdater**: Real-time model updating with background processing
- Momentum optimization and adaptive learning rates

### 4. Reinforcement Learning
- **RLAgent**: Q-Learning based trading agent
- **PPOAgent**: Proximal Policy Optimization agent
- **DQNAgent**: Deep Q-Network with experience replay
- **TradingEnvironment**: Realistic market simulation environment

### 5. Storage Management
- **AIModelDatabaseManager**: Complete model lifecycle management
- Model parameters, training history, and prediction persistence
- Performance tracking and model versioning
- SQLite integration for data storage

## Quick Start

### Import Core Components

```python
# Core AI models
from module_03_ai_models import (
    LSTMModel, LSTMModelConfig, LSTMPrediction,
    TransformerPredictor, TransformerConfig, TemporalTransformer,
    EnsemblePredictor, EnsembleConfig, EnsemblePrediction,
    OnlineLearner, OnlineLearningConfig, OnlineLearningResult,
    RLAgent, RLConfig, RLState, RLAction, Action,
    PPOAgent, PPOConfig, TradingEnvironment,
    get_ai_model_database_manager
)

# Utility functions
from module_03_ai_models.utils import (
    prepare_features_for_training,
    train_ensemble_model,
    evaluate_model_performance,
    create_lstm_predictor,
    create_online_learner
)

# Advanced components
from module_03_ai_models.deep_learning.cnn_model import (
    CNNPredictor, CNNModelConfig, create_cnn_predictor
)
from module_03_ai_models.ensemble_methods.model_ensemble import (
    ModelEnsemble, create_model_ensemble
)
from module_03_ai_models.ensemble_methods.stacking import (
    StackingEnsemble, create_stacking_ensemble
)
from module_03_ai_models.online_learning.adaptive_model import (
    AdaptiveModel, create_adaptive_model
)
from module_03_ai_models.reinforcement_learning.dqn_agent import (
    DQNAgent, create_dqn_agent
)
```

### Basic Usage Example

```python
# 1. Prepare data (integrated with Module 01 and 02)
from module_01_data_pipeline import AkshareDataCollector
from module_02_feature_engineering import TechnicalIndicators
from datetime import datetime, timedelta
import pandas as pd

# Collect data
collector = AkshareDataCollector()
end_date = datetime.now().strftime("%Y%m%d")
start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
stock_data = collector.fetch_stock_history("000001", start_date, end_date)

# Calculate technical indicators
calculator = TechnicalIndicators()
features = calculator.calculate_all_indicators(stock_data)

# 2. Create and configure LSTM model
config = LSTMModelConfig(
    sequence_length=5,
    hidden_size=32,
    num_layers=1,
    dropout=0.2,
    learning_rate=0.001,
    batch_size=16,
    epochs=10
)

lstm_model = LSTMModel(config)
lstm_model.set_model_id("lstm_predictor_v1")

# 3. Prepare training data
features['returns'] = features['close'].pct_change().fillna(0)
X, y = lstm_model.prepare_data(features, 'returns')

# 4. Train model
training_metrics = lstm_model.train(X, y)
print(f"Training completed: {training_metrics}")

# 5. Make predictions
test_features = features.drop(columns=['returns']).values[-5:]
predictions = lstm_model.predict(test_features)
print(f"Predictions: {predictions.predictions}")

# 6. Save model
success = lstm_model.save_model("lstm_predictor_v1")
print(f"Model saved: {success}")
```

## API Reference

### 1. Deep Learning Models

#### LSTMModel

PyTorch-based LSTM for financial time series prediction.

**Constructor**
```python
LSTMModel(config: LSTMModelConfig)
```

**Configuration**
```python
@dataclass
class LSTMModelConfig:
    sequence_length: int = 60    # Input sequence length
    hidden_size: int = 50        # LSTM hidden size
    num_layers: int = 2          # Number of LSTM layers
    dropout: float = 0.2         # Dropout rate
    learning_rate: float = 0.001 # Learning rate
    batch_size: int = 32         # Training batch size
    epochs: int = 100            # Training epochs
```

**Key Methods**

- `set_model_id(model_id: str)` - Set model identifier
- `prepare_data(data: pd.DataFrame, target_column: str) -> Tuple[torch.Tensor, torch.Tensor]` - Prepare LSTM training data
- `train(X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]` - Train LSTM model
- `predict(X: np.ndarray) -> LSTMPrediction` - Make predictions
- `save_model(model_id: str) -> bool` - Save model to database
- `load_model(model_id: str) -> bool` - Load model from database

#### TransformerPredictor

Temporal transformer with attention mechanisms for sequence modeling.

**Constructor**
```python
TransformerPredictor(config: TransformerConfig)
```

**Key Methods**

- `train(X: np.ndarray, y: np.ndarray) -> Dict[str, float]` - Train transformer
- `predict(X: np.ndarray) -> TransformerPrediction` - Make predictions
- `get_attention_weights() -> np.ndarray` - Get attention weights

#### CNNPredictor

1D CNN for pattern recognition in financial data.

**Constructor**
```python
CNNPredictor(config: CNNModelConfig)
```

**Key Methods**

- `prepare_data(data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]` - Prepare CNN data
- `train(X: np.ndarray, y: np.ndarray) -> Dict[str, float]` - Train CNN
- `predict(X: np.ndarray) -> CNNPrediction` - Make predictions

### 2. Ensemble Methods

#### EnsemblePredictor

Multi-model ensemble with intelligent weighting.

**Constructor**
```python
EnsemblePredictor(config: EnsembleConfig)
```

**Configuration**
```python
@dataclass
class EnsembleConfig:
    models: List[Dict[str, Any]] = None
    voting_strategy: str = "weighted"  # weighted, average, majority
    weights: Optional[List[float]] = None
    performance_weights: bool = True
```

**Key Methods**

- `add_model(name: str, model: Any, weight: float = 1.0)` - Add model to ensemble
- `train_ensemble(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]` - Train ensemble
- `predict(X: np.ndarray) -> EnsemblePrediction` - Ensemble prediction
- `get_model_importance() -> Dict[str, float]` - Get model importance weights
- `save_ensemble() -> bool` - Save ensemble to database

#### StackingEnsemble

Cross-validation based stacking with meta-learners.

**Constructor**
```python
StackingEnsemble(config: StackingConfig)
```

**Key Methods**

- `add_base_model(name: str, model: Any)` - Add base learner
- `fit(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]` - Train stacking ensemble
- `predict(X: np.ndarray) -> StackingResult` - Stacking prediction
- `get_cv_performance() -> Dict[str, float]` - Get cross-validation performance

### 3. Online Learning

#### OnlineLearner

Adaptive online learning with concept drift detection.

**Constructor**
```python
OnlineLearner(config: OnlineLearningConfig)
```

**Configuration**
```python
@dataclass
class OnlineLearningConfig:
    learning_rate: float = 0.01      # Base learning rate
    buffer_size: int = 100           # Sample buffer size
    update_frequency: int = 10       # Update frequency
    decay_rate: float = 0.95         # Learning rate decay
    momentum: float = 0.9            # Momentum coefficient
    use_adaptive_lr: bool = True     # Adaptive learning rate
    drift_detection: bool = True     # Concept drift detection
```

**Key Methods**

- `add_sample(features: np.ndarray, target: float)` - Add new sample
- `predict(features: np.ndarray) -> OnlineLearningResult` - Online prediction
- `get_model_state() -> Dict[str, Any]` - Get complete model state
- `reset_learning()` - Reset learning state
- `save_state() -> bool` - Save learner state

#### AdaptiveModel

Model adaptation with concept drift detection.

**Constructor**
```python
AdaptiveModel(base_model: Any, config: AdaptiveConfig)
```

**Key Methods**

- `update(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]` - Update model with drift detection
- `predict(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]` - Predict with meta info
- `get_adaptation_history() -> List[Dict[str, Any]]` - Get adaptation history
- `get_performance_summary() -> Dict[str, Any]` - Get performance summary

### 4. Reinforcement Learning

#### RLAgent

Q-Learning based trading agent.

**Constructor**
```python
RLAgent(config: RLConfig)
```

**Configuration**
```python
@dataclass
class RLConfig:
    state_dim: int = 10              # State dimension
    action_dim: int = 3              # Action dimension (BUY, SELL, HOLD)
    learning_rate: float = 0.1       # Q-learning rate
    epsilon: float = 0.1             # Exploration rate
    gamma: float = 0.95              # Discount factor
    epsilon_decay: float = 0.995     # Epsilon decay rate
```

**Key Methods**

- `select_action(state: np.ndarray) -> int` - Select action using epsilon-greedy
- `update_q_table(state: np.ndarray, action: int, reward: float, next_state: np.ndarray)` - Update Q-table
- `train_episode(env) -> Dict[str, float]` - Train one episode
- `get_q_values(state: np.ndarray) -> np.ndarray` - Get Q-values for state

#### PPOAgent

Proximal Policy Optimization agent for complex trading strategies.

**Constructor**
```python
PPOAgent(config: PPOConfig)
```

**Configuration**
```python
@dataclass
class PPOConfig:
    state_dim: int = 10              # State dimension
    action_dim: int = 3              # Action dimension
    hidden_dims: List[int] = None    # Hidden layer dimensions
    learning_rate: float = 3e-4      # Learning rate
    gamma: float = 0.99              # Discount factor
    eps_clip: float = 0.2            # PPO clipping parameter
    k_epochs: int = 4                # PPO update epochs
```

**Key Methods**

- `select_action(state: np.ndarray) -> Tuple[int, float]` - Select action with probability
- `update_policy(states, actions, rewards, old_probs)` - Update PPO policy
- `train_episode(env) -> Dict[str, float]` - Train one episode
- `save_model(filepath: str) -> bool` - Save PPO model

#### DQNAgent

Deep Q-Network with experience replay.

**Constructor**
```python
DQNAgent(config: DQNConfig)
```

**Key Methods**

- `select_action(state: np.ndarray, training: bool = True) -> int` - Select action
- `store_transition(state, action, next_state, reward, done)` - Store experience
- `train_step() -> Optional[float]` - Execute one training step
- `train_episode(env, max_steps: int = 1000) -> Dict[str, float]` - Train episode
- `evaluate(env, num_episodes: int = 10) -> Dict[str, float]` - Evaluate agent

#### TradingEnvironment

Realistic market simulation environment for RL training.

**Constructor**
```python
TradingEnvironment(data: pd.DataFrame, config: EnvironmentConfig)
```

**Configuration**
```python
@dataclass
class EnvironmentConfig:
    initial_cash: float = 100000.0   # Initial cash
    transaction_cost: float = 0.001  # Transaction cost
    max_position_size: float = 1.0   # Max position ratio
    lookback_window: int = 20        # State window size
    reward_scaling: float = 1.0      # Reward scaling factor
    risk_penalty: float = 0.1        # Risk penalty
    use_risk_management: bool = True # Enable risk management
```

**Key Methods**

- `reset() -> np.ndarray` - Reset environment
- `step(action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]` - Execute action
- `get_performance_metrics() -> Dict[str, float]` - Get performance metrics
- `render(mode: str = 'human') -> Optional[str]` - Render environment state

### 5. Storage Management

#### AIModelDatabaseManager

Complete model lifecycle management with SQLite integration.

**Usage**
```python
db_manager = get_ai_model_database_manager()
```

**Key Methods**

- `save_model_info(model_id: str, model_type: str, model_name: str, config: Dict[str, Any]) -> bool` - Save model basic info
- `save_training_history(model_id: str, epoch: int, train_loss: float, val_loss: float = None) -> bool` - Save training history
- `save_model_prediction(model_id: str, symbol: str, prediction_date: str, prediction_value: float, confidence: float = None) -> bool` - Save prediction result
- `save_model_performance(model_id: str, metric_name: str, metric_value: float) -> bool` - Save performance metric
- `save_model_parameters(model_id: str, parameters: Any) -> bool` - Save model parameters
- `get_model_predictions(model_id: str, symbol: str = None) -> List[Dict[str, Any]]` - Query predictions
- `get_model_performance(model_id: str) -> List[Dict[str, Any]]` - Query performance metrics
- `get_database_stats() -> Dict[str, Any]` - Get database statistics

## Utility Functions

Module 03 provides utility functions to simplify common operations:

### prepare_features_for_training

Prepare training features from Module 01 and 02:

```python
def prepare_features_for_training(
    symbols: List[str],
    start_date: str,
    end_date: str,
    feature_types: List[str] = ['technical']
) -> pd.DataFrame:
    """Prepare features for AI model training
    
    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)
        feature_types: Types of features to include
    
    Returns:
        DataFrame with features and target data
    """
```

### train_ensemble_model

Train ensemble model with multiple base models:

```python
def train_ensemble_model(
    models: List[Dict[str, Any]],
    features: pd.DataFrame,
    target_column: str,
    model_name: str
) -> EnsemblePredictor:
    """Train ensemble model
    
    Args:
        models: List of model dictionaries with 'name', 'model', 'weight'
        features: Feature DataFrame
        target_column: Target column name
        model_name: Ensemble model name
    
    Returns:
        Trained ensemble predictor
    """
```

### evaluate_model_performance

Evaluate model performance and save results:

```python
def evaluate_model_performance(
    model: Any,
    test_features: pd.DataFrame,
    target_column: str,
    model_id: str
) -> Dict[str, float]:
    """Evaluate model performance
    
    Args:
        model: Trained model
        test_features: Test feature data
        target_column: Target column name
        model_id: Model identifier
    
    Returns:
        Performance metrics dictionary
    """
```

### create_lstm_predictor

Create LSTM predictor with default settings:

```python
def create_lstm_predictor(
    sequence_length: int = 60,
    hidden_size: int = 50,
    num_layers: int = 2,
    learning_rate: float = 0.001
) -> LSTMModel:
    """Create LSTM predictor with optimized settings"""
```

### create_online_learner

Create online learner with default settings:

```python
def create_online_learner(
    learning_rate: float = 0.01,
    buffer_size: int = 100,
    model_name: str = "online_learner"
) -> OnlineLearner:
    """Create online learner with optimized settings"""
```

## Integration with Other Modules

### Data Flow Architecture

```
Module 01 (Data Pipeline) 
    ↓ Raw market data
Module 02 (Feature Engineering)
    ↓ Technical indicators & features
Module 03 (AI Models) [THIS MODULE]
    ↓ Predictions & trading signals
Module 04/05/08/09 (Applications)
```

### Integration Examples

#### With Module 01 - Data Collection
```python
from module_01_data_pipeline import AkshareDataCollector, get_database_manager

# Collect raw data
collector = AkshareDataCollector()
stock_data = collector.fetch_stock_history("000001", "20240101", "20241201")

# Use data for AI training
lstm_model = LSTMModel(config)
X, y = lstm_model.prepare_data(stock_data, 'close')
```

#### With Module 02 - Feature Engineering
```python
from module_02_feature_engineering import TechnicalIndicators

# Calculate technical indicators
calculator = TechnicalIndicators()
features = calculator.calculate_all_indicators(stock_data)

# Use features for ML models
ensemble = EnsemblePredictor(config)
ensemble.train_ensemble(features.values, features['returns'].values)
```

#### Providing Services to Other Modules
```python
# Module 04 - Market Analysis
from module_03_ai_models import LSTMModel, get_ai_model_database_manager

def get_market_prediction(symbol: str):
    """Get AI prediction for market analysis"""
    lstm_model = LSTMModel.load_model(f"lstm_{symbol}")
    prediction = lstm_model.predict(latest_features)
    return prediction

# Module 05 - Risk Management
from module_03_ai_models import OnlineLearner

def get_risk_assessment(portfolio_data):
    """Get real-time risk assessment"""
    risk_learner = OnlineLearner.load_state("risk_model")
    risk_score = risk_learner.predict(portfolio_features)
    return risk_score

# Module 08 - Execution
from module_03_ai_models import PPOAgent

def get_trading_decision(market_state):
    """Get trading decision from RL agent"""
    ppo_agent = PPOAgent.load_model("trading_agent")
    action, confidence = ppo_agent.select_action(market_state)
    return action, confidence
```

## Interface Types

### Programmatic API
Module 03 provides **programmatic interfaces** - direct Python function and class calls:
- AI model classes: `LSTMModel`, `TransformerPredictor`, `EnsemblePredictor`, `OnlineLearner`
- Reinforcement learning classes: `PPOAgent`, `RLAgent`, `DQNAgent`, `TradingEnvironment`
- Storage management: `AIModelDatabaseManager`
- Utility functions: `prepare_features_for_training`, `train_ensemble_model`, `evaluate_model_performance`

### Data Service Integration
Module 03 **does not provide REST API endpoints**. It serves as AI infrastructure for other modules:
- **Module 04 (Market Analysis)**: Provides real-time predictions and agent decisions
- **Module 05 (Risk Management)**: Provides risk predictions and portfolio optimization
- **Module 08 (Execution)**: Provides trading signals and strategy decisions
- **Module 09 (Backtesting)**: Provides historical predictions for strategy backtesting

## Testing and Validation

### Run Complete Test Suite
```bash
cd /Users/victor/Desktop/25fininnov/FinLoom-server
# Must be run in study conda environment
conda activate study
python tests/module03_ai_models_test.py
```

### Test Coverage
The test suite includes:
- **Data Integration**: Integration with Module 01 and 02 (✅ 100% pass)
- **LSTM Deep Learning**: Model training and prediction (✅ 100% pass)
- **Ensemble Methods**: Multi-model ensemble training (✅ 100% pass)
- **Online Learning**: Adaptive learning and concept drift (✅ 100% pass)
- **Reinforcement Learning**: RL agent training and decision making (✅ 100% pass)
- **Database Operations**: Model persistence and retrieval (✅ 100% pass)
- **End-to-End Workflow**: Complete AI pipeline testing (✅ 100% pass)

### Performance Metrics
- **Test Success Rate**: 100% (6/6 tests passing)
- **Data Processing**: Successfully handles 123+ records with 29 technical indicators
- **Model Training**: LSTM achieves convergence with proper loss reduction
- **Prediction Quality**: Ensemble models provide confidence-weighted predictions
- **Real-time Processing**: Online learner processes 50+ samples with adaptive learning
- **Database Operations**: All CRUD operations working with SQLite integration

### Integration Validation
- ✅ **Module 01 Integration**: Successfully fetches real market data via akshare
- ✅ **Module 02 Integration**: Uses calculated technical indicators for training
- ✅ **Database Separation**: Uses dedicated SQLite database for AI model data
- ✅ **No Mock Data**: All data comes from real market sources
- ✅ **Environment Compatibility**: Runs correctly in study conda environment
