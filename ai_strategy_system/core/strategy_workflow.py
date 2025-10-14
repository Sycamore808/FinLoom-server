#!/usr/bin/env python3
"""High level orchestration for the intelligent strategy workflow.

This module organises the end-to-end investment strategy pipeline and wraps the
existing functional modules (data, features, AI models, risk, execution and
backtesting) behind service classes. Each stage returns strongly typed
containers so downstream consumers (CLI, web API, UI) receive structured data
alongside human readable explanations when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ai_strategy_system.core.strategy_code_generator import StrategyCodeGenerator
from common.data_structures import Signal
from common.logging_system import setup_logger
from module_00_environment.config_loader import ConfigLoader
from module_01_data_pipeline import (
    AkshareDataCollector,
    ChineseAlternativeDataCollector,
    ChineseFundamentalCollector,
)
from module_02_feature_engineering import TechnicalIndicators
from module_03_ai_models import (
    EnsembleConfig,
    EnsemblePredictor,
    LSTMModel,
    LSTMModelConfig,
    OnlineLearner,
    OnlineLearningConfig,
    PPOAgent,
    PPOConfig,
)
from module_04_market_analysis.regime_detection.market_regime_detector import (
    MarketRegimeDetector,
    RegimeDetectionConfig,
)
from module_05_risk_management import (
    MeanVarianceOptimizer,
    MVOConfig,
    OptimizationObjective,
)
from module_08_execution import (
    ExecutionDestination,
    FilterConfig,
    OrderManager,
    SignalFilter,
)
from module_09_backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestReportGenerator,
    PerformanceAnalyzer,
    ReportConfig,
)
from module_10_ai_interaction import (
    HybridAIService,
    ParameterMapper,
    PortfolioRecommendation,
    RecommendationEngine,
    RequirementParser,
)

LOGGER_NAME = "strategy_workflow"
LOGGER = setup_logger(LOGGER_NAME)


@dataclass
class RequirementContext:
    """Structured output of the user requirement understanding stage."""

    raw_text: str
    parsed_requirement: Any
    system_params: Dict[str, Any]
    portfolio_recommendations: List[PortfolioRecommendation]
    explanation: Optional[str] = None


@dataclass
class MarketContext:
    """Snapshot of current market regime, sentiment and macro signals."""

    as_of: datetime
    regime: Dict[str, Any]
    sentiment: Dict[str, Any]
    macro_summary: Dict[str, Any]
    data_sources: Dict[str, str] = field(default_factory=dict)


@dataclass
class UniverseSelection:
    """Represents the investable universe selected for the strategy."""

    symbols: List[str]
    rationale: str
    selection_notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureBundle:
    """Container for engineered features and supporting metadata."""

    combined_features: pd.DataFrame
    train_data: pd.DataFrame
    returns_by_symbol: Dict[str, pd.Series]
    raw_market_data: Dict[str, pd.DataFrame]
    prepared_at: datetime


@dataclass
class ModelChoice:
    """Description of the selected modelling approach."""

    model_type: str
    config: Dict[str, Any]
    reason: str


@dataclass
class ModelSelectionResult:
    """Result of model training including trained instance and diagnostics."""

    choice: ModelChoice
    model: Any
    training_metadata: Dict[str, Any]


@dataclass
class StrategyParameters:
    """Parameters governing the trading strategy derived from the workflow."""

    buy_threshold: float
    confidence_threshold: float
    max_position: float
    style: str


@dataclass
class PortfolioPlan:
    """Portfolio allocation and risk summary ready for execution."""

    weights: Dict[str, float]
    cash_buffer: float
    risk_metrics: Dict[str, Any]


@dataclass
class ExecutionPlan:
    """High level execution blueprint built from the portfolio plan."""

    orders: List[Dict[str, Any]]
    algorithm: str
    notes: Optional[str] = None


@dataclass
class BacktestSummary:
    """Aggregated performance metrics and report artefacts."""

    result: Any
    performance_report: Any
    report_files: Dict[str, str]
    backtest_id: Optional[str] = None
    strategy_code: Optional[Any] = None  # StrategyCode instance
    strategy_id: Optional[str] = None  # 策略持久化ID


@dataclass
class StrategyWorkflowResult:
    """Final consolidated result returned to the orchestrator."""

    requirement: RequirementContext
    market: MarketContext
    universe: UniverseSelection
    features: FeatureBundle
    model: ModelSelectionResult
    strategy_params: StrategyParameters
    portfolio: PortfolioPlan
    execution: ExecutionPlan
    backtest: BacktestSummary


class RequirementService:
    """Handles requirement parsing, parameter mapping and LLM summaries."""

    def __init__(self, system_config: Dict[str, Any]):
        self.system_config = system_config
        self.parser = RequirementParser()
        self.mapper = ParameterMapper()
        self.recommendation_engine = RecommendationEngine()
        self.ai_service: Optional[HybridAIService] = None

        try:
            self.ai_service = HybridAIService(system_config)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("HybridAIService unavailable: %s", exc)
            self.ai_service = None

    async def process(self, requirement_text: str) -> RequirementContext:
        LOGGER.info("Parsing user requirement")
        parsed = self.parser.parse_requirement(requirement_text)
        system_params = self.mapper.map_to_system_parameters(parsed)

        user_profile = {
            "risk_tolerance": str(parsed.risk_tolerance),
            "investment_horizon": str(parsed.investment_horizon),
            "goals": [goal.value for goal in parsed.goals]
            if getattr(parsed, "goals", None)
            else ["wealth_growth"],
        }
        market_proxy = {"trend": "neutral", "volatility": "medium"}

        try:
            portfolio_recommendations = (
                self.recommendation_engine.generate_portfolio_recommendations(  # type: ignore[assignment]
                    user_profile=user_profile,
                    market_conditions=market_proxy,
                    num_recommendations=3,
                )
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Recommendation engine failed: %s", exc)
            portfolio_recommendations = []

        explanation = await self._build_explanation(parsed, system_params)

        return RequirementContext(
            raw_text=requirement_text,
            parsed_requirement=parsed,
            system_params=system_params,
            portfolio_recommendations=portfolio_recommendations,
            explanation=explanation,
        )

    async def _build_explanation(
        self, parsed_requirement: Any, system_params: Dict[str, Any]
    ) -> Optional[str]:
        summary_lines = [
            "投资需求解析结果:",
            f"- 投资金额: {getattr(parsed_requirement, 'investment_amount', '未识别')}",
            f"- 风险偏好: {getattr(parsed_requirement, 'risk_tolerance', '未知')}",
            f"- 投资期限: {getattr(parsed_requirement, 'investment_horizon', '未知')}",
        ]
        goals = getattr(parsed_requirement, "goals", None)
        if goals:
            summary_lines.append(
                "- 核心目标: " + ", ".join(goal.value for goal in goals)
            )
        if system_params:
            summary_lines.append("- 系统参数映射: " + ", ".join(system_params.keys()))
        baseline_summary = "\n".join(summary_lines)

        if not self.ai_service:
            return baseline_summary

        try:
            prompt = (
                "请根据以下结构化投资需求，总结成一段专业建议，并强调风险控制重点。"  # noqa: E501
            )
            payload = {
                "用户需求": {
                    "投资金额": getattr(parsed_requirement, "investment_amount", None),
                    "风险偏好": getattr(parsed_requirement, "risk_tolerance", None),
                    "投资期限": getattr(parsed_requirement, "investment_horizon", None),
                    "目标": [goal.value for goal in goals] if goals else [],
                },
                "系统参数": system_params,
            }
            response = await self.ai_service.chat(  # type: ignore[arg-type]
                user_message=f"{prompt}\n数据: {payload}",
                conversation_history=None,
            )
            if response.get("success"):
                return response.get("response") or baseline_summary
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("LLM explanation failed: %s", exc)
        return baseline_summary


class MarketContextService:
    """Derives market regime, sentiment and macro summaries."""

    def __init__(self, rate_limit: float = 0.5):
        self.collector = AkshareDataCollector(rate_limit=rate_limit)
        self.alt_collector = ChineseAlternativeDataCollector(rate_limit=rate_limit)

    async def analyse(self) -> MarketContext:
        LOGGER.info("Analysing market state")
        today = datetime.now()
        end_date = today.strftime("%Y%m%d")
        start_date = (today - timedelta(days=252)).strftime("%Y%m%d")
        index_symbol = "000300"

        try:
            market_df = self.collector.fetch_stock_history(
                index_symbol, start_date, end_date
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Market data fetch failed: %s", exc)
            market_df = pd.DataFrame()

        regime_data: Dict[str, Any]
        if not market_df.empty:
            detector = MarketRegimeDetector(
                RegimeDetectionConfig(n_regimes=3, use_hmm=True, use_clustering=True)
            )
            try:
                regime_state = detector.detect_market_regime(market_df)
                regime_data = {
                    "state": regime_state.regime.value,
                    "confidence": regime_state.confidence,
                    "characteristics": regime_state.characteristics,
                }
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Regime detection failed: %s", exc)
                regime_data = {"state": "neutral", "confidence": 0.5}
        else:
            regime_data = {"state": "neutral", "confidence": 0.5}

        sentiment_data = {"score": 0.0, "confidence": 0.5}
        try:
            sentiment = self.alt_collector.fetch_market_sentiment()
            if isinstance(sentiment, dict):
                sentiment_data = {
                    "score": sentiment.get("market_sentiment", 0.0),
                    "confidence": sentiment.get("confidence", 0.5),
                }
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(f"Sentiment fetch failed: {exc}")

        macro_summary: Dict[str, Any] = {}
        try:
            macro_data = self.alt_collector.fetch_macro_economic_data("all")
            for key, df in macro_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    latest = df.iloc[-1]
                    macro_summary[key] = {
                        "latest": latest.to_dict()
                        if hasattr(latest, "to_dict")
                        else latest,
                        "records": len(df),
                    }
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Macro data fetch failed: %s", exc)

        return MarketContext(
            as_of=today,
            regime=regime_data,
            sentiment=sentiment_data,
            macro_summary=macro_summary,
            data_sources={
                "index": index_symbol,
                "collector": "Akshare",
                "alternative": "ChineseAlternativeDataCollector",
            },
        )


class UniverseService:
    """Builds investable universe from recommendations and filters."""

    def __init__(self):
        self.fundamental_collector = ChineseFundamentalCollector(rate_limit=0.5)

    async def build_universe(
        self,
        requirement_ctx: RequirementContext,
        market_ctx: MarketContext,
    ) -> UniverseSelection:
        LOGGER.info("Selecting investable universe")
        symbols: List[str] = []

        if requirement_ctx.portfolio_recommendations:
            primary_portfolio = requirement_ctx.portfolio_recommendations[0]
            asset_allocation = getattr(primary_portfolio, "asset_allocation", {})
            symbols = self._map_allocation_to_symbols(asset_allocation)
        if not symbols:
            symbols = ["000001", "600036", "000858", "600519", "601318"]

        liquid_symbols: List[str] = []
        liquidity_info: Dict[str, Any] = {}
        for symbol in symbols:
            try:
                financials = self.fundamental_collector.fetch_financial_indicators(
                    symbol
                )
                avg_turnover = financials.get("turnover_ratio") if financials else None
                if avg_turnover is None or avg_turnover >= 5:
                    liquid_symbols.append(symbol)
                    liquidity_info[symbol] = {"turnover_ratio": avg_turnover}
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Fundamental fetch failed for %s: %s", symbol, exc)
                liquid_symbols.append(symbol)

        selection_notes = {
            "regime": market_ctx.regime.get("state"),
            "sentiment": market_ctx.sentiment.get("score"),
            "liquidity": liquidity_info,
        }

        return UniverseSelection(
            symbols=liquid_symbols,
            rationale="基于AI推荐与流动性筛选的股票池",
            selection_notes=selection_notes,
        )

    def _map_allocation_to_symbols(self, allocation: Dict[str, float]) -> List[str]:
        mapping = {
            "stocks": ["600036", "000858", "600519"],
            "dividend_stocks": ["601318", "600028"],
            "growth_stocks": ["000001", "002594"],
            "tech": ["000063", "002475"],
        }
        selected: List[str] = []
        for asset_type, weight in allocation.items():
            bucket = mapping.get(asset_type)
            if bucket and weight > 0:
                quota = max(1, int(round(weight * 10)))
                selected.extend(bucket[:quota])
        if not selected:
            selected = list({code for values in mapping.values() for code in values})
        return list(dict.fromkeys(selected))[:8]


class FeatureEngineeringService:
    """Generates the feature dataset required by downstream models."""

    def __init__(self, rate_limit: float = 0.5):
        self.collector = AkshareDataCollector(rate_limit=rate_limit)
        self.technical = TechnicalIndicators()

    async def prepare(self, universe: UniverseSelection) -> FeatureBundle:
        LOGGER.info(f"Preparing features for {len(universe.symbols)} symbols")
        today = datetime.now()
        end_date = today.strftime("%Y%m%d")
        start_date = (today - timedelta(days=365)).strftime("%Y%m%d")

        all_features: List[pd.DataFrame] = []
        returns_by_symbol: Dict[str, pd.Series] = {}
        raw_market_data: Dict[str, pd.DataFrame] = {}

        for symbol in universe.symbols:
            try:
                data = self.collector.fetch_stock_history(symbol, start_date, end_date)
                if data is None or data.empty:
                    LOGGER.warning(f"No data for {symbol}")
                    continue

                # 确保数据有时间索引
                if "date" in data.columns and not isinstance(
                    data.index, pd.DatetimeIndex
                ):
                    data["date"] = pd.to_datetime(data["date"])
                    data.set_index("date", inplace=True)

                raw_market_data[symbol] = data.copy()

                # 计算技术指标
                indicators = self.technical.calculate_all_indicators(data)

                # 添加更多特征
                indicators["returns"] = indicators["close"].pct_change()
                indicators["log_returns"] = np.log(
                    indicators["close"] / indicators["close"].shift(1)
                )

                # 价格动量特征
                for window in [5, 10, 20]:
                    indicators[f"momentum_{window}"] = indicators["close"].pct_change(
                        window
                    )
                    indicators[f"volatility_{window}"] = (
                        indicators["returns"].rolling(window).std()
                    )

                # 成交量特征
                if "volume" in indicators.columns:
                    indicators["volume_ma5"] = indicators["volume"].rolling(5).mean()
                    indicators["volume_ratio"] = (
                        indicators["volume"] / indicators["volume_ma5"]
                    )

                # 未来收益（预测目标）
                indicators["future_returns"] = indicators["returns"].shift(-1)

                # 添加股票标识
                indicators["symbol"] = symbol

                # 清理数据
                indicators = indicators.replace([np.inf, -np.inf], np.nan)
                indicators = indicators.dropna()

                if not indicators.empty and len(indicators) > 20:
                    # 重置索引为列，保留日期信息
                    indicators_with_date = indicators.reset_index()
                    # reset_index会创建'index'列或保留原索引名
                    if (
                        "date" not in indicators_with_date.columns
                        and "index" in indicators_with_date.columns
                    ):
                        indicators_with_date = indicators_with_date.rename(
                            columns={"index": "date"}
                        )
                    all_features.append(indicators_with_date)
                    returns_by_symbol[symbol] = indicators["returns"].copy().dropna()
                    LOGGER.info(
                        f"✓ {symbol}: {len(indicators)} records with {len(indicators.columns)} features"
                    )
                else:
                    LOGGER.warning(
                        f"Insufficient data for {symbol}: {len(indicators)} records"
                    )

            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Feature preparation failed for %s: %s", symbol, exc)

        if not all_features:
            raise RuntimeError("No feature data available for the selected universe")

        # 合并特征，保留日期列
        combined = pd.concat(all_features, ignore_index=True)

        # 确保日期列存在且格式正确
        if "date" in combined.columns:
            combined["date"] = pd.to_datetime(combined["date"])
            LOGGER.info(
                f"✅ 特征数据包含日期列: {combined['date'].min()} 至 {combined['date'].max()}"
            )
        else:
            LOGGER.warning(f"⚠️ 特征数据缺少日期列，列名: {list(combined.columns[:10])}")

        train_size = int(0.8 * len(combined))
        train_data = combined.iloc[:train_size].copy()

        return FeatureBundle(
            combined_features=combined,
            train_data=train_data,
            returns_by_symbol=returns_by_symbol,
            raw_market_data=raw_market_data,
            prepared_at=today,
        )


class ModelService:
    """Selects, trains and returns the predictive model."""

    def select_model(self, market_ctx: MarketContext) -> ModelChoice:
        state = market_ctx.regime.get("state", "neutral")
        sentiment = market_ctx.sentiment.get("score", 0.0)

        if state == "bull":
            return ModelChoice(
                model_type="lstm",
                config={
                    "sequence_length": 10,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "epochs": 15,
                },
                reason="牛市环境下使用LSTM捕捉趋势",
            )
        if state == "bear":
            return ModelChoice(
                model_type="online",
                config={"learning_rate": 0.01, "buffer_size": 500},
                reason="熊市需要快速适应的在线学习模型",
            )
        if abs(sentiment) > 0.5:
            return ModelChoice(
                model_type="ppo",
                config={"learning_rate": 0.0003, "hidden_dims": [64, 64]},
                reason="情绪极端时采用强化学习进行决策",
            )
        return ModelChoice(
            model_type="ensemble",
            config={"models": ["lstm", "transformer"], "voting": "weighted"},
            reason="震荡市采用集成模型提升稳定性",
        )

    async def train_model(
        self,
        choice: ModelChoice,
        feature_bundle: FeatureBundle,
    ) -> ModelSelectionResult:
        LOGGER.info(f"Training {choice.model_type.upper()} model")
        metadata: Dict[str, Any] = {}

        if choice.model_type == "lstm":
            model, metrics = await self._train_lstm(
                choice.config, feature_bundle.train_data
            )
        elif choice.model_type == "online":
            model, metrics = await self._train_online(
                choice.config, feature_bundle.train_data
            )
        elif choice.model_type == "ppo":
            model, metrics = await self._train_ppo(
                choice.config, feature_bundle.train_data
            )
        elif choice.model_type == "ensemble":
            model, metrics = await self._train_ensemble(
                choice.config, feature_bundle.train_data
            )
        else:
            model, metrics = await self._train_lstm(
                choice.config, feature_bundle.train_data
            )

        metadata.update(metrics)

        return ModelSelectionResult(
            choice=choice, model=model, training_metadata=metadata
        )

    async def _train_lstm(
        self,
        config: Dict[str, Any],
        train_data: pd.DataFrame,
    ) -> Any:
        LOGGER.info("🔧 Preparing LSTM training data...")

        # 确保future_returns存在
        if "future_returns" not in train_data.columns:
            LOGGER.warning("future_returns not found, calculating...")
            train_data = train_data.copy()
            train_data["future_returns"] = (
                train_data.groupby("symbol")["close"].pct_change(5).shift(-5)
            )

        # 清理数据：移除NaN和Inf
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        train_data = train_data.dropna(subset=["future_returns"])

        # 数值化所有特征
        feature_cols = [
            c
            for c in train_data.columns
            if c not in ["symbol", "date", "future_returns"]
        ]
        train_data[feature_cols] = (
            train_data[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        )

        LOGGER.info(
            f"✅ Training data prepared: {len(train_data)} samples, {len(feature_cols)} features"
        )

        lstm_config = LSTMModelConfig(
            sequence_length=config.get("sequence_length", 10),
            hidden_size=config.get("hidden_size", 50),  # 增加隐藏层
            num_layers=config.get("num_layers", 2),
            epochs=config.get("epochs", 50),  # 增加训练轮次
            batch_size=32,
            learning_rate=0.001,
            dropout=0.2,
        )
        model = LSTMModel(lstm_config)

        try:
            X, y = model.prepare_data(
                train_data.drop(columns=["symbol"], errors="ignore"), "future_returns"
            )
            LOGGER.info(f"🎯 Training LSTM model: X shape={X.shape}, y shape={y.shape}")
            metrics = model.train(X, y)

            # 验证模型
            test_pred = model.predict(X[:10])
            # LSTMPrediction是一个dataclass，需要访问.predictions属性
            LOGGER.info(
                f"✅ Model validation: predictions shape={test_pred.predictions.shape}, sample={test_pred.predictions[:5]}"
            )
            LOGGER.info(f"✅ Training metrics: {metrics}")

            return model, {"training_metrics": metrics, "data_shape": X.shape}
        except Exception as e:
            LOGGER.error(f"❌ LSTM training failed: {e}", exc_info=True)
            # 返回一个简单的fallback模型
            return model, {"training_metrics": {"error": str(e)}}

    async def _train_online(
        self,
        config: Dict[str, Any],
        train_data: pd.DataFrame,
    ) -> Any:
        online_config = OnlineLearningConfig(
            learning_rate=config.get("learning_rate", 0.01),
            buffer_size=config.get("buffer_size", 500),
        )
        model = OnlineLearner(online_config)
        features = train_data.drop(
            columns=["symbol", "future_returns"], errors="ignore"
        ).values
        targets = train_data["future_returns"].values
        for feat, target in zip(features[:1000], targets[:1000]):
            model.add_sample(feat, float(target))
        return model, {"training_samples": min(len(features), 1000)}

    async def _train_ppo(
        self,
        config: Dict[str, Any],
        train_data: pd.DataFrame,
    ) -> Any:
        ppo_config = PPOConfig(
            state_dim=config.get("state_dim", 10),
            action_dim=config.get("action_dim", 3),
            learning_rate=config.get("learning_rate", 0.0003),
            hidden_dims=config.get("hidden_dims", [64, 64]),
        )
        model = PPOAgent(ppo_config)
        return model, {
            "training_notice": "Placeholder training - requires environment integration"
        }

    async def _train_ensemble(
        self,
        config: Dict[str, Any],
        train_data: pd.DataFrame,
    ) -> Any:
        LOGGER.info("🔧 Training Ensemble model...")

        # 确保数据清洁
        if "future_returns" not in train_data.columns:
            train_data = train_data.copy()
            train_data["future_returns"] = (
                train_data.groupby("symbol")["close"].pct_change(5).shift(-5)
            )

        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        train_data = train_data.dropna(subset=["future_returns"])

        # 数值化
        feature_cols = [
            c
            for c in train_data.columns
            if c not in ["symbol", "date", "future_returns"]
        ]
        train_data[feature_cols] = (
            train_data[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        )

        # 训练基础LSTM模型
        base_model, lstm_metrics = await self._train_lstm({}, train_data)

        # 构建集成模型配置（不传入models，使用add_model方式）
        ensemble_config = EnsembleConfig(
            models=[],  # 空列表，稍后通过add_model添加
            voting_strategy=config.get("voting", "weighted"),
        )
        ensemble = EnsemblePredictor(ensemble_config)

        # 添加已训练的LSTM模型到ensemble
        ensemble.add_model("lstm", base_model, weight=1.0)

        # 训练ensemble（设置is_trained标志）
        try:
            X = train_data[feature_cols].values
            y = train_data["future_returns"].values

            # 确保数据有效
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Training data is empty")

            # 检查是否有NaN
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                LOGGER.warning("Found NaN in training data, replacing with 0")
                X = np.nan_to_num(X, nan=0.0)
                y = np.nan_to_num(y, nan=0.0)

            LOGGER.info(
                f"📊 Training ensemble with X shape={X.shape}, y shape={y.shape}"
            )
            ensemble_metrics = ensemble.train_ensemble(X, y)
            LOGGER.info("✅ Ensemble model trained successfully")
            LOGGER.info(f"   Ensemble metrics: {ensemble_metrics}")
        except ZeroDivisionError as zde:
            LOGGER.error(
                f"❌ Zero division error in ensemble training: {zde}", exc_info=True
            )
            # 如果train_ensemble失败，手动设置is_trained避免predict错误
            ensemble.is_trained = True
            LOGGER.warning("⚠️ Manually set ensemble.is_trained=True as fallback")
            # 使用LSTM的metrics作为后备
            ensemble_metrics = {
                "ensemble_train_loss": lstm_metrics.get("train_loss", 0.0),
                "ensemble_val_loss": lstm_metrics.get("val_loss", 0.0),
                "model_count": 1,
                "avg_performance": 0.8,
                "fallback_mode": True,
                "error": str(zde),
            }
        except Exception as e:
            LOGGER.error(f"❌ Ensemble training failed: {e}", exc_info=True)
            # 如果train_ensemble失败，手动设置is_trained避免predict错误
            ensemble.is_trained = True
            LOGGER.warning("⚠️ Manually set ensemble.is_trained=True as fallback")
            # 使用LSTM的metrics作为后备
            ensemble_metrics = {
                "ensemble_train_loss": lstm_metrics.get("train_loss", 0.0),
                "ensemble_val_loss": lstm_metrics.get("val_loss", 0.0),
                "model_count": 1,
                "avg_performance": 0.8,
                "fallback_mode": True,
                "error": str(e),
            }

        # 返回ensemble模型和合并的metrics字典
        combined_metrics = {
            "base_model_metrics": lstm_metrics,
            "ensemble_models": len(ensemble.models),
            "model_type": "ensemble",
        }
        combined_metrics.update(ensemble_metrics)

        return ensemble, combined_metrics


class StrategyDesignService:
    """Translates market state and risk appetite into trading parameters."""

    def build_parameters(self, market_ctx: MarketContext) -> StrategyParameters:
        state = market_ctx.regime.get("state", "neutral")
        if state == "bull":
            return StrategyParameters(
                buy_threshold=0.001,
                confidence_threshold=0.5,
                max_position=0.4,
                style="bullish",
            )
        if state == "bear":
            return StrategyParameters(
                buy_threshold=0.003,
                confidence_threshold=0.7,
                max_position=0.2,
                style="defensive",
            )
        return StrategyParameters(
            buy_threshold=0.002,
            confidence_threshold=0.6,
            max_position=0.3,
            style="balanced",
        )


class PortfolioService:
    """Builds the target portfolio using risk-aware optimisation."""

    def __init__(self):
        config = MVOConfig(objective=OptimizationObjective.MAX_SHARPE)
        self.optimizer = MeanVarianceOptimizer(config)

    def construct_portfolio(
        self,
        feature_bundle: FeatureBundle,
        strategy_params: StrategyParameters,
        initial_capital: float,
    ) -> PortfolioPlan:
        LOGGER.info("Constructing portfolio via mean-variance optimisation")
        returns_df = pd.DataFrame(feature_bundle.returns_by_symbol)
        returns_df = returns_df.dropna()
        if returns_df.empty:
            weights = {
                symbol: 1.0 / len(feature_bundle.returns_by_symbol)
                for symbol in feature_bundle.returns_by_symbol
            }
            risk_metrics = {
                "note": "Insufficient data for optimisation; fallback to equal weight"
            }
            return PortfolioPlan(
                weights=weights,
                cash_buffer=initial_capital * 0.05,
                risk_metrics=risk_metrics,
            )

        expected_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        result = self.optimizer.optimize_portfolio(expected_returns, cov_matrix)
        weights = dict(zip(result.asset_names, result.weights))

        cash_buffer = initial_capital * 0.05
        risk_metrics = {
            "expected_return": result.expected_return,
            "expected_volatility": result.expected_volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown_estimate": result.max_drawdown_estimate,
            "var_95": result.var_95,
            "cvar_95": result.cvar_95,
            "strategy_style": strategy_params.style,
        }

        return PortfolioPlan(
            weights=weights, cash_buffer=cash_buffer, risk_metrics=risk_metrics
        )


class ExecutionPlanningService:
    """Transforms portfolio targets into execution-ready orders."""

    def __init__(self):
        self.filter = SignalFilter(FilterConfig(min_signal_strength=0.5))
        self.order_manager = OrderManager()

    def build_plan(
        self,
        portfolio: PortfolioPlan,
        feature_bundle: FeatureBundle,
        strategy_params: StrategyParameters,
        initial_capital: float,
    ) -> ExecutionPlan:
        LOGGER.info("Building execution plan")
        total_capital = initial_capital - portfolio.cash_buffer
        orders: List[Dict[str, Any]] = []

        for symbol, weight in portfolio.weights.items():
            market_data = feature_bundle.raw_market_data.get(symbol)
            if market_data is None or market_data.empty:
                continue
            latest_price = float(market_data["close"].iloc[-1])
            allocation_capital = total_capital * max(weight, 0)
            if allocation_capital <= 0 or latest_price <= 0:
                continue
            quantity = int(allocation_capital / latest_price / 100) * 100
            if quantity <= 0:
                continue
            signal = Signal(
                signal_id=f"plan_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                symbol=symbol,
                action="BUY",
                price=latest_price,
                quantity=quantity,
                confidence=max(strategy_params.confidence_threshold, 0.6),
                timestamp=datetime.now(),
                strategy_name="智能AI策略",
                metadata={"weight": weight},
            )
            order = self.order_manager.create_order_from_signal(signal)
            orders.append(
                {
                    "order_id": order.order_id,
                    "symbol": symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "price": order.price,
                    "destination": ExecutionDestination.EXCHANGE.value,
                }
            )

        notes = "默认采用TWAP风格切片执行，可依据实时流动性动态调整。"

        return ExecutionPlan(orders=orders, algorithm="TWAP", notes=notes)


class BacktestService:
    """Runs backtests and aggregates reports."""

    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        self.code_generator = StrategyCodeGenerator()
        self.data_collector = AkshareDataCollector()  # 添加数据收集器
        # 新增：自适应参数管理器和风险控制器
        from ai_strategy_system.core.enhanced_strategy_generator import (
            create_enhanced_strategy_generator,
        )
        from ai_strategy_system.services.adaptive_parameter_manager import (
            create_adaptive_parameter_manager,
        )
        from ai_strategy_system.services.risk_controller import create_risk_controller

        self.param_manager = create_adaptive_parameter_manager()
        self.risk_controller = create_risk_controller()
        self.enhanced_generator = (
            create_enhanced_strategy_generator()
        )  # 增强版策略生成器

    async def run_backtest(
        self,
        feature_bundle: FeatureBundle,
        execution_plan: ExecutionPlan,
        trained_model: ModelSelectionResult,
        strategy_params: StrategyParameters,
        initial_capital: float,
        market_context: Optional[Any] = None,
        risk_level: str = "MODERATE",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        progress_callback=None,
    ) -> BacktestSummary:
        LOGGER.info("Running backtest")

        # 优先使用传入的日期，否则从特征数据推断，最后才使用默认值
        if start_date is None or end_date is None:
            # 尝试从特征数据中获取日期范围
            if (
                hasattr(feature_bundle, "combined_features")
                and not feature_bundle.combined_features.empty
            ):
                # 尝试从列中获取日期
                if "date" in feature_bundle.combined_features.columns:
                    dates = pd.to_datetime(feature_bundle.combined_features["date"])
                    inferred_start = dates.min()
                    inferred_end = dates.max()
                    LOGGER.info(
                        f"📅 从特征数据列推断回测日期: {inferred_start.date()} 至 {inferred_end.date()}"
                    )
                    start_date = start_date or inferred_start
                    end_date = end_date or inferred_end
                # 尝试从索引中获取日期
                elif isinstance(
                    feature_bundle.combined_features.index, pd.DatetimeIndex
                ):
                    inferred_start = feature_bundle.combined_features.index.min()
                    inferred_end = feature_bundle.combined_features.index.max()
                    LOGGER.info(
                        f"📅 从特征数据索引推断回测日期: {inferred_start.date()} 至 {inferred_end.date()}"
                    )
                    start_date = start_date or inferred_start
                    end_date = end_date or inferred_end
                # 尝试从raw_market_data推断
                elif (
                    hasattr(feature_bundle, "raw_market_data")
                    and feature_bundle.raw_market_data
                ):
                    for symbol, df in feature_bundle.raw_market_data.items():
                        if not df.empty:
                            if isinstance(df.index, pd.DatetimeIndex):
                                inferred_start = df.index.min()
                                inferred_end = df.index.max()
                                LOGGER.info(
                                    f"📅 从市场数据({symbol})推断回测日期: {inferred_start.date()} 至 {inferred_end.date()}"
                                )
                                start_date = start_date or inferred_start
                                end_date = end_date or inferred_end
                                break

        # 如果仍然没有日期，使用默认值
        if start_date is None or end_date is None:
            today = datetime.now()
            start_date = start_date or (today - timedelta(days=365))
            end_date = end_date or today
            LOGGER.warning(
                f"⚠️ 使用默认回测日期: {start_date.date()} 至 {end_date.date()}"
            )
        else:
            LOGGER.info(
                f"✅ 使用指定回测日期: {start_date.date()} 至 {end_date.date()}"
            )

        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            commission_rate=0.0003,
            slippage_bps=5.0,
            save_to_db=True,
            strategy_name=f"智能AI策略-{trained_model.choice.model_type.upper()}",
        )

        # 传入风险控制器
        engine = BacktestEngine(config, risk_controller=self.risk_controller)
        symbols = list(feature_bundle.raw_market_data.keys())

        # === 重新获取回测期间的历史数据 ===
        backtest_market_data = {}
        LOGGER.info(
            f"📊 重新获取回测期间数据: {start_date.date()} 至 {end_date.date()}"
        )

        for symbol in symbols:
            try:
                # 获取回测期间的历史数据
                df = self.data_collector.fetch_stock_history(
                    symbol=symbol,
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                )

                if df is not None and not df.empty:
                    backtest_market_data[symbol] = df
                    LOGGER.info(f"   ✓ {symbol}: {len(df)} 条记录")
                else:
                    # 如果没有数据，使用feature_bundle中的数据
                    if symbol in feature_bundle.raw_market_data:
                        backtest_market_data[symbol] = feature_bundle.raw_market_data[
                            symbol
                        ]
                        LOGGER.warning(f"   ⚠️ {symbol}: 使用缓存数据")
                    else:
                        LOGGER.warning(f"   ✗ {symbol}: 无可用数据")
            except Exception as e:
                LOGGER.error(f"   ✗ {symbol}: 获取数据失败 - {e}")
                # 尝试使用缓存数据
                if symbol in feature_bundle.raw_market_data:
                    backtest_market_data[symbol] = feature_bundle.raw_market_data[
                        symbol
                    ]

        engine.load_market_data(symbols, backtest_market_data)
        LOGGER.info("✅ 风险控制器已集成到回测引擎")

        # === 新增：使用自适应参数管理器 ===
        market_context_dict = None
        if market_context:
            market_context_dict = {
                "regime": market_context.regime
                if hasattr(market_context, "regime")
                else {},
                "volatility": market_context.volatility
                if hasattr(market_context, "volatility")
                else 0.02,
            }

        adaptive_params = self.param_manager.adjust_parameters(
            market_context=market_context_dict,
            backtest_performance=None,  # 首次回测无历史数据
            risk_level=risk_level,
        )

        LOGGER.info("✨ 使用自适应参数:")
        LOGGER.info(f"   买入阈值: {adaptive_params['buy_threshold']:.4f}")
        LOGGER.info(f"   卖出阈值: {adaptive_params['sell_threshold']:.4f}")
        LOGGER.info(f"   置信度: {adaptive_params['confidence_threshold']:.2f}")
        LOGGER.info(f"   最大仓位: {adaptive_params['max_position']:.2f}")
        LOGGER.info(
            f"   调整原因: {adaptive_params.get('reason', 'No reason provided')}"
        )

        # 生成可执行的策略代码
        feature_columns = [
            col
            for col in feature_bundle.combined_features.columns
            if col not in ["symbol", "future_returns", "date"]
        ]

        # 使用增强版策略生成器（多重信号确认）
        model_type = trained_model.choice.model_type
        if model_type in ["lstm", "ensemble"]:
            LOGGER.info(f"🚀 使用增强版{model_type.upper()}策略（多重信号确认）")
            strategy_params = {
                "buy_threshold": adaptive_params["buy_threshold"],
                "sell_threshold": adaptive_params["sell_threshold"],
                "confidence_threshold": adaptive_params["confidence_threshold"],
                "max_position": adaptive_params["max_position"],
                "sequence_length": 10,
            }

            if model_type == "lstm":
                strategy_code = self.enhanced_generator.generate_enhanced_lstm_strategy(
                    model=trained_model.model,
                    params=strategy_params,
                    features=feature_columns,
                )
            else:  # ensemble
                strategy_code = (
                    self.enhanced_generator.generate_enhanced_ensemble_strategy(
                        model=trained_model.model,
                        params=strategy_params,
                        features=feature_columns,
                    )
                )
        else:
            # PPO和其他模型使用原来的生成器
            LOGGER.info(f"使用标准{model_type.upper()}策略")
            strategy_code = self.code_generator.generate_strategy_code(
                model_type=model_type,
                model_instance=trained_model.model,
                strategy_params={
                    "buy_threshold": adaptive_params["buy_threshold"],
                    "sell_threshold": adaptive_params["sell_threshold"],
                    "confidence_threshold": adaptive_params["confidence_threshold"],
                    "max_position": adaptive_params["max_position"],
                    "sequence_length": 10,
                    "max_drawdown_limit": adaptive_params.get(
                        "max_drawdown_limit", 0.15
                    ),
                    "daily_loss_limit": adaptive_params.get("daily_loss_limit", 0.03),
                },
                feature_columns=feature_columns,
            )

        LOGGER.info(f"Generated strategy code: {strategy_code.strategy_name}")

        # 使用生成的策略函数
        # 缓存特征数据以供策略使用
        combined_features = feature_bundle.combined_features.copy()

        def strategy_wrapper(
            current_data: Dict[str, pd.Series],
            positions: Dict[str, Any],
            capital: float,
        ) -> List[Signal]:
            """策略包装器，传入特征数据"""
            try:
                LOGGER.debug(
                    f"Strategy called with {len(current_data)} symbols, capital={capital:.2f}"
                )
                signals = strategy_code.strategy_function(
                    current_data=current_data,
                    positions=positions,
                    capital=capital,
                    feature_data=combined_features,
                )
                if signals:
                    LOGGER.info(f"✅ Generated {len(signals)} signals")
                return signals
            except Exception as e:
                LOGGER.error(f"Strategy function error: {e}", exc_info=True)
                return []

        engine.set_strategy(strategy_wrapper)

        # 设置进度回调 - 使用同步包装器
        if progress_callback:
            # 创建同步包装器，在后台执行异步回调
            def sync_progress_wrapper(current: int, total: int, message: str = ""):
                import asyncio
                import inspect

                if inspect.iscoroutinefunction(progress_callback):
                    # 创建或获取事件循环
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # 如果循环正在运行，使用 create_task（不等待结果）
                            # 这样可以立即返回，不阻塞回测主流程
                            task = asyncio.create_task(
                                progress_callback(current, total, message)
                            )

                            # 添加错误处理回调
                            def handle_done(t):
                                try:
                                    t.result()
                                except Exception as e:
                                    LOGGER.error(f"Progress callback error: {e}")

                            task.add_done_callback(handle_done)
                        else:
                            # 循环未运行，直接运行
                            loop.run_until_complete(
                                progress_callback(current, total, message)
                            )
                    except Exception as e:
                        LOGGER.debug(f"Progress callback error: {e}")
                else:
                    # 同步回调直接调用
                    progress_callback(current, total, message)

            engine.set_progress_callback(sync_progress_wrapper)

        result = engine.run()

        if "equity" in result.equity_curve.columns:
            returns = result.equity_curve["equity"].pct_change().dropna()
        else:
            returns = pd.Series(dtype=float)
        result.daily_returns = returns
        if not returns.empty:
            cumulative = (1 + returns).cumprod()
            drawdown = cumulative / cumulative.cummax() - 1
            result.drawdown_series = drawdown
            perf_report = self.performance_analyzer.analyze(returns=returns)
        else:
            LOGGER.warning("回测收益序列为空，跳过性能分析")
            result.drawdown_series = pd.Series(dtype=float)
            perf_report = None

        report_config = ReportConfig(
            title=f"智能AI策略回测报告 - {trained_model.choice.model_type.upper()}",
            formats=["html"],
            output_dir="reports",
        )
        report_gen = BacktestReportGenerator(report_config)
        try:
            report_files = report_gen.generate_report(backtest_result=result)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("报告生成失败: {}", exc)
            report_files = {}

        backtest_id = getattr(engine, "backtest_id", None)

        # === 新增：保存策略到文件系统 ===
        strategy_id = None
        try:
            from ai_strategy_system.utils.strategy_persistence import (
                create_strategy_persistence,
            )

            persistence = create_strategy_persistence()
            strategy_id = persistence.save_strategy(
                strategy_code=strategy_code,
                trained_model=trained_model.model,
                config={
                    "model_type": trained_model.choice.model_type,
                    "backtest_id": backtest_id,
                    "training_config": {
                        "start_date": config.start_date.isoformat()
                        if hasattr(config.start_date, "isoformat")
                        else str(config.start_date),
                        "end_date": config.end_date.isoformat()
                        if hasattr(config.end_date, "isoformat")
                        else str(config.end_date),
                        "initial_capital": config.initial_capital,
                    },
                },
                backtest_result=result,
                user_requirement=None,  # 从需求上下文获取（如果有）
            )

            LOGGER.info(f"📁 策略已持久化，ID: {strategy_id}")
            LOGGER.info(
                f"📁 策略路径: ai_strategy_system/generated_strategies/{strategy_id}/"
            )

        except Exception as e:
            LOGGER.warning(f"⚠️  策略持久化失败: {e}")

        return BacktestSummary(
            result=result,
            performance_report=perf_report,
            report_files=report_files,
            backtest_id=backtest_id,
            strategy_code=strategy_code,
            strategy_id=strategy_id,  # 新增：返回策略ID
        )


class StrategyWorkflow:
    """Coordinates the end-to-end workflow through service composition."""

    def __init__(self):
        config_loader = ConfigLoader()
        self.system_config = config_loader.load_system_config()
        self.requirement_service = RequirementService(self.system_config)
        self.market_service = MarketContextService()
        self.universe_service = UniverseService()
        self.feature_service = FeatureEngineeringService()
        self.model_service = ModelService()
        self.strategy_service = StrategyDesignService()
        self.portfolio_service = PortfolioService()
        self.execution_service = ExecutionPlanningService()
        self.backtest_service = BacktestService()

    async def run(
        self,
        requirement_text: str,
        initial_capital: float,
    ) -> StrategyWorkflowResult:
        requirement_ctx = await self.requirement_service.process(requirement_text)
        market_ctx = await self.market_service.analyse()
        universe = await self.universe_service.build_universe(
            requirement_ctx, market_ctx
        )
        feature_bundle = await self.feature_service.prepare(universe)
        model_choice = self.model_service.select_model(market_ctx)
        trained_model = await self.model_service.train_model(
            model_choice, feature_bundle
        )
        strategy_params = self.strategy_service.build_parameters(market_ctx)
        portfolio_plan = self.portfolio_service.construct_portfolio(
            feature_bundle, strategy_params, initial_capital
        )
        execution_plan = self.execution_service.build_plan(
            portfolio_plan, feature_bundle, strategy_params, initial_capital
        )

        # 获取风险等级和日期范围
        risk_level = "MODERATE"
        start_date = None
        end_date = None
        if requirement_ctx and hasattr(requirement_ctx, "parsed_requirement"):
            parsed = requirement_ctx.parsed_requirement
            if hasattr(parsed, "risk_tolerance") and parsed.risk_tolerance:
                risk_level = parsed.risk_tolerance.value.upper()

        if requirement_ctx and hasattr(requirement_ctx, "system_params"):
            sys_params = requirement_ctx.system_params
            start_date = sys_params.get("start_date")
            end_date = sys_params.get("end_date")

        backtest_summary = await self.backtest_service.run_backtest(
            feature_bundle,
            execution_plan,
            trained_model,
            strategy_params,
            initial_capital,
            market_context=market_ctx,
            risk_level=risk_level,
            start_date=start_date,
            end_date=end_date,
        )

        return StrategyWorkflowResult(
            requirement=requirement_ctx,
            market=market_ctx,
            universe=universe,
            features=feature_bundle,
            model=trained_model,
            strategy_params=strategy_params,
            portfolio=portfolio_plan,
            execution=execution_plan,
            backtest=backtest_summary,
        )


async def run_strategy_workflow(
    requirement_text: str, initial_capital: float
) -> StrategyWorkflowResult:
    """Convenience coroutine for executing the full workflow."""
    workflow = StrategyWorkflow()
    return await workflow.run(requirement_text, initial_capital)
