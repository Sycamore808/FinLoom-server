"""
投资组合权重优化器模块
整合多种优化方法计算最优投资组合权重
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize

from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("portfolio_weight_optimizer")


class OptimizationMethod(Enum):
    """优化方法枚举"""

    MEAN_VARIANCE = "mean_variance"  # 均值-方差优化
    MIN_VARIANCE = "min_variance"  # 最小方差
    MAX_SHARPE = "max_sharpe"  # 最大夏普比率
    RISK_PARITY = "risk_parity"  # 风险平价
    BLACK_LITTERMAN = "black_litterman"  # Black-Litterman
    EQUAL_WEIGHT = "equal_weight"  # 等权重
    INVERSE_VOLATILITY = "inverse_volatility"  # 逆波动率
    MAX_DIVERSIFICATION = "max_diversification"  # 最大分散化
    HIERARCHICAL_RISK_PARITY = "hrp"  # 层次风险平价
    ENSEMBLE = "ensemble"  # 集成多种方法


class OptimizationObjective(Enum):
    """优化目标枚举"""

    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MAXIMIZE_SORTINO = "maximize_sortino"
    MINIMIZE_CVaR = "minimize_cvar"
    MAXIMIZE_UTILITY = "maximize_utility"


@dataclass
class OptimizationConfig:
    """优化配置"""

    # 约束条件
    min_weight: float = 0.0  # 最小权重
    max_weight: float = 0.30  # 最大权重
    min_total_weight: float = 0.95  # 最小总权重
    max_total_weight: float = 1.0  # 最大总权重

    # 风险参数
    target_return: Optional[float] = None  # 目标收益率
    target_volatility: Optional[float] = 0.15  # 目标波动率
    risk_free_rate: float = 0.03  # 无风险利率
    risk_aversion: float = 3.0  # 风险厌恶系数

    # 优化参数
    lookback_period: int = 252  # 回看期
    covariance_method: str = "sample"  # 协方差估计方法
    shrinkage_factor: float = 0.1  # 收缩因子

    # 约束
    max_turnover: Optional[float] = 0.3  # 最大换手率
    sector_constraints: Optional[Dict[str, Tuple[float, float]]] = None  # 行业约束

    # Black-Litterman参数
    confidence_level: float = 0.5  # 观点置信度
    market_cap_weight: bool = True  # 是否使用市值加权

    # 集成方法
    ensemble_methods: List[str] = field(
        default_factory=lambda: ["mean_variance", "risk_parity", "max_sharpe"]
    )
    ensemble_weights: Optional[List[float]] = None  # 集成权重


@dataclass
class OptimizationResult:
    """优化结果"""

    weights: pd.Series  # 优化后的权重
    expected_return: float  # 预期收益率
    volatility: float  # 波动率
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # 索提诺比率
    max_drawdown: float  # 最大回撤
    diversification_ratio: float  # 分散化比率
    effective_n: float  # 有效资产数量

    method_used: str  # 使用的方法
    objective: str  # 优化目标
    success: bool  # 是否成功
    message: str  # 优化信息

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PortfolioWeightOptimizer:
    """投资组合权重优化器

    提供多种优化方法：
    - 均值-方差优化（Markowitz）
    - 风险平价
    - Black-Litterman模型
    - 最大分散化
    - 层次风险平价
    - 集成方法
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """初始化优化器

        Args:
            config: 优化配置
        """
        self.config = config or OptimizationConfig()
        self.optimization_history: List[OptimizationResult] = []
        logger.info(f"Initialized PortfolioWeightOptimizer with config: {self.config}")

    def optimize(
        self,
        returns_data: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
        objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_SHARPE,
        current_weights: Optional[pd.Series] = None,
        views: Optional[Dict[str, float]] = None,
        view_confidences: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """执行投资组合优化

        Args:
            returns_data: 收益率数据DataFrame
            method: 优化方法
            objective: 优化目标
            current_weights: 当前权重（用于turnover约束）
            views: 投资观点（用于Black-Litterman）
            view_confidences: 观点置信度

        Returns:
            优化结果
        """
        try:
            logger.info(f"Optimizing portfolio using {method.value} method")

            # 数据预处理
            returns_data = returns_data.dropna(axis=1, how="all")
            returns_data = returns_data.fillna(0)

            if returns_data.empty or len(returns_data.columns) == 0:
                raise ValueError("No valid return data provided")

            # 计算统计量
            expected_returns = returns_data.mean() * 252
            cov_matrix = self._estimate_covariance(returns_data)

            # 根据方法选择优化策略
            if method == OptimizationMethod.ENSEMBLE:
                weights = self._ensemble_optimization(
                    returns_data, expected_returns, cov_matrix, current_weights
                )
            elif method == OptimizationMethod.MEAN_VARIANCE:
                weights = self._mean_variance_optimization(
                    expected_returns, cov_matrix, objective
                )
            elif method == OptimizationMethod.MIN_VARIANCE:
                weights = self._min_variance_optimization(cov_matrix)
            elif method == OptimizationMethod.MAX_SHARPE:
                weights = self._max_sharpe_optimization(expected_returns, cov_matrix)
            elif method == OptimizationMethod.RISK_PARITY:
                weights = self._risk_parity_optimization(cov_matrix)
            elif method == OptimizationMethod.BLACK_LITTERMAN:
                weights = self._black_litterman_optimization(
                    returns_data, expected_returns, cov_matrix, views, view_confidences
                )
            elif method == OptimizationMethod.EQUAL_WEIGHT:
                weights = self._equal_weight_optimization(returns_data.columns)
            elif method == OptimizationMethod.INVERSE_VOLATILITY:
                weights = self._inverse_volatility_optimization(returns_data)
            elif method == OptimizationMethod.MAX_DIVERSIFICATION:
                weights = self._max_diversification_optimization(cov_matrix)
            elif method == OptimizationMethod.HIERARCHICAL_RISK_PARITY:
                weights = self._hierarchical_risk_parity_optimization(
                    returns_data, cov_matrix
                )
            else:
                raise ValueError(f"Unsupported optimization method: {method}")

            # 应用约束
            weights = self._apply_weight_constraints(weights, current_weights)

            # 计算组合指标
            result = self._calculate_portfolio_metrics(
                weights, expected_returns, cov_matrix, returns_data, method, objective
            )

            self.optimization_history.append(result)
            logger.info(
                f"Optimization completed: Return={result.expected_return:.2%}, "
                f"Volatility={result.volatility:.2%}, Sharpe={result.sharpe_ratio:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            raise QuantSystemError(f"Optimization failed: {e}")

    def optimize_with_constraints(
        self,
        returns_data: pd.DataFrame,
        sector_mapping: Dict[str, str],
        sector_limits: Dict[str, Tuple[float, float]],
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
    ) -> OptimizationResult:
        """带行业约束的优化

        Args:
            returns_data: 收益率数据
            sector_mapping: 股票到行业的映射
            sector_limits: 行业限制 {sector: (min_weight, max_weight)}
            method: 优化方法

        Returns:
            优化结果
        """
        try:
            logger.info("Optimizing portfolio with sector constraints")

            expected_returns = returns_data.mean() * 252
            cov_matrix = self._estimate_covariance(returns_data)

            n_assets = len(returns_data.columns)
            assets = returns_data.columns.tolist()

            # 构建行业约束矩阵
            sectors = list(set(sector_mapping.values()))
            sector_constraints = []

            for sector in sectors:
                sector_assets = [
                    i
                    for i, asset in enumerate(assets)
                    if sector_mapping.get(asset) == sector
                ]

                if sector in sector_limits:
                    min_weight, max_weight = sector_limits[sector]

                    # 构建约束矩阵行
                    constraint_row = np.zeros(n_assets)
                    constraint_row[sector_assets] = 1

                    sector_constraints.append((constraint_row, min_weight, max_weight))

            # 执行优化
            if method == OptimizationMethod.MAX_SHARPE:
                weights = self._max_sharpe_with_constraints(
                    expected_returns, cov_matrix, sector_constraints
                )
            else:
                weights = self._min_variance_with_constraints(
                    cov_matrix, sector_constraints
                )

            weights_series = pd.Series(weights, index=returns_data.columns)

            # 计算指标
            result = self._calculate_portfolio_metrics(
                weights_series,
                expected_returns,
                cov_matrix,
                returns_data,
                method,
                OptimizationObjective.MAXIMIZE_SHARPE,
            )

            return result

        except Exception as e:
            logger.error(f"Constrained optimization failed: {e}")
            raise QuantSystemError(f"Constrained optimization failed: {e}")

    def backtest_optimization(
        self,
        returns_data: pd.DataFrame,
        rebalance_frequency: str = "monthly",
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
    ) -> Dict[str, Any]:
        """回测优化策略

        Args:
            returns_data: 收益率数据
            rebalance_frequency: 再平衡频率
            method: 优化方法

        Returns:
            回测结果
        """
        try:
            logger.info(f"Backtesting {method.value} strategy")

            # 确定再平衡日期
            rebalance_dates = self._get_rebalance_dates(
                returns_data.index, rebalance_frequency
            )

            portfolio_values = [1.0]
            portfolio_returns = []
            weights_history = []

            for i in range(len(rebalance_dates) - 1):
                start_date = rebalance_dates[i]
                end_date = rebalance_dates[i + 1]

                # 获取优化期数据
                lookback_data = returns_data.loc[:start_date].tail(
                    self.config.lookback_period
                )

                if len(lookback_data) < 30:
                    continue

                # 执行优化
                result = self.optimize(lookback_data, method=method)
                weights = result.weights
                weights_history.append(
                    {"date": start_date, "weights": weights.to_dict()}
                )

                # 计算持有期收益
                holding_returns = returns_data.loc[start_date:end_date]

                if not holding_returns.empty:
                    # 确保权重和收益对齐
                    aligned_weights = weights.reindex(
                        holding_returns.columns, fill_value=0
                    )
                    aligned_weights = aligned_weights / aligned_weights.sum()

                    portfolio_return = (holding_returns * aligned_weights).sum(axis=1)
                    portfolio_returns.extend(portfolio_return.tolist())

                    # 更新组合价值
                    period_cumulative = (1 + portfolio_return).prod()
                    portfolio_values.append(portfolio_values[-1] * period_cumulative)

            # 计算回测指标
            portfolio_returns_series = pd.Series(portfolio_returns)

            backtest_results = {
                "total_return": (portfolio_values[-1] - 1) * 100,
                "annualized_return": (portfolio_returns_series.mean() * 252) * 100,
                "annualized_volatility": (portfolio_returns_series.std() * np.sqrt(252))
                * 100,
                "sharpe_ratio": (
                    (portfolio_returns_series.mean() - self.config.risk_free_rate / 252)
                    / portfolio_returns_series.std()
                    * np.sqrt(252)
                    if portfolio_returns_series.std() > 0
                    else 0
                ),
                "max_drawdown": self._calculate_max_drawdown(portfolio_values) * 100,
                "n_rebalances": len(weights_history),
                "portfolio_values": portfolio_values,
                "portfolio_returns": portfolio_returns,
                "weights_history": weights_history,
            }

            logger.info(
                f"Backtest completed: Total Return={backtest_results['total_return']:.2f}%, "
                f"Sharpe={backtest_results['sharpe_ratio']:.2f}"
            )

            return backtest_results

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise QuantSystemError(f"Backtest failed: {e}")

    def _estimate_covariance(self, returns_data: pd.DataFrame) -> np.ndarray:
        """估计协方差矩阵

        Args:
            returns_data: 收益率数据

        Returns:
            协方差矩阵
        """
        if self.config.covariance_method == "sample":
            cov_matrix = returns_data.cov().values * 252
        elif self.config.covariance_method == "shrinkage":
            cov_matrix = self._ledoit_wolf_shrinkage(returns_data)
        elif self.config.covariance_method == "exponential":
            cov_matrix = (
                returns_data.ewm(span=60)
                .cov()
                .iloc[-len(returns_data.columns) :]
                .values
                * 252
            )
        else:
            cov_matrix = returns_data.cov().values * 252

        return cov_matrix

    def _ledoit_wolf_shrinkage(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Ledoit-Wolf收缩估计器

        Args:
            returns_data: 收益率数据

        Returns:
            收缩后的协方差矩阵
        """
        sample_cov = returns_data.cov().values * 252
        n_assets = len(sample_cov)

        # 目标矩阵（常数相关性模型）
        volatilities = np.sqrt(np.diag(sample_cov))
        avg_corr = (sample_cov / np.outer(volatilities, volatilities)).sum() / (
            n_assets * (n_assets - 1)
        )
        target = avg_corr * np.outer(volatilities, volatilities)
        np.fill_diagonal(target, np.diag(sample_cov))

        # 收缩
        shrunk_cov = (
            1 - self.config.shrinkage_factor
        ) * sample_cov + self.config.shrinkage_factor * target

        return shrunk_cov

    def _mean_variance_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray,
        objective: OptimizationObjective,
    ) -> pd.Series:
        """均值-方差优化"""
        n_assets = len(expected_returns)

        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights

        def portfolio_return(weights):
            return weights @ expected_returns.values

        def negative_sharpe(weights):
            ret = portfolio_return(weights)
            vol = np.sqrt(portfolio_variance(weights))
            return -(ret - self.config.risk_free_rate) / vol if vol > 0 else 0

        # 选择目标函数
        if objective == OptimizationObjective.MINIMIZE_RISK:
            objective_func = portfolio_variance
        elif objective == OptimizationObjective.MAXIMIZE_SHARPE:
            objective_func = negative_sharpe
        else:
            objective_func = lambda w: -portfolio_return(w)

        # 约束和边界
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets

        # 初始猜测
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = minimize(
            objective_func,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        return pd.Series(result.x, index=expected_returns.index)

    def _min_variance_optimization(self, cov_matrix: np.ndarray) -> pd.Series:
        """最小方差优化"""
        n_assets = len(cov_matrix)

        def objective(weights):
            return weights @ cov_matrix @ weights

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets
        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        return pd.Series(result.x, index=range(n_assets))

    def _max_sharpe_optimization(
        self, expected_returns: pd.Series, cov_matrix: np.ndarray
    ) -> pd.Series:
        """最大夏普比率优化"""
        n_assets = len(expected_returns)

        def negative_sharpe(weights):
            ret = weights @ expected_returns.values
            vol = np.sqrt(weights @ cov_matrix @ weights)
            return -(ret - self.config.risk_free_rate) / vol if vol > 0 else 1e10

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets
        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            negative_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        return pd.Series(result.x, index=expected_returns.index)

    def _risk_parity_optimization(self, cov_matrix: np.ndarray) -> pd.Series:
        """风险平价优化"""
        n_assets = len(cov_matrix)

        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol

            # 目标：所有资产的风险贡献相等
            target_contrib = portfolio_vol / n_assets
            return np.sum((risk_contrib - target_contrib) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets
        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            risk_parity_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        return pd.Series(result.x, index=range(n_assets))

    def _black_litterman_optimization(
        self,
        returns_data: pd.DataFrame,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray,
        views: Optional[Dict[str, float]] = None,
        view_confidences: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """Black-Litterman优化"""
        n_assets = len(expected_returns)

        # 市场均衡收益
        if self.config.market_cap_weight:
            market_weights = np.ones(n_assets) / n_assets  # 简化：等权重
        else:
            market_weights = np.ones(n_assets) / n_assets

        pi = self.config.risk_aversion * cov_matrix @ market_weights

        # 如果有观点，融合观点
        if views and len(views) > 0:
            # 构建观点矩阵
            assets = expected_returns.index.tolist()
            P = []
            Q = []
            omega_diag = []

            for asset, view_return in views.items():
                if asset in assets:
                    idx = assets.index(asset)
                    p_row = np.zeros(n_assets)
                    p_row[idx] = 1
                    P.append(p_row)
                    Q.append(view_return)

                    # 观点不确定性
                    confidence = (
                        view_confidences.get(asset, self.config.confidence_level)
                        if view_confidences
                        else self.config.confidence_level
                    )
                    omega_diag.append(cov_matrix[idx, idx] / confidence)

            if P:
                P = np.array(P)
                Q = np.array(Q)
                Omega = np.diag(omega_diag)

                # Black-Litterman公式
                tau = 0.05
                M_inv = np.linalg.inv(tau * cov_matrix + P.T @ np.linalg.inv(Omega) @ P)
                bl_returns = M_inv @ (
                    tau * cov_matrix @ pi + P.T @ np.linalg.inv(Omega) @ Q
                )
            else:
                bl_returns = pi
        else:
            bl_returns = pi

        # 使用BL收益进行均值-方差优化
        bl_returns_series = pd.Series(bl_returns, index=expected_returns.index)
        return self._max_sharpe_optimization(bl_returns_series, cov_matrix)

    def _equal_weight_optimization(self, assets: pd.Index) -> pd.Series:
        """等权重优化"""
        n_assets = len(assets)
        weights = np.ones(n_assets) / n_assets
        return pd.Series(weights, index=assets)

    def _inverse_volatility_optimization(self, returns_data: pd.DataFrame) -> pd.Series:
        """逆波动率优化"""
        volatilities = returns_data.std() * np.sqrt(252)
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        return weights

    def _max_diversification_optimization(self, cov_matrix: np.ndarray) -> pd.Series:
        """最大分散化优化"""
        n_assets = len(cov_matrix)
        volatilities = np.sqrt(np.diag(cov_matrix))

        def negative_diversification_ratio(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            weighted_vol = weights @ volatilities
            return -weighted_vol / portfolio_vol if portfolio_vol > 0 else 1e10

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets
        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            negative_diversification_ratio,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return pd.Series(result.x, index=range(n_assets))

    def _hierarchical_risk_parity_optimization(
        self, returns_data: pd.DataFrame, cov_matrix: np.ndarray
    ) -> pd.Series:
        """层次风险平价优化"""
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import squareform

        # 计算距离矩阵
        corr_matrix = returns_data.corr()
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))

        # 层次聚类
        condensed_dist = squareform(dist_matrix)
        link = linkage(condensed_dist, method="single")

        # 获取排序
        sorted_indices = self._get_quasi_diag(link)

        # 递归二分
        weights = self._recursive_bisection(cov_matrix, sorted_indices)

        weights_series = pd.Series(
            [weights[sorted_indices.index(i)] for i in range(len(weights))],
            index=returns_data.columns,
        )

        return weights_series

    def _ensemble_optimization(
        self,
        returns_data: pd.DataFrame,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray,
        current_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """集成多种优化方法"""
        logger.info("Running ensemble optimization")

        methods_map = {
            "mean_variance": lambda: self._mean_variance_optimization(
                expected_returns, cov_matrix, OptimizationObjective.MAXIMIZE_SHARPE
            ),
            "risk_parity": lambda: self._risk_parity_optimization(cov_matrix),
            "max_sharpe": lambda: self._max_sharpe_optimization(
                expected_returns, cov_matrix
            ),
            "min_variance": lambda: self._min_variance_optimization(cov_matrix),
            "inverse_volatility": lambda: self._inverse_volatility_optimization(
                returns_data
            ),
            "max_diversification": lambda: self._max_diversification_optimization(
                cov_matrix
            ),
        }

        # 执行所有方法
        all_weights = []
        valid_methods = []

        for method_name in self.config.ensemble_methods:
            if method_name in methods_map:
                try:
                    weights = methods_map[method_name]()
                    all_weights.append(weights)
                    valid_methods.append(method_name)
                    logger.info(f"Ensemble: {method_name} completed")
                except Exception as e:
                    logger.warning(f"Ensemble: {method_name} failed: {e}")

        if not all_weights:
            logger.warning("All ensemble methods failed, using equal weights")
            return self._equal_weight_optimization(returns_data.columns)

        # 加权平均
        if self.config.ensemble_weights:
            weights = self.config.ensemble_weights[: len(all_weights)]
        else:
            weights = [1.0 / len(all_weights)] * len(all_weights)

        # 确保权重和为1
        weights = np.array(weights)
        weights = weights / weights.sum()

        # 计算加权平均
        ensemble_weights = sum(
            w * method_weights for w, method_weights in zip(weights, all_weights)
        )
        ensemble_weights = ensemble_weights / ensemble_weights.sum()

        return ensemble_weights

    def _max_sharpe_with_constraints(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray,
        sector_constraints: List[Tuple[np.ndarray, float, float]],
    ) -> np.ndarray:
        """带约束的最大夏普比率优化"""
        n_assets = len(expected_returns)

        def negative_sharpe(weights):
            ret = weights @ expected_returns.values
            vol = np.sqrt(weights @ cov_matrix @ weights)
            return -(ret - self.config.risk_free_rate) / vol if vol > 0 else 1e10

        # 基本约束
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        # 添加行业约束
        for constraint_row, min_w, max_w in sector_constraints:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, row=constraint_row, min_val=min_w: w @ row
                    - min_val,
                }
            )
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, row=constraint_row, max_val=max_w: max_val
                    - w @ row,
                }
            )

        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets
        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            negative_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        return result.x

    def _min_variance_with_constraints(
        self,
        cov_matrix: np.ndarray,
        sector_constraints: List[Tuple[np.ndarray, float, float]],
    ) -> np.ndarray:
        """带约束的最小方差优化"""
        n_assets = len(cov_matrix)

        def objective(weights):
            return weights @ cov_matrix @ weights

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        for constraint_row, min_w, max_w in sector_constraints:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, row=constraint_row, min_val=min_w: w @ row
                    - min_val,
                }
            )
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, row=constraint_row, max_val=max_w: max_val
                    - w @ row,
                }
            )

        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets
        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        return result.x

    def _apply_weight_constraints(
        self, weights: pd.Series, current_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """应用权重约束"""
        # 应用最小最大权重约束
        weights = weights.clip(
            lower=self.config.min_weight, upper=self.config.max_weight
        )

        # 重新归一化
        total_weight = weights.sum()
        if total_weight > 0:
            weights = weights / total_weight

        # 应用换手率约束
        if current_weights is not None and self.config.max_turnover is not None:
            # 计算换手率
            aligned_current = current_weights.reindex(weights.index, fill_value=0)
            turnover = (weights - aligned_current).abs().sum()

            if turnover > self.config.max_turnover:
                # 调整权重以满足换手率约束
                scale = self.config.max_turnover / turnover
                weights = aligned_current + (weights - aligned_current) * scale
                weights = weights / weights.sum()
                logger.info(
                    f"Applied turnover constraint: {turnover:.2%} -> {self.config.max_turnover:.2%}"
                )

        return weights

    def _calculate_portfolio_metrics(
        self,
        weights: pd.Series,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray,
        returns_data: pd.DataFrame,
        method: OptimizationMethod,
        objective: OptimizationObjective,
    ) -> OptimizationResult:
        """计算投资组合指标"""
        # 对齐权重和期望收益
        aligned_weights = weights.reindex(expected_returns.index, fill_value=0)
        aligned_weights = aligned_weights / aligned_weights.sum()

        # 基本指标
        portfolio_return = aligned_weights @ expected_returns
        portfolio_variance = (
            aligned_weights.values @ cov_matrix @ aligned_weights.values
        )
        portfolio_volatility = np.sqrt(portfolio_variance)

        # 夏普比率
        sharpe_ratio = (
            (portfolio_return - self.config.risk_free_rate) / portfolio_volatility
            if portfolio_volatility > 0
            else 0
        )

        # 索提诺比率
        portfolio_returns = (returns_data * aligned_weights).sum(axis=1)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = (
            downside_returns.std() * np.sqrt(252)
            if len(downside_returns) > 0
            else portfolio_volatility
        )
        sortino_ratio = (
            (portfolio_return - self.config.risk_free_rate) / downside_std
            if downside_std > 0
            else 0
        )

        # 最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        max_drawdown = self._calculate_max_drawdown(cumulative.tolist())

        # 分散化比率
        asset_volatilities = np.sqrt(np.diag(cov_matrix))
        weighted_volatility = aligned_weights.values @ asset_volatilities
        diversification_ratio = (
            weighted_volatility / portfolio_volatility
            if portfolio_volatility > 0
            else 1.0
        )

        # 有效资产数量（Herfindahl指数的倒数）
        effective_n = (
            1 / (aligned_weights**2).sum()
            if (aligned_weights**2).sum() > 0
            else len(weights)
        )

        return OptimizationResult(
            weights=aligned_weights,
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            diversification_ratio=diversification_ratio,
            effective_n=effective_n,
            method_used=method.value,
            objective=objective.value,
            success=True,
            message="Optimization completed successfully",
            metadata={
                "n_assets": len(weights),
                "n_non_zero_weights": (weights > 0.01).sum(),
                "max_weight": weights.max(),
                "min_weight": weights.min(),
                "weight_concentration": (weights**2).sum(),
            },
        )

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """获取准对角化排序"""
        link = link.astype(int)
        sorted_indices = []
        clusters = {i: [i] for i in range(link.shape[0] + 1)}

        for i in range(link.shape[0]):
            cluster1 = int(link[i, 0])
            cluster2 = int(link[i, 1])
            clusters[link.shape[0] + 1 + i] = clusters[cluster1] + clusters[cluster2]
            del clusters[cluster1]
            del clusters[cluster2]

        return clusters[max(clusters.keys())]

    def _recursive_bisection(
        self, cov_matrix: np.ndarray, sorted_indices: List[int]
    ) -> np.ndarray:
        """递归二分法"""

        def _bisect(indices):
            if len(indices) == 1:
                return np.array([1.0])

            mid = len(indices) // 2
            left_indices = indices[:mid]
            right_indices = indices[mid:]

            # 计算两侧的方差
            cov_left = cov_matrix[np.ix_(left_indices, left_indices)]
            cov_right = cov_matrix[np.ix_(right_indices, right_indices)]

            inv_vol_left = 1 / np.sqrt(
                np.ones(len(left_indices)) @ cov_left @ np.ones(len(left_indices))
            )
            inv_vol_right = 1 / np.sqrt(
                np.ones(len(right_indices)) @ cov_right @ np.ones(len(right_indices))
            )

            alpha = inv_vol_left / (inv_vol_left + inv_vol_right)

            weights_left = _bisect(left_indices) * alpha
            weights_right = _bisect(right_indices) * (1 - alpha)

            return np.concatenate([weights_left, weights_right])

        weights_sorted = _bisect(list(range(len(sorted_indices))))

        # 恢复原始顺序
        weights = np.zeros(len(sorted_indices))
        for i, idx in enumerate(sorted_indices):
            weights[idx] = weights_sorted[i]

        return weights

    def _get_rebalance_dates(
        self, date_index: pd.DatetimeIndex, frequency: str
    ) -> List:
        """获取再平衡日期"""
        if frequency == "daily":
            return date_index.tolist()
        elif frequency == "weekly":
            return date_index[date_index.to_series().dt.dayofweek == 0].tolist()
        elif frequency == "monthly":
            return date_index[date_index.to_series().dt.is_month_end].tolist()
        elif frequency == "quarterly":
            return date_index[date_index.to_series().dt.is_quarter_end].tolist()
        else:
            return date_index[::21].tolist()  # 默认约每月

    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """计算最大回撤"""
        if not values:
            return 0.0

        values = np.array(values)
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max

        return abs(drawdown.min()) if len(drawdown) > 0 else 0.0


# 便捷函数
def optimize_portfolio(
    returns_data: pd.DataFrame,
    method: str = "max_sharpe",
    config: Optional[OptimizationConfig] = None,
) -> OptimizationResult:
    """优化投资组合的便捷函数

    Args:
        returns_data: 收益率数据
        method: 优化方法名称
        config: 优化配置

    Returns:
        优化结果
    """
    optimizer = PortfolioWeightOptimizer(config)
    method_enum = OptimizationMethod(method)
    return optimizer.optimize(returns_data, method=method_enum)
