"""
风险敞口分析器模块
实现投资组合风险敞口的全面分析
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("risk_exposure_analyzer")


@dataclass
class ExposureConfig:
    """风险敞口配置"""

    max_single_asset_exposure: float = 0.20  # 单资产最大敞口20%
    max_sector_exposure: float = 0.40  # 单行业最大敞口40%
    max_total_leverage: float = 1.0  # 最大杠杆率
    max_net_exposure: float = 1.0  # 最大净敞口
    max_gross_exposure: float = 1.5  # 最大总敞口
    risk_free_rate: float = 0.03  # 无风险利率


@dataclass
class ExposureResult:
    """风险敞口分析结果"""

    total_exposure: float
    net_exposure: float
    gross_exposure: float
    leverage: float
    asset_exposures: Dict[str, float]
    sector_exposures: Dict[str, float]
    concentration_risk: float
    exposure_violations: List[Dict[str, Any]]
    diversification_score: float
    beta_exposure: float
    currency_exposures: Dict[str, float]
    metadata: Dict[str, Any]


class RiskExposureAnalyzer:
    """风险敞口分析器"""

    def __init__(self, config: Optional[ExposureConfig] = None):
        """初始化风险敞口分析器

        Args:
            config: 风险敞口配置
        """
        self.config = config or ExposureConfig()
        logger.info("Initialized RiskExposureAnalyzer")

    def analyze_portfolio_exposure(
        self,
        portfolio: Dict[str, Dict[str, Any]],
        market_data: pd.DataFrame,
        sector_mapping: Optional[Dict[str, str]] = None,
    ) -> ExposureResult:
        """分析投资组合风险敞口

        Args:
            portfolio: 投资组合 {symbol: {'weight': float, 'shares': int, 'value': float}}
            market_data: 市场数据
            sector_mapping: 股票到行业的映射

        Returns:
            风险敞口分析结果
        """
        try:
            logger.info("Analyzing portfolio exposure...")

            # 计算总敞口
            total_exposure = self._calculate_total_exposure(portfolio)

            # 计算净敞口和总敞口
            net_exposure, gross_exposure = self._calculate_net_gross_exposure(portfolio)

            # 计算杠杆率
            leverage = self._calculate_leverage(portfolio)

            # 计算资产敞口
            asset_exposures = self._calculate_asset_exposures(portfolio)

            # 计算行业敞口
            sector_exposures = self._calculate_sector_exposures(
                portfolio, sector_mapping
            )

            # 计算集中度风险
            concentration_risk = self._calculate_concentration_risk(asset_exposures)

            # 检查敞口违规
            violations = self._check_exposure_violations(
                asset_exposures,
                sector_exposures,
                leverage,
                net_exposure,
                gross_exposure,
            )

            # 计算分散化得分
            diversification_score = self._calculate_diversification_score(
                asset_exposures
            )

            # 计算Beta敞口
            beta_exposure = self._calculate_beta_exposure(portfolio, market_data)

            # 计算货币敞口（简化版）
            currency_exposures = self._calculate_currency_exposures(portfolio)

            result = ExposureResult(
                total_exposure=total_exposure,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                leverage=leverage,
                asset_exposures=asset_exposures,
                sector_exposures=sector_exposures,
                concentration_risk=concentration_risk,
                exposure_violations=violations,
                diversification_score=diversification_score,
                beta_exposure=beta_exposure,
                currency_exposures=currency_exposures,
                metadata={
                    "analysis_time": datetime.now().isoformat(),
                    "n_positions": len(portfolio),
                    "n_sectors": len(sector_exposures),
                },
            )

            logger.info(
                f"Exposure analysis completed: {len(violations)} violations found"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to analyze portfolio exposure: {e}")
            raise QuantSystemError(f"Exposure analysis failed: {e}")

    def calculate_marginal_exposure(
        self,
        portfolio: Dict[str, Dict[str, Any]],
        new_position: Dict[str, Any],
        symbol: str,
    ) -> Dict[str, float]:
        """计算新头寸的边际敞口影响

        Args:
            portfolio: 当前投资组合
            new_position: 新头寸信息
            symbol: 股票代码

        Returns:
            边际敞口影响
        """
        try:
            # 当前敞口
            current_exposure = self._calculate_total_exposure(portfolio)

            # 添加新头寸后的投资组合
            new_portfolio = portfolio.copy()
            new_portfolio[symbol] = new_position

            # 新敞口
            new_exposure = self._calculate_total_exposure(new_portfolio)

            # 边际影响
            marginal_exposure = new_exposure - current_exposure

            # 计算各维度的边际影响
            result = {
                "marginal_exposure": marginal_exposure,
                "marginal_exposure_pct": marginal_exposure / current_exposure
                if current_exposure > 0
                else 0,
                "new_total_exposure": new_exposure,
                "position_weight": new_position["weight"],
            }

            return result

        except Exception as e:
            logger.error(f"Failed to calculate marginal exposure: {e}")
            return {}

    def monitor_exposure_limits(
        self, portfolio: Dict[str, Dict[str, Any]]
    ) -> Dict[str, bool]:
        """监控敞口限制

        Args:
            portfolio: 投资组合

        Returns:
            限制检查结果
        """
        try:
            checks = {}

            # 计算各项指标
            asset_exposures = self._calculate_asset_exposures(portfolio)
            net_exposure, gross_exposure = self._calculate_net_gross_exposure(portfolio)
            leverage = self._calculate_leverage(portfolio)

            # 检查单资产敞口
            max_asset_exposure = max(asset_exposures.values()) if asset_exposures else 0
            checks["single_asset_limit"] = (
                max_asset_exposure <= self.config.max_single_asset_exposure
            )

            # 检查净敞口
            checks["net_exposure_limit"] = (
                abs(net_exposure) <= self.config.max_net_exposure
            )

            # 检查总敞口
            checks["gross_exposure_limit"] = (
                gross_exposure <= self.config.max_gross_exposure
            )

            # 检查杠杆率
            checks["leverage_limit"] = leverage <= self.config.max_total_leverage

            # 总体是否合规
            checks["all_limits_ok"] = all(checks.values())

            return checks

        except Exception as e:
            logger.error(f"Failed to monitor exposure limits: {e}")
            return {"all_limits_ok": False, "error": str(e)}

    def _calculate_total_exposure(self, portfolio: Dict[str, Dict[str, Any]]) -> float:
        """计算总敞口

        Args:
            portfolio: 投资组合

        Returns:
            总敞口
        """
        total = sum(abs(pos.get("weight", 0)) for pos in portfolio.values())
        return total

    def _calculate_net_gross_exposure(
        self, portfolio: Dict[str, Dict[str, Any]]
    ) -> Tuple[float, float]:
        """计算净敞口和总敞口

        Args:
            portfolio: 投资组合

        Returns:
            (净敞口, 总敞口)
        """
        net_exposure = sum(pos.get("weight", 0) for pos in portfolio.values())
        gross_exposure = sum(abs(pos.get("weight", 0)) for pos in portfolio.values())

        return net_exposure, gross_exposure

    def _calculate_leverage(self, portfolio: Dict[str, Dict[str, Any]]) -> float:
        """计算杠杆率

        Args:
            portfolio: 投资组合

        Returns:
            杠杆率
        """
        # 杠杆率 = 总敞口 / 净资产
        # 这里假设净资产为1
        gross_exposure = sum(abs(pos.get("weight", 0)) for pos in portfolio.values())
        return gross_exposure

    def _calculate_asset_exposures(
        self, portfolio: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """计算资产敞口

        Args:
            portfolio: 投资组合

        Returns:
            资产敞口字典
        """
        exposures = {}
        for symbol, pos in portfolio.items():
            exposures[symbol] = abs(pos.get("weight", 0))

        return exposures

    def _calculate_sector_exposures(
        self,
        portfolio: Dict[str, Dict[str, Any]],
        sector_mapping: Optional[Dict[str, str]],
    ) -> Dict[str, float]:
        """计算行业敞口

        Args:
            portfolio: 投资组合
            sector_mapping: 行业映射

        Returns:
            行业敞口字典
        """
        sector_exposures = {}

        if sector_mapping is None:
            # 如果没有行业映射，返回空字典
            return sector_exposures

        for symbol, pos in portfolio.items():
            sector = sector_mapping.get(symbol, "Unknown")
            weight = abs(pos.get("weight", 0))

            if sector in sector_exposures:
                sector_exposures[sector] += weight
            else:
                sector_exposures[sector] = weight

        return sector_exposures

    def _calculate_concentration_risk(self, asset_exposures: Dict[str, float]) -> float:
        """计算集中度风险（赫芬达尔指数）

        Args:
            asset_exposures: 资产敞口

        Returns:
            集中度风险得分
        """
        if not asset_exposures:
            return 0.0

        # 赫芬达尔指数
        weights = np.array(list(asset_exposures.values()))
        total_weight = weights.sum()

        if total_weight > 0:
            normalized_weights = weights / total_weight
            herfindahl = np.sum(normalized_weights**2)
        else:
            herfindahl = 0

        return herfindahl

    def _check_exposure_violations(
        self,
        asset_exposures: Dict[str, float],
        sector_exposures: Dict[str, float],
        leverage: float,
        net_exposure: float,
        gross_exposure: float,
    ) -> List[Dict[str, Any]]:
        """检查敞口违规

        Args:
            asset_exposures: 资产敞口
            sector_exposures: 行业敞口
            leverage: 杠杆率
            net_exposure: 净敞口
            gross_exposure: 总敞口

        Returns:
            违规列表
        """
        violations = []

        # 检查单资产敞口
        for asset, exposure in asset_exposures.items():
            if exposure > self.config.max_single_asset_exposure:
                violations.append(
                    {
                        "type": "single_asset_exposure",
                        "asset": asset,
                        "exposure": exposure,
                        "limit": self.config.max_single_asset_exposure,
                        "excess": exposure - self.config.max_single_asset_exposure,
                    }
                )

        # 检查行业敞口
        for sector, exposure in sector_exposures.items():
            if exposure > self.config.max_sector_exposure:
                violations.append(
                    {
                        "type": "sector_exposure",
                        "sector": sector,
                        "exposure": exposure,
                        "limit": self.config.max_sector_exposure,
                        "excess": exposure - self.config.max_sector_exposure,
                    }
                )

        # 检查杠杆率
        if leverage > self.config.max_total_leverage:
            violations.append(
                {
                    "type": "leverage",
                    "leverage": leverage,
                    "limit": self.config.max_total_leverage,
                    "excess": leverage - self.config.max_total_leverage,
                }
            )

        # 检查净敞口
        if abs(net_exposure) > self.config.max_net_exposure:
            violations.append(
                {
                    "type": "net_exposure",
                    "exposure": net_exposure,
                    "limit": self.config.max_net_exposure,
                    "excess": abs(net_exposure) - self.config.max_net_exposure,
                }
            )

        # 检查总敞口
        if gross_exposure > self.config.max_gross_exposure:
            violations.append(
                {
                    "type": "gross_exposure",
                    "exposure": gross_exposure,
                    "limit": self.config.max_gross_exposure,
                    "excess": gross_exposure - self.config.max_gross_exposure,
                }
            )

        return violations

    def _calculate_diversification_score(
        self, asset_exposures: Dict[str, float]
    ) -> float:
        """计算分散化得分

        Args:
            asset_exposures: 资产敞口

        Returns:
            分散化得分（0-1，1表示完全分散）
        """
        if not asset_exposures or len(asset_exposures) == 1:
            return 0.0

        # 基于有效资产数量
        weights = np.array(list(asset_exposures.values()))
        total_weight = weights.sum()

        if total_weight > 0:
            normalized_weights = weights / total_weight
            # 有效N = 1 / sum(w_i^2)
            effective_n = 1 / np.sum(normalized_weights**2)
            # 归一化到0-1
            max_n = len(asset_exposures)
            diversification_score = (effective_n - 1) / (max_n - 1) if max_n > 1 else 0
        else:
            diversification_score = 0

        return diversification_score

    def _calculate_beta_exposure(
        self, portfolio: Dict[str, Dict[str, Any]], market_data: pd.DataFrame
    ) -> float:
        """计算Beta敞口

        Args:
            portfolio: 投资组合
            market_data: 市场数据

        Returns:
            投资组合Beta
        """
        try:
            # 简化计算：假设所有股票的Beta都是1
            # 实际应该根据历史数据计算每个股票的Beta
            weighted_beta = sum(pos.get("weight", 0) for pos in portfolio.values())

            return weighted_beta

        except Exception as e:
            logger.warning(f"Failed to calculate beta exposure: {e}")
            return 1.0

    def _calculate_currency_exposures(
        self, portfolio: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """计算货币敞口

        Args:
            portfolio: 投资组合

        Returns:
            货币敞口字典
        """
        # 简化版本：假设所有都是人民币
        currency_exposures = {"CNY": self._calculate_total_exposure(portfolio)}

        return currency_exposures


# 便捷函数
def analyze_exposure(
    portfolio: Dict[str, Dict[str, Any]],
    market_data: pd.DataFrame,
    config: Optional[ExposureConfig] = None,
) -> ExposureResult:
    """分析投资组合敞口的便捷函数

    Args:
        portfolio: 投资组合
        market_data: 市场数据
        config: 敞口配置

    Returns:
        敞口分析结果
    """
    analyzer = RiskExposureAnalyzer(config)
    return analyzer.analyze_portfolio_exposure(portfolio, market_data)
