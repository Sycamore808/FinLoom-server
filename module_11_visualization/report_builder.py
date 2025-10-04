"""
报告生成器模块
负责生成各类投资报告
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import markdown
import numpy as np
import pandas as pd
import pdfkit
from jinja2 import Environment, FileSystemLoader, Template

from common.data_structures import Position, Signal
from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("report_builder")


@dataclass
class ReportConfig:
    """报告配置数据类"""

    report_type: str  # 'daily', 'weekly', 'monthly', 'performance', 'risk'
    template_name: str
    output_format: str  # 'html', 'pdf', 'markdown', 'json'
    include_charts: bool = True
    include_tables: bool = True
    include_summary: bool = True
    custom_sections: List[str] = field(default_factory=list)


@dataclass
class ReportSection:
    """报告章节数据类"""

    section_id: str
    title: str
    content_type: str  # 'text', 'table', 'chart', 'metric'
    content: Any
    order: int
    visible: bool = True


@dataclass
class PerformanceMetrics:
    """绩效指标数据类"""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    total_trades: int
    winning_trades: int
    losing_trades: int


class ReportBuilder:
    """报告生成器类"""

    TEMPLATE_DIR = "module_11_visualization/templates"
    OUTPUT_DIR = "module_11_visualization/reports"

    DEFAULT_TEMPLATES = {
        "daily": "daily_report.html",
        "weekly": "weekly_report.html",
        "monthly": "monthly_report.html",
        "performance": "performance_report.html",
        "risk": "risk_report.html",
    }

    def __init__(self, template_dir: Optional[str] = None):
        """初始化报告生成器

        Args:
            template_dir: 模板目录
        """
        self.template_dir = template_dir or self.TEMPLATE_DIR
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
        self.sections: Dict[str, ReportSection] = {}
        self.metadata: Dict[str, Any] = {}

    def generate_daily_report(
        self,
        date: datetime,
        portfolio_data: Dict[str, Any],
        positions: List[Position],
        signals: List[Signal],
        market_data: pd.DataFrame,
        config: Optional[ReportConfig] = None,
    ) -> str:
        """生成日报

        Args:
            date: 报告日期
            portfolio_data: 组合数据
            positions: 持仓列表
            signals: 信号列表
            market_data: 市场数据
            config: 报告配置

        Returns:
            报告内容
        """
        if config is None:
            config = ReportConfig(
                report_type="daily",
                template_name="daily_report.html",
                output_format="html",
            )

        # 清空之前的章节
        self.sections.clear()

        # 设置元数据
        self.metadata = {
            "report_date": date.strftime("%Y-%m-%d"),
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "report_type": "Daily Report",
        }

        # 添加概要章节
        if config.include_summary:
            self._add_summary_section(portfolio_data, date)

        # 添加持仓章节
        self._add_positions_section(positions)

        # 添加交易信号章节
        self._add_signals_section(signals, date)

        # 添加市场概览章节
        self._add_market_overview_section(market_data)

        # 添加绩效章节
        self._add_performance_section(portfolio_data)

        # 生成报告
        return self._render_report(config)

    def generate_weekly_summary(
        self,
        start_date: datetime,
        end_date: datetime,
        weekly_data: Dict[str, Any],
        config: Optional[ReportConfig] = None,
    ) -> str:
        """生成周报

        Args:
            start_date: 开始日期
            end_date: 结束日期
            weekly_data: 周数据
            config: 报告配置

        Returns:
            报告内容
        """
        if config is None:
            config = ReportConfig(
                report_type="weekly",
                template_name="weekly_report.html",
                output_format="html",
            )

        self.sections.clear()

        self.metadata = {
            "report_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "report_type": "Weekly Summary",
        }

        # 添加周概要
        self._add_section(
            ReportSection(
                section_id="weekly_overview",
                title="Weekly Overview",
                content_type="text",
                content=self._format_weekly_overview(weekly_data),
                order=1,
            )
        )

        # 添加周绩效
        if "performance" in weekly_data:
            self._add_section(
                ReportSection(
                    section_id="weekly_performance",
                    title="Weekly Performance",
                    content_type="table",
                    content=self._format_performance_table(weekly_data["performance"]),
                    order=2,
                )
            )

        # 添加交易统计
        if "trades" in weekly_data:
            self._add_section(
                ReportSection(
                    section_id="trade_statistics",
                    title="Trade Statistics",
                    content_type="table",
                    content=self._format_trade_statistics(weekly_data["trades"]),
                    order=3,
                )
            )

        return self._render_report(config)

    def generate_performance_report(
        self,
        performance_data: Dict[str, Any],
        metrics: PerformanceMetrics,
        config: Optional[ReportConfig] = None,
    ) -> str:
        """生成绩效报告

        Args:
            performance_data: 绩效数据
            metrics: 绩效指标
            config: 报告配置

        Returns:
            报告内容
        """
        if config is None:
            config = ReportConfig(
                report_type="performance",
                template_name="performance_report.html",
                output_format="html",
            )

        self.sections.clear()

        self.metadata = {
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "report_type": "Performance Analysis Report",
        }

        # 添加绩效概要
        self._add_section(
            ReportSection(
                section_id="performance_summary",
                title="Performance Summary",
                content_type="metric",
                content=self._format_metrics(metrics),
                order=1,
            )
        )

        # 添加收益分析
        self._add_section(
            ReportSection(
                section_id="return_analysis",
                title="Return Analysis",
                content_type="table",
                content=self._format_return_analysis(performance_data),
                order=2,
            )
        )

        # 添加风险分析
        self._add_section(
            ReportSection(
                section_id="risk_analysis",
                title="Risk Analysis",
                content_type="table",
                content=self._format_risk_analysis(performance_data),
                order=3,
            )
        )

        # 添加交易分析
        self._add_section(
            ReportSection(
                section_id="trade_analysis",
                title="Trade Analysis",
                content_type="table",
                content=self._format_trade_analysis(metrics),
                order=4,
            )
        )

        return self._render_report(config)

    def create_custom_report(
        self, title: str, sections: List[ReportSection], config: ReportConfig
    ) -> str:
        """创建自定义报告

        Args:
            title: 报告标题
            sections: 章节列表
            config: 报告配置

        Returns:
            报告内容
        """
        self.sections.clear()

        self.metadata = {
            "report_title": title,
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "report_type": "Custom Report",
        }

        for section in sections:
            self._add_section(section)

        return self._render_report(config)

    def export_to_pdf(self, html_content: str, output_path: str) -> bool:
        """导出为PDF

        Args:
            html_content: HTML内容
            output_path: 输出路径

        Returns:
            是否成功
        """
        try:
            options = {
                "page-size": "A4",
                "margin-top": "0.75in",
                "margin-right": "0.75in",
                "margin-bottom": "0.75in",
                "margin-left": "0.75in",
                "encoding": "UTF-8",
                "no-outline": None,
                "enable-local-file-access": None,
            }

            pdfkit.from_string(html_content, output_path, options=options)
            logger.info(f"Report exported to PDF: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export PDF: {e}")
            return False

    def export_to_markdown(self, content: Dict[str, Any], output_path: str) -> bool:
        """导出为Markdown

        Args:
            content: 内容字典
            output_path: 输出路径

        Returns:
            是否成功
        """
        try:
            md_content = self._generate_markdown(content)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Report exported to Markdown: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export Markdown: {e}")
            return False

    def schedule_report_delivery(
        self,
        report_type: str,
        schedule: str,  # 'daily', 'weekly', 'monthly'
        recipients: List[str],
        delivery_method: str = "email",
    ) -> bool:
        """设置报告定时发送

        Args:
            report_type: 报告类型
            schedule: 发送计划
            recipients: 接收者列表
            delivery_method: 发送方式

        Returns:
            是否成功设置
        """
        # 实际实现需要集成任务调度器
        logger.info(f"Scheduled {report_type} report delivery: {schedule}")
        return True

    def _add_section(self, section: ReportSection) -> None:
        """添加报告章节

        Args:
            section: 章节对象
        """
        self.sections[section.section_id] = section

    def _add_summary_section(
        self, portfolio_data: Dict[str, Any], date: datetime
    ) -> None:
        """添加概要章节

        Args:
            portfolio_data: 组合数据
            date: 日期
        """
        summary = {
            "Date": date.strftime("%Y-%m-%d"),
            "Total Value": f"${portfolio_data.get('total_value', 0):,.2f}",
            "Daily P&L": f"${portfolio_data.get('daily_pnl', 0):,.2f}",
            "Daily Return": f"{portfolio_data.get('daily_return', 0):.2%}",
            "YTD Return": f"{portfolio_data.get('ytd_return', 0):.2%}",
        }

        self._add_section(
            ReportSection(
                section_id="summary",
                title="Portfolio Summary",
                content_type="metric",
                content=summary,
                order=0,
            )
        )

    def _add_positions_section(self, positions: List[Position]) -> None:
        """添加持仓章节

        Args:
            positions: 持仓列表
        """
        if not positions:
            return

        positions_data = []
        for pos in positions:
            positions_data.append(
                {
                    "Symbol": pos.symbol,
                    "Quantity": pos.quantity,
                    "Avg Cost": f"${pos.avg_cost:.2f}",
                    "Current Price": f"${pos.current_price:.2f}",
                    "Market Value": f"${pos.market_value:,.2f}",
                    "Unrealized P&L": f"${pos.unrealized_pnl:,.2f}",
                    "Return": f"{pos.return_pct:.2%}",
                }
            )

        df = pd.DataFrame(positions_data)

        self._add_section(
            ReportSection(
                section_id="positions",
                title="Current Positions",
                content_type="table",
                content=df.to_html(classes="table table-striped", index=False),
                order=1,
            )
        )

    def _add_signals_section(self, signals: List[Signal], date: datetime) -> None:
        """添加信号章节

        Args:
            signals: 信号列表
            date: 日期
        """
        # 筛选当日信号
        today_signals = [s for s in signals if s.timestamp.date() == date.date()]

        if not today_signals:
            return

        signals_data = []
        for signal in today_signals:
            signals_data.append(
                {
                    "Time": signal.timestamp.strftime("%H:%M:%S"),
                    "Symbol": signal.symbol,
                    "Action": signal.action,
                    "Quantity": signal.quantity,
                    "Price": f"${signal.price:.2f}",
                    "Confidence": f"{signal.confidence:.2%}",
                    "Strategy": signal.strategy_name,
                }
            )

        df = pd.DataFrame(signals_data)

        self._add_section(
            ReportSection(
                section_id="signals",
                title="Today's Trading Signals",
                content_type="table",
                content=df.to_html(classes="table table-striped", index=False),
                order=2,
            )
        )

    def _add_market_overview_section(self, market_data: pd.DataFrame) -> None:
        """添加市场概览章节

        Args:
            market_data: 市场数据
        """
        # 计算市场统计
        overview = {
            "Market Trend": self._determine_market_trend(market_data),
            "Volatility": f"{market_data['close'].pct_change().std() * np.sqrt(252):.2%}",
            "Top Gainer": self._find_top_mover(market_data, "gainer"),
            "Top Loser": self._find_top_mover(market_data, "loser"),
            "Trading Volume": f"{market_data['volume'].sum():,.0f}",
        }

        self._add_section(
            ReportSection(
                section_id="market_overview",
                title="Market Overview",
                content_type="metric",
                content=overview,
                order=3,
            )
        )

    def _add_performance_section(self, portfolio_data: Dict[str, Any]) -> None:
        """添加绩效章节

        Args:
            portfolio_data: 组合数据
        """
        metrics = {
            "Sharpe Ratio": f"{portfolio_data.get('sharpe_ratio', 0):.2f}",
            "Max Drawdown": f"{portfolio_data.get('max_drawdown', 0):.2%}",
            "Win Rate": f"{portfolio_data.get('win_rate', 0):.2%}",
            "Profit Factor": f"{portfolio_data.get('profit_factor', 0):.2f}",
        }

        self._add_section(
            ReportSection(
                section_id="performance",
                title="Performance Metrics",
                content_type="metric",
                content=metrics,
                order=4,
            )
        )

    def _render_report(self, config: ReportConfig) -> str:
        """渲染报告

        Args:
            config: 报告配置

        Returns:
            报告内容
        """
        # 获取模板
        template = self.env.get_template(config.template_name)

        # 准备数据
        sorted_sections = sorted(self.sections.values(), key=lambda x: x.order)

        context = {
            "metadata": self.metadata,
            "sections": sorted_sections,
            "config": config,
        }

        # 渲染HTML
        html_content = template.render(**context)

        # 根据输出格式转换
        if config.output_format == "html":
            return html_content
        elif config.output_format == "markdown":
            return self._html_to_markdown(html_content)
        elif config.output_format == "json":
            return json.dumps(context, default=str, indent=2)
        else:
            return html_content

    def _format_metrics(self, metrics: PerformanceMetrics) -> Dict[str, str]:
        """格式化绩效指标

        Args:
            metrics: 绩效指标

        Returns:
            格式化后的指标字典
        """
        return {
            "Total Return": f"{metrics.total_return:.2%}",
            "Annualized Return": f"{metrics.annualized_return:.2%}",
            "Volatility": f"{metrics.volatility:.2%}",
            "Sharpe Ratio": f"{metrics.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{metrics.sortino_ratio:.2f}",
            "Max Drawdown": f"{metrics.max_drawdown:.2%}",
            "Win Rate": f"{metrics.win_rate:.2%}",
            "Profit Factor": f"{metrics.profit_factor:.2f}",
            "Total Trades": str(metrics.total_trades),
            "Winning Trades": str(metrics.winning_trades),
            "Losing Trades": str(metrics.losing_trades),
        }

    def _format_weekly_overview(self, weekly_data: Dict[str, Any]) -> str:
        """格式化周概览

        Args:
            weekly_data: 周数据

        Returns:
            格式化的文本
        """
        return f"""
        This week's portfolio performance showed a return of {weekly_data.get("weekly_return", 0):.2%} 
        with {weekly_data.get("total_trades", 0)} trades executed. 
        The portfolio value changed from ${weekly_data.get("start_value", 0):,.2f} 
        to ${weekly_data.get("end_value", 0):,.2f}.
        """

    def _format_performance_table(self, performance: Dict) -> str:
        """格式化绩效表格

        Args:
            performance: 绩效数据

        Returns:
            HTML表格
        """
        df = pd.DataFrame([performance])
        return df.to_html(classes="table table-striped", index=False)

    def _format_trade_statistics(self, trades: List[Dict]) -> str:
        """格式化交易统计

        Args:
            trades: 交易列表

        Returns:
            HTML表格
        """
        df = pd.DataFrame(trades)
        return df.to_html(classes="table table-striped", index=False)

    def _format_return_analysis(self, data: Dict) -> str:
        """格式化收益分析

        Args:
            data: 数据

        Returns:
            HTML表格
        """
        return pd.DataFrame([data]).to_html(classes="table table-striped", index=False)

    def _format_risk_analysis(self, data: Dict) -> str:
        """格式化风险分析

        Args:
            data: 数据

        Returns:
            HTML表格
        """
        return pd.DataFrame([data]).to_html(classes="table table-striped", index=False)

    def _format_trade_analysis(self, metrics: PerformanceMetrics) -> str:
        """格式化交易分析

        Args:
            metrics: 绩效指标

        Returns:
            HTML表格
        """
        trade_stats = {
            "Average Win": f"${metrics.avg_win:.2f}",
            "Average Loss": f"${metrics.avg_loss:.2f}",
            "Best Trade": f"${metrics.best_trade:.2f}",
            "Worst Trade": f"${metrics.worst_trade:.2f}",
            "Win/Loss Ratio": f"{metrics.avg_win / abs(metrics.avg_loss) if metrics.avg_loss != 0 else 0:.2f}",
        }
        return pd.DataFrame([trade_stats]).to_html(
            classes="table table-striped", index=False
        )

    def _determine_market_trend(self, market_data: pd.DataFrame) -> str:
        """判断市场趋势

        Args:
            market_data: 市场数据

        Returns:
            趋势描述
        """
        if len(market_data) < 2:
            return "Unknown"

        returns = market_data["close"].pct_change().mean()
        if returns > 0.01:
            return "Bullish"
        elif returns < -0.01:
            return "Bearish"
        else:
            return "Neutral"

    def _find_top_mover(self, market_data: pd.DataFrame, mover_type: str) -> str:
        """查找涨跌幅最大的标的

        Args:
            market_data: 市场数据
            mover_type: 'gainer' or 'loser'

        Returns:
            标的名称和涨跌幅
        """
        if "symbol" not in market_data.columns:
            return "N/A"

        returns = market_data.groupby("symbol")["close"].pct_change().last()

        if mover_type == "gainer":
            top = returns.idxmax()
            return f"{top} ({returns[top]:.2%})"
        else:
            top = returns.idxmin()
            return f"{top} ({returns[top]:.2%})"

    def _html_to_markdown(self, html_content: str) -> str:
        """HTML转Markdown

        Args:
            html_content: HTML内容

        Returns:
            Markdown内容
        """
        # 简化的转换，实际应使用专门的库
        import html2text

        h = html2text.HTML2Text()
        h.ignore_links = False
        return h.handle(html_content)

    def _generate_markdown(self, content: Dict[str, Any]) -> str:
        """生成Markdown内容

        Args:
            content: 内容字典

        Returns:
            Markdown字符串
        """
        md_lines = []

        # 添加标题
        md_lines.append(f"# {content.get('title', 'Report')}")
        md_lines.append(
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        )
        md_lines.append("")

        # 添加章节
        for section_id, section_content in content.items():
            if section_id == "title":
                continue

            md_lines.append(f"## {section_id.replace('_', ' ').title()}")

            if isinstance(section_content, dict):
                for key, value in section_content.items():
                    md_lines.append(f"- **{key}**: {value}")
            elif isinstance(section_content, list):
                for item in section_content:
                    md_lines.append(f"- {item}")
            else:
                md_lines.append(str(section_content))

            md_lines.append("")

        return "\n".join(md_lines)


# 模块级别函数
def generate_quick_report(data: Dict[str, Any], report_type: str = "daily") -> str:
    """快速生成报告

    Args:
        data: 报告数据
        report_type: 报告类型

    Returns:
        报告内容
    """
    builder = ReportBuilder()
    config = ReportConfig(
        report_type=report_type,
        template_name=ReportBuilder.DEFAULT_TEMPLATES.get(
            report_type, "daily_report.html"
        ),
        output_format="html",
    )

    sections = []
    for idx, (key, value) in enumerate(data.items()):
        sections.append(
            ReportSection(
                section_id=key,
                title=key.replace("_", " ").title(),
                content_type="text" if isinstance(value, str) else "metric",
                content=value,
                order=idx,
            )
        )

    return builder.create_custom_report("Quick Report", sections, config)


def export_report(report_content: str, filename: str, format: str = "html") -> bool:
    """导出报告

    Args:
        report_content: 报告内容
        filename: 文件名
        format: 格式

    Returns:
        是否成功
    """
    try:
        if format == "html":
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content)
        elif format == "pdf":
            builder = ReportBuilder()
            return builder.export_to_pdf(report_content, filename)
        else:
            logger.warning(f"Unsupported format: {format}")
            return False

        logger.info(f"Report exported to {filename}")
        return True

    except Exception as e:
        logger.error(f"Failed to export report: {e}")
        return False
