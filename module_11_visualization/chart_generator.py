"""
图表生成器模块
负责生成各种类型的金融图表
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from common.exceptions import QuantSystemError
from common.logging_system import setup_logger
from plotly.subplots import make_subplots

logger = setup_logger("chart_generator")


@dataclass
class ChartConfig:
    """图表配置数据类"""

    chart_type: str
    title: str
    x_label: str = ""
    y_label: str = ""
    theme: str = "plotly_dark"
    width: int = 1200
    height: int = 600
    show_legend: bool = True
    show_grid: bool = True
    annotations: List[Dict[str, Any]] = None


@dataclass
class CandlestickData:
    """K线数据数据类"""

    timestamps: List[datetime]
    opens: List[float]
    highs: List[float]
    lows: List[float]
    closes: List[float]
    volumes: Optional[List[int]] = None


class ChartGenerator:
    """图表生成器类"""

    CHART_THEMES = {
        "dark": "plotly_dark",
        "light": "plotly_white",
        "presentation": "presentation",
        "ggplot": "ggplot2",
        "seaborn": "seaborn",
    }

    COLOR_SCHEMES = {
        "default": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"],
        "profit_loss": ["#00CC96", "#EF553B"],
        "heatmap": "RdYlGn",
        "diverging": "RdBu",
    }

    def __init__(self, default_theme: str = "dark"):
        """初始化图表生成器

        Args:
            default_theme: 默认主题
        """
        self.default_theme = self.CHART_THEMES.get(default_theme, "plotly_dark")
        self.figure_cache: Dict[str, go.Figure] = {}

    def generate_candlestick_chart(
        self,
        data: Union[CandlestickData, pd.DataFrame],
        indicators: Optional[Dict[str, pd.Series]] = None,
        volume_subplot: bool = True,
        config: Optional[ChartConfig] = None,
    ) -> go.Figure:
        """生成K线图

        Args:
            data: K线数据
            indicators: 技术指标
            volume_subplot: 是否显示成交量子图
            config: 图表配置

        Returns:
            Plotly图表对象
        """
        if config is None:
            config = ChartConfig(
                chart_type="candlestick", title="Price Chart", theme=self.default_theme
            )

        # 转换数据格式
        if isinstance(data, pd.DataFrame):
            candlestick_data = CandlestickData(
                timestamps=data.index.tolist(),
                opens=data["open"].tolist(),
                highs=data["high"].tolist(),
                lows=data["low"].tolist(),
                closes=data["close"].tolist(),
                volumes=data.get("volume", pd.Series()).tolist()
                if "volume" in data
                else None,
            )
        else:
            candlestick_data = data

        # 创建子图
        rows = 2 if volume_subplot and candlestick_data.volumes else 1
        row_heights = [0.7, 0.3] if rows == 2 else [1.0]

        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            subplot_titles=["Price", "Volume"] if rows == 2 else ["Price"],
        )

        # 添加K线图
        fig.add_trace(
            go.Candlestick(
                x=candlestick_data.timestamps,
                open=candlestick_data.opens,
                high=candlestick_data.highs,
                low=candlestick_data.lows,
                close=candlestick_data.closes,
                name="Price",
                increasing_line_color="#00CC96",
                decreasing_line_color="#EF553B",
            ),
            row=1,
            col=1,
        )

        # 添加技术指标
        if indicators:
            for name, values in indicators.items():
                fig.add_trace(
                    go.Scatter(
                        x=candlestick_data.timestamps,
                        y=values,
                        mode="lines",
                        name=name,
                        line=dict(width=1),
                    ),
                    row=1,
                    col=1,
                )

        # 添加成交量
        if volume_subplot and candlestick_data.volumes:
            colors = [
                "#00CC96" if c >= o else "#EF553B"
                for c, o in zip(candlestick_data.closes, candlestick_data.opens)
            ]
            fig.add_trace(
                go.Bar(
                    x=candlestick_data.timestamps,
                    y=candlestick_data.volumes,
                    name="Volume",
                    marker_color=colors,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # 更新布局
        fig.update_layout(
            title=config.title,
            template=config.theme,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )

        # 更新坐标轴
        fig.update_xaxes(
            title_text=config.x_label if config.x_label else "Date", row=rows, col=1
        )
        fig.update_yaxes(
            title_text=config.y_label if config.y_label else "Price", row=1, col=1
        )
        if rows == 2:
            fig.update_yaxes(title_text="Volume", row=2, col=1)

        # 添加注释
        if config.annotations:
            for annotation in config.annotations:
                fig.add_annotation(annotation)

        return fig

    def generate_performance_chart(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        config: Optional[ChartConfig] = None,
    ) -> go.Figure:
        """生成收益率曲线图

        Args:
            returns: 收益率序列
            benchmark: 基准收益率
            config: 图表配置

        Returns:
            Plotly图表对象
        """
        if config is None:
            config = ChartConfig(
                chart_type="line",
                title="Cumulative Returns",
                x_label="Date",
                y_label="Cumulative Return (%)",
                theme=self.default_theme,
            )

        fig = go.Figure()

        # 计算累计收益率
        cumulative_returns = (1 + returns).cumprod() - 1

        # 添加策略收益曲线
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=cumulative_returns * 100,
                mode="lines",
                name="Strategy",
                line=dict(color="#00CC96", width=2),
            )
        )

        # 添加基准收益曲线
        if benchmark is not None:
            benchmark_cumulative = (1 + benchmark).cumprod() - 1
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index,
                    y=benchmark_cumulative * 100,
                    mode="lines",
                    name="Benchmark",
                    line=dict(color="#636EFA", width=2),
                )
            )

        # 添加零线
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        # 更新布局
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            template=config.theme,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend,
            hovermode="x unified",
        )

        return fig

    def generate_heatmap(
        self, data: pd.DataFrame, config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """生成热力图

        Args:
            data: 数据矩阵
            config: 图表配置

        Returns:
            Plotly图表对象
        """
        if config is None:
            config = ChartConfig(
                chart_type="heatmap",
                title="Correlation Heatmap",
                theme=self.default_theme,
            )

        fig = go.Figure(
            data=go.Heatmap(
                z=data.values,
                x=data.columns,
                y=data.index,
                colorscale=self.COLOR_SCHEMES["heatmap"],
                colorbar=dict(title="Correlation"),
                text=np.round(data.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )

        fig.update_layout(
            title=config.title,
            template=config.theme,
            width=config.width,
            height=config.height,
            xaxis=dict(tickangle=-45),
        )

        return fig

    def generate_risk_metrics_chart(
        self, metrics: Dict[str, float], config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """生成风险指标图表

        Args:
            metrics: 风险指标字典
            config: 图表配置

        Returns:
            Plotly图表对象
        """
        if config is None:
            config = ChartConfig(
                chart_type="bar", title="Risk Metrics", theme=self.default_theme
            )

        # 准备数据
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        # 根据值的正负设置颜色
        colors = ["#00CC96" if v >= 0 else "#EF553B" for v in metric_values]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    marker_color=colors,
                    text=[f"{v:.2f}" for v in metric_values],
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title=config.title,
            xaxis_title="Metric",
            yaxis_title="Value",
            template=config.theme,
            width=config.width,
            height=config.height,
            showlegend=False,
        )

        return fig

    def generate_portfolio_composition(
        self, positions: Dict[str, float], config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """生成组合构成饼图

        Args:
            positions: 持仓字典 {symbol: weight}
            config: 图表配置

        Returns:
            Plotly图表对象
        """
        if config is None:
            config = ChartConfig(
                chart_type="pie",
                title="Portfolio Composition",
                theme=self.default_theme,
            )

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(positions.keys()),
                    values=list(positions.values()),
                    hole=0.3,
                    textinfo="label+percent",
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title=config.title,
            template=config.theme,
            width=config.width,
            height=config.height,
            showlegend=True,
        )

        return fig

    def generate_drawdown_chart(
        self, returns: pd.Series, config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """生成回撤图

        Args:
            returns: 收益率序列
            config: 图表配置

        Returns:
            Plotly图表对象
        """
        if config is None:
            config = ChartConfig(
                chart_type="area",
                title="Drawdown Analysis",
                x_label="Date",
                y_label="Drawdown (%)",
                theme=self.default_theme,
            )

        # 计算累计收益和回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100

        fig = go.Figure()

        # 添加回撤区域
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                fill="tozeroy",
                mode="lines",
                name="Drawdown",
                line=dict(color="#EF553B", width=1),
                fillcolor="rgba(239, 85, 59, 0.3)",
            )
        )

        # 标记最大回撤
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()

        fig.add_annotation(
            x=max_dd_idx,
            y=max_dd_value,
            text=f"Max DD: {max_dd_value:.2f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#EF553B",
            ax=0,
            ay=-40,
        )

        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            template=config.theme,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend,
            hovermode="x unified",
        )

        return fig

    def generate_multi_asset_comparison(
        self,
        data: Dict[str, pd.Series],
        normalize: bool = True,
        config: Optional[ChartConfig] = None,
    ) -> go.Figure:
        """生成多资产对比图

        Args:
            data: 资产数据字典 {asset_name: price_series}
            normalize: 是否归一化
            config: 图表配置

        Returns:
            Plotly图表对象
        """
        if config is None:
            config = ChartConfig(
                chart_type="line",
                title="Multi-Asset Comparison",
                x_label="Date",
                y_label="Normalized Price" if normalize else "Price",
                theme=self.default_theme,
            )

        fig = go.Figure()

        for asset_name, prices in data.items():
            if normalize:
                normalized = prices / prices.iloc[0] * 100
                y_data = normalized
            else:
                y_data = prices

            fig.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=y_data,
                    mode="lines",
                    name=asset_name,
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            template=config.theme,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend,
            hovermode="x unified",
        )

        return fig

    def save_chart(self, fig: go.Figure, filename: str, format: str = "html") -> bool:
        """保存图表

        Args:
            fig: Plotly图表对象
            filename: 文件名
            format: 格式 ('html', 'png', 'pdf', 'svg')

        Returns:
            是否成功保存
        """
        try:
            if format == "html":
                fig.write_html(filename)
            elif format in ["png", "pdf", "svg"]:
                fig.write_image(filename, format=format)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Chart saved to {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to save chart: {e}")
            return False

    def create_subplot_layout(
        self,
        charts: List[go.Figure],
        rows: int,
        cols: int,
        subplot_titles: Optional[List[str]] = None,
    ) -> go.Figure:
        """创建子图布局

        Args:
            charts: 图表列表
            rows: 行数
            cols: 列数
            subplot_titles: 子图标题

        Returns:
            组合后的图表
        """
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        for idx, chart in enumerate(charts):
            row = idx // cols + 1
            col = idx % cols + 1

            for trace in chart.data:
                fig.add_trace(trace, row=row, col=col)

        fig.update_layout(
            showlegend=True, template=self.default_theme, height=400 * rows
        )

        return fig


# 模块级别函数
def quick_chart(
    data: Union[pd.Series, pd.DataFrame, Dict],
    chart_type: str = "line",
    title: str = "Chart",
) -> go.Figure:
    """快速生成图表

    Args:
        data: 数据
        chart_type: 图表类型
        title: 标题

    Returns:
        Plotly图表对象
    """
    generator = ChartGenerator()
    config = ChartConfig(chart_type=chart_type, title=title)

    if chart_type == "line" and isinstance(data, pd.Series):
        fig = go.Figure(data=go.Scatter(x=data.index, y=data.values, mode="lines"))
    elif chart_type == "bar" and isinstance(data, dict):
        fig = generator.generate_risk_metrics_chart(data, config)
    elif chart_type == "heatmap" and isinstance(data, pd.DataFrame):
        fig = generator.generate_heatmap(data, config)
    elif chart_type == "pie" and isinstance(data, dict):
        fig = generator.generate_portfolio_composition(data, config)
    else:
        fig = go.Figure()

    fig.update_layout(title=title, template="plotly_dark")
    return fig


def export_chart_collection(
    charts: Dict[str, go.Figure], output_dir: str, format: str = "html"
) -> bool:
    """批量导出图表集合

    Args:
        charts: 图表字典
        output_dir: 输出目录
        format: 导出格式

    Returns:
        是否全部成功
    """
    os.makedirs(output_dir, exist_ok=True)
    generator = ChartGenerator()

    success = True
    for name, fig in charts.items():
        filename = os.path.join(output_dir, f"{name}.{format}")
        if not generator.save_chart(fig, filename, format):
            success = False

    return success
