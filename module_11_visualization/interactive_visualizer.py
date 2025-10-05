"""
交互式可视化器模块
负责创建交互式数据可视化组件
"""

import asyncio
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from bokeh.layouts import column, gridplot, row
from bokeh.models import ColumnDataSource, HoverTool, Range1d
from bokeh.palettes import Category20
from bokeh.plotting import figure, output_file, save
from plotly.subplots import make_subplots

from common.data_structures import MarketData, Position, Signal
from common.exceptions import QuantSystemError
from common.logging_system import setup_logger

logger = setup_logger("interactive_visualizer")


@dataclass
class InteractiveConfig:
    """交互式配置数据类"""

    enable_zoom: bool = True
    enable_pan: bool = True
    enable_hover: bool = True
    enable_selection: bool = True
    enable_crosshair: bool = True
    enable_rangeselector: bool = True
    enable_annotations: bool = True
    animation_duration: int = 500
    default_height: int = 600
    default_width: int = 1200
    theme: str = "dark"


@dataclass
class ChartElement:
    """图表元素数据类"""

    element_id: str
    element_type: str  # 'trace', 'annotation', 'shape', 'indicator'
    data: Any
    layout_update: Optional[Dict[str, Any]] = None
    row: Optional[int] = None
    col: Optional[int] = None
    secondary_y: bool = False


@dataclass
class InteractionEvent:
    """交互事件数据类"""

    event_type: str  # 'click', 'hover', 'select', 'zoom', 'pan'
    timestamp: datetime
    element_id: str
    data_point: Dict[str, Any]
    user_data: Optional[Dict[str, Any]] = None


class InteractiveVisualizer:
    """交互式可视化器类"""

    CHART_THEMES = {
        "dark": {
            "template": "plotly_dark",
            "bg_color": "#111111",
            "grid_color": "#333333",
            "text_color": "#FFFFFF",
        },
        "light": {
            "template": "plotly_white",
            "bg_color": "#FFFFFF",
            "grid_color": "#E0E0E0",
            "text_color": "#000000",
        },
        "seaborn": {
            "template": "seaborn",
            "bg_color": "#F5F5F5",
            "grid_color": "#CCCCCC",
            "text_color": "#333333",
        },
    }

    INDICATOR_CONFIGS = {
        "rsi": {"period": 14, "overbought": 70, "oversold": 30},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "bollinger": {"period": 20, "std_dev": 2},
        "stochastic": {"k_period": 14, "d_period": 3, "smooth_k": 3},
    }

    def __init__(self, config: Optional[InteractiveConfig] = None):
        """初始化交互式可视化器

        Args:
            config: 交互配置
        """
        self.config = config or InteractiveConfig()
        self.theme_config = self.CHART_THEMES.get(
            self.config.theme, self.CHART_THEMES["dark"]
        )
        self.figures: Dict[str, go.Figure] = {}
        self.elements: Dict[str, List[ChartElement]] = {}
        self.event_handlers: Dict[str, List[Callable]] = {
            "click": [],
            "hover": [],
            "select": [],
            "zoom": [],
            "pan": [],
        }
        self.animation_queue: List[Dict[str, Any]] = []
        self.is_animating = False

    def create_interactive_candlestick(
        self,
        df: pd.DataFrame,
        symbol: str,
        indicators: Optional[List[str]] = None,
        overlays: Optional[List[str]] = None,
        volume: bool = True,
        annotations: Optional[List[Dict[str, Any]]] = None,
    ) -> go.Figure:
        """创建交互式K线图

        Args:
            df: OHLCV数据
            symbol: 标的代码
            indicators: 技术指标列表
            overlays: 叠加指标列表
            volume: 是否显示成交量
            annotations: 注释列表

        Returns:
            交互式K线图
        """
        # 验证数据
        required_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            raise QuantSystemError(f"Missing required columns: {required_cols}")

        # 计算子图布局
        rows = 1
        row_heights = [1.0]
        subplot_titles = [f"{symbol} Price"]

        if volume and "volume" in df.columns:
            rows += 1
            row_heights = [0.7, 0.3]
            subplot_titles.append("Volume")

        if indicators:
            for indicator in indicators:
                if indicator.lower() in ["rsi", "stochastic", "williams"]:
                    rows += 1
                    if rows == 2:
                        row_heights = [0.6, 0.4]
                    elif rows == 3:
                        row_heights = [0.5, 0.25, 0.25]
                    else:
                        row_heights = [0.4] + [0.2] * (rows - 1)
                    subplot_titles.append(indicator.upper())

        # 创建子图
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )

        # 添加K线
        candlestick = go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#00CC96",
            decreasing_line_color="#EF553B",
        )
        fig.add_trace(candlestick, row=1, col=1)

        # 添加叠加指标
        if overlays:
            for overlay in overlays:
                overlay_data = self._calculate_overlay(df, overlay)
                if overlay_data is not None:
                    for trace in overlay_data:
                        fig.add_trace(trace, row=1, col=1)

        # 添加成交量
        current_row = 2
        if volume and "volume" in df.columns:
            colors = [
                "#00CC96" if c >= o else "#EF553B"
                for c, o in zip(df["close"], df["open"])
            ]

            volume_bar = go.Bar(
                x=df.index,
                y=df["volume"],
                name="Volume",
                marker_color=colors,
                showlegend=False,
            )
            fig.add_trace(volume_bar, row=current_row, col=1)
            current_row += 1

        # 添加独立指标
        if indicators:
            for indicator in indicators:
                if indicator.lower() in ["rsi", "stochastic", "williams"]:
                    indicator_data = self._calculate_indicator(df, indicator)
                    if indicator_data is not None:
                        for trace in indicator_data:
                            fig.add_trace(trace, row=current_row, col=1)

                        # 添加超买超卖线
                        if indicator.lower() == "rsi":
                            fig.add_hline(
                                y=70,
                                line_dash="dash",
                                line_color="red",
                                opacity=0.5,
                                row=current_row,
                                col=1,
                            )
                            fig.add_hline(
                                y=30,
                                line_dash="dash",
                                line_color="green",
                                opacity=0.5,
                                row=current_row,
                                col=1,
                            )
                        current_row += 1

        # 添加注释
        if annotations:
            for annotation in annotations:
                fig.add_annotation(annotation)

        # 更新布局
        fig.update_layout(
            title=f"{symbol} Interactive Chart",
            template=self.theme_config["template"],
            height=self.config.default_height,
            width=self.config.default_width,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            dragmode="zoom" if self.config.enable_zoom else "pan",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
                if self.config.theme == "dark"
                else "rgba(255,255,255,0.8)",
            ),
        )

        # 添加范围选择器
        if self.config.enable_rangeselector:
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=1, label="1D", step="day", stepmode="backward"),
                            dict(count=5, label="5D", step="day", stepmode="backward"),
                            dict(
                                count=1, label="1M", step="month", stepmode="backward"
                            ),
                            dict(
                                count=3, label="3M", step="month", stepmode="backward"
                            ),
                            dict(
                                count=6, label="6M", step="month", stepmode="backward"
                            ),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(step="all", label="All"),
                        ]
                    ),
                    bgcolor="rgba(0,0,0,0.5)"
                    if self.config.theme == "dark"
                    else "rgba(255,255,255,0.8)",
                    activecolor="#00CC96",
                ),
                row=1,
                col=1,
            )

        # 保存图表
        fig_id = f"candlestick_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.figures[fig_id] = fig

        return fig

    def create_heatmap_matrix(
        self,
        data: pd.DataFrame,
        title: str = "Correlation Matrix",
        color_scale: str = "RdBu",
        show_values: bool = True,
        clustering: bool = False,
    ) -> go.Figure:
        """创建热力图矩阵

        Args:
            data: 数据矩阵
            title: 标题
            color_scale: 颜色方案
            show_values: 是否显示数值
            clustering: 是否进行层次聚类

        Returns:
            热力图
        """
        # 层次聚类重排
        if clustering and len(data) > 2:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import pdist

            # 计算距离矩阵和聚类
            distances = pdist(data.values)
            linkage_matrix = linkage(distances, method="ward")
            dendro = dendrogram(linkage_matrix, no_plot=True)

            # 重新排序
            order = dendro["leaves"]
            data = data.iloc[order, order]

        # 创建hover文本
        hover_text = []
        for i in range(len(data)):
            hover_text.append([])
            for j in range(len(data.columns)):
                hover_text[-1].append(
                    f"{data.index[i]} vs {data.columns[j]}<br>"
                    + f"Value: {data.iloc[i, j]:.3f}"
                )

        # 创建热力图
        fig = go.Figure(
            data=go.Heatmap(
                z=data.values,
                x=data.columns.tolist(),
                y=data.index.tolist(),
                colorscale=color_scale,
                zmid=0 if color_scale == "RdBu" else None,
                text=data.values.round(2) if show_values else None,
                texttemplate="%{text}" if show_values else None,
                textfont={"size": 10},
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=hover_text,
                colorbar=dict(
                    title=dict(text="Value", side="right"), thickness=15, len=0.7
                ),
            )
        )

        # 更新布局
        fig.update_layout(
            title=title,
            template=self.theme_config["template"],
            height=self.config.default_height,
            width=self.config.default_width,
            xaxis=dict(tickangle=-45, showgrid=False, side="bottom"),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )

        # 保存图表
        fig_id = f"heatmap_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.figures[fig_id] = fig

        return fig

    def create_3d_surface(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        title: str = "3D Surface",
        colorscale: str = "Viridis",
        show_contours: bool = True,
    ) -> go.Figure:
        """创建3D曲面图

        Args:
            x: X轴数据
            y: Y轴数据
            z: Z轴数据矩阵
            title: 标题
            colorscale: 颜色方案
            show_contours: 是否显示等高线

        Returns:
            3D曲面图
        """
        # 创建曲面
        surface = go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="Value", thickness=20, len=0.7),
            contours=dict(
                z=dict(
                    show=show_contours,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project=dict(z=True),
                )
            )
            if show_contours
            else None,
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
        )

        # 创建图表
        fig = go.Figure(data=[surface])

        # 更新布局
        fig.update_layout(
            title=title,
            template=self.theme_config["template"],
            scene=dict(
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                zaxis_title="Z Axis",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), center=dict(x=0, y=0, z=0)),
                aspectmode="cube",
            ),
            height=700,
            width=900,
        )

        # 保存图表
        fig_id = f"surface3d_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.figures[fig_id] = fig

        return fig

    def create_network_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        title: str = "Network Graph",
        layout_type: str = "spring",
        node_size_field: Optional[str] = None,
        node_color_field: Optional[str] = None,
    ) -> go.Figure:
        """创建网络图

        Args:
            nodes: 节点列表 [{'id': str, 'label': str, 'x': float, 'y': float, ...}]
            edges: 边列表 [{'source': str, 'target': str, 'weight': float, ...}]
            title: 标题
            layout_type: 布局类型 ('spring', 'circular', 'random', 'grid')
            node_size_field: 节点大小字段
            node_color_field: 节点颜色字段

        Returns:
            网络图
        """
        import networkx as nx

        # 创建网络图对象
        G = nx.Graph()

        # 添加节点
        for node in nodes:
            G.add_node(node["id"], **node)

        # 添加边
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1.0))

        # 计算布局
        if layout_type == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout_type == "circular":
            pos = nx.circular_layout(G)
        elif layout_type == "random":
            pos = nx.random_layout(G)
        else:  # grid
            pos = nx.spectral_layout(G)

        # 提取节点位置
        node_x = []
        node_y = []
        node_text = []
        node_ids = []

        for node_id in G.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            node_data = G.nodes[node_id]
            node_text.append(node_data.get("label", node_id))
            node_ids.append(node_id)

        # 创建边轨迹
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace["x"] += (x0, x1, None)
            edge_trace["y"] += (y0, y1, None)

        # 节点大小
        if node_size_field:
            node_sizes = [
                G.nodes[node_id].get(node_size_field, 10) for node_id in node_ids
            ]
        else:
            # 基于度数
            node_sizes = [10 + 5 * G.degree(node_id) for node_id in node_ids]

        # 节点颜色
        if node_color_field:
            node_colors = [
                G.nodes[node_id].get(node_color_field, 0) for node_id in node_ids
            ]
        else:
            # 基于度数
            node_colors = [G.degree(node_id) for node_id in node_ids]

        # 创建节点轨迹
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hovertemplate="Node: %{text}<br>Connections: %{marker.color}<extra></extra>",
            marker=dict(
                showscale=True,
                colorscale="YlOrRd",
                size=node_sizes,
                color=node_colors,
                colorbar=dict(
                    thickness=15, title="Connections", xanchor="left", titleside="right"
                ),
                line=dict(width=2, color="white"),
            ),
        )

        # 创建图表
        fig = go.Figure(data=[edge_trace, node_trace])

        # 更新布局
        fig.update_layout(
            title=title,
            template=self.theme_config["template"],
            showlegend=False,
            hovermode="closest",
            height=self.config.default_height,
            width=self.config.default_width,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        # 保存图表
        fig_id = f"network_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.figures[fig_id] = fig

        return fig

    def create_animated_timeline(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_columns: List[str],
        title: str = "Animated Timeline",
        animation_speed: int = 100,
    ) -> go.Figure:
        """创建动画时间线

        Args:
            df: 数据框
            date_column: 日期列名
            value_columns: 数值列名列表
            title: 标题
            animation_speed: 动画速度（毫秒）

        Returns:
            动画图表
        """
        # 排序数据
        df = df.sort_values(date_column)

        # 创建帧
        frames = []
        for i in range(len(df)):
            frame_data = []
            for col in value_columns:
                frame_data.append(
                    go.Scatter(
                        x=df[date_column].iloc[: i + 1],
                        y=df[col].iloc[: i + 1],
                        mode="lines+markers",
                        name=col,
                        line=dict(width=2),
                    )
                )
            frames.append(go.Frame(data=frame_data, name=str(i)))

        # 创建初始图表
        fig = go.Figure(
            data=[
                go.Scatter(x=[], y=[], mode="lines+markers", name=col)
                for col in value_columns
            ],
            frames=frames,
        )

        # 添加播放按钮
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=animation_speed, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                    x=0.1,
                    y=1.1,
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(
                            args=[
                                [f.name],
                                dict(
                                    frame=dict(duration=animation_speed, redraw=True),
                                    mode="immediate",
                                ),
                            ],
                            label=str(df[date_column].iloc[i])[:10],
                            method="animate",
                        )
                        for i, f in enumerate(frames)
                    ],
                    x=0.1,
                    len=0.9,
                    xanchor="left",
                    y=0,
                    yanchor="top",
                )
            ],
        )

        # 更新布局
        fig.update_layout(
            title=title,
            template=self.theme_config["template"],
            height=self.config.default_height,
            width=self.config.default_width,
            showlegend=True,
            hovermode="x unified",
        )

        # 设置坐标轴范围
        fig.update_xaxes(range=[df[date_column].min(), df[date_column].max()])
        y_min = min(df[col].min() for col in value_columns)
        y_max = max(df[col].max() for col in value_columns)
        fig.update_yaxes(range=[y_min * 0.9, y_max * 1.1])

        # 保存图表
        fig_id = f"timeline_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.figures[fig_id] = fig

        return fig

    def register_event_handler(
        self, event_type: str, handler: Callable[[InteractionEvent], None]
    ) -> None:
        """注册事件处理器

        Args:
            event_type: 事件类型
            handler: 处理函数
        """
        if event_type not in self.event_handlers:
            logger.warning(f"Unknown event type: {event_type}")
            return

        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type} event")

    def export_interactive_html(
        self,
        fig: go.Figure,
        filename: str,
        include_plotlyjs: str = "cdn",
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """导出交互式HTML

        Args:
            fig: 图表对象
            filename: 文件名
            include_plotlyjs: JS包含方式 ('cdn', 'inline', 'directory')
            config: Plotly配置

        Returns:
            是否成功
        """
        try:
            if config is None:
                config = {
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "chart",
                        "height": 600,
                        "width": 1200,
                        "scale": 2,
                    },
                }

            fig.write_html(filename, include_plotlyjs=include_plotlyjs, config=config)

            logger.info(f"Exported interactive chart to {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to export interactive HTML: {e}")
            return False

    def _calculate_overlay(
        self, df: pd.DataFrame, overlay_type: str
    ) -> List[go.Scatter]:
        """计算叠加指标

        Args:
            df: 价格数据
            overlay_type: 叠加类型

        Returns:
            图表轨迹列表
        """
        traces = []

        if overlay_type.lower() == "ma":
            # 移动平均线
            for period in [20, 50, 200]:
                if len(df) >= period:
                    ma = df["close"].rolling(window=period).mean()
                    traces.append(
                        go.Scatter(
                            x=df.index,
                            y=ma,
                            mode="lines",
                            name=f"MA{period}",
                            line=dict(width=1),
                            visible="legendonly",
                        )
                    )

        elif overlay_type.lower() == "bollinger":
            # 布林带
            config = self.INDICATOR_CONFIGS["bollinger"]
            period = config["period"]
            std_dev = config["std_dev"]

            if len(df) >= period:
                sma = df["close"].rolling(window=period).mean()
                std = df["close"].rolling(window=period).std()

                traces.append(
                    go.Scatter(
                        x=df.index,
                        y=sma + std_dev * std,
                        mode="lines",
                        name="BB Upper",
                        line=dict(color="rgba(250,128,114,0.5)", width=1),
                        visible="legendonly",
                    )
                )

                traces.append(
                    go.Scatter(
                        x=df.index,
                        y=sma,
                        mode="lines",
                        name="BB Middle",
                        line=dict(color="rgba(250,128,114,0.5)", width=1, dash="dash"),
                        visible="legendonly",
                    )
                )

                traces.append(
                    go.Scatter(
                        x=df.index,
                        y=sma - std_dev * std,
                        mode="lines",
                        name="BB Lower",
                        line=dict(color="rgba(250,128,114,0.5)", width=1),
                        fill="tonexty",
                        fillcolor="rgba(250,128,114,0.1)",
                        visible="legendonly",
                    )
                )

        elif overlay_type.lower() == "ema":
            # 指数移动平均
            for period in [12, 26]:
                if len(df) >= period:
                    ema = df["close"].ewm(span=period, adjust=False).mean()
                    traces.append(
                        go.Scatter(
                            x=df.index,
                            y=ema,
                            mode="lines",
                            name=f"EMA{period}",
                            line=dict(width=1),
                            visible="legendonly",
                        )
                    )

        return traces

    def _calculate_indicator(
        self, df: pd.DataFrame, indicator_type: str
    ) -> List[go.Scatter]:
        """计算技术指标

        Args:
            df: 价格数据
            indicator_type: 指标类型

        Returns:
            图表轨迹列表
        """
        traces = []

        if indicator_type.lower() == "rsi":
            # RSI指标
            config = self.INDICATOR_CONFIGS["rsi"]
            period = config["period"]

            if len(df) >= period:
                delta = df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                traces.append(
                    go.Scatter(
                        x=df.index,
                        y=rsi,
                        mode="lines",
                        name="RSI",
                        line=dict(color="orange", width=2),
                    )
                )

        elif indicator_type.lower() == "macd":
            # MACD指标
            config = self.INDICATOR_CONFIGS["macd"]

            if len(df) >= config["slow"]:
                exp1 = df["close"].ewm(span=config["fast"], adjust=False).mean()
                exp2 = df["close"].ewm(span=config["slow"], adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=config["signal"], adjust=False).mean()
                histogram = macd - signal

                traces.append(
                    go.Scatter(
                        x=df.index,
                        y=macd,
                        mode="lines",
                        name="MACD",
                        line=dict(color="blue", width=2),
                    )
                )

                traces.append(
                    go.Scatter(
                        x=df.index,
                        y=signal,
                        mode="lines",
                        name="Signal",
                        line=dict(color="red", width=1),
                    )
                )

                traces.append(
                    go.Bar(
                        x=df.index, y=histogram, name="Histogram", marker_color="gray"
                    )
                )

        elif indicator_type.lower() == "stochastic":
            # 随机指标
            config = self.INDICATOR_CONFIGS["stochastic"]

            if len(df) >= config["k_period"]:
                low_min = df["low"].rolling(window=config["k_period"]).min()
                high_max = df["high"].rolling(window=config["k_period"]).max()

                k_percent = 100 * ((df["close"] - low_min) / (high_max - low_min))
                k_percent = k_percent.rolling(window=config["smooth_k"]).mean()
                d_percent = k_percent.rolling(window=config["d_period"]).mean()

                traces.append(
                    go.Scatter(
                        x=df.index,
                        y=k_percent,
                        mode="lines",
                        name="%K",
                        line=dict(color="blue", width=2),
                    )
                )

                traces.append(
                    go.Scatter(
                        x=df.index,
                        y=d_percent,
                        mode="lines",
                        name="%D",
                        line=dict(color="red", width=1, dash="dash"),
                    )
                )

        return traces


# 模块级别函数
def create_interactive_chart(
    data: pd.DataFrame, chart_type: str = "candlestick", **kwargs
) -> go.Figure:
    """快速创建交互式图表

    Args:
        data: 数据
        chart_type: 图表类型
        **kwargs: 其他参数

    Returns:
        交互式图表
    """
    visualizer = InteractiveVisualizer()

    if chart_type == "candlestick":
        return visualizer.create_interactive_candlestick(
            data,
            kwargs.get("symbol", "STOCK"),
            indicators=kwargs.get("indicators"),
            overlays=kwargs.get("overlays"),
            volume=kwargs.get("volume", True),
        )
    elif chart_type == "heatmap":
        return visualizer.create_heatmap_matrix(
            data,
            title=kwargs.get("title", "Heatmap"),
            show_values=kwargs.get("show_values", True),
        )
    elif chart_type == "network":
        return visualizer.create_network_graph(
            kwargs.get("nodes", []),
            kwargs.get("edges", []),
            title=kwargs.get("title", "Network"),
        )
    elif chart_type == "timeline":
        return visualizer.create_animated_timeline(
            data,
            kwargs.get("date_column", "date"),
            kwargs.get("value_columns", data.columns.tolist()),
            title=kwargs.get("title", "Timeline"),
        )
    else:
        # 默认创建线图
        fig = go.Figure()
        for col in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[col], mode="lines", name=col))
        fig.update_layout(title=kwargs.get("title", "Chart"), template="plotly_dark")
        return fig


def export_all_charts(
    charts: Dict[str, go.Figure], output_dir: str, format: str = "html"
) -> bool:
    """批量导出图表

    Args:
        charts: 图表字典
        output_dir: 输出目录
        format: 格式

    Returns:
        是否全部成功
    """
    os.makedirs(output_dir, exist_ok=True)
    visualizer = InteractiveVisualizer()

    success = True
    for name, fig in charts.items():
        filename = os.path.join(output_dir, f"{name}.{format}")
        if format == "html":
            if not visualizer.export_interactive_html(fig, filename):
                success = False
        else:
            try:
                if format in ["png", "jpg", "svg", "pdf"]:
                    fig.write_image(filename, format=format)
                else:
                    logger.warning(f"Unsupported format: {format}")
                    success = False
            except Exception as e:
                logger.error(f"Failed to export {name}: {e}")
                success = False

    return success
