"""
仪表板管理器模块
负责创建和管理实时监控仪表板
"""

import asyncio
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from common.data_structures import MarketData, Position, Signal
from common.exceptions import QuantSystemError
from common.logging_system import setup_logger
from dash import Input, Output, State, callback_context, dcc, html
from plotly.subplots import make_subplots

logger = setup_logger("dashboard_manager")


@dataclass
class DashboardConfig:
    """仪表板配置数据类"""

    title: str
    refresh_interval_ms: int
    layout: str  # 'grid', 'tabs', 'single'
    theme: str  # 'dark', 'light', 'auto'
    components: List[str]
    auto_refresh: bool = True
    show_header: bool = True
    show_sidebar: bool = True
    responsive: bool = True


@dataclass
class DashboardComponent:
    """仪表板组件数据类"""

    component_id: str
    component_type: str  # 'chart', 'table', 'metric', 'alert'
    title: str
    data_source: str
    update_callback: Optional[Callable] = None
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(
        default_factory=dict
    )  # {'row': 0, 'col': 0, 'width': 6, 'height': 4}


@dataclass
class MetricCard:
    """指标卡片数据类"""

    metric_id: str
    title: str
    value: Union[float, int, str]
    change: Optional[float] = None
    change_period: str = "1d"
    icon: Optional[str] = None
    color: str = "primary"
    format_type: str = "number"  # 'number', 'currency', 'percentage'


class DashboardManager:
    """仪表板管理器类"""

    def __init__(self, config: DashboardConfig):
        """初始化仪表板管理器

        Args:
            config: 仪表板配置
        """
        self.config = config
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.DARKLY if config.theme == "dark" else dbc.themes.BOOTSTRAP
            ],
            suppress_callback_exceptions=True,
        )
        self.components: Dict[str, DashboardComponent] = {}
        self.data_cache: Dict[str, Any] = {}
        self.update_callbacks: Dict[str, Callable] = {}
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None
        self._setup_layout()
        self._setup_callbacks()

    def create_portfolio_dashboard(
        self,
        portfolio_data: Dict[str, Any],
        positions: List[Position],
        signals: List[Signal],
    ) -> dash.Dash:
        """创建投资组合仪表板

        Args:
            portfolio_data: 组合数据
            positions: 持仓列表
            signals: 信号列表

        Returns:
            Dash应用实例
        """
        # 添加组合总览组件
        self.add_component(
            DashboardComponent(
                component_id="portfolio_overview",
                component_type="metric",
                title="Portfolio Overview",
                data_source="portfolio",
                position={"row": 0, "col": 0, "width": 12, "height": 2},
            )
        )

        # 添加持仓表格
        self.add_component(
            DashboardComponent(
                component_id="positions_table",
                component_type="table",
                title="Current Positions",
                data_source="positions",
                position={"row": 2, "col": 0, "width": 6, "height": 4},
            )
        )

        # 添加性能图表
        self.add_component(
            DashboardComponent(
                component_id="performance_chart",
                component_type="chart",
                title="Performance Trend",
                data_source="performance",
                position={"row": 2, "col": 6, "width": 6, "height": 4},
            )
        )

        # 添加信号面板
        self.add_component(
            DashboardComponent(
                component_id="signals_panel",
                component_type="table",
                title="Recent Signals",
                data_source="signals",
                position={"row": 6, "col": 0, "width": 12, "height": 3},
            )
        )

        # 更新数据缓存
        self.data_cache["portfolio"] = portfolio_data
        self.data_cache["positions"] = positions
        self.data_cache["signals"] = signals

        self._refresh_all_components()
        return self.app

    def add_component(self, component: DashboardComponent) -> None:
        """添加仪表板组件

        Args:
            component: 组件对象
        """
        self.components[component.component_id] = component
        logger.info(f"Added dashboard component: {component.component_id}")

    def update_component_data(self, component_id: str, data: Any) -> None:
        """更新组件数据

        Args:
            component_id: 组件ID
            data: 新数据
        """
        if component_id not in self.components:
            logger.warning(f"Component not found: {component_id}")
            return

        self.data_cache[component_id] = data
        if self.is_running:
            self._refresh_component(component_id)

    def start_auto_refresh(self) -> None:
        """启动自动刷新"""
        if not self.config.auto_refresh:
            return

        self.is_running = True
        self.update_thread = threading.Thread(
            target=self._auto_refresh_loop, daemon=True
        )
        self.update_thread.start()
        logger.info("Dashboard auto-refresh started")

    def stop_auto_refresh(self) -> None:
        """停止自动刷新"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("Dashboard auto-refresh stopped")

    def _setup_layout(self) -> None:
        """设置仪表板布局"""
        # 创建导航栏
        navbar = (
            dbc.NavbarSimple(
                brand=self.config.title,
                brand_href="#",
                color="dark" if self.config.theme == "dark" else "light",
                dark=self.config.theme == "dark",
                fluid=True,
            )
            if self.config.show_header
            else None
        )

        # 创建侧边栏
        sidebar = (
            html.Div(
                [
                    html.H2("Controls", className="display-6"),
                    html.Hr(),
                    html.Div(id="sidebar-content"),
                ],
                style={
                    "position": "fixed",
                    "top": "56px",
                    "left": 0,
                    "bottom": 0,
                    "width": "200px",
                    "padding": "20px",
                    "background-color": "#f8f9fa"
                    if self.config.theme == "light"
                    else "#222",
                },
            )
            if self.config.show_sidebar
            else None
        )

        # 创建主内容区域
        content_style = {
            "margin-left": "220px" if self.config.show_sidebar else "0px",
            "margin-right": "20px",
            "padding": "20px",
        }

        content = html.Div(id="main-content", style=content_style)

        # 组装布局
        layout_components = []
        if navbar:
            layout_components.append(navbar)
        if sidebar:
            layout_components.append(sidebar)
        layout_components.append(content)

        # 添加自动刷新组件
        if self.config.auto_refresh:
            layout_components.append(
                dcc.Interval(
                    id="interval-component",
                    interval=self.config.refresh_interval_ms,
                    n_intervals=0,
                )
            )

        self.app.layout = html.Div(layout_components)

    def _setup_callbacks(self) -> None:
        """设置回调函数"""
        if self.config.auto_refresh:

            @self.app.callback(
                Output("main-content", "children"),
                Input("interval-component", "n_intervals"),
            )
            def update_dashboard(n):
                return self._generate_content()

    def _generate_content(self) -> List[Any]:
        """生成仪表板内容

        Returns:
            Dash组件列表
        """
        content = []

        if self.config.layout == "grid":
            # 网格布局
            rows = {}
            for comp_id, component in self.components.items():
                row = component.position.get("row", 0)
                if row not in rows:
                    rows[row] = []
                rows[row].append(self._render_component(component))

            for row_num in sorted(rows.keys()):
                content.append(dbc.Row(rows[row_num], className="mb-3"))

        elif self.config.layout == "tabs":
            # 标签页布局
            tabs = []
            for comp_id, component in self.components.items():
                tabs.append(
                    dbc.Tab(self._render_component(component), label=component.title)
                )
            content.append(dbc.Tabs(tabs))

        else:
            # 单列布局
            for comp_id, component in self.components.items():
                content.append(self._render_component(component))

        return content

    def _render_component(self, component: DashboardComponent) -> Any:
        """渲染单个组件

        Args:
            component: 组件对象

        Returns:
            渲染后的Dash组件
        """
        data = self.data_cache.get(component.data_source)

        if component.component_type == "chart":
            return self._render_chart(component, data)
        elif component.component_type == "table":
            return self._render_table(component, data)
        elif component.component_type == "metric":
            return self._render_metric(component, data)
        elif component.component_type == "alert":
            return self._render_alert(component, data)
        else:
            return html.Div(f"Unknown component type: {component.component_type}")

    def _render_chart(self, component: DashboardComponent, data: Any) -> dcc.Graph:
        """渲染图表组件

        Args:
            component: 组件配置
            data: 数据

        Returns:
            Graph组件
        """
        fig = go.Figure()

        if isinstance(data, pd.DataFrame):
            # 如果是DataFrame，创建线图
            for col in data.columns:
                if col != "timestamp" and col != "date":
                    fig.add_trace(
                        go.Scatter(
                            x=data.index
                            if data.index.name
                            else data.get("timestamp", data.get("date")),
                            y=data[col],
                            mode="lines",
                            name=col,
                        )
                    )

        elif isinstance(data, dict):
            # 如果是字典，根据类型创建图表
            chart_type = component.config.get("chart_type", "line")
            if chart_type == "bar":
                fig = go.Figure(
                    data=[go.Bar(x=list(data.keys()), y=list(data.values()))]
                )
            elif chart_type == "pie":
                fig = go.Figure(
                    data=[go.Pie(labels=list(data.keys()), values=list(data.values()))]
                )
            else:
                fig = go.Figure(
                    data=[go.Scatter(x=list(data.keys()), y=list(data.values()))]
                )

        fig.update_layout(
            title=component.title,
            template="plotly_dark" if self.config.theme == "dark" else "plotly_white",
            height=component.position.get("height", 4) * 100,
        )

        return dcc.Graph(
            id=f"graph-{component.component_id}",
            figure=fig,
            style={"width": f"{component.position.get('width', 6) * 8.33}%"},
        )

    def _render_table(self, component: DashboardComponent, data: Any) -> dbc.Table:
        """渲染表格组件

        Args:
            component: 组件配置
            data: 数据

        Returns:
            Table组件
        """
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, list):
            # 如果是对象列表，转换为DataFrame
            if data and hasattr(data[0], "__dict__"):
                df = pd.DataFrame([item.__dict__ for item in data])
            else:
                df = pd.DataFrame(data)
        else:
            df = pd.DataFrame()

        # 限制显示行数
        max_rows = component.config.get("max_rows", 10)
        if len(df) > max_rows:
            df = df.head(max_rows)

        return dbc.Table.from_dataframe(
            df,
            striped=True,
            bordered=True,
            hover=True,
            dark=self.config.theme == "dark",
            responsive=True,
            id=f"table-{component.component_id}",
        )

    def _render_metric(self, component: DashboardComponent, data: Any) -> dbc.Card:
        """渲染指标卡片

        Args:
            component: 组件配置
            data: 数据

        Returns:
            Card组件
        """
        if isinstance(data, dict):
            cards = []
            for key, value in data.items():
                card = dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4(key, className="card-title"),
                            html.H2(
                                f"{value:,.2f}"
                                if isinstance(value, (int, float))
                                else str(value)
                            ),
                        ]
                    ),
                    style={"width": "200px", "margin": "10px"},
                )
                cards.append(card)
            return dbc.Row(cards)
        else:
            return dbc.Card(
                dbc.CardBody(
                    [
                        html.H4(component.title, className="card-title"),
                        html.H2(str(data)),
                    ]
                )
            )

    def _render_alert(self, component: DashboardComponent, data: Any) -> dbc.Alert:
        """渲染警报组件

        Args:
            component: 组件配置
            data: 数据

        Returns:
            Alert组件
        """
        if isinstance(data, dict):
            color = data.get("color", "info")
            message = data.get("message", "")
            is_open = data.get("is_open", True)
        else:
            color = "info"
            message = str(data)
            is_open = True

        return dbc.Alert(
            message,
            color=color,
            is_open=is_open,
            dismissable=True,
            id=f"alert-{component.component_id}",
        )

    def _refresh_all_components(self) -> None:
        """刷新所有组件"""
        for component_id in self.components:
            self._refresh_component(component_id)

    def _refresh_component(self, component_id: str) -> None:
        """刷新单个组件

        Args:
            component_id: 组件ID
        """
        component = self.components.get(component_id)
        if not component:
            return

        # 执行更新回调
        if component.update_callback:
            try:
                new_data = component.update_callback()
                self.data_cache[component.data_source] = new_data
            except Exception as e:
                logger.error(f"Failed to update component {component_id}: {e}")

    def _auto_refresh_loop(self) -> None:
        """自动刷新循环"""
        import time

        while self.is_running:
            try:
                self._refresh_all_components()
                time.sleep(self.config.refresh_interval_ms / 1000.0)
            except Exception as e:
                logger.error(f"Error in auto-refresh loop: {e}")
                time.sleep(5)

    def run_server(
        self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False
    ) -> None:
        """运行仪表板服务器

        Args:
            host: 主机地址
            port: 端口号
            debug: 是否开启调试模式
        """
        logger.info(f"Starting dashboard server on {host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


# 模块级别函数
def create_default_dashboard() -> DashboardManager:
    """创建默认仪表板

    Returns:
        仪表板管理器实例
    """
    config = DashboardConfig(
        title="Quantum Investment System Dashboard",
        refresh_interval_ms=5000,
        layout="grid",
        theme="dark",
        components=["portfolio", "positions", "performance", "signals"],
    )
    return DashboardManager(config)


def quick_dashboard(data: Dict[str, Any], title: str = "Quick Dashboard") -> None:
    """快速创建并运行仪表板

    Args:
        data: 要显示的数据
        title: 仪表板标题
    """
    config = DashboardConfig(
        title=title,
        refresh_interval_ms=5000,
        layout="single",
        theme="auto",
        components=list(data.keys()),
    )

    manager = DashboardManager(config)

    # 为每个数据项创建组件
    for idx, (key, value) in enumerate(data.items()):
        if isinstance(value, pd.DataFrame):
            comp_type = "table" if len(value) < 100 else "chart"
        elif isinstance(value, dict):
            comp_type = "metric"
        else:
            comp_type = "metric"

        manager.add_component(
            DashboardComponent(
                component_id=key,
                component_type=comp_type,
                title=key.replace("_", " ").title(),
                data_source=key,
                position={"row": idx, "col": 0, "width": 12, "height": 3},
            )
        )

    manager.data_cache = data
    manager.run_server()
