"""
可视化模块初始化文件
提供图表生成、仪表板、报告生成等可视化功能
"""

from module_11_visualization.chart_generator import (
    CandlestickData,
    ChartConfig,
    ChartGenerator,
    export_chart_collection,
    quick_chart,
)
from module_11_visualization.dashboard_manager import (
    DashboardComponent,
    DashboardConfig,
    DashboardManager,
    MetricCard,
    create_default_dashboard,
    quick_dashboard,
)
from module_11_visualization.database_manager import (
    VisualizationDatabaseManager,
    get_visualization_database_manager,
)
from module_11_visualization.export_manager import (
    ExportConfig,
    ExportManager,
    ExportResult,
    ExportTask,
    create_data_backup,
    export_with_compression,
    quick_export,
)
from module_11_visualization.interactive_visualizer import (
    InteractionEvent,
    InteractiveConfig,
    InteractiveVisualizer,
    create_interactive_chart,
    export_all_charts,
)
from module_11_visualization.report_builder import (
    PerformanceMetrics,
    ReportBuilder,
    ReportConfig,
    ReportSection,
    export_report_data,
    generate_quick_report,
)
from module_11_visualization.template_engine import (
    RenderResult,
    TemplateConfig,
    TemplateEngine,
    TemplateVariable,
    create_report_from_template,
    get_default_engine,
    render_quick_template,
)

__all__ = [
    # Chart Generator
    "ChartGenerator",
    "ChartConfig",
    "CandlestickData",
    "quick_chart",
    "export_chart_collection",
    # Dashboard Manager
    "DashboardManager",
    "DashboardConfig",
    "DashboardComponent",
    "MetricCard",
    "create_default_dashboard",
    "quick_dashboard",
    # Database Manager
    "VisualizationDatabaseManager",
    "get_visualization_database_manager",
    # Export Manager
    "ExportManager",
    "ExportConfig",
    "ExportTask",
    "ExportResult",
    "quick_export",
    "export_with_compression",
    "create_data_backup",
    # Interactive Visualizer
    "InteractiveVisualizer",
    "InteractiveConfig",
    "InteractionEvent",
    "create_interactive_chart",
    "export_all_charts",
    # Report Builder
    "ReportBuilder",
    "ReportConfig",
    "ReportSection",
    "PerformanceMetrics",
    "generate_quick_report",
    "export_report_data",
    # Template Engine
    "TemplateEngine",
    "TemplateConfig",
    "TemplateVariable",
    "RenderResult",
    "get_default_engine",
    "render_quick_template",
    "create_report_from_template",
]

# 版本信息
__version__ = "1.0.0"
__author__ = "FinLoom Team"
__description__ = "Visualization module for FinLoom quantitative trading system"
