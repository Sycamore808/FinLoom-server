"""
导出管理器模块
负责管理各种格式的数据和报告导出
"""

import csv
import json
import os
import pickle
import tarfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, TextIO, Union

import h5py
import numpy as np
import openpyxl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from common.exceptions import QuantSystemError
from common.logging_system import setup_logger
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

logger = setup_logger("export_manager")


@dataclass
class ExportConfig:
    """导出配置数据类"""

    export_format: str  # 'csv', 'excel', 'json', 'parquet', 'hdf5', 'pdf', 'html'
    output_path: str
    compression: Optional[str] = None  # 'gzip', 'bz2', 'zip', 'xz'
    encoding: str = "utf-8"
    include_metadata: bool = True
    include_index: bool = False
    chunk_size: Optional[int] = None
    append_mode: bool = False


@dataclass
class ExportTask:
    """导出任务数据类"""

    task_id: str
    task_type: str  # 'data', 'report', 'model', 'config'
    source_data: Any
    config: ExportConfig
    status: str = "pending"  # 'pending', 'running', 'completed', 'failed'
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class ExportResult:
    """导出结果数据类"""

    success: bool
    file_path: str
    file_size: int
    export_time: datetime
    rows_exported: Optional[int] = None
    compression_ratio: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExportManager:
    """导出管理器类"""

    SUPPORTED_FORMATS = {
        "csv": [".csv"],
        "excel": [".xlsx", ".xls"],
        "json": [".json"],
        "parquet": [".parquet", ".pq"],
        "hdf5": [".h5", ".hdf5"],
        "pdf": [".pdf"],
        "html": [".html", ".htm"],
        "pickle": [".pkl", ".pickle"],
        "feather": [".feather"],
        "stata": [".dta"],
        "spss": [".sav"],
    }

    COMPRESSION_EXTENSIONS = {
        "gzip": ".gz",
        "bz2": ".bz2",
        "zip": ".zip",
        "xz": ".xz",
        "tar": ".tar",
    }

    def __init__(self, default_output_dir: str = "exports"):
        """初始化导出管理器

        Args:
            default_output_dir: 默认输出目录
        """
        self.default_output_dir = Path(default_output_dir)
        self.default_output_dir.mkdir(parents=True, exist_ok=True)

        self.export_tasks: Dict[str, ExportTask] = {}
        self.export_history: List[ExportResult] = []
        self.max_history_size = 1000

    def export_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        format: str = "csv",
        config: Optional[ExportConfig] = None,
    ) -> ExportResult:
        """导出DataFrame

        Args:
            df: 数据框
            filename: 文件名
            format: 格式
            config: 导出配置

        Returns:
            导出结果
        """
        if config is None:
            config = ExportConfig(
                export_format=format,
                output_path=str(self.default_output_dir / filename),
            )

        start_time = datetime.now()

        try:
            # 确保输出目录存在
            output_path = Path(config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 根据格式导出
            if format == "csv":
                self._export_csv(df, output_path, config)
            elif format == "excel":
                self._export_excel(df, output_path, config)
            elif format == "json":
                self._export_json(df, output_path, config)
            elif format == "parquet":
                self._export_parquet(df, output_path, config)
            elif format == "hdf5":
                self._export_hdf5(df, output_path, config)
            elif format == "pickle":
                self._export_pickle(df, output_path, config)
            elif format == "feather":
                self._export_feather(df, output_path, config)
            elif format == "stata":
                df.to_stata(output_path, write_index=config.include_index)
            elif format == "html":
                self._export_html(df, output_path, config)
            else:
                raise ValueError(f"Unsupported format: {format}")

            # 应用压缩
            if config.compression:
                output_path = self._compress_file(output_path, config.compression)

            # 获取文件信息
            file_size = output_path.stat().st_size

            result = ExportResult(
                success=True,
                file_path=str(output_path),
                file_size=file_size,
                export_time=datetime.now(),
                rows_exported=len(df),
                metadata={
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict(),
                },
            )

            # 保存历史
            self._add_to_history(result)

            logger.info(f"Exported {len(df)} rows to {output_path}")
            return result

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                file_path="",
                file_size=0,
                export_time=datetime.now(),
                metadata={"error": str(e)},
            )

    def export_multiple(
        self,
        data_dict: Dict[str, pd.DataFrame],
        base_filename: str,
        format: str = "excel",
        create_archive: bool = True,
    ) -> ExportResult:
        """导出多个数据集

        Args:
            data_dict: 数据字典 {sheet_name: dataframe}
            base_filename: 基础文件名
            format: 格式
            create_archive: 是否创建归档

        Returns:
            导出结果
        """
        start_time = datetime.now()
        exported_files = []

        try:
            if format == "excel":
                # Excel支持多个工作表
                output_path = self.default_output_dir / f"{base_filename}.xlsx"
                with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                    for sheet_name, df in data_dict.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

                        # 格式化工作表
                        worksheet = writer.sheets[sheet_name]
                        self._format_excel_sheet(worksheet, df)

                exported_files.append(output_path)

            else:
                # 其他格式需要分别导出
                for name, df in data_dict.items():
                    filename = f"{base_filename}_{name}.{format}"
                    result = self.export_dataframe(df, filename, format)
                    if result.success:
                        exported_files.append(Path(result.file_path))

            # 创建归档
            if create_archive and len(exported_files) > 1:
                archive_path = self._create_archive(
                    exported_files, f"{base_filename}_archive.zip"
                )
                final_path = archive_path
                file_size = archive_path.stat().st_size
            else:
                final_path = exported_files[0] if exported_files else None
                file_size = final_path.stat().st_size if final_path else 0

            return ExportResult(
                success=True,
                file_path=str(final_path) if final_path else "",
                file_size=file_size,
                export_time=datetime.now(),
                rows_exported=sum(len(df) for df in data_dict.values()),
                metadata={"datasets": list(data_dict.keys())},
            )

        except Exception as e:
            logger.error(f"Multiple export failed: {e}")
            return ExportResult(
                success=False,
                file_path="",
                file_size=0,
                export_time=datetime.now(),
                metadata={"error": str(e)},
            )

    def export_report(
        self,
        report_content: str,
        filename: str,
        format: str = "pdf",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExportResult:
        """导出报告

        Args:
            report_content: 报告内容
            filename: 文件名
            format: 格式
            metadata: 元数据

        Returns:
            导出结果
        """
        try:
            output_path = self.default_output_dir / filename

            if format == "pdf":
                self._export_pdf_report(report_content, output_path, metadata)
            elif format == "html":
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report_content)
            elif format == "markdown":
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report_content)
            else:
                raise ValueError(f"Unsupported report format: {format}")

            file_size = output_path.stat().st_size

            return ExportResult(
                success=True,
                file_path=str(output_path),
                file_size=file_size,
                export_time=datetime.now(),
                metadata=metadata or {},
            )

        except Exception as e:
            logger.error(f"Report export failed: {e}")
            return ExportResult(
                success=False,
                file_path="",
                file_size=0,
                export_time=datetime.now(),
                metadata={"error": str(e)},
            )

    def export_model(
        self,
        model: Any,
        model_name: str,
        format: str = "pickle",
        include_metadata: bool = True,
    ) -> ExportResult:
        """导出模型

        Args:
            model: 模型对象
            model_name: 模型名称
            format: 格式
            include_metadata: 是否包含元数据

        Returns:
            导出结果
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.{format}"
            output_path = self.default_output_dir / "models" / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "pickle":
                with open(output_path, "wb") as f:
                    pickle.dump(model, f)

            elif format == "joblib":
                import joblib

                joblib.dump(model, output_path)

            elif format == "onnx":
                # 需要模型支持ONNX导出
                if hasattr(model, "to_onnx"):
                    model.to_onnx(output_path)
                else:
                    raise ValueError("Model does not support ONNX export")

            else:
                raise ValueError(f"Unsupported model format: {format}")

            # 保存元数据
            if include_metadata:
                metadata = {
                    "model_name": model_name,
                    "model_type": type(model).__name__,
                    "export_time": datetime.now().isoformat(),
                    "format": format,
                }

                # 添加模型特定元数据
                if hasattr(model, "get_params"):
                    metadata["params"] = model.get_params()

                metadata_path = output_path.with_suffix(".json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

            file_size = output_path.stat().st_size

            return ExportResult(
                success=True,
                file_path=str(output_path),
                file_size=file_size,
                export_time=datetime.now(),
                metadata={"model_type": type(model).__name__},
            )

        except Exception as e:
            logger.error(f"Model export failed: {e}")
            return ExportResult(
                success=False,
                file_path="",
                file_size=0,
                export_time=datetime.now(),
                metadata={"error": str(e)},
            )

    def create_backup(
        self,
        data_sources: List[str],
        backup_name: Optional[str] = None,
        compression: str = "zip",
    ) -> ExportResult:
        """创建备份

        Args:
            data_sources: 数据源路径列表
            backup_name: 备份名称
            compression: 压缩格式

        Returns:
            导出结果
        """
        try:
            if backup_name is None:
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            backup_dir = self.default_output_dir / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)

            # 收集文件
            files_to_backup = []
            for source in data_sources:
                source_path = Path(source)
                if source_path.exists():
                    if source_path.is_file():
                        files_to_backup.append(source_path)
                    elif source_path.is_dir():
                        files_to_backup.extend(source_path.rglob("*"))

            # 创建归档
            if compression == "zip":
                archive_path = backup_dir / f"{backup_name}.zip"
                with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file in files_to_backup:
                        if file.is_file():
                            zf.write(file, file.relative_to(Path.cwd()))

            elif compression == "tar":
                archive_path = backup_dir / f"{backup_name}.tar.gz"
                with tarfile.open(archive_path, "w:gz") as tf:
                    for file in files_to_backup:
                        if file.is_file():
                            tf.add(file, file.relative_to(Path.cwd()))

            else:
                raise ValueError(f"Unsupported compression: {compression}")

            file_size = archive_path.stat().st_size

            return ExportResult(
                success=True,
                file_path=str(archive_path),
                file_size=file_size,
                export_time=datetime.now(),
                metadata={
                    "files_backed_up": len(files_to_backup),
                    "compression": compression,
                },
            )

        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return ExportResult(
                success=False,
                file_path="",
                file_size=0,
                export_time=datetime.now(),
                metadata={"error": str(e)},
            )

    def schedule_export(
        self, task: ExportTask, run_at: Optional[datetime] = None
    ) -> str:
        """调度导出任务

        Args:
            task: 导出任务
            run_at: 运行时间

        Returns:
            任务ID
        """
        self.export_tasks[task.task_id] = task

        if run_at is None:
            # 立即执行
            self._execute_task(task)
        else:
            # 调度执行（需要集成调度器）
            logger.info(f"Scheduled export task {task.task_id} at {run_at}")

        return task.task_id

    def get_export_history(self, limit: int = 100) -> List[ExportResult]:
        """获取导出历史

        Args:
            limit: 限制数量

        Returns:
            导出结果列表
        """
        return self.export_history[-limit:]

    def clean_old_exports(self, days: int = 30, pattern: Optional[str] = None) -> int:
        """清理旧导出文件

        Args:
            days: 保留天数
            pattern: 文件模式

        Returns:
            清理的文件数
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0

        for file_path in self.default_output_dir.rglob(pattern or "*"):
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_date:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")

        logger.info(f"Cleaned {cleaned_count} old export files")
        return cleaned_count

    def _export_csv(
        self, df: pd.DataFrame, output_path: Path, config: ExportConfig
    ) -> None:
        """导出CSV文件"""
        df.to_csv(
            output_path,
            index=config.include_index,
            encoding=config.encoding,
            compression=config.compression,
            chunksize=config.chunk_size,
        )

    def _export_excel(
        self, df: pd.DataFrame, output_path: Path, config: ExportConfig
    ) -> None:
        """导出Excel文件"""
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=config.include_index)

            # 格式化工作表
            worksheet = writer.sheets["Sheet1"]
            self._format_excel_sheet(worksheet, df)

    def _export_json(
        self, df: pd.DataFrame, output_path: Path, config: ExportConfig
    ) -> None:
        """导出JSON文件"""
        df.to_json(
            output_path,
            orient="records",
            date_format="iso",
            indent=2,
            compression=config.compression,
        )

    def _export_parquet(
        self, df: pd.DataFrame, output_path: Path, config: ExportConfig
    ) -> None:
        """导出Parquet文件"""
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression=config.compression or "snappy")

    def _export_hdf5(
        self, df: pd.DataFrame, output_path: Path, config: ExportConfig
    ) -> None:
        """导出HDF5文件"""
        df.to_hdf(
            output_path,
            key="data",
            mode="w" if not config.append_mode else "a",
            complevel=9 if config.compression else 0,
        )

    def _export_pickle(
        self, df: pd.DataFrame, output_path: Path, config: ExportConfig
    ) -> None:
        """导出Pickle文件"""
        df.to_pickle(output_path, compression=config.compression)

    def _export_feather(
        self, df: pd.DataFrame, output_path: Path, config: ExportConfig
    ) -> None:
        """导出Feather文件"""
        df.to_feather(output_path)

    def _export_html(
        self, df: pd.DataFrame, output_path: Path, config: ExportConfig
    ) -> None:
        """导出HTML文件"""
        html = df.to_html(
            index=config.include_index,
            classes="table table-striped table-bordered",
            table_id="data_table",
        )

        # 添加样式和脚本
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Export</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
            <script src="https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js"></script>
            <script src="https://cdn.datatables.net/1.10.24/js/dataTables.bootstrap4.min.js"></script>
            <link rel="stylesheet" href="https://cdn.datatables.net/1.10.24/css/dataTables.bootstrap4.min.css">
        </head>
        <body>
            <div class="container mt-5">
                <h2>Data Export</h2>
                {html}
            </div>
            <script>
                $(document).ready(function() {{
                    $('#data_table').DataTable();
                }});
            </script>
        </body>
        </html>
        """

        with open(output_path, "w", encoding=config.encoding) as f:
            f.write(full_html)

    def _export_pdf_report(
        self, content: str, output_path: Path, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """导出PDF报告"""
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # 添加标题
        if metadata and "title" in metadata:
            story.append(Paragraph(metadata["title"], styles["Title"]))
            story.append(Spacer(1, 12))

        # 添加内容
        for line in content.split("\n"):
            if line.strip():
                story.append(Paragraph(line, styles["Normal"]))
                story.append(Spacer(1, 6))

        doc.build(story)

    def _format_excel_sheet(self, worksheet: Any, df: pd.DataFrame) -> None:
        """格式化Excel工作表"""
        # 设置列宽
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass

            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

        # 设置表头样式
        header_font = Font(bold=True)
        header_fill = PatternFill(
            start_color="CCCCCC", end_color="CCCCCC", fill_type="solid"
        )
        header_alignment = Alignment(horizontal="center", vertical="center")

        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment

    def _compress_file(self, file_path: Path, compression: str) -> Path:
        """压缩文件"""
        if compression == "gzip":
            import gzip

            compressed_path = file_path.with_suffix(file_path.suffix + ".gz")
            with open(file_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    f_out.writelines(f_in)

        elif compression == "zip":
            compressed_path = file_path.with_suffix(".zip")
            with zipfile.ZipFile(compressed_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(file_path, file_path.name)

        elif compression == "bz2":
            import bz2

            compressed_path = file_path.with_suffix(file_path.suffix + ".bz2")
            with open(file_path, "rb") as f_in:
                with bz2.open(compressed_path, "wb") as f_out:
                    f_out.writelines(f_in)

        else:
            raise ValueError(f"Unsupported compression: {compression}")

        # 删除原文件
        file_path.unlink()

        return compressed_path

    def _create_archive(self, files: List[Path], archive_name: str) -> Path:
        """创建归档文件"""
        archive_path = self.default_output_dir / archive_name

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in files:
                if file.exists():
                    zf.write(file, file.name)

        return archive_path

    def _execute_task(self, task: ExportTask) -> None:
        """执行导出任务"""
        task.status = "running"

        try:
            if task.task_type == "data":
                result = self.export_dataframe(
                    task.source_data,
                    Path(task.config.output_path).name,
                    task.config.export_format,
                    task.config,
                )
            elif task.task_type == "report":
                result = self.export_report(
                    task.source_data,
                    Path(task.config.output_path).name,
                    task.config.export_format,
                )
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            task.status = "completed" if result.success else "failed"
            task.completed_at = datetime.now()

            if not result.success:
                task.error_message = result.metadata.get("error", "Unknown error")

        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            task.completed_at = datetime.now()

    def _add_to_history(self, result: ExportResult) -> None:
        """添加到历史记录"""
        self.export_history.append(result)

        # 限制历史大小
        if len(self.export_history) > self.max_history_size:
            self.export_history.pop(0)


# 模块级别函数
def quick_export(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    filename: str,
    format: str = "csv",
) -> bool:
    """快速导出数据

    Args:
        data: 数据
        filename: 文件名
        format: 格式

    Returns:
        是否成功
    """
    manager = ExportManager()

    if isinstance(data, pd.DataFrame):
        result = manager.export_dataframe(data, filename, format)
    else:
        result = manager.export_multiple(data, filename.split(".")[0], format)

    return result.success


def export_with_compression(
    df: pd.DataFrame, filename: str, compression: str = "gzip"
) -> str:
    """导出并压缩数据

    Args:
        df: 数据框
        filename: 文件名
        compression: 压缩格式

    Returns:
        文件路径
    """
    manager = ExportManager()
    config = ExportConfig(
        export_format="csv", output_path=filename, compression=compression
    )

    result = manager.export_dataframe(df, filename, "csv", config)
    return result.file_path if result.success else ""


def create_data_backup(data_dirs: List[str], backup_name: Optional[str] = None) -> str:
    """创建数据备份

    Args:
        data_dirs: 数据目录列表
        backup_name: 备份名称

    Returns:
        备份文件路径
    """
    manager = ExportManager()
    result = manager.create_backup(data_dirs, backup_name)
    return result.file_path if result.success else ""
