# -*- coding: utf-8 -*-
"""
文件存储管理模块
支持本地文件的读写、序列化等
"""

import json
import os
from typing import Any

import pandas as pd

from common.exceptions import DataError
from common.logging_system import setup_logger

logger = setup_logger("file_storage")


class FileStorage:
    """
    文件存储管理，支持csv、json、parquet等格式
    """

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_csv(self, df: pd.DataFrame, filename: str) -> None:
        try:
            path = os.path.join(self.base_dir, filename)
            df.to_csv(path, index=False)
            logger.info(f"Saved CSV: {path}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {filename}, error: {e}")
            raise DataError(f"Save CSV failed: {e}")

    def load_csv(self, filename: str) -> pd.DataFrame:
        try:
            path = os.path.join(self.base_dir, filename)
            df = pd.read_csv(path)
            logger.info(f"Loaded CSV: {path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV: {filename}, error: {e}")
            raise DataError(f"Load CSV failed: {e}")

    def save_json(self, obj: Any, filename: str) -> None:
        try:
            path = os.path.join(self.base_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved JSON: {path}")
        except Exception as e:
            logger.error(f"Failed to save JSON: {filename}, error: {e}")
            raise DataError(f"Save JSON failed: {e}")

    def load_json(self, filename: str) -> Any:
        try:
            path = os.path.join(self.base_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            logger.info(f"Loaded JSON: {path}")
            return obj
        except Exception as e:
            logger.error(f"Failed to load JSON: {filename}, error: {e}")
            raise DataError(f"Load JSON failed: {e}")
