"""
全局模型管理器 - 单例模式
避免重复加载大型模型到内存
"""

from typing import Any, Optional


class ModelManager:
    """全局模型管理器（单例）"""

    _instance = None
    _fin_r1_model = None
    _fin_r1_tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_fin_r1(self, model: Any, tokenizer: Any):
        """设置 FIN-R1 模型实例"""
        self._fin_r1_model = model
        self._fin_r1_tokenizer = tokenizer

    def get_fin_r1(self) -> tuple[Optional[Any], Optional[Any]]:
        """获取 FIN-R1 模型实例"""
        return self._fin_r1_model, self._fin_r1_tokenizer

    def has_fin_r1(self) -> bool:
        """检查是否已加载 FIN-R1 模型"""
        return self._fin_r1_model is not None and self._fin_r1_tokenizer is not None


# 全局单例实例
model_manager = ModelManager()
