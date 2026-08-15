"""最小运行配置（压缩引擎独立包裁剪版）。

主仓 storynexus 的 app/config.py 含 LLM/搜索/数据路径全套配置；
独立压缩包仅需 embedding 后端切换（STORYNEXUS_EMBEDDING）。
"""
from __future__ import annotations

import os


class Settings:
    """压缩引擎运行设置：仅 embedding 后端（null|hash|fastembed）。"""

    def __init__(self) -> None:
        self.embedding: str = os.getenv("STORYNEXUS_EMBEDDING", "null")


settings = Settings()
