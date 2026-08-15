"""原子文件写入（压缩引擎独立包裁剪版）。

主仓 storynexus 的 app/infra/storage.py 含完整 NovelRepository/Parquet 存储；
独立压缩包仅需要 AtomicFileWriter（narrative_registry 持久化伏笔/阵营用）。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


class AtomicFileWriter:
    """临时文件 + os.replace 的原子写。"""

    @staticmethod
    def write_text(path: Path, content: str) -> None:
        AtomicFileWriter._atomic(path, content.encode("utf-8"))

    @staticmethod
    def write_bytes(path: Path, content: bytes) -> None:
        AtomicFileWriter._atomic(path, content)

    @staticmethod
    def _atomic(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
