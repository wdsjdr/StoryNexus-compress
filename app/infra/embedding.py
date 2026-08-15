"""Embedding 服务接口（CSA 兜底 B）。

默认 NullEmbedding（返回零向量，兜底路径无结果），可在环境变量
STORYNEXUS_EMBEDDING 切换：
  - "fastembed": 本地 FastEmbed 模型（BAAI/bge-small-zh-v1.5，首次调用下载）
  - "hash": 确定性字符哈希嵌入（无依赖、可复现，测试/离线环境）
B 兜底仅当 A（事实三元组）召回不足时触发。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol


class EmbeddingService(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def similarity(self, a: list[float], b: list[float]) -> float: ...


class NullEmbedding:
    """零向量实现：不产生任何召回结果（纯接口占位）。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def similarity(self, a: list[float], b: list[float]) -> float:
        return 0.0


class FastEmbedService:
    """本地 FastEmbed 实现（惰性加载，首次调用下载模型）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure()
        return [vec.tolist() for vec in model.embed(texts)]

    def similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


class HashEmbedding:
    """确定性字符哈希嵌入（M10 兜底 B 的无依赖实现）。

    用 MD5 对每个字符做稳定哈希落到 dim 维桶，L2 归一化。
    相似文本（共享字符 n-gram）产生近似向量，可在无模型环境下
    验证向量召回与时序排序链路；FastEmbed 可无缝替换。
    """

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs: list[list[float]] = []
        for text in texts:
            v = [0.0] * self._dim
            for ch in text:
                digest = hashlib.md5(ch.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:2], "big") % self._dim
                v[bucket] += 1.0
            norm = sum(x * x for x in v) ** 0.5
            if norm > 0:
                v = [x / norm for x in v]
            vecs.append(v)
        return vecs

    def similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return dot  # 双方均已 L2 归一化


def get_embedding_service() -> EmbeddingService:
    """M14: 环境变量优先（运行期查询，测试可控）← settings.json 覆盖默认。"""
    from app.config import settings

    backend = os.getenv("STORYNEXUS_EMBEDDING") or (settings.embedding or "null")
    if backend == "fastembed":
        return FastEmbedService()
    if backend == "hash":
        return HashEmbedding()
    return NullEmbedding()
