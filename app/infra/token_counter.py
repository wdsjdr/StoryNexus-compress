"""TokenCounter 单例：基于 tiktoken 的精确 Token 计数。

决策 3（ADR §12）：必须使用 tiktoken (cl100k_base)，拒绝字符估算。
预留 backend="huggingface" 以便未来精确对接 DeepSeek 官方 tokenizer。
相同文本段落只计一次，结果缓存在内存 dict 中。
"""
from __future__ import annotations

import threading

import tiktoken


class TokenCounter:
    """线程安全的 Token 计数单例。"""

    _instance: "TokenCounter | None" = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "TokenCounter":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, backend: str = "tiktoken") -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._backend = backend
        self._cache: dict[str, int] = {}
        self._encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def backend(self) -> str:
        return self._backend

    def count(self, text: str) -> int:
        """统计 token 数；相同文本命中内存缓存。"""
        if not text:
            return 0
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        n = len(self._encoding.encode(text))
        # 简单内存上限，避免无限膨胀
        if len(self._cache) < 100_000:
            self._cache[text] = n
        return n

    def count_many(self, texts: list[str]) -> int:
        return sum(self.count(t) for t in texts)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """按 token 数截断文本（保留前缀）。max_tokens <= 0 返回空串。"""
        if max_tokens <= 0 or not text:
            return ""
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])

    def clear_cache(self) -> None:
        self._cache.clear()


def get_counter() -> TokenCounter:
    return TokenCounter()
