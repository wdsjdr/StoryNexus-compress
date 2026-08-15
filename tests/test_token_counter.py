"""M0 测试：TokenCounter 单例。"""
from __future__ import annotations

from app.infra.token_counter import get_counter


def test_singleton_same_instance():
    assert get_counter() is get_counter()


def test_count_zh_mixed():
    c = get_counter()
    n = c.count("沈烬与白续在废墟首次对峙。chapter 050")
    assert n > 0
    assert isinstance(n, int)


def test_count_cached():
    c = get_counter()
    text = "不可变前缀：作者意图 + 世界观压缩摘要"
    first = c.count(text)
    second = c.count(text)
    assert first == second


def test_empty_string():
    assert get_counter().count("") == 0
