"""M2 测试：SWA 滑动窗口。"""
from __future__ import annotations

from app.context.swa import SwaAssembler


class TestSwa:
    def test_window_selection(self):
        store = {48: "章48", 49: "章49", 50: "章50", 47: "章47"}
        w = SwaAssembler(window_chapters=3).build(50, store)
        assert w.chapter_nos == [48, 49, 50]

    def test_missing_chapter_skipped(self):
        store = {49: "章49", 50: "章50"}
        w = SwaAssembler(window_chapters=3).build(50, store)
        assert w.chapter_nos == [49, 50]

    def test_token_cap_drops_oldest(self):
        long_zh = "这是一段非常非常长的中文章节正文内容用于超出预算校核" * 10
        store = {
            50: long_zh,  # 最老，超预算被丢弃
            51: "短章51",
            52: "短章52",
        }
        w = SwaAssembler(window_chapters=3).build(52, store, max_tokens=30)
        assert w.chapter_nos == [51, 52]
