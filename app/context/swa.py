"""SWA 滑动窗口：当前章 + 近 N 章原文，不做压缩。

模拟 DeepSeek-V4 局部窗口的 100% 注意力。窗口大小由
novel.yaml 的 context_budget.swa_window_tokens 约束，超限时丢最老章。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.infra.token_counter import get_counter


@dataclass
class SwaWindow:
    chapters: list[str]  # 按章号升序的原文列表
    chapter_nos: list[int]
    token_count: int


class SwaAssembler:
    """从章节存储中选择滑动窗口内的原文。"""

    def __init__(self, window_chapters: int = 3) -> None:
        self._window_chapters = window_chapters

    def build(
        self,
        current_chapter_no: int,
        chapter_store: dict[int, str],
        *,
        max_tokens: int = 32000,
    ) -> SwaWindow:
        """选择 (current-window+1 .. current) 章原文，超 token 上限时丢最老章。"""
        counter = get_counter()
        selected: list[tuple[int, str]] = []
        total = 0
        # 从当前章往前取
        for no in range(current_chapter_no, max(0, current_chapter_no - self._window_chapters), -1):
            text = chapter_store.get(no)
            if text is None:
                continue
            n = counter.count(text)
            if total + n > max_tokens and selected:
                break  # 丢最老章（保留已选的更近章节）
            selected.append((no, text))
            total += n
        selected.reverse()
        return SwaWindow(
            chapters=[t for _, t in selected],
            chapter_nos=[no for no, _ in selected],
            token_count=total,
        )
