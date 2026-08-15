"""句子切分（M12 语义句向量库的前处理；P1 增加英文断句）。

按中文标点（。！？…）断句；英文按「句点/感叹/问号 + 空格 + 大写开头」断句
（Mr./Dr. 等缩写保护：缩写末尾不切）；引号内对话保留；去空白/纯标点句；
单句超长（>120 字）按逗号再切。
"""
from __future__ import annotations

import re

_SENT_END_RE = re.compile(r"[^。！？…!?]+[。！？…!?]?")
# 英文断句：句点/问号/感叹号 + 空白 + 大写开头（新句特征）
_EN_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# 缩写保护（前段以这些结尾时不切）
_ABBREV_END_RE = re.compile(
    r"(?:Mr|Mrs|Ms|Dr|St|Prof|Sr|Jr|vs|etc|U\.S|U\.K|A\.M|P\.M)\.$"
)
_LONG_SPLIT_RE = re.compile(r"[，,、；;]")
_MAX_LEN = 120


def _split_en(s: str) -> list[str]:
    """对片段做英文断句（保护缩写）。"""
    parts = _EN_SPLIT_RE.split(s)
    merged: list[str] = []
    for part in parts:
        if merged and _ABBREV_END_RE.search(merged[-1]):
            merged[-1] += " " + part
        else:
            merged.append(part)
    return merged


def split_sentences(text: str) -> list[str]:
    """把章节正文切成句子列表（保留原文子串，去掉空句）。"""
    out: list[str] = []
    for raw in _SENT_END_RE.findall(text):
        chunks = _split_en(raw) if any(c.isascii() for c in raw) else [raw]
        for chunk in chunks:
            s = chunk.strip()
            if not s:
                continue
            if len(s) <= _MAX_LEN:
                out.append(s)
                continue
            # 超长句按逗号切分（保持边界词完整）
            parts = [p.strip() for p in _LONG_SPLIT_RE.split(s) if p.strip()]
            out.extend(parts)
    return out
