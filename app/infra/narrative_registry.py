"""叙事资产登记表（M12 冷库）：伏笔 ForeshadowRegistry + 阵营 FactionRegistry。

冷库概念：远古章节不整体进上下文，关键叙事资产（伏笔/阵营）登记在此，
需要时按出场人物/主题检索（装配时自动注入 + 工具族查询）。

- 伏笔：自动登记自 slice profile 的"约定/承诺"事实（"约好七年之后见面"），
  人工可标记回收；主题词提取自约定文本中的 motif/时间词。
- 阵营：自动登记自"加入/结盟/联合"类事实与势力实体。

存储：data/data/meta/{novel_id}_narrative.json（线程安全 + 原子写）。
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.infra.storage import AtomicFileWriter


@dataclass
class ForeshadowEntry:
    id: str
    novel_id: str
    subject: str  # 涉及人物
    keyword: str  # 主题词（约定文本中的线索词）
    planted_chapter: int
    summary: str = ""
    status: str = "open"  # open | paid_off
    paid_off_chapter: int = 0
    source: str = "heuristic"  # heuristic | llm | manual

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "novel_id": self.novel_id,
            "subject": self.subject,
            "keyword": self.keyword,
            "planted_chapter": self.planted_chapter,
            "summary": self.summary,
            "status": self.status,
            "paid_off_chapter": self.paid_off_chapter,
            "source": self.source,
        }


@dataclass
class FactionEntry:
    id: str
    novel_id: str
    name: str
    members: list[str] = field(default_factory=list)
    first_seen_chapter: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "novel_id": self.novel_id,
            "name": self.name,
            "members": list(self.members),
            "first_seen_chapter": self.first_seen_chapter,
            "note": self.note,
        }


class NarrativeRegistry:
    """伏笔 + 阵营登记表（JSON 持久化，线程安全）。"""

    def __init__(self, meta_dir: str | Path) -> None:
        self._dir = Path(meta_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._foreshadows: dict[str, ForeshadowEntry] = {}
        self._factions: dict[str, FactionEntry] = {}
        self._load()

    def _path(self, novel_id: str) -> Path:
        return self._dir / f"{novel_id}_narrative.json"

    def _load(self) -> None:
        # 所有作品的登记文件统一加载（文件量小）
        for p in self._dir.glob("*_narrative.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for f in data.get("foreshadows", []):
                    self._foreshadows[f["id"]] = ForeshadowEntry(**f)
                for g in data.get("factions", []):
                    self._factions[g["id"]] = FactionEntry(**g)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    def _save(self, novel_id: str) -> None:
        payload = {
            "foreshadows": [
                f.to_dict() for f in self._foreshadows.values()
                if f.novel_id == novel_id
            ],
            "factions": [
                g.to_dict() for g in self._factions.values()
                if g.novel_id == novel_id
            ],
        }
        AtomicFileWriter.write_text(
            self._path(novel_id),
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    # ── 伏笔 ──
    def register_foreshadow(
        self,
        novel_id: str,
        *,
        subject: str,
        keyword: str,
        chapter: int,
        summary: str = "",
        source: str = "heuristic",
    ) -> ForeshadowEntry | None:
        """登记一条伏笔。同 (subject, keyword) 已开放则去重。"""
        if not subject or not keyword:
            return None
        with self._lock:
            for f in self._foreshadows.values():
                if (
                    f.novel_id == novel_id
                    and f.subject == subject
                    and f.keyword == keyword
                    and f.status == "open"
                ):
                    return None
            entry = ForeshadowEntry(
                id=uuid.uuid4().hex[:10],
                novel_id=novel_id,
                subject=subject,
                keyword=keyword,
                planted_chapter=chapter,
                summary=summary[:80],
                source=source,
            )
            self._foreshadows[entry.id] = entry
            self._save(novel_id)
            return entry

    def mark_paid_off(
        self, novel_id: str, fs_id: str, *, chapter: int = 0
    ) -> bool:
        with self._lock:
            f = self._foreshadows.get(fs_id)
            if f is None or f.novel_id != novel_id:
                return False
            f.status = "paid_off"
            f.paid_off_chapter = chapter
            self._save(novel_id)
            return True

    def list_foreshadows(
        self, novel_id: str, *, status: str | None = None
    ) -> list[dict]:
        with self._lock:
            out = [
                f.to_dict() for f in self._foreshadows.values()
                if f.novel_id == novel_id
                and (status is None or f.status == status)
            ]
        return sorted(out, key=lambda d: d["planted_chapter"])

    def open_for_subjects(self, novel_id: str, subjects: list[str]) -> list[dict]:
        """出场人物的开放伏笔（装配时自动注入数据源）。"""
        if not subjects:
            return []
        with self._lock:
            out = [
                f.to_dict() for f in self._foreshadows.values()
                if f.novel_id == novel_id
                and f.status == "open"
                and f.subject in subjects
            ]
        return sorted(out, key=lambda d: d["planted_chapter"])

    # ── 阵营 ──
    def register_faction(
        self,
        novel_id: str,
        *,
        name: str,
        member: str = "",
        chapter: int = 0,
    ) -> FactionEntry:
        """登记/更新阵营（幂等：同名阵营并集成员）。"""
        with self._lock:
            existing = next(
                (g for g in self._factions.values()
                 if g.novel_id == novel_id and g.name == name),
                None,
            )
            if existing is None:
                existing = FactionEntry(
                    id=uuid.uuid4().hex[:10],
                    novel_id=novel_id,
                    name=name,
                    first_seen_chapter=chapter,
                )
                self._factions[existing.id] = existing
            if member and member not in existing.members:
                existing.members.append(member)
            if chapter and (existing.first_seen_chapter == 0 or chapter < existing.first_seen_chapter):
                existing.first_seen_chapter = chapter
            self._save(novel_id)
            return existing

    def list_factions(self, novel_id: str) -> list[dict]:
        with self._lock:
            out = [
                g.to_dict() for g in self._factions.values()
                if g.novel_id == novel_id
            ]
        return sorted(out, key=lambda d: d["first_seen_chapter"])


def extract_promise_keyword(quote: str, motifs: set[str]) -> str:
    """从约定文本提取主题词：优先 motif 词，其次时间/数字词，最后前 8 字。

    P1/P4: 兼容英文（promise/vow/swear 的西幻原文）与翻译体时间词。
    """
    for m in motifs:
        if m in quote:
            return m
    import re

    m = re.search(
        r"第[0-9零一二三四五六七八九十百千]+次|七年|三年|五年|十年|明天|下次|到时候|"
        r"来年|明年|春天|夏天|秋天|冬天|"
        r"\b(?:next\s+(?:spring|summer|autumn|winter|year)|in\s+\d+\s+years)\b",
        quote,
    )
    if m:
        return m.group(0)
    return quote[:8]
