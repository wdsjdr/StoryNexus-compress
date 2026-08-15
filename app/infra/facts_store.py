"""事实落盘：与正文/记忆原子提交的 facts JSON。

决策 3（M6）：在事务提交时将 FactTriple 列表写入
    /data/facts/{novel_id}/ch_{chapter_no:04d}.json
与正文、记忆原子提交。重建索引时遍历这些 JSON 确定性重建，不调用 LLM。
"""
from __future__ import annotations

import json
from pathlib import Path

from app.domain.models.timeline import FactTriple
from app.infra.storage import AtomicFileWriter


class FactsStore:
    """facts JSON 读写（金丝雀重建索引的数据源）。"""

    def __init__(self, facts_dir: str | Path) -> None:
        self._dir = Path(facts_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        novel_id: str,
        chapter_no: int,
        facts: list[FactTriple],
        *,
        summary_5ch: str = "",
        status: str = "extracted",
    ) -> Path:
        path = self._fact_path(novel_id, chapter_no)
        AtomicFileWriter.write_text(
            path,
            json.dumps(
                {
                    "novel_id": novel_id,
                    "chapter_no": chapter_no,
                    "facts": [f.model_dump() for f in facts],
                    "summary_5ch": summary_5ch,
                    "status": status,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return path

    def load_summary(self, novel_id: str, chapter_no: int) -> str:
        """读取某章的 5 章窗口摘要（无则空串）。"""
        path = self._fact_path(novel_id, chapter_no)
        if not path.exists():
            return ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data.get("summary_5ch", "") or "")
        except json.JSONDecodeError:
            return ""

    def latest_summary(self, novel_id: str) -> str:
        """取最新章节的摘要（若存在）。"""
        chapters = self.load_all(novel_id)
        for no in sorted(chapters.keys(), reverse=True):
            summary = self.load_summary(novel_id, no)
            if summary:
                return summary
        return ""

    def load(self, novel_id: str, chapter_no: int) -> list[FactTriple]:
        path = self._fact_path(novel_id, chapter_no)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [FactTriple.model_validate(f) for f in data.get("facts", [])]
        except (json.JSONDecodeError, ValueError):
            return []

    def load_all(self, novel_id: str) -> dict[int, list[FactTriple]]:
        """按章号升序读取全部事实（重建索引用）。"""
        result: dict[int, list[FactTriple]] = {}
        for path in sorted((self._dir / novel_id).glob("ch_*.json")):
            try:
                no = int(path.stem.split("_")[-1])
            except ValueError:
                continue
            result[no] = self.load(novel_id, no)
        return result

    def _fact_path(self, novel_id: str, chapter_no: int) -> Path:
        return self._dir / novel_id / f"ch_{chapter_no:04d}.json"
