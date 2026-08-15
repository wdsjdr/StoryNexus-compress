"""原子时间线索引（CSA 主干 A）：事实三元组时序倒排索引。

每章提交后，由 LLM 极轻量提炼出该章的「事实三元组 (主语-谓语-宾语)」，
存入 SQLite 并附章节时间戳 (chapter_no)。召回时绝不向量搜索，直接 SQL：
    SELECT * FROM facts WHERE subject=? ORDER BY chapter_no DESC LIMIT 1
保证时序正确（不会把 ch_002 的记忆当作 ch_048 的前置条件）。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from app.domain.models.timeline import FactTriple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fact_triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    novel_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    chapter_no INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_subject_chapter
    ON fact_triples(novel_id, subject, chapter_no DESC);
CREATE INDEX IF NOT EXISTS idx_facts_chapter
    ON fact_triples(novel_id, chapter_no);
CREATE TABLE IF NOT EXISTS chapter_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    novel_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    chapter_no INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    vector TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vectors_subject
    ON chapter_vectors(novel_id, subject, chapter_no);
CREATE TABLE IF NOT EXISTS sentence_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    novel_id TEXT NOT NULL,
    chapter_no INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    text TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sent_vectors_chapter
    ON sentence_vectors(novel_id, chapter_no);
"""


class TimelineIndex:
    """SQLite 事实三元组时序索引。"""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：FastAPI 线程池/测试客户端跨线程访问
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        self._conn.close()

    def add(self, novel_id: str, chapter_no: int, facts: list[FactTriple]) -> None:
        rows = [
            (novel_id, f.subject, f.predicate, f.object, f.chapter_no)
            for f in facts
        ]
        if not rows:
            return
        self._conn.executemany(
            "INSERT INTO fact_triples (novel_id, subject, predicate, object, chapter_no) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def query_by_subjects(
        self,
        novel_id: str,
        subjects: list[str],
        *,
        limit_per_subject: int = 5,
        before_chapter: int | None = None,
        after_chapter: int | None = None,
    ) -> list[FactTriple]:
        """按出场人物召回事实。before_chapter 限定只看当前章之前的记忆；
        after_chapter 限定热区窗口下界（M12：近 50 章窗口化 CSA）。"""
        if not subjects:
            return []
        placeholders = ",".join("?" * len(subjects))
        sql = (
            "SELECT subject, predicate, object, chapter_no FROM fact_triples "
            f"WHERE novel_id=? AND subject IN ({placeholders})"
        )
        params: list = [novel_id, *subjects]
        if before_chapter is not None:
            sql += " AND chapter_no < ?"
            params.append(before_chapter)
        if after_chapter is not None:
            sql += " AND chapter_no >= ?"
            params.append(after_chapter)
        sql += " ORDER BY chapter_no DESC"
        rows = self._conn.execute(sql, params).fetchall()

        result: list[FactTriple] = []
        seen: dict[str, int] = {}
        for row in rows:
            key = f"{row['subject']}|{row['predicate']}"
            if seen.get(key, 0) >= limit_per_subject:
                continue
            seen[key] = seen.get(key, 0) + 1
            result.append(
                FactTriple(
                    subject=row["subject"],
                    predicate=row["predicate"],
                    object=row["object"],
                    chapter_no=row["chapter_no"],
                )
            )
        return result

    def latest_fact(
        self, novel_id: str, subject: str, predicate: str
    ) -> FactTriple | None:
        """最新事实查询（守卫求值/漂移检测用）。"""
        row = self._conn.execute(
            "SELECT subject, predicate, object, chapter_no FROM fact_triples "
            "WHERE novel_id=? AND subject=? AND predicate=? "
            "ORDER BY chapter_no DESC LIMIT 1",
            (novel_id, subject, predicate),
        ).fetchone()
        if row is None:
            return None
        return FactTriple(
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            chapter_no=row["chapter_no"],
        )

    def chapter_facts(self, novel_id: str, chapter_no: int) -> list[FactTriple]:
        rows = self._conn.execute(
            "SELECT subject, predicate, object, chapter_no FROM fact_triples "
            "WHERE novel_id=? AND chapter_no=? ORDER BY id",
            (novel_id, chapter_no),
        ).fetchall()
        return [
            FactTriple(subject=r["subject"], predicate=r["predicate"], object=r["object"], chapter_no=r["chapter_no"])
            for r in rows
        ]

    # ── 金丝雀：抽样与确定性重建 ──
    def sample(self, n: int = 5) -> list[FactTriple]:
        """随机抽取 n 条记录并反序列化，用于完整性校验。损坏返回空。"""
        rows = self._conn.execute(
            "SELECT subject, predicate, object, chapter_no FROM fact_triples "
            "ORDER BY RANDOM() LIMIT ?",
            (n,),
        ).fetchall()
        result: list[FactTriple] = []
        for r in rows:
            try:
                result.append(
                    FactTriple(
                        subject=r["subject"] or "",
                        predicate=r["predicate"] or "",
                        object=r["object"] or "",
                        chapter_no=int(r["chapter_no"]),
                    )
                )
            except (TypeError, ValueError):
                return []
        return result

    # ── M9: 事实修正（人工入口，修正 CSA 索引错误） ──
    def list_facts(self, novel_id: str, *, limit: int = 100) -> list[dict]:
        """列出全部事实（含 id，供修正面板/CLI）。"""
        rows = self._conn.execute(
            "SELECT id, subject, predicate, object, chapter_no FROM fact_triples "
            "WHERE novel_id=? ORDER BY chapter_no DESC LIMIT ?",
            (novel_id, limit),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "chapter_no": r["chapter_no"],
            }
            for r in rows
        ]

    def delete_fact(self, novel_id: str, fact_id: int) -> bool:
        """删除错误事实（如"沈烬拥有断刃"已过时）。"""
        cur = self._conn.execute(
            "DELETE FROM fact_triples WHERE novel_id=? AND id=?", (novel_id, fact_id)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update_fact(self, novel_id: str, fact_id: int, **fields) -> bool:
        """纠正事实字段（subject/predicate/object）。"""
        allowed = {"subject", "predicate", "object"}
        sets = {k: v for k, v in fields.items() if k in allowed and v}
        if not sets:
            return False
        assignments = ", ".join(f"{k}=?" for k in sets)
        cur = self._conn.execute(
            f"UPDATE fact_triples SET {assignments} WHERE novel_id=? AND id=?",
            (*sets.values(), novel_id, fact_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def rebuild(self, novel_id: str, facts_by_chapter: dict[int, list[FactTriple]]) -> int:
        """确定性重建：删除该作品全部索引，从 facts JSON 回插。返回插入条数。

        全程不调用 LLM（金丝雀修复动作），结果与首次提取完全一致。
        """
        self._conn.execute("DELETE FROM fact_triples WHERE novel_id=?", (novel_id,))
        self._conn.execute("DELETE FROM chapter_vectors WHERE novel_id=?", (novel_id,))
        self._conn.execute("DELETE FROM sentence_vectors WHERE novel_id=?", (novel_id,))
        total = 0
        for chapter_no in sorted(facts_by_chapter):
            facts = facts_by_chapter[chapter_no]
            if not facts:
                continue
            rows = [
                (novel_id, f.subject, f.predicate, f.object, f.chapter_no)
                for f in facts
            ]
            self._conn.executemany(
                "INSERT INTO fact_triples (novel_id, subject, predicate, object, chapter_no) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            total += len(rows)
        self._conn.commit()
        return total

    # ── M10: 章节向量表（CSA 兜底 B 的存储侧） ──
    def add_vector(
        self,
        novel_id: str,
        chapter_no: int,
        subject: str,
        vector: list[float],
    ) -> None:
        """写入一条 (subject, chapter) 的记忆向量。"""
        self._conn.execute(
            "INSERT INTO chapter_vectors (novel_id, subject, chapter_no, dim, vector) "
            "VALUES (?, ?, ?, ?, ?)",
            (novel_id, subject, chapter_no, len(vector),
             ",".join(f"{x:.6f}" for x in vector)),
        )
        self._conn.commit()

    def add_vectors(
        self,
        novel_id: str,
        chapter_no: int,
        vectors: dict[str, list[float]],
    ) -> None:
        """批量写入：{subject: vector}。"""
        for subject, vector in vectors.items():
            self.add_vector(novel_id, chapter_no, subject, vector)

    def query_vectors(
        self,
        novel_id: str,
        subjects: list[str],
        *,
        before_chapter: int | None = None,
    ) -> list[dict]:
        """召回某组主语的全部记忆向量（含章号，调用方做相似度排序）。"""
        if not subjects:
            return []
        placeholders = ",".join("?" * len(subjects))
        sql = (
            "SELECT subject, chapter_no, vector FROM chapter_vectors "
            f"WHERE novel_id=? AND subject IN ({placeholders})"
        )
        params: list = [novel_id, *subjects]
        if before_chapter is not None:
            sql += " AND chapter_no < ?"
            params.append(before_chapter)
        sql += " ORDER BY chapter_no DESC"
        rows = self._conn.execute(sql, params).fetchall()
        result: list[dict] = []
        for row in rows:
            try:
                vec = [float(x) for x in row["vector"].split(",") if x]
            except ValueError:
                continue
            result.append({
                "subject": row["subject"],
                "chapter_no": row["chapter_no"],
                "vector": vec,
            })
        return result

    def delete_vectors(self, novel_id: str) -> None:
        self._conn.execute("DELETE FROM chapter_vectors WHERE novel_id=?", (novel_id,))
        self._conn.commit()

    # ── M12: 句子向量库（冷库语义检索引擎） ──
    def add_sentences(
        self,
        novel_id: str,
        chapter_no: int,
        sentences: list[tuple[str, list[float]]],
    ) -> int:
        """写入本章句子向量：(text, vector) 列表。返回写入条数。"""
        if not sentences:
            return 0
        rows = [
            (novel_id, chapter_no, i, text, len(vec),
             ",".join(f"{x:.6f}" for x in vec))
            for i, (text, vec) in enumerate(sentences)
        ]
        self._conn.executemany(
            "INSERT INTO sentence_vectors (novel_id, chapter_no, seq, text, dim, vector) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def query_sentences(
        self,
        novel_id: str,
        query_vectors: list[list[float]],
        *,
        before_chapter: int | None = None,
        after_chapter: int | None = None,
        topk: int = 5,
    ) -> list[dict]:
        """语义句召回：query_vectors 与库内句子余弦相似度排序（全扫，毫秒级）。

        返回 topk 条 {chapter_no, seq, text, score}，按章号升序（时序）。
        """
        if not query_vectors:
            return []
        sql = (
            "SELECT chapter_no, seq, text, vector FROM sentence_vectors "
            "WHERE novel_id=?"
        )
        params: list = [novel_id]
        if before_chapter is not None:
            sql += " AND chapter_no < ?"
            params.append(before_chapter)
        if after_chapter is not None:
            sql += " AND chapter_no >= ?"
            params.append(after_chapter)
        rows = self._conn.execute(sql, params).fetchall()
        scored: list[tuple[float, dict]] = []
        for row in rows:
            try:
                vec = [float(x) for x in row["vector"].split(",") if x]
            except ValueError:
                continue
            best = 0.0
            for qv in query_vectors:
                best = max(best, _cosine(qv, vec))
            if best > 0:
                scored.append((best, {
                    "chapter_no": row["chapter_no"],
                    "seq": row["seq"],
                    "text": row["text"],
                }))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:topk]
        top.sort(key=lambda t: (t[1]["chapter_no"], t[1]["seq"]))
        return [{"score": round(s, 4), **r} for s, r in top]

    def delete_sentences(self, novel_id: str) -> None:
        self._conn.execute("DELETE FROM sentence_vectors WHERE novel_id=?", (novel_id,))
        self._conn.commit()


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
