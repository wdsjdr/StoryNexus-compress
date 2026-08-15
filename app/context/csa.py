"""CSA 稀疏压缩：A+B 混合路线。

- 主干 A：事实三元组时序索引（TimelineIndex），按出场人物 SQL 过滤，
  保证时序正确、省 Token。
- 兜底 B：向量召回（EmbeddingService），仅当 A 召回不足时触发；
  B 结果同样带章号并强制按时间序排列。
- M9 静态锚点层：出场人物角色卡 core（无章号）作为"创作意图快照"，
  与动态事实分层注入，冲突由仲裁规则处理（动态事实优先 + 警示）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.models.packet import CsaCompressed
from app.infra.embedding import EmbeddingService, get_embedding_service
from app.infra.timeline_index import TimelineIndex
from app.infra.token_counter import get_counter


@dataclass
class CsaContext:
    compressed: CsaCompressed
    source: str  # "facts" | "facts+vectors" | "sentences" | "empty"
    token_count: int
    anchors: list[str] = field(default_factory=list)  # M9: 静态锚点层文本
    conflicts: list[str] = field(default_factory=list)  # M9: 锚点vs事实冲突
    foreshadows: list[str] = field(default_factory=list)  # M12: 出场人物开放伏笔
    factions: list[str] = field(default_factory=list)  # M12: 相关阵营


class CsaAssembler:
    def __init__(
        self,
        index: TimelineIndex,
        *,
        embedding: EmbeddingService | None = None,
        min_facts: int = 3,
        vector_topk: int = 3,
        registry=None,  # CardRegistry（M9 锚点层）
        narrative=None,  # NarrativeRegistry（M12: 伏笔/阵营冷库）
        semantic_topk: int = 5,  # M12: 语义句兜底条数（冷库检索）
        window_chapters: int = 50,  # M12: 热区窗口（近 N 章事实）
    ) -> None:
        self._index = index
        self._embedding = embedding or get_embedding_service()
        self._min_facts = min_facts
        self._vector_topk = vector_topk
        self._registry = registry
        self._narrative = narrative
        self._semantic_topk = semantic_topk
        self._window = window_chapters

    def build(
        self,
        novel_id: str,
        on_stage: list[str],
        *,
        current_chapter_no: int | None = None,
        limit_per_subject: int = 5,
    ) -> CsaContext:
        """组装 CSA。on_stage 为出场人物列表（FSM 屏蔽未出场角色在此实现）。

        M12 窗口化：只召回热区（近 window_chapters 章）事实，远古由冷库
        （伏笔/阵营登记表 + 语义句）接管；on_stage 为角色 id 时按 id+name 双键展开。
        """
        query_keys = list(on_stage)
        if self._registry is not None:
            for cid in on_stage:
                card = self._registry.get_card(cid)
                if card is not None and card.name:
                    query_keys.append(card.name)
        window_start = None
        if self._window and current_chapter_no is not None:
            window_start = max(1, current_chapter_no - self._window)
        facts = self._index.query_by_subjects(
            novel_id,
            query_keys,
            limit_per_subject=limit_per_subject,
            before_chapter=current_chapter_no,
            after_chapter=window_start,
        )
        fact_texts: list[str] = []
        for f in facts:
            # object 已是 "chN"（登场类）时不再重复附加章号
            suffix = "" if str(f.object).startswith("ch") else f"(ch{f.chapter_no})"
            fact_texts.append(f"{f.subject}{f.predicate}{f.object}{suffix}")

        # M9: 静态锚点层（角色卡 core，无章号）
        anchors: list[str] = []
        conflicts: list[str] = []
        if self._registry is not None:
            anchors, conflicts = self._build_anchors(on_stage, facts)

        vectors: list[list[float]] = []
        indices: list[int] = []
        sentences: list[str] = []
        source = "facts" if facts else "empty"

        if len(facts) < self._min_facts:
            # 兜底 B：向量召回（仅当 A 不足时）
            vectors, indices = self._vector_recall(
                on_stage, fact_texts,
                novel_id=novel_id,
                before_chapter=current_chapter_no,
                after_chapter=window_start,
            )
            if vectors:
                source = "facts+vectors"
            # 兜底 C（M12）：语义句召回（冷库检索引擎）——即使 B 命中也可并行
            sentences = self._sentence_recall(
                on_stage, novel_id,
                before_chapter=current_chapter_no,
                after_chapter=window_start,
            )
            if sentences and source == "empty":
                source = "sentences"

        # M12: 冷库注入——出场人物的开放伏笔 + 相关阵营
        foreshadows: list[str] = []
        factions: list[str] = []
        if self._narrative is not None:
            name_keys = set(query_keys)
            if self._registry is not None:
                for cid in on_stage:
                    card = self._registry.get_card(cid)
                    if card is not None and card.name:
                        name_keys.add(card.name)
            for f in self._narrative.open_for_subjects(novel_id, list(name_keys))[:6]:
                foreshadows.append(
                    f"[伏笔·{f['planted_chapter']}章] {f['subject']} {f['summary'][:30]}"
                )
            for g in self._narrative.list_factions(novel_id)[:2]:
                factions.append(
                    f"[阵营] {g['name']}: {'、'.join(g['members'][:4])}"
                )

        compressed = CsaCompressed(
            facts=fact_texts, vectors=vectors, indices=indices,
            sentences=sentences,
        )
        token_count = get_counter().count_many(
            fact_texts + sentences + foreshadows + factions
        )
        return CsaContext(
            compressed=compressed,
            source=source,
            token_count=token_count,
            anchors=anchors,
            conflicts=conflicts,
            foreshadows=foreshadows,
            factions=factions,
        )

    def _build_anchors(
        self, on_stage: list[str], facts: list
    ) -> tuple[list[str], list[str]]:
        """锚点层 + 冲突检测（决策 3：动态事实优先 + 警示）。

        - 锚点 = 角色卡 core（attributes/inventory/relationships）
        - 冲突：事实主语与某角色同名且事实内容与该角色卡 relationships
          stance 明显矛盾（如 stance 为敌意但事实是合作）→ 记录警示
        """
        anchors: list[str] = []
        conflicts: list[str] = []
        for cid in on_stage:
            card = self._registry.get_card(cid)
            if card is None:
                continue
            referenced = self._registry.guard_referenced_fields(cid)
            core = card.to_runtime_core(referenced)
            anchor_line = (
                f"[卡·{cid}] 属性:{core['attributes']} "
                f"物品:{core['inventory']} "
                f"关系:{[(r['target'], r['stance']) for r in core['relationships']]}"
            )
            anchors.append(anchor_line)
            # 冲突启发式：敌意 stance + 合作类事实谓语
            hostile = any(
                "敌" in r["stance"] or "仇" in r["stance"] or "恨" in r["stance"]
                for r in core["relationships"]
            )
            if hostile:
                # 事实主语是中文名（如"沈烬"），需匹配角色 id 或 name
                names = {cid, card.name}
                for f in facts:
                    if f.subject in names and f.predicate in ("合作", "结盟", "联合", "救"):
                        conflicts.append(
                            f"[冲突] {cid} 卡 stance=敌意，但 ch{f.chapter_no} 事实: "
                            f"{f.subject}{f.predicate}{f.object}（动态事实优先，请确认是否有意为之）"
                        )
        return anchors, conflicts

    def _vector_recall(
        self,
        query_texts: list[str],
        known_fact_texts: list[str],
        *,
        novel_id: str,
        before_chapter: int | None = None,
        after_chapter: int | None = None,
    ) -> tuple[list[list[float]], list[int]]:
        """向量兜底：以出场人物名为查询，召回相似的历史记忆。

        返回 (向量列表, 章节号列表)。按章节号升序保证时序。
        M10：接入章节向量表（chapter_vectors）；查询键为出场角色 id+中文名。
        """
        keys = list(query_texts)
        # id → 中文名展开（vector 表用中文名存储时才能命中）
        if self._registry is not None:
            for cid in query_texts:
                card = self._registry.get_card(cid)
                if card is not None and card.name:
                    keys.append(card.name)
        rows = self._index.query_vectors(
            novel_id, keys, before_chapter=before_chapter
        )
        if not rows:
            return [], []
        query_vec = self._embedding.embed(query_texts)
        if not query_vec:
            return [], []
        scored: list[tuple[float, dict]] = []
        for row in rows:
            best = 0.0
            for qv in query_vec:
                best = max(best, self._embedding.similarity(qv, row["vector"]))
            if best > 0:
                scored.append((best, row))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[: self._vector_topk]
        top.sort(key=lambda t: t[1]["chapter_no"])  # 时序升序
        vectors = [r["vector"] for _, r in top]
        chapters = [r["chapter_no"] for _, r in top]
        return vectors, chapters

    def _sentence_recall(
        self,
        on_stage: list[str],
        novel_id: str,
        *,
        before_chapter: int | None = None,
        after_chapter: int | None = None,
    ) -> list[str]:
        """冷库语义句召回（M12 兜底 C）：出场人物名 embed → 句子向量库 topk。

        返回原文句列表（按章号升序）。仅当 embedding 可用（非 Null）时生效。
        """
        if not on_stage:
            return []
        from app.infra.embedding import NullEmbedding

        if isinstance(self._embedding, NullEmbedding):
            return []
        query_vec = self._embedding.embed(on_stage)
        if not query_vec:
            return []
        rows = self._index.query_sentences(
            novel_id, query_vec,
            before_chapter=before_chapter,
            after_chapter=after_chapter,
            topk=self._semantic_topk,
        )
        return [f"ch{r['chapter_no']}·{r['text']}" for r in rows]
