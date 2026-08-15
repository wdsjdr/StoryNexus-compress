"""M2 测试：CSA A+B 混合。"""
from __future__ import annotations

import pytest

from app.context.csa import CsaAssembler
from app.domain.models.timeline import FactTriple
from app.infra.embedding import NullEmbedding
from app.infra.timeline_index import TimelineIndex


@pytest.fixture
def csa(tmp_path):
    idx = TimelineIndex(tmp_path / "csa.db")
    idx.add(
        "novel_001",
        45,
        [
            FactTriple(subject="沈烬", predicate="受伤", object="左臂", chapter_no=45),
            FactTriple(subject="沈烬", predicate="获得", object="破刃", chapter_no=48),
            FactTriple(subject="白续", predicate="暴露身份", object="神秘人", chapter_no=12),
        ],
    )
    assembler = CsaAssembler(idx, embedding=NullEmbedding(), min_facts=2)
    yield assembler
    idx.close()


class TestCsa:
    def test_primary_facts_filtered_by_on_stage(self, csa):
        ctx = csa.build("novel_001", ["沈烬"], current_chapter_no=50)
        assert ctx.source == "facts"
        assert all("沈烬" in f for f in ctx.compressed.facts)
        # 白续被 FSM 屏蔽，不出现在候选池
        assert all("白续" not in f for f in ctx.compressed.facts)

    def test_fallback_when_facts_insufficient(self, tmp_path):
        idx = TimelineIndex(tmp_path / "empty.db")
        assembler = CsaAssembler(idx, embedding=NullEmbedding(), min_facts=3)
        ctx = assembler.build("novel_001", ["沈烬"])
        assert ctx.compressed.facts == []
        assert ctx.source in ("empty", "facts+vectors")
        idx.close()

    def test_before_chapter_excludes_current(self, csa):
        ctx = csa.build("novel_001", ["沈烬"], current_chapter_no=46)
        assert all("ch48" not in f for f in ctx.compressed.facts)
