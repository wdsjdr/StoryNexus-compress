"""M2 测试：原子时间线索引（CSA 主干 A）。"""
from __future__ import annotations

import pytest

from app.domain.models.timeline import FactTriple
from app.infra.timeline_index import TimelineIndex


@pytest.fixture
def index(tmp_path):
    idx = TimelineIndex(tmp_path / "timeline.db")
    yield idx
    idx.close()


def make_facts(novel="novel_001"):
    return [
        FactTriple(subject="沈烬", predicate="受伤", object="左臂", chapter_no=45),
        FactTriple(subject="沈烬", predicate="受伤", object="左臂", chapter_no=30),
        FactTriple(subject="白续", predicate="暴露身份", object="神秘人", chapter_no=12),
        FactTriple(subject="沈烬", predicate="获得", object="破刃", chapter_no=48),
        FactTriple(subject="路人甲", predicate="登场", object="归墟港", chapter_no=2),
    ]


class TestTimelineIndex:
    def test_add_and_query_by_subjects(self, index):
        index.add("novel_001", 45, make_facts())
        facts = index.query_by_subjects("novel_001", ["沈烬"])
        # 时序：新章在前
        assert facts[0].chapter_no == 48
        subjects = {f.subject for f in facts}
        assert subjects == {"沈烬"}

    def test_before_chapter_filters_future_memory(self, index):
        index.add("novel_001", 45, make_facts())
        facts = index.query_by_subjects("novel_001", ["沈烬"], before_chapter=46)
        assert all(f.chapter_no < 46 for f in facts)

    def test_latest_fact(self, index):
        index.add("novel_001", 45, make_facts())
        latest = index.latest_fact("novel_001", "沈烬", "受伤")
        assert latest is not None
        assert latest.chapter_no == 45  # 最近一次，而非 ch30

    def test_limit_per_subject(self, index):
        index.add("novel_001", 45, make_facts())
        facts = index.query_by_subjects("novel_001", ["沈烬"], limit_per_subject=1)
        # 同一主语+谓语只保留最新一条
        assert len(facts) <= 3

    def test_empty_subjects(self, index):
        assert index.query_by_subjects("novel_001", []) == []
