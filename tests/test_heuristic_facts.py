"""M12 测试：profile 化启发式事实提取（实体驱动 + motif 层 + 约定/关系）。

验证：垃圾片段不产出、真实实体召回（含无姓氏高频人名）、motif 线索层、
主语/宾语约束、slice 约定/信物事件、cultivation 兼容性。
"""
from __future__ import annotations

from dataclasses import replace

from app.infra.heuristic_facts import (
    SLICE_PROFILE,
    CULTIVATION_PROFILE,
    discover_entities,
    extract_heuristic_facts,
)

# 小文本测试用 profile（降频次/跨章阈值；真实场景用 SLICE_PROFILE 默认值）
LIGHT_SLICE = replace(
    SLICE_PROFILE,
    min_motif_freq=3,
    min_motif_chapters=1,
    high_freq_min=3,
    high_freq_span=1,
)

# 取自真实文本 ch23 的原文片段（曾产生 "远比表面""上去要" 等垃圾）
CH23_FRAGMENT = (
    "这个女人，远比表面看上去要来得有趣。"
    "按照张墨合和宋霆的交情，宋霆上前说道：“玉尘仙君，这边请。”"
    "东嫦曦微微一笑，潮水般涌来的傀儡退去。"
    "曲乐绫坐在亭中，等着宋霆回来。"
)

# 全书（精简版）实体发现用文本（修仙事件流）
FULL_TEXT = (
    "宋霆进入合欢宗。玉尘仙君坐在台上。东嫦曦来到宋霆身边。"
    "曲乐绫跟随宋霆离开。赵紫柔送给宋霆一枚丹药。"
    "玉尘仙君抬起头。宋霆杀死一头妖兽。合欢宗的长老在开会。"
    "宋霆。曲乐绫。东嫦曦。赵紫柔。合欢宗。"
)

# 宇航前几章风格文本（slice 日常文；重复以满足 motif 频次门槛，
# 独立成词的"红绳。""薄荷糖。"满足 motif 的 bounded 验证）
SLICE_TEXT = (
    "林宇航走进教室。苏棠坐在窗边。爱民戴上眼镜看向黑板。" * 3
    + "苏棠递给林宇航一颗薄荷糖。" * 3
    + "林宇航接过薄荷糖，林宇航系上红绳。" * 3
    + "苏棠说：“约好七年之后见面。”" * 3
    + "林宇航说：“我等你。”" * 3
    + "林宇航和苏棠牵手走在槐树下。" * 3
    + "红绳。" * 4 + "薄荷糖。" * 4 + "槐树。" * 4
)


class TestDiscoverEntities:
    def test_title_suffix_entities(self):
        disc = discover_entities(FULL_TEXT, min_freq=1)
        assert "玉尘仙君" in disc.entities
        assert "合欢宗的长老" not in disc.entities  # 含"的"的主体被过滤
        assert "宋霆" in disc.entities
        assert "东嫦曦" in disc.entities

    def test_frequent_proper_nouns(self):
        disc = discover_entities(FULL_TEXT, min_freq=1)
        assert "曲乐绫" in disc.entities
        assert "赵紫柔" in disc.entities
        assert "合欢宗" in disc.entities

    def test_garbage_words_excluded(self):
        disc = discover_entities(
            "修士修士修士 气息气息气息 一声一声一声 力量力量力量 东西东西东西 "
            "上来上来 下来下来 宋霆 宋霆 宋霆 宋霆 东嫦曦 东嫦曦 曲乐绫 曲乐绫",
            min_freq=1,
        )
        for garbage in ("修士", "气息", "一声", "力量", "东西", "上来"):
            assert garbage not in disc.entities
        assert "宋霆" in disc.entities
        assert "东嫦曦" in disc.entities

    def test_min_freq_threshold(self):
        disc = discover_entities("路人甲" * 2 + " 路人乙", min_freq=3)
        assert "路人甲" not in disc.entities  # 2 次 < 3

    # ── M12: slice profile ──
    def test_slice_high_freq_no_surname_entity(self):
        """无姓氏高频跨章人名（爱民类）：slice 第 3 证据。"""
        text = "爱民说：" * 30 + "爱民抬头。" * 10 + "爱民。" * 5
        disc = discover_entities(text, min_freq=2, profile=LIGHT_SLICE)
        assert "爱民" in disc.entities

    def test_slice_no_surname_requires_person_context(self):
        """无姓氏高频词但无人语境（学校类）→ 不入实体。"""
        text = "学校学校" * 100
        disc = discover_entities(text, min_freq=2, profile=LIGHT_SLICE)
        assert "学校" not in disc.entities

    def test_motif_discovery(self):
        text = (
            "他把红绳系在手腕。" * 12 + "红绳。" * 4
            + "薄荷糖从口袋里掉出来。" * 6 + "薄荷糖。" * 3
            + "玻璃罐放在窗台。" * 8 + "玻璃罐。" * 4
        )
        # 合成 8 章让跨章跨度满足（真实场景由章节边界提供）
        chapters = {i: f"第{i}章 {text}" for i in range(1, 9)}
        disc = discover_entities(text, min_freq=2, profile=LIGHT_SLICE, chapters=chapters)
        assert "红绳" in disc.motifs
        assert "薄荷糖" in disc.motifs
        assert "玻璃罐" in disc.motifs
        # 重叠窗口碎片（璃罐/荷糖）不独立成词 → 不入选
        assert "璃罐" not in disc.motifs
        assert "荷糖" not in disc.motifs

    def test_card_names_prior(self):
        """卡先验：卡名直接并入实体（不依赖统计证据）。"""
        disc = discover_entities("正文不包含该名字", min_freq=1,
                                 card_names={"爱民", "苏棠"})
        assert "爱民" in disc.entities and "苏棠" in disc.entities


class TestExtractHeuristicFacts:
    def _disc(self, text=None):
        return discover_entities(text or FULL_TEXT, min_freq=1)

    def test_no_garbage_fragments(self):
        """曾产出垃圾的原文片段现在必须零垃圾。"""
        disc = discover_entities(CH23_FRAGMENT, min_freq=1)
        facts = extract_heuristic_facts(CH23_FRAGMENT, 23, disc.entities)
        subjects = {f.subject for f in facts}
        for garbage in ("远比表面", "上去要", "和元若的", "潮水般涌"):
            assert garbage not in subjects

    def test_subject_always_in_entities(self):
        disc = self._disc()
        facts = extract_heuristic_facts(FULL_TEXT, 1, disc.entities)
        assert facts
        for f in facts:
            assert f.subject in disc.entities, f"主语 {f.subject} 不在实体集"

    def test_real_entity_recall(self):
        disc = self._disc()
        assert {"宋霆", "东嫦曦", "曲乐绫", "赵紫柔", "玉尘仙君", "合欢宗"} <= disc.entities
        facts = extract_heuristic_facts(FULL_TEXT, 1, disc.entities)
        subjects = {f.subject for f in facts}
        assert "宋霆" in subjects

    def test_own_requires_both_entities(self):
        disc = self._disc()
        facts = extract_heuristic_facts(
            "宋霆的木盒 修士的气息 东嫦曦的嫁妆 路上的一朵野花", 2, disc.entities
        )
        own = [f for f in facts if f.predicate == "拥有"]
        assert all(f.subject in disc.entities and f.object in disc.entities for f in own)

    def test_action_whitelist(self):
        disc = self._disc()
        facts = extract_heuristic_facts(
            "宋霆杀死东嫦曦 玉尘仙君放走宋霆", 3, disc.entities
        )
        predicates = {f.predicate for f in facts}
        assert "杀死" in predicates

    def test_no_appearance_facts(self):
        """M12: 登场事实已移除。"""
        disc = self._disc()
        facts = extract_heuristic_facts(
            "宋霆 东嫦曦 曲乐绫 赵紫柔 玉尘仙君 合欢宗", 4, disc.entities
        )
        assert all(f.predicate != "登场" for f in facts)

    def test_max_facts_per_chapter(self):
        disc = self._disc()
        long_text = "宋霆的宋霆 东嫦曦的东嫦曦 " * 20
        facts = extract_heuristic_facts(long_text, 5, disc.entities)
        assert len(facts) <= 8

    def test_dedupe(self):
        disc = self._disc()
        facts = extract_heuristic_facts("宋霆的宋霆 宋霆的宋霆", 6, disc.entities)
        keys = [(f.subject, f.predicate, f.object) for f in facts]
        assert len(keys) == len(set(keys))

    # ── M12: slice 日常事件 ──
    @staticmethod
    def _slice_chapters(text: str) -> dict[int, str]:
        return {i: f"第{i}章 {text}" for i in range(1, 9)}

    def test_slice_gift_and_motif(self):
        """苏棠递给林宇航一颗薄荷糖 → (苏棠, 递给, 薄荷糖)。"""
        chapters = self._slice_chapters(SLICE_TEXT)
        disc = discover_entities(SLICE_TEXT, min_freq=1, profile=LIGHT_SLICE,
                                 chapters=chapters)
        facts = extract_heuristic_facts(
            SLICE_TEXT, 1, disc.entities, disc.motifs, SLICE_PROFILE
        )
        triples = {(f.subject, f.predicate, f.object) for f in facts}
        assert ("苏棠", "递给", "薄荷糖") in triples
        assert ("林宇航", "系上", "红绳") in triples

    def test_slice_promise_from_dialogue(self):
        chapters = self._slice_chapters(SLICE_TEXT)
        disc = discover_entities(SLICE_TEXT, min_freq=1, profile=LIGHT_SLICE,
                                 chapters=chapters)
        facts = extract_heuristic_facts(
            SLICE_TEXT, 1, disc.entities, disc.motifs, SLICE_PROFILE
        )
        promises = [f for f in facts if f.predicate == "约定"]
        assert promises
        assert any(f.subject == "苏棠" for f in promises)

    def test_slice_relationship(self):
        chapters = self._slice_chapters(SLICE_TEXT)
        disc = discover_entities(SLICE_TEXT, min_freq=1, profile=LIGHT_SLICE,
                                 chapters=chapters)
        facts = extract_heuristic_facts(
            SLICE_TEXT, 1, disc.entities, disc.motifs, SLICE_PROFILE
        )
        rel = [f for f in facts if f.predicate == "牵手"]
        assert rel and rel[0].subject == "林宇航" and rel[0].object == "苏棠"

    def test_slice_gift_object_fragment_rejected(self):
        """碎片宾语（荷糖/璃罐）不入选，整词（薄荷糖/玻璃罐）入选。"""
        text = (
            "苏棠递给林宇航一颗薄荷糖。" * 8
            + "薄荷糖。" * 4
            + "玻璃罐放在桌上。" * 6
            + "玻璃罐。" * 4
        )
        chapters = self._slice_chapters(text)
        disc = discover_entities(text, min_freq=2, profile=LIGHT_SLICE,
                                 chapters=chapters)
        facts = extract_heuristic_facts(text, 1, disc.entities, disc.motifs, LIGHT_SLICE)
        objs = {f.object for f in facts}
        assert "薄荷糖" in objs
        assert "荷糖" not in objs
        assert "璃罐" not in objs
