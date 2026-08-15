"""P1 测试：语言感知基座 —— 拉丁实体 / 音译名 / 英文说话人 / 英文断句。"""
from __future__ import annotations

import sys
from pathlib import Path

from app.infra.heuristic_facts import (
    CULTIVATION_PROFILE,
    WESTERN_PROFILE,
    _find_speaker,
    discover_entities,
    discover_latin_entities,
    extract_heuristic_facts,
    get_profile,
)
from app.infra.sentences import split_sentences

EN_BOOK = """Chapter 1

Dorothy lived in the midst of the great Kansas prairies.
"I shall go to the Emerald City," said Dorothy. "The Wizard will help me."
"Dorothy said we should follow the yellow brick road," replied Scarecrow.
The Scarecrow talked with the Tin Woodman about the journey.
Dorothy asked the Tin Woodman for help.
"Be careful," whispered the Wizard. "The Wicked Witch is dangerous."

Chapter 2

The Wicked Witch sent flying monkeys to capture Dorothy.
The Scarecrow and the Tin Woodman rescued their friend Dorothy.
Dorothy said the Lion was brave after all.
"""


def test_latin_entity_discovery():
    text = "\n".join(EN_BOOK.splitlines())
    chs = {1: EN_BOOK.split("Chapter 2")[0], 2: EN_BOOK.split("Chapter 2")[1]}
    entities = discover_latin_entities(text, min_freq=2, chapters=chs)
    assert "Dorothy" in entities
    assert "Scarecrow" in entities
    assert "Wizard" in entities
    assert "Tin" not in entities  # Tin Woodman 拆词后 Tin 语境不足
    assert "The" not in entities  # 停用表
    assert "He" not in entities
    assert "said" not in entities


def test_latin_entities_merged_into_discover():
    text = "\n".join(EN_BOOK.splitlines())
    chs = {1: EN_BOOK.split("Chapter 2")[0], 2: EN_BOOK.split("Chapter 2")[1]}
    profile = CULTIVATION_PROFILE  # 默认 latin_names=False
    d = discover_entities(text, min_freq=2, profile=profile, chapters=chs)
    assert "Dorothy" not in d.entities  # 未启用 latin 时不混入
    # 启用 latin 的 western profile 在 P2 覆盖；这里直接测函数本身


def test_transliterated_name_with_dot():
    text = (
        "哈利·波特住在女贞路。哈利·波特收到霍格沃茨来信。"
        "邓布利多对哈利·波特说：“你是个巫师。”"
    )
    chs = {1: text}
    d = discover_entities(text, min_freq=2, profile=CULTIVATION_PROFILE, chapters=chs)
    assert "哈利·波特" in d.entities


def test_evidence3_generalized_no_surname():
    """无姓氏高频人名（甘道夫）在 cultivation 下也应命中（证据 3 全开放）。"""
    # 证据 3 门槛：freq ≥ max(20, 5×min_freq)、跨章 ≥5、人物语境 ≥2
    chapter = "甘道夫走进精灵王城，甘道夫对阿拉贡说：“魔戒必须毁掉。”甘道夫抽出长剑。"
    lines = [chapter] * 6  # 6 章 × 每章 3 次 = 18 次
    lines.append("阿拉贡望着甘道夫，甘道夫召唤巨鹰。甘道夫念出咒语。")
    lines.append("甘道夫与阿拉贡并肩而战，甘道夫微微一笑。")
    text = "\n".join(lines)
    chs = {i + 1: chunk for i, chunk in enumerate(text.splitlines())}
    d = discover_entities(text, min_freq=2, profile=CULTIVATION_PROFILE, chapters=chs)
    assert "甘道夫" in d.entities  # 无姓氏 + 高频 + 跨章 + 人物语境
    # 阿拉贡 freq 仅 8（< 20 门槛）——低于证据 3 门槛属预期，依赖更高词频或卡先验


def test_english_speaker_find():
    text = '"I will destroy the ring," said Gandalf. "It must be done."'
    entities = {"Gandalf", "Frodo"}
    speaker = _find_speaker(text, text.find("It must"), entities)
    assert speaker == "Gandalf"
    # 中文说话人仍工作
    zh = "林晚说：“明天见。”苏棠道：“好的。”"
    assert _find_speaker(zh, zh.find("好的"), {"林晚", "苏棠"}) == "苏棠"


def test_english_sentence_split():
    text = (
        "Dorothy lived in Kansas. She followed the yellow brick road! "
        "Was the Wizard real? Then they met Mr. Smith in the city. He was kind."
    )
    sents = split_sentences(text)
    assert any("lived in Kansas" in s for s in sents)
    assert any("followed the yellow brick road" in s for s in sents)
    assert any("Was the Wizard real" in s for s in sents)
    # Mr. 缩写保护：不把 "Mr." 单独切出
    assert not any(s.strip() == "Mr." for s in sents)
    assert any("met Mr. Smith" in s for s in sents)


def test_chinese_sentence_split_unchanged():
    text = "林晚到了江南。苏棠递给她一枚红绳，约好来年再见！"
    sents = split_sentences(text)
    assert any("林晚到了江南" in s for s in sents)
    assert any("苏棠递给她一枚红绳" in s for s in sents)


# ═══════════════════════════ P2: western profile ═══════════════════════════

WESTERN_BOOK_ZH = (
    "甘道夫对阿拉贡说：“魔戒必须毁掉。”阿拉贡拔出长剑。"
    "骑士罗兰发誓效忠女王。罗兰骑上战马，吟唱守护咒语。"
) * 3 + (
    "甘道夫召唤巨鹰。阿拉贡与罗兰缔结同盟。罗兰杀死了一只兽人。"
    "女王加冕为北方之王。甘道夫立下誓约，来年春天在白色之城相见。"
) * 2


def test_western_profile_registered():
    assert get_profile("western") is WESTERN_PROFILE
    assert WESTERN_PROFILE.latin_names is True
    assert "骑士" in WESTERN_PROFILE.title_suffixes


def test_western_zh_entity_and_facts():
    # 证据 3 门槛（freq≥20/跨章≥5/语境≥2）按真实书级设计，测试文本造足频率
    line = (
        "甘道夫召唤巨鹰，阿拉贡拔出长剑。甘道夫对阿拉贡说：“魔戒必须毁掉。”"
        "骑士罗兰发誓效忠女王。罗兰骑上战马。甘道夫立下誓约。"
    )
    text = "\n".join([line] * 10)
    chs = {i + 1: chunk for i, chunk in enumerate(text.splitlines())}
    d = discover_entities(text, min_freq=2, profile=WESTERN_PROFILE, chapters=chs)
    assert "甘道夫" in d.entities
    assert "阿拉贡" in d.entities
    assert "罗兰" in d.entities  # 称谓证据（骑士罗兰）+ 高频
    facts = extract_heuristic_facts(text, 1, d.entities, d.motifs, WESTERN_PROFILE)
    predicates = {f.predicate for f in facts}
    assert "召唤" in predicates or "拔出" in predicates  # 西幻动作模板
    assert "立誓" in predicates or "约定" in predicates  # 立下誓约
    assert all(f.subject in d.entities for f in facts)


def test_western_latin_book():
    """英文原文西幻：拉丁人名 + 英文引号说话人。
    注意：启发式动作模板为中文句式，英文原文的事实提取依赖 LLM 双轨
    （--llm-facts）；本测试验证实体与说话人层面（语言感知基座）。
    """
    en = (
        '"The ring must be destroyed," said Gandalf. "We must go to Mordor."\n'
        "Aragorn drew his sword. Aragorn rode to the battle.\n"
        '"Follow me," said Aragorn. Gandalf cast a spell upon the company.\n'
        "Aragorn swore to protect the king. Gandalf whispered the ancient words.\n"
        "The king promised to meet Aragorn in spring.\n"
    )
    chs = {1: en}
    d = discover_entities(en, min_freq=2, profile=WESTERN_PROFILE, chapters=chs)
    assert "Gandalf" in d.entities
    assert "Aragorn" in d.entities
    # 说话人回溯（英文标签）
    speaker = _find_speaker(en, en.find("We must go to Mordor"), d.entities)
    assert speaker == "Gandalf"
    # 英文断句（语义句库前置）
    sents = split_sentences(en)
    assert any("said Gandalf" in s for s in sents)


# ═══════════════════════════ P4: 冷库关键词扩展 ═══════════════════════════

def test_promise_keyword_english_and_season():
    from app.infra.narrative_registry import extract_promise_keyword

    assert extract_promise_keyword("next spring we meet in the white city", set()) == "next spring"
    assert extract_promise_keyword("in 3 years I shall return", set()) == "in 3 years"
    assert extract_promise_keyword("来年春天在白色之城相见", set()) == "来年"
    assert extract_promise_keyword("第七次花开时重逢", set()) == "第七次"
    assert extract_promise_keyword("红绳还在，来年再见", {"红绳"}) == "红绳"  # motif 优先
