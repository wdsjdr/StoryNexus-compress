"""M2 测试：HCA 重度压缩 + 场景切片。"""
from __future__ import annotations

from app.context.hca import HcaAssembler, HcaRuleBlock


def make_assembler():
    blocks = [
        HcaRuleBlock(id="power_rules", title="灵力运转规则", tags=["battle"], content="灵力运行规则正文……"),
        HcaRuleBlock(id="politics_rules", title="政治权谋规则", tags=["politics"], content="朝堂博弈规则正文……"),
        HcaRuleBlock(id="relationship_rules", title="人物关系潜规则", tags=["romance"], content="关系规则正文……"),
    ]
    return HcaAssembler("全书大纲摘要", blocks)


class TestHca:
    def test_battle_scene_slices_only_power_rules(self):
        a = make_assembler()
        bundle = a.build("battle")
        assert [b.id for b in bundle.rule_blocks] == ["power_rules"]
        rendered = a.render(bundle)
        assert "灵力运转规则" in rendered
        assert "政治权谋规则" not in rendered

    def test_none_scene_keeps_only_outline(self):
        a = make_assembler()
        bundle = a.build(None)
        assert bundle.rule_blocks == []

    def test_max_tokens_hard_cap(self):
        a = make_assembler()
        bundle = a.build("battle", max_tokens=10)
        assert bundle.token_count <= 10

    def test_unknown_scene_no_rules(self):
        a = make_assembler()
        bundle = a.build("cooking")
        assert bundle.rule_blocks == []
