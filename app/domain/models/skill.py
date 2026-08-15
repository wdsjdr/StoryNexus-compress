"""题材特化 Skill 数据契约 (skills/{skill_id}.yaml)。

Skill = 题材特化写作技能包：
- 提示词插槽（writer/evaluator/patcher 增量约束）
- 题材曲率（评分维度权重）
- AI 味硬拦截词表（StyleLint）
- 风格指纹（M4 起用于 KL 散度对齐，当前仅日志）
- 场景关键词（SceneResolver 预测 scene_type）
- 场景规则块（M9：HCA 动态段 YAML 化，作者可编辑）
"""
from __future__ import annotations

from pydantic import BaseModel, Field

SKILL_SCHEMA_VERSION = "1.0"


class StyleFingerprint(BaseModel):
    """风格指纹目标值。0 表示不约束。"""

    avg_sentence_len: float = Field(default=0, ge=0)
    exclamation_density: float = Field(default=0, ge=0, le=1)
    dialogue_ratio: float = Field(default=0, ge=0, le=1)


class SceneRuleBlock(BaseModel):
    """HCA 场景规则块（M9：从 skill YAML 读取，替代硬编码 DEFAULT_RULES）。"""

    id: str
    title: str
    tags: list[str] = Field(default_factory=list)
    content: str = ""


class SkillSpec(BaseModel):
    schema_version: str = Field(default=SKILL_SCHEMA_VERSION, pattern=r"^1\.\d+$")
    skill_id: str
    name: str = ""
    description: str = ""
    version: int = Field(default=1, ge=1)

    # 提示词插槽（注入锚点，增量约束）
    writer_prompt: str = ""
    evaluator_criteria: str = ""
    patcher_style: str = ""

    # 题材曲率：5 维度权重
    weights: dict[str, float] = Field(default_factory=dict)

    # StyleLint 硬拦截词表
    forbidden_words: list[str] = Field(default_factory=list)

    # 风格指纹
    style_fingerprint: StyleFingerprint = Field(default_factory=StyleFingerprint)

    # 场景关键词：scene_type -> 关键词列表
    scene_keywords: dict[str, list[str]] = Field(default_factory=dict)

    # M9: HCA 场景规则块（作者可编辑，替代 hca.py 硬编码）
    scene_rule_blocks: list[SceneRuleBlock] = Field(default_factory=list)
    scene_rule_map: dict[str, list[str]] = Field(default_factory=dict)

    # M12: 事实提取 profile（cultivation=修仙/权谋事件流 / slice=日常文线索 /
    # generic=通用未知文体兜底）
    # 驱动 heuristic_facts.FactProfile；benchmark --skill 与导入管线联动
    fact_profile: str = Field(default="generic", pattern=r"^(cultivation|slice|western|generic)$")

    def effective_weights(self, defaults: dict[str, float]) -> dict[str, float]:
        """权重合并：skill 缺省的维度回退到 defaults。"""
        merged = dict(defaults)
        merged.update(self.weights)
        return merged
