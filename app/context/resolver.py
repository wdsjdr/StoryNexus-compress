"""Resolver：scene_type 三级来源解析 + FSM 快照组装。

scene_type 优先级（决策 2，低→高）：
  1. 预测推荐：启发式预测（如根据 chapter_goal 关键词）
  2. 大纲节点硬锚：outline node tags 存在即锁定
  3. 作者指定（重载）：最高优先级
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_KEYWORD_MAP: dict[str, list[str]] = {
    "battle": ["战斗", "对峙", "厮杀", "突袭", "追杀", "决战"],
    "politics": ["权谋", "朝堂", "联盟", "谈判", "密谋", "政变"],
    "romance": ["重逢", "心动", "告白", "暧昧", "约定"],
}


@dataclass
class SceneResolution:
    scene_type: str | None
    source: str  # author | outline | predicted | none


class SceneResolver:
    """场景类型三级来源解析器。

    关键词表可被题材 Skill 覆盖：skill.scene_keywords 与全局表合并，
    命中计数最高者胜出（skill 词表优先计数，不影响全局兜底）。
    """

    def __init__(
        self,
        skill_keywords: dict[str, list[str]] | None = None,
    ) -> None:
        # 合并：skill 关键词追加到全局表之后（全局兜底，skill 加权在前）
        merged: dict[str, list[str]] = {k: list(v) for k, v in _KEYWORD_MAP.items()}
        for scene, kws in (skill_keywords or {}).items():
            merged[scene] = [*kws, *merged.get(scene, [])]
        self._keyword_map = merged

    def resolve(
        self,
        *,
        author_override: str | None = None,
        outline_tags: list[str] | None = None,
        chapter_goal: str = "",
    ) -> SceneResolution:
        if author_override:
            return SceneResolution(scene_type=author_override, source="author")
        if outline_tags:
            return SceneResolution(scene_type=outline_tags[0], source="outline")
        predicted = self._predict(chapter_goal)
        if predicted:
            return SceneResolution(scene_type=predicted, source="predicted")
        return SceneResolution(scene_type=None, source="none")

    def _predict(self, chapter_goal: str) -> str | None:
        best: str | None = None
        best_count = 0
        for scene, keywords in self._keyword_map.items():
            hits = sum(1 for kw in keywords if kw in chapter_goal)
            if hits > best_count:
                best_count = hits
                best = scene
        return best


class FsmResolver:
    """组装 FSM 快照：仅当前出场人物的 current_state + card 运行时数据。

    M8（决策 3）：快照由 CardRegistry 构建——attributes/inventory/relationships
    来自角色卡（唯一真源），context_payload 已退役。
    M9（决策 2）：默认只带 core（守卫引用字段+inventory+stance）；
    first_appearance 集合中的角色附带 detail 全量卡（首次出场注入）。
    """

    def __init__(self, registry, *, first_appearance: set[str] | None = None) -> None:
        self._registry = registry
        self._first_appearance = first_appearance or set()

    def snapshot(self, on_stage: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in on_stage:
            fsm = self._registry.get_fsm(name)
            card = self._registry.get_card(name)
            if fsm is None and card is None:
                result[name] = {"state": "unknown", "card": {}}
                continue
            entry: dict[str, Any] = {
                "state": fsm.current_state if fsm else "unknown",
                "card": {},
            }
            if card is not None:
                if name in self._first_appearance:
                    # 首次出场：全量 detail 注入
                    entry["card"] = card.to_runtime_detail()
                    entry["first_appearance"] = True
                else:
                    # 常规：core（守卫引用字段 + inventory + stance）
                    referenced = self._registry.guard_referenced_fields(name)
                    entry["card"] = card.to_runtime_core(referenced)
            result[name] = entry
        return result
