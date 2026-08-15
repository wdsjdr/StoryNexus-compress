"""SkillRegistry：装载 /src/skills/*.yaml → SkillSpec 注册表。

- 扫描目录加载所有 *.yaml（忽略 *.yaml.template 模板壳）
- get(skill_id)：未知 id 回退 default
- novel.yaml 的 skill_id 字段驱动题材切换
"""
from __future__ import annotations

from pathlib import Path

import yaml

from app.domain.models.skill import SkillSpec

DEFAULT_SKILL_ID = "default"


class SkillRegistryError(ValueError):
    pass


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def load_dir(self, skills_dir: str | Path) -> int:
        """装载目录下所有 skill yaml，返回加载数量。"""
        path = Path(skills_dir)
        if not path.is_dir():
            return 0
        loaded = 0
        for f in sorted(path.glob("*.yaml")):
            spec = self._load_file(f)
            self._skills[spec.skill_id] = spec
            loaded += 1
        return loaded

    @staticmethod
    def _load_file(f: Path) -> SkillSpec:
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise SkillRegistryError(f"Skill YAML 解析失败: {f}: {exc}") from exc
        if not isinstance(data, dict):
            raise SkillRegistryError(f"Skill 文件必须是映射结构: {f}")
        try:
            return SkillSpec.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            raise SkillRegistryError(f"Skill 校验失败: {f}: {exc}") from exc

    def register(self, spec: SkillSpec) -> None:
        self._skills[spec.skill_id] = spec

    def get(self, skill_id: str | None) -> SkillSpec:
        """取 skill；未知/空 id 回退 default。"""
        if skill_id and skill_id in self._skills:
            return self._skills[skill_id]
        return self._skills.get(DEFAULT_SKILL_ID, SkillSpec(skill_id=DEFAULT_SKILL_ID))

    def has(self, skill_id: str) -> bool:
        return skill_id in self._skills

    def ids(self) -> list[str]:
        return sorted(self._skills.keys())
