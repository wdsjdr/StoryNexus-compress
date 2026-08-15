"""HCA 重度压缩：全书大纲 + 世界观规则库，按场景类型切片。

模拟 DeepSeek-V4 的重度压缩注意力：
- 静态段（全书高维大纲 + 世界设定）—— 固定前缀，配合语义缓存复用 KV。
- 动态段（场景规则块）—— 仅当前场景类型对应的规则进入上下文，
  如"战斗"场景只带"灵力运转规则"，忽略"政治权谋"。

scene_type 来源（决策 2，优先级低→高）：
  1. 预测推荐 (resolver 启发式)
  2. 大纲节点硬锚 (outline node tags)
  3. 作者指定 (重载，最高)
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from app.infra.token_counter import get_counter

# 延迟导入避免循环依赖（evaluator 不依赖 hca，仅类型/构造使用）
from app.agent.evaluator import PatchInstruction

logger = logging.getLogger(__name__)

DEFAULT_RULES: dict[str, list[str]] = {
    # 场景类型 -> 规则块 id 列表
    "battle": ["power_rules"],
    "politics": ["politics_rules"],
    "romance": ["relationship_rules"],
}

_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?…\n]+")


class HcaRuleBlock(BaseModel):
    id: str
    title: str
    tags: list[str] = field(default_factory=list)
    content: str = ""


@dataclass
class HcaBundle:
    outline_summary: str
    rule_blocks: list[HcaRuleBlock]
    token_count: int


class HcaAssembler:
    """重度压缩装配器：大纲固定前缀 + 场景规则切片。

    M9：规则块/映射从 SkillSpec 读取（作者可编辑 YAML），
    DEFAULT_RULES 仅作无 skill 兜底。
    """

    def __init__(
        self,
        outline_summary: str,
        rule_blocks: list[HcaRuleBlock] | None = None,
        *,
        scene_rule_map: dict[str, list[str]] | None = None,
        skill=None,  # SkillSpec（M9：scene_rule_blocks + scene_rule_map）
    ) -> None:
        self._outline = outline_summary
        if skill is not None:
            self._blocks = {b.id: HcaRuleBlock(**b.model_dump()) for b in skill.scene_rule_blocks}
            self._scene_rule_map = skill.scene_rule_map or DEFAULT_RULES
        else:
            self._blocks = {b.id: b for b in (rule_blocks or [])}
            self._scene_rule_map = scene_rule_map or DEFAULT_RULES

    @property
    def outline_summary(self) -> str:
        return self._outline

    def build(self, scene_type: str | None, *, max_tokens: int = 2000) -> HcaBundle:
        """按场景类型切片组装 HCA，硬上限 max_tokens（tiktoken 精确计数）。"""
        counter = get_counter()
        blocks: list[HcaRuleBlock] = []
        if scene_type:
            for block_id in self._scene_rule_map.get(scene_type, []):
                block = self._blocks.get(block_id)
                if block is not None:
                    blocks.append(block)
        total = counter.count(self._outline)
        kept: list[HcaRuleBlock] = []
        for b in blocks:
            n = counter.count(b.content)
            if total + n > max_tokens:
                break
            kept.append(b)
            total += n
        return HcaBundle(
            outline_summary=self._outline,
            rule_blocks=kept,
            token_count=total,
        )

    def render(self, bundle: HcaBundle) -> str:
        """渲染为送入上下文的纯文本。"""
        parts = [f"[全书大纲]\n{bundle.outline_summary}"]
        for b in bundle.rule_blocks:
            parts.append(f"[{b.title}]\n{b.content}")
        return "\n\n".join(parts)


@dataclass
class StyleSnapshot:
    """当前章风格指纹快照（风格熵减接口，M3.5 仅统计与日志）。"""

    avg_sentence_len: float = 0.0
    exclamation_density: float = 0.0
    dialogue_ratio: float = 0.0
    char_count: int = 0

    def __str__(self) -> str:
        return (
            f"StyleSnapshot(avg_sentence_len={self.avg_sentence_len:.1f}, "
            f"exclamation_density={self.exclamation_density:.4f}, "
            f"dialogue_ratio={self.dialogue_ratio:.3f})"
        )


def extract_style_stats(draft: str) -> StyleSnapshot:
    """统计修辞特征：平均句长、感叹号密度、对话比例。

    注意：M3.5 为接口 Stub，只做统计与日志；
    KL 散度对齐与 Evaluator style_drift 告警在 M4 实现。
    """
    if not draft.strip():
        return StyleSnapshot()
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(draft) if s.strip()]
    avg_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0.0
    char_count = len(draft)
    exclamation = draft.count("！") + draft.count("!")
    exclamation_density = exclamation / char_count if char_count else 0.0
    # 对话比例：以引号包裹的内容占比近似
    quoted = len(re.findall(r"[“”「」]", draft))
    dialogue_ratio = (quoted / 2) / max(1, len(sentences))
    return StyleSnapshot(
        avg_sentence_len=avg_len,
        exclamation_density=exclamation_density,
        dialogue_ratio=min(dialogue_ratio, 1.0),
        char_count=char_count,
    )


def log_style_stats(snapshot: StyleSnapshot, skill=None) -> None:
    """M3.5：仅日志记录（不触发告警/纠正）。M4 接 KL 散度阈值。"""
    target = getattr(skill, "style_fingerprint", None) if skill else None
    target_str = ""
    if target is not None and target.avg_sentence_len > 0:
        target_str = f", target_avg_len={target.avg_sentence_len:.1f}"
    logger.info(
        "[Style] 当前章风格指纹: avg_len=%.1f, exclam=%.4f, dialogue=%.3f%s",
        snapshot.avg_sentence_len,
        snapshot.exclamation_density,
        snapshot.dialogue_ratio,
        target_str,
    )


# ── M10: 风格指纹 KL 散度 + style_drift 告警 + Patcher 纠正 ──

_MAX_SENTENCE_LEN = 60  # 句长归一化上限（超过视为"长句密集"饱和）
_EPS = 1e-9


def _snapshot_distribution(snapshot: StyleSnapshot) -> list[float]:
    """当前章特征 → 概率分布（3 维，平滑后归一化）。"""
    raw = [
        min(snapshot.avg_sentence_len / _MAX_SENTENCE_LEN, 1.0),
        snapshot.exclamation_density,
        snapshot.dialogue_ratio,
    ]
    total = sum(raw) + _EPS * len(raw)
    return [x / total for x in raw]


def _target_distribution(target) -> list[float]:
    """skill.style_fingerprint → 目标分布（值域 [0,1] 直接归一化）。"""
    raw = [
        min(target.avg_sentence_len / _MAX_SENTENCE_LEN, 1.0),
        target.exclamation_density,
        target.dialogue_ratio,
    ]
    total = sum(raw) + _EPS * len(raw)
    return [x / total for x in raw]


def _kl(p: list[float], q: list[float]) -> float:
    """KL 散度 p||q（q 平滑防除零）。"""
    return sum(p[i] * math.log((p[i] + _EPS) / (q[i] + _EPS)) for i in range(len(p)))


class StyleGuard:
    """风格漂移守卫：当前章 vs skill.style_fingerprint 的对称 KL 散度。

    M10：目标指纹全零 = 不启用（向后兼容）。超阈值返回 PatchInstruction，
    交给 Patcher 生成"恢复既定风格"的微创修改。
    """

    def __init__(self, target=None, threshold: float = 0.15) -> None:
        self._target = target
        self._threshold = threshold

    @property
    def enabled(self) -> bool:
        if self._target is None:
            return False
        return any(
            (getattr(self._target, dim, 0.0) or 0.0) > 0
            for dim in ("avg_sentence_len", "exclamation_density", "dialogue_ratio")
        )

    def drift(self, snapshot: StyleSnapshot) -> float:
        """对称 KL：(p||q + q||p)/2，p=当前章，q=目标。"""
        p = _snapshot_distribution(snapshot)
        q = _target_distribution(self._target)
        return 0.5 * (_kl(p, q) + _kl(q, p))

    def check(self, draft: str) -> tuple[float, PatchInstruction | None]:
        """返回 (KL 散度, 修补指令|None)。"""
        if not self.enabled or not draft.strip():
            return 0.0, None
        snapshot = extract_style_stats(draft)
        value = self.drift(snapshot)
        if value <= self._threshold:
            return value, None
        instruction = PatchInstruction(
            line_range="全文",
            issue=(
                f"风格漂移 KL={value:.3f}（阈值 {self._threshold}）："
                f"句长/感叹密度/对话比例偏离 skill 风格指纹"
            ),
            suggestion=(
                "恢复前文既定风格：校准句长节奏、收敛情绪化感叹、"
                "维持对话与叙述的比例"
            ),
        )
        return value, instruction
