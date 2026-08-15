"""Packet Compiler：组装 CanonicalPacket 并执行预算校核。

装配流水线：
  ① 解析出场人物 / scene_type
  ② SWA：当前章+近N章原文（不压缩）
  ③ CSA：事实三元组按出场人物过滤（A 主干，B 兜底）
  ④ HCA：大纲固定前缀 + 场景规则切片
  ⑤ FSM 快照：仅出场人物
  ⑥ 预算校核（TokenCounter 精确计数）——超限按 HCA > CSA > SWA 顺序牺牲
  ⑦ 生成构成报告（供 Studio 上下文探针饼图使用）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain.models.packet import CanonicalPacket, CsaCompressed, ImmutableCore
from app.infra.token_counter import get_counter


@dataclass
class CompositionReport:
    """Token 构成报告：SWA(蓝)/CSA(绿)/HCA(红) 占比 + 被屏蔽废弃 Token 数。"""

    swa_tokens: int = 0
    csa_tokens: int = 0
    hca_tokens: int = 0
    fsm_tokens: int = 0
    core_tokens: int = 0
    discarded_tokens: int = 0
    total_tokens: int = 0
    conflicts: list[str] = field(default_factory=list)  # M9: 锚点vs事实冲突

    def as_dict(self) -> dict:
        return {
            "swa_tokens": self.swa_tokens,
            "csa_tokens": self.csa_tokens,
            "hca_tokens": self.hca_tokens,
            "fsm_tokens": self.fsm_tokens,
            "core_tokens": self.core_tokens,
            "discarded_tokens": self.discarded_tokens,
            "total_tokens": self.total_tokens,
            "conflicts": list(self.conflicts),
        }


@dataclass
class CompiledPacket:
    packet: CanonicalPacket
    report: CompositionReport
    prefix_cache_hit: bool = False


class PacketCompiler:
    """上下文包编译器。输入各层装配结果，输出 CanonicalPacket + 构成报告。"""

    def __init__(
        self,
        *,
        swa_window_tokens: int = 32000,
        csa_compression_ratio: float = 0.1,
        hca_global_tokens: int = 2000,
    ) -> None:
        self._budget = {
            "swa": swa_window_tokens,
            "csa": swa_window_tokens,  # 占位，实际按压缩率对 SWA 原文计算
            "hca": hca_global_tokens,
        }
        self._csa_ratio = csa_compression_ratio
        self._counter = get_counter()

    def compile(
        self,
        *,
        session_id: str,
        immutable_core: ImmutableCore,
        swa_chapters: list[str],
        csa: CsaCompressed,
        hca_text: str,
        fsm_snapshot: dict[str, Any],
        raw_recent_tokens: int,  # SWA 未压缩原文总 token（用于 CSA 压缩率核算）
        csa_token_budget: int | None = None,  # 显式 CSA 预算，缺省按压缩率推导
        conflicts: list[str] | None = None,  # M9: CSA 锚点vs事实冲突警示
    ) -> CompiledPacket:
        counter = self._counter
        report = CompositionReport()

        core_tokens = counter.count(immutable_core.author_intent) + counter.count(immutable_core.chapter_goal)
        report.core_tokens = core_tokens

        # SWA 原文保持，超预算丢最老章
        kept_swa: list[str] = []
        swa_tokens = 0
        for ch in swa_chapters:
            n = counter.count(ch)
            if swa_tokens + n > self._budget["swa"]:
                report.discarded_tokens += n
                continue
            kept_swa.append(ch)
            swa_tokens += n
        report.swa_tokens = swa_tokens

        # CSA：按压缩率核算，超限丢弃事实条目（先丢最老，即列表尾部）
        csa_budget = (
            csa_token_budget
            if csa_token_budget is not None
            else int(raw_recent_tokens * self._csa_ratio)
        )
        kept_facts: list[str] = []
        csa_tokens = 0
        for f in csa.facts:
            n = counter.count(f)
            if csa_tokens + n > csa_budget:
                report.discarded_tokens += n
                continue
            kept_facts.append(f)
            csa_tokens += n
        report.csa_tokens = csa_tokens

        # HCA：硬上限（token 精确截断，禁止字符估算）
        hca_tokens = counter.count(hca_text)
        if hca_tokens > self._budget["hca"]:
            report.discarded_tokens += hca_tokens - self._budget["hca"]
            hca_text = counter.truncate_to_tokens(hca_text, self._budget["hca"])
            hca_tokens = self._budget["hca"]
        report.hca_tokens = hca_tokens

        fsm_text = " ".join(
            f"{k}:{v.get('state', '?')}" for k, v in fsm_snapshot.items()
        )
        report.fsm_tokens = counter.count(fsm_text)

        report.total_tokens = (
            core_tokens + swa_tokens + csa_tokens + hca_tokens + report.fsm_tokens
        )
        if conflicts:
            report.conflicts = list(conflicts)

        packet = CanonicalPacket(
            session_id=session_id,
            immutable_core=immutable_core,
            swa_context=kept_swa,
            csa_compressed=CsaCompressed(
                facts=kept_facts,
                vectors=csa.vectors,
                indices=csa.indices,
            ),
            hca_global=hca_text,
            fsm_snapshot=fsm_snapshot,
        )
        return CompiledPacket(packet=packet, report=report)
