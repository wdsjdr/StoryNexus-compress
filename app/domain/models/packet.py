"""Canonical Packet：Dante 实际接收的编译后上下文包。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ImmutableCore(BaseModel):
    author_intent: str = ""
    chapter_goal: str = ""


class CsaCompressed(BaseModel):
    """CSA 稀疏压缩：事实三元组为主干(A)，向量为兜底(B)，语义句为冷库检索(C)。"""

    facts: list[str] = Field(default_factory=list)  # 主干：按出场人物过滤的事实文本
    vectors: list[list[float]] = Field(default_factory=list)  # 兜底 B：向量质心
    indices: list[int] = Field(default_factory=list)  # 对应章节号，保证时序
    sentences: list[str] = Field(default_factory=list)  # M12: 语义句召回（冷库检索）


class CanonicalPacket(BaseModel):
    session_id: str
    immutable_core: ImmutableCore = Field(default_factory=ImmutableCore)
    swa_context: list[str] = Field(default_factory=list)  # 近 N 章原文（未压缩）
    csa_compressed: CsaCompressed = Field(default_factory=CsaCompressed)
    hca_global: str = ""  # 全书大纲 + 当前场景切片的重度摘要
    fsm_snapshot: dict[str, Any] = Field(default_factory=dict)  # 仅当前出场人物

    def token_estimate(self) -> int:
        """仅作粗略估算占位；精确计数由 TokenCounter 在 compiler 中负责。"""
        return (
            len(self.immutable_core.author_intent)
            + len(self.immutable_core.chapter_goal)
            + sum(len(c) for c in self.swa_context)
            + sum(len(f) for f in self.csa_compressed.facts)
            + len(self.hca_global)
        )
