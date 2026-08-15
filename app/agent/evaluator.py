"""最小 PatchInstruction（压缩引擎独立包裁剪版）。

主仓 storynexus 中该定义位于 app/agent/evaluator.py（完整 Evaluator 依赖
LLM 网关）；独立压缩包只保留此数据契约供 hca.StyleGuard 引用。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatchInstruction:
    """病灶修补指令：行区间 + 问题 + 建议（Patcher 消费）。"""

    line_range: str
    issue: str
    suggestion: str
