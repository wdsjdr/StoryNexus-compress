"""共享测试夹具（压缩引擎独立包：无卡片/FSM 依赖，保留空夹具占位）。"""
from __future__ import annotations

import pytest


@pytest.fixture
def card_registry(tmp_path):
    """占位夹具：独立包无卡片注册表（主仓 M8 体系），返回 None。"""
    return None
