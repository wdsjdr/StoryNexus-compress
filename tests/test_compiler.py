"""M2 测试：PacketCompiler 预算校核 + 构成报告。"""
from __future__ import annotations

from app.context.compiler import PacketCompiler
from app.domain.models.packet import CsaCompressed, ImmutableCore


def make_compiler(**kw):
    return PacketCompiler(swa_window_tokens=100, csa_compression_ratio=0.1, hca_global_tokens=50, **kw)


class TestCompiler:
    def test_assembly_and_report(self):
        comp = make_compiler()
        result = comp.compile(
            session_id="ch_050_write",
            immutable_core=ImmutableCore(
                author_intent="写紧张对峙", chapter_goal="沈烬与白续对峙"
            ),
            swa_chapters=["章49正文", "章50正文"],
            csa=CsaCompressed(facts=["沈烬受伤左臂(ch45)", "白续暴露身份(ch12)"]),
            hca_text="全书大纲 + 战斗规则",
            fsm_snapshot={
                "shen_jin": {"state": "tentative_hostility", "payload": {}},
                "white_xu": {"state": "mysterious", "payload": {}},
            },
            raw_recent_tokens=1000,
        )
        p = result.packet
        assert p.session_id == "ch_050_write"
        assert len(p.swa_context) == 2
        assert p.fsm_snapshot["shen_jin"]["state"] == "tentative_hostility"
        r = result.report
        assert r.total_tokens == (
            r.core_tokens + r.swa_tokens + r.csa_tokens + r.hca_tokens + r.fsm_tokens
        )
        assert r.swa_tokens > 0
        assert r.csa_tokens > 0

    def test_swa_over_budget_drops_oldest(self):
        comp = make_compiler()
        long_ch = "很长的章节内容" * 100  # 远超预算
        short_ch = "短章"
        result = comp.compile(
            session_id="s",
            immutable_core=ImmutableCore(),
            swa_chapters=[long_ch, short_ch],
            csa=CsaCompressed(),
            hca_text="",
            fsm_snapshot={},
            raw_recent_tokens=2000,
        )
        assert len(result.packet.swa_context) == 1
        assert result.packet.swa_context == [short_ch]
        assert result.report.discarded_tokens > 0

    def test_hca_hard_cap_truncates(self):
        comp = make_compiler()
        big_hca = "世界观规则" * 200
        result = comp.compile(
            session_id="s",
            immutable_core=ImmutableCore(),
            swa_chapters=[],
            csa=CsaCompressed(),
            hca_text=big_hca,
            fsm_snapshot={},
            raw_recent_tokens=0,
        )
        assert result.report.hca_tokens <= 50
