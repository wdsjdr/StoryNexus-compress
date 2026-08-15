"""1M 上下文窗口大规模压测（M12：热区/冷库架构 + 题材 profile + 双轨对照）。

数据：--src 指定小说全文（UTF-8），默认 <你的书.txt>
（可用环境变量 STORYNEXUS_BENCH_SRC 设置默认路径）

压测内容：
  1. 全书 Token 实测 vs token-budget.md 系数
  2. SWA 装配：丢最老章明细（丢弃章号/token/当前章超预算标记）
  3. CSA 装配：题材 profile（--skill 的 fact_profile）启发式提取
     + motif 线索层 + 卡先验（--cards-novel）+ 冷库登记（伏笔/阵营）
     + 句子向量语义句召回（STORYNEXUS_EMBEDDING=hash|fastembed）
     + 热区窗口（近 50 章事实）
  4. HCA 装配（--skill 题材规则块）
  5. LLM 双轨对照（--llm-facts，需 litellm 后端 + API key）
  6. 信息完整性评估：motif 覆盖率 / 跨章呼应探测 / 人物关系保留率 /
     伏笔可及性 / 卡先验对比
  7. 压缩数据导出 compressed/

输出：benchmark_report.md / benchmark_report.json / compressed/*
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path

from app.context.csa import CsaAssembler
from app.context.hca import HcaAssembler
from app.context.swa import SwaAssembler
from app.domain.models.timeline import FactTriple
from app.infra.embedding import HashEmbedding, NullEmbedding, get_embedding_service
from app.infra.facts_store import FactsStore
from app.infra.heuristic_facts import (
    complete_word_count,
    discover_entities,
    extract_heuristic_facts,
    get_profile,
)
from app.infra.narrative_registry import NarrativeRegistry, extract_promise_keyword
from app.infra.sentences import split_sentences
from app.infra.skill_registry import SkillRegistry
from app.infra.timeline_index import TimelineIndex
from app.infra.token_counter import get_counter

DEFAULT_SRC = Path(
    os.getenv("STORYNEXUS_BENCH_SRC", "")
) if os.getenv("STORYNEXUS_BENCH_SRC") else Path("<你的书.txt>")
COMPRESSED_DIR_NAME = "compressed"
SKILLS_DIR = Path(__file__).resolve().parent.parent / "data" / "src" / "skills"
HOT_ZONE_CHAPTERS = 50  # M12: 热区窗口（近 N 章事实）

# 切章边界：兼容「第X章」/「第X回」（中文，容忍全角空格缩进）与
# 「Chapter N」/「Chapter I」（英文原文西幻/公版书，含罗马数字章号）；
# 与 importer.py 同源语义
_CHAPTER_RE = re.compile(r"^第\s*([0-9零一二三四五六七八九十百千]+)\s*[章回].*$")
_EN_CHAPTER_RE = re.compile(r"^Chapter\s+([0-9]+|[IVXLCDM]+)[.:]?\s*$", re.IGNORECASE)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s: str) -> int:
    """罗马数字（I/IV/V/IX/X…）转整数；非法返回 0。"""
    total, prev = 0, 0
    try:
        for ch in reversed(s.upper()):
            v = _ROMAN_VALUES[ch]
            total += -v if v < prev else v
            prev = v
    except KeyError:
        return 0
    return total
_CN_DIGITS = {c: i for i, c in enumerate("零一二三四五六七八九")}


def cn_to_int(s: str) -> int:
    """中文数字（≤999）转整数；阿拉伯数字直接解析。"""
    s = s.strip()
    if s.isdigit():
        return int(s)
    s = s.replace("零", "")
    if "千" in s or "百" in s:
        total, cur = 0, 0
        for ch in s:
            if ch in _CN_DIGITS:
                cur = _CN_DIGITS[ch]
            elif ch == "十":
                cur = (cur or 1) * 10
                total += cur
                cur = 0
            elif ch in ("百", "千"):
                total += (cur or 1) * (100 if ch == "百" else 1000)
                cur = 0
        return total + cur
    if "十" in s:
        a, _, b = s.partition("十")
        tens = (_CN_DIGITS.get(a, 1) if a else 1) * 10
        return tens + (_CN_DIGITS.get(b, 0) if b else 0)
    return _CN_DIGITS[s]


def split_chapters(text: str) -> dict[int, str]:
    """按行首 第N章/第N回/Chapter N|I（阿拉伯/中文/罗马数字，容忍全角空格）切章。"""
    lines = text.splitlines()
    chapters: dict[int, str] = {}
    current_no: int | None = None
    current: list[str] = []
    for line in lines:
        stripped = line.lstrip("\u3000\u2003\u00a0 \t")
        m = _CHAPTER_RE.match(stripped)
        if m is None:
            m = _EN_CHAPTER_RE.match(stripped)
        if m:
            if current_no is not None:
                chapters[current_no] = "\n".join(current)
            num = m.group(1)
            if num.isdigit() or not num.isascii():
                current_no = cn_to_int(num)
            else:
                current_no = roman_to_int(num)
            current = [line]
        else:
            current.append(line)
    if current_no is not None:
        chapters[current_no] = "\n".join(current)
    return chapters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1M 上下文窗口压测（M12 热区/冷库架构）")
    p.add_argument("--src", default=str(DEFAULT_SRC),
                   help=f"小说全文路径（默认 {DEFAULT_SRC}）")
    p.add_argument("--out-dir", default="",
                   help="报告/压缩数据输出目录（默认：--src 所在目录）")
    p.add_argument("--source-name", default="",
                   help="报告中的数据源名（默认：文件名 stem（压测））")
    p.add_argument("--key-entities", default="",
                   help="事实质量抽检的关键实体，逗号分隔（缺省取前 2 个实体）")
    p.add_argument("--key-clues", default="",
                   help="跨章呼应探测的人工金标准线索词，逗号分隔（缺省取 motif 层）")
    p.add_argument("--fallback-entity", default="",
                   help="CSA 无出场实体时的兜底召回实体（缺省取首个实体）")
    p.add_argument("--skill", default="romance",
                   help="HCA 题材 skill id（data/src/skills/*.yaml，其 fact_profile 驱动提取）")
    p.add_argument("--cards-novel", default="",
                   help="卡先验来源作品 id（从 data/src/{id}/ 载入角色卡/物品卡名；"
                        "缺省不启用卡先验）")
    p.add_argument("--hca-scenes", default="battle,politics,romance",
                   help="HCA 装配的场景类型，逗号分隔")
    p.add_argument("--garbage", default="远比表面,上去要,潮水般涌",
                   help="旧版垃圾主语抽检片段，逗号分隔")
    p.add_argument("--llm-facts", action="store_true",
                   help="LLM 双轨对照：逐章 FactExtractionService 提取（需 litellm 后端）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    compressed_dir = out_dir / COMPRESSED_DIR_NAME
    source_name = args.source_name or f"{src.stem}（压测）"
    key_entities = [e for e in args.key_entities.split(",") if e]
    key_clues = [c for c in args.key_clues.split(",") if c]
    fallback_entity = args.fallback_entity or (key_entities[0] if key_entities else "")
    hca_scenes = [s.strip() for s in args.hca_scenes.split(",") if s.strip()]
    garbage_fragments = [g for g in args.garbage.split(",") if g]
    counter = get_counter()
    compressed_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. 读入 + 切章 ──
    t0 = time.perf_counter()
    with io.open(src, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    chapters = split_chapters(text)
    print(f"[0] 读取 {src.name}: {len(text):,} 字符, 切出 {len(chapters)} 章, "
          f"{time.perf_counter() - t0:.1f}s")
    if 0 in chapters:
        chapters.pop(0)
    if not chapters:
        raise SystemExit(f"[FATAL] {src.name} 未切出任何章（行首需为『第N章』）")
    chapter_store = {no: ch for no, ch in chapters.items()}
    full_text = "\n".join(chapters.values())

    # ── 1. 全书 Token 实测 vs 系数预测 ──
    t1 = time.perf_counter()
    no_space = len(re.sub(r"\s", "", text))
    full_tokens = counter.count(text)
    predict_tokens = no_space * 1.8
    ratio = full_tokens / no_space if no_space else 0
    print(f"[1] 全书去空白 {no_space:,} 字 → 实测 {full_tokens:,} token "
          f"(系数 {ratio:.3f}, 预测 {predict_tokens:,.0f}) [{time.perf_counter() - t1:.1f}s]")

    # ── 2. SWA 装配（丢章明细） ──
    swa_asm = SwaAssembler(window_chapters=3)
    swa_rows: list[dict] = []
    for no in sorted(chapters):
        t = time.perf_counter()
        win = swa_asm.build(no, chapter_store, max_tokens=32000)
        dt = (time.perf_counter() - t) * 1000
        candidates = [n for n in range(no, max(0, no - 3), -1) if n in chapter_store]
        dropped = [n for n in candidates if n not in win.chapter_nos]
        cur_tokens = counter.count(chapter_store[no])
        swa_rows.append({
            "chapter": no,
            "tokens": win.token_count,
            "chapters_kept": win.chapter_nos,
            "dropped": dropped,
            "dropped_tokens": sum(counter.count(chapter_store[n]) for n in dropped),
            "over_budget_alone": cur_tokens > 32000,
            "ms": round(dt, 2),
        })

    # ── 2b. 档位 B 模拟（15,000 字/章截断） ──
    tier_b_store = {
        no: counter.truncate_to_tokens(chapter_store[no], 27_000)
        for no in sorted(chapters)
    }
    tier_b_rows = []
    for no in sorted(chapters):
        win = swa_asm.build(no, tier_b_store, max_tokens=32000)
        tier_b_rows.append({"chapter": no, "tokens": win.token_count,
                            "chapters_kept": len(win.chapter_nos)})

    # ── 3. CSA 装配：题材 profile 提取 → 时序索引 → 窗口化召回 ──
    skill_reg = SkillRegistry()
    skill_reg.load_dir(SKILLS_DIR)
    hca_skill = skill_reg.get(args.skill)
    profile = get_profile(getattr(hca_skill, "fact_profile", "cultivation"))

    # 卡先验（--cards-novel 从 registry 载入）
    card_names: set[str] = set()
    item_names: set[str] = set()
    if args.cards_novel:
        from app.infra.card_repo import RegistryPool

        from app.config import settings

        try:
            reg = RegistryPool(settings.src_dir).get(args.cards_novel)
            card_names = {c.name for c in reg.cards.values() if c.name}
            item_names = {i.name for i in reg.list_items() if i.name}
        except Exception as exc:  # noqa: BLE001
            print(f"[3] 卡先验载入失败（忽略）: {exc}")
    print(f"[3] 提取 profile: {profile.id} | 卡先验: "
          f"{len(card_names)} 角色 + {len(item_names)} 物品")

    t3 = time.perf_counter()
    for stale in (out_dir / "benchmark_timeline.db",):
        if stale.exists():
            stale.unlink()
    stale_facts = out_dir / "benchmark_facts"
    if stale_facts.exists():
        shutil.rmtree(stale_facts)
    stale_meta = out_dir / "meta"  # M12: 冷库登记表（防上次运行残留污染去重）
    if stale_meta.exists():
        shutil.rmtree(stale_meta)
    tmp_index = TimelineIndex(out_dir / "benchmark_timeline.db")
    facts_store = FactsStore(stale_facts)
    narrative = NarrativeRegistry(out_dir / "meta")

    t3a = time.perf_counter()
    discovery = discover_entities(
        full_text, min_freq=8, profile=profile,
        card_names=card_names, item_names=item_names, chapters=chapters,
    )
    entities, motifs = discovery.entities, discovery.motifs
    print(f"[3a] 实体集 {len(entities)} 个 | motif 线索 {len(motifs)} 个 "
          f"[{time.perf_counter() - t3a:.1f}s]")

    all_facts: dict[int, list[FactTriple]] = {}
    for no, content in chapters.items():
        facts = extract_heuristic_facts(content, no, entities, motifs, profile)
        all_facts[no] = facts
        if facts:
            tmp_index.add("bench", no, facts)
            facts_store.save("bench", no, facts)
        # M12: 句子向量库（冷库检索引擎存储侧）
        sentences = split_sentences(content)
        if sentences:
            pairs = [(s, v) for s, v in zip(sentences, _embed(sentences))]
            tmp_index.add_sentences("bench", no, pairs)
    total_facts = sum(len(v) for v in all_facts.values())
    print(f"[3b] 启发式事实: {total_facts} 条 [{time.perf_counter() - t3:.1f}s]")

    # M12: 冷库登记（伏笔/阵营）+ 建卡提议统计
    foreshadow_count = faction_count = 0
    for no, fs in all_facts.items():
        for f in fs:
            if f.predicate == "约定":
                if narrative.register_foreshadow(
                    "bench", subject=f.subject,
                    keyword=extract_promise_keyword(f.object, motifs),
                    chapter=no, summary=f.object,
                ):
                    foreshadow_count += 1
            elif f.predicate in ("加入", "结盟", "联合", "效忠", "宣战"):  # P4: 西幻阵营词
                narrative.register_faction("bench", name=f.object,
                                           member=f.subject, chapter=no)
                faction_count += 1
    print(f"[3c] 冷库登记: 伏笔 {foreshadow_count} 条 | 阵营 {faction_count} 个")

    # 质量抽检
    garbage_subjects = sorted({f.subject for fs in all_facts.values() for f in fs} - entities)
    legacy_garbage = [
        f.subject for fs in all_facts.values() for f in fs
        if any(b in f.subject for b in garbage_fragments)
    ]
    print(f"[3d] 非实体主语 {len(garbage_subjects)} | 垃圾片段 {len(legacy_garbage)} "
          f"| 关键实体 {key_entities} 保留: "
          f"{all(x in entities for x in key_entities)}")

    embedding = get_embedding_service()
    if isinstance(embedding, NullEmbedding):
        # 无配置时用确定性 HashEmbedding 兜底（保证句子库/语义召回可测可复现）
        embedding = HashEmbedding()
        print("[3e] embedding 未配置 → 使用 HashEmbedding（确定性兜底）")
    csa_asm = CsaAssembler(
        tmp_index,
        embedding=embedding,
        min_facts=3, vector_topk=0,
        narrative=narrative,
        semantic_topk=5,
        window_chapters=HOT_ZONE_CHAPTERS,
    )
    csa_rows: list[dict] = []
    for no in sorted(chapters):
        t = time.perf_counter()
        subjects = {f.subject for f in all_facts.get(no, [])}
        ctx = csa_asm.build("bench", list(subjects) or [fallback_entity],
                            current_chapter_no=no, limit_per_subject=5)
        dt = (time.perf_counter() - t) * 1000
        csa_rows.append({
            "chapter": no, "facts": len(ctx.compressed.facts),
            "sentences": len(ctx.compressed.sentences),
            "foreshadows": len(ctx.foreshadows),
            "tokens": ctx.token_count, "source": ctx.source, "ms": round(dt, 2),
        })

    # ── 4. HCA 装配 ──
    hca_asm = HcaAssembler(f"全书大纲（{src.stem}）", skill=hca_skill)
    hca_rows: list[dict] = []
    for scene in hca_scenes:
        t = time.perf_counter()
        bundle = hca_asm.build(scene, max_tokens=2000)
        hca_rows.append({
            "scene": scene, "blocks": [b.id for b in bundle.rule_blocks],
            "tokens": bundle.token_count,
            "ms": round((time.perf_counter() - t) * 1000, 2),
        })

    # ── 5. LLM 双轨对照（--llm-facts） ──
    llm_report: dict | None = None
    if args.llm_facts:
        llm_report = _run_llm_facts(chapters, tmp_index, csa_rows)

    # ── 6. 信息完整性评估 ──
    last_no = sorted(chapters)[-1]
    last_tokens = counter.count(chapters[last_no])
    win_1m = SwaAssembler(window_chapters=3).build(
        last_no, chapter_store, max_tokens=128_000
    )
    swa_window_set = set(win_1m.chapter_nos)
    csa_all_text = " ".join(
        f"{f.subject}{f.predicate}{f.object}" for fs in all_facts.values() for f in fs
    )
    sent_all_text = " ".join(
        s for no in chapters for s in _sentences_of(chapters[no])
    )

    motif_coverage: list[dict] = []
    for m in sorted(motifs):
        appear = [no for no, c in chapters.items() if m in c]
        csa_hit = [no for no, fs in all_facts.items()
                   if any(m in f.object or m in f.subject for f in fs)]
        motif_coverage.append({
            "motif": m,
            "freq": full_text.count(m),
            "appear_chapters": len(appear),
            "first_chapter": min(appear) if appear else 0,
            "last_chapter": max(appear) if appear else 0,
            "csa_chapters": len(csa_hit),
            "in_csa_any": m in csa_all_text,
            "in_sentences": m in sent_all_text,
        })

    # 跨章呼应探测：写最后一章时，线索早期信息的三层保留矩阵
    echo_probe: list[dict] = []
    for clue in sorted(set(key_clues) | set(motifs)):
        appear = [no for no, c in chapters.items() if clue in c]
        if len(appear) < 2:
            continue
        early = [n for n in appear if n not in swa_window_set]
        if not early:
            continue  # 早期出现都在 SWA 窗口内 → 安全
        lost = not (clue in csa_all_text or clue in sent_all_text)
        echo_probe.append({
            "clue": clue,
            "appear_chapters": appear,
            "outside_swa": early,
            "in_csa": clue in csa_all_text,
            "in_sentences": clue in sent_all_text,
            "risk": lost,
        })

    # 人物关系保留率：共现人物对 vs 关系类事实
    char_pairs: set[tuple[str, str]] = set()
    rel_facts = 0
    for no, fs in all_facts.items():
        for f in fs:
            if f.predicate in ("牵手", "拥抱", "告白", "和好", "吵架", "结婚",
                               "结盟", "联合", "合作", "敌对", "约定"):
                rel_facts += 1
    for no, content in chapters.items():
        present = [e for e in entities if e in content]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                char_pairs.add(tuple(sorted((present[i], present[j]))))

    # 伏笔可及性：写最后一章时，窗口外伏笔是否经 CSA 注入可及
    foreshadow_entries = narrative.list_foreshadows("bench", status="open")
    fs_out_of_swa = [
        f for f in foreshadow_entries if f["planted_chapter"] not in swa_window_set
    ]
    last_subjects = {f.subject for f in all_facts.get(last_no, [])} or {fallback_entity}
    last_ctx = csa_asm.build("bench", list(last_subjects),
                             current_chapter_no=last_no, limit_per_subject=5)
    injected_text = " ".join(last_ctx.foreshadows)
    fs_csa_cover = sum(
        1 for f in fs_out_of_swa
        if f["keyword"] in injected_text or f["subject"] in injected_text
    )

    compressed_context = (
        win_1m.token_count + csa_rows[-1]["tokens"] + hca_rows[0]["tokens"]
    )
    compress_ratio = last_tokens / max(1, compressed_context)
    per_chapter_ctx = compressed_context + 1600
    coverage_1m = 1_000_000 // max(1, per_chapter_ctx)
    coverage_raw = 1_000_000 // max(1, last_tokens)

    # ── 7. 压缩数据导出 ──
    def build_packet(no: int) -> dict:
        row = next(r for r in swa_rows if r["chapter"] == no)
        swa_chapter_nos = row["chapters_kept"]
        swa_preview = {
            n: (chapter_store[n][:800] + ("…" if len(chapter_store[n]) > 800 else ""))
            for n in swa_chapter_nos
        }
        subjects = {f.subject for f in all_facts.get(no, [])} or {fallback_entity}
        ctx = csa_asm.build("bench", list(subjects),
                            current_chapter_no=no, limit_per_subject=5)
        epoch_facts = [f for n in range(max(1, no - 4), no + 1) for f in all_facts.get(n, [])]
        epoch_summary = "\n".join(
            f"ch{f.chapter_no}: {f.subject}{f.predicate}{f.object}" for f in epoch_facts[:40]
        )
        bundle = hca_asm.build(hca_scenes[0], max_tokens=2000)
        return {
            "chapter_no": no,
            "source": source_name,
            "说明": "SWA=近章原文(压缩前), CSA=窗口事实+伏笔+阵营+语义句, "
                    "HCA=题材规则块, facts_epoch=前5章剧情摘要",
            "swa": {
                "chapter_nos": swa_chapter_nos,
                "tokens": row["tokens"],
                "dropped": row["dropped"],
                "preview_800_chars": swa_preview,
            },
            "csa": {
                "tokens": ctx.token_count,
                "source": ctx.source,
                "facts_text": list(ctx.compressed.facts),
                "sentences": list(ctx.compressed.sentences),
                "foreshadows": list(ctx.foreshadows),
                "factions": list(ctx.factions),
            },
            "hca": {
                "blocks": [b.id for b in bundle.rule_blocks],
                "tokens": bundle.token_count,
                "contents": [b.content for b in bundle.rule_blocks],
            },
            "facts_epoch": {
                "epoch": no // 5,
                "summary_tokens": counter.count(epoch_summary),
                "summary": epoch_summary,
            },
        }

    exports: list[Path] = []
    for no in list(sorted(chapters))[:3] + [last_no]:
        packet = build_packet(no)
        path = compressed_dir / f"packet_ch{no:03d}.json"
        path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
        exports.append(path)

    # ── 报告 ──
    swa_total = sum(r["tokens"] for r in swa_rows)
    dropped_total = sum(1 for r in swa_rows if r["dropped"])
    over_budget = [r["chapter"] for r in swa_rows if r["over_budget_alone"]]
    csa_total = sum(r["tokens"] for r in csa_rows)
    tier_b_dropped = sum(1 for r in tier_b_rows if r["chapters_kept"] < 3)
    tier_b_swa = sum(r["tokens"] for r in tier_b_rows)

    report = {
        "source": str(src),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "skill": args.skill,
        "fact_profile": profile.id,
        "stats": {
            "total_chars": len(text),
            "chars_no_space": no_space,
            "total_tokens_measured": full_tokens,
            "tokens_per_char": round(ratio, 4),
            "predicted_total_tokens_1p8": round(predict_tokens),
            "chapters": len(chapters),
            "avg_tokens_per_chapter": round(full_tokens / max(1, len(chapters))),
        },
        "swa": {
            "budget_max_tokens": 32000,
            "chapters_with_drop": dropped_total,
            "chapters_over_budget_alone": over_budget,
            "total_swa_tokens": swa_total,
            "avg_ms": round(sum(r["ms"] for r in swa_rows) / max(1, len(swa_rows)), 2),
            "drops": [
                {"chapter": r["chapter"], "dropped": r["dropped"],
                 "dropped_tokens": r["dropped_tokens"]}
                for r in swa_rows if r["dropped"]
            ],
        },
        "tier_b_simulation": {
            "chapters_truncated_to_15k_chars": len(tier_b_rows),
            "swa_drop_with_15k_chapters": tier_b_dropped,
            "total_swa_tokens_15k_chapters": tier_b_swa,
            "avg_swa_per_chapter_15k": round(tier_b_swa / max(1, len(tier_b_rows))),
        },
        "csa": {
            "total_facts": total_facts,
            "total_csa_tokens": csa_total,
            "avg_ms": round(sum(r["ms"] for r in csa_rows) / max(1, len(csa_rows)), 2),
            "sentence_recall_hits": sum(r["sentences"] for r in csa_rows),
            "foreshadow_injections": sum(r["foreshadows"] for r in csa_rows),
            "hot_zone_chapters": HOT_ZONE_CHAPTERS,
        },
        "fact_quality": {
            "entities": len(entities),
            "motifs": sorted(motifs),
            "facts": total_facts,
            "key_entities": key_entities,
            "key_entities_kept": all(x in entities for x in key_entities),
            "non_entity_subjects": len(garbage_subjects),
            "garbage_fragments": len(legacy_garbage),
            "card_prior": {
                "enabled": bool(args.cards_novel),
                "card_names": sorted(card_names),
                "item_names": sorted(item_names),
            },
            "note": f"profile={profile.id}；实体/motif/卡先验；登场已移除",
        },
        "narrative": {
            "foreshadows": foreshadow_count,
            "factions": faction_count,
            "open_foreshadows": len(foreshadow_entries),
            "outside_swa_foreshadows": len(fs_out_of_swa),
        },
        "hca": hca_rows,
        "one_million_window": {
            "last_chapter": last_no,
            "last_chapter_tokens": last_tokens,
            "swa_1m_tokens": win_1m.token_count,
            "compressed_context_tokens": compressed_context,
            "compress_ratio": round(compress_ratio, 3),
            "coverage_1m_compressed_chapters": coverage_1m,
            "coverage_1m_raw_chapters": coverage_raw,
            "meaning": "1M 窗口可覆盖的章数（压缩 vs 原文）",
        },
        "coverage": {
            "motif_coverage": motif_coverage,
            "echo_probe": echo_probe,
            "relationship_retention": {
                "character_pairs": len(char_pairs),
                "relation_facts": rel_facts,
            },
            "foreshadow_reachability": {
                "open": len(foreshadow_entries),
                "outside_swa": len(fs_out_of_swa),
                "injected_via_csa": fs_csa_cover,
            },
        },
        "budget_hit_rate": {
            "swa_within_32k": round(
                sum(1 for r in swa_rows if r["tokens"] <= 32000) / max(1, len(swa_rows)), 3
            ),
            "hca_within_2k": round(
                sum(1 for r in hca_rows if r["tokens"] <= 2000) / max(1, len(hca_rows)), 3
            ),
        },
        "llm_facts": llm_report,
        "files": [str(p) for p in exports],
    }
    report_path = out_dir / "benchmark_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_md(report, src, args, key_entities, garbage_fragments, hca_rows,
              swa_rows, tier_b_rows, all_facts, entities, motifs, no_space,
              full_tokens, ratio, predict_tokens, swa_total, dropped_total,
              over_budget, tier_b_dropped, tier_b_swa, total_facts, csa_total,
              last_no, last_tokens, win_1m, compressed_context, compress_ratio,
              coverage_1m, coverage_raw, exports, profile, key_clues, out_dir,
              narrative, foreshadow_count, faction_count, echo_probe,
              motif_coverage, char_pairs, rel_facts, fs_out_of_swa, fs_csa_cover)

    print(f"[6] 报告: {report_path} / {out_dir / 'benchmark_report.md'}")
    print(f"    压缩数据: {compressed_dir} ({len(exports)} 个 packet)")
    print(f"[OK] 全书 {full_tokens:,} token | 实体 {len(entities)} / motif {len(motifs)} "
          f"| 事实 {total_facts} | 伏笔 {foreshadow_count} | "
          f"1M 窗口覆盖 {coverage_1m} 章压缩上下文 (原文 {coverage_raw} 章)")


# ═══════════════════════════ helpers ═══════════════════════════

_emb_cache: dict | None = None


def _embed(sentences: list[str]) -> list[list[float]]:
    global _emb_cache
    if _emb_cache is None:
        _emb_cache = get_embedding_service()
    if isinstance(_emb_cache, NullEmbedding):
        # 无 embedding 服务时用确定性 HashEmbedding 兜底（保证句子库可测）
        _emb_cache = HashEmbedding()
    return _emb_cache.embed(sentences)


def _sentences_of(content: str) -> list[str]:
    return split_sentences(content)


def _csa_facts_for(all_facts: dict, no: int, entities, motifs, profile) -> list[str]:
    return [
        f"{f.subject}{f.predicate}{f.object}" for f in all_facts.get(no, [])
    ]


def _run_llm_facts(chapters: dict, index: TimelineIndex, csa_rows) -> dict:
    """LLM 双轨对照：逐章 FactExtractionService 提取（litellm 后端）。"""
    from app.config import settings

    if settings.llm_backend != "litellm":
        return {"skipped": True, "reason": "llm_backend != litellm（需真实 API key）"}

    from app.agent.fact_extractor import FactExtractionService
    from app.infra.llm_gateway import get_gateway

    service = FactExtractionService(get_gateway())
    total_tokens_est = 0
    facts_per_chapter: dict[int, int] = {}

    async def _run() -> list[tuple[int, int]]:
        results = []
        for no, content in chapters.items():
            res = await service.extract(content, no, on_stage=[])
            facts_per_chapter[no] = len(res.facts)
            total_est = len(content) * 1.5 + 300  # 粗估 prompt+输出 token
            results.append((no, total_est))
        return results

    rows = asyncio.run(_run())
    total_tokens_est = sum(r for _, r in rows)
    llm_facts_total = sum(facts_per_chapter.values())
    return {
        "skipped": False,
        "facts_total": llm_facts_total,
        "facts_per_chapter": facts_per_chapter,
        "estimated_tokens": total_tokens_est,
        "note": "LLM 提取对照（T=0.0 通用 prompt）；成本 = 估算 prompt+输出 token",
    }


def _write_md(report, src, args, key_entities, garbage_fragments, hca_rows,
              swa_rows, tier_b_rows, all_facts, entities, motifs, no_space,
              full_tokens, ratio, predict_tokens, swa_total, dropped_total,
              over_budget, tier_b_dropped, tier_b_swa, total_facts, csa_total,
              last_no, last_tokens, win_1m, compressed_context, compress_ratio,
              coverage_1m, coverage_raw, exports, profile, key_clues, out_dir,
              narrative, foreshadow_count, faction_count, echo_probe,
              motif_coverage, char_pairs, rel_facts, fs_out_of_swa, fs_csa_cover) -> None:
    lines = [
        f"# 文枢 1M 上下文窗口压测报告（{src.stem}）",
        "",
        f"- 数据源：`{src.name}`（约 {report['stats']['total_chars']:,} 字符 / "
        f"{report['stats']['chapters']} 章）",
        f"- 生成时间：{report['generated_at']}",
        f"- HCA 题材 skill：`{args.skill}` · 事实提取 profile：`{profile.id}`",
        "",
        "## 1. 全书 Token 实测（tiktoken cl100k_base）",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 去空白字符数 | {no_space:,} |",
        f"| 实测全书 token | {full_tokens:,} |",
        f"| 实测系数 | **{ratio:.3f} token/字** |",
        f"| token-budget.md 预测系数 | 1.8 token/字（偏差 {(ratio - 1.8) / 1.8 * 100:+.1f}%） |",
        f"| 平均每章 | {full_tokens // max(1, report['stats']['chapters']):,} token |",
        "",
        "## 2. SWA 装配（窗口 3 章，预算 32,000 token）",
        "",
        f"- 触发『丢最老章』降级的章数：**{dropped_total} / {report['stats']['chapters']}**"
        f"（{dropped_total / max(1, report['stats']['chapters']) * 100:.0f}%）",
        f"- SWA 合计 token：{swa_total:,}（全书视角）",
        f"- **当前章单独超预算**（整章 >32k，无条件保留）：{over_budget or '无'}",
        f"- 预算命中率（≤32,000）：{report['budget_hit_rate']['swa_within_32k'] * 100:.0f}%",
        "",
        "### 丢章明细（Q4：丢弃的具体章号）",
        "",
        "| 写章 | 丢弃章号 | 丢弃 token |",
        "|---|---|---|",
    ]
    drops = report["swa"]["drops"]
    if drops:
        for d in drops:
            lines.append(
                f"| ch{d['chapter']} | {', '.join(f'ch{n}' for n in d['dropped']) or '—'} "
                f"| {d['dropped_tokens']:,} |"
            )
    else:
        lines.append("| — | 无丢弃 | — |")
    lines += [
        "",
        "## 2b. 档位 B 模拟（15,000 字/章截断）",
        "",
        f"- 截断后 SWA 平均 {tier_b_swa // max(1, len(tier_b_rows)):,} token/章",
        f"- 触发『丢最老章』：**{tier_b_dropped} / {len(tier_b_rows)} 章**"
        f"（该档位假设下窗口无法完整容纳 3 章）",
        "",
        "## 3. CSA 装配（M12：热区窗口 + 冷库登记 + 语义句召回）",
        "",
        f"- 事实提取 profile：`{profile.id}`（SkillSpec.fact_profile 驱动）",
        f"- 实体集：**{len(entities)} 个** | motif 线索层：**{len(motifs)} 个**"
        f"{f"（{', '.join(sorted(motifs))}）" if motifs else ''}",
        f"- 卡先验：{'启用（' + ', '.join(report['fact_quality']['card_prior']['card_names']) + '）' if report['fact_quality']['card_prior']['enabled'] else '未启用'}",
        f"- 启发式事实总数：**{total_facts} 条**（登场已移除；事件/关系/约定/信物）",
        f"- 冷库登记：伏笔 **{foreshadow_count}** 条 | 阵营 **{faction_count}** 个",
        f"- 语义句召回命中：**{report['csa']['sentence_recall_hits']}** 句"
        f"（STORYNEXUS_EMBEDDING 非空时；hash 确定性兜底）",
        f"- 伏笔装配注入：**{report['csa']['foreshadow_injections']}** 次",
        f"- CSA 热区窗口：近 **{HOT_ZONE_CHAPTERS}** 章事实（远古由冷库/检索接管）",
        f"- 非实体主语：**{report['fact_quality']['non_entity_subjects']}**；"
        f"旧版垃圾片段（{'/'.join(garbage_fragments)}）：**{report['fact_quality']['garbage_fragments']}**",
        f"- 关键实体保留：{' / '.join(key_entities)} = "
        f"{'全部保留' if report['fact_quality']['key_entities_kept'] else '缺失！'}",
        "",
        f"## 4. HCA 装配（{args.skill} skill）",
        "",
        "| 场景 | 规则块 | token |",
        "|---|---|---|",
    ]
    for r in hca_rows:
        lines.append(f"| {r['scene']} | {', '.join(r['blocks'])} | {r['tokens']} |")
    lines += [
        "",
        f"## 5. 1M 窗口场景（最后一章 ch{last_no}）",
        "",
        "| 项 | token |",
        "|---|---|",
        f"| 最后一章原文 | {last_tokens:,} |",
        f"| SWA 1M 段（窗口 3 章 ≤128k） | {win_1m.token_count:,} |",
        f"| 压缩后单章上下文 | {compressed_context:,} |",
        f"| **压缩率** | **1 : {1 / max(compress_ratio, 1e-9):.1f}**（压缩后为原文 "
        f"{compress_ratio * 100:.0f}%） |",
        f"| **1M 窗口覆盖** | **{coverage_1m} 章压缩上下文**（原文只能装 {coverage_raw} 章） |",
        "",
        "## 6. 信息完整性评估（Q3：压缩后是否保留关键剧情节点）",
        "",
        "### 6a. motif 线索覆盖率",
        "",
        "| 线索 | 词频 | 出现章数 | 首末章 | CSA 事实覆盖章数 | 在 CSA/句子库 |",
        "|---|---|---|---|---|---|",
    ]
    for m in motif_coverage:
        lines.append(
            f"| {m['motif']} | {m['freq']} | {m['appear_chapters']} | "
            f"ch{m['first_chapter']}-ch{m['last_chapter']} | {m['csa_chapters']} | "
            f"{'是' if (m['in_csa_any'] or m['in_sentences']) else '**否·丢失**'} |"
        )
    lines += [
        "",
        "### 6b. 跨章呼应探测（写最后一章时，早期线索的保留矩阵）",
        "",
        "| 线索 | 出现章 | 窗口外早期章 | CSA | 句子库 | 风险 |",
        "|---|---|---|---|---|---|",
    ]
    if echo_probe:
        for e in echo_probe:
            lines.append(
                f"| {e['clue']} | {','.join(f'ch{n}' for n in e['appear_chapters'])} | "
                f"{','.join(f'ch{n}' for n in e['outside_swa'])} | "
                f"{'✓' if e['in_csa'] else '✗'} | {'✓' if e['in_sentences'] else '✗'} | "
                f"{'**丢失风险**' if e['risk'] else '可及'} |"
            )
    else:
        lines.append("| — | 无跨章线索 | — | — | — | 安全 |")
    lines += [
        "",
        "### 6c. 人物关系保留率 + 伏笔可及性",
        "",
        f"- 共现人物对：**{len(char_pairs)}** 对 | 关系类事实：**{rel_facts}** 条",
        f"- 开放伏笔：**{len(fs_out_of_swa)}** 条在 SWA 窗口外；"
        f"经 CSA 伏笔注入可及 **{fs_csa_cover}** 条（其余依赖语义句/工具查询）",
        "",
        "### 6d. LLM 双轨对照",
        "",
    ]
    llm = report["llm_facts"]
    if llm is None:
        lines.append("- 未启用（--llm-facts 需 litellm 后端 + API key）")
    elif llm.get("skipped"):
        lines.append(f"- 已跳过：{llm.get('reason')}")
    else:
        lines.append(
            f"- LLM 提取事实：**{llm['facts_total']}** 条（启发式 {total_facts} 条）"
            f" | 估算 token 成本：{llm['estimated_tokens']:,}"
        )
    lines += [
        "",
        "## 7. 压缩数据文件",
        "",
    ]
    lines += [f"- `{p.name}`" for p in exports]
    md_path = out_dir / "benchmark_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
