# 文枢 StoryNexus 压缩引擎（storynexus-compress）

> 长篇小说上下文压缩引擎的**独立发布包**：SWA/CSA/HCA 三层压缩 + 题材化事实提取
> + 多语言实体发现（中文/拉丁/音译）+ 压测基准。
>
> [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

本仓库是 [storynexus](https://github.com/wdsjdr/storynexus)（完整 AI 写作平台）的
**上下文压缩子项目**——从主仓抽取的可独立运行核心，聚焦"如何在 1M 上下文窗口内
把长篇小说压缩为高质量写作上下文"。

## 核心能力

| 模块 | 说明 |
|---|---|
| **SWA** 滑动窗口 | 当前章 + 近 N 章原文，预算内丢最老章（含当前章超预算标记） |
| **CSA** 稀疏压缩 | 事实三元组时序索引（A 主干）+ 章节向量（B 兜底）+ 句子向量语义召回（C）+ 冷库注入（伏笔/阵营） |
| **HCA** 重度压缩 | 全书大纲静态段 + 场景规则块切片（Skill YAML 驱动）+ 风格指纹 KL 守卫 |
| **事实提取** | 题材 profile 化（cultivation / slice / **western**）+ 四证据实体发现（称谓/姓氏/高频跨章/卡先验）+ motif 线索层 |
| **多语言基座** | 拉丁人名（Aragorn）+ 中文音译名（甘道夫/哈利·波特）+ 文言"X曰" + 英文断句/说话人回溯 |
| **压测基准** | `benchmark_1m.py`：丢章明细、motif 覆盖率、跨章呼应探测、伏笔可及性、三题材矩阵 |

## 快速开始

```bash
pip install -e .[bench,dev]

# 压测：古典演义（三国演义，120 回）
python -m scripts.benchmark_1m \
  --src samples/三国演义/三国演义.txt --skill default \
  --key-entities "刘备,曹操,诸葛亮,云长,孔明"

# 压测：西幻英文（绿野仙踪，24 章）
python -m scripts.benchmark_1m \
  --src samples/wizard_of_oz/wizard_of_oz.txt --skill western \
  --key-entities "Dorothy,Scarecrow,Toto,Wizard"
```

测试：`python -m pytest -q`

## 压测示例（samples/）

| 子目录 | 题材 | 要点 |
|---|---|---|
| `三国演义/` | 古典演义（cultivation） | 切章「第X回」、文言"X曰"模板、实体 126 / 事实 133 |
| `wizard_of_oz/` | 西幻英文（western） | 拉丁人名 39 个、Chapter N/I 切章、英文断句 |

每目录含 `<书名>.txt`（公版原文）+ `benchmark_report.md/json` + `compressed/packet_*.json`。
完整说明见 [samples/README.md](samples/README.md)。

## 与主仓的关系

- 主仓 [storynexus](https://github.com/wdsjdr/storynexus) 是完整平台（WebUI/写章管线/FSM/Dante），
  本包为其 `backend/app/context` + `backend/app/infra` 压缩相关核心的独立发布，
  代码同源、同许可（Apache-2.0）。
- 差异：本包裁剪了 LLM 网关/应用服务/前端等平台部分，并提供独立的 `PatchInstruction`
  最小定义（主仓在 `app/agent/evaluator.py`）；benchmark 与测试与主仓保持同步演进。

## 许可

Apache-2.0。示例数据为公版作品（《三国演义》罗贯中；*The Wonderful Wizard of Oz*
L. Frank Baum，Project Gutenberg #55）。
