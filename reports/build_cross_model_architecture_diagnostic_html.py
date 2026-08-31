#!/usr/bin/env python3
"""Build the self-contained HTML edition of the architecture diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import build_future_research_roadmap_html as shared


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md"
OUTPUT = HERE / "CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC_STANDALONE.html"


def replace_required(document: str, old: str, new: str) -> str:
    if old not in document:
        raise RuntimeError(f"shared HTML template marker was not found: {old[:80]!r}")
    return document.replace(old, new)


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    shared.SOURCE = SOURCE
    shared.OUTPUT = OUTPUT
    shared.main()
    document = OUTPUT.read_text(encoding="utf-8")
    replacements = {
        '<meta name="description" content="MS-CADM 后续研究路线决策、实验协议与论文蓝图">':
            '<meta name="description" content="风电场景生成跨模型诊断、架构选择与下一阶段实验门">',
        '<title>MS-CADM 后续研究路线决策与实验蓝图</title>':
            '<title>跨模型诊断与下一阶段架构路线</title>',
        'content: "RESEARCH ROADMAP";': 'content: "ARCHITECTURE DIAGNOSTIC";',
        '<a class="brand" href="#top">MS-CADM 后续路线</a>':
            '<a class="brand" href="#top">跨模型诊断与下一阶段路线</a>',
        '<div class="hero-kicker">RESEARCH ROADMAP · DECISION DOCUMENT</div>':
            '<div class="hero-kicker">CROSS-MODEL DIAGNOSTIC · FALSIFIABLE NEXT ROUTE</div>',
        '<h1>MS-CADM 后续研究路线<br>决策与实验蓝图</h1>':
            '<h1>风电场景生成跨模型诊断<br>与下一阶段架构路线</h1>',
        '<p>从已确认的仓库证据出发，明确下一步研究问题、模型结构、消融、统计协议、停止门槛与投稿边界。</p>':
            '<p>在严格配对的冻结场景档案上，对 ramp、跨时滞依赖、边界状态持续期和 NWP 天气分层做机制诊断，再决定 diffusion、flow 与状态空间路线。</p>',
        '<div class="stat"><strong>3</strong><span>条分级主路线</span></div>':
            '<div class="stat"><strong>300</strong><span>个不重复日期，两面板分开</span></div>',
        "生成源：FUTURE_RESEARCH_ROADMAP.md":
            "生成源：CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md",
        "生成器：build_future_research_roadmap_html.py":
            "生成器：build_cross_model_architecture_diagnostic_html.py（共享离线模板）",
    }
    for old, new in replacements.items():
        document = replace_required(document, old, new)
    OUTPUT.write_text(document, encoding="utf-8")
    print(
        json.dumps(
            {
                "source": str(SOURCE),
                "output": str(OUTPUT),
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "bytes": OUTPUT.stat().st_size,
                "mode": "self_contained_offline_html",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
