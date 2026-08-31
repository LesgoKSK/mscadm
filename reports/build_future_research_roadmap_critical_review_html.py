#!/usr/bin/env python3
"""Build the self-contained HTML edition of the roadmap critical review.

The shared roadmap builder owns the offline CSS, navigation, image embedding,
dark mode, print layout, and search behavior.  This wrapper preserves that
single presentation implementation while substituting review-specific source
and cover metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import build_future_research_roadmap_html as shared


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "FUTURE_RESEARCH_ROADMAP_CRITICAL_REVIEW.md"
OUTPUT = HERE / "FUTURE_RESEARCH_ROADMAP_CRITICAL_REVIEW_STANDALONE.html"


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
            '<meta name="description" content="MS-CADM 后续研究路线十分详细复审、文献碰撞、实验协议与停止门">',
        "<title>MS-CADM 后续研究路线决策与实验蓝图</title>":
            "<title>MS-CADM 后续研究路线十分详细复审</title>",
        'content: "RESEARCH ROADMAP";': 'content: "CRITICAL REVIEW";',
        '<a class="brand" href="#top">MS-CADM 后续路线</a>':
            '<a class="brand" href="#top">MS-CADM 路线复审</a>',
        '<div class="hero-kicker">RESEARCH ROADMAP · DECISION DOCUMENT</div>':
            '<div class="hero-kicker">CRITICAL REVIEW · LITERATURE AUDIT · 2026</div>',
        '<h1>MS-CADM 后续研究路线<br>决策与实验蓝图</h1>':
            '<h1>MS-CADM 后续研究路线<br>十分详细复审</h1>',
        '<p>从已确认的仓库证据出发，明确下一步研究问题、模型结构、消融、统计协议、停止门槛与投稿边界。</p>':
            '<p>逐节红队审计原路线图，以代码与冻结结果校正事实，以 2023–2026 原始文献核验碰撞，并给出可证伪的新方案、实验协议与停止门。</p>',
        '<div class="stat"><strong>3</strong><span>条分级主路线</span></div>':
            '<div class="stat"><strong>7</strong><span>条重排研究路线</span></div>',
        "生成源：FUTURE_RESEARCH_ROADMAP.md":
            "生成源：FUTURE_RESEARCH_ROADMAP_CRITICAL_REVIEW.md",
        "生成器：build_future_research_roadmap_html.py":
            "生成器：build_future_research_roadmap_critical_review_html.py（共享离线模板）",
    }
    for old, new in replacements.items():
        document = replace_required(document, old, new)

    OUTPUT.write_text(document, encoding="utf-8")
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "source": str(SOURCE),
                "output": str(OUTPUT),
                "source_sha256": source_sha,
                "bytes": OUTPUT.stat().st_size,
                "mode": "critical_review_shared_offline_template",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
