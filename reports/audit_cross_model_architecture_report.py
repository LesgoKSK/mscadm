#!/usr/bin/env python3
"""Fail-fast structural audit for the diagnostic report and its artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs" / "cross_model_diagnostics"
MARKDOWN = REPORTS / "CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md"
HTML = REPORTS / "CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC_STANDALONE.html"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def csv_rows(name: str) -> int:
    path = OUTPUTS / name
    require(path.is_file(), f"missing diagnostic CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    require(MARKDOWN.is_file(), f"missing report source: {MARKDOWN}")
    require(HTML.is_file(), f"missing standalone report: {HTML}")
    markdown = MARKDOWN.read_text(encoding="utf-8")
    document = HTML.read_text(encoding="utf-8")
    for token in (
        "结论先行",
        "Ramp 诊断",
        "Cross-lag",
        "Atom duration",
        "NWP regime",
        "架构路线",
        "状态空间",
        "解释边界",
    ):
        require(token in markdown, f"report is missing required topic: {token}")
    require("TODO" not in markdown and "PLACEHOLDER" not in markdown, "report has placeholders")
    require(document.startswith("<!doctype html>"), "standalone HTML lacks doctype")
    require('<meta charset="utf-8">' in document, "standalone HTML lacks UTF-8 charset")
    require("data:image/png;base64," in document, "standalone HTML contains no embedded PNG")
    require(len(re.findall(r"data:image/png;base64,", document)) >= 4, "fewer than four figures embedded")
    require("<script src=" not in document, "standalone HTML depends on an external script")
    require("<link rel=\"stylesheet\"" not in document, "standalone HTML depends on a stylesheet")
    require("mathjax.org" not in document.lower(), "report should not load MathJax")
    require("katex.min.js" not in document.lower(), "report should not load KaTeX")

    expected_csv = {
        "per_day_metrics.csv": 1,
        "primary_paired_bootstrap.csv": 1,
        "primary_regime_bootstrap.csv": 1,
        "primary_regime_interaction_bootstrap.csv": 1,
        "secondary_paired_bootstrap.csv": 1,
        "lag1_state_attribution_bootstrap.csv": 1,
        "lagged_variogram_summary.csv": 1,
        "atom_event_primary_bootstrap.csv": 1,
        "generated_transition_decomposition_summary.csv": 1,
    }
    counts = {name: csv_rows(name) for name in expected_csv}
    require(all(counts[name] >= minimum for name, minimum in expected_csv.items()), "an output CSV is empty")

    manifest = json.loads((OUTPUTS / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = OUTPUTS / entry["path"]
        require(path.is_file(), f"manifest target missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == entry["sha256"], f"manifest checksum mismatch: {path.name}")

    result = {
        "markdown_bytes": MARKDOWN.stat().st_size,
        "html_bytes": HTML.stat().st_size,
        "embedded_png_count": len(re.findall(r"data:image/png;base64,", document)),
        "csv_rows": counts,
        "manifest_files_verified": len(manifest["files"]),
        "status": "PASS",
    }
    qa_path = REPORTS / "qa" / "cross_model_architecture_diagnostic_audit.json"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result["audit_file"] = str(qa_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
