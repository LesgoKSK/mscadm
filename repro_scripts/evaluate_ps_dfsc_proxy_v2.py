"""Evaluate validation proxy with the registered per-unit Clarabel QP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import repro_scripts.evaluate_ps_dfsc_proxy as proxy
from ps_dfsc.fast_differentiable_suc_publication_v3 import (
    ClarabelPublicationDifferentiableSUC,
)


def main() -> None:
    proxy.PublicationDifferentiableSUC = (
        ClarabelPublicationDifferentiableSUC
    )
    proxy.main()
    try:
        output_index = sys.argv.index("--output") + 1
        output = Path(sys.argv[output_index])
    except (ValueError, IndexError) as error:
        raise RuntimeError("proxy output argument is unavailable") from error
    summary_path = output.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "schema": "ps_dfsc_validation_proxy_v2",
            "relaxation_solver": "Clarabel",
            "power_scaling": "per_unit_1200MW",
            "registered_strong_convexity": float(
                summary["strong_convexity"]
            ),
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
