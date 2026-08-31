"""Validate one completed formal candidate in an isolated process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repro_scripts.schedule_ps_dfsc_candidates_v16 import (
    _validate_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evidence = _validate_candidate(
        Path(args.checkpoint),
        outer=args.outer,
        beta=args.beta,
        seed=args.seed,
    )
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
