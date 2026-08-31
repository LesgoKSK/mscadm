"""Create a fail-closed publication lock before confirmation access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ps_dfsc.manifest_publication import (
    build_lock_manifest,
    verify_lock_manifest,
)


def _assert_confirmation_absent(base_output: str | Path) -> None:
    root = Path(base_output)
    if not root.exists():
        return
    forbidden = [
        path
        for path in root.rglob("*")
        if path.is_file() and "confirmation_test" in path.name
    ]
    if forbidden:
        raise RuntimeError(
            "confirmation artifacts already exist before lock: "
            + ", ".join(str(path) for path in forbidden[:5])
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the PS-DFSC publication lock manifest"
    )
    parser.add_argument("--splits", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data-files", nargs="+", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--base-output", default="outputs/ps_dfsc/base")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _assert_confirmation_absent(args.base_output)
    manifest = build_lock_manifest(
        split_registry=args.splits,
        config_path=args.config,
        model_paths=args.models,
        data_paths=args.data_files,
        selection_path=args.selection,
        outer=args.outer,
    )
    verify_lock_manifest(manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "outer": args.outer,
                "candidate_id": manifest["candidate_id"],
                "used_identity_fallback": manifest[
                    "used_identity_fallback"
                ],
                "status": "locked_before_confirmation_access",
            }
        )
    )


if __name__ == "__main__":
    main()
