from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command], stderr=subprocess.DEVNULL, text=True, encoding="utf-8"
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def scenario_record(path: Path) -> dict[str, object]:
    archive = np.load(path, allow_pickle=False)
    record: dict[str, object] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "scenario_shape": list(archive["scenarios"].shape),
        "observation_shape": list(archive["observations"].shape),
    }
    if "metadata" in archive.files:
        try:
            record["metadata"] = json.loads(str(archive["metadata"]))
        except json.JSONDecodeError:
            record["metadata_raw"] = str(archive["metadata"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a hash-verified experiment manifest")
    parser.add_argument("--root", default="outputs/full_reproduction")
    parser.add_argument("--config", default="repro_configs/paper.json")
    args = parser.parse_args()
    root, config = Path(args.root), Path(args.config)
    checkpoints = sorted(root.rglob("*.pt"))
    scenarios = sorted((root / "scenarios").rglob("*.npz"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_status": git(["status", "--short"]),
        "config": {
            "path": config.as_posix(),
            "bytes": config.stat().st_size,
            "sha256": sha256(config),
        },
        "checkpoints": [
            {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in checkpoints
        ],
        "scenarios": [scenario_record(path) for path in scenarios],
    }
    destination = root / "experiment_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destination.resolve())


if __name__ == "__main__":
    main()
