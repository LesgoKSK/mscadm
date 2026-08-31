from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in config.get("models", {}):
        raise KeyError(f"Model {name!r} is absent from configuration")
    result = deepcopy(config["models"][name])
    result.setdefault("seed", config.get("seed", 0))
    result.setdefault("device", config.get("device", "cuda"))
    return result


__all__ = ["load_config", "model_config"]
