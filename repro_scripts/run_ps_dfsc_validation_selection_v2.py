"""Validation/locking automation using the audited v3 training config."""

from __future__ import annotations

from pathlib import Path

import repro_scripts.run_ps_dfsc_validation_selection_v1 as pipeline


CONFIG_V2 = pipeline.ROOT / "repro_configs" / "ps_dfsc_v2.json"
CONFIG_V3 = pipeline.ROOT / "repro_configs" / "ps_dfsc_v3.json"
ORIGINAL_PYTHON = pipeline._python
ORIGINAL_RUN_STEP = pipeline._run_step


def python_with_v3_config(module: str, *arguments: str) -> list[str]:
    command = ORIGINAL_PYTHON(module, *arguments)
    if module == "repro_scripts.lock_ps_dfsc_publication":
        index = command.index("--config") + 1
        command[index] = str(CONFIG_V3)
    return command


def run_step_with_v3_config(**kwargs) -> None:
    if kwargs.get("label") == "publication_lock":
        kwargs["inputs"] = [
            CONFIG_V3 if Path(path) == CONFIG_V2 else path
            for path in kwargs["inputs"]
        ]
    ORIGINAL_RUN_STEP(**kwargs)


def main() -> None:
    pipeline._python = python_with_v3_config
    pipeline._run_step = run_step_with_v3_config
    pipeline.main()


if __name__ == "__main__":
    main()
