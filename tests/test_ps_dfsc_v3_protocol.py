from __future__ import annotations

import json
from pathlib import Path

import repro_scripts.run_ps_dfsc_validation_selection_v2 as validation


ROOT = Path(__file__).resolve().parents[1]


def test_v3_config_records_stochastic_training_and_exact_confirmation():
    value = json.loads(
        (ROOT / "repro_configs" / "ps_dfsc_v3.json").read_text(
            encoding="utf-8"
        )
    )
    assert value["training"]["canonical_entrypoint"].endswith("v18")
    assert value["training"]["proper_warmup_days_per_epoch"] == 100
    assert value["training"]["decision_days_per_epoch"] == 8
    assert value["training"]["commitment_refresh"] == "midpoint_all_100_days"
    assert value["training"]["midpoint_time_limit_seconds"] == 600.0
    assert value["exact_suc"]["time_limit_seconds"] == 600.0
    assert (
        value["exact_suc"]["no_incumbent_policy"]
        == "explicit_failed_row_without_substitution"
    )


def test_lock_command_uses_v3_config():
    command = validation.python_with_v3_config(
        "repro_scripts.lock_ps_dfsc_publication",
        "--config",
        str(validation.CONFIG_V2),
    )
    assert Path(command[command.index("--config") + 1]) == (
        validation.CONFIG_V3
    )
