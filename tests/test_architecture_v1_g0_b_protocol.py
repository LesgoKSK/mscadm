"""Protocol-only checks for the frozen G0-B tiny-denoiser utility Probe."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "repro_configs" / "architecture_v1_g0_b_tiny_denoiser.json"
REPORT = ROOT / "reports" / "ARCHITECTURE_V1_G0_B_TINY_DENOISER_PROTOCOL.md"
G0_A_RESULT = (
    ROOT / "outputs" / "architecture_v1_g0_a_predictability" / "G0_A_RESULT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _date_sha256(days: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(days)).encode("ascii")).hexdigest()


def test_frozen_config_sidecar_scope_and_lineage() -> None:
    value = _config()
    assert value["schema"] == "architecture_v1_g0_b_tiny_denoiser_utility_v1"
    assert value["status"] == (
        "frozen_before_any_G0_B_target_aware_P0_or_tiny_denoiser_training"
    )
    assert value["evidence_class"] == (
        "prospective_train_only_nested_cross_fitted_mechanism_probe_not_validation_or_generation"
    )
    assert value["lineage"]["g0_a_required_status"] == (
        "G0_A_NWP_MODE_UNCERTAINTY_GO"
    )
    assert value["lineage"]["g0_b0_required_status"] == (
        "G0_B0_SCHEDULE_FEASIBILITY_GO"
    )
    sidecar = CONFIG.with_name(CONFIG.name + ".sha256")
    assert sidecar.read_text(encoding="ascii").split() == [_sha256(CONFIG), CONFIG.name]
    assert value["role_access"]["dry_run_allowed_target_roles"] == []
    assert value["role_access"]["P0_allowed_target_roles"] == ["train"]
    assert set(value["role_access"]["forbidden_target_roles"]) == {
        "validation",
        "calibration",
        "selection",
        "r_seen",
        "final",
    }
    assert value["freeze_and_resume"]["formal_training_requires_P0_GO"] is True


def test_exact_six_path_registry_and_causal_control() -> None:
    value = _config()
    paths = value["paths"]
    assert [path["id"] for path in paths] == [
        "IID",
        "CW_GROUP",
        "FIXED_BAND",
        "MULAN_LITE",
        "PA_RWF",
        "PA_SHUFFLE",
    ]
    assert len(paths) == value["training"]["paths"] == 6
    by_name = {path["id"]: path for path in paths}
    assert by_name["PA_RWF"][
        "must_numerically_equal_G0_B0_eta_0.5_on_frozen_G0_A_rows"
    ] is True
    assert by_name["PA_SHUFFLE"]["fixed_points_allowed"] == 0
    assert "correct same-day" in by_name["PA_SHUFFLE"]["budget_and_reserve"]
    assert "correct" in by_name["PA_SHUFFLE"]["denoiser_condition"]
    assert by_name["CW_GROUP"]["full_CW_Gen_replication"] is False
    assert by_name["MULAN_LITE"]["scheduler_parameters"] == 870
    assert value["shared_proxy_allocation"]["eta"] == 0.5
    assert value["shared_proxy_allocation"]["eta_tunable"] is False


def test_training_matrix_architecture_and_common_compute_are_exact() -> None:
    value = _config()
    training = value["training"]
    runs = (
        training["outer_folds"]
        * training["data_fractions"]
        * training["paths"]
        * len(training["model_seeds"])
    )
    assert runs == training["expected_retained_runs"] == 324
    assert training["model_seeds"] == [3, 4, 5]
    assert training["optimizer_updates_per_run"] == 1024
    assert training["calendar_day_corruption_exposures_per_run"] == (
        training["optimizer_updates_per_run"] * training["batch_calendar_days"]
    )
    assert training["validation_early_stopping"] is False
    assert training["retained_checkpoint"] == "final EMA after update 1024 only"

    model = value["tiny_denoiser"]
    assert model["input_dimension"] == 240 + 240 + 47 + 6 + 16 == 549
    expected = 2 * 549
    expected += 549 * 64 + 64
    expected += 64 * 64 + 64
    expected += 64 * 240 + 240
    assert expected == model["expected_trainable_parameters_without_scheduler"] == 56058
    overhead = 870 / expected
    assert math.isclose(
        overhead,
        model["MULAN_LITE_parameter_overhead_fraction"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert overhead <= model["maximum_allowed_schedule_parameter_overhead_fraction"]


def test_learning_curve_registry_is_nested_and_matches_g0a_dates_when_available() -> None:
    value = _config()
    curve = value["learning_curve"]
    assert curve["fractions"] == [0.25, 0.5, 1.0]
    assert curve["primary_fraction"] == 0.5
    assert curve["nested_subsets_required"] is True
    assert curve["nuisance_models_refit_per_fraction"] is False
    assert set(curve["subset_registry"]) == {str(index) for index in range(6)}
    if not G0_A_RESULT.is_file():
        return

    evidence = json.loads(G0_A_RESULT.read_text(encoding="utf-8"))
    rows = evidence["per_day"]
    root = int(curve["subset_seed_root"])
    tag = "architecture_v1_g0_b_tiny_v1"
    for fold in range(6):
        registry = curve["subset_registry"][str(fold)]
        outer_train = sorted(
            str(record["day"])
            for record in rows
            if int(record["outer_fold"]) != fold
        )
        ranked = sorted(
            outer_train,
            key=lambda day: (
                hashlib.sha256(f"{tag}|{root + fold}|{day}".encode()).digest(),
                day,
            ),
        )
        previous: set[str] = set()
        for fraction in curve["fractions"]:
            count = (
                len(outer_train)
                if fraction == 1.0
                else math.ceil(float(fraction) * len(outer_train))
            )
            selected = set(ranked[:count])
            frozen = registry[f"fraction_{fraction}"]
            assert frozen["days"] == count
            assert frozen["date_sha256"] == _date_sha256(list(selected))
            assert previous.issubset(selected)
            previous = selected


def test_confirmatory_metrics_gates_and_no_generation_boundary() -> None:
    value = _config()
    metrics = value["metrics"]
    assert metrics["raw_training_loss_used_as_formal_evidence"] is False
    assert "cell_MSE" in metrics
    assert "increment_MSE" in metrics
    assert "joint_day_normalized_SSE" in metrics
    assert "oracle_efficiency_ratio" in metrics
    assert len(value["inference"]["confirmatory_contrasts"]) == 13
    gates = value["go_no_go"]
    assert gates["PA_vs_IID_balanced_risk_min_relative_improvement_at_50pct"] == 0.02
    assert gates["PA_vs_SHUFFLE_balanced_risk_min_relative_improvement_at_50pct"] == 0.01
    assert gates["PA_vs_MULAN_AULC_min_relative_improvement"] == 0.02
    assert gates["PA_vs_IID_oracle_efficiency_min_relative_improvement_at_50pct"] == 0.01
    assert gates["any_non_GO_status_authorizes_full_model"] is False
    assert value["continuous_residual"]["atom_model_or_scenario_reconstruction"] is False


def test_protocol_report_exposes_the_machine_contract() -> None:
    value = _config()
    text = REPORT.read_text(encoding="utf-8")
    assert _sha256(CONFIG) in text
    for path in ("IID", "CW_GROUP", "FIXED_BAND", "MULAN_LITE", "PA_RWF", "PA_SHUFFLE"):
        assert path in text
    assert "324 retained runs" in text
    assert value["go_no_go"]["GO_status"] in text
    assert "不会生成 100 条场景" in text
