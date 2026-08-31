from __future__ import annotations

from contextlib import contextmanager, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from architecture_v1.evaluation import load_scenario_archive
from architecture_v1.protocol import file_sha256, load_architecture_config
from repro_scripts import run_architecture_v1_smoke as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "repro_configs" / "architecture_v1.json"
DATA = ROOT / "Data"


@contextmanager
def _raises(exception: type[BaseException], pattern: str):
    try:
        yield
    except exception as error:
        assert pattern.lower() in str(error).lower(), str(error)
    else:
        raise AssertionError(f"expected {exception.__name__}: {pattern}")


def _numpy_random_states_equal(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_dry_run_is_default_and_never_creates_output() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "must_not_exist"
        result = runner.main(
            [
                "--config",
                str(CONFIG),
                "--data-dir",
                str(DATA),
                "--output-dir",
                str(output),
            ]
        )
        assert result["mode"] == "dry_run"
        assert result["executed"] is False
        assert result["active_date_counts"] == runner.EXPECTED_SMOKE_COUNTS
        assert not output.exists()

    # There is intentionally no ambiguous --run alias: only the fully named
    # execution flag may mutate the filesystem.
    with _raises(SystemExit, "2"):
        with redirect_stderr(io.StringIO()):
            runner.build_parser().parse_args(["--run"])


def test_resolved_plan_consumes_registered_bounded_values_and_rejects_drift() -> None:
    _, source = load_architecture_config(CONFIG)
    resolved = runner.resolved_smoke_config(source)
    assert resolved["scientific_status"] == "smoke_only_non_publishable"
    assert resolved["hardware"] == {
        "device": "cpu",
        "dtype": "float32",
        "amp": False,
        "torch_num_threads": 1,
        "deterministic_algorithms": True,
    }
    assert resolved["model"]["encoder_dim"] == 32
    assert resolved["model"]["encoder_depth"] == 1
    assert resolved["model"]["flow_dim"] == 32
    assert resolved["model"]["flow_depth"] == 1
    assert resolved["training"]["r0_atom_steps"] == 2
    assert resolved["training"]["r0_flow_steps"] == 2
    assert resolved["training"]["t0_atom_steps"] == 0
    assert resolved["training"]["t0_flow_steps"] == 2
    assert resolved["sampling"]["members"] == 4
    assert resolved["sampling"]["steps"] == 2
    assert resolved["sampling"]["method"] == "heun"
    assert resolved["sampling"]["per_path_nfe"] == 3
    assert resolved["seeds"]["shared_sampling"] == 20000

    enlarged = deepcopy(source)
    enlarged["smoke"]["training"]["updates_per_stage"] = 200
    with _raises(RuntimeError, "formal or enlarged"):
        runner.resolved_smoke_config(enlarged)

    enlarged = deepcopy(source)
    enlarged["smoke"]["sampling"]["members"] = 100
    with _raises(RuntimeError, "formal or enlarged"):
        runner.resolved_smoke_config(enlarged)


def test_r_seen_final_and_nonregistered_roles_fail_closed() -> None:
    assert runner._require_allowed_roles(("train", "validation", "selection")) == (
        "train",
        "validation",
        "selection",
    )
    for role in ("r_seen", "final"):
        with _raises(RuntimeError, "refuses R-SEEN/final"):
            runner._require_allowed_roles((role,))
    with _raises(RuntimeError, "unregistered"):
        runner._require_allowed_roles(("calibration",))


def test_real_cpu_smoke_executes_fixed_plan_and_writes_auditable_artifacts() -> None:
    previous_threads = torch.get_num_threads()
    previous_dtype = torch.get_default_dtype()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_python_random = random.getstate()
    previous_numpy_random = np.random.get_state()
    previous_torch_random = torch.random.get_rng_state().clone()

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "architecture_v1_smoke"
        result = runner.execute_smoke(
            config_path=CONFIG,
            data_dir=DATA,
            output_dir=output,
        )
        assert result["status"] == "complete"
        assert result["scientific_status"] == "smoke_only_non_publishable"
        assert result["formal_training_performed"] is False
        assert result["roles_used"] == ["train", "validation", "selection"]
        assert result["roles_never_used"] == ["calibration", "r_seen", "final"]

        marker = json.loads((output / "SMOKE_ONLY.json").read_text(encoding="utf-8"))
        completion = json.loads(
            (output / "completion.json").read_text(encoding="utf-8")
        )
        assert marker["status"] == "complete"
        assert marker["formal_training_authorized"] is False
        assert marker["r_seen_or_final_use_authorized"] is False
        assert completion["status"] == "complete"

        protocol_path = output / "architecture_v1.protocol.manifest.json"
        assert protocol_path.is_file()
        assert protocol_path.with_name(protocol_path.name + ".sha256").is_file()
        for variant in ("R0", "T0"):
            flow = result["training"][variant]["flow"]
            assert flow["steps"] == 2
            assert Path(flow["best"]).is_file()
            assert file_sha256(flow["best"]) == flow["best_sha256"]
        assert result["training"]["R0"]["atom"]["steps"] == 2
        assert (
            result["training"]["R0"]["common_shell_sha256"]
            == result["training"]["T0"]["common_shell_sha256_after_flow"]
        )
        assert result["training"]["T0"]["common_shell_frozen_unchanged"] is True

        archives = {}
        for variant in ("R0", "T0"):
            report = result["evaluation"][variant]
            archive = load_scenario_archive(
                report["archive"],
                expected_metadata={
                    "candidate_id": variant,
                    "split_role": "selection",
                    "scientific_status": "smoke_only_non_publishable",
                    "estimand": "finite_dependent_scenario_set",
                    "score_semantics": "empirical_v_stat",
                    "common_random_numbers_group": (
                        "R0_T0_selection_smoke_seed20000"
                    ),
                    "plan_id": "selection_M4_heun2_crn_v1",
                },
            )
            archives[variant] = archive
            assert archive.scenarios.shape == (2, 4, 10, 24)
            assert archive.metadata["integrator"] == "heun"
            assert archive.metadata["integration_steps"] == 2
            assert archive.metadata["flow_nfe"] == 3
            assert report["sampling"]["batched_forward_calls"] == 6
            assert report["summary"]["days"] == 2
            assert report["summary"]["members"] == 4
            assert Path(report["metrics"]).is_file()
        assert np.array_equal(archives["R0"].states, archives["T0"].states)
        assert np.array_equal(
            archives["R0"].zero_probability,
            archives["T0"].zero_probability,
        )
        assert np.array_equal(
            archives["R0"].one_probability,
            archives["T0"].one_probability,
        )

        with _raises(FileExistsError, "refusing to overwrite"):
            runner.execute_smoke(
                config_path=CONFIG,
                data_dir=DATA,
                output_dir=output,
            )

    assert torch.get_num_threads() == previous_threads
    assert torch.get_default_dtype() == previous_dtype
    assert torch.are_deterministic_algorithms_enabled() == previous_deterministic
    assert (
        torch.is_deterministic_algorithms_warn_only_enabled() == previous_warn_only
    )
    assert random.getstate() == previous_python_random
    assert _numpy_random_states_equal(np.random.get_state(), previous_numpy_random)
    assert torch.equal(torch.random.get_rng_state(), previous_torch_random)


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
