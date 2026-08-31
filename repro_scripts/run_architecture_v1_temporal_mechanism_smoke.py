"""Synthetic-only smoke for matched T0/T1 temporal mechanisms.

The command is predictor-only by default.  ``--execute-smoke`` performs three
CPU updates on deterministic synthetic tensors and never imports a repository
target loader.  Its purpose is wiring and semantic verification, not evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from architecture_v1.model import (
    T0MemorylessRectifiedFlow,
    T1FeatureRectifiedFlow,
    T1ShuffleRectifiedFlow,
    T1SourceRectifiedFlow,
)
from architecture_v1.training import file_sha256, tensor_state_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "repro_configs" / "architecture_v1_temporal_mechanisms_v3_0.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "architecture_v1_temporal_mechanism_smoke_v3_0"


def _write_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    digest = file_sha256(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "architecture_v1_temporal_mechanisms_v3_0":
        raise ValueError("unexpected temporal-mechanism config schema")
    roles = config["role_access"]
    if roles["allowed_target_roles"] != ["train", "validation"]:
        raise ValueError("formal target-role contract drifted")
    if roles["selection_state"] != "sealed" or roles["calibration_state"] != "sealed":
        raise ValueError("sealed roles cannot be opened by a smoke config")
    if config["smoke"]["data"] != "deterministic_synthetic_only":
        raise ValueError("smoke may use synthetic data only")
    if config["smoke"]["real_target_roles_accessed"]:
        raise ValueError("smoke cannot register real target access")
    return config


def dry_run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_config(config_path)
    return {
        "mode": "predictor_only_no_targets_loaded_no_files_created",
        "config_sha256": file_sha256(config_path),
        "candidate_ids": list(config["candidates"]),
        "real_target_roles_accessed": [],
        "selection_state": "sealed",
        "calibration_state": "sealed",
    }


def _kwargs(config: dict[str, Any]) -> dict[str, Any]:
    common = dict(config["common_model"])
    for key in (
        "continuous_latent_scope",
        "common_parameter_topology_required",
        "common_initial_tensor_bank_required",
    ):
        common.pop(key)
    common.update(config["smoke"]["model_overrides"])
    return common


def _models(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = _kwargs(config)
    order = tuple(config["shuffle_control"]["hour_order_zero_based"])
    torch.manual_seed(31001)
    t0 = T0MemorylessRectifiedFlow(**kwargs)
    initial = t0.state_dict()
    models = {
        "T0": t0,
        "T1_feature": T1FeatureRectifiedFlow(**kwargs),
        "T1_source": T1SourceRectifiedFlow(**kwargs),
        "T1_feature_shuffle": T1ShuffleRectifiedFlow(
            temporal_hour_order=order, shuffle_target="feature", **kwargs
        ),
        "T1_source_shuffle": T1ShuffleRectifiedFlow(
            temporal_hour_order=order, shuffle_target="source", **kwargs
        ),
    }
    for model in models.values():
        model.load_state_dict(initial, strict=True)
        model.freeze_shared(True)
    return models


def _synthetic(config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layout = config["common_model"]
    generator = torch.Generator().manual_seed(32001)
    condition = torch.randn(
        config["smoke"]["batch_days"],
        layout["zones"],
        layout["hours"],
        layout["condition_dim"],
        generator=generator,
    )
    target = torch.sigmoid(
        0.35 * condition[..., 0]
        - 0.20 * condition[..., 1]
        + 0.15 * torch.sin(torch.arange(layout["hours"])[None, None])
    )
    target[:, 0, 0] = 0.0
    target[:, 1, 1] = 1.0
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[0, 2, 3] = False
    return condition.float(), target.float(), observed


def _grad_norm(parameter: torch.Tensor | None) -> float:
    if parameter is None:
        return 0.0
    return float(parameter.detach().norm().cpu())


def execute(config_path: Path, output: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {output}")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    models = _models(config)
    condition, target, observed = _synthetic(config)

    parameter_names = {name: list(model.state_dict()) for name, model in models.items()}
    parameter_shapes = {
        name: [list(value.shape) for value in model.state_dict().values()]
        for name, model in models.items()
    }
    initial_hashes = {
        name: tensor_state_sha256(model.state_dict()) for name, model in models.items()
    }
    topology_equal = len({json.dumps(value) for value in parameter_names.values()}) == 1
    shapes_equal = len({json.dumps(value) for value in parameter_shapes.values()}) == 1
    initial_equal = len(set(initial_hashes.values())) == 1

    histories: dict[str, list[float]] = {name: [] for name in models}
    recurrent_gradient_max: dict[str, dict[str, float]] = {
        name: {"feature": 0.0, "source": 0.0} for name in models
    }
    optimizers = {
        name: torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
            weight_decay=0.0,
        )
        for name, model in models.items()
    }
    for update in range(config["smoke"]["updates"]):
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            loss = model.flow_loss(
                condition,
                target,
                observed_mask=observed,
                generator=torch.Generator().manual_seed(33001 + update),
            )["loss"]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite smoke loss for {name}")
            loss.backward()
            recurrent_gradient_max[name]["feature"] = max(
                recurrent_gradient_max[name]["feature"],
                _grad_norm(model.feature_cell.recurrent_projection.weight.grad),
            )
            recurrent_gradient_max[name]["source"] = max(
                recurrent_gradient_max[name]["source"],
                _grad_norm(model.source_cell.recurrent_projection.weight.grad),
            )
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                5.0,
            )
            optimizer.step()
            histories[name].append(float(loss.detach()))

    reference = models["T0"]
    with torch.no_grad():
        statistics = reference.atom(reference.encode_condition(condition))
        allocation = reference.atom.allocate(
            statistics, members=config["smoke"]["members"], seed=34001
        )
        iid = torch.randn(
            len(condition),
            config["smoke"]["members"],
            config["common_model"]["zones"],
            config["common_model"]["hours"],
            generator=torch.Generator().manual_seed(35001),
        )
    scenarios: dict[str, dict[str, Any]] = {}
    state_reference: torch.Tensor | None = None
    all_states_equal = True
    all_atom_latents_zero = True
    for name, model in models.items():
        scenario = model.sample(
            condition,
            members=config["smoke"]["members"],
            steps=config["smoke"]["integration_steps"],
            seed=36001,
            member_chunk=2,
            allocation=allocation,
            initial_noise=iid,
        )
        if state_reference is None:
            state_reference = scenario.states
        else:
            all_states_equal &= bool(torch.equal(state_reference, scenario.states))
        all_atom_latents_zero &= bool(
            torch.all(scenario.interior_latent[~scenario.active_mask] == 0.0)
        )
        raw = scenario.values.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        scenarios[name] = {
            "finite": bool(torch.isfinite(scenario.values).all()),
            "mean": float(scenario.values.mean()),
            "std": float(scenario.values.std()),
            "values_sha256": hashlib.sha256(raw).hexdigest(),
            "per_path_nfe": scenario.per_path_nfe,
        }

    gradient_checks = {
        "T0_feature_recurrent_dormant": recurrent_gradient_max["T0"]["feature"] == 0.0,
        "T0_source_recurrent_dormant": recurrent_gradient_max["T0"]["source"] == 0.0,
        "T1_feature_edge_active": recurrent_gradient_max["T1_feature"]["feature"] > 0.0,
        "T1_feature_source_edge_dormant": recurrent_gradient_max["T1_feature"]["source"] == 0.0,
        "T1_source_feature_edge_dormant": recurrent_gradient_max["T1_source"]["feature"] == 0.0,
        "T1_source_edge_active": recurrent_gradient_max["T1_source"]["source"] > 0.0,
        "feature_shuffle_edge_active": recurrent_gradient_max["T1_feature_shuffle"]["feature"] > 0.0,
        "source_shuffle_edge_active": recurrent_gradient_max["T1_source_shuffle"]["source"] > 0.0,
    }
    checks = {
        "parameter_names_equal": topology_equal,
        "parameter_shapes_equal": shapes_equal,
        "initial_tensor_bank_equal": initial_equal,
        "losses_finite": all(
            all(torch.isfinite(torch.tensor(value)) for value in history)
            for history in histories.values()
        ),
        "common_atom_states_equal": all_states_equal,
        "atom_latents_strictly_zero": all_atom_latents_zero,
        "scenario_values_finite": all(value["finite"] for value in scenarios.values()),
        **gradient_checks,
    }
    result = {
        "schema": "architecture_v1_temporal_mechanism_smoke_v1",
        "status": "SMOKE_PASS" if all(checks.values()) else "SMOKE_FAIL",
        "scientific_evidence": False,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": file_sha256(config_path),
        "runner_sha256": file_sha256(Path(__file__)),
        "model_code_sha256": file_sha256(ROOT / "architecture_v1" / "model.py"),
        "real_target_roles_accessed": [],
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "checks": checks,
        "initial_state_sha256": initial_hashes,
        "loss_history": histories,
        "recurrent_gradient_max": recurrent_gradient_max,
        "model_specs": {name: model.model_spec() for name, model in models.items()},
        "scenarios": scenarios,
    }
    output.mkdir(parents=True, exist_ok=False)
    result_sha = _write_json(output / "SMOKE_RESULT.json", result)
    freeze = {
        "schema": "architecture_v1_temporal_mechanism_smoke_freeze_v1",
        "status": result["status"],
        "result": "SMOKE_RESULT.json",
        "result_sha256": result_sha,
        "selection_authorized": False,
        "retained_weight_training_authorized": result["status"] == "SMOKE_PASS",
    }
    _write_json(output / "smoke.freeze.json", freeze)
    if result["status"] != "SMOKE_PASS":
        raise RuntimeError("temporal mechanism smoke failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute-smoke", action="store_true")
    args = parser.parse_args()
    if args.execute_smoke:
        result = execute(args.config.resolve(), args.output.resolve())
    else:
        result = dry_run(args.config.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
