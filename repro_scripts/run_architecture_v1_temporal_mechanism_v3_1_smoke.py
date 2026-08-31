"""Synthetic train/sample smoke for the variance-preserving v3.1 mechanisms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from architecture_v1.model import (
    T0StableSourceRectifiedFlow,
    T1FeatureStableSourceRectifiedFlow,
    T1SourceStableRectifiedFlow,
    T1StableShuffleRectifiedFlow,
)
from architecture_v1.training import configure_stage, file_sha256, tensor_state_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "repro_configs" / "architecture_v1_temporal_mechanisms_v3_1.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "architecture_v1_temporal_mechanism_v3_1_smoke"


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "architecture_v1_temporal_mechanisms_v3_1":
        raise ValueError("unexpected v3.1 config schema")
    roles = config["role_access"]
    if roles["allowed_target_roles"] != ["train", "validation"]:
        raise ValueError("role-access contract drifted")
    if roles["selection_state"] != "sealed" or roles["calibration_state"] != "sealed":
        raise ValueError("smoke requires sealed selection and calibration")
    if config["smoke"]["data"] != "deterministic_synthetic_only":
        raise ValueError("smoke data must be synthetic")
    if config["smoke"]["real_target_roles_accessed"]:
        raise ValueError("smoke cannot access real targets")
    return config


def _write_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = file_sha256(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def dry_run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_config(config_path)
    return {
        "mode": "predictor_only_no_files_created",
        "config_sha256": file_sha256(config_path),
        "candidates": list(config["pilot_candidates"]),
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "real_target_roles_accessed": [],
    }


def _kwargs() -> dict[str, Any]:
    return {
        "condition_dim": 4,
        "zones": 2,
        "hours": 24,
        "encoder_dim": 8,
        "encoder_depth": 1,
        "flow_dim": 8,
        "flow_depth": 1,
        "heads": 2,
        "ff_multiplier": 2,
        "dropout": 0.0,
        "atom_hidden_dim": 8,
        "atom_fixed_one_probability": 1e-4,
        "stable_source_rho_max": 0.95,
    }


def _models(order: tuple[int, ...]) -> dict[str, Any]:
    torch.manual_seed(31101)
    reference = T0StableSourceRectifiedFlow(**_kwargs())
    initial = reference.state_dict()
    models = {
        "T0": reference,
        "T1_feature": T1FeatureStableSourceRectifiedFlow(**_kwargs()),
        "T1_source": T1SourceStableRectifiedFlow(**_kwargs()),
        "T1_feature_shuffle": T1StableShuffleRectifiedFlow(
            temporal_hour_order=order, shuffle_target="feature", **_kwargs()
        ),
        "T1_source_shuffle": T1StableShuffleRectifiedFlow(
            temporal_hour_order=order, shuffle_target="source", **_kwargs()
        ),
    }
    for model in models.values():
        model.load_state_dict(initial, strict=True)
        model.freeze_shared(True)
    return models


def execute(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = _load_config(config_path)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    order = tuple(config["shuffle_control"]["hour_order_zero_based"])
    models = _models(order)

    hashes = {name: tensor_state_sha256(model.state_dict()) for name, model in models.items()}
    if len(set(hashes.values())) != 1:
        raise RuntimeError("candidate initial tensor banks differ")
    if len({model.parameter_count() for model in models.values()}) != 1:
        raise RuntimeError("candidate parameter counts differ")

    generator = torch.Generator().manual_seed(31102)
    condition = torch.randn(2, 2, 24, 4, generator=generator)
    target = torch.sigmoid(0.4 * condition[..., 0] - 0.2 * condition[..., 1])
    target[0, 0, 0] = 0.0
    target[1, 1, 1] = 1.0
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[0, 1, 3] = False

    histories: dict[str, list[float]] = {}
    rho_gradient_max: dict[str, float] = {}
    for name, model in models.items():
        parameters = configure_stage(model, "flow")
        optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0.0)
        losses: list[float] = []
        maximum = 0.0
        for update in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = model.flow_loss(
                condition,
                target,
                observed_mask=observed,
                generator=torch.Generator().manual_seed(31103 + update),
            )["loss"]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss for {name}")
            loss.backward()
            gradient = model.stable_source.rho_head.weight.grad
            maximum = max(maximum, 0.0 if gradient is None else float(gradient.norm()))
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        histories[name] = losses
        rho_gradient_max[name] = maximum
    if rho_gradient_max["T0"] != 0.0 or rho_gradient_max["T1_feature"] != 0.0:
        raise RuntimeError("IID controls unexpectedly trained the AR correlation head")
    if rho_gradient_max["T1_source"] <= 0.0:
        raise RuntimeError("chronological AR source received no gradient")

    reference = models["T0"]
    with torch.no_grad():
        encoded = reference.encode_condition(condition)
        statistics = reference.atom(encoded)
        allocation = reference.atom.allocate(statistics, members=4, seed=31110)
        iid = torch.randn(2, 4, 2, 24, generator=torch.Generator().manual_seed(31111))
    state_hashes: dict[str, str] = {}
    atom_latent_max: dict[str, float] = {}
    nfe: dict[str, int] = {}
    for name, model in models.items():
        scenario = model.sample(
            condition,
            members=4,
            steps=2,
            seed=31112,
            method="heun",
            member_chunk=2,
            allocation=allocation,
            initial_noise=iid,
        )
        state_hashes[name] = tensor_state_sha256({"states": scenario.states})
        inactive = ~scenario.active_mask
        atom_latent_max[name] = (
            float(scenario.interior_latent[inactive].abs().max())
            if bool(inactive.any())
            else 0.0
        )
        nfe[name] = scenario.per_path_nfe
        if not bool(torch.isfinite(scenario.values).all()):
            raise RuntimeError(f"non-finite scenarios for {name}")
    if len(set(state_hashes.values())) != 1:
        raise RuntimeError("common atom allocation was not preserved")
    if max(atom_latent_max.values()) != 0.0:
        raise RuntimeError("an atom coordinate received a continuous latent")
    if set(nfe.values()) != {3}:
        raise RuntimeError("two-step Heun must report three flow evaluations")

    # Large synthetic source bank checks the engineering invariant directly.
    source_model = models["T1_source"]
    with torch.no_grad():
        large_iid = torch.randn(12000, 1, 24, generator=torch.Generator().manual_seed(31120))
        large_encoded = torch.zeros(12000, 1, 24, source_model.encoder_dim)
        large_active = torch.ones_like(large_iid, dtype=torch.bool)
        transformed = source_model.stable_source(
            large_iid, large_encoded, large_active, use_correlation=True
        )
        source_mean = float(transformed.mean())
        source_variance = float(transformed.var(unbiased=True))
        rho_max = float(source_model.stable_source.correlation(large_encoded[:1]).abs().max())
    gates = {
        "common_topology": len({tuple(model.state_dict()) for model in models.values()}) == 1,
        "common_initial_tensors": len(set(hashes.values())) == 1,
        "iid_control_rho_gradient_zero": rho_gradient_max["T0"] == 0.0 and rho_gradient_max["T1_feature"] == 0.0,
        "source_rho_gradient_positive": rho_gradient_max["T1_source"] > 0.0,
        "source_mean": abs(source_mean) <= 0.03,
        "source_variance": 0.92 <= source_variance <= 1.08,
        "source_rho_bounded": rho_max <= 0.95,
        "common_atom_allocation": len(set(state_hashes.values())) == 1,
        "atom_latent_zero": max(atom_latent_max.values()) == 0.0,
        "nfe_accounting": set(nfe.values()) == {3},
    }
    result = {
        "schema": "architecture_v1_temporal_mechanism_v3_1_smoke_result",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scientific_evidence": False,
        "config_sha256": file_sha256(config_path),
        "real_target_roles_accessed": [],
        "selection_state": "sealed",
        "calibration_state": "sealed",
        "gates": gates,
        "loss_history": histories,
        "rho_gradient_max": rho_gradient_max,
        "source_diagnostic": {"mean": source_mean, "variance": source_variance, "rho_abs_max": rho_max},
        "atom_latent_max": atom_latent_max,
        "per_path_nfe": nfe,
    }
    _write_json(output / "SMOKE_RESULT.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("v3.1 smoke gates failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute-smoke", action="store_true")
    args = parser.parse_args()
    if not args.execute_smoke:
        print(json.dumps(dry_run(args.config), indent=2, sort_keys=True))
        return
    result = execute(args.config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
