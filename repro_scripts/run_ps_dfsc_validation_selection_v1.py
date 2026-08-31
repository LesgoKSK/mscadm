"""Automate validation calibration, safety, exact selection, and locking."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BETA_SPECS = (
    ("beta000", 0.0, 0),
    ("beta025", 0.25, 25),
    ("beta050", 0.5, 50),
    ("beta100", 1.0, 100),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes(paths) -> dict[str, str]:
    return {str(path.resolve()): _sha256(path) for path in paths}


def _done_valid(
    done: Path,
    *,
    command: list[str],
    inputs: list[Path],
    outputs: list[Path],
) -> bool:
    if not done.is_file() or not all(path.is_file() for path in outputs):
        return False
    payload = json.loads(done.read_text(encoding="utf-8"))
    return bool(
        payload.get("command") == command
        and payload.get("input_sha256") == _hashes(inputs)
        and payload.get("output_sha256") == _hashes(outputs)
    )


def _run_step(
    *,
    outer_dir: Path,
    label: str,
    command: list[str],
    inputs: list[Path],
    outputs: list[Path],
    attempts: int,
) -> None:
    logs = outer_dir / "validation_automation"
    logs.mkdir(parents=True, exist_ok=True)
    done = logs / f"{label}.done.json"
    if _done_valid(
        done, command=command, inputs=inputs, outputs=outputs
    ):
        return
    stdout = logs / f"{label}.stdout.log"
    stderr = logs / f"{label}.stderr.log"
    for attempt in range(1, attempts + 1):
        with stdout.open("ab") as out, stderr.open("ab") as err:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=out,
                stderr=err,
                check=False,
            )
        if result.returncode == 0 and all(
            path.is_file() for path in outputs
        ):
            payload = {
                "schema": "ps_dfsc_validation_step_v1",
                "completed_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "label": label,
                "attempt": attempt,
                "command": command,
                "input_sha256": _hashes(inputs),
                "output_sha256": _hashes(outputs),
                "test_truth_accessed": False,
            }
            temporary = done.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(done)
            return
    raise RuntimeError(
        f"validation step failed closed after {attempts} attempts: {label}"
    )


def _wait_for_candidates(poll_seconds: float) -> None:
    required = [
        ROOT
        / "outputs"
        / "ps_dfsc"
        / f"outer{outer}"
        / "candidates"
        / f"{name}.validation.json"
        for outer in (1, 2, 3)
        for name, _beta, _offset in BETA_SPECS
    ]
    while not all(path.is_file() for path in required):
        time.sleep(poll_seconds)


def _python(module: str, *arguments: str) -> list[str]:
    return [sys.executable, "-u", "-m", module, *arguments]


def _record_command(
    *,
    name: str,
    beta: float,
    checkpoint: Path,
    proxy_summary: Path,
    calibrated: Path,
    gate: Path,
    output: Path,
    exact_summary: Path | None = None,
) -> list[str]:
    command = _python(
        "repro_scripts.build_ps_dfsc_candidate_record_publication",
        "--candidate-id",
        name,
        "--beta",
        str(beta),
        "--checkpoint",
        str(checkpoint),
        "--proxy-summary",
        str(proxy_summary),
        "--calibrated",
        str(calibrated),
        "--gate",
        str(gate),
    )
    if exact_summary is not None:
        command += ["--exact-pair-summary", str(exact_summary)]
    command += ["--output", str(output)]
    return command


def _run_outer(outer: int, attempts: int) -> None:
    outer_dir = ROOT / "outputs" / "ps_dfsc" / f"outer{outer}"
    base_dir = ROOT / "outputs" / "ps_dfsc" / "base" / f"outer{outer}"
    candidate_dir = outer_dir / "candidates"
    selection_dir = outer_dir / "selection"
    exact_dir = outer_dir / "validation_exact"
    selection_dir.mkdir(parents=True, exist_ok=True)
    exact_dir.mkdir(parents=True, exist_ok=True)
    pooled_train = base_dir / "pooled" / "development_train_M100.npz"
    pooled_validation = (
        base_dir / "pooled" / "development_validation_M100.npz"
    )
    prepared_train = outer_dir / "development_train_prepared.npz"
    prepared_validation = (
        outer_dir / "development_validation_prepared.npz"
    )
    identity_train = outer_dir / "identity_train.npz"
    identity_validation = outer_dir / "identity_validation.npz"
    mapping = ROOT / "repro_configs" / f"ps_dfsc_mapping_outer{outer}.json"
    artifacts: dict[str, dict[str, Path]] = {}
    for name, beta, seed_offset in BETA_SPECS:
        checkpoint = candidate_dir / f"{name}.pt"
        calibrated = candidate_dir / f"{name}_validation.npz"
        gate = selection_dir / f"{name}.safety.json"
        gate_metrics = gate.with_suffix(".metrics.npz")
        proxy_csv = selection_dir / f"{name}.proxy.csv"
        proxy_summary = proxy_csv.with_suffix(".summary.json")
        record = selection_dir / f"{name}.record.json"
        seed = outer * 10_000 + seed_offset + 1
        _run_step(
            outer_dir=outer_dir,
            label=f"{name}_calibrate",
            command=_python(
                "repro_scripts.run_ps_dfsc_canonical_v17",
                "calibrate",
                "--input",
                str(pooled_validation),
                "--scenario-key",
                "scenarios",
                "--checkpoint",
                str(checkpoint),
                "--mapping",
                str(mapping),
                "--clusters",
                "20",
                "--device",
                "cuda",
                "--output",
                str(calibrated),
            ),
            inputs=[pooled_validation, checkpoint, mapping],
            outputs=[calibrated],
            attempts=attempts,
        )
        _run_step(
            outer_dir=outer_dir,
            label=f"{name}_safety",
            command=_python(
                "repro_scripts.run_ps_dfsc_canonical_v17",
                "safety-gate",
                "--candidate",
                str(calibrated),
                "--baseline",
                str(identity_validation),
                "--truth",
                str(pooled_validation),
                "--regime-reference",
                str(prepared_train),
                "--bootstrap-samples",
                "10000",
                "--seed",
                str(seed),
                "--output",
                str(gate),
            ),
            inputs=[
                calibrated,
                identity_validation,
                pooled_validation,
                prepared_train,
            ],
            outputs=[gate, gate_metrics],
            attempts=attempts,
        )
        _run_step(
            outer_dir=outer_dir,
            label=f"{name}_proxy",
            command=_python(
                "repro_scripts.evaluate_ps_dfsc_proxy_v2",
                "--candidate",
                str(calibrated),
                "--identity",
                str(identity_validation),
                "--prepared",
                str(prepared_validation),
                "--mapping",
                str(mapping),
                "--clusters",
                "20",
                "--strong-convexity",
                "1e-4",
                "--risk-weight",
                "0.25",
                "--output",
                str(proxy_csv),
            ),
            inputs=[
                calibrated,
                identity_validation,
                prepared_validation,
                mapping,
            ],
            outputs=[proxy_csv, proxy_summary],
            attempts=attempts,
        )
        _run_step(
            outer_dir=outer_dir,
            label=f"{name}_record_pre",
            command=_record_command(
                name=name,
                beta=beta,
                checkpoint=checkpoint,
                proxy_summary=proxy_summary,
                calibrated=calibrated,
                gate=gate,
                output=record,
            ),
            inputs=[checkpoint, proxy_summary, calibrated, gate],
            outputs=[record],
            attempts=attempts,
        )
        artifacts[name] = {
            "checkpoint": checkpoint,
            "calibrated": calibrated,
            "gate": gate,
            "gate_metrics": gate_metrics,
            "proxy_csv": proxy_csv,
            "proxy_summary": proxy_summary,
            "record": record,
        }
    records = {
        name: json.loads(paths["record"].read_text(encoding="utf-8"))
        for name, paths in artifacts.items()
    }
    shortlist = sorted(
        (
            value
            for value in records.values()
            if bool(value["gate"]["passed"])
        ),
        key=lambda value: (
            float(value["proxy_objective"]),
            -float(value["mean_ess"]),
            float(value["mean_transport"]),
            value["candidate_id"],
        ),
    )[:3]
    shortlist_names = {value["candidate_id"] for value in shortlist}
    identity_csv = exact_dir / "identity.csv"
    identity_summary = identity_csv.with_suffix(".summary.json")
    if shortlist:
        _run_step(
            outer_dir=outer_dir,
            label="identity_exact",
            command=_python(
                "repro_scripts.run_ps_dfsc_exact_validation_cached_v1",
                "--calibrated",
                str(identity_validation),
                "--truth",
                str(pooled_validation),
                "--mapping",
                str(mapping),
                "--outer",
                str(outer),
                "--method",
                "identity",
                "--mip-gap",
                "0.001",
                "--time-limit",
                "600",
                "--per-day-attempts",
                "3",
                "--cache-dir",
                str(exact_dir / "cache_identity"),
                "--output",
                str(identity_csv),
            ),
            inputs=[identity_validation, pooled_validation, mapping],
            outputs=[identity_csv, identity_summary],
            attempts=attempts,
        )
    for name, beta, _seed_offset in BETA_SPECS:
        paths = artifacts[name]
        exact_pair = None
        if name in shortlist_names:
            candidate_csv = exact_dir / f"{name}.csv"
            candidate_summary = candidate_csv.with_suffix(".summary.json")
            _run_step(
                outer_dir=outer_dir,
                label=f"{name}_exact",
                command=_python(
                    "repro_scripts.run_ps_dfsc_exact_validation_cached_v1",
                    "--calibrated",
                    str(paths["calibrated"]),
                    "--truth",
                    str(pooled_validation),
                    "--mapping",
                    str(mapping),
                    "--outer",
                    str(outer),
                    "--method",
                    name,
                    "--mip-gap",
                    "0.001",
                    "--time-limit",
                    "600",
                    "--per-day-attempts",
                    "3",
                    "--cache-dir",
                    str(exact_dir / f"cache_{name}"),
                    "--output",
                    str(candidate_csv),
                ),
                inputs=[paths["calibrated"], pooled_validation, mapping],
                outputs=[candidate_csv, candidate_summary],
                attempts=attempts,
            )
            exact_pair = exact_dir / f"{name}.paired.summary.json"
            _run_step(
                outer_dir=outer_dir,
                label=f"{name}_exact_pair",
                command=_python(
                    "repro_scripts.summarize_ps_dfsc_exact_pair",
                    "--candidate-csv",
                    str(candidate_csv),
                    "--baseline-csv",
                    str(identity_csv),
                    "--risk-weight",
                    "0.25",
                    "--output",
                    str(exact_pair),
                ),
                inputs=[candidate_csv, identity_csv],
                outputs=[exact_pair],
                attempts=attempts,
            )
        command = _record_command(
            name=name,
            beta=beta,
            checkpoint=paths["checkpoint"],
            proxy_summary=paths["proxy_summary"],
            calibrated=paths["calibrated"],
            gate=paths["gate"],
            output=paths["record"],
            exact_summary=exact_pair,
        )
        record_inputs = [
            paths["checkpoint"],
            paths["proxy_summary"],
            paths["calibrated"],
            paths["gate"],
        ]
        if exact_pair is not None:
            record_inputs.append(exact_pair)
        _run_step(
            outer_dir=outer_dir,
            label=f"{name}_record_final",
            command=command,
            inputs=record_inputs,
            outputs=[paths["record"]],
            attempts=attempts,
        )
    decision = selection_dir / "decision.json"
    record_paths = [artifacts[name]["record"] for name, *_ in BETA_SPECS]
    _run_step(
        outer_dir=outer_dir,
        label="candidate_selection",
        command=_python(
            "repro_scripts.select_ps_dfsc_candidate_publication",
            "--candidates",
            *(str(path) for path in record_paths),
            "--output",
            str(decision),
        ),
        inputs=record_paths,
        outputs=[decision],
        attempts=attempts,
    )
    model_files = [
        base_dir / "runs" / f"seed{seed}" / "final.pt"
        for seed in range(3)
    ]
    for name, *_ in BETA_SPECS:
        model_files += [
            candidate_dir / f"{name}.pt",
            candidate_dir / f"{name}.epoch_resume.pt",
            candidate_dir / f"{name}.history.json",
        ]
    data_files = [
        pooled_train,
        pooled_validation,
        prepared_train,
        prepared_validation,
        identity_train,
        identity_validation,
        mapping,
        *record_paths,
    ]
    for name, *_ in BETA_SPECS:
        paths = artifacts[name]
        data_files += [
            paths["calibrated"],
            paths["gate"],
            paths["gate_metrics"],
            paths["proxy_csv"],
            paths["proxy_summary"],
            candidate_dir / f"{name}.validation.json",
        ]
    if shortlist:
        data_files += [identity_csv, identity_summary]
        for name in shortlist_names:
            data_files += [
                exact_dir / f"{name}.csv",
                exact_dir / f"{name}.summary.json",
                exact_dir / f"{name}.paired.summary.json",
            ]
    lock = outer_dir / "lock_manifest.json"
    config = ROOT / "repro_configs" / "ps_dfsc_v2.json"
    splits = ROOT / "repro_configs" / "ps_dfsc_splits.json"
    _run_step(
        outer_dir=outer_dir,
        label="publication_lock",
        command=_python(
            "repro_scripts.lock_ps_dfsc_publication",
            "--splits",
            str(splits),
            "--config",
            str(config),
            "--models",
            *(str(path) for path in model_files),
            "--data-files",
            *(str(path) for path in data_files),
            "--selection",
            str(decision),
            "--outer",
            str(outer),
            "--base-output",
            str(ROOT / "outputs" / "ps_dfsc" / "base"),
            "--output",
            str(lock),
        ),
        inputs=[splits, config, decision, *model_files, *data_files],
        outputs=[lock],
        attempts=attempts,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    _wait_for_candidates(args.poll_seconds)
    for outer in (1, 2, 3):
        _run_outer(outer, args.attempts)
    completion = ROOT / "outputs" / "ps_dfsc" / "validation_locks.complete.json"
    completion.write_text(
        json.dumps(
            {
                "schema": "ps_dfsc_validation_locks_complete_v1",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "locks": {
                    f"outer{outer}": _sha256(
                        ROOT
                        / "outputs"
                        / "ps_dfsc"
                        / f"outer{outer}"
                        / "lock_manifest.json"
                    )
                    for outer in (1, 2, 3)
                },
                "test_truth_accessed": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
