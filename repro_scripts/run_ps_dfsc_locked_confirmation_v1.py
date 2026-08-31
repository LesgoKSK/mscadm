"""Run all locked confirmation phases with restartable exact evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ps_dfsc.manifest_publication import verify_lock_manifest


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes(paths) -> dict[str, str]:
    return {str(path.resolve()): _sha256(path) for path in paths}


def _run_step(
    *,
    label: str,
    command: list[str],
    inputs: list[Path],
    outputs: list[Path],
    attempts: int,
) -> None:
    logs = ROOT / "outputs" / "ps_dfsc" / "confirmation_automation"
    logs.mkdir(parents=True, exist_ok=True)
    done = logs / f"{label}.done.json"
    if done.is_file() and all(path.is_file() for path in outputs):
        payload = json.loads(done.read_text(encoding="utf-8"))
        if (
            payload.get("command") == command
            and payload.get("input_sha256") == _hashes(inputs)
            and payload.get("output_sha256") == _hashes(outputs)
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
                "schema": "ps_dfsc_confirmation_step_v1",
                "completed_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "label": label,
                "attempt": attempt,
                "command": command,
                "input_sha256": _hashes(inputs),
                "output_sha256": _hashes(outputs),
            }
            temporary = done.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(done)
            return
    raise RuntimeError(
        f"confirmation step failed closed after {attempts} attempts: {label}"
    )


def _python(module: str, *arguments: str) -> list[str]:
    return [sys.executable, "-u", "-m", module, *arguments]


def _wait_for_locks(poll_seconds: float) -> Path:
    marker = (
        ROOT / "outputs" / "ps_dfsc" / "validation_locks.complete.json"
    )
    while not marker.is_file():
        time.sleep(poll_seconds)
    return marker


def _load_locks() -> dict[int, tuple[Path, dict]]:
    result = {}
    for outer in (1, 2, 3):
        path = (
            ROOT
            / "outputs"
            / "ps_dfsc"
            / f"outer{outer}"
            / "lock_manifest.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        verify_lock_manifest(payload)
        if int(payload["outer"]) != outer:
            raise ValueError("confirmation lock outer mismatch")
        result[outer] = (path, payload)
    return result


def _absence_audit(
    marker: Path, locks: dict[int, tuple[Path, dict]]
) -> Path:
    output = (
        ROOT
        / "outputs"
        / "ps_dfsc"
        / "pre_confirmation_absence.audit.json"
    )
    lock_hashes = {
        f"outer{outer}": _sha256(path)
        for outer, (path, _payload) in locks.items()
    }
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload["lock_sha256"] != lock_hashes:
            raise ValueError("pre-confirmation audit lock hashes changed")
        return output
    base = ROOT / "outputs" / "ps_dfsc" / "base"
    forbidden = [
        path
        for path in base.rglob("*confirmation_test*")
        if path.is_file()
    ]
    confirmation = ROOT / "outputs" / "ps_dfsc" / "confirmation"
    if confirmation.exists():
        forbidden += [path for path in confirmation.rglob("*") if path.is_file()]
    if forbidden:
        raise RuntimeError(
            "confirmation artifacts exist before the absence audit: "
            + ", ".join(str(path) for path in forbidden[:10])
        )
    payload = {
        "schema": "ps_dfsc_pre_confirmation_absence_v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_lock_marker_sha256": _sha256(marker),
        "lock_sha256": lock_hashes,
        "confirmation_artifacts_present": False,
        "test_truth_accessed": False,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def _confirmation_args(
    *,
    outer: int,
    lock_path: Path,
    lock: dict,
    phase: str,
) -> list[str]:
    command = _python(
        "repro_scripts.run_ps_dfsc_confirmation_canonical_v7",
        "--base-config",
        str(ROOT / "repro_configs" / "ps_dfsc_base.json"),
        "--lock",
        str(lock_path),
        "--outer",
        str(outer),
        "--mapping",
        str(
            ROOT
            / "repro_configs"
            / f"ps_dfsc_mapping_outer{outer}.json"
        ),
        "--phase",
        phase,
        "--device",
        "cuda",
        "--clusters",
        "20",
        "--output-root",
        str(ROOT / "outputs" / "ps_dfsc" / "confirmation"),
    )
    if not bool(lock["used_identity_fallback"]):
        command += [
            "--checkpoint",
            str(lock["selection"]["candidate"]["checkpoint"]),
        ]
    return command


def _run_outer(
    outer: int,
    lock_path: Path,
    lock: dict,
    absence: Path,
    attempts: int,
) -> list[Path]:
    base_outer = (
        ROOT / "outputs" / "ps_dfsc" / "base" / f"outer{outer}"
    )
    scenario_files = [
        base_outer / "scenarios" / f"seed{seed}_confirmation_test.npz"
        for seed in range(3)
    ]
    pooled = (
        base_outer / "pooled" / "confirmation_test_M100.npz"
    )
    confirmation = (
        ROOT
        / "outputs"
        / "ps_dfsc"
        / "confirmation"
        / f"outer{outer}"
    )
    calibrated = confirmation / "ps_dfsc_test.npz"
    identity = confirmation / "identity_test.npz"
    mapping = (
        ROOT / "repro_configs" / f"ps_dfsc_mapping_outer{outer}.json"
    )
    base_models = [
        base_outer / "runs" / f"seed{seed}" / "final.pt"
        for seed in range(3)
    ]
    _run_step(
        label=f"outer{outer}_generate",
        command=_confirmation_args(
            outer=outer,
            lock_path=lock_path,
            lock=lock,
            phase="generate",
        ),
        inputs=[lock_path, absence, mapping, *base_models],
        outputs=[*scenario_files, pooled],
        attempts=attempts,
    )
    calibration_inputs = [lock_path, pooled, mapping]
    if not bool(lock["used_identity_fallback"]):
        calibration_inputs.append(
            Path(lock["selection"]["candidate"]["checkpoint"])
        )
    _run_step(
        label=f"outer{outer}_calibrate",
        command=_confirmation_args(
            outer=outer,
            lock_path=lock_path,
            lock=lock,
            phase="calibrate",
        ),
        inputs=calibration_inputs,
        outputs=[calibrated, identity],
        attempts=attempts,
    )
    exact_candidate = confirmation / "exact_ps_dfsc.csv"
    exact_candidate_summary = exact_candidate.with_suffix(".summary.json")
    exact_identity = confirmation / "exact_identity.csv"
    exact_identity_summary = exact_identity.with_suffix(".summary.json")
    _run_step(
        label=f"outer{outer}_candidate_exact",
        command=_python(
            "repro_scripts.run_ps_dfsc_exact_cached_v2",
            "--split-role",
            "confirmation",
            "--lock",
            str(lock_path),
            "--calibrated",
            str(calibrated),
            "--truth",
            str(pooled),
            "--mapping",
            str(mapping),
            "--outer",
            str(outer),
            "--method",
            (
                "PS-DFSC identity fallback"
                if bool(lock["used_identity_fallback"])
                else "PS-DFSC"
            ),
            "--mip-gap",
            "0.001",
            "--time-limit",
            "600",
            "--per-day-attempts",
            "3",
            "--cache-dir",
            str(confirmation / "cache_ps_dfsc"),
            "--output",
            str(exact_candidate),
        ),
        inputs=[lock_path, calibrated, pooled, mapping],
        outputs=[exact_candidate, exact_candidate_summary],
        attempts=attempts,
    )
    _run_step(
        label=f"outer{outer}_identity_exact",
        command=_python(
            "repro_scripts.run_ps_dfsc_exact_cached_v2",
            "--split-role",
            "confirmation",
            "--lock",
            str(lock_path),
            "--calibrated",
            str(identity),
            "--truth",
            str(pooled),
            "--mapping",
            str(mapping),
            "--outer",
            str(outer),
            "--method",
            "MM-JDWind identity",
            "--mip-gap",
            "0.001",
            "--time-limit",
            "600",
            "--per-day-attempts",
            "3",
            "--cache-dir",
            str(confirmation / "cache_identity"),
            "--output",
            str(exact_identity),
        ),
        inputs=[lock_path, identity, pooled, mapping],
        outputs=[exact_identity, exact_identity_summary],
        attempts=attempts,
    )
    safety = confirmation / "confirmation_safety.json"
    safety_metrics = safety.with_suffix(".metrics.npz")
    _run_step(
        label=f"outer{outer}_safety",
        command=_python(
            "repro_scripts.run_ps_dfsc_confirmation_safety_v1",
            "--lock",
            str(lock_path),
            "--outer",
            str(outer),
            "--candidate",
            str(calibrated),
            "--baseline",
            str(identity),
            "--truth",
            str(pooled),
            "--regime-reference",
            str(
                ROOT
                / "outputs"
                / "ps_dfsc"
                / f"outer{outer}"
                / "development_train_prepared.npz"
            ),
            "--seed",
            str(10_000 + outer),
            "--output",
            str(safety),
        ),
        inputs=[
            lock_path,
            calibrated,
            identity,
            pooled,
            ROOT
            / "outputs"
            / "ps_dfsc"
            / f"outer{outer}"
            / "development_train_prepared.npz",
        ],
        outputs=[safety, safety_metrics],
        attempts=attempts,
    )
    manifest = confirmation / "confirmation_access_manifest.json"
    artifacts = [
        *scenario_files,
        pooled,
        calibrated,
        identity,
        exact_candidate,
        exact_candidate_summary,
        exact_identity,
        exact_identity_summary,
        safety,
        safety_metrics,
    ]
    payload = {
        "schema": "ps_dfsc_confirmation_access_manifest_v1",
        "outer": outer,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_path": str(lock_path.resolve()),
        "lock_sha256": _sha256(lock_path),
        "pre_confirmation_absence_sha256": _sha256(absence),
        "artifact_sha256": _hashes(artifacts),
        "generated_after_selection_lock": True,
        "test_truth_accessed": True,
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return [*artifacts, manifest]


def _audit_final(
    locks: dict[int, tuple[Path, dict]],
    absence: Path,
    report: Path,
    markdown: Path,
    outer_artifacts: dict[int, list[Path]],
) -> Path:
    absence_payload = json.loads(absence.read_text(encoding="utf-8"))
    audit_time = datetime.fromisoformat(
        absence_payload["audited_at_utc"]
    )
    checks = {}
    total_rows = 0
    for outer, (lock_path, lock) in locks.items():
        verify_lock_manifest(lock)
        lock_time = datetime.fromisoformat(lock["created_utc"])
        artifacts = outer_artifacts[outer]
        after_lock = all(
            datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            )
            > lock_time
            for path in artifacts
        )
        after_audit = all(
            datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            )
            > audit_time
            for path in artifacts
        )
        candidate_rows = len(
            pd.read_csv(
                ROOT
                / "outputs"
                / "ps_dfsc"
                / "confirmation"
                / f"outer{outer}"
                / "exact_ps_dfsc.csv"
            )
        )
        identity_rows = len(
            pd.read_csv(
                ROOT
                / "outputs"
                / "ps_dfsc"
                / "confirmation"
                / f"outer{outer}"
                / "exact_identity.csv"
            )
        )
        total_rows += min(candidate_rows, identity_rows)
        checks[f"outer{outer}"] = {
            "lock_sha256": _sha256(lock_path),
            "all_artifacts_after_lock": after_lock,
            "all_artifacts_after_absence_audit": after_audit,
            "candidate_rows": candidate_rows,
            "identity_rows": identity_rows,
        }
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    complete = bool(
        total_rows == 150
        and int(report_payload["paired_cases"]) == 150
        and all(
            value["all_artifacts_after_lock"]
            and value["all_artifacts_after_absence_audit"]
            and value["candidate_rows"] == 50
            and value["identity_rows"] == 50
            for value in checks.values()
        )
    )
    output = (
        ROOT
        / "outputs"
        / "ps_dfsc"
        / "confirmation"
        / "completion_audit.json"
    )
    output.write_text(
        json.dumps(
            {
                "schema": "ps_dfsc_confirmation_completion_audit_v1",
                "completed_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "complete": complete,
                "paired_rows": total_rows,
                "report_sha256": _sha256(report),
                "markdown_sha256": _sha256(markdown),
                "absence_audit_sha256": _sha256(absence),
                "outer_checks": checks,
                "test_results_used_for_model_selection": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if not complete:
        raise RuntimeError("confirmation completion audit failed closed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    marker = _wait_for_locks(args.poll_seconds)
    locks = _load_locks()
    absence = _absence_audit(marker, locks)
    outer_artifacts = {
        outer: _run_outer(
            outer,
            lock_path,
            lock,
            absence,
            args.attempts,
        )
        for outer, (lock_path, lock) in locks.items()
    }
    report = (
        ROOT / "outputs" / "ps_dfsc" / "confirmation" / "report.json"
    )
    markdown = report.with_suffix(".md")
    exact_inputs = [
        ROOT
        / "outputs"
        / "ps_dfsc"
        / "confirmation"
        / f"outer{outer}"
        / name
        for outer in (1, 2, 3)
        for name in (
            "exact_ps_dfsc.csv",
            "exact_identity.csv",
            "confirmation_safety.json",
        )
    ]
    _run_step(
        label="final_report",
        command=_python(
            "repro_scripts.ps_dfsc_final_report_publication",
            "--confirmation-root",
            str(ROOT / "outputs" / "ps_dfsc" / "confirmation"),
            "--output-json",
            str(report),
            "--output-markdown",
            str(markdown),
        ),
        inputs=exact_inputs,
        outputs=[report, markdown],
        attempts=args.attempts,
    )
    _audit_final(
        locks, absence, report, markdown, outer_artifacts
    )


if __name__ == "__main__":
    main()
