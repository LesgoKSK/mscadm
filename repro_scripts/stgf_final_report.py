from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STGF_ROOT = ROOT / "outputs" / "stgf_confirmation_v2"
DDPM_ROOT = ROOT / "outputs" / "ddpm_stgf_confirmation_v2"
OUTPUT = STGF_ROOT / "final_analysis"
OUTERS = (1, 2, 3)
SEEDS = (0, 1, 2)
DAY_METRICS = (
    "CRPS",
    "MAE",
    "coverage_90",
    "width_90",
    "joint_ES_240",
    "adjacency_VS",
    "aggregate_CRPS",
    "ramp_CRPS",
    "zero_Brier",
    "daily_zone_any_zero_Brier",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def coordinate_crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    observed = np.asarray(truth, dtype=np.float64)
    members = values.shape[1]
    first = np.abs(values - observed[:, None]).mean(axis=1)
    ordered = np.sort(values, axis=1)
    weights = 2 * np.arange(members, dtype=np.float64) - members + 1
    pair_half = np.sum(
        ordered
        * weights.reshape((1, members) + (1,) * (values.ndim - 2)),
        axis=1,
    ) / members**2
    return (first - pair_half).reshape(len(values), -1).mean(axis=1)


def energy_per_day(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64).reshape(
        len(samples), samples.shape[1], -1
    )
    observed = np.asarray(truth, dtype=np.float64).reshape(len(truth), -1)
    result = np.empty(len(values), dtype=np.float64)
    for index, (ensemble, target) in enumerate(zip(values, observed)):
        first = np.linalg.norm(ensemble - target[None], axis=1).mean()
        pair = np.linalg.norm(
            ensemble[:, None] - ensemble[None, :], axis=-1
        ).mean()
        result[index] = first - 0.5 * pair
    return result


def adjacency_vs_per_day(
    samples: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    temporal_truth = np.abs(truth[:, :, 1:] - truth[:, :, :-1]) ** 0.5
    temporal_sample = np.abs(
        samples[:, :, :, 1:] - samples[:, :, :, :-1]
    ) ** 0.5
    spatial_truth = np.abs(truth[:, 1:] - truth[:, :-1]) ** 0.5
    spatial_sample = np.abs(
        samples[:, :, 1:] - samples[:, :, :-1]
    ) ** 0.5
    return (
        (temporal_truth - temporal_sample.mean(axis=1))
        .reshape(len(samples), -1)
        .square()
        .mean(axis=1)
        + (spatial_truth - spatial_sample.mean(axis=1))
        .reshape(len(samples), -1)
        .square()
        .mean(axis=1)
    )


def per_day(
    scenarios: np.ndarray, observations: np.ndarray
) -> dict[str, np.ndarray]:
    values = np.asarray(scenarios, dtype=np.float64)
    truth = np.asarray(observations, dtype=np.float64)
    mean = values.mean(axis=1)
    lower = np.quantile(values, 0.05, axis=1)
    upper = np.quantile(values, 0.95, axis=1)
    aggregate = values.mean(axis=2)
    aggregate_truth = truth.mean(axis=1)
    ramps = np.diff(values, axis=-1)
    truth_ramps = np.diff(truth, axis=-1)
    zero_probability = (values == 0.0).mean(axis=1)
    sample_any_zero = (values == 0.0).any(axis=-1).mean(axis=1)
    truth_any_zero = (truth == 0.0).any(axis=-1)
    return {
        "CRPS": coordinate_crps(values, truth),
        "MAE": np.abs(mean - truth).reshape(len(values), -1).mean(axis=1),
        "coverage_90": (
            (truth >= lower) & (truth <= upper)
        ).reshape(len(values), -1).mean(axis=1),
        "width_90": (upper - lower).reshape(len(values), -1).mean(axis=1),
        "joint_ES_240": energy_per_day(values, truth),
        "adjacency_VS": adjacency_vs_per_day(values, truth),
        "aggregate_CRPS": coordinate_crps(aggregate, aggregate_truth),
        "ramp_CRPS": coordinate_crps(ramps, truth_ramps),
        "zero_Brier": (
            zero_probability - (truth == 0.0)
        ).reshape(len(values), -1).square().mean(axis=1),
        "daily_zone_any_zero_Brier": (
            sample_any_zero - truth_any_zero
        ).square().mean(axis=1),
    }


def stgf_archive(outer: int, mode: str, seed: int) -> Path:
    return (
        STGF_ROOT
        / f"outer{outer}"
        / "scenarios"
        / f"{mode}_seed{seed}_test.npz"
    )


def load_days(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return (
            per_day(stored["scenarios"], stored["observations"]),
            stored["day"].copy(),
        )


def metric_record(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def stgf_metric(outer: int, mode: str, seed: int) -> dict[str, float]:
    return metric_record(
        stgf_archive(outer, mode, seed).with_suffix(
            ".common.metrics.json"
        )
    )


def ddpm_archive(outer: int) -> Path:
    return (
        DDPM_ROOT
        / f"outer{outer}"
        / "scenarios"
        / "ddpm_seed0_test.npz"
    )


def summarize(
    records: list[dict[str, float]]
) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean([record[metric] for record in records])),
            "std": float(
                np.std([record[metric] for record in records], ddof=1)
            )
            if len(records) > 1
            else 0.0,
            "n_evaluation_cells": len(records),
        }
        for metric in records[0]
    }


def average_seed_days(mode: str) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for outer in OUTERS:
        values = [load_days(stgf_archive(outer, mode, seed))[0] for seed in SEEDS]
        result[outer] = {
            metric: np.mean([item[metric] for item in values], axis=0)
            for metric in DAY_METRICS
        }
    return result


def one_model_days(
    mode: str, seed: int
) -> dict[int, dict[str, np.ndarray]]:
    return {
        outer: load_days(stgf_archive(outer, mode, seed))[0]
        for outer in OUTERS
    }


def ddpm_days() -> dict[int, dict[str, np.ndarray]]:
    return {outer: load_days(ddpm_archive(outer))[0] for outer in OUTERS}


def stratified_bootstrap(
    deltas: dict[int, np.ndarray], *, seed: int, replicates: int = 20_000
) -> dict[str, float | int | str]:
    generator = np.random.default_rng(seed)
    observed = float(np.mean(np.concatenate(list(deltas.values()))))
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = [
            values[generator.integers(0, len(values), size=len(values))]
            for values in deltas.values()
        ]
        samples[index] = np.mean(np.concatenate(selected))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_difference": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_difference_below_zero": float(
            np.mean(samples < 0.0)
        ),
        "bootstrap_replicates": replicates,
        "bootstrap_unit": "calendar day, stratified within frozen test block",
    }


def comparison_bootstrap(
    left: dict[int, dict[str, np.ndarray]],
    right: dict[int, dict[str, np.ndarray]],
    *,
    seed_base: int,
) -> dict[str, Any]:
    return {
        metric: stratified_bootstrap(
            {
                outer: left[outer][metric] - right[outer][metric]
                for outer in OUTERS
            },
            seed=seed_base + index,
        )
        for index, metric in enumerate(DAY_METRICS)
    }


def relative_improvement(smaller: float, baseline: float) -> float:
    return float((baseline - smaller) / baseline)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mode_records = {
        mode: [stgf_metric(outer, mode, 0) for outer in OUTERS]
        for mode in (
            "time_domain",
            "graph_only",
            "time_frequency",
            "stgf",
        )
    }
    stgf_all_records = [
        stgf_metric(outer, "stgf", seed)
        for outer in OUTERS
        for seed in SEEDS
    ]
    ddpm_records = [
        metric_record(
            ddpm_archive(outer).with_suffix(".metrics.json")
        )
        for outer in OUTERS
    ]
    summaries = {
        "time_domain_seed0": summarize(mode_records["time_domain"]),
        "graph_only_seed0": summarize(mode_records["graph_only"]),
        "time_frequency_seed0": summarize(
            mode_records["time_frequency"]
        ),
        "stgf_seed0": summarize(mode_records["stgf"]),
        "stgf_all_seeds": summarize(stgf_all_records),
        "ddpm_seed0": summarize(ddpm_records),
    }
    stgf_avg_days = average_seed_days("stgf")
    stgf0_days = one_model_days("stgf", 0)
    time_days = one_model_days("time_domain", 0)
    time_frequency_days = one_model_days("time_frequency", 0)
    ddpm_day_values = ddpm_days()
    bootstrap = {
        "stgf_seed_average_vs_time_domain_seed0": comparison_bootstrap(
            stgf_avg_days, time_days, seed_base=2_026_700
        ),
        "stgf_seed0_vs_time_domain_seed0": comparison_bootstrap(
            stgf0_days, time_days, seed_base=2_026_800
        ),
        "stgf_seed0_vs_time_frequency_seed0": comparison_bootstrap(
            stgf0_days, time_frequency_days, seed_base=2_026_900
        ),
        "stgf_seed_average_vs_ddpm_seed0": comparison_bootstrap(
            stgf_avg_days, ddpm_day_values, seed_base=2_027_000
        ),
    }
    outer_direction: dict[str, dict[str, int]] = {}
    for metric in DAY_METRICS:
        outer_direction[metric] = {
            "stgf_seed_average_better_than_time_domain": sum(
                np.mean(stgf_avg_days[outer][metric])
                < np.mean(time_days[outer][metric])
                for outer in OUTERS
            ),
            "stgf_seed0_better_than_time_frequency": sum(
                np.mean(stgf0_days[outer][metric])
                < np.mean(time_frequency_days[outer][metric])
                for outer in OUTERS
            ),
            "stgf_seed_average_better_than_ddpm": sum(
                np.mean(stgf_avg_days[outer][metric])
                < np.mean(ddpm_day_values[outer][metric])
                for outer in OUTERS
            ),
        }
    primary = summaries["stgf_all_seeds"]
    baseline = summaries["time_domain_seed0"]
    secondary = (
        "MAE",
        "aggregate_CRPS",
        "ramp_CRPS",
        "spectral_energy_MAE",
        "cross_zone_correlation_Frobenius",
    )
    secondary_degradation = {
        metric: float(
            primary[metric]["mean"] / baseline[metric]["mean"] - 1.0
        )
        for metric in secondary
    }
    gates = {
        "CRPS_relative_improvement_at_least_1pct": (
            relative_improvement(
                primary["CRPS"]["mean"], baseline["CRPS"]["mean"]
            )
            >= 0.01
        ),
        "joint_ES_240_relative_improvement_at_least_1pct": (
            relative_improvement(
                primary["joint_ES_240"]["mean"],
                baseline["joint_ES_240"]["mean"],
            )
            >= 0.01
        ),
        "adjacency_VS_relative_improvement_at_least_1pct": (
            relative_improvement(
                primary["adjacency_VS"]["mean"],
                baseline["adjacency_VS"]["mean"],
            )
            >= 0.01
        ),
        "coverage_90_in_0.88_to_0.92": (
            0.88 <= primary["coverage_90"]["mean"] <= 0.92
        ),
        "CRPS_better_on_at_least_2_of_3_blocks": (
            outer_direction["CRPS"][
                "stgf_seed_average_better_than_time_domain"
            ]
            >= 2
        ),
        "no_secondary_metric_degrades_more_than_1pct": all(
            value <= 0.01 for value in secondary_degradation.values()
        ),
    }
    source_paths = [
        ROOT / "repro_configs" / "stgf_confirmation_splits.json",
        ROOT / "repro_configs" / "stgf_confirmation_v2.json",
        ROOT / "repro_configs" / "ddpm_stgf_confirmation_v2.json",
        ROOT
        / "outputs"
        / "stgf_development_v2"
        / "selection.common_basis.lock.json",
        STGF_ROOT / "confirmation_manifest.json",
    ]
    result = {
        "schema": "stgf_flow_final_analysis_v2",
        "evidence_scope": (
            "post-lock internal confirmation on 150 new frozen days; "
            "not external validation"
        ),
        "design": {
            "unique_models": (
                "6 STGF-family models: four representations at seed0 plus "
                "STGF seeds1/2"
            ),
            "evaluation": (
                "each unique model evaluated on three mutually exclusive "
                "50-day blocks; all blocks share the same universal 481-day "
                "training split"
            ),
            "ddpm": "one seed0 model evaluated on the same three blocks",
            "ensemble_members": 100,
            "stgf_nfe": 16,
            "ddpm_sampling_steps": 250,
            "test_days": 150,
        },
        "summaries": summaries,
        "paired_day_bootstrap": bootstrap,
        "outer_direction": outer_direction,
        "relative_improvements": {
            "stgf_all_seeds_vs_time_domain": {
                metric: relative_improvement(
                    primary[metric]["mean"], baseline[metric]["mean"]
                )
                for metric in (
                    "CRPS",
                    "joint_ES_240",
                    "adjacency_VS",
                )
            }
        },
        "secondary_relative_degradation": secondary_degradation,
        "predeclared_success_gates": gates,
        "source_hashes": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in source_paths
        ],
        "interpretation": {
            "full_graph_frequency_hypothesis": (
                "not confirmed if any primary gate fails"
            ),
            "observed_pattern": (
                "The locked STGF model widens intervals and improves "
                "coverage, but the graph transform does not deliver the "
                "predeclared CRPS/joint-score gains over time-domain or "
                "time-frequency ablations."
            ),
            "strong_result": (
                "The joint Rectified Flow remains competitive with or better "
                "than the 250-step independent DDPM on several marginal and "
                "joint scores at only 16 flow evaluations."
            ),
        },
    }
    json_path = OUTPUT / "stgf_final_analysis.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    def value(model: str, metric: str) -> str:
        item = summaries[model][metric]
        return f"{item['mean']:.6f} ± {item['std']:.6f}"

    ci_time = bootstrap[
        "stgf_seed_average_vs_time_domain_seed0"
    ]["CRPS"]
    ci_frequency = bootstrap[
        "stgf_seed0_vs_time_frequency_seed0"
    ]["CRPS"]
    ci_ddpm = bootstrap["stgf_seed_average_vs_ddpm_seed0"]["CRPS"]
    passed = sum(bool(value) for value in gates.values())
    markdown = f"""# STGF-Flow 完整实验报告

证据范围：**锁定后内部确认实验，150 个此前未用于 CAA/MM test 的新冻结日期；不是外部验证。**

## 实验设计

- 训练集：排除全部 150 个确认日期后的 481 日；validation 50 日，calibration 50 日。
- 六个唯一 flow 模型：time-domain、graph-only、time-frequency、STGF 的 seed0，以及 STGF seed1/2。
- 三个互斥 test blocks 各 50 日；由于训练/validation 完全相同，同一模型在三块上评估，而不是伪装成三次独立重训。
- 所有模型生成 100 条 `10 Zone × 24 h` 联合轨迹。STGF 使用 16 NFE；DDPM 使用 250 个 DDIM 步。
- 所有频谱结构指标统一使用仅由 481 日训练集拟合的 GFT×DCT 评估基。

## 确认结果

| 模型 | CRPS ↓ | MAE ↓ | Coverage90 | Width90 ↓ | Joint ES ↓ | Adj. VS ↓ | Spectral MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time-domain seed0 | {value('time_domain_seed0', 'CRPS')} | {value('time_domain_seed0', 'MAE')} | {value('time_domain_seed0', 'coverage_90')} | {value('time_domain_seed0', 'width_90')} | {value('time_domain_seed0', 'joint_ES_240')} | {value('time_domain_seed0', 'adjacency_VS')} | {value('time_domain_seed0', 'spectral_energy_MAE')} |
| Graph-only seed0 | {value('graph_only_seed0', 'CRPS')} | {value('graph_only_seed0', 'MAE')} | {value('graph_only_seed0', 'coverage_90')} | {value('graph_only_seed0', 'width_90')} | {value('graph_only_seed0', 'joint_ES_240')} | {value('graph_only_seed0', 'adjacency_VS')} | {value('graph_only_seed0', 'spectral_energy_MAE')} |
| Time-frequency seed0 | {value('time_frequency_seed0', 'CRPS')} | {value('time_frequency_seed0', 'MAE')} | {value('time_frequency_seed0', 'coverage_90')} | {value('time_frequency_seed0', 'width_90')} | {value('time_frequency_seed0', 'joint_ES_240')} | {value('time_frequency_seed0', 'adjacency_VS')} | {value('time_frequency_seed0', 'spectral_energy_MAE')} |
| STGF seed0 | {value('stgf_seed0', 'CRPS')} | {value('stgf_seed0', 'MAE')} | {value('stgf_seed0', 'coverage_90')} | {value('stgf_seed0', 'width_90')} | {value('stgf_seed0', 'joint_ES_240')} | {value('stgf_seed0', 'adjacency_VS')} | {value('stgf_seed0', 'spectral_energy_MAE')} |
| STGF all seeds | {value('stgf_all_seeds', 'CRPS')} | {value('stgf_all_seeds', 'MAE')} | {value('stgf_all_seeds', 'coverage_90')} | {value('stgf_all_seeds', 'width_90')} | {value('stgf_all_seeds', 'joint_ES_240')} | {value('stgf_all_seeds', 'adjacency_VS')} | {value('stgf_all_seeds', 'spectral_energy_MAE')} |
| DDPM seed0 | {value('ddpm_seed0', 'CRPS')} | {value('ddpm_seed0', 'MAE')} | {value('ddpm_seed0', 'coverage_90')} | {value('ddpm_seed0', 'width_90')} | {value('ddpm_seed0', 'joint_ES_240')} | {value('ddpm_seed0', 'adjacency_VS')} | {value('ddpm_seed0', 'spectral_energy_MAE')} |

STGF 三 seed 平均与 time-domain 的逐日配对 CRPS 差值（STGF−baseline）为 {ci_time['mean_difference']:.6f}，分层 bootstrap 95% CI [{ci_time['ci95_low']:.6f}, {ci_time['ci95_high']:.6f}]。

STGF seed0 与 time-frequency seed0 的 CRPS 差值为 {ci_frequency['mean_difference']:.6f}，95% CI [{ci_frequency['ci95_low']:.6f}, {ci_frequency['ci95_high']:.6f}]。

STGF 三 seed 平均与 DDPM 的 CRPS 差值为 {ci_ddpm['mean_difference']:.6f}，95% CI [{ci_ddpm['ci95_low']:.6f}, {ci_ddpm['ci95_high']:.6f}]。

## 预注册成功门槛

通过 {passed}/{len(gates)} 项：

{json.dumps(gates, ensure_ascii=False, indent=2)}

## 结论

完整的“图频域带来净增益”假设**未被确认**。锁定的 STGF 确实显著修复了 time-domain 的欠覆盖，但代价是更宽的区间，且没有达到预设的 CRPS、Joint ES、邻接 Variogram 改善门槛。消融证据进一步表明，时间 DCT 是更稳定的有效成分；训练相关图 GFT 加在 time-frequency 上并未产生可靠增益。

仍然存在一个可发表的正结果：联合 Rectified Flow 用 16 NFE 即可在同一新冻结测试集上与 250-step 独立 DDPM 竞争，并原生生成跨 10 Zone 的联合轨迹。更合理的下一版论文主线应从“固定相关图有效”转向“高效联合 flow + 可学习/动态图结构”，把固定 GFT 作为经严格否证的消融，而不是继续包装成已证实贡献。
"""
    markdown_path = OUTPUT / "STGF_FLOW_FINAL_REPORT.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "analysis": str(json_path.resolve()),
                "report": str(markdown_path.resolve()),
                "gates": gates,
                "stgf_vs_time_CRPS": ci_time,
                "stgf_vs_time_frequency_CRPS": ci_frequency,
                "stgf_vs_ddpm_CRPS": ci_ddpm,
            }
        )
    )


if __name__ == "__main__":
    main()
