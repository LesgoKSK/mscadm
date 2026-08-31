# MS-CADM formal reproduction status

Last updated: 2026-07-16 (Asia/Shanghai)

This file separates completed numerical results from still-running experiments. It is intentionally not a claim of exact reproduction.

## Completed protocol checks

- Exact GEFCom2014 wind split used by the public Dumas preprocessing: 631/50/50 days per zone for train/validation/test.
- Ten weather features per hour and zone, in executable-reference order: `U10,V10,U100,V100,ws10,ws100,we10,we100,wd10,wd100`.
- Missing target values are forward-filled and weather/target standardizers are fit on training data only.
- Evaluation uses 100 scenarios, 24 hours, 50 test days, and ten zones.
- RAND reproduces the paper nearly exactly, providing an end-to-end check of data selection and metric definitions.

## All-zone scenario metrics

| Implementation | MAE | RMSE | CRPS | QS | ES | VS |
|---|---:|---:|---:|---:|---:|---:|
| Paper MS-CADM | 0.1191 | 0.1645 | 0.0873 | 0.0441 | 0.5380 | 18.14 |
| MS-CADM, 250 ancestral steps | 0.124248 | 0.177632 | 0.107400 | 0.054139 | 0.681950 | 22.019999 |
| MS-CADM reconstruction branch | 0.126226 | 0.174407 | 0.099098 | 0.050123 | 0.611520 | 19.140327 |
| Reference-behavior VAE | 0.123630 | 0.167013 | 0.087727 | 0.044337 | 0.545616 | 17.717475 |
| Reference-schedule WGAN-GP | 0.129290 | 0.177066 | 0.096574 | 0.048859 | 0.592980 | 18.596385 |
| Conditional RealNVP approximation | 0.117057 | 0.164107 | 0.085050 | 0.043012 | 0.532192 | 16.948224 |
| Public-energy WaveNet DDPM | 0.120106 | 0.160717 | 0.080505 | 0.040624 | 0.506541 | 15.870887 |
| RAND | 0.258707 | 0.300767 | 0.168710 | 0.085216 | 0.959265 | 23.172145 |

The VAE and RAND results are close to the paper's values, while the reconstructed MS-CADM is not. The discrepancy is therefore not plausibly explained by the split or metrics alone. The most likely remaining sources are undisclosed architecture/training details and sampling calibration. The reconstructed model is substantially under-dispersed: at nominal 90% coverage, its observed coverage is 35.95% (original branch) or 49.58% (feature-mask/linear-schedule branch).

The RealNVP and DDPM rows are strong, documented approximations; they are not represented as exact author implementations. Both outperform the paper's reported MS-CADM on several metrics, which makes implementation parity and baseline tuning important threats to the paper's comparison.

## Seven-day RTS-24 operational evaluation

The following values are averages over seven test days. The workflow uses ten clustered scenarios for day-ahead commitment, then replays actual wind under fixed commitment. Penalty cost is reported separately instead of being hidden in total cost.

| Model | Total cost | Penalty cost | Load shedding | Wind curtailment |
|---|---:|---:|---:|---:|
| DDPM | 352829.40 | 5031.82 | 0.7087 | 54.0394 |
| MS-CADM | 371161.36 | 4084.52 | 4.0139 | 0.8825 |
| NF | 552754.49 | 199916.50 | 195.1205 | 59.9498 |
| VAE reference | 367094.74 | 9995.04 | 6.9583 | 37.9595 |
| WGAN reference | 404085.87 | 68459.07 | 66.5294 | 24.1212 |

These costs are not directly claimed to reproduce the paper's Table 5 because the paper does not disclose enough case-mapping, representative-day selection, or scenario-reduction detail. They are a transparent reconstruction with all reliability components retained.

## Still running / queued

- Formal GPU QRGBM (24 hourly models, 99 quantiles, 300 boosted trees per quantile).
- QRGBM seven-day RTS-24 evaluation after generation.
- MS-CADM denoising-step sweep for 10/20/50/100/250 steps.
- Independent single-zone Table 2 training runs.
- Independent Table 4 ablations.
- Final regeneration of tables and figures after all archives are present.

Machine-readable outputs are under `outputs/full_reproduction/`; the authoritative scenario table is `outputs/full_reproduction/tables/table1_metrics.csv`, interval calibration is `outputs/full_reproduction/tables/interval_metrics.json`, and the operational outputs are under `outputs/full_reproduction/suc/`.
