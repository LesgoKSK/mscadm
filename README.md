# MS-CADM Reproduction and Joint Wind Scenario Research

This repository contains a clean-room reproduction of **“Wind power scenario
generation via multi-scale condition adaptive diffusion model”** and the
subsequent research code used to diagnose and improve multi-zone wind-power
scenario generation.

The current project is broader than a paper reproduction. It studies whether
joint generative models can preserve:

- calibrated marginal uncertainty;
- realistic hour-to-hour ramps;
- cross-zone and lagged temporal dependence; and
- exact boundary states at 0 and 1.

The main setting is GEFCom2014 Wind with 10 zones, 24 forecast hours, and 100
joint scenario members per forecast day.

## Current research status

The repository records both positive and negative results.

1. **MS-CADM reproduction:** the full data, training, sampling, and evaluation
   chain was reconstructed. The local headline CRPS is 0.1074 versus 0.0873
   reported in the paper, and local 90% coverage is 0.3595. The implementation
   therefore does not reproduce the paper’s headline probabilistic result.
2. **Cross-model diagnosis:** marginal level CRPS is close across the archived
   models, while ramp, total-variation, maximum-jump, and lag-1 diagnostics
   expose a temporal-dynamics gap, especially under dynamic NWP regimes.
3. **Architecture-v1 protocol:** leakage controls, fixed validation randomness,
   common atom allocation, and three-seed training substantially reduce seed
   spread and provide a fair comparison base.
4. **Temporal mechanisms:** chronological ordering consistently outperforms
   fixed shuffle controls, so temporal information is real. The tested
   `T1-feature` and `T1-source` mechanisms nevertheless fail preregistered
   non-inferiority or stability gates and are recorded as **No-Go**.
5. **Flow versus Joint DDPM:** the corrected v-prediction DDPM improves lagged
   dependence, while Flow is better on marginal/joint quality and coverage.
   Neither family satisfies the preregistered dominance rule; the formal winner
   remains unresolved.

The next planned experiment is a **Temporal Utility Probe**: 8 log-SNR bins ×
chronological/shuffled ordering × 3 training seeds. It will determine whether
temporal intervention should depend on NWP dynamicity, diffusion stage, both,
or neither.

> All current architecture conclusions are validation-stage findings. Sealed
> selection/calibration roles and final external testing are not treated as
> completed evidence.

## Repository layout

| Path | Purpose |
| --- | --- |
| `mscadm/` | Original MS-CADM-style implementation |
| `repro/` | Clean-room reproduction components and baseline models |
| `architecture_v1/` | Leakage-controlled joint Flow/DDPM and temporal-mechanism framework |
| `cross_model_diagnostics/` | Ramp, lagged-dependence, atom, and NWP-regime diagnostics |
| `caa_rahc/`, `cr_mscadm/`, `mm_jdwind/`, `ps_dfsc/`, `rahc/`, `stgf_flow/` | Earlier and adjacent candidate methods |
| `repro_scripts/` | Training, evaluation, audit, and formal experiment runners |
| `repro_configs/` | Frozen and preregistered experiment configurations |
| `tests/` | Unit and protocol regression tests |
| `reports/` | Research reports, standalone HTML presentations, and figure sources |

## Data and generated artifacts

The GEFCom2014 data, downloaded papers, checkpoints, and generated experiment
outputs are intentionally not committed.

Expected local-only locations include:

```text
Data/
literature/
outputs/
checkpoints/
tmp/
```

These paths are excluded by `.gitignore`. Obtain the data and papers from their
authorized sources and review their redistribution terms before use.

## Environment

Use Python 3.11 and install the reproduction dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-repro.txt
```

Some adjacent experiments require the additional packages listed in
`requirements-ps-dfsc.txt`.

## Basic checks

Run the test suite:

```bash
python -m pytest -q
```

Run the small MS-CADM smoke configuration:

```bash
python scripts/train.py --config configs/smoke.yaml
```

Evaluate a generated checkpoint:

```bash
python scripts/evaluate.py \
  --checkpoint outputs/smoke/latest.pt \
  --max-batches 1
```

The formal Architecture-v1 experiments are protocol-controlled. Read the
corresponding frozen configuration and report before running them; entry points
are documented in `reports/ARCHITECTURE_V1_IMPLEMENTATION_RUNBOOK.md` and the
matching `reports/ARCHITECTURE_V1_*_PROTOCOL.md` files.

## Main reports

- [Current research progress — standalone HTML](reports/CURRENT_RESEARCH_PROGRESS_GROUP_MEETING_STANDALONE.html)
- [Current research progress — clear narrative DOCX](reports/CURRENT_RESEARCH_PROGRESS_CLEAR_NARRATIVE.docx)
- [MS-CADM reproduction report](reports/MSCADM_PAPER_REPRODUCTION_REPORT.md)
- [Cross-model architecture diagnostic](reports/CROSS_MODEL_ARCHITECTURE_DIAGNOSTIC.md)
- [Temporal mechanism v3.2 result](reports/ARCHITECTURE_V1_TEMPORAL_MECHANISM_V3_2_RESULT.md)
- [Flow versus Joint DDPM formal comparison](reports/ARCHITECTURE_V1_FAMILY_V1_1_FORMAL_COMPARISON_RESULT.md)

## Reproducibility and interpretation

- Historical diagnostic dates are marked `R-SEEN` and are not reused as unseen
  confirmation evidence.
- Validation-informed architecture decisions are distinguished from sealed
  selection/calibration and final external tests.
- Negative results and failed sampler contracts remain in the repository to
  preserve the actual decision history.
- Exact author-code equivalence is not claimed because the source paper does
  not disclose all implementation details.

## License and third-party material

No open-source license has been selected for this repository yet. Public
visibility alone does not grant redistribution rights. Third-party papers,
datasets, and generated model artifacts are not included; see
`THIRD_PARTY_NOTICES.md` for additional attribution notes.
