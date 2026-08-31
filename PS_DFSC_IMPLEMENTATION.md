# PS-DFSC implementation and execution protocol

## Implemented scope

The repository now contains a complete, restartable implementation of
Proper-Score-Constrained Decision-Focused Scenario Calibration:

- frozen fresh 3x50 confirmation split;
- outer-specific leakage-audited MM-JDWind training data;
- training-only ten-zone to six-farm mapping;
- DeepSets scenario reweighting and bounded monotone transport;
- weighted proper scores, conditional coverage diagnostics and bootstrap gate;
- truth-independent identity fallback;
- differentiable fixed-commitment dispatch/reserve QP;
- exact binary two-stage DC-network SUC and realized recourse;
- beta candidate selection, lock manifest, locked confirmation runner;
- exact/relaxed mismatch and final three-outer report utilities.

The stable public Python imports are in `ps_dfsc.api`. The canonical training
CLI is `python -m repro_scripts.run_ps_dfsc_final`.

## Frozen protocol

`repro_configs/ps_dfsc_splits.json` has already been frozen:

- development train: 100 prior MM-JDWind confirmation dates;
- development validation: 50 prior MM-JDWind confirmation dates;
- fresh eligible pool: 181 dates;
- confirmation: three disjoint blocks of 50 dates;
- confirmation union SHA-256:
  `723d74022dafe32fc744a1bd16e455367b16281c89e74ef22691c17c452651ec`.

Allocation used calendar month and NWP WS100 terciles only. It did not use
wind-power targets.

Outer mappings have also been frozen:

- outer 1: `(1,7), (2,10), (3,9), (4), (5,6), (8)`;
- outer 2: `(1,7), (2,10), (3,9), (4), (5,6), (8)`;
- outer 3: `(1,9), (2,10), (3), (4), (5,6), (7,8)`.

The JSON files use zero-based zone indices; the list above is one-based for
readability.

## Environment

Use the documented Python 3.11 environment. The differentiable dependencies
are listed separately:

```powershell
python -m pip install -r requirements-ps-dfsc.txt
```

The installed and tested versions are CVXPY 1.7.5, cvxpylayers 0.1.9 and
diffcp 1.1.8.

## Full experiment order

### 1. Retrain the frozen MM-JDWind bases

Run for each outer. Training all three seeds is intentionally a long GPU job.

```powershell
python -m repro_scripts.run_ps_dfsc_base --phase train --outer 1 --device cuda
python -m repro_scripts.run_ps_dfsc_base --phase generate --outer 1 --role development_train --device cuda
python -m repro_scripts.run_ps_dfsc_base --phase generate --outer 1 --role development_validation --device cuda
python -m repro_scripts.run_ps_dfsc_base --phase pool --outer 1 --role development_train
python -m repro_scripts.run_ps_dfsc_base --phase pool --outer 1 --role development_validation
```

Repeat for outers 2 and 3. Do not generate `confirmation_test` yet.

Each pooled archive contains exactly 100 members: 34 from seed 0 and 33 each
from seeds 1 and 2.

### 2. Prepare identity costs and commitment caches

For each outer and each development role:

```powershell
python -m repro_scripts.prepare_ps_dfsc_training_v2 `
  --input outputs/ps_dfsc/base/outer1/pooled/development_train_M100.npz `
  --mapping repro_configs/ps_dfsc_mapping_outer1.json `
  --clusters 20 `
  --cache-dir outputs/ps_dfsc/outer1/baseline_cache/train `
  --training-output outputs/ps_dfsc/outer1/development_train_prepared.npz `
  --identity-output outputs/ps_dfsc/outer1/identity_train.npz
```

Run the same command for development validation. Per-day cache files make
interrupted exact-MILP preparation resumable.

### 3. Train the four beta candidates

```powershell
python -m repro_scripts.run_ps_dfsc_final train `
  --input outputs/ps_dfsc/outer1/development_train_prepared.npz `
  --mapping repro_configs/ps_dfsc_mapping_outer1.json `
  --clusters 20 --beta 0.25 --device cuda `
  --output outputs/ps_dfsc/outer1/candidates/beta025.pt
```

Repeat with beta `0`, `0.25`, `0.5` and `1`, and for all outers.

Training performs:

1. 20 proper-score warm-up epochs;
2. decision-focused fixed-commitment QP training;
3. one midpoint exact-MILP commitment refresh over all 100 development days;
4. continued QP training with the refreshed commitment cache.

The training QP uses a copper-plate relaxation for tractable gradients. All
candidate validation costs and final results use the exact DC-network MILP.

### 4. Calibrate and gate development validation

For each candidate:

```powershell
python -m repro_scripts.run_ps_dfsc_final calibrate `
  --input outputs/ps_dfsc/base/outer1/pooled/development_validation_M100.npz `
  --scenario-key scenarios `
  --checkpoint outputs/ps_dfsc/outer1/candidates/beta025.pt `
  --mapping repro_configs/ps_dfsc_mapping_outer1.json `
  --clusters 20 `
  --output outputs/ps_dfsc/outer1/candidates/beta025_validation.npz

python -m repro_scripts.run_ps_dfsc_final safety-gate `
  --candidate outputs/ps_dfsc/outer1/candidates/beta025_validation.npz `
  --baseline outputs/ps_dfsc/outer1/identity_validation.npz `
  --truth outputs/ps_dfsc/base/outer1/pooled/development_validation_M100.npz `
  --regime-reference outputs/ps_dfsc/outer1/development_train_prepared.npz `
  --output outputs/ps_dfsc/outer1/candidates/beta025_gate.json
```

Only safety-passing candidates enter the exact validation shortlist. Run
`exact-evaluate` for the top three proxy candidates, then assemble records with
`build_ps_dfsc_candidate_record` and apply
`select_ps_dfsc_candidate`.

If no candidate passes both the safety and exact decision gates, the selection
file records identity fallback. This is a valid outcome.

### 5. Measure relaxation mismatch

```powershell
python -m repro_scripts.evaluate_ps_dfsc_mismatch `
  --training-archive outputs/ps_dfsc/outer1/development_train_prepared.npz `
  --checkpoint outputs/ps_dfsc/outer1/candidates/beta025.pt `
  --mapping repro_configs/ps_dfsc_mapping_outer1.json `
  --output outputs/ps_dfsc/outer1/mismatch.csv
```

The output includes commitment Hamming distance, day-ahead dispatch MAE,
reserve MAE, objective mismatch, exact gap and solve time.

### 6. Lock before confirmation access

The lock manifest must include the selected calibrator and all three MM-JDWind
base checkpoints. The safety JSON and candidate selection must already exist.

```powershell
python -m repro_scripts.run_ps_dfsc_final lock `
  --splits repro_configs/ps_dfsc_splits.json `
  --models <selected-calibrator> <base-seed0> <base-seed1> <base-seed2> `
  --safety <selected-gate-json> `
  --outer 1 --beta 0.25 --candidate-id <locked-id> `
  --output outputs/ps_dfsc/outer1/selection.lock.json
```

### 7. Locked confirmation

Only the confirmation runner should generate or evaluate the fresh test block:

```powershell
python -m repro_scripts.run_ps_dfsc_confirmation `
  --lock outputs/ps_dfsc/outer1/selection.lock.json `
  --outer 1 `
  --checkpoint <selected-calibrator> `
  --mapping repro_configs/ps_dfsc_mapping_outer1.json `
  --regime-reference outputs/ps_dfsc/outer1/development_train_prepared.npz `
  --phase all --device cuda
```

The runner verifies the lock before generating test scenarios and again before
opening test observations for exact evaluation. Test safety metrics are
reported after locking and never trigger model reselection.

### 8. Aggregate the paper result

```powershell
python -m repro_scripts.ps_dfsc_final_report `
  --output-json outputs/ps_dfsc/final_report.json `
  --output-markdown PS_DFSC_RESULTS.md
```

The report permits a confirmatory improvement claim only when:

- all 150 paired exact-MILP cases are available;
- the stratified paired one-sided 95% cost bound is below zero;
- all three confirmation safety gates pass.

The 1–3% mean-cost and at-least-5% CVaR improvements remain engineering
targets, not promised outcomes.

## Verification completed

- New PS-DFSC unit/integration tests: 11 passed.
- Entire repository: 125 passed using a workspace-local pytest temp directory.
- Exact SUC planned/realized identity consistency verified.
- Full fixed-commitment DC QP produces finite nonzero gradients.
- A separate three-hour QP finite-difference check matches autograd.
- Two-day end-to-end smoke completed for training, calibration, fallback gate,
  exact planned/realized/oracle evaluation, candidate fallback and lock
  verification.

The full 3 outer x 3 base-seed retraining and 150-day confirmation experiment
has not been run as part of implementation because it is a multi-hour/day
research experiment. No cost or CVaR improvement is claimed yet.
