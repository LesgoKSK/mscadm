# PS-DFSC publication runbook

This runbook supersedes older `v1`/`v2` helper entry points for the formal
experiment.

## Canonical entry points

- Base MM-JDWind training/generation: `repro_scripts.run_ps_dfsc_base`
- Audited identity preparation: `repro_scripts.prepare_ps_dfsc_publication`
- Candidate training/calibration/safety/exact validation:
  `repro_scripts.run_ps_dfsc_canonical_v4`
- Locked confirmation:
  `repro_scripts.run_ps_dfsc_confirmation_canonical_v3`

The canonical exact wrapper reports both the raw HiGHS gap in the termination
string and the gap recomputed from the reported operating-cost incumbent and
the constant-adjusted dual bound. A time-limit incumbent remains a valid binary
solution, but its solver-success flag is false and it must not be described as
optimal.

## Frozen data order

1. Train the 3 seeds for each of the 3 outer splits.
2. Generate only `development_train` and `development_validation`.
3. Pool 34/33/33 members into 100-member outer distributions.
4. Prepare train and validation identity archives with the audited preparer.
5. Train beta candidates `0`, `0.25`, `0.5`, and `1`.
6. Calibrate development validation, run the paired-day publication safety
   gate, then exact-evaluate at most the three proxy-shortlisted safe candidates.
7. Select a candidate or lock the whole outer to identity fallback.
8. Create and verify the lock manifest.
9. Only after step 8 may the locked confirmation runner generate
   `confirmation_test`.

No command before step 8 may use the `confirmation_test` role.

## Audited identity preparation

Example for outer 1 training:

```powershell
python -m repro_scripts.prepare_ps_dfsc_publication `
  --input outputs/ps_dfsc/base/outer1/pooled/development_train_M100.npz `
  --mapping repro_configs/ps_dfsc_mapping_outer1.json `
  --clusters 20 --mip-gap 0.001 --time-limit 600 `
  --cache-dir outputs/ps_dfsc/outer1/baseline_cache/train `
  --training-output outputs/ps_dfsc/outer1/development_train_prepared.npz `
  --identity-output outputs/ps_dfsc/outer1/identity_train.npz
```

Each daily cache records input/day/mapping hashes, incumbent cost, adjusted
dual bound, actual gap, node count, runtime, solver-success flag, termination
message, and cache origin. Legacy caches are reused only when their saved gap
already certifies the requested threshold.

## Candidate commands

```powershell
python -m repro_scripts.run_ps_dfsc_canonical_v4 train `
  --input outputs/ps_dfsc/outer1/development_train_prepared.npz `
  --mapping repro_configs/ps_dfsc_mapping_outer1.json `
  --clusters 20 --beta 0.25 --epochs 50 --warmup-epochs 20 `
  --strong-convexity 1e-4 --mip-gap 0.001 --time-limit 600 `
  --device cuda `
  --output outputs/ps_dfsc/outer1/candidates/beta025.pt

python -m repro_scripts.run_ps_dfsc_canonical_v4 calibrate `
  --input outputs/ps_dfsc/base/outer1/pooled/development_validation_M100.npz `
  --scenario-key scenarios `
  --checkpoint outputs/ps_dfsc/outer1/candidates/beta025.pt `
  --mapping repro_configs/ps_dfsc_mapping_outer1.json `
  --clusters 20 --device cuda `
  --output outputs/ps_dfsc/outer1/candidates/beta025_validation.npz

python -m repro_scripts.run_ps_dfsc_canonical_v4 safety-gate `
  --candidate outputs/ps_dfsc/outer1/candidates/beta025_validation.npz `
  --baseline outputs/ps_dfsc/outer1/identity_validation.npz `
  --truth outputs/ps_dfsc/base/outer1/pooled/development_validation_M100.npz `
  --regime-reference outputs/ps_dfsc/outer1/development_train_prepared.npz `
  --bootstrap-samples 10000 --seed 10001 `
  --output outputs/ps_dfsc/outer1/candidates/beta025_gate.json
```

Calibration stores the original calendar dates and the fixed K-means
assignments fitted on untransported base scenarios. The safety gate uses
paired-day resampling, a 2.5%/97.5% interval for coverage, and recomputes
equal-group conditional ACE inside every bootstrap replicate.

## Locked identity fallback

If validation selects identity fallback, use candidate ID `identity`, include
the three frozen base checkpoints in the lock, and omit `--checkpoint` from the
locked confirmation command. The confirmation archive is then an exact copy of
the pooled base distribution with uniform probabilities, zero transport,
`used_fallback=True`, and reason `outer_identity_fallback`.

For a selected non-identity candidate, `--checkpoint` is required and the
resolved checkpoint path must be present in the lock manifest. Any mismatch
fails before confirmation data generation.
