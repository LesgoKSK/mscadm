# MS-CADM full reproduction track

The authoritative implementation is under `repro/`, with entry points under
`repro_scripts/` and explicit configurations under `repro_configs/`. The older
`mscadm/` and `scripts/` files are retained only as the initial prototype.

## Reproduction protocol

- Raw GEFCom2014 wind files are read directly from `Data/`.
- Missing targets are forward-filled exactly as in the public Dumas reference.
- Ten derived/raw NWP features and ten zone indicators are used.
- The Dumas random-day split is reproduced exactly: per zone 631 learning,
  50 validation, and 50 test days (`random_state=0`).
- Targets and inputs are standardized using learning-set statistics only.
- Every evaluated forecast has 100 24-hour scenarios and is clipped to [0, 1]
  after inverse standardization.
- Six paper metrics are computed: MAE, RMSE, CRPS, QS, ES, and VS. Interval
  coverage, ACE, and PIAW are exported separately.

## Paper-disclosed and reconstructed choices

The paper discloses Adam, learning rate 1e-4, batch size 256, 26,000 updates,
100 scenarios, AdaLN, multi-scale condition embedding, learned variance, and
random condition masking. It does not disclose Transformer width/depth, head
count, convolution widths, masking probability, diffusion schedule, learned
variance interpolation, VLB weight, QRGBM hyperparameters, or the procedure for
coupling hourly QRGBM quantiles. Those choices are therefore explicit in
`repro_configs/paper.json` and are reconstruction assumptions, not claimed
author code.

RAND has two versions: `rand` exactly reproduces the reference behavior by
sampling observations from the evaluation split itself; `rand_train` samples
only from the learning split and is the leakage-free control.

## Commands

Smoke train one model:

```powershell
python -m repro_scripts.train --config repro_configs/smoke.json --model mscadm
```

Formal training uses `repro_configs/paper.json`; replace `mscadm` with `ddpm`,
`vae`, `wgan`, or `nf`. Generate scenarios with `repro_scripts.generate`, build
classical baselines with `repro_scripts.classical`, and aggregate metrics with
`repro_scripts.evaluate`.

## Reproduction status

Implemented and tested: exact data protocol, MS-CADM with four ablation
switches, learned variance/VLB, ancestral and respaced DDIM sampling, VAE,
WGAN-GP, conditional NF, conditional DDPM, RAND, QRGBM+ECC, scenario archives,
and all forecast metrics.

Still required before claiming a complete numerical reproduction: formal model
training, Tables 1-4, Figures 4-7, and the IEEE RTS-24 stochastic unit commitment
experiment for Table 5.
