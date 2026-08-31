# MS-CADM reproducibility audit

This file separates exact reproductions, explicit reconstructions, and claims
that cannot be uniquely reproduced from the article.

## High-confidence exact items

- Raw GEFCom2014 train/test predictors and targets are merged into 731 complete
  days per zone. Missing target values are forward-filled as in the Dumas code.
- The split is the Dumas two-stage shuffled split with `random_state=0`: 631
  learning, 50 validation, and 50 test days per zone.
- The Dumas feature formulas are reproduced, including degrees and the unusual
  `atan2(U, V)` argument order.
- Input/target scalers are fitted on the learning split only.
- Evaluation uses 100 24-hour scenarios and the six article metrics.
- RAND reproduces the reference's evaluation-split resampling behavior. Its
  formal scores closely match Table 1, validating the pipeline.
- RTS-24 uses the public Ordoudis 12-unit data (3375 MW), six 200 MW wind
  farms, 34 branches, and the published demand profile.

## Explicit reconstruction assumptions

- The article provides no model width, depth, head count, convolution channel
  count, convolution kernel sizes, dropout, or positional encoding. These are
  declared in `repro_configs/paper.json`.
- The diffusion beta schedule, learned-variance parameterization, VLB weight,
  and condition-mask probability are absent. The implementation uses cosine
  betas, Improved-DDPM-style variance interpolation, VLB weight 1e-3, and mask
  probability 0.1.
- The article does not state how 10/20/50/100-step sampling is obtained from a
  250-step model. The reproduction uses respaced DDIM with configurable eta.
- QRGBM tree hyperparameters and conversion from 24 marginal quantiles to
  trajectories are absent. The reproduction uses multi-quantile XGBoost plus
  same-zone residual-rank ECC.
- The exact NF transform is not disclosed. The reproduction provides a
  conditional RealNVP baseline; it is not claimed to be the author's private
  NF implementation.
- The selected single zone is not named. Zone 1 is the default and is the
  closest match to the Table-2 RAND point metrics, but this remains an
  inference.
- The number of K-Means representatives `k` is not stated. The declared default
  is 10.

## Internal inconsistencies or methodological vulnerabilities

1. Algorithm 2 samples a fresh `x_t` inside every reverse step, which breaks the
   Markov reverse chain if read literally. It also omits the reverse stochastic
   variance term despite learned variance being a claimed contribution.
2. The text motivates a KL/VLB learned-variance objective but the displayed
   simplified loss only shows noise-prediction MSE. The relative weighting is
   absent.
3. Alpha notation alternates between per-step and cumulative quantities, making
   the displayed reverse update ambiguous.
4. The Table-1 MS-CADM row exactly equals the 250-step column in Table 3 even
   though the paper concludes that 50 steps is the chosen balance.
5. The SUC paragraph says seven randomly selected days, while the next paragraph
   says Table 5 averages 100 days.
6. The PIAW equation subtracts upper from lower, producing negative width. The
   implementation uses the standard positive `upper - lower` definition.
7. RAND samples from validation/test observations being evaluated. This is
   preserved as `rand` for numerical comparability and corrected as
   `rand_train` for a leakage-free control.
8. The paper calls all-zone samples scenarios but trains on pooled zone-day
   records with a zone one-hot. It does not generate a joint ten-zone spatial
   trajectory, so multi-zone spatial dependence is not tested.
9. Equation 13 orders raw wind components differently from the inherited Dumas
   code. The reproduction follows the executable Dumas order because the paper
   explicitly says it inherits that protocol.
10. No seeds, number of repeated runs, uncertainty bars, checkpoint selection,
    code, or trained weights are supplied. Reported four-decimal rankings may
    therefore reflect single-run stochastic variation.
11. Table 4 calls CE, AdaLN, LV, and RCM “three” components although four
    ablations are listed.
12. The two SUC stages are described in a nonstandard order, and reserve rules,
    representative-scenario count, network formulation, wind-farm allocation,
    and exact test days are absent. Table 5 cannot be uniquely reconstructed.

## Claim threshold

“Implementation complete” means every disclosed experiment has an executable,
tested path. “Numerically reproduced” is reserved for a table/figure only after
formal runs have produced artifacts and the deltas against
`repro_configs/paper_reported_results.json` have been inspected.
