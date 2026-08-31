# Baseline fidelity findings

The formal RAND reproduction and the Dumas-behavior VAE reproduction nearly
match the article's Table 1 at all six metrics. This is stronger evidence than
architecture-name matching alone.

## RAND

Formal reproduction: `0.2587, 0.3008, 0.1687, 0.0852, 0.9593, 23.17`.
Article: `0.2583, 0.3012, 0.1692, 0.0855, 0.9615, 23.21`.

This confirms that the article's RAND is consistent with sampling the very test
observations being evaluated, as done in the public Dumas reference. The
`rand_train` archive is retained as the leakage-free scientific control.

## VAE

Standard 26,000-update VAE reproduction:
`0.1364, 0.1861, 0.1245, 0.0626, 0.7631, 25.70`.

Dumas-behavior VAE reproduction:
`0.1236, 0.1670, 0.0877, 0.0443, 0.5456, 17.72`.

Article: `0.1244, 0.1677, 0.0880, 0.0445, 0.5482, 17.87`.

The matching version preserves the public code's nonstandard
`z = mean + exp(log_sigma) * noise` behavior and its 200-epoch/2,000-update
schedule. This is evidence that “identical experimental conditions” in Section
4.2 should not be interpreted as every baseline receiving the stated 26,000
updates. Both variants are retained so this conclusion remains auditable.

The same strategy is applied to WGAN: `wgan_reference` follows the public
300-epoch/3,000-generator-update schedule, while `wgan` retains the explicit
26,000-update control.
