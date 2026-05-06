# Theory validation run

- Max RR privacy ratio: `2.718282`; exp(epsilon): `2.718282`.
- Privacy-ratio check passed: `True`.
- Unclipped inverse max absolute Monte Carlo bias: `0.009530` over `40` repetitions.
- Bootstrap minimum marginal coverage: `0.875` over `8` repetitions.
- Epsilon/k grid max privacy-ratio absolute error: `1.776e-15`.
- Epsilon/k grid max inverse-recovery L1 error: `6.384e-16`.
- Clipped+renormalised estimator max bias in the finite-sample check: `0.053040`.

The unclipped inverse is the theoretical unbiased estimator. Production reports clip and renormalise finite-sample estimates, so clipped outputs trade exact unbiasedness for valid probability vectors. The grid checks validate this relationship over multiple epsilon and category-count settings.
