# PyHGF uncertainty-choice diagnostic

This diagnostic compares official-pyhgf learners on one fixed MindRL split:

- 20 complete training trajectories (2,400 trials)
- 60 held-out trajectories from disjoint human subjects (7,200 trials)
- 2 NumPyro chains
- 500 warm-up and 500 retained draws per chain
- 200 population-posterior draws for held-out scoring

The uncertainty-aware choice score is:

```text
beta * predicted_mean
+ uncertainty_weight * predicted_standard_deviation
+ repetition_weight * previous_choice_indicator
```

The last term is present only in the combined model. Each coefficient enters
the categorical logit independently.

## Results

| Model | Held-out NLL | Eligible | Max R-hat | Min ESS |
|---|---:|:---:|---:|---:|
| Sticky dual-alpha RW | 0.5526 | No (borderline) | 1.0105 | 218 |
| PyHGF + uncertainty + repetition | 0.6710 | Yes | 1.0023 | 365 |
| PyHGF + uncertainty | 0.7548 | Yes | 1.0076 | 231 |
| PyHGF + coupled sticky term | 0.7787 | No | 1.0212 | 188 |
| PyHGF value-only | 1.3921 | Yes | 1.0040 | 830 |
| Uniform random | 1.3863 | n/a | n/a | n/a |

Lower NLL is better. The complete machine-readable results are in
`research/incremental_model_comparison_results.json`.

"Eligible" uses the project's declared gate over population intercepts and
random-effect scales. NumPyro still warned that some participant-level offsets
had R-hat above 1.01 in the combined fit, so the complete posterior should not
yet be treated as fully converged at the individual level.

## Population estimates

For PyHGF + uncertainty:

```text
ghgf_omega          -1.442  (94% interval -1.998 to -0.947)
beta                 0.161  (94% interval  0.095 to  0.245)
uncertainty_weight  -2.639  (94% interval -3.411 to -2.118)
```

For PyHGF + uncertainty + repetition:

```text
ghgf_omega          -0.998  (94% interval -1.540 to -0.580)
beta                 0.165  (94% interval  0.093 to  0.267)
uncertainty_weight  -0.905  (94% interval -1.176 to -0.712)
repetition_weight    1.698  (94% interval  1.481 to  1.927)
```

The negative uncertainty coefficients indicate uncertainty avoidance, not
directed exploration: holding expected reward and repetition constant, a more
uncertain arm is less likely to be selected. The combined model retains both a
negative uncertainty effect and a positive repetition effect.

The very small beta estimates show that expected HGF reward means contribute
little to choice in these fits. The combined model improves prediction mainly
through certainty preference and repetition. Because an arm becomes more
certain when it is sampled, those signals are related even though their
coefficients are separately parameterized.

## Interpretation and next gate

The uncertainty-aware response model materially improves PyHGF and fixes the
previous beta/stickiness product confound. It does not beat the matched sticky
dual-alpha RW model. These are diagnostic, two-chain fits and not production
estimates.

Before interpreting a 100/300 production run, simulate behavior with known
uncertainty and repetition coefficients and confirm parameter and model
recovery. A four-chain combined-model production run was scheduled after the
recovery experiment below passed its model-recovery checks.

## Recovery follow-up

The recovery workflow in `research/recover_pyhgf_choice.py` simulated 12
training and 12 validation agents per case against real complete reward
schedules. It compared population maximum-likelihood fits of value-only,
uncertainty, and uncertainty-plus-repetition candidates.

The value-only case generated `omega=-3` and `beta=4`. The value-only fit
recovered `omega=-2.90` and `beta=4.41`. Adding uncertainty changed validation
NLL by less than 0.0002 and estimated an uncertainty effect near zero; BIC
favoured the simpler value-only model.

The uncertainty-averse case generated `omega=-2`, `beta=3`, and uncertainty
weight `-1.5`. The uncertainty model recovered `omega=-1.71`, `beta=2.90`, and
uncertainty weight `-1.55`. It clearly beat value-only validation NLL (1.206 vs
1.257), while the combined model estimated repetition near zero.

The diagnostic-like combined case generated `omega=-1`, `beta=0.2`, uncertainty
weight `-0.9`, and repetition weight `1.7`. The combined fit was correctly
selected with validation NLL 0.701, versus 0.867 for uncertainty-only and 1.386
for value-only. It recovered `omega=-0.54`, `beta=0.42`, uncertainty weight
`-0.73`, and repetition weight `1.67`.

This supports model recovery and the signs of the choice effects. It also shows
that beta and volatility are only approximately recoverable in the low-beta
regime. The full machine-readable output is in
`research/pyhgf_uncertainty_recovery_results.json`.
