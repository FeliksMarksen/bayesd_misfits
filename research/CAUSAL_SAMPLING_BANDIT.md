# Causal sampling bandit — BMS-inspired process model

This is the implementation of the "causal models" direction: a Bayesian
Mutation Sampler (Kolvoort, Temme & van Maanen, 2023) adapted to the MindRL
4-arm drifting bandit.

## What was built

Two modules, both tested:

| File | What it is |
|------|-----------|
| `bayesd_misfits/bms.py` | Clean Python/JAX port of the BMS (3-variable causal networks). Reproduces the paper's Figure 2 / Table 1. |
| `bayesd_misfits/causal_sampling_bandit.py` | The bandit adaptation: a mutation sampler over discretized arm values, warm-started, with a Beta prior. |

Plus:

- `tests/test_bms.py`, `tests/test_causal_sampling_bandit.py` (24 tests)
- `research/reproduce_bms_figure2.py` (reproduces Table 1 + Figure 2 means)

## The model

The agent represents a hypothesis about the *latent causes* of reward — the
four arm values — as a discretized level per arm (default 5 levels on
[0.1, 0.9] for rewards normalized to [0, 1]). It does **not** compute a point
estimate. Each trial it runs a Metropolis-Hastings "mutation" chain:

1. **Warm start** — the chain begins at the previous trial's final state.
2. **Mutation** — propose moving *one arm* up or down *one level*.
3. **Accept/reject** — Metropolis-Hastings with a Gaussian likelihood
   `P(history | levels) = ∏ N(reward_t | value(level, arm_t), σ²)`.
4. **Limited N** — only `n_samples` steps (the resource budget).
5. **Beta prior** — the "which arm is best" posterior is regularized:
   `logit_i = log(count_i + beta)`, pulling choice toward uniform.

Mechanism → bias mapping (the point of the model):

| Mechanism | Bias |
|-----------|------|
| warm start | perseveration |
| limited N | anchoring / slow drift tracking |
| mutation | small incremental hypothesis changes |
| Beta prior | conservatism / bias toward indifferent choice |

## Verification against the paper

`research/reproduce_bms_figure2.py` reproduces the paper's Table 1 (Mutation
Sampler, `beta=0`) for the query `P(X1=1 | Y=1, X2=0)` on the common-cause
structure. With 20,000 chains the breakdown matches the paper within Monte
Carlo noise:

| chain len | computed | 50% | 100% |
|----------:|---------:|----:|-----:|
| 2  | 0.00 (paper 0.00) | 0.93 (0.93) | 0.06 (0.06) |
| 12 | 0.18 (0.18) | 0.39 (0.39) | 0.34 (0.33) |
| 48 | 0.74 (0.73) | 0.02 (0.02) | 0.22 (0.22) |

The Figure 2 means also show the expected pattern: the MS mean is biased away
from the normative answer, and the BMS (`beta=1`) mean is pulled toward 0.5
(conservatism).

## Fitting path (important caveat)

`beta` (the Beta-prior strength) enters the logits smoothly and is
differentiable, so it can be fit with HSSM gradient-based inference. The
mutation chain itself depends on `obs_sigma` and `n_samples` only through
discrete accept/reject steps, so those are **hyperparameters** (constructor
arguments) to be selected by grid search / PDA — exactly as the BMS paper fits
its discrete chain length by grid search.

The `NArmCausalSamplingBandit` HSSM adapter therefore exposes only `beta` as a
free parameter and runs the chain once per trial inside `update_jax`, storing
the best-arm counts in the state so `compute_jax` only applies the
differentiable `log(count + beta)` mapping. This mirrors the `last_choice`
pattern in the RW learners and keeps the chain's discrete steps out of the
gradient path.

## How to run

```bash
# BMS reference (Table 1 + Figure 2 means)
.venv/bin/python research/reproduce_bms_figure2.py --n-chains 20000

# Hyperparameter grid search (forward model, no MCMC)
.venv/bin/python research/grid_search_causal_sampling.py --n-valid 20

# Tests
.venv/bin/python -m pytest tests/test_bms.py tests/test_causal_sampling_bandit.py -q
```

## Grid-search result (forward model, 20 held-out trajectories)

The forward model was swept over `n_samples x obs_sigma x beta` on a small
subject-disjoint split.  Best operating point:

| n_samples | obs_sigma | beta | NLL | accuracy |
|--:|--:|--:|--:|--:|
| 20 | 0.1 | 5.0 | 1.10 | 0.55 |

Uniform random is NLL 1.386.  The model learns (accuracy ~0.55 vs 0.25
chance) but is well below the submitted RW+sticky (NLL 0.69, accuracy 0.76).
The sweep is fairly flat (NLL 1.10-1.18 across the grid), which points to the
coarse 5-level discretization -- not the hyperparameters -- as the main
bottleneck.  This is the expected outcome for a process model: it trades
predictive accuracy for a mechanistic account of perseveration and
conservatism.

## run_comparison.py integration

`NArmCausalSamplingBandit` is registered in `research/run_comparison.py` as
`CausalSamplingBandit`, with `prior_beta` as the only free parameter and the
`categorical_logits` decision process.  The chain hyperparameters are
configurable via environment variables:

```bash
CSB_N_SAMPLES=20 CSB_OBS_SIGMA=0.1 \
  .venv/bin/python research/run_comparison.py
```

## Honest expectations

This is a **process-model contribution, not a leaderboard bet**. The BMS's
signature phenomena (conservatism, extreme avoidance) are about a 0–100
judgment scale, and the bandit records discrete choices, so the distinctive
distributional predictions are only partially visible. What *does* transfer is
the mechanistic account of perseveration (warm start + limited N) and
conservatism (Beta prior), which the current RW+sticky model captures only as
free parameters. Expect it to roughly match — not clearly beat — the
`CausalScaleSticky` baseline on held-out NLL, while explaining *why* the
perseveration is there.

## Next steps

1. (done) Grid-search `n_samples` and `obs_sigma` on a small split.
2. (done) Wire `NArmCausalSamplingBandit` into `research/run_comparison.py`.
3. Run the full HSSM fit of `prior_beta` (with the grid-search operating
   point) and compare against `CausalScaleSticky`.
4. Parameter/model recovery before interpreting `prior_beta` psychologically.
5. (optional) Finer value discretization (more levels) and/or a scale-free
   version using running reward standardization, to close the gap to RW.
