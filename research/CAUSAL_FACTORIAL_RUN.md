# Causal-scale factorial model comparison

This comparison answers three separate questions without changing the causal,
scale-free reward handling used by the submitted agent:

1. Does a second learning rate for negative prediction errors improve
   out-of-subject prediction?
2. Does choice history help, and is a gradual trace better than only the
   immediately previous choice?
3. Do random effects for real human subjects improve population predictions
   enough to justify a hierarchical model?

## Correct grouping

The two identifiers now have separate jobs:

- `participant_id` is HSSM's required sequence key. It contains one remapped
  trajectory ID, so Q values, running reward statistics, and choice history
  reset for each 120-trial task trajectory.
- `subject_id` is the actual human identity. Subject-hierarchical models use
  `(1|subject_id)` and the train/validation split is disjoint on this field.

Never switch the loader to `group_by="subject"` as a shortcut. In the current
loader that concatenates a person's task trajectories and would carry RL state
across different reward schedules.

## Model matrix

The core suite fits 12 models: every combination of

- pooled or real-subject hierarchical parameters;
- one learning rate or separate positive/negative rates;
- no choice-history term, an immediate previous-choice term, or a gradual
  choice trace.

The gradual choice trace maintains a decaying trace for each action. A fitted
`choice_trace_rate` of 1 is exactly the immediate-choice model; smaller values
retain a longer choice history. Its logit contribution stays separate from
`beta`, so reward sensitivity and perseveration are not conflated.

This design follows the gradual-perseveration model studied by Sugawara and
Katahira, who showed that choice autocorrelation and asymmetric learning can
be mistaken for one another when either process is omitted:

- Sugawara, M., & Katahira, K. (2021). *Dissociation between asymmetric value
  updating and perseverance in human reinforcement learning*.
  <https://doi.org/10.1038/s41598-020-80593-7>
- Senta, Bishop, and Collins (2025) also use a multi-choice kernel and note that omitted
  perseveration can be allocated incorrectly to asymmetric updating:
  <https://doi.org/10.1371/journal.pcbi.1012872>

The existing `Resource` model is available only in the `extended` suite. It is
an exploratory fatigue heuristic, not a faithful implementation of the
sampling model in Bruckner et al. (2025), and it still uses the older
fixed-scale fitting architecture. It is not eligible for deployment without a
separate causal-scale rewrite:
<https://doi.org/10.1037/rev0000526>.

## Production run

From the repository root:

```bash
bash research/run_causal_factorial_server.sh core
```

Defaults:

- 300 training trajectories and 300 held-out trajectories;
- four chains, 1,000 tuning steps, and 1,000 retained draws;
- 500 posterior draws for held-out scoring;
- NumPyro and four virtual JAX CPU devices;
- incremental JSON after every model;
- one log per model;
- completed models skipped when the same command is restarted.

Run only the faster pooled models first:

```bash
bash research/run_causal_factorial_server.sh pooled
```

Resume or add the subject models later using the same `RUN_TAG`:

```bash
RUN_TAG=causal_factorial_n300 bash research/run_causal_factorial_server.sh subject
```

Run the core matrix plus the exploratory Resource and selected pyHGF models:

```bash
bash research/run_causal_factorial_server.sh extended
```

Posterior files are disabled by default because 12 hierarchical NetCDF files
can be large. Enable them if disk space permits:

```bash
SAVE_POSTERIORS=1 bash research/run_causal_factorial_server.sh core
```

Override compute settings normally, for example:

```bash
N_TRAIN=100 N_VALID=100 N_TUNE=500 N_DRAWS=500 \
  bash research/run_causal_factorial_server.sh pooled
```

Do not combine results created with different sampling or data-size settings
in one output file. The runner detects incompatible metadata and stops.

## Reading the result

Use only models with `eligible: true`. Among eligible models compare
`heldout_nll_mean` on the identical validation split. Interpret the factorial
contrasts before selecting a winner:

- Single versus dual isolates the value of asymmetric updating.
- No-history versus immediate isolates one-step perseveration.
- Immediate versus trace tests whether deeper choice history is useful.
- Pooled versus subject tests whether a hierarchy helps after grouping by the
  real human rather than by trajectory.

A slightly lower NLL from a non-converged model is not sufficient evidence to
deploy it.

Before giving the trace rate a strong psychological interpretation, run
parameter- and model-recovery checks. The factorial run establishes predictive
value and sampling behavior; it does not by itself prove that gradual choice
history is uniquely identifiable from asymmetric learning in this dataset.
