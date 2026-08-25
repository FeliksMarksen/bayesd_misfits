# Causal diagnostic baselines

This analysis separates reward learning from simpler sources of predictability.
It uses the same subject-disjoint split as the factorial comparison but fits
deterministic pooled parameters, then freezes them for held-out evaluation.

The suite contains:

- uniform choice;
- fixed action-label frequencies learned from training subjects;
- causal within-trajectory action frequencies;
- previous-choice repetition without rewards;
- a gradual choice trace without rewards;
- causal-scale single-alpha RW without choice history;
- the same RW model with immediate repetition;
- the same RW model with a gradual choice trace.

All dynamic baselines reset at trajectory boundaries. During held-out scoring,
their state uses only choices and rewards already revealed in that trajectory.
No validation outcomes, future trials, or participant-wide evaluation data are
used to fit parameters.

Run a quick smoke analysis:

```bash
N_TRAIN=20 N_VALID=20 DIAGNOSTIC_RESTARTS=2 \
  uv run python research/run_diagnostic_baselines.py
```

Run the 300/300 comparison:

```bash
N_TRAIN=300 N_VALID=300 DIAGNOSTIC_RESTARTS=5 \
DIAGNOSTIC_OUTPUT_PATH=research/diagnostic_baselines_n300_results.json \
  uv run python research/run_diagnostic_baselines.py
```

The result file includes per-trajectory held-out NLLs and paired cluster
bootstrap contrasts. The bootstrap resamples complete held-out human subjects,
not individual trajectories, so repeated trajectories from one person are not
treated as independent. Positive `mean_nll_improvement` means the candidate
outperformed its named reference on the same trajectories.
