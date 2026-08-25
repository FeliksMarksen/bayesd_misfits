# MindRL Model Comparison — Colab GPU Runner Usage Guide

This document explains how to run the `bayesd_misfits` model comparison on a
Google Colab T4 GPU using the `google-colab-cli`. It covers both the
**one-shot full comparison** and the recommended **per-model incremental**
workflow, which is safer on free-tier Colab.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick reference](#quick-reference)
3. [Why the per-model workflow is recommended](#why-the-per-model-workflow-is-recommended)
4. [One-shot full comparison (fragile)](#one-shot-full-comparison-fragile)
5. [Recommended: one model at a time](#recommended-one-model-at-a-time)
   - [Step 1 — start a model](#step-1--start-a-model)
   - [Step 2 — poll and download](#step-2--poll-and-download)
   - [Step 3 — repeat](#step-3--repeat)
6. [Choosing the best model](#choosing-the-best-model)
7. [Troubleshooting](#troubleshooting)
8. [Files in this folder](#files-in-this-folder)

---

## Prerequisites

- `uv` and `google-colab-cli` installed locally.
- Authenticated with Google Colab:
  ```bash
  colab version
  colab sessions
  ```
- Run all commands from the repo root:
  ```bash
  cd /path/to/bayesd_misfits
  ```

---

## Quick reference

| Task | Command |
|------|---------|
| See remaining models | `bash research/colab_run_all.sh` |
| Start a single model in background | `bash research/colab_run_single.sh <MODEL>` |
| Poll / download / stop VM | `bash research/colab_poll_single.sh <MODEL>` |
| Full one-shot run (fragile) | `bash research/colab_run.sh` |
| Quick smoke test (small settings) | `N_CHAINS=2 N_TUNE=100 N_DRAWS=100 N_TRAIN=10 N_VALID=30 bash research/colab_run_single.sh RW` |

---

## Why the per-model workflow is recommended

Free-tier Colab has dynamic limits and can interrupt long GPU sessions. The
original `colab_run.sh` runs all 10 models in one session. If that session
terminates early, all progress is lost.

The per-model scripts fix this:

- Each model gets its own fresh Colab session.
- The model runs as a **detached background job** inside the VM, so you can
  close your laptop.
- Results are appended to `research/incremental_model_comparison_results.json`
  on the VM and downloaded as `results_colab_incremental.json`.
- If a model fails, you only lose that model, and `colab_run_all.sh` will
  skip completed models when you resume.

---

## One-shot full comparison (fragile)

If you are on a paid Colab plan with guaranteed runtime, or just want to try
one command:

```bash
bash research/colab_run.sh
```

This:
1. Creates/reuses a `mindrl` Colab session.
2. Installs dependencies and fixes version conflicts.
3. Starts the full comparison as a detached VM background job.
4. Prints a message and returns to your shell.

You can then poll with:

```bash
bash research/colab_poll.sh
```

On free-tier Colab, the long runtime may be interrupted before completion.

---

## Recommended: one model at a time

### Step 1 — start a model

```bash
bash research/colab_run_single.sh RW
```

This command:
- uploads the local code to the Colab VM,
- installs HSSM and reconciles dependency versions (`jax[cuda12]`,
  `numba>=0.61`, `numpyro>=0.21`),
- starts `research/run_single_model.py RW` as a **detached background process**,
- prints a message and returns to your shell within 1–2 minutes.

You can now **close your laptop**.

### Step 2 — poll and download

After ~30–60 minutes, check whether the model finished:

```bash
bash research/colab_poll_single.sh RW
```

This:
- prints the tail of the remote log,
- downloads `research/incremental_model_comparison_results.json` from the VM
  to `./results_colab_incremental.json`,
- stops the VM to save free GPU quota.

If results are not ready yet, the script will keep polling every 2 minutes.
Press `Ctrl+C` to stop polling without stopping the VM.

### Step 3 — repeat

Once a model is done, start the next one:

```bash
bash research/colab_run_single.sh RW+Decay
# ...later...
bash research/colab_poll_single.sh RW+Decay
```

To see which models are still pending:

```bash
bash research/colab_run_all.sh
```

This prints a list with the exact `colab_run_single.sh` and
`colab_poll_single.sh` commands for each remaining model.

---

## Choosing the best model

After a model finishes, open `results_colab_incremental.json`. For each
model you care about:

1. **Check eligibility** — `status` must be `"ok"` and `eligible` must be
   `true`. This means the sampler diagnostics passed:
   - R-hat ≤ 1.01,
   - bulk ESS above threshold,
   - BFMI ≥ 0.30,
   - zero divergences.

2. **Compare held-out NLL** — among eligible models, the one with the
   **lowest `heldout_nll_mean`** is preferred. The script already prints a
   summary table in the log:

   ```text
   Model                  Mean        SE    Diagnostics
   RW                  0.xxxx    0.xxxx        PASS
   DualAlpha           0.xxxx    0.xxxx        PASS
   ...
   ```

Models with `status: "fit_failed"` cannot be selected.

---

## Troubleshooting

### `TooManyAssignmentsError`

You already have an active Colab session. Stop it:

```bash
colab stop -s mindrl
```

### `ImportError: cannot import name 'xla_pmap_p'`

This means `numpyro` is too old for the installed JAX. The scripts already
upgrade to `numpyro>=0.21`; if you see this error, make sure you are using the
latest version of `research/colab_run_single.sh`.

### Session dies overnight / no results

Free-tier Colab may reclaim VMs after prolonged GPU use. This is expected.
Just re-run the start command for the model that did not finish; the
incremental results file lets you resume without re-running completed models.

### Check VM status quickly

```bash
colab sessions
colab status -s mindrl
```

### Read the remote log manually

```bash
colab exec -s mindrl --timeout 30 << 'PYEOF'
with open("/root/run_single_RW.log") as f:
    print(f.read()[-2000:])
PYEOF
```

Replace `RW` with the model name you are checking.

---

## Files in this folder

| File | Purpose |
|------|---------|
| `colab_run.sh` | One-shot full comparison (detached background job) |
| `colab_poll.sh` | Poll the one-shot run, download results, stop VM |
| `colab_run_single.sh` | Start one model as a detached background job |
| `colab_poll_single.sh` | Poll one model, download results, stop VM |
| `colab_run_all.sh` | Print remaining models and exact commands |
| `run_single_model.py` | Python helper: fit one model and append incremental results |
| `COLAB_USAGE.md` | This guide |
