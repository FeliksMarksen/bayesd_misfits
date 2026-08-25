# Bayes'd Misfits — MindRL Challenge 2026

Cognitive modeling for the [MindRL Hub Modeling Challenge](https://rldmjc.github.io/posts/modelingchallenge/):
4-arm drifting (restless) bandit, modeled with [HSSM](https://github.com/lnccbrown/HSSM) / `ssms.rl`.

## Submission model

The revised submitted agent is a **scale-free Dual-Alpha Rescorla-Wagner
learner with an independent repetition effect**, fit offline via hierarchical
MCMC. Raw action values are standardized for choice using the running mean and
sample standard deviation of rewards already revealed in the current
trajectory. This removes the previous fixed 1-100 reward-scale assumption.
The key files at the repo root form the MindRL submission contract:

| File | Purpose |
|------|---------|
| `agent.py` | The `Agent` class the evaluator imports and calls |
| `config.yaml` | Runtime config + fitted model hyperparameters |
| `submission.yaml` | Team and submission metadata for organizers |
| `requirements.txt` | Pip dependencies for the evaluator |
| `interpretation_card.md` | Structured scientific write-up |
| `LICENSE_OR_POLICY_NOTICE.md` | Evaluation-only IP notice |
| `artifacts/` | Small local helper files (large weights → HuggingFace) |

The model's learning rules live in the Python package:

| File | Purpose |
|------|---------|
| `bayesd_misfits/causal_rw.py` | Scale-free submitted learner used for fitting |
| `bayesd_misfits/model.py` | RW learners: single-alpha, dual-alpha, sticky |
| `bayesd_misfits/hgf.py` | Hierarchical Gaussian Filter learners |
| `bayesd_misfits/data.py` | JSONL → HSSM-ready DataFrame import functions |

The offline comparison now keeps two identities separate: HSSM's
`participant_id` column is a trajectory-level sequence key so RL state resets
between tasks, while real-human random effects use `subject_id`. The parameters
currently frozen in `config.yaml` still come from the earlier fit; do not update
them until a corrected comparison has converged and passed held-out evaluation.

## Quick start

```bash
# 1. Clone
git clone git@github.com:FeliksMarksen/bayesd_misfits.git
cd bayesd_misfits

# 2. Create environment (Python 3.12 + HSSM stack)
uv sync

# 3. Download the dataset
./research/download_data.sh

# 4. Run the model comparison
.venv/bin/python research/run_comparison.py

# 5. Or explore the notebooks
jupyter lab research/notebooks/01_data_loading.ipynb
```

Select the **"Bayes'd Misfits (Python 3.12)"** kernel in Jupyter / VS Code.

## Project structure

```
bayesd_misfits/
├── agent.py                        # submission agent (the evaluator imports this)
├── config.yaml                     # alpha_pos/neg, beta, repetition + normalization declaration
├── submission.yaml                 # team + submission metadata
├── requirements.txt                # evaluator pip deps
├── interpretation_card.md          # scientific write-up
├── LICENSE_OR_POLICY_NOTICE.md     # IP notice
├── pyproject.toml                  # uv project — HSSM, ssms.rl, Python 3.12
├── uv.lock                         # pinned dependency tree
│
├── bayesd_misfits/                 # Python package (offline fitting code)
│   ├── __init__.py
│   ├── causal_rw.py                # causal scale-free submitted learner
│   ├── model.py                    # RW learners (single, dual-alpha, sticky)
│   ├── hgf.py                      # HGF learners (base, drift, sticky)
│   └── data.py                     # data loading + HSSM validation
│
├── tests/
│   ├── test_causal_rw.py           # invariance, parity, reset + gradient tests
│   └── test_hgf.py                 # HGF numpy/JAX parity + gradient tests
│
├── research/                       # offline experiments (not needed at eval time)
│   ├── run_comparison.py           # 7-model fitting + held-out NLL comparison
│   ├── download_data.sh            # fetch dataset from HuggingFace
│   ├── model_comparison_results.json  # latest comparison results
│   └── notebooks/
│       ├── 01_data_loading.ipynb
│       ├── 02_mvp_rlssm_model.ipynb
│       ├── 03_hgf_vs_rw_comparison.ipynb
│       └── 04_four_model_comparison.ipynb
│
└── artifacts/                      # submission artifacts (gitignored except README)
```

## Data

The dataset is **not committed** — download it with:

```bash
./research/download_data.sh
```

Data lands in `hf_cache/public/` (gitignored, sibling to the repo).

## How it works

```
Offline (once):                       Submission (evaluator runs):
research/run_comparison.py            agent.py
  HSSM MCMC fit                         Agent(config) reads config.yaml
  → causal running scaler               reset(context) clears all state
  → raw Dual-Alpha Q updates             predict(history) uses past rewards only
  → value + repetition logits            update(a, r) updates Q + scaler
  → fitted population parameters         no optimization during evaluation
```

The fitted parameters are the bridge: the slow Bayesian fit produces numbers;
those numbers go into `config.yaml`; the fast `Agent` reads them and plays the
model forward. Nothing in `agent.py` does inference — it only uses the result.

## Official-pyhgf prototype

`bayesd_misfits/pyhgf_bandit.py` contains a separate continuous generalized-HGF
prototype built with the official `pyhgf.model.Network` API. It creates one
independent reward/value/volatility branch per arm and uses pyhgf observation
masks so only the chosen arm receives feedback. It does not change the submitted
agent.

The module also exposes `NArmPyHGF` and the nested `NArmPyHGFSticky` as JAX
learners for HSSM. The first comparison fits only value-level tonic volatility
(`ghgf_omega`) and decision inverse temperature; the sticky variant adds one
perseveration parameter. Sensory precision, volatility coupling, and the
higher-level dynamics remain fixed until parameter-recovery tests justify a
larger model.

Two uncertainty-aware variants use predicted value standard deviations directly
in choice. Their custom categorical logits keep expected-value sensitivity,
uncertainty preference, and repetition as separate coefficients. See
`research/pyhgf_uncertainty_diagnostic.md` for the matched diagnostic results.

Run its correctness tests and replay diagnostic with:

```bash
.venv/bin/pytest -q tests/test_pyhgf_bandit.py
MPLCONFIGDIR=/tmp/matplotlib-bayesd \
  .venv/bin/python research/replay_pyhgf.py \
  --trajectory 0 --output /tmp/pyhgf_replay.png
```

## Causal-scale factorial comparison

The restartable server comparison crosses pooled versus real-subject
hierarchical fitting, single versus dual learning rates, and no choice history
versus immediate or gradual choice traces:

```bash
bash research/run_causal_factorial_server.sh core
```

See `research/CAUSAL_FACTORIAL_RUN.md` for the model matrix, research basis,
resource-model caveat, resume behavior, and compute overrides.
