# Bayes'd Misfits — MindRL Challenge 2026

Cognitive modeling for the [MindRL Hub Modeling Challenge](https://rldmjc.github.io/posts/modelingchallenge/):
4-arm drifting (restless) bandit, modeled with [HSSM](https://github.com/lnccbrown/HSSM) / `ssms.rl`.

## Submission model

The submitted agent is a **Dual-Alpha Rescorla-Wagner learner with choice
stickiness**, fit offline via hierarchical MCMC. The key files at the repo
root form the MindRL submission contract:

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
| `bayesd_misfits/model.py` | RW learners: single-alpha, dual-alpha, sticky |
| `bayesd_misfits/hgf.py` | Hierarchical Gaussian Filter learners |
| `bayesd_misfits/data.py` | JSONL → HSSM-ready DataFrame import functions |

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
├── config.yaml                     # fitted hyperparameters: alpha_pos, alpha_neg, sticky, beta
├── submission.yaml                 # team + submission metadata
├── requirements.txt                # evaluator pip deps
├── interpretation_card.md          # scientific write-up
├── LICENSE_OR_POLICY_NOTICE.md     # IP notice
├── pyproject.toml                  # uv project — HSSM, ssms.rl, Python 3.12
├── uv.lock                         # pinned dependency tree
│
├── bayesd_misfits/                 # Python package (offline fitting code)
│   ├── __init__.py
│   ├── model.py                    # RW learners (single, dual-alpha, sticky)
│   ├── hgf.py                      # HGF learners (base, drift, sticky)
│   └── data.py                     # data loading + HSSM validation
│
├── tests/
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
Offline (once):                      Submission (evaluator runs):
research/run_comparison.py           agent.py
  HSSM MCMC fit                        Agent(config) reads config.yaml
  → fitted alpha_pos/neg/sticky/beta   reset(context)  → sets actions + Q=0.5
  → written into config.yaml           predict(history) → softmax(Q + sticky) → probs
                                       update(a, r)     → Q[a] += alpha*(r - Q[a])
```

The fitted parameters are the bridge: the slow Bayesian fit produces numbers;
those numbers go into `config.yaml`; the fast `Agent` reads them and plays the
model forward. Nothing in `agent.py` does inference — it only uses the result.