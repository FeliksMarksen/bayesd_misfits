# Bayes'd Misfits — MindRL Challenge 2026

Cognitive modeling for the [MindRL Hub Modeling Challenge](https://rldmjc.github.io/posts/modelingchallenge/):
4-arm drifting (restless) bandit, modeled with [HSSM](https://github.com/lnccbrown/HSSM) / `ssms.rl`.

## Quick start

```bash
# 1. Clone
git clone git@github.com:FeliksMarksen/bayesd_misfits.git
cd bayesd_misfits

# 2. Create environment (Python 3.12 + HSSM stack)
uv sync

# 3. Run the notebook
jupyter lab notebooks/01_data_loading.ipynb
```

Select the **"Bayes'd Misfits (Python 3.12)"** kernel in Jupyter / VS Code.

## Project structure

```
bayesd_misfits/
├── pyproject.toml              # uv project — HSSM 0.4.0, ssms 0.13.2, Python 3.12
├── uv.lock                     # pinned dependency tree (committed)
├── scripts/
│   └── download_data.sh        # fetches the dataset from HuggingFace
├── bayesd_misfits/
│   ├── __init__.py
│   ├── data.py                 # JSONL → HSSM-ready DataFrame import functions
│   └── main.py                 # quick-start entry point
├── notebooks/
│   └── 01_data_loading.ipynb   # data loading (no visualization)
└── hf_cache/                   # downloaded data (gitignored)
    └── public/
        ├── public_train.jsonl
        ├── public_train_reward_schedules.jsonl
        ├── schema.json
        └── task_description.md
```

## Data

The dataset is **not committed** — the notebook downloads it automatically on
first run via `ensure_data_downloaded()` (using `huggingface_hub`, which comes
with `uv sync`). Data lands in `hf_cache/public/` (gitignored).

To download from the command line instead:

```bash
./scripts/download_data.sh
```