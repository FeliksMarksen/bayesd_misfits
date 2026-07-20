# MindRL Challenge — Submission Guide (Bayes'd Misfits)

This is the team-facing guide for turning this repo into a valid MindRL
Challenge submission. It distills the official submission template's
requirements. The submission **contract files** live at the repo root:

```
agent.py                     # Agent class — the evaluator imports this
config.yaml                  # runtime + model hyperparameters
submission.yaml              # team + submission metadata (incl. runtime_profile)
requirements.txt             # pip deps the evaluator installs
interpretation_card.md       # structured scientific write-up
LICENSE_OR_POLICY_NOTICE.md  # evaluation-only IP notice
artifacts/                   # small local helper files (large weights → HuggingFace)
```

## 1. The Agent API (the only thing the evaluator calls)

```python
class Agent:
    def __init__(self, config=None)      # once, at construction; config = parsed config.yaml
    def reset(self, context)             # once per trajectory; context has available_actions, task ids
    def predict(self, history) -> dict   # every step; returns {"action_probs": {action: prob}}
    def update(self, action, reward, info=None)  # after each trial's true outcome is revealed
```

### Hard invariants

- `predict` **always** returns `{"action_probs": {action_label: probability, ...}}`.
- Probabilities are **non-negative** and **sum to 1.0**.
- Keys are the **exact action labels** from `context["available_actions"]` (same type/identity).
- **Every** valid action is a key; no extras, no missing.
- `history` contains **only past trials** — never peek at the current step's outcome.
- `reset` is called **per trajectory**; `__init__` **once**. Initialize per-episode state in `reset`.

### Optional response-time target

RT modeling is optional and does not affect the main choice leaderboard. To
opt in, set in `submission.yaml`:

```yaml
optional_targets:
  - response_time
```

and return a positive, finite RT in milliseconds on every `predict`:

```python
return {"action_probs": {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}, "rt_ms": 715.0}
```

RT is scored separately via log-RT RMSE and MAE.

## 2. Replacing the placeholder baseline

`agent.py` currently ships a valid **Win-Stay-Lose-Shift** baseline so the
bundle passes validation. To ship the real model:

1. **Replace `agent.py`** — keep the four-method `Agent` API and the
   `{"action_probs": {...}}` return shape; swap in your model's logic
   (HGF / HSSM-fitted RL / etc.). Helper modules may sit alongside `agent.py`
   as long as imports resolve via `requirements.txt`.
2. **Edit `config.yaml`** — put your model's hyperparameters under `model.*`;
   keep `runtime` and `evaluation`.
3. **Fill `submission.yaml`** — real `commit_hash`, team member details, and an
   accurate `runtime_profile` (GPU? external API? cost estimate?).
4. **Rewrite `interpretation_card.md`** — replace the placeholder WSLS
   sections with your model's claims, mechanism mapping, discriminative
   tests, failure conditions, and evidence.
5. **Trim `requirements.txt`** — only what `agent.py` actually imports at eval
   time. Fit offline (notebooks/HSSM) and ship fitted params in `config.yaml`
   so the heavy stack isn't needed in the evaluator.

## 3. Runtime profile (mandatory in `submission.yaml`)

```yaml
runtime_profile:
  execution_type: local_python        # local_python | gpu_model | external_api | hybrid
  model_family: symbolic_cognitive     # symbolic_cognitive | neural | llm_agent | auditing | hybrid | other
  requires_gpu: false
  gpu_type: null
  requires_external_api: false
  required_secrets: []                 # e.g. ["OPENAI_API_KEY"]; never commit the keys
  estimated_eval_cost: none            # none | low | medium | high
  expected_runtime_minutes: null
  notes: "..."
```

Even GPU/API/LLM submissions must still expose a callable `Agent` class — this
block only declares routing. Never commit secrets; list secret **names** only.

## 4. Artifact / checkpoint guidance

- **Small** auxiliary files may live under `artifacts/`.
- **Large** checkpoints (HSSM traces, HGF posteriors) belong on Hugging Face:
  record the URI in `submission.yaml` (`artifacts.hf_checkpoint`) or
  `config.yaml`, and keep this git repo small.
- Never commit API keys, SSH keys, or raw identifiable participant data.

## 5. Validation

Validation uses the **core evaluator repo** (not this repo). From the core
repo root:

```bash
python scripts/validate_submission.py \
  --repo-url https://github.com/FeliksMarksen/bayesd_misfits.git \
  --commit-hash <full-40-char-sha> \
  --agent-path agent.py \
  --config-path config.yaml \
  --requirements-path requirements.txt \
  --data examples/toy_data/toy_bandit_trajectories.jsonl \
  --output validation_report.json
```

Use only toy/public data paths; never commit private evaluation sets.

## 6. Rolling submission workflow

Official submissions are self-service via the **Rolling submission** issue form
in `mindrl-challenge-core`:
https://github.com/mindrl-challenge/mindrl-challenge-core/issues/new/choose

1. Push code to this repo.
2. Pin a **full 40-character commit SHA** (do not submit branch names like `main`/`master`).
3. Open the issue form with the registry metadata.
4. The intake workflow opens/updates a registry PR that touches only
   `submissions_registry.yaml`; CI validates the pinned commit on toy
   trajectories; organizers review before merge.

## 7. Visibility & confidentiality

Default visibility in `submission.yaml` is **`internal`**: shared for
evaluation, review, adversarial analysis, mentorship, and challenge-internal
discussion. See `LICENSE_OR_POLICY_NOTICE.md`. If the team later opts into
public release, update `submission.yaml` (`policy.public_release_opt_in`) and
your own licensing.

## 8. Submission checklist

- [ ] `Agent` API matches the spec and loads from `agent.py`.
- [ ] `config.yaml` reflects your runtime (seed, device, checkpoint references).
- [ ] `submission.yaml` has a complete `runtime_profile` matching your eval needs.
- [ ] `submission.yaml` has real `repo_url` / `commit_hash` and team fields.
- [ ] `interpretation_card.md` is complete and non-sensitive.
- [ ] `requirements.txt` lists only imported deps; no heavy unused packages.
- [ ] No secrets, private data, or hidden benchmark files in the repo.
- [ ] Validation run completed using the official core script on allowed data.
- [ ] `scores.json` is gitignored and not submitted unless instructed.
- [ ] Rolling submission issue opened with the full pinned commit SHA.