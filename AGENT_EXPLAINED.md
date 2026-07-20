# Agent & Submission Files — Line by Line

This document explains, in plain language and line by line, every submission-relevant file in this repo: what it does, why each piece is there, and how the pieces fit together. It is written for someone who wants to actually *understand* the submission, not just ship it.

---

## 0. The big picture: two jobs, split across two places

Before the line-by-line, the one mental model that makes everything click:

```
  NOTEBOOKS / scripts/run_comparison.py        agent.py
  (the "lab" — done offline, once)             (the "player" — the evaluator runs this)
  ───────────────────────────────────────      ────────────────────────────────────────
  HSSM fits your model to the data via MCMC,   runs the model FORWARD with FIXED params,
  finds the best alpha_pos, alpha_neg,         produces action probabilities each trial,
  sticky, beta. (slow)                          scored by the evaluator. (fast)
  output: idata (posterior)          →         input: fixed params read from config.yaml
```

You fit once, offline, with HSSM. You take the fitted numbers and write them into `config.yaml`. The evaluator then instantiates your `Agent(config)` and calls `predict`/`update` over and over. The Agent never fits anything — it just *acts* using the params you already found. Everything below is the machinery that makes that handoff work.

---

## 1. The contract the evaluator enforces (read this first)

The evaluator imports a class named **`Agent`** from **`agent.py`** and calls four methods:

```python
class Agent:
    def __init__(self, config=None)      # once, at construction; config = parsed config.yaml
    def reset(self, context)             # once per trajectory; context has available_actions
    def predict(self, history) -> dict   # every step; returns {"action_probs": {action: prob}}
    def update(self, action, reward, info=None)  # after each trial's true outcome is revealed
```

**Hard invariants:**
- `predict` always returns `{"action_probs": {action_label: probability, ...}}`.
- Probabilities are non-negative and sum to 1.0.
- Keys are the **exact action labels** from `context["available_actions"]` (same type/identity).
- Every valid action is a key; no extras, no missing.
- `history` contains **only past trials** — never peek at the current step's outcome.
- `reset` is called once per trajectory; `__init__` once overall.

The challenge raw data format is `{"context": {...}, "trials": [{trial_index, action, reward, info}, ...]}` — so the action field is `"action"` and the reward field is `"reward"`, and the evaluator passes rewards to agents on a **0–1 normalized scale** (the official WSLS baseline's `win_threshold = 0.5` is only sensible at that scale). Raw challenge rewards are 1–100; your params were fit on `reward/100` (normalized), so the agent must keep rewards on that 0–1 scale.

---

## 2. `agent.py` — line by line

### Module docstring & imports

```python
from __future__ import annotations
from collections.abc import Mapping, Sequence
import math
from typing import Any
```
- `from __future__ import annotations` — lets the type hints (`Mapping[str, Any] | Any | None`) work on older Pythons; purely cosmetic.
- `Mapping`, `Sequence` — used to detect whether `config`/`context`/`history` are dict-like or list-like without assuming exact types.
- `math` — only for `math.exp` in the softmax.
- `Any` — type hints.

Only the standard library is imported. `agent.py` does **not** import numpy, HSSM, or pyhgf — the heavy fitting stack is never needed at eval time. That's why `requirements.txt` stays tiny.

### Helper functions

```python
def get_field(obj, name, default=None):
    if obj is None: return default
    if isinstance(obj, Mapping): return obj.get(name, default)
    return getattr(obj, name, default)
```
Reads `name` from `obj` whether `obj` is a dict (`obj.get`) or an arbitrary object/dataclass (`getattr`). The evaluator may pass trials as dicts or as objects; this handles both. Used in `reset`/`predict` to fish out `available_actions`, `action`, `reward`.

```python
def normalize_probs(probs):
    clipped = {a: max(0.0, float(p)) for a, p in probs.items()}
    total = sum(clipped.values())
    if total <= 0.0: ... uniform fallback ...
    return {a: p / total for a, p in clipped.items()}
```
Defensive: clips any negative to 0, then divides by the sum so probabilities total exactly 1.0. The softmax already produces non-negative values, so in practice this just renormalizes after the `1e-5` floor is applied. It's a safety net against float drift.

```python
def _as_history_list(history):
    if history is None: return []
    if isinstance(history, (str, bytes)): return []
    if isinstance(history, Sequence) ...: return list(history)
    return []
```
Coerces whatever `history` the evaluator passes (list, tuple, generator, None) into a plain list of trials. Strings/bytes are excluded so a stray string isn't iterated character-by-character.

```python
def _default_actions():
    return [1, 2, 3, 4]
```
Fallback action set if `context` doesn't provide `available_actions`. The challenge uses 4 arms, so `[1,2,3,4]` is a safe default.

### The `Agent` class

```python
class Agent:
    """Dual-Alpha + Sticky RL submission agent. ..."""
```
A **Dual-Alpha Rescorla-Wagner learner with choice stickiness**, fit offline with HSSM. Conceptually it keeps a value `Q` per arm, updates `Q` of the chosen arm after each trial (with a *different* learning rate for good vs bad news), adds a "stickiness" bonus to the last-chosen arm before choosing, and turns values into choice probabilities via a softmax. The fitted parameters live in `config.yaml`.

#### `__init__`

```python
def __init__(self, config=None):
    self._config = config or {}
    model = get_field(self._config, "model", default={}) or {}
```
`config` is the parsed `config.yaml`. The agent reads its hyperparameters **once**, here, from the `model:` block, and stores them on `self` so the other methods don't re-read config.

```python
    self._rl_alpha_pos = float(get_field(model, "rl_alpha_pos", 0.380))
    self._rl_alpha_neg = float(get_field(model, "rl_alpha_neg", 0.763))
    self._sticky      = float(get_field(model, "sticky", 0.126))
    self._beta        = float(get_field(model, "beta", 8.331))
    self._initial_q   = float(get_field(model, "initial_q", 0.5))
```
The five fitted parameters (defaults are the fitted values, so the agent works even if a key is missing):
- `rl_alpha_pos` (0.380) — learning rate for **positive** prediction errors (reward better than expected). Moderate.
- `rl_alpha_neg` (0.763) — learning rate for **negative** prediction errors (disappointment). Larger: people prune bad options faster than they reinforce good ones.
- `sticky` (0.126) — a value-independent bonus added to the last-chosen arm's logit, capturing perseveration (the tendency to repeat a choice regardless of value).
- `beta` (8.331) — inverse temperature: how decisively value maps to choice. High beta ≈ the person usually picks the best-looking arm.
- `initial_q` (0.5) — the starting value for every arm (neutral on the 0–1 reward scale).

```python
    self._available_actions: list[Any] = []
    self._q: dict[Any, float] = {}
    self._last_choice: Any = None
    self._history_len = 0
```
State placeholders, filled by `reset`:
- `_available_actions` — the action labels for this trajectory (set in `reset`).
- `_q` — `{action_label: value}` — the model's running belief about each arm's quality.
- `_last_choice` — the action chosen on the previous trial (for the stickiness bonus).
- `_history_len` — how many trials the agent has *internally* processed; used to detect whether `self._q` is in sync with the history the evaluator passed (see `predict`).

#### `reset`

```python
def reset(self, context):
    actions = get_field(context, "available_actions", default=None)
    ...
    self._available_actions = list(actions) if actions is not None else []
    if not self._available_actions:
        self._available_actions = list(_default_actions())
```
Called once per trajectory. Reads the actual action **labels** from `context["available_actions"]`. These labels become the **keys** of every probability dict `predict` returns — so they must be preserved exactly. Falls back to `[1,2,3,4]` if none provided.

```python
    self._q = {a: self._initial_q for a in self._available_actions}
    self._last_choice = None
    self._history_len = 0
```
Starts every arm at the neutral value `0.5`, clears the last-choice memory, and zeroes the history counter. This is the per-trajectory fresh start.

#### `predict` — "which arm will the person pick?"

```python
def predict(self, history):
    actions = list(self._available_actions)
    if not actions:
        return {"action_probs": {}}
    hist = _as_history_list(history)
```
Grab the action labels; coerce history to a list. If there are no actions, return an empty (invalid but safe) dict.

```python
    if len(hist) != self._history_len:
        # Reconstruct state from history
        q = {a: self._initial_q for a in actions}
        last_choice = None
        max_rew = 0.0
        for trial in hist:
            rew = get_field(trial, "reward", default=None)
            ...
            max_rew = max(max_rew, abs(float(rew)))
        scale = 100.0 if max_rew > 1.0 else 1.0
```
**Desync fallback (Design A).** Normally the agent carries state in `self._q` (updated by `update`). But if the number of trials in `history` doesn't match `self._history_len` — meaning `update` was skipped, or the evaluator replayed history out of order — the agent rebuilds its state from scratch by replaying `history`. This is what makes the agent robust: it works whether or not the evaluator calls `update`. The `scale` line is a reward-scale guard: if any reward in history exceeds 1.0, divide all by 100 (handles raw 1–100 rewards); otherwise use rewards as-is (the normal 0–1 case).

```python
        for trial in hist:
            act = get_field(trial, "action", default=None)
            rew = get_field(trial, "reward", default=None)
            if act is not None and act in q and rew is not None:
                r = float(rew) / scale
                pe = r - q[act]
                alpha = self._rl_alpha_pos if pe >= 0.0 else self._rl_alpha_neg
                q[act] += alpha * pe
                last_choice = act
        self._q = q
        self._last_choice = last_choice
        self._history_len = len(hist)
```
The replay loop applies the **identical** learning rule as `update` (see below), trial by trial, to rebuild `q` and `last_choice`, then writes them back to `self`. So whether state comes from `update` or from replay, the numbers are the same.

```python
    values = {}
    for a in actions:
        val = self._q[a]
        if self._last_choice is not None and a == self._last_choice:
            val += self._sticky      # perseveration bonus on the last-chosen arm
        values[a] = val
```
Build the "effective value" of each arm: its Q-value, plus the `sticky` bonus if it was the last-chosen arm. This is the choice-stickiness mechanism — it tilts choice toward repeating the previous action independent of value.

```python
    max_val = max(values.values())
    exp_vals = {}
    for a in actions:
        exp_vals[a] = math.exp(self._beta * (values[a] - max_val))
```
**Numerically stable softmax.** `exp(beta * value)` for each arm, shifted by the max so the largest exponent is `exp(0)=1` and nothing overflows. Subtracting the same constant from every exponent doesn't change the final probabilities (it cancels in the division) but keeps `exp` from blowing up when `beta * value` is large.

```python
    floored = {a: max(1e-5, exp_vals[a]) for a in actions}
    return {"action_probs": normalize_probs(floored)}
```
A tiny `1e-5` floor so no probability is *exactly* zero (which would give `-inf` log-loss in scoring), then renormalize so they sum to 1.0. The returned dict's keys are the action labels from `available_actions`, its values are the choice probabilities. This is the contract.

> Note: `history` is only used in the desync-fallback branch. In the normal (in-sync) path, `predict` ignores `history` and reads `self._q` — because `update` has been keeping it current. Both paths produce identical numbers; the fallback just guarantees correctness if `update` is ever skipped.

#### `update` — "learn from what just happened"

```python
def update(self, action, reward, info=None):
    if action not in self._q or reward is None:
        return
```
Guard: do nothing if the action isn't valid or there's no reward.

```python
    try:
        r = float(reward)
        if r > 1.0:
            r = r / 100.0
```
**Reward normalization, in one place.** The params were fit on normalized (0–1) rewards. The evaluator passes 0–1 rewards, so normally no division happens. The `> 1.0` guard catches a raw 1–100 reward if one ever slips through and divides by 100 to match the fit scale.

```python
        pe = r - self._q[action]                       # prediction error
        alpha = self._rl_alpha_pos if pe >= 0.0 else self._rl_alpha_neg  # dual-alpha
        self._q[action] += alpha * pe                 # the one learning line
        self._last_choice = action                     # sticky bonus uses this next step
        self._history_len += 1
    except (TypeError, ValueError):
        pass
```
The heart of the model, in one line: `Q[chosen] += alpha * (reward - Q[chosen])`. The prediction error `pe` is the surprise; `alpha` is the learning rate — `alpha_pos` if the surprise was pleasant, `alpha_neg` if disappointing (this is the "dual-alpha" part). Then `last_choice` is recorded for the stickiness bonus on the next `predict`, and `_history_len` is bumped so the desync check in `predict` stays accurate.

That's the whole model: `predict` reads state → softmax → probabilities; `update` changes state → one Q-learning line. The alternation `predict → update → predict → …` *is* the model.

---

## 3. `config.yaml` — line by line

```yaml
runtime:
  random_seed: 2026
  device: cpu
  execution_type: local_python
  requires_gpu: false
  requires_external_api: false
```
Declares how the evaluator should run the agent: on CPU, in-process, no GPU, no external API. The agent genuinely needs none of those — it's pure Python with `math.exp`.

```yaml
model:
  name: dual_alpha_sticky_rl
  type: cognitive_rl_model
  rl_alpha_pos: 0.380
  rl_alpha_neg: 0.763
  sticky: 0.126
  beta: 8.331
  initial_q: 0.5
```
The fitted hyperparameters, read by `Agent.__init__`. These are the values HSSM found offline (the posterior estimates). `name`/`type` are labels for humans/organizers; the agent only reads the numeric keys. To re-fit, you change these numbers — no code change needed.

```yaml
evaluation:
  allow_online_state_updates: true
```
Tells the evaluator the agent may keep state across trials (i.e. `update` is allowed to change `self._q`). This must be `true` for the stateful design; the desync-fallback in `predict` is the safety net if it were ever `false`.

---

## 4. `submission.yaml` — section by section

```yaml
team:
  team_id: bayesd_misfits
  display_name: Bayes'd Misfits
  members:
    - name: FeliksMarksen
      email: feliks@marksen.de
      role: modeler
```
Team identity for the organizers.

```yaml
submission:
  visibility: internal
  method_family: bayesian_cognitive_model
  short_description: "..."
  repo_url: https://github.com/FeliksMarksen/bayesd_misfits.git
  commit_hash: 21870cc3095b9d0922a118b5ea41a1c569273850
  agent_path: agent.py
  config_path: config.yaml
  requirements_path: requirements.txt
  interpretation_card_path: interpretation_card.md
```
The **mandatory routing metadata**. `commit_hash` must be a full 40-character SHA pinned at submission time (not a branch name). The `*_path` fields tell the evaluator where each file lives — all relative to the repo root.

```yaml
optional_targets: []
```
Empty = choice-only (the main leaderboard). Add `["response_time"]` to opt into the auxiliary RT leaderboard, which would require `predict` to also return a positive finite `"rt_ms"`.

```yaml
runtime_profile:
  execution_type: local_python
  model_family: symbolic_cognitive
  requires_gpu: false
  gpu_type: null
  requires_external_api: false
  required_secrets: []
  estimated_eval_cost: none
  expected_runtime_minutes: null
  notes: "..."
```
Mandatory block the organizers read *before* cloning, to route evaluation. Even GPU/API submissions must still expose a callable `Agent` — this only declares routing. No secrets are ever committed; `required_secrets` lists *names* only (empty here).

```yaml
artifacts:
  hf_checkpoint: null
  external_weights: null
  notes: "Fitted hyperparameters are shipped directly in config.yaml."
```
Large checkpoints would go on Hugging Face and be referenced by URI here. None needed — the fitted params are small and live in `config.yaml`.

```yaml
policy:
  ip_owner: submitting_team
  access_scope: challenge_internal
  public_release_opt_in: false
```
Legal: evaluation-only, IP stays with the team, not opted into public release. Matches `LICENSE_OR_POLICY_NOTICE.md`.

---

## 5. `requirements.txt`

```text
numpy>=1.26,<2.3
pyyaml>=6.0
```
The evaluator `pip install`s these before importing `agent.py`. `pyyaml` parses `config.yaml`; `numpy` is listed for CI compatibility even though the agent itself only uses the standard library. The heavy HSSM/pyhgf stack is **deliberately absent** — fitting happens offline in notebooks, and only the resulting numbers ship in `config.yaml`. Keep this minimal: add a package only if `agent.py` actually imports it at eval time.

---

## 6. `interpretation_card.md`

A 9-section scientific write-up (community-visible). Sections: Core Claim, Mechanism Mapping, Alternative Explanations, Discriminative Test, Predictive Role, Failure Conditions, Evidence Summary, Reproducibility Notes, Confidentiality. It documents the **Dual-Alpha + Sticky** model: asymmetric learning rates (`alpha_pos=0.38`, `alpha_neg=0.76`), choice stickiness (`sticky=0.126`), and the model-comparison ranking that selected it (NLL 0.6097, best of six). The numbers in the card come from `scripts/run_comparison.py`'s output. If you re-fit and the ranking changes, update this card to match.

---

## 7. How it all fits together

```
Offline (once):                      Submission (evaluator runs):
scripts/run_comparison.py           agent.py
  HSSM MCMC fit                        Agent(config) reads config.yaml
  → fitted alpha_pos/neg/sticky/beta   reset(context)  → sets actions + Q=0.5
  → written into config.yaml           predict(history) → softmax(Q + sticky) → probs
                                       update(a, r)     → Q[a] += alpha*(r - Q[a])
                                       (repeat for every trial)
```

The fitted parameters are the bridge: the slow, offline Bayesian fit produces numbers; those numbers go into `config.yaml`; the fast, deterministic `Agent` reads them and plays the model forward. Nothing in `agent.py` does inference — it only *uses* the inference result. That separation is why the submission can run on a CPU in milliseconds while the fitting took minutes-to-hours.

---

## 8. The submission checklist (from the template)

- [x] `Agent` API matches the spec and loads from `agent.py`.
- [x] `config.yaml` reflects runtime + fitted model params.
- [x] `submission.yaml` has a complete `runtime_profile`.
- [x] `submission.yaml` has real `repo_url` / `commit_hash` + team fields.
- [x] `interpretation_card.md` is complete and non-sensitive.
- [x] `requirements.txt` lists only needed deps; no heavy unused packages.
- [x] No secrets, private data, or hidden benchmark files committed.
- [ ] Validation run via the official core `validate_submission.py` (run before final submit).
- [ ] Pin a fresh commit SHA in `submission.yaml` at submission time.
- [ ] Open the rolling-submission issue in `mindrl-challenge-core`.