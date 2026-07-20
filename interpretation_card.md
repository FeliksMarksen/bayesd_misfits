# Interpretation Card — Bayes'd Misfits

> **Status:** This card currently documents the **placeholder WSLS baseline**
> so the submission validates. Replace every section below with the claims and
> evidence for your actual model (HGF / HSSM-fitted RL / etc.) before submitting.

## 1. Core Claim

The placeholder model captures a simple reinforcement heuristic: **repeat the
last chosen action after outcomes that are sufficiently good, and switch away
from that action after poor outcomes.** It posits no latent states, value
functions, or lookahead planning — only a local mapping from the most recent
(action, reward) pair to a stochastic choice rule.

## 2. Mechanism Mapping

| Concept | Where it lives |
|--------|----------------|
| Win vs loss | `win_threshold` in `config.yaml` / `Agent._win_threshold` — compares the last observed reward to this scalar. |
| Repeat-after-win tendency | `stay_probability` — after a win, mass on the previous action; remainder split evenly across other actions. |
| Shift-after-loss | Complementary mass on the previous action vs others (mirror of the win case using the same `stay_probability`). |
| Exploration / no zero mass | `epsilon` — applied as a floor on every action before renormalization. |
| Action set | `reset(context)` reads `available_actions`; if missing, defaults to `[1, 2, 3, 4]`. |

## 3. Alternative Explanations

- **Uniform or weakly structured random choice** — could mimic parts of the policy when rewards are uninformative or history is empty.
- **Tabular Q-learning or other value-based RL** — can produce win-stay / lose-shift-like patterns but implies slower credit assignment and sustained value tracking.
- **Recency-weighted value learning** — uses a running summary of past rewards, not a single-step threshold rule.
- **Perseveration independent of reward** — sticky choice behavior without coupling to reward sign or magnitude.

## 4. Discriminative Test

- **High vs low reward on the same action:** manipulate only the last reward and compare `predict` outputs; WSLS should assign higher mass to repeating the last action after high reward and higher mass to switching after low reward.
- **Ablate `win_threshold`:** sweep the threshold and check the stay/shift boundary aligns with the intended win/loss definition.
- **Compare to a reward-insensitive perseveration model:** WSLS should diverge most clearly when the last reward crosses the threshold.

## 5. Predictive Role

WSLS improves short-horizon, locally reward-contingent prediction when
participants approximate "repeat after success, explore or switch after
failure." Removing reward gating collapses the model toward structureless
stickiness or symmetry.

## 6. Failure Conditions

- **Gradual value learning** — outcomes cumulate into a slowly changing value estimate rather than a one-step win/loss cut.
- **Structured generalization** — choices depend on stimuli or contexts not in the one-step history.
- **Long-horizon planning** — decisions depend on future contingencies beyond the last reward.
- **Non-stationary latent structure** — the implicit "win" definition drifts while `win_threshold` stays fixed.
- **Mismatched reward scale** — if rewards are rarely above `win_threshold`, the model behaves as if almost always "losing."

## 7. Evidence Summary

This repository currently ships a **template baseline** for exercising the
submission interface and validator, not a claim of strong empirical fit on
challenge benchmarks. Replace with your model's evidence once fitted.

## 8. Reproducibility Notes

- Agent entrypoint: `agent.py` (`Agent` class).
- Hyperparameters: `config.yaml` under `model.*`.
- Validate with the official `validate_submission.py` from the **core** evaluator repo (see `SUBMISSION.md`).

## 9. Confidentiality Note

Do not add secrets, private URLs with credentials, or identifiable participant
data to this card or repository.