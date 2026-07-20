# Interpretation Card — Bayes'd Misfits

## 1. Core Claim

The agent implements a **Dual-Alpha Rescorla-Wagner model with choice stickiness**. It maintains a value estimate $Q_a$ per arm, updated with asymmetric learning rates for positive vs. negative prediction errors, and selects actions via softmax with a perseveration bonus on the last-chosen arm.

**Update rule** (`update()`): $\delta = r - Q_a$, then $Q_a \leftarrow Q_a + \alpha\,\delta$ where $\alpha = \alpha_{+}$ if $\delta \geq 0$, else $\alpha_{-}$.

**Choice rule** (`predict()`): $P(a) \propto \exp\!\bigl(\beta \cdot [Q_a + \mathbb{1}(a = a_{\text{prev}}) \cdot s]\bigr)$

## 2. Mechanism Mapping

| Parameter | Value | Role in `agent.py` |
|-----------|-------|---------------------|
| `rl_alpha_pos` | 0.380 | Learning rate for positive prediction errors (reward ≥ expectation) |
| `rl_alpha_neg` | 0.763 | Learning rate for negative prediction errors (disappointment) |
| `sticky` | 0.126 | Value-independent bonus added to the last-chosen arm's logit |
| `beta` | 8.331 | Softmax inverse temperature |
| `initial_q` | 0.5 | Starting Q-value for all arms (neutral on [0, 1] scale) |

## 3. Alternative Explanations

- **HGF:** Tracks environment volatility dynamically but underperforms under softmax because uncertainty of neglected arms grows, causing overcorrection on re-sampling.
- **Symmetric RW + Decay:** A single learning rate with forgetting fits worse because humans update asymmetrically from wins vs. losses.

## 4. Discriminative Test

- Removing `sticky` causes $\beta$ to saturate at the prior boundary (~9.9), showing it is forced to absorb perseveration as choice certainty.
- Forcing $\alpha_{+} = \alpha_{-}$ degrades fit, confirming asymmetric updating.

## 5. Predictive Role

Q-values are updated online via `update()` after every trial, so the agent learns from each reward. $\alpha_{-} > \alpha_{+}$ means bad options are pruned quickly while good options are reinforced gradually. The stickiness term captures choice inertia independently of value, preventing $\beta$ from being overestimated.

## 6. Failure Conditions

- Hyperparameters ($\alpha_{+}$, $\alpha_{-}$, sticky, $\beta$) are fixed from config; the model cannot adapt its learning speed to sudden volatility shifts. (Q-values do update online each trial.)
- No exploration bonus for uncertain arms (pure softmax).
- Assumes rewards on a [0, 1] scale (divides by 100 if raw rewards > 1.0).

## 7. Evidence Summary

Hierarchical MCMC fitting on training data. Parameters in `config.yaml` are the group-level posterior means.

## 8. Reproducibility Notes

- Agent: `agent.py` (`Agent` class).
- Parameters: `config.yaml` under `model.*`.
- Fitting: `scripts/run_comparison.py`.

## 9. Confidentiality Note

No secrets, private URLs, or identifiable participant data in this repository.