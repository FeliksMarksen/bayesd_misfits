# Interpretation Card — Bayes'd Misfits

> **Status:** Fully updated for the winning **Dual-Alpha + Sticky RL** submission.

## 1. Core Claim

Our submission agent implements a **Dual-Alpha Rescorla-Wagner reinforcement learning model with choice stickiness (perseveration)**. It asserts that human choice patterns in this multi-arm restless bandit task are dominated by two main cognitive processes:
1. **Asymmetric Learning rates:** Separate updating speeds for positive and negative prediction errors (PE), reflecting differing updates from rewards vs. omissions.
2. **Choice Perseveration (Stickiness):** A significant, value-independent bias to repeat the previous choice, which prevents choice noise ($\beta$) from being overestimated.

## 2. Mechanism Mapping

| Concept | Where it lives |
|--------|----------------|
| Pos/Neg Learning Rates | `rl_alpha_pos` (0.380) / `rl_alpha_neg` (0.763) in `config.yaml` — positive errors update moderately, while negative errors update aggressively to prune bad options. |
| Choice Stickiness | `sticky` (0.126) — a value bonus added to the last-chosen arm's logit before softmax. |
| Choice Consistency / Noise | `beta` (8.331) — inverse-temperature mapping value difference to choice probability. |
| Initial Beliefs | `initial_q` (0.5) — prior expected value for all arms, corresponding to the normalized [0, 1] reward scale. |

## 3. Alternative Explanations

- **Hierarchical Gaussian Filter (HGF):** Posits that agents track environment volatility dynamically. MCMC comparisons showed HGF performed significantly worse (NLL 0.74 vs 0.61) under a softmax choice rule because the uncertainty of neglected arms grows, causing overcorrection when they are eventually sampled.
- **Symmetric RW + Decay:** Posits a single learning rate and forgetting. This fits worse because humans learn asymmetric lessons from success vs. failure, and decay-based memory is too short-sighted.

## 4. Discriminative Test

- **Ablation of Stickiness:** Removing the `sticky` parameter causes the inverse-temperature $\beta$ estimate to saturate at the boundary ($\approx 9.9$), showing it is forced to capture perseveration as decision certainty.
- **Asymmetry Test:** Forcing `rl_alpha_pos` = `rl_alpha_neg` degrades fit, showing participants update their beliefs faster when disappointed by a loss (alpha neg = 0.76) than when confirmed by a win (alpha pos = 0.38).

## 5. Predictive Role

Including both learning rate asymmetry and choice stickiness drops the one-step-ahead choice prediction NLL from **0.708 (standard RW)** to **0.609 (Sticky)**, making it our primary model for the MindRL Challenge.

## 6. Failure Conditions

- **Sudden Volatility Shifts:** Since the learning rates and stickiness parameters are fixed, the model cannot dynamically speed up learning if the environment's drift rate changes dramatically.
- **Uncertainty-guided exploration:** The softmax decision rule does not actively choose options because they are uncertain (no exploration bonus like UCB/Thompson sampling).

## 7. Evidence Summary

Formal hierarchical MCMC comparisons (N=30 participants, 120 trials each, 1000 draws) ranked the models as follows:
1. **Dual-Alpha + Sticky RL**: NLL = **0.6097** 🥇
2. **Dual-Alpha**: NLL = 0.6999
3. **RW + Decay**: NLL = 0.7043
4. **Standard RW**: NLL = 0.7081
5. **HGF**: NLL = 0.7382

## 8. Reproducibility Notes

- Agent entrypoint: `agent.py` (`Agent` class).
- Hyperparameters: `config.yaml` under `model.*`.
- Model comparisons: `scripts/run_comparison.py`.