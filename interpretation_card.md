# Interpretation Card Bayes'd Misfits

## 1. Core Claim

People update their beliefs faster after disappointments than after confirmations, and they tend to repeat their previous choice regardless of its value. The agent models this with per-arm value estimates, dual learning rates, and a perseveration bonus in the softmax choice rule.

## 2. Mechanism Mapping

| Concept                          | Where it lives                                                   |
| -------------------------------- | ---------------------------------------------------------------- |
| Positive prediction error update | `rl_alpha_pos` (0.380): learning rate when reward >= expectation |
| Negative prediction error update | `rl_alpha_neg` (0.763): learning rate when reward < expectation  |
| Choice perseveration             | `sticky` (0.126): bonus added to the last-chosen arm's logit     |
| Choice consistency               | `beta` (8.331): softmax inverse temperature                      |
| Prior expectation                | `initial_q` (0.5): starting Q-value for all arms                 |
| Action values                    | `self._q`: updated online each trial via `update()`              |

## 3. Discriminative Test

- Removing `sticky` causes `beta` to saturate at ~9.9, because it has to absorb perseveration as choice certainty.
- Setting `rl_alpha_pos` = `rl_alpha_neg` degrades fit, confirming asymmetric updating.
- The model should outperform WSLS on sequences where cumulative reward history matters beyond the last trial.

## 4. Predictive Role

The dual learning rates let the model capture how quickly people abandon bad options vs. how gradually they commit to good ones. The stickiness term picks up choice inertia that would otherwise inflate beta. Both components are needed; removing either one hurts fit.

## 5. Failure Conditions

- Learning rates are fixed, so the model cannot adapt to sudden changes in environment drift.
- No exploration bonus: the softmax rule does not direct choices toward uncertain arms.
- Same parameters for all arms and trials, so arm- or phase-specific learning is not captured.
- Rewards outside [0, 1] trigger a heuristic division by 100, which may not suit all tasks.

## 6. Evidence Summary

Parameters were fitted offline with hierarchical MCMC on training data. The values in `config.yaml` are group-level posterior means. No private data or challenge scores are included.

## 7. Reproducibility Notes

- Agent entrypoint: `agent.py` (`Agent` class).
- Hyperparameters: `config.yaml` under `model.*`.
- Fitting script: `scripts/run_comparison.py`.

## 8. Confidentiality Note

Do not add secrets, private URLs with credentials, or identifiable participant data to this card or repository.
