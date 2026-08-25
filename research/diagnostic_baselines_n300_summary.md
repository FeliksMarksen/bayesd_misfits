# Diagnostic baseline result: 300/300 subject-disjoint split

Run date: 2026-08-16. Parameters were estimated on 300 trajectories from 235
human subjects and frozen before evaluation on 300 trajectories from 172
disjoint subjects. All dynamic state reset at trajectory boundaries and used
only already revealed history.

## Held-out scores

| Model | Mean NLL |
|---|---:|
| Causal single-alpha RW + gradual choice trace | **0.608160** |
| Gradual choice trace without rewards | 0.661316 |
| Causal single-alpha RW + previous choice | 0.667389 |
| Causal single-alpha RW without choice history | 0.805184 |
| Previous choice without rewards | 0.817026 |
| Causal running choice frequency | 0.889548 |
| Uniform | 1.386294 |
| Fixed training-set action frequencies | 1.387129 |

The deterministic frozen-parameter RW scores reproduce the corresponding
pooled MCMC posterior-predictive scores to within approximately 0.0002 NLL.
All five multistart optimizations converged with selected gradient norms below
`7e-9`.

## Paired mechanism contrasts

Intervals are paired 5,000-draw cluster bootstraps over the 172 held-out human
subjects. Positive improvement means the candidate has lower NLL than its
reference.

| Candidate contrast | Mean improvement | 95% interval |
|---|---:|---:|
| Running choice frequency vs uniform | 0.49675 | 0.45224 to 0.53957 |
| Previous choice vs uniform | 0.56927 | 0.51338 to 0.62247 |
| Choice trace vs previous choice | 0.15571 | 0.14287 to 0.16888 |
| Reward-only causal RW vs uniform | 0.58111 | 0.52280 to 0.63389 |
| RW + previous choice vs reward-only RW | 0.13779 | 0.09587 to 0.18239 |
| RW + trace vs RW + previous choice | 0.05923 | 0.04497 to 0.07555 |
| RW + trace vs choice trace alone | 0.05316 | 0.04350 to 0.06181 |
| Choice trace alone vs RW + previous choice | 0.00607 | -0.01702 to 0.03097 |
| Fixed action frequencies vs uniform | -0.00084 | -0.00651 to 0.00506 |

## Conclusions

1. There is no useful population-wide action-label bias.
2. Choices are strongly autocorrelated within trajectories. A gradual trace
   captures substantially more than only the immediately prior choice.
3. Choice history alone performs about as well as reward learning combined
   with only an immediate repetition term. This means an underspecified
   history model can make reward-learning parameters absorb autocorrelation.
4. Rewards still contain unique predictive information. Adding causal RW to
   the choice trace improves NLL reliably for 87.8% of held-out subjects.
5. The best frozen diagnostic model combines reward learning and gradual
   choice history. Its fitted learning rate is high (`0.9726`), so that value
   should not receive a strong psychological interpretation before model and
   parameter recovery.
6. The completed factorial comparison gives no predictive justification for
   dual learning rates or the current subject hierarchy: dual rates add
   essentially nothing in pooled models, and every completed hierarchy scores
   worse than its pooled counterpart. Several hierarchical fits also fail
   convergence thresholds.

These findings motivate decomposing action persistence and task structure
before adding another complicated learning architecture. HGF or
resource-rational variants should be compared against the reward-free trace
and should include a separately controlled choice-history component, rather
than asking their learning parameters to explain autocorrelation.
