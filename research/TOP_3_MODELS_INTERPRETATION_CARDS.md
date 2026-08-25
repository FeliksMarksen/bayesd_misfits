# Top 3 MindRL Modeling Challenge Candidate Models: Scientific Evidence & Interpretation Cards
### Team: Bayes'd Misfits

This document provides complete, self-contained scientific interpretation cards, mathematical formulations, literature citations, parameter estimates, and empirical verification for the **Top 3 Candidate Models** developed in this codebase.

---

## 1. Candidate 1: Causal Scale-Free Rescorla-Wagner + Gradual Choice Trace (`CausalRWTrace`)
**Recommended Role:** Top Predictive Performer (Lowest NLL in Codebase: **0.6082**, Accuracy: **79.5%**)

### Core Theoretical Claim
Human decision-making in multi-armed restless bandits is governed by two interacting processes:
1. **Causal, scale-free reward learning**: Error-driven value estimation that is invariant to arbitrary units and origin shifts in reward feedback.
2. **Gradual motor inertia / policy compression**: An exponential choice-history trace capturing multi-trial action autocorrelation that operates independently of reward.

### Mathematical Formulation
1. **Causal Running Reward Normalization (Welford's Algorithm)**:
   For rewards $r_1, \dots, r_{t-1}$, update running mean $\mu_t$ and sample variance $\sigma_t^2$.
   Standardized action value: $z_t(a) = \frac{Q_t(a) - \mu_t}{\sigma_t}$ (or $0$ when $\sigma_t \approx 0$).

2. **Value Learning (Rescorla & Wagner, 1972)**:
   $$\delta_t = r_t - Q_t(a_t)$$
   $$Q_{t+1}(a_t) = Q_t(a_t) + \alpha \delta_t, \quad Q_{t+1}(a) = Q_t(a) \text{ for } a \neq a_t$$

3. **Gradual Choice Trace (Lau & Glimcher, 2005; Lai & Gershman, 2024)**:
   $$T_{a, t+1} = T_{a, t} + \eta_{\text{trace}} (\mathbf{1}_{a_t=a} - T_{a, t})$$

4. **Softmax Choice Rule**:
   $$\text{logit}_t(a) = \beta z_t(a) + \omega_{\text{trace}} T_{a, t}$$
   $$P(a_t = a) = \frac{\exp(\text{logit}_t(a))}{\sum_{a'} \exp(\text{logit}_t(a'))}$$

### Fitted Population Parameters (Pooled MCMC, 300 Train Trajectories)
| Parameter | Value | 94% Credible Interval | Role |
|---|---:|:---:|---|
| $\alpha$ (`rl_alpha`) | 0.9330 | [0.9044, 0.9591] | Fast value updating from recent prediction errors |
| $\beta$ (`beta`) | 0.7160 | [0.6920, 0.7398] | Sensitivity to standardized action values |
| $\omega_{\text{trace}}$ (`repetition_weight`) | 2.5442 | [2.4776, 2.6101] | Strength of the gradual choice-history trace |
| $\eta_{\text{trace}}$ (`choice_trace_rate`) | 0.3303 | [0.3168, 0.3444] | Decay rate / recency weighting of past actions |

### Scientific Literature Grounding
- **Rescorla, R. A., & Wagner, A. R. (1972)**. *A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement*. Classical error-driven associative learning.
- **Lau, B., & Glimcher, P. W. (2005)**. *Dynamic response-by-response models of matching behavior in rhesus monkeys*. Journal of the Experimental Analysis of Behavior, 84(3), 555-579. Choice traces capture dynamic reinforcement history.
- **Gershman, S. J. (2020)**. *Origin of perseveration in the trade-off between reward and compression*. Cognition, 201, 104294. Demonstrates that choice perseveration is an information-theoretic bound on policy complexity.
- **Lai, L., & Gershman, S. J. (2024)**. *Policy compression in human reinforcement learning*. Nature Human Behaviour. Explains habit formation as a rate-distortion optimization between value tracking and default choice policy.

---

## 2. Candidate 2: Causal Scale-Free Dual-Alpha RW with Immediate Stickiness (`CausalScaleDualAlphaSticky`)
**Recommended Role:** Robust Production Baseline (Verified 4-Chain MCMC Fit, Local NLL: **0.6944**, Accuracy: **75.9%**)

### Core Theoretical Claim
Learning rates may differ for positive versus negative prediction errors (valenced asymmetry), while choice stickiness captures immediate 1-step perseveration. Causal trajectory-local normalization ensures scale-free deployment across unknown evaluation environments.

### Mathematical Formulation
1. **Valenced Prediction Error Updates (Frank et al., 2004; Niv et al., 2007)**:
   $$\delta_t = r_t - Q_t(a_t)$$
   $$\alpha_t = \begin{cases} \alpha_+ & \text{if } \delta_t \ge 0 \\ \alpha_- & \text{if } \delta_t < 0 \end{cases}$$
   $$Q_{t+1}(a_t) = Q_t(a_t) + \alpha_t \delta_t$$

2. **Choice Policy**:
   $$\text{logit}_t(a) = \beta \left(\frac{Q_t(a) - \mu_t}{\sigma_t}\right) + \omega_{\text{rep}} \mathbf{1}_{a_{t-1}=a}$$

### Fitted Population Parameters (4 Chains $\times$ 1,000 Tune + 1,000 Draws)
| Parameter | Value | 94% Credible Interval | Diagnostics |
|---|---:|:---:|:---:|
| $\alpha_+$ (`rl_alpha_pos`) | 0.7309 | [0.6719, 0.7884] | $\hat{R} = 1.001$, Bulk ESS = 1,420 |
| $\alpha_-$ (`rl_alpha_neg`) | 0.7105 | [0.6216, 0.7963] | $\hat{R} = 1.007$, Bulk ESS = 733 |
| $\beta$ (`beta`) | 1.5025 | [1.3204, 1.6881] | $\hat{R} = 1.002$, Bulk ESS = 1,180 |
| $\omega_{\text{rep}}$ (`repetition_weight`) | 1.3713 | [1.2294, 1.5126] | $\hat{R} = 1.001$, Bulk ESS = 2,150 |

### Scientific Literature Grounding
- **Frank, M. J., Seeberger, L. C., & O'Reilly, R. C. (2004)**. *By carrot or by stick: cognitive reinforcement learning in parkinsonism*. Science, 306(5703), 1940-1943. Identifies distinct D1 (positive PE) and D2 (negative PE) striatal learning mechanisms.
- **Gershman, S. J. (2015)**. *Do learning rates differ for positive and negative prediction errors?* Cognitive, Affective, & Behavioral Neuroscience, 15(4), 848-854. Evaluates evidence for asymmetric learning in bandit tasks.

---

## 3. Candidate 3: Hierarchical Gaussian Filter with Uncertainty & Repetition (`PyHGF+Uncertainty+Sticky`)
**Recommended Role:** Normative Bayesian Perceptual Inference (NLL: **0.7646**, Production MCMC Converged)

### Core Theoretical Claim
Humans dynamically adjust their learning rates according to environmental volatility and subjective precision. Furthermore, action selection is directed by both expected reward and epistemic uncertainty (uncertainty bonus/penalty), balanced with motor repetition.

### Mathematical Formulation
1. **Two-Level Exact Bayesian Volatility Filtering (Mathys et al., 2011, 2014)**:
   - Level 1: Arm reward contingency $\mu_{1, t}(a)$ with precision $\pi_{1, t}(a)$.
   - Level 2: Log-volatility $\omega$ controlling drift rate $\exp(\omega)$.
   - Unchosen arms: Precision decays as $\pi_{1, t}(a)^{-1} \leftarrow \pi_{1, t-1}(a)^{-1} + \exp(\omega)$.
   - Chosen arm: Precision updates with observed outcome precision: $\hat{\pi}_1 = \pi_{1, \text{pred}} + \pi_{\text{obs}}$.
   - Value update: $\Delta \mu_1(a_t) = \frac{\pi_{\text{obs}}}{\hat{\pi}_1} (r_t - \mu_{1, \text{pred}})$.

2. **Uncertainty-Guided Policy (Gershman, 2018; Speekenbrink & Konstantinidis, 2015)**:
   $$\sigma_{1, t}(a) = \frac{1}{\sqrt{\pi_{1, t}(a)}}$$
   $$\text{logit}_t(a) = \beta \mu_{1, t}(a) + \omega_{\text{unc}} \sigma_{1, t}(a) + \omega_{\text{rep}} \mathbf{1}_{a_{t-1}=a}$$

### Fitted Population Parameters (Production MCMC, 4 Chains)
| Parameter | Value | 94% Credible Interval | Diagnostics |
|---|---:|:---:|:---:|
| $\omega$ (`ghgf_omega`) | -0.3643 | [-0.5348, -0.2294] | $\hat{R} = 1.003$, Bulk ESS = 1,004 |
| $\beta$ (`beta`) | 0.0566 | [0.0335, 0.0841] | $\hat{R} = 1.001$, Bulk ESS = 1,840 |
| $\omega_{\text{unc}}$ (`uncertainty_weight`) | -0.4191 | [-0.6074, -0.2312] | $\hat{R} = 1.002$, Bulk ESS = 1,410 |
| $\omega_{\text{rep}}$ (`repetition_weight`) | 1.5831 | [1.3912, 1.7760] | $\hat{R} = 1.001$, Bulk ESS = 2,230 |

### Parameter & Model Recovery Evidence (`research/pyhgf_uncertainty_recovery_results.json`)
- In simulated agents with uncertainty aversion ($\omega_{\text{unc}} = -1.5$), HGF with uncertainty recovered $\hat{\omega}_{\text{unc}} = -1.55$ and beat value-only models (NLL 1.206 vs 1.257).
- In combined agents ($\omega = -1, \beta = 0.2, \omega_{\text{unc}} = -0.9, \omega_{\text{rep}} = 1.7$), recovery accurately identified the combined architecture with validation NLL 0.701 vs 1.386 (value-only).

### Scientific Literature Grounding
- **Mathys, C., Daunizeau, J., Friston, K. J., & Stephan, K. E. (2011)**. *A bayesian foundation for individual learning under uncertainty*. Frontiers in Human Neuroscience, 5, 39.
- **Mathys, C. D. et al. (2014)**. *Uncertainty in perception and the Hierarchical Gaussian Filter*. Frontiers in Human Neuroscience, 8, 825.
- **Gershman, S. J. (2018)**. *Deconstructing the human algorithms for exploration*. Cognition, 173, 34-42.
- **Speekenbrink, M., & Konstantinidis, E. (2015)**. *Uncertainty and exploration in a restless bandit problem*. Topics in Cognitive Science, 7(2), 351-367.
