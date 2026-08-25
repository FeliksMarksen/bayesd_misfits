# Interpretation Card: Bayes'd Misfits (Causal Sampling Bandit + Choice Trace)

## 1. Core Claim

The submitted agent is a **scale-free Causal Sampling Bandit with Choice Trace (CSB+Trace)**. It models human decision-making in restless multi-arm bandits as a **resource-rational cognitive process** rather than an infinite-precision curve-fitter:

1. **Bounded Sampling Hypothesis**: Humans do not maintain exact continuous 64-bit value updates or compute exact Bayesian posteriors. Instead, they draw a bounded budget of mental samples ($N=10$ mutation steps) over discrete value hypotheses (Kolvoort, Temme & van Maanen, 2023).
2. **Mechanistic Biases from Resource Constraints**:
   - **Belief Inertia / Anchoring**: Mental chains are warm-started from the previous trial's hypothesis state; with a limited chain length ($N=10$), beliefs adapt incrementally rather than jumping instantaneously.
   - **Cognitive Conservatism**: Choice probabilities are regularized by a symmetric Dirichlet / Laplace pseudo-count prior ($\text{prior\_beta}$) on best-arm sample frequencies: $\log(\bar{C}_a + \text{prior\_beta})$. This categorical generalization of the BMS Beta prior prevents spurious 0%/100% extremes and models human conservatism toward indifference from first principles.
3. **Motor vs. Cognitive Disentanglement**:
   - Motor inertia (the tendency to physically repeat button presses) is captured by a gradual exponential choice trace ($\text{trace}_t$).
   - This prevents reward-learning parameters from being distorted by physical choice autocorrelation.

---

## 2. Learning and Choice Mechanism

### A. Hypothesis Space & Mutation Chain

The agent maintains hypotheses about the latent standardized values of the 4 bandit arms over discrete levels $\mathcal{V} = \{-2, -1, 0, 1, 2\}$ in running $z$-score units.

Given the history of standardized rewards, the Gaussian log-likelihood for a discrete hypothesis vector $\vec{v} \in \mathcal{V}^4$ with exponential recency decay ($\gamma = 0.98$) is:

$$\log \mathcal{L}(\vec{v}) = -\frac{1}{2 \sigma_{\text{obs}}^2} \sum_{a=0}^3 \sum_{k=1}^{n_a} \gamma^{t - t_k} (z_{a, k} - v_a)^2$$

On each trial $t$, the agent runs an ensemble of $M=32$ parallel Metropolis-Hastings mutation chains for $N=10$ steps, warm-started from the previous trial's hypothesis state. At each step, one arm is chosen at random, shifted by $\pm 1$ level, and accepted/rejected with the standard Hastings ratio.

### B. Expected Policy & Choice Logits

For each chain $m$, the best arm $\arg\max_a v_{m, a}$ is counted (splitting ties evenly). The expected best-arm frequency is averaged across the ensemble: $\bar{C}_a = \frac{1}{M} \sum_{m=1}^M C_{a, m}$.

Choice logits combine the cognitive sampling evidence with the motor choice trace:

$$\text{logit}_a = \text{temperature} \cdot \log(\bar{C}_a + \text{prior\_beta}) + \text{repetition\_weight} \cdot \text{trace}_a(t)$$

$$\text{trace}_a(t) = \text{trace}_a(t-1) + \text{choice\_trace\_rate} \cdot (\mathbb{I}[a_{t-1} = a] - \text{trace}_a(t-1))$$

A softmax transformation converts these logits into action probabilities.

---

## 3. Fitted Parameters and Psychological Roles

Parameters were estimated via 4-chain hierarchical MCMC (NumPyro NUTS, 4 chains × 500 tune + 500 draws, 50 training subjects, 12,000 held-out validation trials; 0 divergences, min bulk ESS = 459.5, max $\hat{R} = 1.019$):

| Parameter | Fitted Value | 94% HDI | Psychological Meaning |
|---|:---:|:---:|---|
| `prior_beta` | **1.134684** | $[0.3531, 2.5587]$ | Dirichlet pseudo-count prior strength; pulls choice toward indifference |
| `temperature` | **0.415717** | $[0.2658, 0.6179]$ | Choice sharpness / sensitivity to sampling count ratios |
| `repetition_weight` | **3.170215** | $[2.9404, 3.3869]$ | Motor perseveration bonus; strength of the action-repetition habit |
| `choice_trace_rate` | **0.412275** | $[0.3691, 0.4545]$ | Memory decay rate of the choice trace (retains ~2–3 past choices) |

---

## 4. Causal Reward Standardization & Recency Structure

The agent assumes no prior knowledge of reward units, scale, or offsets:
1. **Running Standardization**: Each reward is standardized online using Welford's one-pass algorithm:
   $$z_t = \frac{r_t - \mu_{t-1}}{\sigma_{t-1}}$$
   using only rewards already revealed in the current trajectory ($z_t = 0.0$ before at least two distinct rewards are observed).
2. **Scale Stability vs. Evidence Recency**: The global reward mean and variance are estimated across the full trajectory history to maintain a stable coordinate frame for $z$-scores, while per-arm sufficient statistics decay exponentially ($\gamma = 0.98$) so recent outcomes dominate value inference in drifting environments.
3. **Scale-Invariance**: Multiplying all rewards by $c > 0$ and adding $d \in \mathbb{R}$ leaves predicted choice probabilities strictly invariant within floating-point precision ($\Delta P < 10^{-12}$).

---

## 5. Empirical Performance on Held-Out Human Subjects

Evaluated on $100$ held-out trajectories ($12{,}000$ trials) from disjoint human subjects (reported directly from `csb_and_trace_results.json`):

| Model | Held-out NLL (Mean $\pm$ SE) | Total Held-out NLL | Architecture |
|---|:---:|:---:|---|
| **`CSB+Trace` (Submitted)** | **0.6870** ($\pm 0.0092$) | **8,243.55** | **BMS Sampling ($N=10, M=32$) + Gradual Choice Trace** |
| *Workshop Baseline (Dual-Alpha RW + Sticky)* | *0.6912* ($\pm 0.0094$) | *8,294.40* | *Standard heuristic benchmark* |
| *Pure CSB (No Trace)* | *1.1279* ($\pm 0.0070$) | *13,534.33* | *Pure BMS process model (no motor trace)* |
| *Uniform Random* | *1.3863* | *16,635.53* | *Chance baseline* |

---

## 6. Discriminative Tests

1. **Disable Choice Trace (`repetition_weight = 0`)**:
   - Tests whether motor perseveration is distinct from value belief. Without trace, NLL drops to $1.1279$, demonstrating that sampling alone cannot explain motor inertia.
2. **Prior Sensitivity (`prior_beta -> 0` vs `prior_beta = 50`)**:
   - $\beta \rightarrow 0$ produces over-confident spiky choices; $\beta = 50$ produces indifferent exploration. The fitted $\beta = 1.13$ matches theoretical BMS predictions for human conservatism.
3. **Scale Invariance Verification**:
   - Running the agent with $r \rightarrow 10r + 500$ yields identical action probability trajectories ($\Delta P < 10^{-12}$).
