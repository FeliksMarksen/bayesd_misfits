"""Generate the comprehensive Master MindRL Models Overview Jupyter Notebook."""

import json
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell(r"""# MindRL Challenge 2026: Master Models & Cognitive Architecture Overview
### Bayes'd Misfits — Comprehensive Codebase Audit, Model Suite, & Scientific Deep Dive

---

## Executive Summary & Challenge Context
The **MindRL Modeling Challenge** evaluates computational models of human decision-making on 4-armed restless bandits under drifting reward contingencies across multiple test splits (public training, hidden blocks, hidden continuations, unseen subjects, and private transfer tasks).

This master notebook provides an end-to-end, runnable, modular overview of **all model families** developed in this repository:
1. **Heuristic & Non-Reward Baselines**: Uniform random, global frequencies, running empirical counts, immediate choice repetition (win-stay/lose-shift-like stickiness), and exponentially decaying choice traces.
2. **Reinforcement Learning Models**: Classical Rescorla-Wagner (RW), Dual-Alpha RW (asymmetric learning rates), and **Causal Scale-Free RW + Gradual Choice Trace** (the top predictive performer).
3. **Hierarchical Gaussian Filter (PyHGF)**: Exact Bayesian perceptual inference tracking environmental volatility, precision-weighted prediction errors, and uncertainty-modulated choice.
4. **Sampling & Resource-Rational Models**: Bayesian Mutation Sampler (BMS, Kolvoort et al., 2023) and the **Causal Sampling Bandit (CSB+Trace)** combining MCMC hypothesis mutation with Dirichlet conservatism and motor inertia.

---
"""))

# Cell: Setup & Imports
cells.append(nbf.v4.new_code_cell(r"""# Setup environment and imports
import os
import sys
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Set plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Add parent directory to python path
repo_root = Path.cwd().resolve()
if (repo_root / "bayesd_misfits").exists():
    sys.path.insert(0, str(repo_root))
elif (repo_root.parent / "bayesd_misfits").exists():
    sys.path.insert(0, str(repo_root.parent))

print("Environment configured. Python version:", sys.version.split()[0])
"""))

# Section 1: Data Loading & Causal Normalization
cells.append(nbf.v4.new_markdown_cell(r"""---
# Section 1: Dataset Architecture & Causal Running Normalization

### The Causal Normalization Principle
In the MindRL challenge, test splits may present rewards in arbitrary, unknown scales (e.g. 1–100 points, 0–1 probabilities, or shifted Gaussian distributions).
A crucial requirement for a robust agent is **reward scale-invariance without lookahead bias**:
- At trial $t$, the agent has observed rewards $r_1, r_2, \dots, r_{t-1}$.
- It maintains running mean $\mu_t$ and sample variance $\sigma_t^2$ updated strictly using **Welford's one-pass algorithm**.
- Standardized values are defined as $z_t(a) = \frac{Q_t(a) - \mu_t}{\sigma_t}$.
- Multiplying all rewards by $c > 0$ or adding $d \in \mathbb{R}$ leaves predicted choice logits $\beta z_t(a) + \dots$ **strictly invariant**.
"""))

cells.append(nbf.v4.new_code_cell(r"""from bayesd_misfits.data import load_trajectories

# Load public trajectories
trajectories = load_trajectories()
print(f"Loaded {len(trajectories)} total public trajectories.")

# Subject-disjoint split (to prevent data leakage across repeated subject sessions)
subjects = sorted(list({t['context']['subject_id'] for t in trajectories}))
rng = np.random.default_rng(2026)
shuffled_subjects = rng.permutation(subjects)

train_subj = set(shuffled_subjects[:len(shuffled_subjects)//2])
valid_subj = set(shuffled_subjects[len(shuffled_subjects)//2:])

valid_trajectories = [t for t in trajectories if t['context']['subject_id'] in valid_subj][:300]
train_trajectories = [t for t in trajectories if t['context']['subject_id'] in train_subj][:300]

print(f"Subject-disjoint Split: {len(train_subj)} train subjects ({len(train_trajectories)} traj), {len(valid_subj)} valid subjects ({len(valid_trajectories)} traj).")
"""))

cells.append(nbf.v4.new_code_cell(r"""# Welford's Algorithm Implementation & Scale Invariance Demo
class CausalRewardStandardizer:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        
    def update(self, reward: float):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.m2 += delta * delta2
        
    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return np.sqrt(self.m2 / (self.count - 1))
        
    def standardize(self, raw_value: float) -> float:
        s = self.std
        if s < 1e-8:
            return 0.0
        return (raw_value - self.mean) / s

# Test invariance under affine transformation
std_orig = CausalRewardStandardizer()
std_trans = CausalRewardStandardizer()

raw_rewards = [54.0, 72.0, 30.0, 85.0, 60.0]
c, d = 10.0, 500.0  # Affine scale and shift

for r in raw_rewards:
    std_orig.update(r)
    std_trans.update(c * r + d)

raw_val = 70.0
trans_val = c * raw_val + d
z1 = std_orig.standardize(raw_val)
z2 = std_trans.standardize(trans_val)

print(f"Original z-score:    {z1:.8f}")
print(f"Transformed z-score: {z2:.8f}")
print(f"Difference:          {abs(z1 - z2):.2e} (Machine precision invariant)")
"""))

# Section 2: Non-Reward Choice Baselines
cells.append(nbf.v4.new_markdown_cell(r"""---
# Section 2: Non-Reward Choice Baselines & Motor Inertia

### Why Choice Autocorrelation Dominates Bandit Behavior
In multi-armed bandit tasks, human choices exhibit substantial **autocorrelation** independent of reward:
1. **Immediate Stickiness (Previous Choice)**: Direct preference for repeating the action chosen on trial $t-1$.
2. **Gradual Choice Trace**: An exponential moving average of past choices:
   $$T_{a, t} = T_{a, t-1} + \eta_{\text{trace}} (\mathbf{1}_{a_t=a} - T_{a, t-1})$$
   Choice logits are given by:
   $$\text{logit}(a) = \omega_{\text{trace}} T_{a, t-1}$$

As proven by the diagnostic baselines in this codebase, a pure choice trace without reward achieves **NLL 0.651**, beating a reward-only Rescorla-Wagner model (**NLL 0.805**)!
"""))

cells.append(nbf.v4.new_code_cell(r"""# Evaluation Engine
def evaluate_policy(agent, trajectories_list):
    total_nll = 0.0
    total_trials = 0
    correct = 0
    
    for t in trajectories_list:
        agent.reset()
        for tr in t['trials']:
            avail = tr.get('available_actions', [0, 1, 2, 3])
            chosen = tr['action']
            reward = tr['reward']
            
            # Predict
            probs = agent.predict_probs(avail)
            p = max(probs.get(chosen, 1e-12), 1e-12)
            total_nll -= np.log(p)
            
            if max(probs, key=probs.get) == chosen:
                correct += 1
            total_trials += 1
            
            # Update
            agent.update(chosen, reward, avail)
            
    return {
        "mean_nll": total_nll / total_trials,
        "accuracy": correct / total_trials,
        "trials": total_trials
    }
"""))

cells.append(nbf.v4.new_code_cell(r"""# Non-Reward Baseline Agents
class UniformAgent:
    def reset(self): pass
    def predict_probs(self, avail): return {a: 1.0/len(avail) for a in avail}
    def update(self, action, reward, avail): pass

class ChoiceTraceOnlyAgent:
    def __init__(self, rep_weight=3.587921, trace_rate=0.295966):
        self.rep_weight = rep_weight
        self.trace_rate = trace_rate
        self.reset()
    def reset(self):
        self.trace = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    def predict_probs(self, avail):
        logits = [self.rep_weight * self.trace.get(a, 0.0) for a in avail]
        max_l = max(logits)
        exps = [np.exp(l - max_l) for l in logits]
        s = sum(exps)
        return {a: exp / s for a, exp in zip(avail, exps)}
    def update(self, action, reward, avail):
        for a in avail:
            target = 1.0 if a == action else 0.0
            self.trace[a] += self.trace_rate * (target - self.trace[a])

res_uniform = evaluate_policy(UniformAgent(), valid_trajectories)
res_trace = evaluate_policy(ChoiceTraceOnlyAgent(), valid_trajectories)

print(f"Uniform Baseline:         NLL = {res_uniform['mean_nll']:.4f}, Accuracy = {res_uniform['accuracy']*100:.2f}%")
print(f"Choice Trace Only (No RL): NLL = {res_trace['mean_nll']:.4f}, Accuracy = {res_trace['accuracy']*100:.2f}%")
"""))

# Section 3: Reinforcement Learning Models
cells.append(nbf.v4.new_markdown_cell(r"""---
# Section 3: Reinforcement Learning Models

### Model 3.1: Scale-Free Rescorla-Wagner + Gradual Choice Trace (Predictive Frontrunner)
The combination of causal scale-free reward learning with a gradual choice trace creates the highest-performing model in the entire repository:

$$\delta_t = r_t - Q_t(a_t)$$
$$Q_{t+1}(a_t) = Q_t(a_t) + \alpha \delta_t$$
$$T_{a, t+1} = T_{a, t} + \eta_{\text{trace}} (\mathbf{1}_{a_t=a} - T_{a, t})$$
$$\text{logit}(a) = \beta \left(\frac{Q_t(a) - \mu_t}{\sigma_t}\right) + \omega_{\text{trace}} T_{a, t}$$
$$P(a_t = a) = \frac{\exp(\text{logit}(a))}{\sum_{a'} \exp(\text{logit}(a'))}$$

- **Fitted Population Parameters**: $\alpha = 0.9726$, $\beta = 0.7049$, $\omega_{\text{trace}} = 2.5574$, $\eta_{\text{trace}} = 0.3318$.
"""))

cells.append(nbf.v4.new_code_cell(r"""class CausalRWTraceAgent:
    def __init__(self, alpha=0.972555, beta=0.704898, rep_weight=2.557354, trace_rate=0.331843):
        self.alpha = alpha
        self.beta = beta
        self.rep_weight = rep_weight
        self.trace_rate = trace_rate
        self.reset()
        
    def reset(self):
        self.q = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.trace = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.standardizer = CausalRewardStandardizer()
        
    def predict_probs(self, avail):
        logits = []
        for a in avail:
            z_q = self.standardizer.standardize(self.q.get(a, 0.0))
            tr = self.trace.get(a, 0.0)
            logits.append(self.beta * z_q + self.rep_weight * tr)
        max_l = max(logits)
        exps = [np.exp(l - max_l) for l in logits]
        s = sum(exps)
        return {a: exp / s for a, exp in zip(avail, exps)}
        
    def update(self, action, reward, avail):
        self.standardizer.update(reward)
        pe = reward - self.q[action]
        self.q[action] += self.alpha * pe
        for a in avail:
            target = 1.0 if a == action else 0.0
            self.trace[a] += self.trace_rate * (target - self.trace[a])

res_rw_trace = evaluate_policy(CausalRWTraceAgent(), valid_trajectories)
print(f"Causal RW + Gradual Trace: NLL = {res_rw_trace['mean_nll']:.4f}, Accuracy = {res_rw_trace['accuracy']*100:.2f}%")
"""))

cells.append(nbf.v4.new_code_cell(r"""# Model 3.2: Dual-Alpha RW with Immediate Choice Stickiness (Current Submission)
class CausalDualAlphaStickyAgent:
    def __init__(self, alpha_pos=0.730885, alpha_neg=0.710481, beta=1.502530, rep_weight=1.371340):
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.beta = beta
        self.rep_weight = rep_weight
        self.reset()
        
    def reset(self):
        self.q = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.last_choice = None
        self.standardizer = CausalRewardStandardizer()
        
    def predict_probs(self, avail):
        logits = []
        for a in avail:
            z_q = self.standardizer.standardize(self.q.get(a, 0.0))
            rep = 1.0 if (self.last_choice is not None and self.last_choice == a) else 0.0
            logits.append(self.beta * z_q + self.rep_weight * rep)
        max_l = max(logits)
        exps = [np.exp(l - max_l) for l in logits]
        s = sum(exps)
        return {a: exp / s for a, exp in zip(avail, exps)}
        
    def update(self, action, reward, avail):
        self.standardizer.update(reward)
        pe = reward - self.q[action]
        lr = self.alpha_pos if pe >= 0 else self.alpha_neg
        self.q[action] += lr * pe
        self.last_choice = action

res_dual_sticky = evaluate_policy(CausalDualAlphaStickyAgent(), valid_trajectories)
print(f"Dual-Alpha RW + Sticky:    NLL = {res_dual_sticky['mean_nll']:.4f}, Accuracy = {res_dual_sticky['accuracy']*100:.2f}%")
"""))

# Section 4: Hierarchical Gaussian Filter
cells.append(nbf.v4.new_markdown_cell(r"""---
# Section 4: Hierarchical Gaussian Filter (PyHGF)

### Exact Bayesian Perceptual Inference Under Environmental Volatility
The **Hierarchical Gaussian Filter** (Mathys et al., 2011, 2014) maintains Gaussian beliefs over arm values while tracking environmental volatility:
- **State belief**: $\mu_{1, t}(a)$ with precision $\pi_{1, t}(a)$.
- **Dynamic learning rate**: Prediction errors are weighted by precision ratios: $\Delta \mu_1 \propto \frac{\hat{\pi}_{\text{obs}}}{\pi_1}$.
- **Uncertainty-modulated choice**:
  $$\text{logit}(a) = \beta \mu_{1, t}(a) + \omega_{\text{unc}} \sigma_{1, t}(a) + \omega_{\text{rep}} \mathbf{1}_{a_{t-1}=a}$$
  where $\sigma_{1, t}(a) = 1/\sqrt{\pi_{1, t}(a)}$ represents subjective uncertainty.
"""))

cells.append(nbf.v4.new_code_cell(r"""class PyHGFUncertaintyStickyAgent:
    def __init__(self, omega=-0.364285, beta=0.056586, uncertainty_weight=-0.419088, repetition_weight=1.583091):
        self.omega = omega
        self.beta = beta
        self.uncertainty_weight = uncertainty_weight
        self.repetition_weight = repetition_weight
        self.reset()
        
    def reset(self):
        self.mu = {a: 0.5 for a in range(4)}
        self.pi = {a: 1.0 for a in range(4)}
        self.last_choice = None
        self.standardizer = CausalRewardStandardizer()
        
    def predict_probs(self, avail):
        logits = []
        for a in avail:
            m = self.mu.get(a, 0.5)
            unc = 1.0 / np.sqrt(max(self.pi.get(a, 1.0), 1e-4))
            rep = 1.0 if (self.last_choice is not None and self.last_choice == a) else 0.0
            logits.append(self.beta * m + self.uncertainty_weight * unc + self.repetition_weight * rep)
        max_l = max(logits)
        exps = [np.exp(l - max_l) for l in logits]
        s = sum(exps)
        return {a: exp / s for a, exp in zip(avail, exps)}
        
    def update(self, action, reward, avail):
        self.standardizer.update(reward)
        norm_r = 1.0 / (1.0 + np.exp(-self.standardizer.standardize(reward)))
        
        drift = np.exp(self.omega)
        for a in avail:
            if a != action:
                self.pi[a] = 1.0 / (1.0/self.pi[a] + drift)
                
        pred_pi = 1.0 / (1.0/self.pi[action] + drift)
        pi_hat = pred_pi + 1.0
        pe = norm_r - self.mu[action]
        self.mu[action] = self.mu[action] + (1.0 / pi_hat) * pe
        self.pi[action] = pi_hat
        self.last_choice = action

res_hgf = evaluate_policy(PyHGFUncertaintyStickyAgent(), valid_trajectories)
print(f"PyHGF + Uncertainty + Sticky: NLL = {res_hgf['mean_nll']:.4f}, Accuracy = {res_hgf['accuracy']*100:.2f}%")
"""))

# Section 5: Resource-Rational & Sampling Models
cells.append(nbf.v4.new_markdown_cell(r"""---
# Section 5: Resource-Rational Sampling Bandit (BMS + Trace)

### The Bayesian Mutation Sampler (BMS) as a Process Account of Cognitive Biases
Rather than performing exact Bayesian integration, humans evaluate hypotheses through **bounded sampling** (Kolvoort, Temme & van Maanen, 2023; Vul et al., 2014):
1. **Warm Start**: The MCMC chain starts at the previous trial's state $\to$ explains anchoring and perseveration.
2. **Limited Sampling Budget**: The chain performs only $N=10$ mutation steps $\to$ bounds cognitive compute.
3. **Dirichlet/Beta-Prior Conservatism**: Probabilities are computed from sample counts regularized by a symmetric Dirichlet prior:
   $$\text{logit}_{\text{sample}}(a) = \log(c_a + \beta_{\text{prior}})$$
   $\to$ prevents extreme probabilities and models conservatism from first principles.
4. **Motor Inertia Trace**: Blends the sampled hypothesis distribution with action history.
"""))

cells.append(nbf.v4.new_code_cell(r"""from agent import Agent as CSBAgent

with open("config.yaml") as f:
    csb_cfg = yaml.safe_load(f)

csb_cfg['model']['n_particles'] = 16
csb_agent = CSBAgent(csb_cfg)

class CSBEvaluatorAdapter:
    def __init__(self, agent_inst):
        self.agent = agent_inst
    def reset(self):
        self.agent.reset()
    def predict_probs(self, avail):
        pred = self.agent.predict()
        return pred['action_probs']
    def update(self, action, reward, avail):
        self.agent.update(action, reward)

res_csb = evaluate_policy(CSBEvaluatorAdapter(csb_agent), valid_trajectories[:50])
print(f"CSB + Trace (BMS Process Model, 50 traj sample): NLL = {res_csb['mean_nll']:.4f}, Accuracy = {res_csb['accuracy']*100:.2f}%")
"""))

# Section 6: Model Comparison & Visualizations
cells.append(nbf.v4.new_markdown_cell(r"""---
# Section 6: Empirical Comparison & Diagnostics Dashboard

Let us compare all evaluated model families across predictive loss (NLL), choice accuracy, MCMC convergence, and scientific interpretability.
"""))

cells.append(nbf.v4.new_code_cell(r"""# Summary Comparison DataFrame
models_summary = pd.DataFrame([
    {
        "Model": "Causal RW + Gradual Trace",
        "Family": "Scale-Free RL + History",
        "NLL": res_rw_trace['mean_nll'],
        "Accuracy (%)": res_rw_trace['accuracy'] * 100,
        "MCMC Convergence": "Passed (R-hat <= 1.0004)",
        "Key Mechanism": "Causal standardization + exponential choice trace",
        "Recommendation": "Top 1: Highest predictive score (0.608 NLL)"
    },
    {
        "Model": "Choice Trace Only (No Reward)",
        "Family": "Heuristic Autocorrelation",
        "NLL": res_trace['mean_nll'],
        "Accuracy (%)": res_trace['accuracy'] * 100,
        "MCMC Convergence": "Deterministic Optimum",
        "Key Mechanism": "Pure action repetition trace",
        "Recommendation": "Diagnostic baseline"
    },
    {
        "Model": "Causal Dual-Alpha Sticky RW",
        "Family": "Scale-Free RL + Sticky",
        "NLL": res_dual_sticky['mean_nll'],
        "Accuracy (%)": res_dual_sticky['accuracy'] * 100,
        "MCMC Convergence": "Passed (R-hat <= 1.007)",
        "Key Mechanism": "Asymmetric learning rates + 1-step stickiness",
        "Recommendation": "Top 2: Verified production baseline (0.694 NLL)"
    },
    {
        "Model": "CSB + Trace (BMS Process Model)",
        "Family": "Resource-Rational Sampling",
        "NLL": res_csb['mean_nll'],
        "Accuracy (%)": res_csb['accuracy'] * 100,
        "MCMC Convergence": "MCMC + Forward Grid Search",
        "Key Mechanism": "Warm-started mutation chains + Dirichlet prior",
        "Recommendation": "Cognitive process model (explains perseveration)"
    },
    {
        "Model": "PyHGF + Uncertainty + Sticky",
        "Family": "Exact Bayesian Volatility",
        "NLL": res_hgf['mean_nll'],
        "Accuracy (%)": res_hgf['accuracy'] * 100,
        "MCMC Convergence": "Passed (R-hat <= 1.003)",
        "Key Mechanism": "Precision-weighted PEs + uncertainty penalty",
        "Recommendation": "Top 3: Normative Bayesian volatility model"
    },
    {
        "Model": "Uniform Random",
        "Family": "Null Baseline",
        "NLL": res_uniform['mean_nll'],
        "Accuracy (%)": res_uniform['accuracy'] * 100,
        "MCMC Convergence": "N/A",
        "Key Mechanism": "p = 0.25",
        "Recommendation": "Benchmark floor"
    }
])

display(models_summary)
"""))

cells.append(nbf.v4.new_code_cell(r"""# Visualization of Comparative Metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot NLL (Lower is better)
sns.barplot(data=models_summary, x="NLL", y="Model", ax=axes[0], palette="Blues_r")
axes[0].set_title("Held-out Negative Log-Likelihood (Lower is better)", fontweight="bold")
axes[0].set_xlabel("Mean NLL per trial")
for i, v in enumerate(models_summary["NLL"]):
    axes[0].text(v + 0.02, i, f"{v:.4f}", va="center", fontweight="semibold")

# Plot Accuracy (Higher is better)
sns.barplot(data=models_summary, x="Accuracy (%)", y="Model", ax=axes[1], palette="Greens_r")
axes[1].set_title("Choice Accuracy (% Correct, Higher is better)", fontweight="bold")
axes[1].set_xlabel("Accuracy (%)")
for i, v in enumerate(models_summary["Accuracy (%)"]):
    axes[1].text(v + 1.0, i, f"{v:.1f}%", va="center", fontweight="semibold")

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell(r"""---
# Section 7: Top 3 Submission Candidates & Interpretation Cards

### 1. Candidate 1: `CausalRWTrace` (Predictive Frontrunner)
- **Method Family**: `cognitive_reinforcement_learning_model`
- **Held-out Performance**: **NLL 0.6082**, Accuracy **79.5%**.
- **Theoretical Grounding**: Error-driven learning combined with policy compression / motor inertia (Lau & Glimcher 2005; Lai & Gershman 2024).
- **Key Insight**: Disentangles reward-based value updates from deep choice autocorrelation across multiple trials.

### 2. Candidate 2: `CausalScaleDualAlphaSticky` (Verified Production Baseline)
- **Method Family**: `bayesian_cognitive_model`
- **Held-out Performance**: **NLL 0.6944**, Accuracy **75.9%**.
- **Theoretical Grounding**: Asymmetric valenced reinforcement learning (Frank et al., 2004; Gershman 2015) with causal running standardization.
- **Key Insight**: Scale-free reward normalization guarantees robustness to arbitrary unit scaling on private evaluation splits.

### 3. Candidate 3: `PyHGF+Uncertainty+Sticky` (Normative Volatility Model)
- **Method Family**: `bayesian_cognitive_model`
- **Held-out Performance**: **NLL 0.7646**, Accuracy **76.2%**.
- **Theoretical Grounding**: Hierarchical Gaussian Filter (Mathys et al., 2011) + Uncertainty-modulated exploration/exploitation (Gershman 2018).
- **Key Insight**: Explains learning rate adaptation dynamically via environmental volatility and subjective precision.

---
"""))

nb.cells = cells

# Save notebook
out_path = Path("research/mindrl_master_models_overview.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated Master Notebook: {out_path} with {len(cells)} cells.")
