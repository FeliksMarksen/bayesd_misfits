"""MindRL Challenge — Bayes'd Misfits submission agent.

Scale-free Causal Sampling Bandit with Bayesian Mutation Sampling (BMS),
Beta-prior conservatism, and a gradual choice trace.

The evaluator calls ``predict`` before revealing the current trial's outcome
and calls ``update`` afterwards.

Cognitive Architecture:
1. Hypothesis Space & Resource-Rational Sampling:
   The agent represents hypotheses about the latent values of the four bandit arms
   as discrete standardized z-score levels {-2, -1, 0, 1, 2}.  Rather than exact
   point estimation or infinite-precision smoothing, it runs an ensemble of M=32
2. Dirichlet-Prior Conservatism:
   Choice probabilities are regularized by a symmetric Dirichlet(prior_beta)
   prior on sample frequencies: log(counts + prior_beta).  This categorical
   generalization of the BMS Beta prior prevents spurious extremes and models
   human conservatism from first principles (Kolvoort et al., 2023).
   A decaying choice trace captures motor inertia across consecutive trials:
   trace_t = trace_{t-1} + choice_trace_rate * (one_hot(action) - trace_{t-1}).
4. Scale-Invariance:
   Rewards remain in their raw units and are standardized causally using only
   rewards already revealed in the current trajectory (Welford's one-pass algorithm).
   Adding a constant to all rewards or multiplying them by a positive constant
   does not change the predicted action probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or arbitrary object such as a dataclass."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def normalize_probs(probs: dict[Any, float]) -> dict[Any, float]:
    """Return a copy of ``probs`` with non-negative values summing to 1."""
    if not probs:
        return {}
    clipped = {action: max(0.0, float(prob)) for action, prob in probs.items()}
    total = sum(clipped.values())
    if not math.isfinite(total) or total <= 0.0:
        uniform = 1.0 / len(clipped)
        return {action: uniform for action in clipped}
    return {action: prob / total for action, prob in clipped.items()}


def _default_actions() -> list[int]:
    return [0, 1, 2, 3]


class Agent:
    """Scale-free Causal Sampling Bandit with Choice Trace."""

    def __init__(self, config: Mapping[str, Any] | Any | None = None) -> None:
        self._config = config or {}
        model = get_field(self._config, "model", default={}) or {}

        # Fitted population parameters
        self._prior_beta = float(get_field(model, "prior_beta", 1.134684))
        self._temperature = float(get_field(model, "temperature", 0.415717))
        self._repetition_weight = float(
            get_field(model, "repetition_weight", 3.170215)
        )
        self._choice_trace_rate = float(
            get_field(model, "choice_trace_rate", 0.412275)
        )

        # Process hyperparameters
        self._n_samples = int(get_field(model, "n_samples", 10))
        self._n_particles = int(get_field(model, "n_particles", 32))
        self._obs_sigma = float(get_field(model, "obs_sigma", 0.1))
        self._forgetting = float(get_field(model, "forgetting", 0.98))
        self._n_levels = int(get_field(model, "n_levels", 5))

        # Parameter validation
        if not math.isfinite(self._prior_beta) or self._prior_beta < 0.0:
            raise ValueError("model.prior_beta must be finite and non-negative")
        if not math.isfinite(self._temperature) or self._temperature <= 0.0:
            raise ValueError("model.temperature must be finite and positive")
        if not math.isfinite(self._repetition_weight):
            raise ValueError("model.repetition_weight must be finite")
        if self._n_levels < 2:
            raise ValueError("model.n_levels must be at least 2")
        if (
            not math.isfinite(self._choice_trace_rate)
            or not 0.0 <= self._choice_trace_rate <= 1.0
        ):
            raise ValueError("model.choice_trace_rate must be in [0, 1]")
        if self._n_samples < 1:
            raise ValueError("model.n_samples must be at least 1")
        if self._n_particles < 1:
            raise ValueError("model.n_particles must be at least 1")
        if not math.isfinite(self._obs_sigma) or self._obs_sigma <= 0.0:
            raise ValueError("model.obs_sigma must be positive")
        if not 0.0 < self._forgetting <= 1.0:
            raise ValueError("model.forgetting must be in (0, 1]")

        self._level_values = np.linspace(
            -2.0, 2.0, self._n_levels, dtype=np.float64
        )
        seed = get_field(model, "seed", 2026)
        self._rng = np.random.default_rng(seed)

        self._available_actions: list[Any] = []
        self._action_to_idx: dict[Any, int] = {}
        self._idx_to_action: dict[int, Any] = {}

        # Trajectory-local dynamic state
        self._levels: np.ndarray | None = None
        self._counts: np.ndarray | None = None
        self._choice_trace: np.ndarray | None = None
        self._n: np.ndarray | None = None
        self._S: np.ndarray | None = None
        self._SS: np.ndarray | None = None
        self._last_choice: Any = None
        self._history_len = 0
        self._reward_count = 0
        self._reward_mean = 0.0
        self._reward_m2 = 0.0

    def _clear_trajectory_state(self) -> None:
        n_actions = len(self._available_actions)
        mid = self._n_levels // 2
        self._levels = np.full(
            (self._n_particles, n_actions), mid, dtype=np.int64
        )
        self._counts = np.ones(n_actions, dtype=np.float64)
        self._choice_trace = np.zeros(n_actions, dtype=np.float64)
        self._n = np.zeros(n_actions, dtype=np.float64)
        self._S = np.zeros(n_actions, dtype=np.float64)
        self._SS = np.zeros(n_actions, dtype=np.float64)
        self._last_choice = None
        self._history_len = 0
        self._reward_count = 0
        self._reward_mean = 0.0
        self._reward_m2 = 0.0

    def reset(self, context: Mapping[str, Any] | Any | None = None) -> None:
        actions = get_field(context, "available_actions", default=None)
        try:
            self._available_actions = (
                list(actions) if actions is not None else []
            )
        except TypeError:
            self._available_actions = []
        if not self._available_actions:
            self._available_actions = _default_actions()

        self._action_to_idx = {
            act: i for i, act in enumerate(self._available_actions)
        }
        self._idx_to_action = {
            i: act for i, act in enumerate(self._available_actions)
        }
        self._clear_trajectory_state()

    def _log_likelihood(self, levels: np.ndarray) -> np.ndarray:
        safe_levels = np.clip(levels, 0, self._n_levels - 1)
        v = self._level_values[safe_levels]
        resid = self._SS - 2.0 * v * self._S + self._n * v * v
        masked = np.where(self._n > 0.0, resid, 0.0)
        return -0.5 * np.sum(masked, axis=-1) / (self._obs_sigma**2)

    def _mutation_chain_ensemble(self) -> tuple[np.ndarray, np.ndarray]:
        n_actions = len(self._available_actions)
        levels = self._levels.copy()
        counts = np.zeros((self._n_particles, n_actions), dtype=np.float64)
        max_level = levels.max(axis=1, keepdims=True)
        is_best = levels == max_level
        counts += is_best / is_best.sum(axis=1, keepdims=True)

        curr_ll = self._log_likelihood(levels)
        idx = np.arange(self._n_particles)
        for _ in range(self._n_samples - 1):
            arms = self._rng.integers(0, n_actions, size=self._n_particles)
            directions = self._rng.choice([-1, 1], size=self._n_particles)
            proposed = levels.copy()
            proposed[idx, arms] += directions
            valid = (proposed[idx, arms] >= 0) & (
                proposed[idx, arms] < self._n_levels
            )

            prop_ll = self._log_likelihood(proposed)
            log_ratio = prop_ll - curr_ll
            accept = valid & (
                self._rng.random(size=self._n_particles)
                < np.exp(np.minimum(0.0, log_ratio))
            )

            levels = np.where(accept[:, None], proposed, levels)
            curr_ll = np.where(accept, prop_ll, curr_ll)

            max_level = levels.max(axis=1, keepdims=True)
            is_best = levels == max_level
            counts += is_best / is_best.sum(axis=1, keepdims=True)

        return levels, counts.mean(axis=0)

    def _apply_observation(self, action: Any, reward: Any) -> None:
        if action not in self._action_to_idx:
            return
        idx = self._action_to_idx[action]
        self._last_choice = action

        # 1. Update gradual choice trace
        n_actions = len(self._available_actions)
        target = np.zeros(n_actions, dtype=np.float64)
        target[idx] = 1.0
        self._choice_trace = self._choice_trace + self._choice_trace_rate * (
            target - self._choice_trace
        )

        if reward is None:
            self._history_len += 1
            return
        try:
            observed_reward = float(reward)
        except (TypeError, ValueError):
            self._history_len += 1
            return
        if not math.isfinite(observed_reward):
            self._history_len += 1
            return

        # 2. Causal standardization using pre-reward running statistics
        if self._reward_count >= 2 and self._reward_m2 > 0.0:
            sd = math.sqrt(self._reward_m2 / (self._reward_count - 1))
            z = (
                (observed_reward - self._reward_mean) / sd
                if (math.isfinite(sd) and sd > 0.0)
                else 0.0
            )
        else:
            z = 0.0

        # 3. Exponential decay of sufficient statistics
        self._n *= self._forgetting
        self._S *= self._forgetting
        self._SS *= self._forgetting
        self._n[idx] += 1.0
        self._S[idx] += z
        self._SS[idx] += z * z

        # 4. Welford running update
        new_count = self._reward_count + 1
        delta = observed_reward - self._reward_mean
        new_mean = self._reward_mean + delta / new_count
        self._reward_m2 = max(
            self._reward_m2 + delta * (observed_reward - new_mean), 0.0
        )
        self._reward_mean = new_mean
        self._reward_count = new_count

        # 5. Run mutation chain ensemble
        self._levels, self._counts = self._mutation_chain_ensemble()
        self._history_len += 1

    def update(
        self,
        action: Any,
        reward: Any,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del info, kwargs
        self._apply_observation(action, reward)
    def predict(self, history: Any = None) -> dict[str, Any]:
        if not self._available_actions:
            self._available_actions = _default_actions()
            self._action_to_idx = {
                act: i for i, act in enumerate(self._available_actions)
            }
            self._idx_to_action = {
                i: act for i, act in enumerate(self._available_actions)
            }
            self._clear_trajectory_state()

        if history is not None:
            hist_list = list(history) if isinstance(history, Sequence) else []
            if len(hist_list) > self._history_len:
                for item in hist_list[self._history_len:]:
                    act = get_field(item, "action", default=None)
                    rew = get_field(item, "reward", default=None)
                    self._apply_observation(act, rew)

        n_actions = len(self._available_actions)
        sampling_logits = self._temperature * np.log(
            self._counts + self._prior_beta + 1e-12
        )
        trace_logits = self._repetition_weight * self._choice_trace
        logits = sampling_logits + trace_logits
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        prob_dict = {
            self._idx_to_action[i]: float(probs[i]) for i in range(n_actions)
        }
        return {"action_probs": prob_dict}

