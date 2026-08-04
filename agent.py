"""
MindRL Challenge — Bayes'd Misfits submission agent.

This module defines the public ``Agent`` class the evaluator imports and calls.
It implements the fitted Dual-Alpha + Sticky reinforcement-learning model while
keeping the public API stable:

    class Agent:
        def __init__(self, config=None)
        def reset(self, context)
        def predict(self, history) -> dict      # {"action_probs": {action: prob}}
        def update(self, action, reward, info=None)

Hard invariants the evaluator enforces:
  * ``predict`` returns ``{"action_probs": {...}}`` (optionally ``"rt_ms"``).
  * Probabilities are non-negative and sum to 1.0.
  * Keys are the exact action labels from ``context["available_actions"]``.
  * Every valid action is a key; no extras, no missing.
  * ``history`` contains only PAST trials — never peek at the current outcome.
  * ``reset`` is called once per trajectory; ``__init__`` once overall.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or arbitrary object (e.g. dataclass)."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def normalize_probs(probs: dict[Any, float]) -> dict[Any, float]:
    """Return a copy of ``probs`` with non-negative values summing to 1."""
    if not probs:
        return {}
    clipped = {a: max(0.0, float(p)) for a, p in probs.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        n = len(clipped)
        u = 1.0 / n if n else 0.0
        return {a: u for a in clipped}
    return {a: p / total for a, p in clipped.items()}


def _as_history_list(history: Any) -> list[Any]:
    if history is None:
        return []
    if isinstance(history, (str, bytes)):
        return []
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        try:
            return list(history)
        except TypeError:
            return []
    return []


def _default_actions() -> list[Any]:
    return [1, 2, 3, 4]


class Agent:
    """
    Dual-Alpha + Sticky RL submission agent with optional resource-rational
    extensions inspired by Bruckner et al. (2025, Psychological Review).

    Maintains action values updated via positive/negative learning rates,
    applies a choice-stickiness perseveration bonus, and selects options
    via a softmax choice rule.

    When ``fatigue_rate`` > 0, a trial-varying resource level degrades the
    effective learning rate and amplifies stickiness over the session —
    modelling within-session cognitive fatigue.  When ``surprise_gain`` > 0,
    the learning rate is modulated by prediction-error magnitude, consistent
    with the paper's "criterion level of accuracy" stopping rule.  Both default
    to 0, recovering the original Dual-Alpha + Sticky model.
    """

    def __init__(self, config: Mapping[str, Any] | Any | None = None) -> None:
        self._config = config or {}
        model = get_field(self._config, "model", default={}) or {}
        
        # Load hyperparameters from config (defaulting to converged fit)
        self._rl_alpha_pos = float(get_field(model, "rl_alpha_pos", 0.380))
        self._rl_alpha_neg = float(get_field(model, "rl_alpha_neg", 0.763))
        self._sticky = float(get_field(model, "sticky", 0.126))
        self._beta = float(get_field(model, "beta", 8.331))
        self._initial_q = float(get_field(model, "initial_q", 0.5))
        self._reward_scale = float(get_field(model, "reward_scale", 1.0))
        if not math.isfinite(self._reward_scale) or self._reward_scale <= 0.0:
            raise ValueError("model.reward_scale must be a positive finite number")

        # --- Resource-rational extensions (Bruckner et al. 2025) -----------
        # All default to 0 → exact backward compatibility with the original model.
        self._fatigue_rate = float(get_field(model, "fatigue_rate", 0.0))
        self._sticky_gain = float(get_field(model, "sticky_gain", 0.0))
        self._surprise_gain = float(get_field(model, "surprise_gain", 0.0))

        self._available_actions: list[Any] = []
        self._q: dict[Any, float] = {}
        self._last_choice: Any = None
        self._history_len = 0

    def _resource_level(self, trial: int) -> float:
        """Trial-varying cognitive resource level rho_t in (0, 1].

        rho_t = 1 / (1 + fatigue_rate * t)
        At t=0, rho=1 (full resources).  As t grows, rho → 0 (depleted).
        When fatigue_rate == 0, rho is always 1 (no fatigue).
        """
        if self._fatigue_rate <= 0.0:
            return 1.0
        return 1.0 / (1.0 + self._fatigue_rate * trial)

    def _effective_alpha(self, pe: float, trial: int) -> float:
        """Compute the resource- and surprise-adjusted learning rate."""
        rho = self._resource_level(trial)
        base = self._rl_alpha_pos if pe >= 0.0 else self._rl_alpha_neg
        alpha = base * rho
        if self._surprise_gain > 0.0:
            alpha *= 1.0 + self._surprise_gain * abs(pe)
        return min(alpha, 0.9999)

    def _effective_sticky(self, trial: int) -> float:
        """Compute the resource-adjusted stickiness bonus."""
        rho = self._resource_level(trial)
        return self._sticky + self._sticky_gain * (1.0 - rho)

    def reset(self, context: Mapping[str, Any] | Any) -> None:
        actions = get_field(context, "available_actions", default=None)
        if actions is None:
            actions = []
        try:
            self._available_actions = list(actions) if actions is not None else []
        except TypeError:
            self._available_actions = []
        if not self._available_actions:
            self._available_actions = list(_default_actions())
            
        # Re-initialize action values and choice history
        self._q = {a: self._initial_q for a in self._available_actions}
        self._last_choice = None
        self._history_len = 0

    def _normalize_reward(self, reward: Any) -> float:
        """Convert evaluator rewards to the 0-1 scale used during fitting."""
        return float(reward) / self._reward_scale

    def predict(self, history: Any) -> dict[str, dict[Any, float]]:
        actions = list(self._available_actions)
        if not actions:
            return {"action_probs": {}}

        hist = _as_history_list(history)

        if len(hist) != self._history_len:
            # Reconstruct state from history (must mirror update() exactly)
            q = {a: self._initial_q for a in actions}
            last_choice = None
            trial_idx = 0
            
            for trial in hist:
                act = get_field(trial, "action", default=None)
                rew = get_field(trial, "reward", default=None)
                if act is not None and act in q and rew is not None:
                    try:
                        r = self._normalize_reward(rew)
                        pe = r - q[act]
                        alpha = self._effective_alpha(pe, trial_idx)
                        q[act] += alpha * pe
                        last_choice = act
                        trial_idx += 1
                    except (TypeError, ValueError):
                        pass
            self._q = q
            self._last_choice = last_choice
            self._history_len = len(hist)

        values = {}
        sticky_eff = self._effective_sticky(self._history_len)
        for a in actions:
            val = self._q[a]
            if self._last_choice is not None and a == self._last_choice:
                # Q-scale perseveration bonus; its logit contribution is beta * sticky_eff.
                val += sticky_eff
            values[a] = val

        # Numerically stable softmax (shift by max so exp never overflows)
        max_val = max(values.values())
        exp_vals = {}
        for a in actions:
            exp_vals[a] = math.exp(self._beta * (values[a] - max_val))

        # Tiny floor so no probability is exactly zero (avoids -inf log-loss)
        floored = {a: max(1e-5, exp_vals[a]) for a in actions}
        return {"action_probs": normalize_probs(floored)}

    def _uniform_distribution(self, actions: list[Any]) -> dict[Any, float]:
        n = len(actions)
        if n == 0:
            return {}
        base = 1.0 / n
        floored = {a: max(1e-5, base) for a in actions}
        return normalize_probs(floored)

    def update(self, action: Any, reward: Any, info: Any | None = None) -> None:
        if action not in self._q or reward is None:
            return
        try:
            r = self._normalize_reward(reward)
            pe = r - self._q[action]                       # prediction error
            alpha = self._effective_alpha(pe, self._history_len)
            self._q[action] += alpha * pe                 # the one learning line
            self._last_choice = action                     # sticky bonus uses this next step
            self._history_len += 1
        except (TypeError, ValueError):
            pass
