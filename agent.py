"""
MindRL Challenge — Bayes'd Misfits submission agent.

This module defines the public ``Agent`` class the evaluator imports and calls.
It currently ships a **valid Win-Stay-Lose-Shift (WSLS) cognitive baseline** so
the bundle passes validation while the real model (HGF / HSSM / RL) is being
developed. Replace the body of the four methods with your own model while
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
    Win-Stay-Lose-Shift placeholder baseline.

    Favor repeating the last action after a sufficiently good reward; after a
    poor reward, favor switching away. Replace this with the Bayes'd Misfits
    model (HGF / HSSM-fitted RL / etc.) while keeping the four-method API.
    """

    def __init__(self, config: Mapping[str, Any] | Any | None = None) -> None:
        self._config = config or {}
        model = get_field(self._config, "model", default={}) or {}
        self._win_threshold = float(get_field(model, "win_threshold", 0.5))
        self._stay_probability = float(get_field(model, "stay_probability", 0.85))
        self._epsilon = float(get_field(model, "epsilon", 0.05))
        self._available_actions: list[Any] = []
        # Optional online cache; predict() uses history when present.
        self._last_action: Any = None
        self._last_reward: Any = None

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
        self._last_action = None
        self._last_reward = None

    def predict(self, history: Any) -> dict[str, dict[Any, float]]:
        actions = list(self._available_actions)
        if not actions:
            return {"action_probs": {}}

        hist = _as_history_list(history)
        if not hist:
            return {"action_probs": self._uniform_distribution(actions)}

        last = hist[-1]
        prev_action = get_field(last, "action", default=None)
        prev_reward = get_field(last, "reward", default=None)

        try:
            won = prev_reward is not None and float(prev_reward) >= self._win_threshold
        except (TypeError, ValueError):
            won = False

        if prev_action is None or prev_action not in actions:
            return {"action_probs": self._uniform_distribution(actions)}

        n = len(actions)
        if n == 1:
            return {"action_probs": normalize_probs({actions[0]: 1.0})}

        others = [a for a in actions if a != prev_action]
        m = len(others)
        if m == 0:
            return {"action_probs": self._uniform_distribution(actions)}

        if won:
            p_prev = self._stay_probability
            p_other = (1.0 - self._stay_probability) / m
        else:
            p_prev = 1.0 - self._stay_probability
            p_other = self._stay_probability / m

        raw: dict[Any, float] = {prev_action: p_prev}
        for a in others:
            raw[a] = p_other

        floored = {a: max(self._epsilon, raw.get(a, 0.0)) for a in actions}
        return {"action_probs": normalize_probs(floored)}

    def _uniform_distribution(self, actions: list[Any]) -> dict[Any, float]:
        n = len(actions)
        if n == 0:
            return {}
        base = 1.0 / n
        floored = {a: max(self._epsilon, base) for a in actions}
        return normalize_probs(floored)

    def update(self, action: Any, reward: Any, info: Any | None = None) -> None:
        self._last_action = action
        self._last_reward = reward