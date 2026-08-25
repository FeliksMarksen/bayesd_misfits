"""Four-arm continuous generalized HGF built with the official pyhgf API.

This module is deliberately separate from :mod:`bayesd_misfits.hgf`, which
contains the project's earlier hand-written approximation.  Here pyhgf owns
the graph, prediction schedule, missing-observation semantics, and posterior
updates.

Each arm is an independent three-node branch::

    observed reward <- latent value <- latent log-volatility

Only the chosen arm is marked as observed on a trial.  The unchosen branches
continue their random walks, so their means remain unchanged while their
uncertainty increases.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from pyhgf.model import Network


@dataclass(frozen=True)
class ContinuousGHGFConfig:
    """Parameters defining a continuous generalized-HGF bandit network."""

    n_actions: int = 4
    sensory_precision: float = 20.0
    initial_value_mean: float = 0.5
    initial_value_precision: float = 4.0
    value_tonic_volatility: float = -4.0
    initial_volatility_mean: float = -1.0
    initial_volatility_precision: float = 1.0
    volatility_tonic_volatility: float = -4.0
    volatility_coupling: float = 1.0
    max_posterior_precision: float = 1e10

    def validate(self) -> None:
        if self.n_actions < 2:
            raise ValueError("n_actions must be at least 2")
        for name in (
            "sensory_precision",
            "initial_value_precision",
            "initial_volatility_precision",
            "max_posterior_precision",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        for name in (
            "initial_value_mean",
            "value_tonic_volatility",
            "initial_volatility_mean",
            "volatility_tonic_volatility",
            "volatility_coupling",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class GHGFReplay:
    """One-step-ahead beliefs and post-outcome states from a trajectory replay."""

    prior_means: np.ndarray
    prior_variances: np.ndarray
    posterior_means: np.ndarray
    posterior_variances: np.ndarray
    volatility_means: np.ndarray
    observed: np.ndarray
    action_probs: np.ndarray


def build_continuous_ghgf(
    config: ContinuousGHGFConfig | None = None,
) -> tuple[Network, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Build four independent continuous-HGF branches with ``pyhgf.Network``.

    Returns the network followed by the observation, value, and volatility node
    indices.  Keeping those indices explicit avoids relying on hidden ordering
    assumptions elsewhere in the adapter.
    """
    config = config or ContinuousGHGFConfig()
    config.validate()

    network = Network(
        volatility_updates="unbounded",
        max_posterior_precision=config.max_posterior_precision,
    )

    observation_nodes = tuple(range(config.n_actions))
    network.add_nodes(
        kind="continuous-state",
        n_nodes=config.n_actions,
        precision=config.sensory_precision,
        expected_precision=config.sensory_precision,
    )

    value_nodes: list[int] = []
    for action in range(config.n_actions):
        value_nodes.append(network.n_nodes)
        network.add_nodes(
            kind="continuous-state",
            mean=config.initial_value_mean,
            expected_mean=config.initial_value_mean,
            precision=config.initial_value_precision,
            expected_precision=config.initial_value_precision,
            tonic_volatility=config.value_tonic_volatility,
            value_children=action,
        )

    volatility_nodes: list[int] = []
    for value_node in value_nodes:
        volatility_nodes.append(network.n_nodes)
        network.add_nodes(
            kind="continuous-state",
            mean=config.initial_volatility_mean,
            expected_mean=config.initial_volatility_mean,
            precision=config.initial_volatility_precision,
            expected_precision=config.initial_volatility_precision,
            tonic_volatility=config.volatility_tonic_volatility,
            volatility_children=(
                [value_node],
                [config.volatility_coupling],
            ),
        )

    network.input_idxs = observation_nodes
    network.create_belief_propagation_fn()
    return network, observation_nodes, tuple(value_nodes), tuple(volatility_nodes)


def encode_partial_feedback(
    actions: Sequence[int] | np.ndarray,
    rewards: Sequence[float] | np.ndarray,
    n_actions: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode chosen rewards as a pyhgf input matrix and observation mask."""
    action_array = np.asarray(actions)
    reward_array = np.asarray(rewards, dtype=np.float64)
    if action_array.ndim != 1 or reward_array.ndim != 1:
        raise ValueError("actions and rewards must be one-dimensional")
    if len(action_array) != len(reward_array):
        raise ValueError("actions and rewards must have the same length")
    if len(action_array) == 0:
        raise ValueError("actions and rewards must not be empty")
    if not np.issubdtype(action_array.dtype, np.integer):
        if not np.all(np.equal(action_array, np.floor(action_array))):
            raise ValueError("actions must be integer indices")
    action_array = action_array.astype(np.int64)
    if np.any((action_array < 0) | (action_array >= n_actions)):
        raise ValueError(f"actions must be in [0, {n_actions - 1}]")
    if not np.isfinite(reward_array).all():
        raise ValueError("rewards must be finite")
    if np.any((reward_array < 0.0) | (reward_array > 1.0)):
        raise ValueError("rewards must be normalized to [0, 1]")

    values = np.zeros((len(action_array), n_actions), dtype=np.float64)
    observed = np.zeros((len(action_array), n_actions), dtype=np.int32)
    rows = np.arange(len(action_array))
    values[rows, action_array] = reward_array
    observed[rows, action_array] = 1
    return values, observed


def softmax_choice_probabilities(
    values: np.ndarray,
    actions: Sequence[int] | np.ndarray,
    *,
    beta: float,
    sticky: float = 0.0,
) -> np.ndarray:
    """Convert pre-outcome value beliefs to trial-wise action probabilities."""
    values = np.asarray(values, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    if values.ndim != 2 or len(actions) != values.shape[0]:
        raise ValueError("values must be trial-by-action and align with actions")
    if np.any((actions < 0) | (actions >= values.shape[1])):
        raise ValueError("actions contain an invalid action index")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be a non-negative finite number")
    if not np.isfinite(sticky):
        raise ValueError("sticky must be finite")

    decision_values = values.copy()
    if len(actions) > 1:
        rows = np.arange(1, len(actions))
        decision_values[rows, actions[:-1]] += sticky
    logits = beta * decision_values
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def replay_continuous_ghgf(
    actions: Sequence[int] | np.ndarray,
    rewards: Sequence[float] | np.ndarray,
    *,
    config: ContinuousGHGFConfig | None = None,
    beta: float = 5.0,
    sticky: float = 0.0,
) -> GHGFReplay:
    """Replay a partial-feedback bandit trajectory through the official filter."""
    config = config or ContinuousGHGFConfig()
    values, observed = encode_partial_feedback(
        actions, rewards, n_actions=config.n_actions
    )
    network, _, value_nodes, volatility_nodes = build_continuous_ghgf(config)
    network.input_data(input_data=values, observed=observed)

    prior_means = np.column_stack([
        np.asarray(network.node_trajectories[node]["expected_mean"], dtype=float)
        for node in value_nodes
    ])
    prior_variances = np.column_stack([
        1.0
        / np.asarray(
            network.node_trajectories[node]["expected_precision"], dtype=float
        )
        for node in value_nodes
    ])
    posterior_means = np.column_stack([
        np.asarray(network.node_trajectories[node]["mean"], dtype=float)
        for node in value_nodes
    ])
    posterior_variances = np.column_stack([
        1.0
        / np.asarray(network.node_trajectories[node]["precision"], dtype=float)
        for node in value_nodes
    ])
    volatility_means = np.column_stack([
        np.asarray(network.node_trajectories[node]["mean"], dtype=float)
        for node in volatility_nodes
    ])
    action_probs = softmax_choice_probabilities(
        prior_means,
        actions,
        beta=beta,
        sticky=sticky,
    )
    return GHGFReplay(
        prior_means=prior_means,
        prior_variances=prior_variances,
        posterior_means=posterior_means,
        posterior_variances=posterior_variances,
        volatility_means=volatility_means,
        observed=observed,
        action_probs=action_probs,
    )


class OnlineContinuousGHGF:
    """Small stateful adapter for checking online and batch pyhgf parity."""

    def __init__(self, config: ContinuousGHGFConfig | None = None) -> None:
        self.config = config or ContinuousGHGFConfig()
        self.reset()

    def reset(self) -> None:
        (
            self.network,
            self.observation_nodes,
            self.value_nodes,
            self.volatility_nodes,
        ) = build_continuous_ghgf(self.config)
        self.attributes = deepcopy(self.network.attributes)

    def predict(self, time_step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Return next-trial prior means and variances without changing state."""
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step must be a positive finite number")
        predicted: dict[int, dict[str, Any]] = deepcopy(self.attributes)
        predicted[-1]["time_step"] = jnp.asarray(time_step)
        assert self.network.update_sequence is not None
        for node_idx, update_fn in self.network.update_sequence.prediction_steps:
            predicted = update_fn(
                attributes=predicted,
                node_idx=node_idx,
                edges=self.network.edges,
            )
        means = np.asarray(
            [predicted[node]["expected_mean"] for node in self.value_nodes],
            dtype=float,
        )
        variances = np.asarray(
            [1.0 / predicted[node]["expected_precision"] for node in self.value_nodes],
            dtype=float,
        )
        return means, variances

    def update(self, action: int, reward: float, time_step: float = 1.0) -> None:
        """Apply one chosen-arm reward observation using pyhgf belief propagation."""
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step must be a positive finite number")
        values, observed = encode_partial_feedback(
            [action], [reward], n_actions=self.config.n_actions
        )
        values_tuple = tuple(
            jnp.asarray(values[:, i]) for i in range(self.config.n_actions)
        )
        observed_tuple = tuple(
            jnp.asarray(observed[:, i]).squeeze()
            for i in range(self.config.n_actions)
        )
        assert self.network.scan_fn is not None
        self.attributes, _ = self.network.scan_fn(
            self.attributes,
            (
                values_tuple,
                observed_tuple,
                jnp.asarray(time_step),
                None,
            ),
        )


class NArmPyHGF:
    """JAX-only official-pyhgf learner implementing the ssms RL protocol.

    The HGF's value-level tonic volatility is the only fitted perceptual
    parameter. Observation precision, volatility coupling, and the higher
    level's own tonic volatility remain fixed in ``config``. This deliberately
    small parameterization is a safer first identification target than fitting
    every HGF quantity simultaneously.

    The learner emits one-step-ahead latent value means. The decision module
    supplies inverse temperature separately, as it does for the project's RW
    learners.
    """

    def __init__(
        self,
        n_actions: int = 4,
        *,
        config: ContinuousGHGFConfig | None = None,
        feedback_field: str = "feedback",
    ) -> None:
        if config is None:
            config = ContinuousGHGFConfig(n_actions=n_actions)
        elif config.n_actions != n_actions:
            raise ValueError(
                "config.n_actions must match n_actions: "
                f"{config.n_actions} != {n_actions}"
            )
        config.validate()
        self._n_actions = n_actions
        self._feedback_field = feedback_field
        self.config = config
        (
            self._network,
            self._observation_nodes,
            self._value_nodes,
            self._volatility_nodes,
        ) = build_continuous_ghgf(config)

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def computed_params(self) -> list[str]:
        return [f"q{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        return ["ghgf_omega"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {"ghgf_omega": (-8.0, 0.0)}

    @property
    def default_params(self) -> dict[str, float]:
        return {"ghgf_omega": self.config.value_tonic_volatility}

    @property
    def available_backends(self) -> tuple[str, ...]:
        # Official pyhgf belief propagation is written in JAX. Exposing a fake
        # NumPy backend would create a second implementation to keep in sync.
        return ("jax",)

    @property
    def supports_gradient(self) -> bool:
        return True

    @property
    def required_context_fields(self) -> list[str]:
        return ["choice", self._feedback_field]

    def _set_fitted_parameters(self, attributes, params):
        # tree_map recreates the dict/list containers without copying immutable
        # JAX leaves, making the following dictionary assignments trace-safe.
        updated = jax.tree_util.tree_map(lambda leaf: leaf, attributes)
        omega = jnp.asarray(params["ghgf_omega"])
        for node_idx in self._value_nodes:
            updated[node_idx]["tonic_volatility"] = omega
        return updated

    def init_jax_state(self) -> dict[str, Any]:
        attributes = self._set_fitted_parameters(
            self._network.attributes,
            self.default_params,
        )
        return {
            "attributes": attributes,
            "last_choice": jnp.asarray(-1, dtype=jnp.int32),
        }

    def _predict_attributes(self, state, params):
        """Run the non-mutating pre-outcome pyhgf prediction pass."""
        predicted = self._set_fitted_parameters(state["attributes"], params)
        predicted[-1]["time_step"] = jnp.asarray(1.0)
        assert self._network.update_sequence is not None
        for node_idx, update_fn in self._network.update_sequence.prediction_steps:
            predicted = update_fn(
                attributes=predicted,
                node_idx=node_idx,
                edges=self._network.edges,
            )
        return predicted

    def predict_moments_jax(self, state, params):
        """Return pre-outcome value means and standard deviations."""
        predicted = self._predict_attributes(state, params)
        means = jnp.stack([
            predicted[node_idx]["expected_mean"] for node_idx in self._value_nodes
        ])
        standard_deviations = jnp.sqrt(jnp.stack([
            1.0 / predicted[node_idx]["expected_precision"]
            for node_idx in self._value_nodes
        ]))
        return means, standard_deviations

    def compute_jax(self, state, params, context):
        """Return pre-outcome expected value means for the decision module."""
        means, _ = self.predict_moments_jax(state, params)
        return {
            f"q{action}": means[action] for action in range(self._n_actions)
        }

    def update_jax(self, state, params, context):
        """Update all branches, observing only the reward for the chosen arm."""
        choice = jnp.asarray(context["choice"], dtype=jnp.int32)
        feedback = jnp.asarray(context[self._feedback_field])
        attributes = self._set_fitted_parameters(state["attributes"], params)
        values = tuple(
            jnp.expand_dims(
                jnp.where(choice == action, feedback, jnp.asarray(0.0)),
                axis=0,
            )
            for action in range(self._n_actions)
        )
        observed = tuple(
            jnp.asarray(choice == action, dtype=jnp.int32)
            for action in range(self._n_actions)
        )
        assert self._network.scan_fn is not None
        attributes, _ = self._network.scan_fn(
            attributes,
            (values, observed, jnp.asarray(1.0), None),
        )
        return {"attributes": attributes, "last_choice": choice}


class NArmPyHGFSticky(NArmPyHGF):
    """Nested pyhgf learner adding one free choice-perseveration parameter."""

    @property
    def free_params(self) -> list[str]:
        return ["ghgf_omega", "sticky"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {**super().param_bounds, "sticky": (-3.0, 3.0)}

    @property
    def default_params(self) -> dict[str, float]:
        return {**super().default_params, "sticky": 0.0}

    def compute_jax(self, state, params, context):
        values = super().compute_jax(state, params, context)
        last_choice = state["last_choice"]
        sticky = jnp.asarray(params["sticky"])
        return {
            f"q{action}": values[f"q{action}"]
            + jnp.where(last_choice == action, sticky, 0.0)
            for action in range(self._n_actions)
        }


class NArmPyHGFUncertainty(NArmPyHGF):
    """Pyhgf learner emitting separately weighted categorical choice logits.

    The decision score for each action is the sum of a value term and a signed
    uncertainty term. ``uncertainty_weight > 0`` expresses directed exploration;
    a negative value expresses uncertainty avoidance. The supplied HSSM decision
    process must consume the resulting ``logit0`` ... ``logitN`` values directly,
    without applying another inverse temperature.
    """

    @property
    def computed_params(self) -> list[str]:
        return [f"logit{i}" for i in range(self._n_actions)]

    @property
    def free_params(self) -> list[str]:
        return ["ghgf_omega", "beta", "uncertainty_weight"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            **super().param_bounds,
            "beta": (0.0, 15.0),
            "uncertainty_weight": (-10.0, 10.0),
        }

    @property
    def default_params(self) -> dict[str, float]:
        return {
            **super().default_params,
            "beta": 5.0,
            "uncertainty_weight": 0.0,
        }

    def compute_jax(self, state, params, context):
        means, standard_deviations = self.predict_moments_jax(state, params)
        logits = (
            jnp.asarray(params["beta"]) * means
            + jnp.asarray(params["uncertainty_weight"]) * standard_deviations
        )
        return {
            f"logit{action}": logits[action]
            for action in range(self._n_actions)
        }


class NArmPyHGFUncertaintySticky(NArmPyHGFUncertainty):
    """Uncertainty-aware pyhgf learner with independent repetition logits."""

    @property
    def free_params(self) -> list[str]:
        return [
            "ghgf_omega",
            "beta",
            "uncertainty_weight",
            "repetition_weight",
        ]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {**super().param_bounds, "repetition_weight": (-5.0, 5.0)}

    @property
    def default_params(self) -> dict[str, float]:
        return {**super().default_params, "repetition_weight": 0.0}

    def compute_jax(self, state, params, context):
        logits = super().compute_jax(state, params, context)
        last_choice = state["last_choice"]
        repetition_weight = jnp.asarray(params["repetition_weight"])
        return {
            f"logit{action}": logits[f"logit{action}"]
            + jnp.where(last_choice == action, repetition_weight, 0.0)
            for action in range(self._n_actions)
        }
