"""Behavioral and compliance tests for the submitted agent."""

from __future__ import annotations
import math
import numpy as np
import pytest
from agent import Agent


CONFIG = {
    "model": {
        "prior_beta": 1.134684,
        "temperature": 0.415717,
        "repetition_weight": 3.170215,
        "choice_trace_rate": 0.412275,
        "n_samples": 10,
        "n_particles": 32,
        "obs_sigma": 0.1,
        "forgetting": 0.98,
        "n_levels": 5,
        "seed": 2026,
    }
}
CONTEXT = {"available_actions": [0, 1, 2, 3]}
CHOICES = [0, 1, 0, 3, 2, 0, 1, 3, 2, 2]
REWARDS = [40.0, 70.0, 55.0, 90.0, 20.0, 62.0, 75.0, 30.0, 45.0, 80.0]


def _assert_distributions_close(left, right, tolerance=1e-12):
    assert left.keys() == right.keys()
    for action in left:
        assert left[action] == pytest.approx(
            right[action], abs=tolerance, rel=tolerance
        )


def test_predictions_are_invariant_to_positive_affine_reward_transformations():
    original = Agent(CONFIG)
    transformed = Agent(CONFIG)
    original.reset(CONTEXT)
    transformed.reset(CONTEXT)

    original_history = []
    transformed_history = []
    for choice, reward in zip(CHOICES, REWARDS, strict=True):
        _assert_distributions_close(
            original.predict(original_history)["action_probs"],
            transformed.predict(transformed_history)["action_probs"],
        )
        original.update(choice, reward)
        transformed.update(choice, 10.0 * reward + 500.0)
        original_history.append({"action": choice, "reward": reward})
        transformed_history.append(
            {"action": choice, "reward": 10.0 * reward + 500.0}
        )

    _assert_distributions_close(
        original.predict(original_history)["action_probs"],
        transformed.predict(transformed_history)["action_probs"],
    )


def test_incremental_updates_and_history_replay_are_identical():
    incremental = Agent(CONFIG)
    replay = Agent(CONFIG)
    incremental.reset(CONTEXT)
    replay.reset(CONTEXT)
    history = []

    for choice, reward in zip(CHOICES, REWARDS, strict=True):
        incremental.update(choice, reward)
        history.append({"action": choice, "reward": reward})

    _assert_distributions_close(
        incremental.predict(history)["action_probs"],
        replay.predict(history)["action_probs"],
    )
    assert incremental._reward_count == replay._reward_count == len(REWARDS)
    assert incremental._reward_mean == pytest.approx(replay._reward_mean)
    assert incremental._reward_m2 == pytest.approx(replay._reward_m2)


def test_agent_defaults_match_the_submitted_parameter_vector():
    configured = Agent(CONFIG)
    defaults = Agent()
    configured.reset(CONTEXT)
    defaults.reset(CONTEXT)
    history = []

    for choice, reward in zip(CHOICES, REWARDS, strict=True):
        _assert_distributions_close(
            configured.predict(history)["action_probs"],
            defaults.predict(history)["action_probs"],
        )
        configured.update(choice, reward)
        defaults.update(choice, reward)
        history.append({"action": choice, "reward": reward})


def test_reset_clears_values_choice_and_reward_statistics():
    agent = Agent(CONFIG)
    agent.reset(CONTEXT)
    for choice, reward in zip(CHOICES[:4], REWARDS[:4], strict=True):
        agent.update(choice, reward)
    assert agent._reward_count == 4
    assert agent._last_choice is not None

    agent.reset(CONTEXT)
    assert agent._reward_count == 0
    assert agent._reward_mean == 0.0
    assert agent._reward_m2 == 0.0
    assert agent._last_choice is None
    assert np.allclose(agent._choice_trace, 0.0)
    assert agent.predict([])["action_probs"] == {
        0: 0.25,
        1: 0.25,
        2: 0.25,
        3: 0.25,
    }


def test_first_or_constant_reward_has_no_value_scale_but_repetition_is_active():
    agent = Agent(CONFIG)
    agent.reset(CONTEXT)
    agent.update(2, 5000.0)
    probabilities = agent.predict([{"action": 2, "reward": 5000.0}])[
        "action_probs"
    ]
    assert probabilities[2] > max(
        probabilities[0], probabilities[1], probabilities[3]
    )
    for unchosen in (0, 1, 3):
        assert probabilities[unchosen] < 0.25

    for _ in range(3):
        agent.update(2, 5000.0)
    probabilities = agent.predict([{"action": 2, "reward": 5000.0}] * 4)[
        "action_probs"
    ]
    assert all(
        math.isfinite(probability) for probability in probabilities.values()
    )
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_missing_or_nonfinite_rewards_do_not_contaminate_scaler():
    agent = Agent(CONFIG)
    agent.reset(CONTEXT)
    for reward in (None, float("nan"), float("inf"), "not-a-number"):
        agent.update(1, reward)
    assert agent._reward_count == 0
    assert agent._last_choice == 1
    probabilities = agent.predict(
        [
            {"action": 1, "reward": reward}
            for reward in (None, float("nan"), float("inf"), "not-a-number")
        ]
    )["action_probs"]
    assert sum(probabilities.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("prior_beta", -0.1),
        ("temperature", -1.0),
        ("choice_trace_rate", 1.5),
        ("repetition_weight", float("nan")),
        ("n_samples", 0),
        ("n_particles", 0),
        ("n_levels", 0),
        ("n_levels", 1),
        ("obs_sigma", 0.0),
        ("forgetting", 0.0),
    ],
)
def test_invalid_parameters_are_rejected(field, value):
    config = {"model": dict(CONFIG["model"], **{field: value})}
    with pytest.raises(ValueError):
        Agent(config)
