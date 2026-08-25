"""Tests for reward-free causal diagnostic learners."""

from __future__ import annotations

import numpy as np

from bayesd_misfits.choice_baselines import (
    NArmChoiceTrace,
    NArmPreviousChoice,
    NArmRunningChoiceFrequency,
)


CHOICES = [0, 1, 1, 3, 2]


def _logits(learner, state, params):
    computed = learner.compute_python(state, params, context={})
    return np.asarray([computed[f"logit{i}"] for i in range(4)])


def test_previous_choice_is_trace_rate_one():
    previous = NArmPreviousChoice(4)
    trace = NArmChoiceTrace(4)
    previous_state = previous.init_state()
    trace_state = trace.init_state()
    previous_params = {"repetition_weight": 1.7}
    trace_params = {"repetition_weight": 1.7, "choice_trace_rate": 1.0}

    for choice in CHOICES:
        np.testing.assert_allclose(
            _logits(previous, previous_state, previous_params),
            _logits(trace, trace_state, trace_params),
        )
        context = {"choice": choice, "feedback": 999.0}
        previous_state = previous.update_python(
            previous_state, previous_params, context
        )
        trace_state = trace.update_python(trace_state, trace_params, context)


def test_choice_trace_ignores_reward_values():
    first = NArmChoiceTrace(4)
    second = NArmChoiceTrace(4)
    first_state = first.init_state()
    second_state = second.init_state()
    params = {"repetition_weight": 2.0, "choice_trace_rate": 0.3}

    for index, choice in enumerate(CHOICES):
        np.testing.assert_allclose(
            _logits(first, first_state, params),
            _logits(second, second_state, params),
        )
        first_state = first.update_python(
            first_state,
            params,
            {"choice": choice, "feedback": float(index)},
        )
        second_state = second.update_python(
            second_state,
            params,
            {"choice": choice, "feedback": float(1000 * index - 50)},
        )


def test_running_frequency_is_uniform_then_updates_only_revealed_choice():
    learner = NArmRunningChoiceFrequency(4, prior_count=1.0)
    state = learner.init_state()
    np.testing.assert_allclose(np.exp(_logits(learner, state, {})), np.ones(4))

    state = learner.update_python(state, {}, {"choice": 2, "feedback": -100.0})
    probabilities = np.exp(_logits(learner, state, {}))
    probabilities /= probabilities.sum()
    np.testing.assert_allclose(probabilities, [0.2, 0.2, 0.4, 0.2])


def test_numpy_and_jax_choice_trace_match():
    learner = NArmChoiceTrace(4)
    python_state = learner.init_state()
    jax_state = learner.init_jax_state()
    params = {"repetition_weight": 1.3, "choice_trace_rate": 0.4}

    for choice in CHOICES:
        computed = learner.compute_jax(jax_state, params, context={})
        jax_logits = np.asarray([computed[f"logit{i}"] for i in range(4)])
        np.testing.assert_allclose(
            _logits(learner, python_state, params), jax_logits, rtol=2e-6
        )
        context = {"choice": choice, "feedback": 0.0}
        python_state = learner.update_python(python_state, params, context)
        jax_state = learner.update_jax(jax_state, params, context)

    np.testing.assert_allclose(
        python_state["choice_trace"], np.asarray(jax_state["choice_trace"]), rtol=2e-6
    )

