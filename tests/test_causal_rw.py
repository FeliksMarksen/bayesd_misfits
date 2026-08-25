"""Tests for causally reward-standardized dual-alpha reinforcement learning."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bayesd_misfits.causal_rw import (
    NArmCausalScaleDualAlphaChoiceTrace,
    NArmCausalScaleDualAlphaNoHistory,
    NArmCausalScaleDualAlphaSticky,
    NArmCausalScaleSingleAlphaChoiceTrace,
    NArmCausalScaleSingleAlphaNoHistory,
    NArmCausalScaleSingleAlphaSticky,
)



CHOICES = np.asarray([0, 1, 0, 3, 2, 0, 1, 3, 2, 2], dtype=int)
REWARDS = np.asarray([40.0, 70.0, 55.0, 90.0, 20.0, 62.0, 75.0, 30.0, 45.0, 80.0])
PARAMS = {
    "rl_alpha_pos": 0.38,
    "rl_alpha_neg": 0.76,
    "beta": 2.5,
    "repetition_weight": 1.1,
}


def _python_logits(learner, state):
    computed = learner.compute_python(state, PARAMS, context={})
    return np.asarray([computed[f"logit{i}"] for i in range(4)])


def _jax_logits(learner, state, params=PARAMS):
    computed = learner.compute_jax(state, params, context={})
    return np.asarray([computed[f"logit{i}"] for i in range(4)])


def test_numpy_and_jax_states_and_logits_match():
    learner = NArmCausalScaleDualAlphaSticky(4)
    python_state = learner.init_state()
    jax_state = learner.init_jax_state()

    for choice, reward in zip(CHOICES, REWARDS, strict=True):
        np.testing.assert_allclose(
            _python_logits(learner, python_state),
            _jax_logits(learner, jax_state),
            rtol=2e-5,
            atol=2e-6,
        )
        context = {"choice": int(choice), "feedback": float(reward)}
        python_state = learner.update_python(python_state, PARAMS, context)
        jax_state = learner.update_jax(jax_state, PARAMS, context)

    np.testing.assert_allclose(
        python_state["q_values"], np.asarray(jax_state["q_values"]), rtol=2e-5
    )
    np.testing.assert_array_equal(
        python_state["action_seen"], np.asarray(jax_state["action_seen"])
    )
    for field in ("reward_count", "reward_mean", "reward_m2", "last_choice"):
        np.testing.assert_allclose(python_state[field], np.asarray(jax_state[field]))


def test_logits_are_invariant_to_positive_affine_reward_transformations():
    original = NArmCausalScaleDualAlphaSticky(4)
    transformed = NArmCausalScaleDualAlphaSticky(4)
    original_state = original.init_state()
    transformed_state = transformed.init_state()

    for choice, reward in zip(CHOICES, REWARDS, strict=True):
        np.testing.assert_allclose(
            _python_logits(original, original_state),
            _python_logits(transformed, transformed_state),
            rtol=1e-12,
            atol=1e-12,
        )
        original_state = original.update_python(
            original_state,
            PARAMS,
            {"choice": int(choice), "feedback": float(reward)},
        )
        transformed_state = transformed.update_python(
            transformed_state,
            PARAMS,
            {"choice": int(choice), "feedback": float(10.0 * reward + 500.0)},
        )

    np.testing.assert_allclose(
        _python_logits(original, original_state),
        _python_logits(transformed, transformed_state),
        rtol=1e-12,
        atol=1e-12,
    )


def test_zero_variance_rewards_are_safe_and_only_repetition_changes_logits():
    learner = NArmCausalScaleDualAlphaSticky(4)
    state = learner.init_state()
    for choice in (2, 2, 0, 1):
        state = learner.update_python(
            state,
            PARAMS,
            {"choice": choice, "feedback": 7.0},
        )
        logits = _python_logits(learner, state)
        expected = np.zeros(4)
        expected[choice] = PARAMS["repetition_weight"]
        np.testing.assert_allclose(logits, expected)
        assert np.isfinite(logits).all()


def test_reset_clears_all_trajectory_local_scaler_state():
    learner = NArmCausalScaleDualAlphaSticky(4)
    learner.reset()
    for choice, reward in zip(CHOICES[:4], REWARDS[:4], strict=True):
        learner.update(int(choice), float(reward), PARAMS)
    assert learner._state["reward_count"] == 4

    learner.reset()
    fresh = learner.init_state()
    for key in fresh:
        np.testing.assert_array_equal(learner._state[key], fresh[key])


def test_jax_parameter_gradients_are_finite():
    learner = NArmCausalScaleDualAlphaSticky(4)

    def objective(vector):
        params = {
            "rl_alpha_pos": vector[0],
            "rl_alpha_neg": vector[1],
            "beta": vector[2],
            "repetition_weight": vector[3],
        }
        state = learner.init_jax_state()
        total = 0.0
        for choice, reward in zip(CHOICES, REWARDS, strict=True):
            computed = learner.compute_jax(state, params, context={})
            logits = jnp.stack([computed[f"logit{i}"] for i in range(4)])
            total = total + jax.nn.log_softmax(logits)[choice]
            state = learner.update_jax(
                state,
                params,
                {"choice": jnp.asarray(choice), "feedback": jnp.asarray(reward)},
            )
        return total

    gradient = np.asarray(jax.grad(objective)(jnp.asarray([0.38, 0.76, 2.5, 1.1])))
    assert np.isfinite(gradient).all()


def test_causal_scale_dual_alpha_sticky_python_and_jax_match():
    learner = NArmCausalScaleDualAlphaSticky(4)
    python_state = learner.init_state()
    jax_state = learner.init_jax_state()

    for choice, reward in zip(CHOICES, REWARDS, strict=True):
        py_logits = learner.compute_python(python_state, PARAMS, context={})
        jx_logits = learner.compute_jax(jax_state, PARAMS, context={})
        for i in range(4):
            assert float(py_logits[f"logit{i}"]) == pytest.approx(
                float(jx_logits[f"logit{i}"]), rel=1e-5, abs=1e-5
            )

        context = {"choice": int(choice), "feedback": float(reward)}
        python_state = learner.update_python(python_state, PARAMS, context)
        jax_state = learner.update_jax(jax_state, PARAMS, context)

def _computed_logits(learner, state, params):
    computed = learner.compute_python(state, params, context={})
    return np.asarray([computed[f"logit{i}"] for i in range(4)])


def test_choice_trace_rate_one_recovers_immediate_repetition():
    immediate = NArmCausalScaleDualAlphaSticky(4)
    gradual = NArmCausalScaleDualAlphaChoiceTrace(4)
    immediate_state = immediate.init_state()
    gradual_state = gradual.init_state()
    trace_params = {**PARAMS, "choice_trace_rate": 1.0}

    for choice, reward in zip(CHOICES, REWARDS, strict=True):
        np.testing.assert_allclose(
            _computed_logits(immediate, immediate_state, PARAMS),
            _computed_logits(gradual, gradual_state, trace_params),
            rtol=1e-12,
            atol=1e-12,
        )
        context = {"choice": int(choice), "feedback": float(reward)}
        immediate_state = immediate.update_python(
            immediate_state, PARAMS, context
        )
        gradual_state = gradual.update_python(
            gradual_state, trace_params, context
        )


def test_gradual_choice_trace_retains_multiple_recent_actions():
    learner = NArmCausalScaleDualAlphaChoiceTrace(4)
    state = learner.init_state()
    params = {**PARAMS, "choice_trace_rate": 0.5}

    state = learner.update_python(
        state, params, {"choice": 0, "feedback": 10.0}
    )
    np.testing.assert_allclose(state["choice_trace"], [0.5, 0.0, 0.0, 0.0])
    state = learner.update_python(
        state, params, {"choice": 1, "feedback": 20.0}
    )
    np.testing.assert_allclose(state["choice_trace"], [0.25, 0.5, 0.0, 0.0])


def test_choice_trace_python_and_jax_match_and_are_scale_free():
    learner = NArmCausalScaleSingleAlphaChoiceTrace(4)
    transformed = NArmCausalScaleSingleAlphaChoiceTrace(4)
    params = {
        "rl_alpha": 0.45,
        "beta": 2.2,
        "repetition_weight": 0.8,
        "choice_trace_rate": 0.35,
    }
    python_state = learner.init_state()
    jax_state = learner.init_jax_state()
    transformed_state = transformed.init_state()

    for choice, reward in zip(CHOICES, REWARDS, strict=True):
        original_logits = _computed_logits(learner, python_state, params)
        np.testing.assert_allclose(
            original_logits,
            _jax_logits(learner, jax_state, params),
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            original_logits,
            _computed_logits(transformed, transformed_state, params),
            rtol=1e-12,
            atol=1e-12,
        )
        context = {"choice": int(choice), "feedback": float(reward)}
        python_state = learner.update_python(python_state, params, context)
        jax_state = learner.update_jax(jax_state, params, context)
        transformed_state = transformed.update_python(
            transformed_state,
            params,
            {"choice": int(choice), "feedback": float(7.0 * reward - 30.0)},
        )


def test_single_alpha_and_dual_alpha_match_when_alphas_are_equal():
    class_pairs = [
        (NArmCausalScaleSingleAlphaNoHistory, NArmCausalScaleDualAlphaNoHistory),
        (NArmCausalScaleSingleAlphaSticky, NArmCausalScaleDualAlphaSticky),
        (NArmCausalScaleSingleAlphaChoiceTrace, NArmCausalScaleDualAlphaChoiceTrace),
    ]
    for single_class, dual_class in class_pairs:
        single = single_class(4)
        dual = dual_class(4)
        single_state = single.init_state()
        dual_state = dual.init_state()
        single_params = {
            "rl_alpha": 0.4,
            "beta": 2.0,
            "repetition_weight": 0.7,
            "choice_trace_rate": 0.6,
        }
        dual_params = {
            "rl_alpha_pos": 0.4,
            "rl_alpha_neg": 0.4,
            "beta": 2.0,
            "repetition_weight": 0.7,
            "choice_trace_rate": 0.6,
        }
        single_params = {
            key: value
            for key, value in single_params.items()
            if key in single.free_params
        }
        dual_params = {
            key: value
            for key, value in dual_params.items()
            if key in dual.free_params
        }
        for choice, reward in zip(CHOICES, REWARDS, strict=True):
            np.testing.assert_allclose(
                _computed_logits(single, single_state, single_params),
                _computed_logits(dual, dual_state, dual_params),
                rtol=1e-12,
                atol=1e-12,
            )
            context = {"choice": int(choice), "feedback": float(reward)}
            single_state = single.update_python(
                single_state, single_params, context
            )
            dual_state = dual.update_python(dual_state, dual_params, context)
