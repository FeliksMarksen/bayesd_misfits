"""Regression tests for the custom N-arm HGF learners."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from bayesd_misfits.hgf import NArmHGF, NArmHGFDriftLearner, NArmHGFSticky


class TestNArmHGF(unittest.TestCase):
    choices = np.asarray([0, 1, 0, 3, 2, 0, 1, 3] * 4)
    rewards = np.asarray([0.0, 1.0, 0.2, 0.8, 0.5, 0.1, 0.9, 0.4] * 4)

    def _run_both(self, kappa: float):
        learner = NArmHGF(4)
        numpy_state = learner.init_state()
        jax_state = learner.init_jax_state()
        params = {"omega": -2.3, "kappa": kappa}
        for choice, reward in zip(self.choices, self.rewards, strict=True):
            context = {"choice": int(choice), "feedback": float(reward)}
            numpy_state = learner.update_python(numpy_state, params, context)
            jax_state = learner.update_jax(jax_state, params, context)
        return numpy_state, jax_state

    def test_numpy_and_jax_updates_match(self):
        for kappa in (0.0, 0.001, 0.7, 2.5):
            with self.subTest(kappa=kappa):
                numpy_state, jax_state = self._run_both(kappa)
                for field in ("mu1", "sigma1", "mu2", "sigma2"):
                    np.testing.assert_allclose(
                        numpy_state[field],
                        np.asarray(jax_state[field]),
                        rtol=2e-5,
                        atol=2e-6,
                    )

    def test_state_stays_finite_with_positive_variances(self):
        for kappa in (0.0, 0.001, 0.7, 2.5, 4.0):
            with self.subTest(kappa=kappa):
                state, _ = self._run_both(kappa)
                for value in state.values():
                    self.assertTrue(np.isfinite(value).all())
                self.assertTrue((state["sigma1"] > 0.0).all())
                self.assertTrue((state["sigma2"] > 0.0).all())

    def test_jax_gradients_are_finite(self):
        def objective(parameters):
            learner = NArmHGF(4)
            state = learner.init_jax_state()
            total = 0.0
            params = {"omega": parameters[0], "kappa": parameters[1]}
            for choice, reward in zip(self.choices, self.rewards, strict=True):
                state = learner.update_jax(
                    state,
                    params,
                    {
                        "choice": jnp.asarray(choice),
                        "feedback": jnp.asarray(reward),
                    },
                )
                total = total + jnp.sum(state["mu1"]) + 0.01 * jnp.sum(state["mu2"])
            return total

        for parameters in (
            jnp.asarray([-2.3, 0.001]),
            jnp.asarray([-2.3, 0.7]),
            jnp.asarray([-6.0, 3.9]),
        ):
            with self.subTest(parameters=np.asarray(parameters)):
                gradient = np.asarray(jax.grad(objective)(parameters))
                self.assertTrue(np.isfinite(gradient).all())

    def test_drift_learner_uses_identical_hgf_state_update(self):
        base = NArmHGF(4)
        drift = NArmHGFDriftLearner(4)
        base_state = base.init_state()
        drift_state = drift.init_state()
        params = {"omega": -2.3, "kappa": 0.7, "scaler": 2.0}
        for choice, reward in zip(self.choices, self.rewards, strict=True):
            context = {"choice": int(choice), "feedback": float(reward)}
            base_state = base.update_python(base_state, params, context)
            drift_state = drift.update_python(drift_state, params, context)
        for field in base_state:
            np.testing.assert_allclose(base_state[field], drift_state[field])

    def test_sticky_bonus_applies_only_after_a_choice(self):
        learner = NArmHGFSticky(4)
        params = {"omega": -2.0, "kappa": 1.0, "sticky": 0.5}
        state = learner.init_state()
        before = learner.compute_python(state, params, context={})
        self.assertEqual(len(set(before.values())), 1)

        state = learner.update_python(
            state, params, {"choice": 2, "feedback": 1.0}
        )
        unbiased = NArmHGF.compute_python(learner, state, params, context={})
        biased = learner.compute_python(state, params, context={})
        self.assertAlmostEqual(biased["q2"] - unbiased["q2"], 0.5)
        for action in (0, 1, 3):
            self.assertAlmostEqual(biased[f"q{action}"] - unbiased[f"q{action}"], 0.0)


if __name__ == "__main__":
    unittest.main()
