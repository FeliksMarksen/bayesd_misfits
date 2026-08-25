"""Tests for the direct categorical-logit HSSM decision process."""

from __future__ import annotations

import numpy as np

from bayesd_misfits.categorical_logits import categorical_logits_simulator


def test_categorical_logits_simulator_respects_supplied_logits():
    result = categorical_logits_simulator(
        logit0=5.0,
        logit1=0.0,
        logit2=0.0,
        logit3=0.0,
        n_samples=4000,
        n_trials=1,
        random_state=7,
    )
    choices = np.asarray(result["choices"])[:, 0, 0]
    assert np.mean(choices == 0) > 0.97
    assert np.all(np.asarray(result["rts"]) == -1.0)
