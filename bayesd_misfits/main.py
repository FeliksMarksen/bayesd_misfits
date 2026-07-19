"""Bayes'd Misfits — MindRL Challenge 2026 entry point.

Quick start
-----------
>>> from bayesd_misfits.data import load_challenge_data, summarize_data
>>> df = load_challenge_data(feedback_transform="normalize")
>>> summarize_data(df)
"""

import pandas as pd

from bayesd_misfits.data import (
    load_challenge_data,
    load_trajectories,
    load_reward_schedules,
    trajectories_to_dataframe,
    reward_schedules_to_dataframe,
    summarize_data,
    validate_for_hssm,
)

__all__ = [
    "load_challenge_data",
    "load_trajectories",
    "load_reward_schedules",
    "trajectories_to_dataframe",
    "reward_schedules_to_dataframe",
    "summarize_data",
    "validate_for_hssm",
]


def main() -> pd.DataFrame:
    """Load the MindRL Challenge data in HSSM-ready format and print a summary."""
    # Choice-only setup: RT placeholder = -1.0, feedback normalized to [0, 1].
    # Adjust parameters for RT-based or binary-feedback models as needed.
    df = load_challenge_data(
        feedback_transform="normalize",
        rt_placeholder=-1.0,
    )
    summarize_data(df)
    validate_for_hssm(df, require_rt=False)
    return df


if __name__ == "__main__":
    mindRL_data = main()