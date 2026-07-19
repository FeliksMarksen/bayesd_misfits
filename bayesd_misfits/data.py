"""Data loading and transformation utilities for the MindRL Challenge dataset.

This module converts the raw MindRL Challenge JSONL trajectories into the
flat trial-level DataFrame format expected by HSSM / ssms.rl.

Reference data contract (from Carney_ARIA_Workshop_2026 notebooks):
    - One row per trial.
    - `response`        : int, the chosen action (0-indexed for multi-arm).
    - `rt`              : float, response time in *seconds* (or -1.0 placeholder
                          for choice-only models).
    - `participant_id`  : int, subject identifier (used for hierarchical
                          random effects: `param ~ 1 + (1|participant_id)`).
    - `trial_id`        : int, within-trajectory trial index.
    - `feedback`        : float, reward received on that trial (used by the RL
                          learner to update Q-values).

MindRL Challenge raw format (public_train.jsonl, one line = one trajectory):
    {"context": {subject_id, trajectory_id, task_id, ...},
     "trials":  [{trial_index, action, reward, info: {rt}}, ...]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# The dataset lives at the repo root under hf_cache/public/.
# Resolve relative to this module so it works regardless of the caller's CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA_DIR = _REPO_ROOT / "hf_cache" / "public"

# ---------------------------------------------------------------------------
# Low-level JSONL readers
# ---------------------------------------------------------------------------


def load_trajectories(
    path: str | Path = _DEFAULT_DATA_DIR / "public_train.jsonl",
) -> list[dict]:
    """Load raw trajectories from a JSONL file.

    Parameters
    ----------
    path : str | Path
        Path to ``public_train.jsonl`` (or any JSONL file with the same schema).

    Returns
    -------
    list[dict]
        A list of trajectory dicts, each with ``context`` and ``trials`` keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trajectory file not found: {path}. "
            "Download it with: "
            "hf download mindrl-hub/mindrl-challenge-public "
            "public_train.jsonl --repo-type dataset --local-dir hf_cache/public"
        )
    trajectories: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    return trajectories


def load_reward_schedules(
    path: str | Path = _DEFAULT_DATA_DIR / "public_train_reward_schedules.jsonl",
) -> list[dict]:
    """Load the reward-schedule sidecar file.

    Each entry contains the full (including unchosen) option payoff schedule
    for every trial in a trajectory.  **If you use this in your model, disclose
    it in your Interpretation Card** (per challenge rules).

    Parameters
    ----------
    path : str | Path
        Path to ``public_train_reward_schedules.jsonl``.

    Returns
    -------
    list[dict]
        Each dict has ``trajectory_id``, ``subject_id``, ``reward_schedule``
        (list of ``{trial_index, available_rewards: [r0, r1, r2, r3]}``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reward schedule file not found: {path}. "
            "Download it with: "
            "hf download mindrl-hub/mindrl-challenge-public "
            "public_train_reward_schedules.jsonl --repo-type dataset "
            "--local-dir hf_cache/public"
        )
    schedules: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                schedules.append(json.loads(line))
    return schedules


# ---------------------------------------------------------------------------
# Core transformation: JSONL trajectories → HSSM DataFrame
# ---------------------------------------------------------------------------


def _subject_id_to_int(subject_id: str) -> int:
    """Convert ``"sub_000848"`` → ``848``."""
    return int(subject_id.split("_")[-1])


def _trajectory_id_to_int(trajectory_id: str) -> int:
    """Convert ``"traj_005472"`` → ``5472``."""
    return int(trajectory_id.split("_")[-1])


def trajectories_to_dataframe(
    trajectories: list[dict],
    *,
    rt_unit: Literal["ms", "s"] = "ms",
    feedback_transform: Literal["raw", "normalize", "binary"] = "normalize",
    binary_threshold: float = 50.0,
    group_by: Literal["subject", "trajectory"] = "subject",
    drop_missing_rt: bool = False,
    rt_placeholder: float | None = -1.0,
) -> pd.DataFrame:
    """Flatten trajectory dicts into a trial-level DataFrame for HSSM.

    Parameters
    ----------
    trajectories : list[dict]
        Raw trajectory dicts as returned by :func:`load_trajectories`.
    rt_unit : {"ms", "s"}
        Unit of the ``rt`` field in the raw data.  HSSM expects seconds, so
        ``"ms"`` triggers a /1000 conversion.
    feedback_transform : {"raw", "normalize", "binary"}
        How to transform the reward (1–100) into the ``feedback`` column:
        - ``"raw"``      : keep the original 1–100 scale.
        - ``"normalize"``: scale to [0, 1] via ``reward / 100``.
        - ``"binary"``   : threshold at ``binary_threshold`` → 0 or 1.
    binary_threshold : float
        Threshold for ``feedback_transform="binary"`` (default 50.0).
    group_by : {"subject", "trajectory"}
        Which identifier maps to ``participant_id``:
        - ``"subject"``    : one participant may span multiple trajectories
                             (blocks).  HSSM random effects are per-subject.
                             ``trial_id`` is renumbered continuously across
                             trajectories within each subject (see
                             ``global_trial_id`` column for the within-block
                             index).
        - ``"trajectory"`` : each trajectory is treated as a separate
                             participant.  Use this if trajectories are
                             independent blocks with different reward schedules,
                             or when Q-value carryover between blocks is
                             undesirable.  Produces a balanced panel
                             (108–120 trials each).
    drop_missing_rt : bool
        If ``True``, drop trials where ``rt`` is null.  If ``False``, keep them
        with ``rt`` set to ``rt_placeholder`` (or NaN if placeholder is None).
    rt_placeholder : float | None
        Value to fill for missing RTs.  Set to ``-1.0`` (default) for
        choice-only models, or ``None`` to leave as NaN.

    Returns
    -------
    pd.DataFrame
        Columns: ``participant_id``, ``trial_id``, ``response``, ``rt``,
        ``feedback``, ``subject_id`` (str), ``trajectory_id`` (str),
        ``task_id``, ``block_id``, ``reward_raw``, ``global_trial_id``
        (within-trajectory trial index, preserved as metadata).
    """
    records: list[dict] = []

    for traj in trajectories:
        ctx = traj["context"]
        subj_id_str = ctx["subject_id"]
        traj_id_str = ctx["trajectory_id"]
        subj_int = _subject_id_to_int(subj_id_str)
        traj_int = _trajectory_id_to_int(traj_id_str)
        block_id = ctx.get("metadata", {}).get("block_id", None)
        task_id = ctx.get("task_id", None)

        participant_id = subj_int if group_by == "subject" else traj_int

        for trial in traj["trials"]:
            rt = trial.get("info", {}).get("rt", None)
            if rt is None:
                if drop_missing_rt:
                    continue
                rt_val = rt_placeholder if rt_placeholder is not None else np.nan
            else:
                rt_val = float(rt)
                if rt_unit == "ms":
                    rt_val /= 1000.0

            reward = float(trial["reward"])

            if feedback_transform == "raw":
                feedback = reward
            elif feedback_transform == "normalize":
                feedback = reward / 100.0
            elif feedback_transform == "binary":
                feedback = 1.0 if reward >= binary_threshold else 0.0
            else:
                raise ValueError(f"Unknown feedback_transform: {feedback_transform}")

            records.append(
                {
                    "participant_id": participant_id,
                    "trial_id": trial["trial_index"],  # within-trajectory
                    "global_trial_id": trial["trial_index"],  # may be renumbered below
                    "response": int(trial["action"]),
                    "rt": rt_val,
                    "feedback": feedback,
                    # Metadata columns (not required by HSSM but useful for
                    # analysis, plotting, and cross-referencing reward schedules)
                    "subject_id": subj_id_str,
                    "trajectory_id": traj_id_str,
                    "task_id": task_id,
                    "block_id": block_id,
                    "reward_raw": reward,
                }
            )

    df = pd.DataFrame(records)

    # Ensure correct dtypes
    df["participant_id"] = df["participant_id"].astype(int)
    df["trial_id"] = df["trial_id"].astype(int)
    df["response"] = df["response"].astype(int)
    df["feedback"] = df["feedback"].astype(float)
    df["rt"] = df["rt"].astype(float)

    # When group_by="subject", a subject spans multiple trajectories.
    # The RL learner updates Q-values trial-by-trial, so trial_id must be a
    # *continuous* sequence across all of a subject's trajectories — otherwise
    # the learner would see trial_id=0 repeated for each block.
    # We keep the original within-trajectory index as `global_trial_id` for
    # cross-referencing reward schedules, and renumber `trial_id` to be
    # continuous within each participant.
    if group_by == "subject":
        # Sort by subject, then by trajectory_id (deterministic block order),
        # then by within-trajectory trial index.
        df = df.sort_values(
            ["participant_id", "trajectory_id", "trial_id"]
        ).reset_index(drop=True)
        # Renumber trial_id continuously within each participant
        df["trial_id"] = df.groupby("participant_id").cumcount()
    else:
        # group_by="trajectory": each trajectory is one participant, trial_id
        # is already the correct within-trajectory index.
        df = df.sort_values(
            ["participant_id", "trial_id"]
        ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Reward schedule flattening (sidecar)
# ---------------------------------------------------------------------------


def reward_schedules_to_dataframe(
    schedules: list[dict],
) -> pd.DataFrame:
    """Flatten the reward-schedule sidecar into a trial-level DataFrame.

    This exposes the *full* payoff for every option (including unchosen ones)
    on every trial, which is not part of the standard agent-facing trajectory.

    Returns
    -------
    pd.DataFrame
        Columns: ``trajectory_id`` (int), ``subject_id`` (str),
        ``trial_id`` (int), ``available_rewards`` (list of 4 floats),
        ``reward_0``, ``reward_1``, ``reward_2``, ``reward_3``.
    """
    records: list[dict] = []
    for entry in schedules:
        traj_str = entry["trajectory_id"]
        traj_int = _trajectory_id_to_int(traj_str)
        subj_str = entry["subject_id"]
        for trial in entry["reward_schedule"]:
            rewards = trial["available_rewards"]
            records.append(
                {
                    "trajectory_id": traj_str,      # string key for merging
                    "trajectory_id_int": traj_int,   # numeric key for sorting
                    "subject_id": subj_str,
                    "trial_id": trial["trial_index"],
                    "available_rewards": rewards,
                    "reward_0": rewards[0],
                    "reward_1": rewards[1],
                    "reward_2": rewards[2],
                    "reward_3": rewards[3],
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# High-level convenience loaders
# ---------------------------------------------------------------------------


def load_challenge_data(
    data_dir: str | Path = _DEFAULT_DATA_DIR,
    *,
    rt_unit: Literal["ms", "s"] = "ms",
    feedback_transform: Literal["raw", "normalize", "binary"] = "normalize",
    binary_threshold: float = 50.0,
    group_by: Literal["subject", "trajectory"] = "subject",
    drop_missing_rt: bool = False,
    rt_placeholder: float | None = -1.0,
    include_reward_schedules: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end loader: JSONL files → HSSM-ready DataFrame.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing ``public_train.jsonl`` and optionally
        ``public_train_reward_schedules.jsonl``.
    include_reward_schedules : bool
        If ``True``, also load and return the reward-schedule sidecar
        DataFrame (remember to disclose usage in your Interpretation Card).

    All other parameters are passed through to
    :func:`trajectories_to_dataframe`.

    Returns
    -------
    pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]
        The HSSM-ready trial DataFrame, or a tuple ``(df, schedule_df)``
        if ``include_reward_schedules=True``.

    Examples
    --------
    Choice-only model (no RT, normalized feedback):

    >>> df = load_challenge_data(
    ...     feedback_transform="normalize",
    ...     rt_placeholder=-1.0,
    ... )

    RT-based model (drop missing RTs, raw rewards):

    >>> df = load_challenge_data(
    ...     feedback_transform="raw",
    ...     drop_missing_rt=True,
    ... )

    With reward schedules for analysis:

    >>> df, schedules = load_challenge_data(include_reward_schedules=True)
    """
    data_dir = Path(data_dir)
    trajectories = load_trajectories(data_dir / "public_train.jsonl")
    df = trajectories_to_dataframe(
        trajectories,
        rt_unit=rt_unit,
        feedback_transform=feedback_transform,
        binary_threshold=binary_threshold,
        group_by=group_by,
        drop_missing_rt=drop_missing_rt,
        rt_placeholder=rt_placeholder,
    )

    if include_reward_schedules:
        schedules = load_reward_schedules(
            data_dir / "public_train_reward_schedules.jsonl"
        )
        sched_df = reward_schedules_to_dataframe(schedules)
        return df, sched_df

    return df


# ---------------------------------------------------------------------------
# Validation & inspection helpers
# ---------------------------------------------------------------------------


def summarize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Print a quick summary of the challenge DataFrame and return per-participant stats.

    Useful sanity check before fitting an HSSM model — confirms the panel is
    balanced (equal trials per participant) and shows action/reward distributions.
    """
    n_participants = df["participant_id"].nunique()
    n_trials_total = len(df)
    trials_per = df.groupby("participant_id").size()

    print(f"Participants:         {n_participants}")
    print(f"Total trials:         {n_trials_total}")
    print(
        f"Trials/participant:   "
        f"min={trials_per.min()}, max={trials_per.max()}, "
        f"mean={trials_per.mean():.1f}"
    )
    print(f"Actions (response):   {sorted(df['response'].unique())}")
    print(
        f"Action distribution:  "
        f"{dict(df['response'].value_counts().sort_index())}"
    )
    print(f"Feedback range:       [{df['feedback'].min():.3f}, {df['feedback'].max():.3f}]")
    print(f"RT range:             [{df['rt'].min():.3f}, {df['rt'].max():.3f}]")
    n_missing_rt = df["rt"].isna().sum() + (df["rt"] == -1.0).sum()
    print(f"Missing/placeholder RT: {n_missing_rt} ({n_missing_rt / len(df):.1%})")

    # Per-participant summary
    per_participant = (
        df.groupby("participant_id")
        .agg(
            n_trials=("trial_id", "count"),
            mean_feedback=("feedback", "mean"),
            mean_rt=("rt", lambda x: x[x > 0].mean() if (x > 0).any() else np.nan),
        )
        .reset_index()
    )
    return per_participant


def validate_for_hssm(df: pd.DataFrame, *, require_rt: bool = True) -> None:
    """Check that the DataFrame meets HSSM/ssms.rl expectations.

    Raises
    ------
    ValueError
        If any check fails.
    """
    required = ["participant_id", "trial_id", "response", "feedback"]
    if require_rt:
        required.append("rt")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Response must be non-negative integers for multi-arm models
    if df["response"].min() < 0:
        raise ValueError(
            f"Response values must be >= 0 for multi-arm models, "
            f"got min={df['response'].min()}"
        )

    # Participant IDs must be integers
    if not pd.api.types.is_integer_dtype(df["participant_id"]):
        raise ValueError(
            f"participant_id must be integer, got {df['participant_id'].dtype}"
        )

    # Check panel is balanced (same trials per participant)
    counts = df.groupby("participant_id").size()
    if counts.nunique() > 1:
        print(
            f"WARNING: Unbalanced panel — trials per participant vary "
            f"({counts.min()}–{counts.max()}). "
            "HSSM may require balanced panels depending on the model."
        )

    # No NaN feedback
    if df["feedback"].isna().any():
        raise ValueError("feedback contains NaN values")

    if require_rt:
        if df["rt"].isna().any():
            n_nan = df["rt"].isna().sum()
            print(
                f"WARNING: {n_nan} trials have NaN rt. "
                "Use rt_placeholder=-1.0 for choice-only models, "
                "or drop_missing_rt=True to remove them."
            )

    print("Validation passed ✓")