"""Evaluate the frozen scale-free submission agent on the held-out public split."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import Agent
from run_comparison import SEED, load_split_data


OUTPUT_PATH = Path(
    os.environ.get(
        "OUTPUT_PATH",
        "research/causal_scale_submission_evaluation.json",
    )
)


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    _, valid_data, split_info = load_split_data(include_rt=False)

    total_nll = 0.0
    total_correct = 0
    total_trials = 0
    max_affine_probability_difference = 0.0
    affine_scale = 10.0
    affine_offset = 500.0

    for _, trajectory in valid_data.groupby("participant_id", sort=True):
        trajectory = trajectory.sort_values("trial_id")
        original = Agent(config)
        transformed = Agent(config)
        context = {"available_actions": [0, 1, 2, 3]}
        original.reset(context)
        transformed.reset(context)
        original_history: list[dict[str, float | int]] = []
        transformed_history: list[dict[str, float | int]] = []

        for row in trajectory.itertuples(index=False):
            prediction = original.predict(original_history)["action_probs"]
            affine_prediction = transformed.predict(transformed_history)[
                "action_probs"
            ]
            max_affine_probability_difference = max(
                max_affine_probability_difference,
                max(
                    abs(prediction[action] - affine_prediction[action])
                    for action in context["available_actions"]
                ),
            )

            choice = int(row.response)
            reward = float(row.feedback)
            probability = max(float(prediction[choice]), 1e-300)
            total_nll -= math.log(probability)
            total_correct += int(
                max(prediction, key=prediction.get) == choice
            )
            total_trials += 1

            transformed_reward = affine_scale * reward + affine_offset
            original.update(choice, reward)
            transformed.update(choice, transformed_reward)
            original_history.append({"action": choice, "reward": reward})
            transformed_history.append(
                {"action": choice, "reward": transformed_reward}
            )

    result = {
        "seed": SEED,
        "parameters": config["model"],
        "split": split_info,
        "evaluation": {
            "nll": total_nll / total_trials,
            "accuracy": total_correct / total_trials,
            "num_trials": total_trials,
            "num_trajectories": split_info["validation_trajectories"],
        },
        "affine_invariance_check": {
            "transformation": "10 * reward + 500",
            "max_absolute_probability_difference": (
                max_affine_probability_difference
            ),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
