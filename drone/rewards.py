from __future__ import annotations

import numpy as np


TASKS = {
    "center": np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float32,
    ),
    "left": np.array(
        [-1.5, 0.0, 1.0],
        dtype=np.float32,
    ),
    "right": np.array(
        [1.5, 0.0, 1.0],
        dtype=np.float32,
    ),
    "corner": np.array(
        [1.5, 1.5, 1.2],
        dtype=np.float32,
    ),
    "high": np.array(
        [0.0, 0.0, 1.8],
        dtype=np.float32,
    ),
}


def goal_reward(
    observations: np.ndarray,
    goal: np.ndarray,
    *,
    sigma_position: float = 0.35,
    sigma_velocity: float = 0.50,
) -> np.ndarray:

    position = observations[..., 0:3]
    velocity = observations[..., 3:6]

    distance_sq = np.sum(
        (position - goal) ** 2,
        axis=-1,
    )

    speed_sq = np.sum(
        velocity**2,
        axis=-1,
    )

    position_reward = np.exp(
        -distance_sq
        / (
            2.0
            * sigma_position**2
        )
    )

    velocity_reward = np.exp(
        -speed_sq
        / (
            2.0
            * sigma_velocity**2
        )
    )

    return (
        position_reward
        * velocity_reward
    ).astype(np.float32)