from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .env import CrazyflieEnv


def clip_norm(
    vector: np.ndarray,
    maximum: float,
) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm <= maximum:
        return vector

    return vector * (
        maximum / max(norm, 1e-8)
    )


def waypoint_action(
    obs: np.ndarray,
    waypoint: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:

    position = obs[:3]

    error = waypoint - position

    desired_velocity = (
        0.9 * error
    )

    desired_velocity[:2] = clip_norm(
        desired_velocity[:2],
        1.5,
    )

    desired_velocity[2] = np.clip(
        desired_velocity[2],
        -0.8,
        0.8,
    )

    action = np.array(
        [
            desired_velocity[0] / 2.0,
            desired_velocity[1] / 2.0,
            desired_velocity[2] / 1.0,
            0.0,
        ],
        dtype=np.float32,
    )

    action += noise.astype(
        np.float32
    )

    return np.clip(
        action,
        -1.0,
        1.0,
    )


def collect_episode(
    env: CrazyflieEnv,
    rng: np.random.Generator,
    *,
    xy_range: float,
    z_min: float,
    z_max: float,
) -> dict[str, np.ndarray]:

    ts = env.reset()

    observations = [
        ts.observation.copy()
    ]

    # Match ExoRL convention used by RLDP loader:
    # action[0] is unused/dummy.
    actions = [
        np.zeros(
            4,
            dtype=np.float32,
        )
    ]

    physics = [
        env.get_physics_state()
    ]

    discounts = [
        np.array(
            [1.0],
            dtype=np.float32,
        )
    ]

    waypoint = None
    waypoint_steps_left = 0

    noise = np.zeros(
        4,
        dtype=np.float32,
    )

    for _ in range(
        env.episode_steps
    ):

        if waypoint_steps_left <= 0:
            waypoint = np.array(
                [
                    rng.uniform(
                        -xy_range,
                        xy_range,
                    ),
                    rng.uniform(
                        -xy_range,
                        xy_range,
                    ),
                    rng.uniform(
                        z_min,
                        z_max,
                    ),
                ],
                dtype=np.float32,
            )

            waypoint_steps_left = int(
                rng.integers(
                    50,
                    151,
                )
            )

        # Smooth exploration noise.
        noise = (
            0.95 * noise
            + 0.05
            * rng.normal(
                0.0,
                [
                    0.20,
                    0.20,
                    0.10,
                    0.15,
                ],
            )
        ).astype(np.float32)

        action = waypoint_action(
            ts.observation,
            waypoint,
            noise,
        )

        ts = env.step(action)

        observations.append(
            ts.observation.copy()
        )

        actions.append(
            action.copy()
        )

        physics.append(
            env.get_physics_state()
        )

        discounts.append(
            np.array(
                [
                    ts.discount
                    if ts.discount is not None
                    else 1.0
                ],
                dtype=np.float32,
            )
        )

        waypoint_steps_left -= 1

    return {
        "observation": np.asarray(
            observations,
            dtype=np.float32,
        ),
        "action": np.asarray(
            actions,
            dtype=np.float32,
        ),
        "physics": np.asarray(
            physics,
        ),
        "discount": np.asarray(
            discounts,
            dtype=np.float32,
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--xy-range",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--z-min",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--z-max",
        type=float,
        default=2.0,
    )

    args = parser.parse_args()

    output = (
        Path(args.output)
        / "buffer"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        args.seed
    )

    env = CrazyflieEnv(
        seed=args.seed,
    )

    for episode_idx in range(
        args.episodes
    ):
        data = collect_episode(
            env,
            rng,
            xy_range=args.xy_range,
            z_min=args.z_min,
            z_max=args.z_max,
        )

        path = (
            output
            / f"episode_{episode_idx:06d}_1000.npz"
        )

        np.savez_compressed(
            path,
            **data,
        )

        if episode_idx % 10 == 0:
            print(
                f"{episode_idx}/{args.episodes}"
            )


if __name__ == "__main__":
    main()