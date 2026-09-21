from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .env import CrazyflieEnv


# ---------------------------------------------------------------------
# Collection behavior
# ---------------------------------------------------------------------

# Approach behavior.
APPROACH_KP = 0.8
APPROACH_MAX_XY_SPEED = 1.0
APPROACH_MAX_Z_SPEED = 0.5

# Once near a waypoint, keep making small position corrections
# rather than immediately choosing another target.
HOLD_KP = 0.5
HOLD_MAX_XY_SPEED = 0.25
HOLD_MAX_Z_SPEED = 0.20

# Waypoint is considered reached inside this radius.
WAYPOINT_RADIUS = 0.30

# Environment runs at 50 Hz.
# 40 steps = 0.8 seconds.
HOLD_STEPS = 40

# 300 steps = 6 seconds.
# If a waypoint is not reached within this time, move on.
MAX_APPROACH_STEPS = 300


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


def sample_waypoint(
    rng: np.random.Generator,
    *,
    xy_range: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    return np.array(
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


def position_action(
    obs: np.ndarray,
    waypoint: np.ndarray,
    *,
    kp: float,
    max_xy_speed: float,
    max_z_speed: float,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert position error into the normalized high-level action expected
    by CrazyflieEnv:

        [vx, vy, vz, yaw_rate] in [-1, 1]

    CrazyflieEnv maps these to approximately:

        vx       -> [-2, 2] m/s
        vy       -> [-2, 2] m/s
        vz       -> [-1, 1] m/s
        yaw_rate -> [-1.5, 1.5] rad/s
    """

    position = obs[:3]

    error = waypoint - position

    desired_velocity = (
        kp * error
    )

    desired_velocity[:2] = clip_norm(
        desired_velocity[:2],
        max_xy_speed,
    )

    desired_velocity[2] = np.clip(
        desired_velocity[2],
        -max_z_speed,
        max_z_speed,
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

    if noise is not None:
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

    # Match the ExoRL convention used by the RLDP loader:
    # action[0] is a dummy action.
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

    waypoint = sample_waypoint(
        rng,
        xy_range=xy_range,
        z_min=z_min,
        z_max=z_max,
    )

    phase = "approach"

    approach_steps = 0
    hold_steps_left = 0

    noise = np.zeros(
        4,
        dtype=np.float32,
    )

    for _ in range(
        env.episode_steps
    ):
        position = (
            ts.observation[:3]
        )

        distance = float(
            np.linalg.norm(
                position - waypoint
            )
        )

        # -------------------------------------------------------------
        # State transitions
        # -------------------------------------------------------------

        if phase == "approach":
            if distance <= WAYPOINT_RADIUS:
                phase = "hold"
                hold_steps_left = HOLD_STEPS

                # Remove accumulated exploration noise when entering
                # the hold phase.
                noise[:] = 0.0

            elif approach_steps >= MAX_APPROACH_STEPS:
                # Failed to reach this waypoint in time.
                # Sample another one rather than spending the whole
                # episode chasing an unreachable target.
                waypoint = sample_waypoint(
                    rng,
                    xy_range=xy_range,
                    z_min=z_min,
                    z_max=z_max,
                )

                approach_steps = 0

                noise[:] = 0.0

        # -------------------------------------------------------------
        # Action
        # -------------------------------------------------------------

        if phase == "hold":
            # No exploration noise while holding.
            #
            # We still use a small position controller instead of
            # commanding exactly zero velocity. That lets the drone
            # correct drift while remaining slow near the waypoint.
            action = position_action(
                ts.observation,
                waypoint,
                kp=HOLD_KP,
                max_xy_speed=HOLD_MAX_XY_SPEED,
                max_z_speed=HOLD_MAX_Z_SPEED,
                noise=None,
            )

            hold_steps_left -= 1

        else:
            # Smooth temporally correlated exploration noise.
            noise = (
                0.95 * noise
                + 0.05
                * rng.normal(
                    0.0,
                    [
                        0.12,
                        0.12,
                        0.06,
                        0.10,
                    ],
                )
            ).astype(
                np.float32
            )

            action = position_action(
                ts.observation,
                waypoint,
                kp=APPROACH_KP,
                max_xy_speed=APPROACH_MAX_XY_SPEED,
                max_z_speed=APPROACH_MAX_Z_SPEED,
                noise=noise,
            )

            approach_steps += 1

        # -------------------------------------------------------------
        # Environment transition
        # -------------------------------------------------------------

        ts = env.step(
            action
        )

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

        # -------------------------------------------------------------
        # Finished holding: choose another waypoint.
        # -------------------------------------------------------------

        if (
            phase == "hold"
            and hold_steps_left <= 0
        ):
            waypoint = sample_waypoint(
                rng,
                xy_range=xy_range,
                z_min=z_min,
                z_max=z_max,
            )

            phase = "approach"

            approach_steps = 0
            hold_steps_left = 0

            noise[:] = 0.0

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