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

PHASE_APPROACH = 0
PHASE_HOLD = 1


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
    Convert position error into the normalized high-level action:

        [vx, vy, vz, yaw_rate] in [-1, 1]

    CrazyflieEnv maps these approximately to:

        vx       -> [-2, 2] m/s
        vy       -> [-2, 2] m/s
        vz       -> [-1, 1] m/s
        yaw_rate -> [-1.5, 1.5] rad/s
    """

    position = obs[:3]
    error = waypoint - position

    desired_velocity = kp * error

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
            desired_velocity[2],
            0.0,
        ],
        dtype=np.float32,
    )

    if noise is not None:
        action += noise.astype(np.float32)

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

    # ExoRL convention used by the RLDP loader:
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

    waypoints = [
        waypoint.copy()
    ]

    phases = [
        PHASE_APPROACH
    ]

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

        # Make the metadata for the current state match the phase and
        # waypoint actually used to choose the next action.
        waypoints[-1] = waypoint.copy()

        phases[-1] = (
            PHASE_HOLD
            if phase == "hold"
            else PHASE_APPROACH
        )

        # -------------------------------------------------------------
        # Action
        # -------------------------------------------------------------

        if phase == "hold":
            # No exploration noise during the hold phase.
            #
            # Small position corrections keep the drone close to the
            # waypoint while allowing low-speed states to enter the
            # offline dataset.
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
        # Finished holding: select the next waypoint.
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

        # Metadata for the newly recorded state. If the phase changes
        # at the beginning of the next control step, this entry will be
        # updated before that action is selected.
        waypoints.append(
            waypoint.copy()
        )

        phases.append(
            PHASE_HOLD
            if phase == "hold"
            else PHASE_APPROACH
        )

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

        "waypoint": np.asarray(
            waypoints,
            dtype=np.float32,
        ),
        "phase": np.asarray(
            phases,
            dtype=np.uint8,
        ),
    }


def main() -> None:
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