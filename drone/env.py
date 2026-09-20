from __future__ import annotations

from pathlib import Path

import dm_env
from dm_env import specs
from dm_control import mujoco
import numpy as np

from .controller import VelocityController


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = (
    ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "bitcraze_crazyflie_2"
    / "scene.xml"
)


class CrazyflieEnv(dm_env.Environment):
    """
    Reward-free Crazyflie environment.

    Observation layout:
        [0:3]   position xyz
        [3:6]   linear velocity xyz
        [6:10]  quaternion wxyz
        [10:13] angular velocity xyz

    RL action:
        [vx, vy, vz, yaw_rate], normalized to [-1, 1].
    """

    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_MODEL,
        episode_steps: int = 1000,
        control_timestep: float = 0.02,
        seed: int = 0,
        randomize_start: bool = True,
    ) -> None:
        self.model_path = Path(model_path)

        self.physics = (
            mujoco.Physics.from_xml_path(
                str(self.model_path)
            )
        )

        self.episode_steps = episode_steps
        self.control_timestep = control_timestep
        self.randomize_start = randomize_start

        physics_timestep = float(
            self.physics.timestep()
        )

        ratio = (
            control_timestep
            / physics_timestep
        )

        self._substeps = int(round(ratio))

        if not np.isclose(
            self._substeps * physics_timestep,
            control_timestep,
        ):
            raise ValueError(
                "control_timestep must be an integer "
                "multiple of MuJoCo timestep"
            )

        self.controller = VelocityController()

        self.rng = np.random.default_rng(seed)

        self._step = 0
        self._needs_reset = True

        self._action_spec = specs.BoundedArray(
            shape=(4,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name="velocity_command",
        )

        self._observation_spec = specs.Array(
            shape=(13,),
            dtype=np.float32,
            name="state",
        )

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_state(self) -> np.ndarray:
        qpos = np.asarray(
            self.physics.data.qpos
        )

        qvel = np.asarray(
            self.physics.data.qvel
        )

        position = qpos[:3].copy()
        quaternion = qpos[3:7].copy()

        # q and -q encode identical rotations.
        if quaternion[0] < 0.0:
            quaternion *= -1.0

        linear_velocity = qvel[:3].copy()

        # Body gyro is already available in the model.
        angular_velocity = np.asarray(
            self.physics.named.data.sensordata[
                "body_gyro"
            ]
        ).copy()

        obs = np.concatenate(
            [
                position,
                linear_velocity,
                quaternion,
                angular_velocity,
            ]
        )

        return obs.astype(
            np.float32,
            copy=False,
        )

    def get_physics_state(self) -> np.ndarray:
        return (
            self.physics.get_state()
            .astype(np.float64)
            .copy()
        )

    def reset(self):
        self.physics.reset()

        with self.physics.reset_context():

            if self.randomize_start:
                self.physics.data.qpos[:3] = np.array(
                    [
                        self.rng.uniform(-0.25, 0.25),
                        self.rng.uniform(-0.25, 0.25),
                        self.rng.uniform(0.8, 1.2),
                    ]
                )
            else:
                self.physics.data.qpos[:3] = np.array(
                    [0.0, 0.0, 1.0]
                )

            self.physics.data.qpos[3:7] = np.array(
                [1.0, 0.0, 0.0, 0.0]
            )

            self.physics.data.qvel[:] = 0.0

            # Native Crazyflie hover thrust.
            self.physics.data.ctrl[:] = np.array(
                [0.26487, 0.0, 0.0, 0.0]
            )

        obs = self.get_state()

        self.controller.reset(
            obs[6:10]
        )

        self._step = 0
        self._needs_reset = False

        return dm_env.restart(obs)

    def step(
        self,
        action: np.ndarray,
    ):
        if self._needs_reset:
            return self.reset()

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        self._action_spec.validate(action)

        obs = self.get_state()

        native_control = self.controller.control(
            action=action,
            position=obs[0:3],
            velocity=obs[3:6],
            quaternion=obs[6:10],
            angular_velocity=obs[10:13],
            dt=self.control_timestep,
        )

        self.physics.set_control(
            native_control
        )

        self.physics.step(
            self._substeps
        )

        self._step += 1

        obs = self.get_state()

        # RLDP pretraining is reward-free.
        reward = 0.0

        if self._step >= self.episode_steps:
            self._needs_reset = True

            # This is a time-limit truncation,
            # not an environment terminal.
            return dm_env.truncation(
                reward=reward,
                observation=obs,
                discount=1.0,
            )

        return dm_env.transition(
            reward=reward,
            observation=obs,
            discount=1.0,
        )