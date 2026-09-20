from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion convention: [w, x, y, z]."""

    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quat.astype(np.float64))

    return matrix.reshape(3, 3)


def yaw_from_matrix(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class VelocityControllerConfig:
    mass: float = 0.027
    gravity: float = 9.81

    max_vx: float = 2.0
    max_vy: float = 2.0
    max_vz: float = 1.0
    max_yaw_rate: float = 1.5

    # Outer velocity loop.
    kp_velocity_xy: float = 2.0
    kp_velocity_z: float = 2.5

    max_horizontal_acceleration: float = 4.0
    max_vertical_acceleration: float = 4.0

    # Attitude controller in physical torque units.
    kp_attitude: float = 2.5e-5
    kd_attitude: float = 5.0e-6

    # Crazyflie Menagerie actuator gear magnitude.
    moment_scale: float = 1.0e-5

    min_thrust: float = 0.0
    max_thrust: float = 0.35


class VelocityController:
    """
    High-level action:
        [-1, 1]^4

    maps to:
        vx
        vy
        vz
        yaw_rate

    and ultimately outputs native Crazyflie controls:
        thrust
        x moment
        y moment
        z moment
    """

    def __init__(
        self,
        config: VelocityControllerConfig | None = None,
    ) -> None:
        self.cfg = config or VelocityControllerConfig()

        self.desired_yaw = 0.0

    def reset(
        self,
        quaternion: np.ndarray,
    ) -> None:
        R = quat_to_matrix(quaternion)
        self.desired_yaw = yaw_from_matrix(R)

    def decode_action(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        action = np.asarray(action, dtype=np.float64)

        if action.shape != (4,):
            raise ValueError(
                f"Expected action shape (4,), got {action.shape}"
            )

        action = np.clip(action, -1.0, 1.0)

        velocity = np.array(
            [
                action[0] * self.cfg.max_vx,
                action[1] * self.cfg.max_vy,
                action[2] * self.cfg.max_vz,
            ],
            dtype=np.float64,
        )

        yaw_rate = (
            action[3] * self.cfg.max_yaw_rate
        )

        return velocity, float(yaw_rate)

    def control(
        self,
        *,
        action: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        del position

        cfg = self.cfg

        desired_velocity, desired_yaw_rate = (
            self.decode_action(action)
        )

        # ---------------------------------------
        # Velocity → desired acceleration
        # ---------------------------------------

        velocity_error = desired_velocity - velocity

        acceleration = np.array(
            [
                cfg.kp_velocity_xy * velocity_error[0],
                cfg.kp_velocity_xy * velocity_error[1],
                cfg.kp_velocity_z * velocity_error[2],
            ],
            dtype=np.float64,
        )

        horizontal_norm = np.linalg.norm(
            acceleration[:2]
        )

        if horizontal_norm > cfg.max_horizontal_acceleration:
            acceleration[:2] *= (
                cfg.max_horizontal_acceleration
                / horizontal_norm
            )

        acceleration[2] = np.clip(
            acceleration[2],
            -cfg.max_vertical_acceleration,
            cfg.max_vertical_acceleration,
        )

        # Desired total acceleration includes gravity compensation.
        desired_total_acceleration = (
            acceleration
            + np.array(
                [0.0, 0.0, cfg.gravity],
                dtype=np.float64,
            )
        )

        norm = np.linalg.norm(
            desired_total_acceleration
        )

        if norm < 1e-8:
            desired_body_z = np.array(
                [0.0, 0.0, 1.0]
            )
        else:
            desired_body_z = (
                desired_total_acceleration / norm
            )

        # ---------------------------------------
        # Desired yaw
        # ---------------------------------------

        self.desired_yaw = wrap_angle(
            self.desired_yaw
            + desired_yaw_rate * dt
        )

        heading = np.array(
            [
                np.cos(self.desired_yaw),
                np.sin(self.desired_yaw),
                0.0,
            ]
        )

        desired_body_y = np.cross(
            desired_body_z,
            heading,
        )

        y_norm = np.linalg.norm(
            desired_body_y
        )

        if y_norm < 1e-6:
            desired_body_y = np.array(
                [0.0, 1.0, 0.0]
            )
        else:
            desired_body_y /= y_norm

        desired_body_x = np.cross(
            desired_body_y,
            desired_body_z,
        )

        desired_rotation = np.column_stack(
            (
                desired_body_x,
                desired_body_y,
                desired_body_z,
            )
        )

        # ---------------------------------------
        # Current attitude
        # ---------------------------------------

        rotation = quat_to_matrix(
            quaternion
        )

        # Geometric SO(3) attitude error.
        error_matrix = (
            desired_rotation.T @ rotation
            - rotation.T @ desired_rotation
        )

        attitude_error = 0.5 * np.array(
            [
                error_matrix[2, 1],
                error_matrix[0, 2],
                error_matrix[1, 0],
            ]
        )

        torque = (
            -cfg.kp_attitude * attitude_error
            - cfg.kd_attitude * angular_velocity
        )

        # Menagerie x/y/z moment actuators use
        # negative gear coefficients.
        moment_control = (
            -torque / cfg.moment_scale
        )

        moment_control = np.clip(
            moment_control,
            -1.0,
            1.0,
        )

        # ---------------------------------------
        # Total thrust
        # ---------------------------------------

        thrust = (
            cfg.mass
            * np.dot(
                desired_total_acceleration,
                rotation[:, 2],
            )
        )

        thrust = np.clip(
            thrust,
            cfg.min_thrust,
            cfg.max_thrust,
        )

        return np.array(
            [
                thrust,
                moment_control[0],
                moment_control[1],
                moment_control[2],
            ],
            dtype=np.float64,
        )