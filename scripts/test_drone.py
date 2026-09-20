import numpy as np

from drone.env import CrazyflieEnv


env = CrazyflieEnv(
    randomize_start=False,
)

ts = env.reset()

command = np.array(
    [
        0.0,  # vx
        0.0,  # vy
        0.0,  # vz
        0.0,  # yaw rate
    ],
    dtype=np.float32,
)

positions = []

while not ts.last():
    ts = env.step(command)

    positions.append(
        ts.observation[:3].copy()
    )

positions = np.asarray(positions)

print("Initial:", positions[0])
print("Final:", positions[-1])
print("Z range:", positions[:, 2].min(), positions[:, 2].max())