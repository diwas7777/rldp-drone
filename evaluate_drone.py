from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from drone.env import CrazyflieEnv
from drone.rewards import TASKS, goal_reward

from rldp.nn_models import eval_mode
from rldp.rldp import FBAgent


def sample_offline_observations(
    dataset_root: str | Path,
    *,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Uniformly sample next-state observations from the offline dataset.

    All drone episodes contain 1000 transitions and 1001 observations.
    """

    root = Path(dataset_root) / "buffer"

    files = sorted(root.glob("*.npz"))

    if not files:
        raise RuntimeError(
            f"No .npz episodes found in {root}"
        )

    rng = np.random.default_rng(seed)

    samples: list[np.ndarray] = []
    collected = 0

    while collected < num_samples:
        file_index = int(
            rng.integers(0, len(files))
        )

        with np.load(files[file_index]) as data:
            observations = data["observation"]

            if observations.shape[0] != 1001:
                raise ValueError(
                    f"{files[file_index]} has "
                    f"{observations.shape[0]} observations; "
                    "expected 1001"
                )

            remaining = num_samples - collected

            n = min(
                remaining,
                observations.shape[0] - 1,
            )

            indices = rng.integers(
                1,
                observations.shape[0],
                size=n,
            )

            batch = observations[
                indices
            ].astype(
                np.float32,
                copy=False,
            )

            samples.append(batch)

            collected += n

    return np.concatenate(
        samples,
        axis=0,
    )[:num_samples]


@torch.no_grad()
def infer_task_z(
    agent: FBAgent,
    offline_observations: np.ndarray,
    goal: np.ndarray,
) -> torch.Tensor:
    """
    Infer the RLDP latent task vector from an unseen reward function.
    """

    rewards = goal_reward(
        offline_observations,
        goal,
    )

    next_obs = torch.as_tensor(
        offline_observations,
        dtype=torch.float32,
        device=agent.device,
    )

    reward = torch.as_tensor(
        rewards,
        dtype=torch.float32,
        device=agent.device,
    ).reshape(-1, 1)

    z = agent._model.reward_inference(
        next_obs=next_obs,
        reward=reward,
    )

    return z


def render_frame(
    env: CrazyflieEnv,
    *,
    height: int = 480,
    width: int = 640,
) -> np.ndarray:
    """
    Render the Crazyflie rollout.

    Prefer the Menagerie tracking camera. Fall back to camera 0
    if the named camera is unavailable.
    """

    try:
        return env.physics.render(
            height=height,
            width=width,
            camera_id="track",
        )
    except (ValueError, KeyError):
        return env.physics.render(
            height=height,
            width=width,
            camera_id=0,
        )


def evaluate_episode(
    *,
    agent: FBAgent,
    env: CrazyflieEnv,
    z: torch.Tensor,
    goal: np.ndarray,
    success_radius: float,
    success_speed: float,
    success_hold_steps: int,
    video_path: Path | None = None,
    video_fps: int = 25,
) -> dict[str, float | int | bool]:
    ts = env.reset()

    total_reward = 0.0
    min_distance = float("inf")

    success_counter = 0
    success = False
    time_to_goal = None

    step = 0

    writer = None

    if video_path is not None:
        video_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        writer = imageio.get_writer(
            str(video_path),
            fps=video_fps,
            codec="libx264",
        )

        writer.append_data(
            render_frame(env)
        )

    try:
        while not ts.last():
            obs_np = np.asarray(
                ts.observation,
                dtype=np.float32,
            )

            obs = torch.as_tensor(
                obs_np,
                dtype=torch.float32,
                device=agent.device,
            ).reshape(1, -1)

            with torch.no_grad(), eval_mode(
                agent._model
            ):
                action = agent.act(
                    obs=obs,
                    z=z,
                    mean=True,
                )

            action_np = (
                action
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            ts = env.step(
                action_np
            )

            # Environment runs at 50 Hz.
            # Record every second step for 25 FPS video.
            if (
                writer is not None
                and step % 2 == 0
            ):
                writer.append_data(
                    render_frame(env)
                )

            next_obs = np.asarray(
                ts.observation,
                dtype=np.float32,
            )

            reward = float(
                goal_reward(
                    next_obs[None, :],
                    goal,
                )[0]
            )

            total_reward += reward

            position = next_obs[0:3]
            velocity = next_obs[3:6]

            distance = float(
                np.linalg.norm(
                    position - goal
                )
            )

            speed = float(
                np.linalg.norm(
                    velocity
                )
            )

            min_distance = min(
                min_distance,
                distance,
            )

            if (
                distance <= success_radius
                and speed <= success_speed
            ):
                success_counter += 1
            else:
                success_counter = 0

            if (
                not success
                and success_counter
                >= success_hold_steps
            ):
                success = True

                time_to_goal = (
                    step
                    - success_hold_steps
                    + 2
                ) * env.control_timestep

            step += 1

    finally:
        if writer is not None:
            writer.close()

    final_obs = np.asarray(
        ts.observation,
        dtype=np.float32,
    )

    final_distance = float(
        np.linalg.norm(
            final_obs[0:3] - goal
        )
    )

    return {
        "return": total_reward,
        "success": success,
        "min_distance": min_distance,
        "final_distance": final_distance,
        "time_to_goal": (
            float(time_to_goal)
            if time_to_goal is not None
            else -1.0
        ),
        "steps": step,
    }


def evaluate_task(
    *,
    agent: FBAgent,
    dataset_root: Path,
    task_name: str,
    goal: np.ndarray,
    episodes: int,
    num_inference_samples: int,
    seed: int,
    success_radius: float,
    success_speed: float,
    success_hold_steps: int,
    video_dir: Path | None = None,
) -> tuple[dict, list[dict]]:
    offline_observations = (
        sample_offline_observations(
            dataset_root,
            num_samples=num_inference_samples,
            seed=seed,
        )
    )

    z = infer_task_z(
        agent,
        offline_observations,
        goal,
    )

    episode_results = []

    for episode in range(episodes):
        env = CrazyflieEnv(
            seed=seed + episode,
            randomize_start=True,
        )

        # Record only the first rollout for each task.
        video_path = None

        if (
            video_dir is not None
            and episode == 0
        ):
            video_path = (
                video_dir
                / f"{task_name}.mp4"
            )

        result = evaluate_episode(
            agent=agent,
            env=env,
            z=z,
            goal=goal,
            success_radius=success_radius,
            success_speed=success_speed,
            success_hold_steps=success_hold_steps,
            video_path=video_path,
        )

        result["episode"] = episode
        result["task"] = task_name

        episode_results.append(
            result
        )

    returns = np.asarray(
        [
            x["return"]
            for x in episode_results
        ],
        dtype=np.float64,
    )

    successes = np.asarray(
        [
            x["success"]
            for x in episode_results
        ],
        dtype=np.float64,
    )

    final_distances = np.asarray(
        [
            x["final_distance"]
            for x in episode_results
        ],
        dtype=np.float64,
    )

    min_distances = np.asarray(
        [
            x["min_distance"]
            for x in episode_results
        ],
        dtype=np.float64,
    )

    successful_times = [
        x["time_to_goal"]
        for x in episode_results
        if x["success"]
    ]

    summary = {
        "task": task_name,

        "goal": goal.tolist(),

        "episodes": episodes,

        "return_mean": float(
            returns.mean()
        ),

        "return_std": float(
            returns.std()
        ),

        "success_rate": float(
            successes.mean()
        ),

        "final_distance_mean": float(
            final_distances.mean()
        ),

        "min_distance_mean": float(
            min_distances.mean()
        ),

        "time_to_goal_mean": (
            float(
                np.mean(
                    successful_times
                )
            )
            if successful_times
            else None
        ),
    }

    return (
        summary,
        episode_results,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(
            TASKS.keys()
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--num-inference-samples",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--success-radius",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--success-speed",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--success-hold-steps",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help=(
            "Disable evaluation video recording."
        ),
    )

    args = parser.parse_args()

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_dir = (
        None
        if args.no_video
        else output / "videos"
    )

    agent = FBAgent.load(
        args.checkpoint,
        device=args.device,
    )

    summaries = {}
    all_episodes = []

    for task_name in args.tasks:
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task: {task_name}"
            )

        goal = TASKS[
            task_name
        ]

        print(
            f"Evaluating {task_name}: "
            f"{goal.tolist()}"
        )

        summary, episodes = (
            evaluate_task(
                agent=agent,
                dataset_root=Path(
                    args.dataset
                ),
                task_name=task_name,
                goal=goal,
                episodes=args.episodes,
                num_inference_samples=(
                    args.num_inference_samples
                ),
                seed=args.seed,
                success_radius=(
                    args.success_radius
                ),
                success_speed=(
                    args.success_speed
                ),
                success_hold_steps=(
                    args.success_hold_steps
                ),
                video_dir=video_dir,
            )
        )

        summaries[
            task_name
        ] = summary

        all_episodes.extend(
            episodes
        )

        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

    summary_path = (
        output
        / "summary.json"
    )

    with summary_path.open(
        "w"
    ) as f:
        json.dump(
            summaries,
            f,
            indent=2,
        )

        f.write("\n")

    if all_episodes:
        fieldnames = list(
            all_episodes[
                0
            ].keys()
        )

        episodes_path = (
            output
            / "episodes.csv"
        )

        with episodes_path.open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                all_episodes
            )

    print()
    print(
        f"Summary: {summary_path}"
    )

    if video_dir is not None:
        print(
            f"Videos: {video_dir}"
        )


if __name__ == "__main__":
    main()