from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from rldp.buffers.buffers import DictBuffer
from rldp.rldp import FBAgent, FBAgentConfig


torch.set_float32_matmul_precision(
    "high"
)


def load_dataset(
    dataset_root: str | Path,
    num_episodes: int,
) -> dict:

    root = Path(dataset_root) / "buffer"

    files = sorted(
        root.glob("*.npz")
    )

    files = files[
        : min(
            num_episodes,
            len(files),
        )
    ]

    if not files:
        raise RuntimeError(
            f"No episodes found in {root}"
        )

    storage = {
        "observation": [],
        "action": [],
        "next": {
            "observation": [],
            "terminated": [],
        },
    }

    for path in files:
        data = np.load(path)

        if (
            data["observation"].shape[0]
            != 1001
        ):
            raise ValueError(
                f"{path} must contain 1001 observations"
            )

        storage[
            "observation"
        ].append(
            data["observation"][:-1]
            .astype(np.float32)
        )

        storage[
            "action"
        ].append(
            data["action"][1:]
            .astype(np.float32)
        )

        storage[
            "next"
        ][
            "observation"
        ].append(
            data["observation"][1:]
            .astype(np.float32)
        )

        storage[
              "next"
            ][
              "terminated"
            ].append(
            np.asarray(
              1 - data["discount"][1:],
              dtype=np.bool_,
            ).reshape(-1, 1)
        )

    storage[
        "observation"
    ] = np.concatenate(
        storage["observation"]
    )

    storage[
        "action"
    ] = np.concatenate(
        storage["action"]
    )

    for key in storage["next"]:
        storage["next"][key] = (
            np.concatenate(
                storage["next"][key]
            )
        )

    n = len(
        storage["observation"]
    )

    storage[
        "next_observation_hash"
    ] = np.random.permutation(
        np.arange(n)
    ).reshape(-1, 1)

    return storage


def make_agent_config(
    *,
    device: str,
    representation_steps: int,
    horizon: int,
    ortho_coef: float,
    compile_model: bool,
    cudagraphs: bool,
) -> FBAgentConfig:

    cfg = FBAgentConfig()

    cfg.model.obs_dim = 13
    cfg.model.action_dim = 4

    cfg.model.device = device
    cfg.model.norm_obs = False
    cfg.model.seq_length = 1

    cfg.model.archi.z_dim = 512

    cfg.model.archi.b.norm = True
    cfg.model.archi.norm_z = True

    cfg.model.archi.b.hidden_dim = 256
    cfg.model.archi.b.hidden_layers = 2

    cfg.model.archi.f.hidden_dim = 1024
    cfg.model.archi.f.hidden_layers = 1

    cfg.model.archi.actor.hidden_dim = 1024
    cfg.model.archi.actor.hidden_layers = 1

    cfg.train.batch_size = 1024

    cfg.train.lr_f = 1e-4
    cfg.train.lr_b = 1e-4
    cfg.train.lr_actor = 1e-4

    cfg.train.discount = 0.99

    cfg.train.ortho_coef = ortho_coef
    cfg.train.enc_horizon = horizon
    cfg.train.representation_steps = (
        representation_steps
    )

    cfg.train.train_goal_ratio = 0.5

    cfg.train.fb_pessimism_penalty = 0.0
    cfg.train.actor_pessimism_penalty = 0.5

    cfg.compile = compile_model
    cfg.cudagraphs = cudagraphs

    return cfg


def main():
    parser = argparse.ArgumentParser()

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
        "--episodes",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=300_000,
    )

    parser.add_argument(
        "--representation-steps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--ortho-coef",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--compile",
        action="store_true",
    )

    parser.add_argument(
        "--cudagraphs",
        action="store_true",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    output = Path(args.output)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = make_agent_config(
        device=args.device,
        representation_steps=(
            args.representation_steps
        ),
        horizon=args.horizon,
        ortho_coef=args.ortho_coef,
        compile_model=args.compile,
        cudagraphs=args.cudagraphs,
    )

    agent = FBAgent(
        **dataclasses.asdict(config)
    )

    data = load_dataset(
        args.dataset,
        args.episodes,
    )

    replay = DictBuffer(
        capacity=len(
            data["observation"]
        ),
        device=agent.device,
    )

    replay.extend(data)

    print(replay)

    log_file = (
        output
        / "train.jsonl"
    )

    accumulated = None

    for step in trange(
        args.steps
    ):
        metrics = agent.update(
            {"train": replay},
            step,
        )

        current = {
            k: float(v.mean().item())
            for k, v in metrics.items()
        }

        if accumulated is None:
            accumulated = {
                k: 0.0
                for k in current
            }

        for key, value in current.items():
            accumulated[key] += value

        if (
            (step + 1)
            % args.log_every
            == 0
        ):
            averaged = {
                key: value
                / args.log_every
                for key, value
                in accumulated.items()
            }

            averaged["step"] = (
                step + 1
            )

            print(averaged)

            with log_file.open(
                "a"
            ) as f:
                f.write(
                    json.dumps(
                        averaged
                    )
                    + "\n"
                )

            accumulated = None

        if (
            (step + 1)
            % args.save_every
            == 0
        ):
            checkpoint_dir = (
                output
                / "checkpoints"
                / f"step_{step + 1:07d}"
            )

            checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            agent.save(
                str(checkpoint_dir)
            )

    agent.save(
        str(
            output
            / "checkpoint"
        )
    )


if __name__ == "__main__":
    main()