from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from drone.env import CrazyflieEnv


def render_frame(
    env: CrazyflieEnv,
    *,
    camera: str | int,
    width: int,
    height: int,
) -> np.ndarray:
    try:
        return env.physics.render(
            width=width,
            height=height,
            camera_id=camera,
        )
    except (ValueError, KeyError):
        if camera == "track":
            return env.physics.render(
                width=width,
                height=height,
                camera_id=0,
            )

        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render a recorded Crazyflie .npz trajectory "
            "to an MP4 video."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to an episode .npz file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output MP4 path. "
            "Defaults to <input>.mp4."
        ),
    )

    parser.add_argument(
        "--camera",
        default="track",
        help=(
            "MuJoCo camera name or integer camera ID. "
            "Default: track"
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video width in pixels. Default: 640",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height in pixels. Default: 480",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help=(
            "Render every Nth recorded state. "
            "At 50 Hz, stride=2 gives 25 FPS."
        ),
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Video FPS. Default: 25",
    )

    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
    else:
        dataset_root = input_path.parent.parent
        video_dir = dataset_root / "videos"

        output_path = (
            video_dir
            / input_path.with_suffix(".mp4").name
        )

    if args.stride < 1:
        raise ValueError(
            "--stride must be at least 1"
        )

    # Convert numeric camera strings such as "0"
    # into integer camera IDs.
    camera: str | int

    try:
        camera = int(args.camera)
    except ValueError:
        camera = args.camera

    with np.load(input_path) as data:
        print(f"Loading: {input_path}")
        print(f"Fields: {data.files}")

        if "physics" not in data.files:
            raise RuntimeError(
                "This dataset does not contain a 'physics' field. "
                "Exact trajectory rendering requires the saved "
                "MuJoCo physics states."
            )

        physics_states = np.asarray(
            data["physics"]
        )

        observations = (
            np.asarray(data["observation"])
            if "observation" in data.files
            else None
        )

        print(
            "physics:",
            physics_states.shape,
            physics_states.dtype,
        )

        if observations is not None:
            print(
                "observation:",
                observations.shape,
                observations.dtype,
            )

    env = CrazyflieEnv(
        randomize_start=False,
    )

    # Initialize the model/data before restoring
    # recorded states.
    env.reset()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = imageio.get_writer(
        str(output_path),
        fps=args.fps,
        codec="libx264",
    )

    frame_count = 0

    try:
        for index in range(
            0,
            len(physics_states),
            args.stride,
        ):
            state = physics_states[index]

            env.physics.set_state(
                state
            )

            # Recompute derived MuJoCo quantities after
            # restoring qpos/qvel/etc.
            env.physics.forward()

            frame = render_frame(
                env,
                camera=camera,
                width=args.width,
                height=args.height,
            )

            writer.append_data(
                frame
            )

            frame_count += 1

            if frame_count % 100 == 0:
                print(
                    f"Rendered {frame_count} frames"
                )

    finally:
        writer.close()

    duration = (
        frame_count / args.fps
        if args.fps > 0
        else 0.0
    )

    print()
    print("Done.")
    print(f"Frames:   {frame_count}")
    print(f"FPS:      {args.fps}")
    print(f"Duration: {duration:.2f} s")
    print(f"Video:    {output_path}")


if __name__ == "__main__":
    main()