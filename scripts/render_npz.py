from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from drone.env import CrazyflieEnv


CONTROL_HZ = 50.0

PHASE_APPROACH = 0
PHASE_HOLD = 1

WAYPOINT_RADIUS = 0.30

# Recent trajectory shown more strongly on the XY map.
TRAIL_STEPS = 100


def get_font(
    size: int,
) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSansMono.ttf",
            size,
        )
    except OSError:
        return ImageFont.load_default()


def add_scene_sphere(
    scene,
    *,
    position: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return

    # dm_control exposes scene.geoms only up to scene.ngeom.
    # Increase ngeom first so the newly allocated slot becomes
    # accessible through scene.geoms[-1].
    scene.ngeom += 1

    try:
        geom = scene.geoms[-1]

        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array(
                [
                    radius,
                    radius,
                    radius,
                ],
                dtype=np.float64,
            ),
            np.asarray(
                position,
                dtype=np.float64,
            ),
            np.eye(
                3,
                dtype=np.float64,
            ).reshape(-1),
            np.asarray(
                rgba,
                dtype=np.float32,
            ),
        )

        geom.category = (
            mujoco.mjtCatBit.mjCAT_DECOR
        )

    except Exception:
        # Restore the original scene count if initialization fails.
        scene.ngeom -= 1
        raise


def make_waypoint_scene_callback(
    waypoint: np.ndarray,
):
    waypoint = np.asarray(
        waypoint,
        dtype=np.float64,
    ).copy()

    def callback(
        _physics,
        scene,
    ) -> None:
        # Faint sphere showing the 0.30 m waypoint acceptance region.
        add_scene_sphere(
            scene,
            position=waypoint,
            radius=WAYPOINT_RADIUS,
            rgba=(
                1.0,
                0.72,
                0.05,
                0.08,
            ),
        )

        # Solid marker showing the exact waypoint.
        add_scene_sphere(
            scene,
            position=waypoint,
            radius=0.055,
            rgba=(
                1.0,
                0.72,
                0.05,
                0.95,
            ),
        )

    return callback


def render_frame(
    env: CrazyflieEnv,
    *,
    camera: str | int,
    width: int,
    height: int,
    waypoint: np.ndarray | None,
    show_waypoint: bool,
) -> np.ndarray:
    scene_callback = None

    if (
        show_waypoint
        and waypoint is not None
    ):
        scene_callback = (
            make_waypoint_scene_callback(
                waypoint
            )
        )

    def do_render(
        camera_id: str | int,
    ) -> np.ndarray:
        return env.physics.render(
            width=width,
            height=height,
            camera_id=camera_id,
            scene_callback=scene_callback,
        )

    try:
        return do_render(
            camera
        )

    except KeyError:
        if camera == "track":
            return do_render(
                0
            )

        raise

    except ValueError as exc:
        # Only use camera 0 as a fallback when the problem
        # is actually related to the named camera.
        message = str(
            exc
        ).lower()

        if (
            camera == "track"
            and "camera" in message
        ):
            return do_render(
                0
            )

        raise


def compute_xy_bounds(
    observations: np.ndarray | None,
    waypoints: np.ndarray | None,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    points = []

    if observations is not None:
        points.append(
            observations[:, :2]
        )

    if waypoints is not None:
        points.append(
            waypoints[:, :2]
        )

    if not points:
        return (
            -2.5,
            2.5,
            -2.5,
            2.5,
        )

    xy = np.concatenate(
        points,
        axis=0,
    )

    xmin = float(
        xy[:, 0].min()
    )

    xmax = float(
        xy[:, 0].max()
    )

    ymin = float(
        xy[:, 1].min()
    )

    ymax = float(
        xy[:, 1].max()
    )

    xspan = max(
        xmax - xmin,
        1.0,
    )

    yspan = max(
        ymax - ymin,
        1.0,
    )

    # Small margin around actual trajectory coverage.
    xmin -= 0.08 * xspan
    xmax += 0.08 * xspan

    ymin -= 0.08 * yspan
    ymax += 0.08 * yspan

    # Keep world-space scaling equal in X and Y.
    center_x = 0.5 * (
        xmin + xmax
    )

    center_y = 0.5 * (
        ymin + ymax
    )

    span = max(
        xmax - xmin,
        ymax - ymin,
    )

    half = 0.5 * span

    return (
        center_x - half,
        center_x + half,
        center_y - half,
        center_y + half,
    )


def world_to_map(
    x: float,
    y: float,
    *,
    bounds: tuple[
        float,
        float,
        float,
        float,
    ],
    left: int,
    top: int,
    size: int,
) -> tuple[int, int]:
    xmin, xmax, ymin, ymax = (
        bounds
    )

    u = (
        (x - xmin)
        / max(
            xmax - xmin,
            1e-8,
        )
    )

    v = (
        (y - ymin)
        / max(
            ymax - ymin,
            1e-8,
        )
    )

    px = (
        left
        + int(
            np.clip(
                u,
                0.0,
                1.0,
            )
            * size
        )
    )

    # Image Y grows downward.
    py = (
        top
        + size
        - int(
            np.clip(
                v,
                0.0,
                1.0,
            )
            * size
        )
    )

    return (
        px,
        py,
    )


def draw_telemetry(
    draw: ImageDraw.ImageDraw,
    *,
    index: int,
    observations: np.ndarray | None,
    waypoints: np.ndarray | None,
    phases: np.ndarray | None,
) -> None:
    font = get_font(
        12
    )

    left = 10
    top = 10

    width = 228
    height = 102

    # Much lighter than the previous solid black panel.
    draw.rounded_rectangle(
        (
            left,
            top,
            left + width,
            top + height,
        ),
        radius=7,
        fill=(
            8,
            14,
            22,
            105,
        ),
        outline=(
            255,
            255,
            255,
            35,
        ),
        width=1,
    )

    time_seconds = (
        index / CONTROL_HZ
    )

    phase_name = "UNKNOWN"

    if (
        phases is not None
        and index < len(phases)
    ):
        phase_name = (
            "HOLD"
            if int(
                phases[index]
            )
            == PHASE_HOLD
            else "APPROACH"
        )

    if phase_name == "HOLD":
        phase_fill = (
            120,
            255,
            150,
            255,
        )
    else:
        phase_fill = (
            255,
            215,
            95,
            255,
        )

    draw.text(
        (
            left + 9,
            top + 7,
        ),
        phase_name,
        font=font,
        fill=phase_fill,
    )

    draw.text(
        (
            left + 105,
            top + 7,
        ),
        f"t {time_seconds:5.1f}s",
        font=font,
        fill=(
            235,
            235,
            235,
            255,
        ),
    )

    position = None

    if (
        observations is not None
        and index < len(observations)
    ):
        obs = observations[
            index
        ]

        position = obs[:3]
        velocity = obs[3:6]

        speed = float(
            np.linalg.norm(
                velocity
            )
        )

        draw.text(
            (
                left + 9,
                top + 27,
            ),
            (
                "pos  "
                f"{position[0]:+4.2f} "
                f"{position[1]:+4.2f} "
                f"{position[2]:+4.2f}"
            ),
            font=font,
            fill=(
                245,
                245,
                245,
                255,
            ),
        )

        draw.text(
            (
                left + 9,
                top + 45,
            ),
            f"speed  {speed:4.2f} m/s",
            font=font,
            fill=(
                245,
                245,
                245,
                255,
            ),
        )

    if (
        waypoints is not None
        and index < len(waypoints)
    ):
        waypoint = waypoints[
            index
        ]

        draw.text(
            (
                left + 9,
                top + 63,
            ),
            (
                "wp   "
                f"{waypoint[0]:+4.2f} "
                f"{waypoint[1]:+4.2f} "
                f"{waypoint[2]:+4.2f}"
            ),
            font=font,
            fill=(
                255,
                220,
                90,
                255,
            ),
        )

        if position is not None:
            distance = float(
                np.linalg.norm(
                    position - waypoint
                )
            )

            draw.text(
                (
                    left + 9,
                    top + 81,
                ),
                f"dist   {distance:4.2f} m",
                font=font,
                fill=(
                    245,
                    245,
                    245,
                    255,
                ),
            )


def draw_xy_map(
    draw: ImageDraw.ImageDraw,
    *,
    image_width: int,
    index: int,
    observations: np.ndarray,
    waypoints: np.ndarray | None,
    bounds: tuple[
        float,
        float,
        float,
        float,
    ],
) -> None:
    font = get_font(
        10
    )

    # Deliberately small.
    size = 118

    left = (
        image_width
        - size
        - 12
    )

    top = 12

    padding = 5

    draw.rounded_rectangle(
        (
            left - padding,
            top - padding,
            left + size + padding,
            top + size + padding,
        ),
        radius=7,
        fill=(
            8,
            14,
            22,
            90,
        ),
        outline=(
            255,
            255,
            255,
            35,
        ),
        width=1,
    )

    xmin, xmax, ymin, ymax = (
        bounds
    )

    # X = 0 reference.
    if xmin <= 0.0 <= xmax:
        x0, _ = world_to_map(
            0.0,
            ymin,
            bounds=bounds,
            left=left,
            top=top,
            size=size,
        )

        draw.line(
            (
                x0,
                top,
                x0,
                top + size,
            ),
            fill=(
                210,
                210,
                210,
                65,
            ),
            width=1,
        )

    # Y = 0 reference.
    if ymin <= 0.0 <= ymax:
        _, y0 = world_to_map(
            xmin,
            0.0,
            bounds=bounds,
            left=left,
            top=top,
            size=size,
        )

        draw.line(
            (
                left,
                y0,
                left + size,
                y0,
            ),
            fill=(
                210,
                210,
                210,
                65,
            ),
            width=1,
        )

    # Full history, faint.
    history = observations[
        : min(
            index + 1,
            len(observations),
        ),
        :2,
    ]

    if len(history) >= 2:
        history_pixels = [
            world_to_map(
                float(point[0]),
                float(point[1]),
                bounds=bounds,
                left=left,
                top=top,
                size=size,
            )
            for point in history
        ]

        draw.line(
            history_pixels,
            fill=(
                80,
                170,
                230,
                90,
            ),
            width=1,
        )

    # Recent motion, brighter.
    recent_start = max(
        0,
        index - TRAIL_STEPS,
    )

    recent_end = min(
        index + 1,
        len(observations),
    )

    recent = observations[
        recent_start:recent_end,
        :2,
    ]

    if len(recent) >= 2:
        recent_pixels = [
            world_to_map(
                float(point[0]),
                float(point[1]),
                bounds=bounds,
                left=left,
                top=top,
                size=size,
            )
            for point in recent
        ]

        draw.line(
            recent_pixels,
            fill=(
                90,
                205,
                255,
                220,
            ),
            width=2,
        )

    # Current waypoint.
    if (
        waypoints is not None
        and index < len(waypoints)
    ):
        waypoint = waypoints[
            index
        ]

        wx, wy = world_to_map(
            float(
                waypoint[0]
            ),
            float(
                waypoint[1]
            ),
            bounds=bounds,
            left=left,
            top=top,
            size=size,
        )

        world_span = max(
            xmax - xmin,
            1e-8,
        )

        radius_px = max(
            3,
            int(
                WAYPOINT_RADIUS
                / world_span
                * size
            ),
        )

        draw.ellipse(
            (
                wx - radius_px,
                wy - radius_px,
                wx + radius_px,
                wy + radius_px,
            ),
            outline=(
                255,
                210,
                50,
                210,
            ),
            width=1,
        )

        # Target cross.
        draw.line(
            (
                wx - 5,
                wy,
                wx + 5,
                wy,
            ),
            fill=(
                255,
                210,
                50,
                255,
            ),
            width=2,
        )

        draw.line(
            (
                wx,
                wy - 5,
                wx,
                wy + 5,
            ),
            fill=(
                255,
                210,
                50,
                255,
            ),
            width=2,
        )

    # Current drone location.
    if index < len(
        observations
    ):
        position = observations[
            index,
            :2,
        ]

        dx, dy = world_to_map(
            float(
                position[0]
            ),
            float(
                position[1]
            ),
            bounds=bounds,
            left=left,
            top=top,
            size=size,
        )

        radius = 3

        draw.ellipse(
            (
                dx - radius,
                dy - radius,
                dx + radius,
                dy + radius,
            ),
            fill=(
                255,
                90,
                90,
                255,
            ),
            outline=(
                255,
                255,
                255,
                230,
            ),
            width=1,
        )

    draw.text(
        (
            left + 4,
            top + 3,
        ),
        "XY",
        font=font,
        fill=(
            235,
            235,
            235,
            210,
        ),
    )


def annotate_frame(
    frame: np.ndarray,
    *,
    index: int,
    observations: np.ndarray | None,
    waypoints: np.ndarray | None,
    phases: np.ndarray | None,
    xy_bounds: tuple[
        float,
        float,
        float,
        float,
    ],
) -> np.ndarray:
    image = Image.fromarray(
        frame
    ).convert(
        "RGBA"
    )

    overlay = Image.new(
        "RGBA",
        image.size,
        (
            0,
            0,
            0,
            0,
        ),
    )

    draw = ImageDraw.Draw(
        overlay
    )

    draw_telemetry(
        draw,
        index=index,
        observations=observations,
        waypoints=waypoints,
        phases=phases,
    )

    if observations is not None:
        draw_xy_map(
            draw,
            image_width=image.width,
            index=index,
            observations=observations,
            waypoints=waypoints,
            bounds=xy_bounds,
        )

    result = Image.alpha_composite(
        image,
        overlay,
    ).convert(
        "RGB"
    )

    return np.asarray(
        result
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render a recorded Crazyflie trajectory "
            "to an annotated MP4 video."
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
            "Output MP4 path. Defaults to the "
            "dataset's videos directory."
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

    # Keep within the Menagerie model's default offscreen framebuffer.
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
            "Render every Nth state. "
            "At 50 Hz, stride=2 gives 25 rendered "
            "states per simulated second."
        ),
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Video FPS. Default: 25",
    )

    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help=(
            "Disable the telemetry panel and XY map."
        ),
    )

    parser.add_argument(
        "--no-waypoint-marker",
        action="store_true",
        help=(
            "Disable the 3D waypoint marker."
        ),
    )

    args = parser.parse_args()

    input_path = (
        args.input
        .expanduser()
        .resolve()
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    if args.stride < 1:
        raise ValueError(
            "--stride must be at least 1"
        )

    if args.fps < 1:
        raise ValueError(
            "--fps must be at least 1"
        )

    if args.output is not None:
        output_path = (
            args.output
            .expanduser()
            .resolve()
        )

    else:
        dataset_root = (
            input_path
            .parent
            .parent
        )

        output_path = (
            dataset_root
            / "videos"
            / input_path
            .with_suffix(".mp4")
            .name
        )

    try:
        camera: str | int = int(
            args.camera
        )

    except ValueError:
        camera = args.camera

    with np.load(input_path) as data:
        print(f"Loading: {input_path}")
        print(f"Fields: {data.files}")

        if "physics" not in data.files:
            raise RuntimeError(
                "Dataset does not contain 'physics'. "
                "Exact rendering requires saved "
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

        waypoints = (
            np.asarray(data["waypoint"])
            if "waypoint" in data.files
            else None
        )

        phases = (
            np.asarray(data["phase"])
            if "phase" in data.files
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

        if waypoints is not None:
            print(
                "waypoint:",
                waypoints.shape,
                waypoints.dtype,
            )

        if phases is not None:
            print(
                "phase:",
                phases.shape,
                phases.dtype,
            )

    if observations is not None:
        if (
            len(observations)
            != len(physics_states)
        ):
            raise ValueError(
                "observation and physics lengths do not match"
            )

    if waypoints is not None:
        if (
            len(waypoints)
            != len(physics_states)
        ):
            raise ValueError(
                "waypoint and physics lengths do not match"
            )

    if phases is not None:
        if (
            len(phases)
            != len(physics_states)
        ):
            raise ValueError(
                "phase and physics lengths do not match"
            )

    if waypoints is None:
        print(
            "Note: waypoint metadata is not available."
        )

    xy_bounds = compute_xy_bounds(
        observations,
        waypoints,
    )

    env = CrazyflieEnv(
        randomize_start=False,
    )

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

            waypoint = None

            if (
                waypoints is not None
                and index < len(
                    waypoints
                )
            ):
                waypoint = waypoints[
                    index
                ]

            frame = render_frame(
                env,
                camera=camera,
                width=args.width,
                height=args.height,
                waypoint=waypoint,
                show_waypoint=(
                    not args.no_waypoint_marker
                ),
            )

            if not args.no_overlay:
                frame = annotate_frame(
                    frame,
                    index=index,
                    observations=observations,
                    waypoints=waypoints,
                    phases=phases,
                    xy_bounds=xy_bounds,
                )

            writer.append_data(
                frame
            )

            frame_count += 1

            if (frame_count % 100== 0):
                print(
                    f"Rendered {frame_count} frames"
                )

    finally:
        writer.close()

    duration = (
        frame_count
        / args.fps
    )

    print()
    print("Done.")
    print(f"Frames:   {frame_count}")
    print(f"FPS:      {args.fps}")
    print(f"Duration: {duration:.2f} s")
    print(f"Video:    {output_path}")


if __name__ == "__main__":
    main()