"""Videos and plots for the nav task.

iter_nav_frames(): chase camera behind and above the vehicle in the murky-water look of the control-task
videos (fog / haze from the MJCF; fog end extended for the larger scene, see nav_cpu_env.NAV_FOG_END_M), the
obstacle field, and scene markers that are not part of the physics: the measured sonar fan (one line per
beam, coloured by range: red near -> yellow -> green far; beams with no echo faint cyan), the goal (green
sphere), the start (yellow), the travelled path (white line) and the current vector (cyan arrow, as in
rov_rl/render.py). Requires an OpenGL backend (headless: MUJOCO_GL=egl).

write_nav_videos() streams per-controller and side-by-side MP4s (imageio + imageio-ffmpeg; captions need
Pillow). plot_paths(): overhead (top-down) plot of one obstacle field with the paths of several controllers
(requires matplotlib).
"""

from __future__ import annotations

import math
import pathlib
from typing import Iterator

import mujoco
import numpy as np

from rov_rl import core, nav_core
from rov_rl.config import SonarConfig
from rov_rl.nav_cpu_env import build_scene_model
from rov_rl.nav_rollout import NavEpisodeLog
from rov_rl.render import CURRENT_ARROW_M_PER_MPS, _add_geom, _arrow

CAMERA_DISTANCE_M = 9.0
CAMERA_ELEVATION_DEG = -38.0
CAMERA_YAW_SMOOTHING = 0.05  # per frame, first-order filter on the camera azimuth
SONAR_LINE_WIDTH_PX = 2.5
TRAIL_LINE_WIDTH_PX = 2.0
TRAIL_STRIDE = 5  # control steps between trail vertices
NO_ECHO_RGBA = np.array([0.4, 0.9, 1.0, 0.18])
SEPARATOR_PX = 6


def range_rgba(range_m: float, max_range_m: float) -> np.ndarray:
    """Red (0 m) -> yellow (half range) -> green (max range); faint cyan for no echo."""
    if range_m >= max_range_m - 1e-6:
        return NO_ECHO_RGBA.copy()
    x = float(np.clip(range_m / max_range_m, 0.0, 1.0))
    if x < 0.5:
        return np.array([1.0, 2.0 * x, 0.0, 0.95])
    return np.array([2.0 * (1.0 - x), 1.0, 0.0, 0.95])


def _line(scene: mujoco.MjvScene, a: np.ndarray, b: np.ndarray, rgba: np.ndarray, width_px: float) -> None:
    if scene.ngeom >= scene.maxgeom:
        return  # scene full: drop the marker rather than corrupt the buffer
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width_px, a, b)
    scene.ngeom += 1


def iter_nav_frames(
    log: NavEpisodeLog,
    params: nav_core.NavEpisode,
    sonar: SonarConfig,
    ctrl_dt_s: float,
    frame_stride: int = 2,
    width: int = 960,
    height: int = 540,
    caption: str | None = None,
) -> Iterator[np.ndarray]:
    """Yield one RGB frame every `frame_stride` control steps (real time at fps = 1 / (ctrl_dt_s * frame_stride)).

    A generator, so a 100 s episode never has to be held in memory as a frame list.
    """
    model = build_scene_model(params.scene)
    data = mujoco.MjData(model)
    dirs = nav_core.sonar_dirs_body(sonar)
    mid_row = int(np.argmin(np.abs(np.asarray(sonar.elevation_rows_rad))))
    fan_dirs = dirs[mid_row * sonar.num_beams : (mid_row + 1) * sonar.num_beams]
    mount = np.asarray(sonar.mount_pos_m)
    eye = np.eye(3)
    goal = np.asarray(params.scene.goal_m, dtype=float)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = CAMERA_DISTANCE_M
    cam.elevation = CAMERA_ELEVATION_DEG
    cam_yaw = None
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for i in range(0, log.qpos.shape[0], frame_stride):
            qpos = log.qpos[i]
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            rot = core.quat_to_mat(np, qpos[3:7])
            yaw = float(core.yaw_of(np, rot))
            cam_yaw = yaw if cam_yaw is None else cam_yaw + CAMERA_YAW_SMOOTHING * frame_stride * float(
                core.wrap_angle(np, yaw - cam_yaw))
            cam.azimuth = math.degrees(cam_yaw)
            cam.lookat[:] = qpos[:3]
            renderer.update_scene(data, camera=cam)
            scene = renderer.scene
            scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 1
            _add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.3, 0, 0]), goal, eye,
                      np.array([0.1, 1.0, 0.3, 0.6]))
            _add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.12, 0, 0]), log.qpos[0, :3], eye,
                      np.array([1.0, 0.85, 0.1, 0.8]))
            trail = log.qpos[: i + 1 : TRAIL_STRIDE, :3]
            for a, b in zip(trail[:-1], trail[1:]):
                _line(scene, a, b, np.array([1.0, 1.0, 1.0, 0.6]), TRAIL_LINE_WIDTH_PX)
            origin = qpos[:3] + rot @ mount
            for d_body, r in zip(fan_dirs, log.sonar_range_m[i]):
                _line(scene, origin, origin + float(r) * (rot @ d_body), range_rgba(float(r), sonar.max_range_m),
                      SONAR_LINE_WIDTH_PX)
            current = core.current_velocity(np, i * ctrl_dt_s, params.vehicle)
            _arrow(scene, qpos[:3] + np.array([0.0, 0.0, 0.6]), current * CURRENT_ARROW_M_PER_MPS,
                   np.array([0.2, 0.9, 1.0, 0.9]))
            frame = renderer.render().copy()
            yield label_frame(frame, caption) if caption else frame


def label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a caption into the top-left corner of a frame."""
    from PIL import Image, ImageDraw  # Pillow ships with imageio

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, 16 + 7 * len(text), 30], fill=(0, 0, 0))
    draw.text((12, 12), text, fill=(255, 255, 255))
    return np.asarray(img)


def side_by_side_frame(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    sep = np.zeros((left.shape[0], SEPARATOR_PX, 3), dtype=left.dtype)
    return np.concatenate([left, sep, right], axis=1)


def write_nav_videos(out_dir: pathlib.Path, stem: str, params: nav_core.NavEpisode, logs: dict[str, NavEpisodeLog],
                     sonar: SonarConfig, ctrl_dt_s: float, frame_stride: int,
                     pair: tuple[str, str] | None = None) -> list[pathlib.Path]:
    """Stream one MP4 per controller, plus `pair[0] | pair[1]` side by side when both are present.

    All clips advance in lockstep; a clip that ended earlier holds its last frame in the side-by-side video.
    Requires imageio + imageio-ffmpeg.
    """
    import imageio.v2 as imageio  # optional dependency, only needed for videos

    fps = 1.0 / (ctrl_dt_s * frame_stride)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    gens = {name: iter_nav_frames(log, params, sonar, ctrl_dt_s, frame_stride,
                                  caption=f"{name}: {log.outcome}, {log.t_s[-1]:.0f} s")
            for name, log in logs.items()}
    paths = {name: out_dir / "videos" / f"{stem}_{name}.mp4" for name in logs}
    writer_args = dict(fps=fps, codec="libx264", quality=7, macro_block_size=8)
    writers = {name: imageio.get_writer(path, **writer_args) for name, path in paths.items()}
    use_pair = pair is not None and pair[0] in logs and pair[1] in logs
    if use_pair:
        paths["pair"] = out_dir / "videos" / f"{stem}_{pair[0]}_vs_{pair[1]}.mp4"
        writers["pair"] = imageio.get_writer(paths["pair"], **writer_args)
    last: dict[str, np.ndarray] = {}
    active = set(gens)
    try:
        while active:
            for name in list(active):
                frame = next(gens[name], None)
                if frame is None:
                    active.discard(name)
                    continue
                last[name] = frame
                writers[name].append_data(frame)
            if use_pair and active:
                writers["pair"].append_data(side_by_side_frame(last[pair[0]], last[pair[1]]))
    finally:
        for g in gens.values():
            g.close()  # releases each renderer's GL context even if writing failed part-way
        for w in writers.values():
            w.close()
    return list(paths.values())


def _capsule_footprint(a: np.ndarray, b: np.ndarray, radius_m: float) -> np.ndarray:
    """Top-down outline (K, 2) of a horizontal capsule: a rectangle with semicircular ends."""
    axis = b[:2] - a[:2]
    length = float(np.linalg.norm(axis))
    heading = math.atan2(axis[1], axis[0]) if length > 0.0 else 0.0
    arc = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 12)
    end_b = np.stack([np.cos(heading + arc), np.sin(heading + arc)], -1) * radius_m + b[:2]
    end_a = np.stack([np.cos(heading + math.pi + arc), np.sin(heading + math.pi + arc)], -1) * radius_m + a[:2]
    return np.concatenate([end_b, end_a], axis=0)


def plot_paths(params: nav_core.NavEpisode, logs: dict[str, NavEpisodeLog], path: pathlib.Path, title: str,
               goal_radius_m: float) -> None:
    """Overhead plot: obstacle footprints, start, goal (with goal radius) and each controller's path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    sc = params.scene
    start_z = float(params.vehicle.start_pos_m[2])
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in np.flatnonzero(np.asarray(sc.cap_active) > 0.5):
        a, b, r = np.asarray(sc.cap_a_m[i]), np.asarray(sc.cap_b_m[i]), float(sc.cap_radius_m[i])
        if sc.cap_is_pipe[i] > 0.5:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#8c5a33", lw=2.0, solid_capstyle="round", alpha=0.9,
                    label="pipe span" if "pipe span" not in ax.get_legend_handles_labels()[1] else None)
            ax.add_patch(Polygon(_capsule_footprint(a, b, r), closed=True, color="#8c5a33", alpha=0.35))
        else:
            ax.add_patch(Circle((a[0], a[1]), r, color="#5f675c", alpha=0.9))
    for i in np.flatnonzero(np.asarray(sc.box_active) > 0.5):
        c, h, yaw = np.asarray(sc.box_center_m[i]), np.asarray(sc.box_half_m[i]), float(sc.box_yaw_rad[i])
        rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        corners = (rot @ (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * h[:2]).T).T + c[:2]
        blocking = c[2] + h[2] > start_z - nav_core.VEHICLE_BOUNDING_RADIUS_M
        ax.add_patch(Polygon(corners, closed=True, color="#4a4236" if blocking else "#a89c84", alpha=0.85))
    goal = np.asarray(sc.goal_m)
    ax.add_patch(Circle((goal[0], goal[1]), goal_radius_m, fill=False, color="green", lw=1.5))
    ax.plot([0.0], [0.0], marker="o", color="gold", ms=9, label="start")
    ax.plot([goal[0]], [goal[1]], marker="*", color="green", ms=14, label="goal")
    colors = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c"]
    for (name, log), color in zip(logs.items(), colors):
        xy = log.qpos[:, :2]
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2.0, label=f"{name} ({log.outcome}, {log.t_s[-1]:.0f} s)")
        if log.outcome == "collision":
            ax.plot([xy[-1, 0]], [xy[-1, 1]], marker="x", color=color, ms=12, mew=3)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title + "\n(dark boxes reach the start depth, light boxes lie below it)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
