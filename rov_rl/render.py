"""Offscreen rendering of recorded episodes with the MJCF chase camera.

Requires an OpenGL backend (headless: MUJOCO_GL=egl). Markers drawn into the
scene (not part of the physics model): reference point (green), waypoints
(yellow), and the water-current vector at the vehicle (cyan arrow, 1 m per
0.25 m/s).
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np

from rov_rl import core
from rov_rl.rollout import EpisodeLog
from rov_rl.vehicle import load_model

CURRENT_ARROW_M_PER_MPS = 4.0
MIN_ARROW_SPEED_MPS = 1e-3


def _add_geom(scene: mujoco.MjvScene, gtype: mujoco.mjtGeom, size: np.ndarray, pos: np.ndarray,
              mat: np.ndarray, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return  # scene full: drop the marker rather than corrupt the buffer
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype, size, pos, mat.reshape(-1), rgba)
    scene.ngeom += 1


def _arrow(scene: mujoco.MjvScene, start: np.ndarray, vec: np.ndarray, rgba: np.ndarray) -> None:
    length = float(np.linalg.norm(vec))
    if length < MIN_ARROW_SPEED_MPS * CURRENT_ARROW_M_PER_MPS:
        return
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.015, start, start + vec)
    scene.ngeom += 1


def render_episode(
    log: EpisodeLog,
    params: core.EpisodeParams,
    ctrl_dt_s: float,
    width: int = 960,
    height: int = 540,
) -> list[np.ndarray]:
    """One RGB frame per control step (real-time at fps = 1 / ctrl_dt_s)."""
    model = load_model()
    data = mujoco.MjData(model)
    frames: list[np.ndarray] = []
    eye = np.eye(3)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for i, qpos in enumerate(log.qpos):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)  # also updates camera poses
            renderer.update_scene(data, camera="chase")
            scene = renderer.scene
            scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 1
            for wp in np.asarray(params.waypoints_m):
                _add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.04, 0, 0]), wp, eye,
                          np.array([1.0, 0.85, 0.1, 0.8]))
            _add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.06, 0, 0]), log.ref_pos_m[i], eye,
                      np.array([0.1, 1.0, 0.3, 0.7]))
            current = core.current_velocity(np, i * ctrl_dt_s, params)
            _arrow(scene, qpos[:3] + np.array([0.0, 0.0, 0.45]), current * CURRENT_ARROW_M_PER_MPS,
                   np.array([0.2, 0.9, 1.0, 0.9]))
            frames.append(renderer.render().copy())
    return frames


def write_video(frames: list[np.ndarray], path: pathlib.Path, fps: float) -> None:
    import imageio.v2 as imageio  # optional dependency, only needed for videos

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
