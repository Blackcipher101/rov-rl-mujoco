"""Nav sonar ray casting and clearance vs analytic distances and vs MuJoCo's own ray caster."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from rov_rl import core, nav_core
from rov_rl.config import NAV_TRAIN_SCENE, SonarConfig, nav_eval_scene
from rov_rl.nav_cpu_env import OBSTACLE_GEOM_GROUP, build_scene_model
from rov_rl.vehicle import load_model

NO_HIT = nav_core._NO_HIT
ON = np.ones(1)


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def cap(o, d, a, b, r, active=ON) -> float:
    return float(nav_core.ray_capsules(np, np.asarray(o, float), unit(d)[None], np.asarray([a], float),
                                       np.asarray([b], float), np.asarray([r], float), active)[0])


def box(o, d, c, h, yaw, active=ON) -> float:
    return float(nav_core.ray_boxes(np, np.asarray(o, float), unit(d)[None], np.asarray([c], float),
                                    np.asarray([h], float), np.asarray([yaw], float), active)[0])


def test_ray_vertical_capsule_side_and_oblique():
    a, b = (5.0, 0.0, -10.0), (5.0, 0.0, 10.0)
    assert cap((0, 0, 0), (1, 0, 0), a, b, 0.5) == pytest.approx(4.5)
    assert cap((0, 0, 0), (0, 1, 0), a, b, 0.5) == NO_HIT
    assert cap((0, 0, 0), (-1, 0, 0), a, b, 0.5) == NO_HIT  # behind the ray
    # Oblique ray in the horizontal plane: circle (x - 5)^2 + y^2 = 0.25 along t * (cos, sin).
    ang = math.atan2(0.3, 5.0)
    c, s = math.cos(ang), math.sin(ang)
    t_ref = 5.0 * c - math.sqrt((5.0 * c) ** 2 - (25.0 - 0.25))
    assert cap((0, 0, 0), (c, s, 0), a, b, 0.5) == pytest.approx(t_ref, abs=1e-12)
    # Elevated ray: only the horizontal component sees the circle.
    el = math.radians(10.0)
    assert cap((0, 0, 0), (math.cos(el), 0, math.sin(el)), a, b, 0.5) == pytest.approx(4.5 / math.cos(el))


def test_ray_parallel_to_axis_hits_end_sphere():
    a, b = (0.0, 0.0, 0.0), (0.0, 0.0, 3.0)
    assert cap((0, 0, -5), (0, 0, 1), a, b, 0.5) == pytest.approx(4.5)
    assert cap((0.3, 0, -5), (0, 0, 1), a, b, 0.5) == pytest.approx(5.0 - math.sqrt(0.25 - 0.09))
    assert cap((0.6, 0, -5), (0, 0, 1), a, b, 0.5) == NO_HIT
    assert cap((0, 0, 8), (0, 0, -1), a, b, 0.5) == pytest.approx(4.5)


def test_ray_horizontal_pipe_body_and_end_cap():
    a, b = (-5.0, 3.0, 0.0), (5.0, 3.0, 0.0)
    assert cap((0, 0, 0), (0, 1, 0), a, b, 0.25) == pytest.approx(2.75)
    assert cap((5.2, 0, 0), (0, 1, 0), a, b, 0.25) == pytest.approx(3.0 - math.sqrt(0.0625 - 0.04))
    assert cap((5.3, 0, 0), (0, 1, 0), a, b, 0.25) == NO_HIT
    assert cap((0, 0, 0), (0, 1, 0), a, b, 0.25, active=np.zeros(1)) == NO_HIT


def test_ray_box_faces_yaw_and_corner():
    assert box((0, 0, 0), (1, 0, 0), (4, 0, 0), (1.0, 0.5, 0.5), 0.0) == pytest.approx(3.0)
    assert box((0, 0, 0), (1, 0, 0), (4, 0, 0), (1.0, 0.5, 0.5), math.pi / 2) == pytest.approx(3.5)
    assert box((0, 0, 0), (1, 0, 0), (4, 0, 0), (1.0, 1.0, 1.0), math.pi / 4) == pytest.approx(4.0 - math.sqrt(2.0))
    assert box((0, 0, 0), (1, 0, 0), (4, 2, 0), (1.0, 0.5, 0.5), 0.0) == NO_HIT
    assert box((0, 0, 3), (0, 0, -1), (0, 0, 0), (1.0, 1.0, 1.0), 0.3) == pytest.approx(2.0)  # axis-parallel ray
    assert box((0, 0, 0), (1, 0, 0), (4, 0, 0), (1.0, 0.5, 0.5), 0.0, active=np.zeros(1)) == NO_HIT


def test_ray_seabed():
    o = np.zeros(3)
    dirs = np.array([[0, 0, -1.0], unit([1, 0, -1]), [1, 0, 0], [0, 0, 1.0]])
    t = nav_core.ray_seabed(np, o, dirs, np.float64(-3.0))
    np.testing.assert_allclose(t[:2], [3.0, 3.0 * math.sqrt(2.0)])
    assert t[2] == NO_HIT and t[3] == NO_HIT
    assert nav_core.ray_seabed(np, np.array([0, 0, -4.0]), dirs[:1], np.float64(-3.0))[0] == NO_HIT


def test_point_distances():
    a, b = np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 4.0]])
    d = nav_core.capsule_distances(np, np.array([3.0, 0.0, 2.0]), a, b, np.array([0.5]))
    assert d[0] == pytest.approx(2.5)
    d = nav_core.capsule_distances(np, np.array([0.0, 0.0, 7.0]), a, b, np.array([0.5]))
    assert d[0] == pytest.approx(2.5)  # beyond the end: distance to the end sphere
    c, h = np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 2.0, 3.0]])
    assert nav_core.box_distances(np, np.array([4.0, 0.0, 0.0]), c, h, np.zeros(1))[0] == pytest.approx(3.0)
    assert nav_core.box_distances(np, np.array([0.5, 0.0, 0.0]), c, h, np.zeros(1))[0] == pytest.approx(-0.5)
    assert nav_core.box_distances(np, np.array([2.0, 3.0, 0.0]), c, h, np.zeros(1))[0] == pytest.approx(math.sqrt(2))
    # Box yawed by 90 deg: its 2 m half extent now lies along world x.
    assert nav_core.box_distances(np, np.array([4.0, 0.0, 0.0]), c, h, np.array([math.pi / 2]))[0] == \
        pytest.approx(2.0)


def single_pillar_scene(x_m: float, radius_m: float, seabed_z_m: float = -3.0) -> nav_core.NavScene:
    """A sampled scene with every obstacle switched off except one pillar at (x_m, 0)."""
    scene = nav_core.sample_scene(np, np.random.default_rng(0).random(nav_core.NAV_NUM_UNIFORMS), NAV_TRAIN_SCENE)
    cap_a = scene.cap_a_m.copy()
    cap_b = scene.cap_b_m.copy()
    cap_r = scene.cap_radius_m.copy()
    cap_a[0] = [x_m, 0.0, seabed_z_m - 1.0]
    cap_b[0] = [x_m, 0.0, NAV_TRAIN_SCENE.surface_z_m]
    cap_r[0] = radius_m
    active = np.zeros_like(scene.cap_active)
    active[0] = 1.0
    return scene._replace(cap_a_m=cap_a, cap_b_m=cap_b, cap_radius_m=cap_r, cap_active=active,
                          box_active=np.zeros_like(scene.box_active), seabed_z_m=np.float64(seabed_z_m))


def test_sonar_fan_and_vertical_beams_match_geometry():
    cfg = SonarConfig()
    scene = single_pillar_scene(x_m=6.0, radius_m=0.5, seabed_z_m=-3.0)
    qpos = np.concatenate([[0.0, 0.0, 0.0], core.yaw_quat(np, 0.0, 0.0, 0.0)])
    fan, vertical = nav_core.true_ranges(np, qpos, scene, cfg)
    assert fan.shape == (cfg.num_beams,)
    # Centre beams (azimuth ~0) see the pillar surface 6 - 0.5 - mount offset ahead; the elevated rows are
    # longer, so the per-azimuth minimum is the horizontal row.
    centre = np.argsort(np.abs(nav_core.beam_azimuths_rad(cfg)))[:2]
    az = nav_core.beam_azimuths_rad(cfg)[centre]
    x0 = 6.0 - cfg.mount_pos_m[0]
    expected = x0 * np.cos(az) - np.sqrt(0.25 - (x0 * np.sin(az)) ** 2)
    np.testing.assert_allclose(fan[centre], expected, atol=1e-9)
    # Outer beams: seabed seen by the lowest row (-10 deg) at 3 / sin(10 deg) = 17.3 m > max range.
    assert fan[0] == cfg.max_range_m and fan[-1] == cfg.max_range_m
    assert vertical[0] == pytest.approx(3.0)  # altimeter
    assert vertical[1] == cfg.max_range_m  # nothing overhead
    # Yawing the vehicle by +90 deg moves the pillar out of the 120 deg fan.
    qpos_yawed = np.concatenate([[0.0, 0.0, 0.0], core.yaw_quat(np, 0.0, 0.0, math.pi / 2)])
    fan_yawed, _ = nav_core.true_ranges(np, qpos_yawed, scene, cfg)
    assert np.all(fan_yawed == cfg.max_range_m)


def test_sonar_sees_seabed_when_low():
    cfg = SonarConfig()
    scene = single_pillar_scene(x_m=100.0, radius_m=0.5, seabed_z_m=-1.0)
    qpos = np.concatenate([[0.0, 0.0, 0.0], core.yaw_quat(np, 0.0, 0.0, 0.0)])
    fan, vertical = nav_core.true_ranges(np, qpos, scene, cfg)
    # Lowest row at -10 deg from the mount (z = 0) reaches the seabed 1 m below at 1 / sin(10 deg).
    np.testing.assert_allclose(fan, 1.0 / math.sin(-min(cfg.elevation_rows_rad)), rtol=1e-9)
    assert vertical[0] == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_analytic_rays_match_mujoco_mj_ray(seed):
    """The analytic caster agrees with mujoco.mj_ray on the compiled scene model (capsule, box, plane)."""
    rng = np.random.default_rng(seed)
    scene = nav_core.sample_scene(np, rng.random(nav_core.NAV_NUM_UNIFORMS), nav_eval_scene("high"))
    model = build_scene_model(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    group = np.zeros(mujoco.mjNGROUP, dtype=np.uint8)
    group[OBSTACLE_GEOM_GROUP] = 1
    max_range = 40.0  # well inside the 100 m plane extent of the compiled seabed
    goal = np.asarray(scene.goal_m)
    hits = 0
    for _ in range(400):
        xy = rng.uniform(-0.1, 1.1) * goal[:2] + rng.uniform(-6.0, 6.0, size=2)  # in and around the field
        o = np.array([xy[0], xy[1], rng.uniform(float(scene.seabed_z_m) + 0.1, 3.0)])
        if nav_core.scene_distance(np, o, scene) < 0.05:
            continue  # origin inside / on an obstacle: behaviour intentionally differs (see ray_capsules)
        v = unit(rng.normal(size=3))
        ref = mujoco.mj_ray(model, data, o, v, group, 1, -1, np.zeros(1, np.int32))
        ref = max_range if ref < 0.0 else min(ref, max_range)
        ours = min(float(nav_core.ray_scene(np, o, v[None], scene)[0]), max_range)
        hits += ref < max_range
        assert ours == pytest.approx(ref, abs=1e-6)
    assert hits > 50  # the comparison exercised real intersections


def test_physics_model_has_no_obstacles():
    """Obstacles must never enter the simulated model (contacts stay disabled, fluid is per body)."""
    scene = nav_core.sample_scene(np, np.random.default_rng(5).random(nav_core.NAV_NUM_UNIFORMS),
                                  nav_eval_scene("high"))
    scene_model = build_scene_model(scene)
    base = load_model()
    n_obs = int(np.sum(scene.cap_active) + np.sum(scene.box_active))
    assert scene_model.ngeom == base.ngeom + n_obs
    assert scene_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT
