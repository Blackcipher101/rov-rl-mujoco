"""Obstacle-field generator: reproducibility, keep-out around start and goal, ranges, density levels."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rov_rl import core, nav_core
from rov_rl.config import NAV_DENSITY_LEVELS, NAV_TRAIN_SCENE, NAV_TRAIN_VEHICLE, SimConfig, nav_eval_scene

CFG = NAV_TRAIN_SCENE


def scene(seed: int, cfg=CFG) -> nav_core.NavScene:
    return nav_core.sample_scene(np, np.random.default_rng(seed).random(nav_core.NAV_NUM_UNIFORMS), cfg)


def test_reproducible_by_seed():
    a, b, c = scene(7), scene(7), scene(8)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert not np.allclose(a.goal_m, c.goal_m)


def test_shapes():
    s = scene(0)
    n_cap, n_box = nav_core.num_capsules(CFG), nav_core.num_boxes(CFG)
    assert s.cap_a_m.shape == (n_cap, 3) and s.cap_active.shape == (n_cap,)
    assert s.box_center_m.shape == (n_box, 3) and s.box_active.shape == (n_box,)
    assert int(np.sum(s.cap_is_pipe)) == CFG.num_pipes


def test_too_few_uniforms_raises():
    with pytest.raises(ValueError):
        nav_core.sample_scene(np, np.zeros(nav_core.NAV_NUM_UNIFORMS // 4), CFG)


@pytest.mark.parametrize("density", sorted(NAV_DENSITY_LEVELS))
def test_keepout_around_start_and_goal(density):
    cfg = nav_eval_scene(density)
    min_gap = cfg.keepout_m + nav_core.VEHICLE_BOUNDING_RADIUS_M
    for seed in range(60):
        s = scene(seed, cfg)
        for point in (np.zeros(3), np.asarray(s.goal_m)):
            dc = nav_core.capsule_distances(np, point, s.cap_a_m, s.cap_b_m, s.cap_radius_m)
            db = nav_core.box_distances(np, point, s.box_center_m, s.box_half_m, s.box_yaw_rad)
            assert np.all(dc[s.cap_active > 0.5] >= min_gap - 1e-9)
            assert np.all(db[s.box_active > 0.5] >= min_gap - 1e-9)
            # Start and goal are collision-free with margin (seabed included).
            assert nav_core.clearance_m(np, point, s) >= min(cfg.keepout_m, 0.5)


def test_ranges_and_placement():
    for seed in range(60):
        s = scene(seed)
        goal = np.asarray(s.goal_m)
        assert CFG.goal_distance_m[0] - 1e-9 <= np.linalg.norm(goal[:2]) <= CFG.goal_distance_m[1] + 1e-9
        assert abs(goal[2]) <= CFG.goal_dz_m
        altitude = -float(s.seabed_z_m)
        assert CFG.seabed_altitude_m[0] <= altitude <= CFG.seabed_altitude_m[1]
        # Boxes rest on the seabed.
        np.testing.assert_allclose(s.box_center_m[:, 2] - s.box_half_m[:, 2], s.seabed_z_m, atol=1e-12)
        # Pillars span from below the seabed to the surface; pipes stay above the seabed and are horizontal.
        pillars = s.cap_is_pipe < 0.5
        assert np.all(s.cap_a_m[pillars, 2] < s.seabed_z_m) and np.all(s.cap_b_m[pillars, 2] == CFG.surface_z_m)
        pipes = ~pillars
        np.testing.assert_allclose(s.cap_a_m[pipes, 2], s.cap_b_m[pipes, 2])
        assert np.all(s.cap_a_m[pipes, 2] - s.cap_radius_m[pipes] >= s.seabed_z_m - 1e-9)
        # Lattice obstacles lie inside the corridor around the start-goal line.
        along = goal[:2] / np.linalg.norm(goal[:2])
        cross = np.array([-along[1], along[0]])
        centres = np.concatenate([s.cap_a_m[pillars, :2], s.box_center_m[:, :2]])
        assert np.all(np.abs(centres @ cross) <= CFG.corridor_half_width_m + 1e-9)
        s_along = centres @ along
        assert np.all((s_along >= 0.0) & (s_along <= np.linalg.norm(goal[:2])))
        assert np.all((s.cap_radius_m[pillars] >= CFG.pillar_radius_m[0]) &
                      (s.cap_radius_m[pillars] <= CFG.pillar_radius_m[1]))


def test_density_levels_order_obstacle_counts():
    means = {}
    for density in ("low", "medium", "high"):
        cfg = nav_eval_scene(density)
        counts = [float(np.sum(scene(seed, cfg).cap_active) + np.sum(scene(seed, cfg).box_active))
                  for seed in range(80)]
        means[density] = np.mean(counts)
        assert scene(0, cfg).density == pytest.approx(NAV_DENSITY_LEVELS[density][0])
    assert means["low"] < means["medium"] < means["high"]


def test_episode_starts_level_at_origin_facing_near_goal():
    sim = SimConfig(episode_duration_s=100.0)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        ep = nav_core.make_episode(np, rng.random(core.NUM_UNIFORMS), rng.random(nav_core.NAV_NUM_UNIFORMS),
                                   NAV_TRAIN_VEHICLE, CFG, sim)
        np.testing.assert_array_equal(ep.vehicle.start_pos_m, np.zeros(3))
        rot = core.quat_to_mat(np, ep.vehicle.start_quat)
        assert rot[2, 2] == pytest.approx(1.0)
        bearing = math.atan2(ep.scene.goal_m[1], ep.scene.goal_m[0])
        assert abs(core.wrap_angle(np, core.yaw_of(np, rot) - bearing)) <= CFG.start_heading_offset_rad + 1e-9
        assert not np.any(ep.vehicle.thruster_failed)  # nav training has no thruster failures
