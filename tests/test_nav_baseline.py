"""VFH+ baseline, occupancy grid, pose reconstruction, and the shared velocity controller."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from rov_rl import core, nav_core
from rov_rl.config import NavConfig, SonarConfig, nav_eval_scene, nav_eval_vehicle
from rov_rl.nav_baseline import OccupancyGrid, VfhParams, VfhPlanner, rotation_from_measurement
from rov_rl.nav_cpu_env import NavCpuEnv, make_velocity_controller_params
from rov_rl.nav_rollout import VfhController, nav_episode_metrics, run_nav_episode
from rov_rl.pid_baseline import CascadePid, PidGains
from rov_rl.vehicle import load_model

CFG = NavConfig()
NOISELESS = dataclasses.replace(CFG, sonar=dataclasses.replace(SonarConfig(), range_noise_std_m=0.0,
                                                               range_noise_frac=0.0, dropout_prob=0.0,
                                                               dvl_noise_std_mps=0.0))


def random_state(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    quat = core.yaw_quat(np, *rng.uniform(-0.3, 0.3, 2), rng.uniform(-math.pi, math.pi))
    qpos = np.concatenate([rng.normal(size=3), quat])
    qvel = np.concatenate([rng.normal(scale=0.3, size=3), rng.normal(scale=0.2, size=3)])
    return qpos, qvel


def test_velocity_controller_matches_cascade_pid_inner_loops():
    """With the outer position / yaw loops zeroed, CascadePid.wrench reduces to nav_core.velocity_wrench."""
    model = load_model()
    gains = PidGains.load()
    inner_only = dataclasses.replace(gains, kp_pos_ps=0.0, ki_pos_ps2=0.0, kp_yaw_ps=0.0)
    pid = CascadePid(model, inner_only, 0.04)
    vc = make_velocity_controller_params(model, gains)
    rng = np.random.default_rng(0)
    for _ in range(20):
        qpos, qvel = random_state(rng)
        v_cmd = rng.uniform(-0.3, 0.3, 3)
        r_cmd = rng.uniform(-0.5, 0.5)
        ref = core.Reference(qpos[:3].copy(), v_cmd, np.float64(0.0), np.float64(r_cmd))
        expected, _ = pid.wrench(qpos, qvel, ref)
        np.testing.assert_allclose(nav_core.velocity_wrench(np, qpos, qvel, v_cmd, np.float64(r_cmd), vc), expected,
                                   atol=1e-10)


def test_command_to_thrust_direction_and_limits():
    vc = make_velocity_controller_params(load_model(), PidGains.load())
    qpos = np.concatenate([np.zeros(3), core.yaw_quat(np, 0.0, 0.0, 0.0)])
    qvel = np.zeros(6)
    tam = np.linalg.pinv(vc.tam_pinv)  # allocation matrix (pinv of pinv for a full-rank 6x8 matrix)
    fwd = nav_core.command_to_thrust(np, qpos, qvel, np.array([0.5, 0, 0, 0]), CFG.command, vc)
    wrench = tam @ (fwd * vc.thrust_max_n)
    assert wrench[0] > 10.0 and abs(wrench[1]) < 1e-6 and abs(wrench[5]) < 1e-6
    full = nav_core.command_to_thrust(np, qpos, qvel, np.array([1.0, 1.0, 1.0, 1.0]), CFG.command, vc)
    assert np.max(np.abs(full)) <= 1.0 + 1e-12
    # Sway is along body +y (left) for a vehicle yawed by 90 deg: world -x.
    qpos_yawed = np.concatenate([np.zeros(3), core.yaw_quat(np, 0.0, 0.0, math.pi / 2)])
    sway = nav_core.command_to_thrust(np, qpos_yawed, qvel, np.array([0.0, 0.5, 0, 0]), CFG.command, vc)
    assert (tam @ sway)[1] > 0.0  # body-frame wrench: +y force


def test_pose_reconstruction_from_measurement():
    rng = np.random.default_rng(1)
    scene = nav_core.sample_scene(np, rng.random(nav_core.NAV_NUM_UNIFORMS), nav_eval_scene("low"))
    for _ in range(10):
        qpos, qvel = random_state(rng)
        meas = nav_core.measure(np, qpos, qvel, scene, NOISELESS.sonar, 30.0,
                                np.zeros(nav_core.num_noise_normals(NOISELESS.sonar)),
                                np.ones(nav_core.num_noise_uniforms(NOISELESS.sonar)))
        rot = rotation_from_measurement(meas)
        np.testing.assert_allclose(rot, core.quat_to_mat(np, qpos[3:7]), atol=1e-12)
        np.testing.assert_allclose(-(rot @ meas.goal_body_m), qpos[:3] - scene.goal_m, atol=1e-12)


def test_grid_marks_echo_and_clears_free_space():
    p = VfhParams()
    grid = OccupancyGrid(p)
    origin = np.array([0.0, 0.0])
    dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
    for _ in range(3):
        grid.update(origin, dirs, np.array([5.0, 12.0]), np.array([True, False]), 12.0)
    hit, _ = grid.cell_index(np.array([5.0, 0.0]))
    before, _ = grid.cell_index(np.array([3.0, 0.0]))
    side, _ = grid.cell_index(np.array([0.0, 11.0]))
    g = grid.logodds.reshape(-1)
    assert g[hit] == pytest.approx(min(3 * p.logodds_hit, p.logodds_max))
    assert g[before] < 0.0 and g[side] < 0.0  # no echo: cleared to max range
    centres, _ = grid.occupied_near(origin, 6.0)
    assert centres.shape[0] >= 1 and np.min(np.linalg.norm(centres - [5.0, 0.0], axis=1)) < p.cell_m


def test_seabed_returns_are_gated():
    planner = VfhPlanner(VfhParams(), NOISELESS.sonar, CFG.command)
    nb = NOISELESS.sonar.num_beams
    seabed_range = 1.5 / math.sin(-min(NOISELESS.sonar.elevation_rows_rad))
    meas = nav_core.NavMeasurement(np.full(nb, seabed_range), np.float64(1.5), np.float64(12.0), np.zeros(3),
                                   np.zeros(3), np.array([0.0, 0.0, -1.0]), np.float64(0.0), np.float64(30.0),
                                   np.array([10.0, 0.0, 0.0]))
    assert not np.any(planner.echo_mask(meas))
    assert np.all(planner.echo_mask(meas._replace(sonar_range_m=np.full(nb, 3.0))))


def test_direction_selection():
    planner = VfhPlanner(VfhParams(), CFG.sonar, CFG.command)
    k = planner.p.num_sectors
    assert planner.choose_direction(np.zeros(k, bool), 0.3, 0.0) == pytest.approx(0.3)
    assert planner.choose_direction(np.ones(k, bool), 0.3, 0.0) is None
    blocked = np.abs(planner._sector_angles) < math.radians(30.0)  # obstacle straight ahead, goal behind it
    chosen = planner.choose_direction(blocked, 0.0, 0.0)
    assert chosen is not None and abs(chosen) >= math.radians(30.0)
    sector = int(math.floor((chosen + math.pi) / (2 * math.pi / k))) % k
    assert not blocked[sector]


def measurement(goal_body, ranges=None, yaw=0.0) -> nav_core.NavMeasurement:
    nb = CFG.sonar.num_beams
    r = np.full(nb, CFG.sonar.max_range_m) if ranges is None else ranges
    return nav_core.NavMeasurement(r, np.float64(3.0), np.float64(12.0), np.zeros(3), np.zeros(3),
                                   np.array([0.0, 0.0, -1.0]), np.float64(yaw), np.float64(30.0),
                                   np.asarray(goal_body, float))


def test_act_free_space_goal_ahead_behind_and_close_obstacle():
    planner = VfhPlanner(VfhParams(), CFG.sonar, CFG.command)
    cmd = planner.act(measurement([20.0, 0.0, 0.0]))
    assert cmd[0] > 0.5 and abs(cmd[3]) < 1e-9
    planner.reset()
    cmd = planner.act(measurement([-20.0, 1.0, 0.0]))  # goal behind, slightly left
    assert cmd[0] == 0.0 and cmd[3] > 0.0
    planner.reset()
    close = np.full(CFG.sonar.num_beams, CFG.sonar.max_range_m)
    close[CFG.sonar.num_beams // 2 - 2 : CFG.sonar.num_beams // 2 + 2] = 0.6
    cmd = planner.act(measurement([20.0, 0.0, 0.0], close))
    assert cmd[0] < 0.0  # backs off
    planner.reset()
    cmd = planner.act(measurement([20.0, 0.0, 2.0]))
    assert cmd[2] > 0.0  # climbs towards a goal above


def single_pillar_episode(env: NavCpuEnv, offset_m: float) -> nav_core.NavEpisode:
    """Straight 20 m run along +x with one 0.5 m pillar on (or `offset_m` beside) the line at 10 m."""
    ep = env.sample_params(np.random.default_rng(3))
    s = ep.scene
    cap_a, cap_b, cap_r = s.cap_a_m.copy(), s.cap_b_m.copy(), s.cap_radius_m.copy()
    cap_a[0] = [10.0, offset_m, -4.0]
    cap_b[0] = [10.0, offset_m, 30.0]
    cap_r[0] = 0.5
    active = np.zeros_like(s.cap_active)
    active[0] = 1.0
    scene = s._replace(goal_m=np.array([20.0, 0.0, 0.0]), start_yaw_rad=np.float64(0.0), seabed_z_m=np.float64(-3.0),
                       cap_a_m=cap_a, cap_b_m=cap_b, cap_radius_m=cap_r, cap_active=active,
                       box_active=np.zeros_like(s.box_active))
    vehicle = ep.vehicle._replace(start_quat=core.yaw_quat(np, 0.0, 0.0, 0.0))
    return nav_core.NavEpisode(vehicle=vehicle, scene=scene)


@pytest.mark.parametrize("offset_m", [0.0, 0.4])
def test_closed_loop_avoids_pillar_on_the_line(offset_m):
    env = NavCpuEnv(nav_eval_vehicle(0), nav_eval_scene("low"), CFG)
    ep = single_pillar_episode(env, offset_m)
    log = run_nav_episode(env, VfhController(VfhPlanner(VfhParams.load(), CFG.sonar, CFG.command)), ep,
                          noise_seed=0)
    m = nav_episode_metrics(log, ep)
    assert m["outcome"] == "success", m
    assert m["min_clearance_m"] > 0.2
    # The run may end anywhere on the goal sphere, so a near-direct path can come in just under 1.
    assert 1.0 - CFG.reward.goal_radius_m / m["straight_line_m"] < m["path_ratio"] < 1.5
    assert np.max(np.abs(log.qpos[:, 1])) > 0.5  # it actually went around the pillar
