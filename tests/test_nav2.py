"""nav2: moving obstacles, fan memory, curriculum, VFH+ decay option, eval cells."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from rov_rl import core, nav_core
from rov_rl.config import (
    NAV2_CONFIG,
    NAV2_CURRICULUM,
    NAV_DENSITY_LEVELS,
    NAV_TRAIN_SCENE,
    NavConfig,
    SonarConfig,
    nav_eval_scene,
    nav_eval_scene_moving,
    nav_eval_vehicle,
)
from rov_rl.nav_baseline import OccupancyGrid, VfhParams
from rov_rl.nav_cpu_env import NavCpuEnv

SIM = NavConfig().sim
STATIC_UNIFORMS = 236  # uniforms consumed by the static part of sample_scene (movers are drawn after them)


def episode(seed: int, scene_cfg=nav_eval_scene_moving("high"), level: int = 2) -> nav_core.NavEpisode:
    rng = np.random.default_rng(seed)
    return nav_core.make_episode(np, rng.random(core.NUM_UNIFORMS), rng.random(nav_core.NAV_NUM_UNIFORMS),
                                 nav_eval_vehicle(level), scene_cfg, SIM)


# --- moving obstacles --------------------------------------------------------------------------------------


def test_static_fields_do_not_depend_on_mover_uniforms():
    """Adding movers kept the nav-task fields (and so its published eval episodes) unchanged."""
    u = np.random.default_rng(3).random(nav_core.NAV_NUM_UNIFORMS)
    v = u.copy()
    v[STATIC_UNIFORMS:] = np.random.default_rng(4).random(nav_core.NAV_NUM_UNIFORMS - STATIC_UNIFORMS)
    a = nav_core.sample_scene(np, u, nav_eval_scene_moving("high"))
    b = nav_core.sample_scene(np, v, nav_eval_scene_moving("high"))
    for name in ("goal_m", "seabed_z_m", "cap_a_m", "cap_b_m", "cap_radius_m", "cap_active", "box_center_m",
                 "box_half_m", "box_yaw_rad", "box_active"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert not np.allclose(a.mov_center_m, b.mov_center_m)
    # numpy draws the longer uniform vector as a prefix-extension, so seeds give the same first 256 values.
    np.testing.assert_array_equal(np.random.default_rng(9).random(nav_core.NAV_NUM_UNIFORMS)[:256],
                                  np.random.default_rng(9).random(256))


def test_static_scene_has_no_active_movers():
    for seed in range(20):
        assert not np.any(episode(seed, nav_eval_scene("high")).scene.mov_active)


def test_patrol_motion_is_smooth_bounded_and_in_speed_range():
    cfg = nav_eval_scene_moving("high")
    vmax = cfg.mover_speed_mps[1] * math.sqrt(2.0)
    ts = np.arange(0.0, 100.0, 0.04)
    n_patrol = 0
    for seed in range(15):
        ep = episode(seed)
        s = ep.scene
        pos = np.stack([nav_core.mover_positions(np, s, ep.vehicle, t) for t in ts])  # (T, M, 3)
        for k in np.flatnonzero((s.mov_active > 0.5) & (s.mov_drift < 0.5)):
            n_patrol += 1
            reach = np.linalg.norm(s.mov_axis1_m[k]) + np.linalg.norm(s.mov_axis2_m[k])
            assert np.max(np.linalg.norm(pos[:, k] - s.mov_center_m[k], axis=-1)) <= reach + 1e-9
            speed = np.linalg.norm(np.diff(pos[:, k], axis=0), axis=-1) / 0.04
            assert speed.max() <= vmax + 1e-3
            assert speed.max() >= cfg.mover_speed_mps[0] * 0.9  # it does move
            np.testing.assert_allclose(pos[:, k, 2], s.mov_center_m[k, 2])  # horizontal motion
    assert n_patrol >= 10


def test_drifting_mover_follows_the_current_integral():
    ep = episode(1)
    s = ep.scene._replace(mov_drift=np.ones_like(ep.scene.mov_drift))
    dt = 0.01
    ts = np.arange(0.0, 30.0 + dt / 2, dt)
    vel = np.stack([core.current_velocity(np, t, ep.vehicle) for t in ts])
    integral = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (vel[1:] + vel[:-1]) * dt, axis=0)])
    for t_idx in (0, 1000, 3000):
        pos = nav_core.mover_positions(np, s, ep.vehicle, ts[t_idx])
        np.testing.assert_allclose(pos - s.mov_center_m, np.broadcast_to(integral[t_idx], pos.shape), atol=1e-6)


@pytest.mark.parametrize("density", sorted(NAV_DENSITY_LEVELS))
def test_patrol_loops_keep_out_of_start_and_goal(density):
    cfg = nav_eval_scene_moving(density)
    min_gap = cfg.keepout_m + nav_core.VEHICLE_BOUNDING_RADIUS_M
    ts = np.linspace(0.0, 100.0, 501)
    for seed in range(30):
        ep = episode(seed, cfg)
        s = ep.scene
        for t in ts:
            now = nav_core.scene_at(np, s, ep.vehicle, t)
            movers = slice(s.cap_a_m.shape[0], None)
            for point in (np.zeros(3), np.asarray(s.goal_m)):
                d = nav_core.capsule_distances(np, point, now.cap_a_m[movers], now.cap_b_m[movers],
                                               now.cap_radius_m[movers])
                patrol = (s.mov_active > 0.5) & (s.mov_drift < 0.5)
                assert np.all(d[patrol] >= min_gap - 1e-9)


def test_scene_at_appends_movers_and_moves_them():
    ep = episode(2)
    s = ep.scene
    n_cap, n_mov = s.cap_a_m.shape[0], s.mov_active.shape[0]
    a, b = nav_core.scene_at(np, s, ep.vehicle, 0.0), nav_core.scene_at(np, s, ep.vehicle, 10.0)
    assert a.cap_a_m.shape == (n_cap + n_mov, 3)
    np.testing.assert_array_equal(a.cap_a_m[:n_cap], s.cap_a_m)  # static part unchanged
    np.testing.assert_allclose(a.cap_b_m[n_cap:, 2] - a.cap_a_m[n_cap:, 2], 2.0 * s.mov_half_height_m)
    active = s.mov_active > 0.5
    assert np.any(active)
    assert np.all(np.linalg.norm(a.cap_a_m[n_cap:][active] - b.cap_a_m[n_cap:][active], axis=-1) > 0.05)
    # Without active movers the instantaneous scene gives the same clearance as the static one.
    still = s._replace(mov_active=np.zeros_like(s.mov_active))
    p = np.array([3.0, 1.0, 0.0])
    assert nav_core.clearance_m(np, p, nav_core.scene_at(np, still, ep.vehicle, 5.0)) == pytest.approx(
        nav_core.clearance_m(np, p, still))


def test_mover_crossing_a_stationary_vehicle_is_a_collision_and_is_seen_by_the_sonar():
    env = NavCpuEnv(nav_eval_vehicle(0), nav_eval_scene_moving("low"), NavConfig())
    ep = env.sample_params(np.random.default_rng(0))
    s = ep.scene
    one = np.zeros_like(s.mov_active)
    one[0] = 1.0
    # A mover patrolling along world x through the vehicle's start, 0.5 m/s peak, starting 4 m ahead.
    scene = s._replace(
        goal_m=np.array([25.0, 0.0, 0.0]), cap_active=np.zeros_like(s.cap_active),
        box_active=np.zeros_like(s.box_active), seabed_z_m=np.float64(-4.0), mov_active=one,
        mov_drift=np.zeros_like(one), mov_center_m=np.tile([0.0, 0.0, 0.0], (one.size, 1)),
        mov_axis1_m=np.tile([4.0, 0.0, 0.0], (one.size, 1)), mov_axis2_m=np.zeros((one.size, 3)),
        mov_omega_radps=np.tile([0.125, 0.0], (one.size, 1)),
        mov_phase_rad=np.tile([0.5 * math.pi, 0.0], (one.size, 1)),
        mov_radius_m=np.full(one.size, 0.4), mov_half_height_m=np.full(one.size, 0.4))
    obs = env.reset(params=nav_core.NavEpisode(ep.vehicle._replace(start_quat=core.yaw_quat(np, 0.0, 0.0, 0.0)),
                                               scene), noise_seed=0)
    centre = np.argsort(np.abs(nav_core.beam_azimuths_rad(env.cfg.sonar)))[0]
    assert obs[centre] * env.cfg.sonar.max_range_m < 4.0  # seen ahead at t = 0
    done = False
    while not done:
        obs, _, done, info = env.step(np.zeros(4))
    assert info.collision and info.t_s < 20.0


# --- fan memory --------------------------------------------------------------------------------------------

MEM = NAV2_CONFIG.memory
SONAR = SonarConfig()


def meas(ranges=None, vel=(0.0, 0.0, 0.0), yaw=0.0) -> nav_core.NavMeasurement:
    nb = SONAR.num_beams
    r = np.full(nb, SONAR.max_range_m) if ranges is None else np.asarray(ranges, float)
    return nav_core.NavMeasurement(r, np.float64(3.0), np.float64(12.0), np.asarray(vel, float), np.zeros(3),
                                   np.array([0.0, 0.0, -1.0]), np.float64(yaw), np.float64(30.0), np.zeros(3))


def centre_echo(range_m: float) -> np.ndarray:
    r = np.full(SONAR.num_beams, SONAR.max_range_m)
    r[np.argsort(np.abs(nav_core.beam_azimuths_rad(SONAR)))[0]] = range_m
    return r


def sector_of(angle_rad: float) -> int:
    return int(math.floor((angle_rad + math.pi) / (2.0 * math.pi / MEM.sectors))) % MEM.sectors


def test_nav2_observation_size_and_env_obs():
    assert nav_core.nav_obs_size(SONAR, MEM) == nav_core.nav_obs_size(SONAR) + MEM.sectors == 94
    env = NavCpuEnv(nav_eval_vehicle(0), nav_eval_scene_moving("high"), NAV2_CONFIG)
    obs = env.reset(seed=1)
    assert obs.shape == (94,) == (env.observation_size,)
    for _ in range(30):
        obs, _, _, _ = env.step(np.array([0.5, 0.0, 0.0, 0.1]))
    polar = obs[-MEM.sectors:]
    assert np.all((polar >= 0.0) & (polar <= 1.0)) and np.all(np.isfinite(obs))


def test_memory_stores_echo_in_heading_sector():
    mem = nav_core.memory_init(np, MEM, SONAR, np.float64)
    assert np.all(nav_core.memory_polar(np, mem, 0.0, MEM, SIM.ctrl_dt_s) == 1.0)
    mem = nav_core.memory_update(np, mem, meas(centre_echo(3.0)), SONAR, MEM, 0.0)
    polar = nav_core.memory_polar(np, mem, 0.0, MEM, SIM.ctrl_dt_s)
    az = nav_core.beam_azimuths_rad(SONAR)[np.argsort(np.abs(nav_core.beam_azimuths_rad(SONAR)))[0]]
    expected = math.hypot(SONAR.mount_pos_m[0] + 3.0 * math.cos(az), 3.0 * math.sin(az))
    assert polar[sector_of(az)] == pytest.approx(expected / MEM.max_range_m)
    assert np.sum(polar < 1.0) == 1
    # Echoes beyond the memory range (e.g. the seabed far away) are not stored.
    far = nav_core.memory_update(np, nav_core.memory_init(np, MEM, SONAR, np.float64), meas(centre_echo(8.0)),
                                 SONAR, MEM, 0.0)
    assert np.all(nav_core.memory_polar(np, far, 0.0, MEM, SIM.ctrl_dt_s) == 1.0)


def test_memory_motion_compensation_and_rotation():
    mem = nav_core.memory_init(np, MEM, SONAR, np.float64)
    mem = nav_core.memory_update(np, mem, meas(centre_echo(4.0)), SONAR, MEM, 0.0)
    p0 = mem.points_m[0][mem.valid[0] > 0.5][0].copy()
    # Drive forward at 0.5 m/s for 1 s with an empty fan: the remembered point comes 0.5 m closer.
    for _ in range(25):
        mem = nav_core.memory_update(np, mem, meas(vel=(0.5, 0.0, 0.0)), SONAR, MEM, SIM.ctrl_dt_s)
    p1 = mem.points_m[0][mem.valid[0] > 0.5][0]
    np.testing.assert_allclose(p1, p0 - np.array([0.5, 0.0]), atol=1e-9)
    # Turning the vehicle left by 90 deg puts the point on its right side (heading-relative sectors).
    polar = nav_core.memory_polar(np, mem, math.pi / 2, MEM, SIM.ctrl_dt_s)
    assert np.argmin(polar) == sector_of(math.atan2(p1[1], p1[0]) - math.pi / 2)
    # Motion uses the measured attitude: the body velocity is rotated into world axes by the compass yaw.
    mem2 = nav_core.memory_update(np, nav_core.memory_init(np, MEM, SONAR, np.float64), meas(centre_echo(4.0)),
                                  SONAR, MEM, 0.0)
    q0 = mem2.points_m[0][mem2.valid[0] > 0.5][0].copy()
    mem2 = nav_core.memory_update(np, mem2, meas(vel=(1.0, 0.0, 0.0), yaw=math.pi / 2), SONAR, MEM, 1.0)
    np.testing.assert_allclose(mem2.points_m[0][mem2.valid[0] > 0.5][0], q0 - np.array([0.0, 1.0]), atol=1e-9)


def test_memory_stride_and_expiry():
    mem = nav_core.memory_init(np, MEM, SONAR, np.float64)
    mem = nav_core.memory_update(np, mem, meas(centre_echo(3.0)), SONAR, MEM, 0.0)  # update 0: stored
    for _ in range(MEM.stride_steps - 1):  # updates 1 .. stride-1: echoes ignored
        mem = nav_core.memory_update(np, mem, meas(centre_echo(2.0)), SONAR, MEM, SIM.ctrl_dt_s)
    assert int(np.sum(mem.valid)) == 1
    horizon = nav_core.memory_horizon_s(MEM, SIM.ctrl_dt_s)
    assert horizon == pytest.approx(MEM.slots * MEM.stride_steps * SIM.ctrl_dt_s)
    steps = int(round(horizon / SIM.ctrl_dt_s))
    for _ in range(steps):  # empty fans for one horizon: everything is forgotten
        mem = nav_core.memory_update(np, mem, meas(), SONAR, MEM, SIM.ctrl_dt_s)
    assert np.all(nav_core.memory_polar(np, mem, 0.0, MEM, SIM.ctrl_dt_s) == 1.0)


def test_nav_task_observation_is_unchanged_without_memory():
    env = NavCpuEnv(nav_eval_vehicle(0), nav_eval_scene("high"), NavConfig())
    assert env.reset(seed=3).shape == (nav_core.nav_obs_size(SONAR),) == (70,)


# --- curriculum, presets, baseline decay, eval cells ------------------------------------------------------


def test_curriculum_schedule_ramps_up_and_covers_eval():
    assert sum(st.fraction for st in NAV2_CURRICULUM) == pytest.approx(1.0)
    dens = [st.scene.slot_density[1] for st in NAV2_CURRICULUM]
    cur = [st.vehicle.current_const_mps[1] for st in NAV2_CURRICULUM]
    movers = [st.scene.mover_prob for st in NAV2_CURRICULUM]
    assert dens == sorted(dens) and cur == sorted(cur) and movers == sorted(movers)
    assert movers[0] == 0.0 and movers[-1] > 0.0  # moving obstacles only in later stages
    last = NAV2_CURRICULUM[-1]
    assert last.scene.slot_density[1] >= max(v[0] for v in NAV_DENSITY_LEVELS.values())
    assert last.scene.pipe_prob[1] >= max(v[1] for v in NAV_DENSITY_LEVELS.values())
    assert last.vehicle.current_const_mps[1] >= 0.3  # eval current level 3
    assert all(st.scene.num_movers == NAV_TRAIN_SCENE.num_movers for st in NAV2_CURRICULUM)  # same obs / shapes


def test_nav2_presets_and_stage_steps():
    import train

    for name, preset in train.NAV2_PRESETS.items():
        total = preset["num_timesteps"]
        steps = [int(round(st.fraction * total)) for st in NAV2_CURRICULUM]
        assert abs(sum(steps) - total) <= len(steps), name
    assert train.NAV_PRESETS["full"]["num_timesteps"] == 300_000_000  # nav presets untouched


def test_baseline_decay_forgets_and_zero_decay_does_not():
    for decay, forgets in ((0.0, False), (0.4, True)):
        grid = OccupancyGrid(dataclasses.replace(VfhParams(), logodds_decay_per_s=decay))
        for _ in range(4):
            grid.update(np.zeros(2), np.array([[1.0, 0.0]]), np.array([5.0]), np.array([True]), 12.0, 0.04)
        before = grid.logodds.copy()
        for _ in range(250):  # 10 s without looking there
            grid.update(np.zeros(2), np.array([[0.0, -1.0]]), np.array([12.0]), np.array([False]), 12.0, 0.04)
        hit, _ = grid.cell_index(np.array([5.0, 0.0]))
        g0, g1 = before.reshape(-1)[hit], grid.logodds.reshape(-1)[hit]
        assert (g1 < 0.1 * g0) if forgets else (g1 == g0)
    with pytest.raises(ValueError):
        dataclasses.replace(VfhParams(), logodds_decay_per_s=-1.0).validate()


def test_eval_cells_keep_static_seeds_and_names():
    import eval_nav

    static, moving = eval_nav.NavCell("high", 2), eval_nav.NavCell("high", 2, "moving")
    assert static.name == "high_L2" and moving.name == "high_L2_mov"
    env_s = NavCpuEnv(nav_eval_vehicle(2), static.scene(), NavConfig())
    env_m = NavCpuEnv(nav_eval_vehicle(2), moving.scene(), NavConfig())
    (ep_s, noise_s), = eval_nav.cell_episodes(env_s, static, 1, 2024)
    # The nav-task key for this episode, unchanged.
    ref = env_s.sample_params(np.random.default_rng([2024, eval_nav.DENSITY_ORDER.index("high"), 2, 0]))
    np.testing.assert_array_equal(ep_s.scene.cap_a_m, ref.scene.cap_a_m)
    (ep_m, _), = eval_nav.cell_episodes(env_m, moving, 1, 2024)
    assert np.any(ep_m.scene.mov_active) and not np.any(ep_s.scene.mov_active)
