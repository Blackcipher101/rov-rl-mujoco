"""Nav reward shaping, success / collision / failure termination."""

from __future__ import annotations

import numpy as np
import pytest

from rov_rl import nav_core
from rov_rl.config import NavRewardConfig

CFG = NavRewardConfig()
ZERO4 = np.zeros(4)
ZERO8 = np.zeros(8)


def step(dist_prev=10.0, dist_now=10.0, clearance=5.0, cos_tilt=1.0, thrust=ZERO8, cmd=ZERO4, last=ZERO4,
         finite=True):
    r, out, terms = nav_core.nav_reward_and_done(
        np, np.float64(dist_prev), np.float64(dist_now), np.float64(clearance), np.float64(cos_tilt),
        np.asarray(thrust, float), np.asarray(cmd, float), np.asarray(last, float), np.bool_(finite), CFG)
    return float(r), out, terms


def test_idle_far_from_obstacles_costs_only_time():
    r, out, _ = step()
    assert r == pytest.approx(-CFG.w_time)
    assert float(out.done) == 0.0


def test_progress_sign_and_scale():
    r_closer, _, _ = step(dist_prev=10.0, dist_now=9.98)
    r_away, _, _ = step(dist_prev=10.0, dist_now=10.02)
    assert r_closer == pytest.approx(-CFG.w_time + CFG.w_progress_per_m * 0.02)
    assert r_away == pytest.approx(-CFG.w_time - CFG.w_progress_per_m * 0.02)


def test_success_bonus_and_termination():
    r, out, _ = step(dist_prev=1.02, dist_now=CFG.goal_radius_m - 0.01)
    assert float(out.done) == 1.0 and float(out.success) == 1.0 and float(out.collision) == 0.0
    assert r == pytest.approx(-CFG.w_time + CFG.w_progress_per_m * 0.03 + CFG.success_bonus)


def test_collision_terminates_and_overrides_success():
    r, out, _ = step(clearance=-0.01)
    assert float(out.done) == 1.0 and float(out.collision) == 1.0
    assert r < -CFG.collision_penalty + 1.0
    _, out, _ = step(dist_now=0.5, clearance=-0.01)
    assert float(out.success) == 0.0 and float(out.collision) == 1.0


def test_clearance_shaping_monotonic_and_bounded():
    vals = [step(clearance=c)[2]["clearance"] for c in (2.0, CFG.safe_clearance_m, 0.75, 0.5, 0.25, 0.0)]
    assert vals[0] == 0.0 and vals[1] == 0.0
    assert all(a > b for a, b in zip(vals[1:], vals[2:]))
    assert vals[-1] == pytest.approx(-CFG.w_clearance)


def test_effort_rate_tilt_terms():
    _, _, t = step(thrust=np.full(8, 0.5), cmd=np.full(4, 0.5), last=np.full(4, -0.5), cos_tilt=0.9)
    assert t["effort"] == pytest.approx(-CFG.w_effort * 0.5)
    assert t["action_rate"] == pytest.approx(-CFG.w_action_rate * 1.0)
    assert t["tilt"] == pytest.approx(-CFG.w_tilt * 0.1)


def test_tilt_and_non_finite_failures():
    _, out, _ = step(cos_tilt=CFG.min_cos_tilt - 0.01)
    assert float(out.done) == 1.0 and float(out.failure) == 1.0 and float(out.collision) == 0.0
    r, out, _ = step(dist_now=np.nan, clearance=np.nan, cos_tilt=np.nan, finite=False)
    assert float(out.done) == 1.0 and float(out.failure) == 1.0
    assert r == -CFG.collision_penalty
