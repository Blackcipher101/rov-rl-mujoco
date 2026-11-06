"""Reward shaping and termination."""

from __future__ import annotations

import numpy as np
import pytest

from rov_rl import core
from rov_rl.config import RewardConfig

CFG = RewardConfig()
ZERO = np.zeros(8)


def errors(pos_m: float = 0.0, yaw_rad: float = 0.0, cos_tilt: float = 1.0) -> core.Errors:
    return core.Errors(np.array([pos_m, 0.0, 0.0]), np.float64(pos_m), np.float64(yaw_rad), np.float64(cos_tilt))


def reward(err: core.Errors, action=ZERO, last=ZERO, finite=True) -> tuple[float, float]:
    r, done, _ = core.reward_and_done(
        np, err, np.asarray(action, float), np.asarray(last, float), np.bool_(finite), CFG)
    return float(r), float(done)


def test_perfect_tracking_gives_max_reward():
    r, done = reward(errors())
    assert done == 0.0
    assert r == pytest.approx(CFG.w_pos_coarse + CFG.w_pos_fine + CFG.w_yaw)


def test_reward_decreases_with_position_and_yaw_error():
    pos = [reward(errors(pos_m=e))[0] for e in (0.0, 0.05, 0.2, 1.0, 3.0)]
    yaw = [reward(errors(yaw_rad=e))[0] for e in (0.0, 0.1, 0.5, 2.0)]
    assert all(a > b for a, b in zip(pos, pos[1:]))
    assert all(a > b for a, b in zip(yaw, yaw[1:]))
    assert reward(errors(yaw_rad=0.5))[0] == pytest.approx(reward(errors(yaw_rad=-0.5))[0])


def test_effort_action_rate_and_tilt_penalties():
    base = reward(errors())[0]
    assert reward(errors(), action=np.full(8, 0.5), last=np.full(8, 0.5))[0] == pytest.approx(
        base - CFG.w_effort * 0.5)
    assert reward(errors(), action=np.full(8, 0.5))[0] == pytest.approx(
        base - CFG.w_effort * 0.5 - CFG.w_action_rate * 0.25)
    jump = reward(errors(), action=np.full(8, 0.5), last=np.full(8, -0.5))[0]
    assert jump == pytest.approx(base - CFG.w_effort * 0.5 - CFG.w_action_rate * 1.0)
    assert reward(errors(cos_tilt=0.9))[0] == pytest.approx(base - CFG.w_tilt * 0.1)


def test_termination_conditions():
    r_far, done_far = reward(errors(pos_m=CFG.max_pos_error_m + 0.1))
    assert done_far == 1.0 and r_far < -CFG.termination_penalty + 1.0
    _, done_tilt = reward(errors(cos_tilt=CFG.min_cos_tilt - 0.01))
    assert done_tilt == 1.0
    _, done_ok = reward(errors(pos_m=CFG.max_pos_error_m - 0.1, cos_tilt=CFG.min_cos_tilt + 0.01))
    assert done_ok == 0.0


def test_non_finite_state_terminates_with_finite_penalty():
    err = core.Errors(np.full(3, np.nan), np.float64(np.nan), np.float64(np.nan), np.float64(np.nan))
    r, done = reward(err, finite=False)
    assert done == 1.0
    assert r == -CFG.termination_penalty
