#!/usr/bin/env bash
# Train + evaluate on the GPU workstation (headless).
#
# Usage:  scripts/run_ws1.sh [smoke|full|all|nav_smoke|nav_full|nav]
#           smoke / full / all (default: all = smoke, then full): station keeping + tracking (control task)
#           nav_smoke / nav_full / nav (= nav_smoke, then nav_full): sonar obstacle avoidance, RL vs VFH+
#           nav2_smoke / nav2_full / nav2: fan memory, safety shaping, moving obstacles, staged curriculum
# Env:    MJRL_VENV  (default: $HOME/.venvs/mjrl)  venv with jax[cuda12], mujoco,
#                    mujoco-mjx, playground, brax, imageio, imageio-ffmpeg, pytest (+ matplotlib for nav plots)
#         RUNS_DIR   (default: <repo>/runs)
#         EVAL_EPISODES (default: 20) episodes per evaluation cell in the full stages
#         XLA_PYTHON_CLIENT_PREALLOCATE (default: false) so training can share the GPU with other jobs
#
# Every stage streams line-by-line progress (python -u) and is tee'd to a log
# file under $RUNS_DIR, so `tail -f` works while it runs.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MJRL_VENV:-$HOME/.venvs/mjrl}"
RUNS_DIR="${RUNS_DIR:-$REPO_DIR/runs}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
STAGE="${1:-all}"
PY="$VENV/bin/python"

case "$STAGE" in
  smoke|full|all|nav_smoke|nav_full|nav|nav2_smoke|nav2_full|nav2) ;;
  *) echo "usage: $0 [smoke|full|all|nav_smoke|nav_full|nav|nav2_smoke|nav2_full|nav2]" >&2; exit 2 ;;
esac
if [[ ! -x "$PY" ]]; then
  echo "error: no python at $PY (set MJRL_VENV)" >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1  # avoid unrelated site plugins (e.g. ROS launch_testing)
cd "$REPO_DIR"
mkdir -p "$RUNS_DIR"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo "[$(stamp)] ===== $* =====" ; }

banner "environment"
"$PY" -u - <<'EOF' 2>&1 | tee "$RUNS_DIR/environment.log"
import importlib, platform
print("python", platform.python_version())
for mod in ("mujoco", "mujoco.mjx", "jax", "brax", "mujoco_playground", "imageio", "PIL", "matplotlib"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:18s} {getattr(m, '__version__', 'n/a')}")
    except Exception as exc:  # report every missing piece, then fail below
        print(f"{mod:18s} MISSING ({exc})")
import jax
print("jax devices", jax.devices())
EOF

banner "unit tests (incl. MJX/CPU parity)"
"$PY" -u -m pytest -v 2>&1 | tee "$RUNS_DIR/pytest.log"

run_smoke() {
  local out="$RUNS_DIR/smoke"
  mkdir -p "$out"
  banner "smoke: train"
  "$PY" -u train.py --preset smoke --out-dir "$out" 2>&1 | tee "$out/train.log"
  banner "smoke: eval"
  JAX_PLATFORMS=cpu "$PY" -u eval.py --controllers pid,rl --policy-dir "$out/policy" \
    --episodes 2 --levels 0,2 --scenarios combined \
    --video-cells station_nominal_L2 --out-dir "$out/eval" 2>&1 | tee "$out/eval.log"
}

run_full() {
  local out="$RUNS_DIR/full"
  mkdir -p "$out"
  banner "full: train"
  "$PY" -u train.py --preset full --out-dir "$out" 2>&1 | tee "$out/train.log"
  banner "full: eval"
  JAX_PLATFORMS=cpu "$PY" -u eval.py --controllers pid,rl --policy-dir "$out/policy" \
    --episodes "$EVAL_EPISODES" --out-dir "$out/eval" 2>&1 | tee "$out/eval.log"
  banner "full: results table -> $out/eval/results.md"
}

run_nav_smoke() {
  local out="$RUNS_DIR/nav_smoke"
  mkdir -p "$out"
  banner "nav_smoke: train"
  "$PY" -u train.py --task nav --preset smoke --out-dir "$out" 2>&1 | tee "$out/train.log"
  banner "nav_smoke: eval (RL vs VFH+)"
  JAX_PLATFORMS=cpu "$PY" -u eval_nav.py --controllers baseline,rl --policy-dir "$out/policy" \
    --episodes 2 --densities low,high --current-levels 0,2 \
    --video-cells high_L2 --plot-cell high_L2 --out-dir "$out/nav_eval" 2>&1 | tee "$out/nav_eval.log"
}

run_nav_full() {
  local out="$RUNS_DIR/nav_full"
  mkdir -p "$out"
  banner "nav_full: train"
  "$PY" -u train.py --task nav --preset full --out-dir "$out" 2>&1 | tee "$out/train.log"
  banner "nav_full: eval (RL vs VFH+)"
  JAX_PLATFORMS=cpu "$PY" -u eval_nav.py --controllers baseline,rl --policy-dir "$out/policy" \
    --episodes "$EVAL_EPISODES" --out-dir "$out/nav_eval" 2>&1 | tee "$out/nav_eval.log"
  banner "nav_full: results table -> $out/nav_eval/results.md"
}

# nav2: same eval seeds and VFH+ baseline as nav (static cells are directly comparable), plus moving-obstacle
# cells and the VFH+ variant with log-odds decay (config/nav_baseline_decay.json).
run_nav2_smoke() {
  local out="$RUNS_DIR/nav2_smoke"
  mkdir -p "$out"
  banner "nav2_smoke: train (3-stage curriculum)"
  "$PY" -u train.py --task nav2 --preset smoke --out-dir "$out" 2>&1 | tee "$out/train.log"
  banner "nav2_smoke: eval (RL vs VFH+, static + moving)"
  JAX_PLATFORMS=cpu "$PY" -u eval_nav.py --task nav2 --controllers baseline,baseline_decay,rl \
    --policy-dir "$out/policy" --movers static,moving --episodes 2 --densities low,high --current-levels 0,2 \
    --video-cells high_L2_mov --plot-cell high_L2_mov --out-dir "$out/nav_eval" 2>&1 | tee "$out/nav_eval.log"
}

run_nav2_full() {
  local out="$RUNS_DIR/nav2_full"
  mkdir -p "$out"
  banner "nav2_full: train (3-stage curriculum)"
  "$PY" -u train.py --task nav2 --preset full --out-dir "$out" 2>&1 | tee "$out/train.log"
  banner "nav2_full: eval (RL vs VFH+, static + moving)"
  JAX_PLATFORMS=cpu "$PY" -u eval_nav.py --task nav2 --controllers baseline,baseline_decay,rl \
    --policy-dir "$out/policy" --movers static,moving --episodes "$EVAL_EPISODES" \
    --video-cells high_L2,high_L2_mov --plot-cell high_L2_mov --out-dir "$out/nav_eval" 2>&1 | tee "$out/nav_eval.log"
  banner "nav2_full: results table -> $out/nav_eval/results.md"
}

case "$STAGE" in
  smoke) run_smoke ;;
  full) run_full ;;
  all) run_smoke; run_full ;;
  nav_smoke) run_nav_smoke ;;
  nav_full) run_nav_full ;;
  nav) run_nav_smoke; run_nav_full ;;
  nav2_smoke) run_nav2_smoke ;;
  nav2_full) run_nav2_full ;;
  nav2) run_nav2_smoke; run_nav2_full ;;
esac
banner "done ($STAGE)"
