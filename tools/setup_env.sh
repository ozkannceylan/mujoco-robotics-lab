#!/usr/bin/env bash
#
# tools/setup_env.sh — fresh-clone environment setup for mujoco-robotics-lab.
#
# Idempotent: safe to re-run. Anything already present is skipped.
#
# Usage:
#   ./tools/setup_env.sh
#
# What it does:
#   1. Installs the Python dependencies (note: the real Pinocchio is `pin` on
#      PyPI, NOT the unrelated package named `pinocchio`).
#   2. Sparse-clones the MuJoCo Menagerie robot models the labs need into both
#      lab-2-Ur5e-robotics-lab/models/mujoco_menagerie (labs 2-6) and
#      third_party/mujoco_menagerie (lab 7).
#   3. Best-effort apt install of libegl1 for headless MuJoCo rendering.
#   4. Prints the MUJOCO_GL=egl hint needed for headless runs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MENAGERIE_URL="https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_MODELS=(universal_robots_ur5e robotiq_2f85 unitree_g1)
MENAGERIE_DESTS=(
  "${REPO_ROOT}/lab-2-Ur5e-robotics-lab/models/mujoco_menagerie"
  "${REPO_ROOT}/third_party/mujoco_menagerie"
)

PYTHON_BIN="${PYTHON_BIN:-python3}"

info() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }


# ---------------------------------------------------------------------------
# 1. Python dependencies
# ---------------------------------------------------------------------------
info "Installing Python dependencies"
# `pin` is the real Pinocchio (provides `import pinocchio`).
# The PyPI package literally named `pinocchio` is an unrelated project.
"${PYTHON_BIN}" -m pip install \
  mujoco \
  numpy \
  pin \
  scipy \
  "imageio[ffmpeg]" \
  matplotlib \
  pytest \
  meshcat


# ---------------------------------------------------------------------------
# 2. MuJoCo Menagerie models (sparse checkout, no full history)
# ---------------------------------------------------------------------------
clone_menagerie() {
  local dest="$1"

  # Consider it present if every required model directory already exists.
  local complete=1
  for model in "${MENAGERIE_MODELS[@]}"; do
    [ -d "${dest}/${model}" ] || complete=0
  done
  if [ "${complete}" -eq 1 ]; then
    info "Menagerie already present at ${dest} — skipping"
    return 0
  fi

  if [ -e "${dest}" ] && [ -n "$(ls -A "${dest}" 2>/dev/null)" ] && [ ! -d "${dest}/.git" ]; then
    warn "${dest} exists, is non-empty, and is not a git checkout — leaving it alone."
    warn "Delete it and re-run if you want a clean Menagerie clone."
    return 0
  fi

  info "Fetching MuJoCo Menagerie into ${dest}"
  mkdir -p "${dest}"
  if [ ! -d "${dest}/.git" ]; then
    git clone --filter=blob:none --no-checkout --depth 1 "${MENAGERIE_URL}" "${dest}"
  fi
  git -C "${dest}" sparse-checkout init --cone
  git -C "${dest}" sparse-checkout set "${MENAGERIE_MODELS[@]}"
  git -C "${dest}" checkout

  for model in "${MENAGERIE_MODELS[@]}"; do
    if [ -d "${dest}/${model}" ]; then
      printf '    ok  %s\n' "${model}"
    else
      warn "missing after checkout: ${model}"
    fi
  done
}

for dest in "${MENAGERIE_DESTS[@]}"; do
  clone_menagerie "${dest}"
done


# ---------------------------------------------------------------------------
# 3. Headless rendering system library (best effort)
# ---------------------------------------------------------------------------
info "Checking headless rendering support (libEGL)"
if ldconfig -p 2>/dev/null | grep -q 'libEGL\.so'; then
  echo "    libEGL already available — skipping apt install"
elif ! command -v apt-get >/dev/null 2>&1; then
  warn "apt-get not found — install an EGL runtime manually for headless rendering."
else
  SUDO=""
  if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 && SUDO="sudo"
  fi
  if [ "$(id -u)" -ne 0 ] && [ -z "${SUDO}" ]; then
    warn "Not root and sudo unavailable — skipping 'apt-get install libegl1'."
  else
    ${SUDO} apt-get update -qq \
      && ${SUDO} apt-get install -y --no-install-recommends libegl1 \
      || warn "libegl1 install failed — headless rendering may not work."
  fi
fi


# ---------------------------------------------------------------------------
# 4. Environment hint
# ---------------------------------------------------------------------------
info "Setup complete"
cat <<'EOF'

Headless rendering (no display attached) requires the EGL backend:

    export MUJOCO_GL=egl

Add it to your shell profile, or prefix individual runs:

    MUJOCO_GL=egl python3 lab-5-grasping-manipulation/src/pick_place_demo.py

Quick check:

    MUJOCO_GL=egl python3 -c "import mujoco, pinocchio; print(mujoco.__version__, pinocchio.__version__)"

EOF
