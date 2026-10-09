#!/usr/bin/env bash
# Set up CaP-X (github.com/capgym/cap-x) for the Robosuite and LIBERO-PRO suites, if this node's hardware can
# run them (an NVIDIA GPU with EGL for headless rendering, x86_64); otherwise skip, saying why. Clone it into
# third_party/cap-x at a pinned commit with the submodules those suites need, and build one env per suite
# (their robosuite versions conflict), exactly as pinned in CaP-X's own uv.lock:
#   .venvs/capx-robosuite-<arch>/  Python 3.10, extras robosuite + contactgraspnet
#   .venvs/capx-libero-<arch>/     Python 3.12, extras libero + contactgraspnet
# and fetch LIBERO-PRO's perturbed task files into data/libero-pro/.
# BEHAVIOR is not supported: CaP-X's BEHAVIOR-1K needs Isaac Sim 5.1, which crashes at renderer startup on
# NVIDIA drivers newer than the R580 series (driver 615.71: segfault in librtx.scenedb.plugin.so); Isaac Sim 6.1
# works there, but BEHAVIOR-1K only accepts 5.1.0.
# The main env (.venvs/<arch>-<cuda>/) is not touched.
#
# Usage: bash scripts/setup_capx.sh [robosuite] [libero]    (default: both; also run by setup_env.sh)
# Then:  source scripts/capx_env.sh robosuite|libero
# Check: python scripts/check_env.py --only CaP-X   (or: sbatch -p <gpu partition> scripts/slurm/check_capx.sbatch)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/env.sh

if [ "$RVB_GPU" = "cpu" ]; then
  echo "CaP-X: skipped, no NVIDIA GPU on this node"
  exit 0
fi

SUITES=("$@")
[ ${#SUITES[@]} -eq 0 ] && SUITES=(robosuite libero)
BEHAVIOR_UNSUPPORTED="BEHAVIOR is not supported: CaP-X's BEHAVIOR-1K needs Isaac Sim 5.1, which crashes at renderer \
startup on NVIDIA drivers newer than the R580 series (Isaac Sim 6.1 works, but BEHAVIOR-1K only accepts 5.1.0)."
for suite in "${SUITES[@]}"; do
  case "$suite" in
    robosuite|libero) ;;
    behavior) echo "$BEHAVIOR_UNSUPPORTED" >&2; exit 1 ;;
    *) echo "unknown suite: $suite (use robosuite or libero)" >&2; exit 1 ;;
  esac
done

# Hardware CaP-X can't run on: say why, and record it for this GPU type (cache/<gpu>/capx-check.json), so
# CaP-X runs here skip its suites and the report shows them as N/A
skip() {
  echo "CaP-X: skipped, $1"
  python -c 'import sys; from robovlmbench.benchmarks.capx import SUITES, record_support
for suite in SUITES: record_support(sys.argv[1], sys.argv[2], suite, False, sys.argv[3])' \
    "$RVB_GPU" "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" "$1"
  exit 0
}
[ "$(uname -m)" = x86_64 ] || skip "its envs are set up for x86_64 only; this node is $(uname -m)"
# MuJoCo renders headless through the NVIDIA driver's EGL library
ldconfig -p | grep -q libEGL_nvidia || skip "no NVIDIA EGL library (libEGL_nvidia.so) for headless rendering"

CAPX_URL=https://github.com/capgym/cap-x.git
CAPX_COMMIT=53e9966d7a8e2fa7494676772bccc35280f5c0ed
CAPX_DIR="$REPO_ROOT/third_party/cap-x"
# Submodules the Robosuite and LIBERO envs install from (b1k = BEHAVIOR is left out)
SUBMODULES=(
  capx/third_party/robosuite
  capx/third_party/libero_dependencies/robosuite
  capx/third_party/LIBERO-PRO
  capx/third_party/sam3
  capx/third_party/contact_graspnet_pytorch
  capx/third_party/curobo
  capx/third_party/verl
)

# 1. CaP-X source at the pinned commit
mkdir -p third_party
if [ ! -d "$CAPX_DIR/.git" ]; then
  git clone -q "$CAPX_URL" "$CAPX_DIR"
fi
if [ "$(git -C "$CAPX_DIR" rev-parse HEAD)" != "$CAPX_COMMIT" ]; then
  git -C "$CAPX_DIR" fetch -q origin "$CAPX_COMMIT"
  git -C "$CAPX_DIR" checkout -q --detach "$CAPX_COMMIT"
fi
git -C "$CAPX_DIR" submodule update --init --recursive "${SUBMODULES[@]}"
echo "third_party/cap-x @ ${CAPX_COMMIT:0:12}"

# 2. One env per suite, from CaP-X's uv.lock (--frozen: install the lock as is, without re-resolving)
build_env() {  # <suite> <python> <extras...>
  local suite="$1" python="$2"; shift 2
  local venv="$REPO_ROOT/.venvs/capx-$suite-$(uname -m)"
  local extras=()
  for e in "$@"; do extras+=(--extra "$e"); done
  uv python install "$python"
  # (re)create the env unless it is built on a Python inside the repo
  if ! grep -q "^home = $UV_PYTHON_INSTALL_DIR/" "$venv/pyvenv.cfg" 2>/dev/null; then
    uv venv -q --clear --python "$python" --prompt "capx-$suite" "$venv"
  fi
  (cd "$CAPX_DIR" && UV_PROJECT_ENVIRONMENT="$venv" uv sync --frozen "${extras[@]}")
  echo "env: $venv"
}

for suite in "${SUITES[@]}"; do
  case "$suite" in
    robosuite)
      build_env robosuite 3.10 robosuite contactgraspnet
      ;;
    libero)
      build_env libero 3.12 libero contactgraspnet
      libero_root="$CAPX_DIR/capx/third_party/LIBERO-PRO/libero/libero"
      # LIBERO-PRO's perturbed suites (libero_{goal,spatial,object,10}_{swap,task,object,lan}) need task
      # files published separately as a Hugging Face dataset (~10 MB), at this commit
      libero_pro="${ROBOVLMBENCH_DATA:-$REPO_ROOT/data}/libero-pro"
      python -I - "$libero_pro" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("zhouxueyang/LIBERO-Pro", repo_type="dataset", local_dir=sys.argv[1],
                  revision="c86fc3b8293185a6f373677018ff3e37f8391602")
EOF
      # LIBERO reads every suite from one bddl_files/ and one init_files/ folder: link the suite folders of
      # the submodule and of the dataset into cache/libero/ (the submodule's copy wins where both have one)
      for kind in bddl_files init_files; do
        rm -rf "cache/libero/$kind"
        mkdir -p "cache/libero/$kind"
        for suite_dir in "$libero_root/$kind"/libero_*/ "$libero_pro/$kind"/libero_*/; do
          suite_dir="${suite_dir%/}"
          [ -e "cache/libero/$kind/${suite_dir##*/}" ] || ln -s "$suite_dir" "cache/libero/$kind/${suite_dir##*/}"
        done
      done
      # LIBERO asks for its paths interactively unless config.yaml exists; keep it in the repo, not ~/.libero
      cat > cache/libero/config.yaml <<EOF
assets: $libero_root/assets
bddl_files: $REPO_ROOT/cache/libero/bddl_files
benchmark_root: $libero_root
datasets: $libero_root/../datasets
init_states: $REPO_ROOT/cache/libero/init_files
EOF
      echo "LIBERO-PRO task files: $libero_pro (linked into cache/libero/)"
      ;;
    *) echo "unknown suite: $suite (use robosuite or libero)" >&2; exit 1 ;;
  esac
done

echo
echo "Done. Use with:  source scripts/capx_env.sh robosuite   (or: libero)"
