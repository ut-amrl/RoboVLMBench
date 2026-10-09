# Pick the Python env and caches for this node's hardware. Source it before running anything:
#   source scripts/env.sh
#
# The repo sits on storage shared by every node, so per-hardware pieces are kept apart:
#   .venvs/<arch>-<cuda>/  packages differ only by CPU architecture and CUDA generation
#   cache/<gpu>/           compiled GPU code (vLLM, PyTorch, Triton) is specific to the GPU model
# Code, third_party/, data/, models/ and .venvs/python/ are shared.

RVB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# CUDA generation the driver supports. Every env we define is CUDA 13 (driver >= 580); CPU-only
# nodes have no driver and use the same packages, with the GPU parts unused.
if command -v nvidia-smi >/dev/null 2>&1; then
  _driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
  if [ "$_driver_major" -lt 580 ]; then
    echo "env.sh: NVIDIA driver $_driver_major is older than 580 (CUDA 13); no env is defined for it." >&2
    return 1
  fi
  # e.g. "NVIDIA RTX PRO 6000 Blackwell Server Edition" -> "rtx-pro-6000-blackwell-server-edition"
  RVB_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 \
    | sed -e 's/^NVIDIA //' | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed -e 's/-$//')"
else
  RVB_GPU="cpu"
fi
RVB_ENV_TYPE="$(uname -m)-cu13"
unset _driver_major

export RVB_ROOT RVB_GPU RVB_ENV_TYPE
export RVB_VENV="$RVB_ROOT/.venvs/$RVB_ENV_TYPE"

# Everything the project installs or caches stays inside the repo, never in $HOME:
#   .venvs/tools/<arch>/  uv itself and Python launchers
#   .venvs/python/        Python interpreters the envs are built on
#   cache/                download and build caches (uv, pip, Hugging Face, torch, ...)
export RVB_TOOLS="$RVB_ROOT/.venvs/tools/$(uname -m)"
export UV_PYTHON_INSTALL_DIR="$RVB_ROOT/.venvs/python"
export UV_PYTHON_BIN_DIR="$RVB_TOOLS"
export UV_PROJECT_ENVIRONMENT="$RVB_VENV"            # where `uv sync` installs
export XDG_CACHE_HOME="$RVB_ROOT/cache"              # default cache of uv, pip, HF, torch and most tools
export XDG_DATA_HOME="$RVB_ROOT/.venvs/share"
export VLLM_CONFIG_ROOT="$RVB_ROOT/cache/vllm-config"
export VLLM_NO_USAGE_STATS=1
export APPTAINER_CONFIGDIR="$RVB_ROOT/cache/apptainer"
export APPTAINER_CACHEDIR="$RVB_ROOT/cache/apptainer/cache"
unset UV_CACHE_DIR HF_HOME PIP_CACHE_DIR TORCH_HOME   # if the shell sets these, they would override XDG_CACHE_HOME

export HF_HUB_CACHE="${ROBOVLMBENCH_MODELS:-$RVB_ROOT/models}"  # model weights, shared by all nodes

_cache="$RVB_ROOT/cache/$RVB_GPU"
export VLLM_CACHE_ROOT="$_cache/vllm"
export TRITON_CACHE_DIR="$_cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$_cache/torchinductor"
export FLASHINFER_WORKSPACE_BASE="$_cache"
export CUDA_CACHE_PATH="$_cache/nv"
unset _cache

export PATH="$RVB_VENV/bin:$RVB_TOOLS:$PATH"

# TensorRT-LLM runs in a container (scripts/setup_trtllm.sh)
command -v apptainer >/dev/null 2>&1 || module load tacc-apptainer >/dev/null 2>&1 || true
