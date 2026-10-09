# Environment for CaP-X (third_party/cap-x). Source it instead of scripts/env.sh when working with CaP-X:
#   source scripts/capx_env.sh robosuite   # Robosuite tasks (robosuite 1.5, Python 3.10)
#   source scripts/capx_env.sh libero      # LIBERO-PRO tasks (robosuite 1.4 fork, Python 3.12)
# then run CaP-X from its directory:
#   cd "$CAPX_DIR" && python capx/envs/launch.py --config-path env_configs/... [--server-url ... --model ...]
#
# Robosuite and LIBERO need conflicting robosuite versions, so each suite has its own env,
# .venvs/capx-<suite>-<arch>/ (built by scripts/setup_capx.sh). Everything CaP-X and its
# dependencies cache or write stays inside the repo, never in $HOME.

_capx_suite="${1:-robosuite}"
case "$_capx_suite" in
  robosuite|libero) ;;
  *) echo "capx_env.sh: unknown suite '$_capx_suite' (use robosuite or libero)" >&2; unset _capx_suite; return 1 ;;
esac

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh" || return 1

export CAPX_SUITE="$_capx_suite"
export CAPX_DIR="$RVB_ROOT/third_party/cap-x"
export CAPX_VENV="$RVB_ROOT/.venvs/capx-$CAPX_SUITE-$(uname -m)"
unset _capx_suite

# Point uv and the shell at the CaP-X env, never at the main env
export UV_PROJECT_ENVIRONMENT="$CAPX_VENV"
export VIRTUAL_ENV="$CAPX_VENV"
export PATH="$CAPX_VENV/bin:${PATH//$RVB_VENV\/bin:/}"

# Caches and config dirs that do not follow XDG_CACHE_HOME (or that default to ~/.config, ~/.nv, ...)
export XDG_CONFIG_HOME="$RVB_ROOT/cache/config"
export XDG_STATE_HOME="$RVB_ROOT/cache/state"
export MPLCONFIGDIR="$RVB_ROOT/cache/matplotlib"
export ROBOT_DESCRIPTIONS_CACHE="$RVB_ROOT/cache/robot_descriptions"   # PyRoKi IK server robot URDFs
export IMAGEIO_USERDIR="$RVB_ROOT/cache/imageio"
export PYTHONUSERBASE="$RVB_ROOT/.venvs/share/python-user"              # `pip install --user` fallback
export __GL_SHADER_DISK_CACHE_PATH="$RVB_ROOT/cache/$RVB_GPU/nv-gl"      # NVIDIA OpenGL/EGL shader cache
export MESA_SHADER_CACHE_DIR="$RVB_ROOT/cache/$RVB_GPU/mesa"           # Mesa's, if its EGL is loaded too
export TORCH_EXTENSIONS_DIR="$RVB_ROOT/cache/$RVB_GPU/torch_extensions"
export LIBERO_CONFIG_PATH="$RVB_ROOT/cache/libero"                      # LIBERO's config.yaml (default ~/.libero)

# Headless rendering through EGL on the GPU
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
