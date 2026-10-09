#!/usr/bin/env bash
# Build the environment for this node's hardware: install uv into .venvs/ if missing, clone the official
# evaluation code into third_party/ at pinned commits, and create .venvs/<arch>-<cuda>/
# (vLLM included). On nodes with a GPU TensorRT-LLM supports, also fetch its container; on nodes that can run
# CaP-X's simulators, set up CaP-X (its own envs).
#
# Usage: bash scripts/setup_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/env.sh
echo "hardware: $RVB_ENV_TYPE, GPU: $RVB_GPU"

# 1. uv, in .venvs/tools/<arch>/
if [ ! -x "$RVB_TOOLS/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$RVB_TOOLS" sh  # no receipt or PATH edits in $HOME
fi
echo "uv: $(uv --version)"

# 2. Official evaluation code at pinned commits
mkdir -p third_party logs
while read -r name url commit; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  dir="third_party/$name"
  if [ ! -d "$dir/.git" ]; then
    git init -q "$dir"
    git -C "$dir" remote add origin "$url"
  fi
  if [ "$(git -C "$dir" rev-parse -q --verify HEAD 2>/dev/null)" != "$commit" ]; then
    git -C "$dir" fetch -q --depth 1 origin "$commit"
    git -C "$dir" checkout -q --detach FETCH_HEAD
  fi
  echo "$dir @ ${commit:0:12}"
done < scripts/third_party_repos.txt

# 3. Python environment for this hardware, exactly as pinned in uv.lock
uv sync --locked
echo "env: $RVB_VENV"

# 4. TensorRT-LLM: NVIDIA's container, only on GPUs it supports
bash scripts/setup_trtllm.sh

# 5. CaP-X: its own envs (Robosuite, LIBERO-PRO), only on hardware that can run them
bash scripts/setup_capx.sh

echo
echo "Done. Check it with:  source scripts/env.sh && python scripts/check_env.py"
echo "                 or:  sbatch scripts/slurm/check_env.sbatch"
