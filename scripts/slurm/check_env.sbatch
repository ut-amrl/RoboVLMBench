#!/bin/bash
# Run scripts/check_env.py on a CPU compute node.
# Usage, from the repo root: sbatch scripts/slurm/check_env.sbatch
# (add -A <allocation> if you have more than one). Log: logs/check_env-<jobid>.out
#SBATCH -J check_env
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -o logs/check_env-%j.out
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/check_env.py ]; then
  echo "Submit this from the RoboVLMBench repo root." >&2
  exit 1
fi

source scripts/env.sh
echo "node: $(hostname), env: $RVB_ENV_TYPE, GPU: $RVB_GPU"
python scripts/check_env.py
