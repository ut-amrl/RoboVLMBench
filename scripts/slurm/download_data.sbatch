#!/bin/bash
# Run scripts/download_data.py on a CPU compute node (all benchmarks, about 32 GB).
# Usage, from the repo root: sbatch scripts/slurm/download_data.sbatch [--bench robobench ...]
# (add -A <allocation> if you have more than one). Log: logs/download_data-<jobid>.out
#SBATCH -J download_data
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o logs/download_data-%j.out
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/download_data.py ]; then
  echo "Submit this from the RoboVLMBench repo root." >&2
  exit 1
fi

source scripts/env.sh
echo "node: $(hostname), env: $RVB_ENV_TYPE, GPU: $RVB_GPU"
export HF_HUB_DISABLE_PROGRESS_BARS=1
python scripts/download_data.py "$@"
python scripts/check_data.py "$@"
