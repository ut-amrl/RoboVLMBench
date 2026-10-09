#!/bin/bash
# Run the CaP-X checks of scripts/check_env.py on a GPU node: per suite, headless rendering and a scripted
# episode (CaP-X's oracle code). The result is recorded for the node's GPU type (cache/<gpu>/capx-check.json):
# CaP-X runs there skip the suites that fail, and reports/capx.html shows them as N/A.
# Usage, from the repo root, with a GPU partition:  sbatch -p <partition> scripts/slurm/check_capx.sbatch
# (e.g. -p rtx-small, amd-rtx or h100; add -A <allocation> if you have more than one). Log: logs/check_capx-<jobid>.out
# Set up CaP-X on that hardware first: bash scripts/setup_capx.sh on such a node (also run by setup_env.sh).
#SBATCH -J check_capx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o logs/check_capx-%j.out
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/check_env.py ]; then
  echo "Submit this from the RoboVLMBench repo root." >&2
  exit 1
fi

source scripts/env.sh
echo "node: $(hostname), env: $RVB_ENV_TYPE, GPU: $RVB_GPU"
python scripts/check_env.py --only python packages "GPU + vLLM" CaP-X
