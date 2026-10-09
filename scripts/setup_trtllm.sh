#!/usr/bin/env bash
# Fetch NVIDIA's TensorRT-LLM container as an Apptainer image, on nodes with a GPU it supports.
# The container brings TensorRT-LLM's own PyTorch, CUDA and MPI, which can't share the vLLM env.
# Run it with:  apptainer exec --nv .venvs/trtllm-<version>-<arch>.sif trtllm-serve ...
#
# Usage: bash scripts/setup_trtllm.sh   (also run by setup_env.sh)
set -euo pipefail

TRTLLM_VERSION="1.3.0rc29"  # first release with Qwen3.5/3.8 vision-language models (Qwen3_5ForConditionalGeneration)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/env.sh"

if [ "$RVB_GPU" = "cpu" ]; then
  echo "TensorRT-LLM: skipped, no NVIDIA GPU on this node"
  exit 0
fi
cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
if [ "${cc%%.*}" -lt 8 ]; then
  echo "TensorRT-LLM: skipped, needs compute capability 8.0+ (Ampere or newer); this GPU is $cc"
  exit 0
fi

sif="$REPO_ROOT/.venvs/trtllm-${TRTLLM_VERSION}-$(uname -m).sif"
if [ -f "$sif" ]; then
  echo "TensorRT-LLM: $sif"
  exit 0
fi

if ! command -v apptainer >/dev/null 2>&1; then
  module load tacc-apptainer 2>/dev/null || { echo "TensorRT-LLM: apptainer not found" >&2; exit 1; }
fi
# Unpacking the image is ~30 GB of small files: on node-local disk it takes minutes, on shared
# storage like $SCRATCH it can take over an hour. Build locally, then move the finished image.
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/trtllm-pull.XXXXXX")"
trap 'rm -rf "$build_dir"' EXIT
echo "TensorRT-LLM: pulling container $TRTLLM_VERSION (~13 GB image) via $build_dir"
APPTAINER_TMPDIR="$build_dir" apptainer pull --disable-cache "$build_dir/image.sif" \
  "docker://nvcr.io/nvidia/tensorrt-llm/release:$TRTLLM_VERSION"
mkdir -p "$(dirname "$sif")"
mv "$build_dir/image.sif" "$sif.part"
mv "$sif.part" "$sif"
echo "TensorRT-LLM: $sif"
