"""Evaluate a model on a benchmark.

Usage (after `source scripts/env.sh`, on a GPU node):
    python scripts/run_eval.py --model qwen3.8-27b-nvfp4_nothink --bench robospatial --limit 10
    python scripts/run_eval.py --model qwen3.8-27b-nvfp4_think --bench robospatial --server http://127.0.0.1:8000/v1

Starts the model's server on --gpus unless --server points at one already running.
Results: outputs/<model>/<engine>/<bench>/ (see robovlmbench/runner.py).
"""

import argparse
import json

from robovlmbench.runner import BENCHMARKS, run


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="<model> or <model>_<mode> from configs/models/, e.g. qwen3.8-27b-nvfp4_think")
    parser.add_argument("--bench", required=True, choices=list(BENCHMARKS))
    parser.add_argument("--engine", default="vllm")
    parser.add_argument("--gpus", default="0", help="GPUs for the server, e.g. 0 or 0,1 (with --server: the GPUs it uses, recorded as the hardware)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--limit", type=int, help="only the first N samples (small-scale test)")
    parser.add_argument("--server", help="URL of a server already running, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--shard", help="I/N: run only part I of N, e.g. 1/2 on one GPU and 2/2 on another (capx only)")
    args = parser.parse_args()
    shard = None
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        if not 1 <= i <= n:
            parser.error(f"--shard {args.shard}: needs 1 <= I <= N")
        shard = (i, n)

    scores = run(args.model, args.bench, args.engine, gpus=args.gpus, port=args.port,
                 limit=args.limit, server_url=args.server, shard=shard)
    print(json.dumps({k: v for k, v in scores.items() if k != "per_sample"}, indent=1))


if __name__ == "__main__":
    main()
