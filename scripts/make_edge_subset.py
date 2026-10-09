"""Pick the edge subset from the full runs and pack it (see robovlmbench/edge).

Usage (after `source scripts/env.sh`, on a GPU node; needs the benchmarks' data and the full no-think runs in
outputs/; the first build starts each model's server once to measure its memory, on CUDA_VISIBLE_DEVICES (one model per
GPU at a time) or GPU 0):
    python scripts/make_edge_subset.py                 # -> dist/edge-subset/ and dist/edge-subset.tar.gz
    python scripts/make_edge_subset.py --budget-s 480 --out dist/edge-subset
"""

import argparse
import os
from pathlib import Path

from robovlmbench import REPO_ROOT
from robovlmbench.edge import device_seconds, measure_memory, models, pack, select


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--budget-s", type=float, default=540,
                        help="the most time the slowest model may take on the device, estimated (default: 9 minutes, "
                             "a margin under 10 for the estimate's error)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "dist" / "edge-subset")
    args = parser.parse_args()

    names = models()
    print(f"models: {', '.join(names)}")
    subset = select(names, args.budget_s)
    gpus = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")
    archive = pack(subset, args.out, args.budget_s, measure_memory(subset, gpus))

    by_stratum = {}
    for it in subset.items:
        by_stratum.setdefault(it.benchmark, {}).setdefault(it.stratum, 0)
        by_stratum[it.benchmark][it.stratum] += 1
    print(f"\n{len(subset.items)} requests:")
    for bench, strata in by_stratum.items():
        print(f"  {bench}: " + ", ".join(f"{s} {n}" for s, n in strata.items()))
    print("\nestimated time on the device (s): " + ", ".join(
        f"{m} {sum(device_seconds(subset.fits[m], it.records[m]) for it in subset.items):.0f}" for m in names))
    print("\nerror of run_bench.py's full-benchmark estimate on a simulated device (model left out of the pick):")
    benches = list(next(iter(subset.validation.values())))
    print(f"  {'':34s}" + "".join(f"{b:>13s}" for b in benches))
    for m, errors in subset.validation.items():
        print(f"  {m:34s}" + "".join(f"{errors[b] * 100:12.1f}%" for b in benches))
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"\npackage: {args.out} ({size / 1e6:.1f} MB), {archive} ({archive.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
