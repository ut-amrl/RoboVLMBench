"""Write a benchmark's report, one self-contained HTML page, from all its complete runs in outputs/.

Usage (after `source scripts/env.sh`):
    python scripts/report.py --bench robospatial      # -> reports/robospatial.html
    python scripts/report.py --bench instructpart     # -> reports/instructpart.html
    python scripts/report.py --bench robobench        # -> reports/robobench.html
    python scripts/report.py --bench epic_bench       # -> reports/epic_bench.html
    python scripts/report.py --bench ovdeval          # -> reports/ovdeval.html
    python scripts/report.py --bench capx             # -> reports/capx.html
"""

import argparse
from pathlib import Path

from robovlmbench.benchmarks.capx import report as capx_report
from robovlmbench.benchmarks.epic_bench import report as epic_bench_report
from robovlmbench.benchmarks.instructpart import report as instructpart_report
from robovlmbench.benchmarks.ovdeval import report as ovdeval_report
from robovlmbench.benchmarks.robobench import report as robobench_report
from robovlmbench.benchmarks.robospatial import report as robospatial_report
from robovlmbench.report import REPORTS_DIR, find_runs

REPORTS = {"robospatial": robospatial_report.build, "instructpart": instructpart_report.build,
           "robobench": robobench_report.build, "epic_bench": epic_bench_report.build,
           "ovdeval": ovdeval_report.build, "capx": capx_report.build}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bench", required=True, choices=list(REPORTS))
    parser.add_argument("--out", type=Path, help="default: reports/<bench>.html")
    args = parser.parse_args()

    out = args.out or REPORTS_DIR / f"{args.bench}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(REPORTS[args.bench](find_runs(args.bench)))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
