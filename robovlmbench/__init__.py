"""RoboVLMBench: evaluation harness for VLMs on robot tasks."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
DATA_DIR = Path(os.environ.get("ROBOVLMBENCH_DATA", REPO_ROOT / "data"))
