| Benchmark | Capability | Format and scoring | Download size |
|---|---|---|---|
| **RoboBench** | System-2 reasoning: understanding instructions, perception reasoning, planning (including across different robots), affordances, error analysis | Images and video; mostly question-answer, with open-ended planning answers | 21.6 GB |
| **EPIC-Bench** | Fine-grained grounding in 3 areas: finding targets, navigation (paths, ground, visual matching), manipulation (affordance region, contact, placement region) | The model outputs boxes (points for path tasks), scored by IoU against ground-truth masks; some tasks add a count or yes/no term | 7.75 GB |
| **RoboSpatial-Home** | Spatial reasoning: object relations, pointing to free space, whether something fits; RGB plus depth | Yes/no answers, plus points scored against a free-space mask | 35 MB |
| **OVDEval** | Open-vocabulary detection with misleading descriptions: color, material, position, relationship, negation (plus logos, celebrities, landmarks) | The model boxes only objects matching the correct descriptions; COCO-style scoring per target (average recall and precision); precision needs a confidence per box | 2.3 GB |
| **InstructPart** | Functional-part grounding: from a task instruction ("add some water") find the part to act on (kettle lid); a second version names the part directly | The model outputs points, counted correct if they land on the ground-truth part mask (our scoring; the official metric needs a predicted mask); one target per sample | 0.23 GB |
| **CaP-X** | Closed-loop manipulation: the model writes Python that calls perception and control functions to drive a simulated robot (39 tasks in Robosuite, LIBERO-PRO, BEHAVIOR) | Code generation, single-turn and multi-turn (8 levels); scored by task success | — |

## Installation

Needs Linux, `git` and `curl`.

```bash
git clone <repo-url> RoboVLMBench && cd RoboVLMBench
bash scripts/setup_env.sh    # run once per hardware type (e.g. on a CPU node and on a GPU node)
```

Setup builds a Python env with vLLM for the node's hardware in `.venvs/<arch>-<cuda>/`, clones the official eval code into `third_party/`, and on nodes with a supported GPU fetches NVIDIA's TensorRT-LLM container. The repo can sit on storage shared by many nodes: `source scripts/env.sh` picks the env and the per-GPU caches (`cache/<gpu>/`) for whichever node you are on. Model weights go to `models/` (set `ROBOVLMBENCH_MODELS` to put them elsewhere).

Check the setup (all lines should say PASS, or SKIP where a check doesn't apply to the node):

```bash
source scripts/env.sh
python scripts/check_env.py
# or on Slurm, from a login node: sbatch scripts/slurm/check_env.sbatch  (log in logs/)
```

## Data

Downloads go to `data/` (set `ROBOVLMBENCH_DATA` to put them elsewhere), at the versions pinned in `scripts/dataset_revisions.txt`. Re-running resumes or skips what is already there.

```bash
source scripts/env.sh
python scripts/download_data.py    # all benchmarks; or pick some: --bench ovdeval instructpart
python scripts/check_data.py       # counts and files match the release
# or on Slurm, from a login node: sbatch scripts/slurm/download_data.sbatch  (downloads, then checks)
```

RoboBench has 88k files; set `HF_TOKEN` (or run `hf auth login` after `source scripts/env.sh`) for higher Hugging Face rate limits.

## Speed and memory on an edge device (NVIDIA Jetson Thor)

A small package of requests taken from the full runs times each model on the device and measures the memory it
takes. It needs only the device's own vLLM (0.31.0 matches the full runs) and Python 3.8+; it installs nothing.
Details: [robovlmbench/edge/README.md](robovlmbench/edge/README.md).

1. Download `edge-subset.tar.gz` from the latest pre-release on
   [Releases](https://github.com/ut-amrl/RoboVLMBench/releases) and compare `sha256sum edge-subset.tar.gz`
   with the checksum in the release notes.
2. `tar -xzf edge-subset.tar.gz && cd edge-subset`
3. `python3 run_bench.py --check-system`: reads only, and prints anything missing with the command that would
   fix it. Run such commands only with the device owner's approval.
4. For each model (`python3 run_bench.py` lists them): `python3 run_bench.py --model <model>`. This starts the
   system's vLLM with the memory the model needs (the first start downloads its weights, about 44 GiB for all
   models), times its requests, records its memory, and stops it.
5. Send back the `results/` folder.

The package is built from the full runs with `python scripts/make_edge_subset.py` (needs `outputs/`).
