"""Check that the development environment is set up correctly.

Runs each benchmark's official scorer on a tiny hand-made input whose correct
score is known, and checks the inference engines can use this node's GPU.
Exits non-zero on failure; checks that don't apply to this node are skipped.
The CaP-X checks run a scripted episode per suite and record the result for this GPU type
(cache/<gpu>/capx-check.json): runs on this GPU type skip the suites that fail, and the report shows them as N/A.

Usage: source scripts/env.sh && python scripts/check_env.py [--only CaP-X ...]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


class Skip(Exception):
    """The check doesn't apply to this node."""


def check_python():
    assert sys.version_info[:2] == (3, 12), f"expected Python 3.12, got {sys.version.split()[0]}"
    assert "RVB_VENV" in os.environ, "run `source scripts/env.sh` first"
    venv = Path(os.environ["RVB_VENV"]).resolve()
    assert Path(sys.prefix).resolve() == venv, f"not running in {venv} (sys.prefix={sys.prefix})"
    return f"{sys.version.split()[0]} in .venvs/{os.environ['RVB_ENV_TYPE']}"


def check_third_party():
    pinned = []
    for line in (REPO_ROOT / "scripts" / "third_party_repos.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, _url, commit = line.split()
            head = subprocess.run(
                ["git", "-C", str(REPO_ROOT / "third_party" / name), "rev-parse", "HEAD"],
                capture_output=True, text=True,
            ).stdout.strip()
            assert head == commit, f"third_party/{name} is at {head or 'nothing'}, expected {commit}"
            pinned.append(f"{name}@{commit[:7]}")
    return ", ".join(pinned)


def check_packages():
    import cv2  # noqa: F401  (needed by robobench; fails on nodes missing system libraries)
    from pycocotools import mask  # noqa: F401
    import datasets, openai, PIL, robobench, scipy, yaml  # noqa: F401,E401

    names = ["numpy", "scipy", "pycocotools", "pillow", "opencv-python-headless", "datasets", "robobench",
             "torch", "vllm"]
    return ", ".join(f"{n} {version(n)}" for n in names)


def check_gpu():
    import torch

    if not torch.cuda.is_available():
        raise Skip("no NVIDIA GPU on this node")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    assert arch in torch.cuda.get_arch_list(), f"PyTorch {torch.__version__} has no kernels for {name} ({arch})"

    # Actually run on the GPU: an unsupported GPU fails here, not at import.
    x = torch.randn(256, 256)
    on_gpu = (x.cuda() @ x.cuda()).cpu()
    assert torch.allclose(on_gpu, x @ x, atol=1e-3), "GPU matmul result differs from CPU"

    # vLLM's compiled GPU kernels (vllm._C*; the name varies by version) fail to load if built for a different PyTorch.
    import importlib
    import pkgutil

    import vllm

    compiled = [m.name for m in pkgutil.iter_modules(vllm.__path__) if m.name.startswith("_C")]
    assert compiled, "vLLM has no compiled kernels (vllm._C*)"
    for module in compiled:
        importlib.import_module(f"vllm.{module}")
    return f"{name} ({arch}): PyTorch {torch.__version__} (CUDA {torch.version.cuda}) runs on it; vLLM kernels load"


def check_trtllm():
    if os.environ.get("RVB_GPU", "cpu") == "cpu":
        raise Skip("no NVIDIA GPU on this node")
    images = sorted((REPO_ROOT / ".venvs").glob(f"trtllm-*-{os.uname().machine}.sif"))
    if not images:
        cc = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.split()[0]
        if int(cc.split(".")[0]) < 8:
            raise Skip(f"GPU compute capability {cc} is below TensorRT-LLM's minimum (8.0)")
        raise AssertionError("TensorRT-LLM container missing; run `bash scripts/setup_trtllm.sh`")
    probe = ("import torch, tensorrt_llm; x = torch.randn(64, 64, device='cuda'); (x @ x).sum().item(); "
             "print(tensorrt_llm.__version__, torch.cuda.get_device_name(0))")
    result = subprocess.run(["apptainer", "exec", "--nv", str(images[-1]), "python3", "-c", probe],
                            capture_output=True, text=True)
    assert result.returncode == 0, f"container failed on this GPU: {result.stderr.strip().splitlines()[-1:]}"
    trt_version, gpu = result.stdout.strip().splitlines()[-1].split(" ", 1)
    return f"TensorRT-LLM {trt_version} runs on {gpu} ({images[-1].name})"


def check_robobench():
    from robobench.evaluation import get_evaluator

    results = [
        {"request_id": "right", "response": "<answer>B</answer>", "gt_answer": "B"},
        {"request_id": "wrong", "response": "A", "gt_answer": "C"},
    ]
    scores = get_evaluator("multi_choice").evaluate(
        results, {"gpt_model": None, "normalize_with_gpt": False}
    )
    assert (scores["correct"], scores["total_questions"]) == (1, 2), scores
    return "multiple-choice scorer: 1 right, 1 wrong as expected"


def check_epic_bench():
    from pycocotools import mask as mask_utils

    from robovlmbench.benchmarks.epic_bench import official

    localization_score = official().scorers.metrics.localization.localization_score

    # Ground truth in the dataset's format: compressed RLE with string counts.
    gt = np.zeros((80, 100), dtype=np.uint8)
    gt[10:30, 20:40] = 1
    rle = mask_utils.encode(np.asfortranarray(gt))
    rle["counts"] = rle["counts"].decode()

    cases = [  # (predicted boxes, ground truth, expected score)
        ([[20, 10, 40, 30]], [rle], 1.0),  # box exactly covers the target
        ([[20, 10, 30, 30]], [rle], 0.5),  # box covers half of it
        ([], [], 1.0),                     # correctly says "no target"
        ([[0, 0, 10, 10]], [], 0.0),       # false positive when nothing is there
    ]
    for pred, gt_rle, expected in cases:
        score, _, err = localization_score(pred, gt_rle, format_type="bbox", coord_mode="absolute")
        assert err is None and score == expected, f"pred={pred}: score={score}, err={err}, expected {expected}"
    return f"box-vs-mask scorer correct on {len(cases)} cases (incl. no-target)"


def check_robospatial():
    from PIL import Image

    from robovlmbench.benchmarks.robospatial import official

    evaluate_answer = official().evaluate_answer

    with tempfile.TemporaryDirectory() as tmp:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 60:80] = 255  # free space around normalized point (0.7, 0.3)
        mask_path = str(Path(tmp) / "mask.png")
        Image.fromarray(mask).save(mask_path)

        cases = [  # (ground truth, model answer, mask, expected correct)
            ("Yes", "yes, it is", None, True),
            ("No", "Yes", None, False),
            ("[(0.7, 0.3)]", "[(0.7, 0.3)]", mask_path, True),
            ("[(0.7, 0.3)]", "[(0.1, 0.9)]", mask_path, False),
        ]
        for gt, answer, mask_file, expected in cases:
            correct, *_ = evaluate_answer(gt, answer, mask_path=mask_file)
            assert correct == expected, f"gt={gt} answer={answer}: got {correct}, expected {expected}"
    return f"yes/no and point-in-mask scorer correct on {len(cases)} cases"


def check_ovdeval():
    import contextlib
    import io
    import json

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    from robovlmbench.benchmarks.ovdeval import official

    gt = {  # one image with two red cups; "blue cup" is the misleading description
        "images": [{"id": 0, "file_name": "x.jpg", "height": 100, "width": 100,
                    "text": ["red cup"], "neg_text": ["blue cup"]}],
        "categories": [{"id": 0, "name": "red cup"}, {"id": 1, "name": "blue cup"}],
        "annotations": [  # boxes as [x, y, w, h]; ids start at 1 because COCOeval never matches id 0
            {"id": 1, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 0, "category_id": 0, "bbox": [60, 60, 20, 20], "area": 400, "iscrowd": 0},
        ],
    }
    preds = [  # finds the first cup, labels it with both descriptions, misses the second cup
        {"image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 0, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.6},
    ]
    kept = official().nms_threaded(gt["annotations"], preds, iou_threshold=0.5, num_threads=1)
    assert kept == preds[:1], f"NMS should keep only the higher-scoring box, kept {kept}"

    # Same COCO calls as the official cal_map(), which only prints its scores.
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        gt_path = Path(tmp) / "gt.json"
        gt_path.write_text(json.dumps(gt))
        coco = COCO(str(gt_path))
        evaluator = COCOeval(coco, coco.loadRes(kept), "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    recall = evaluator.stats[8]  # average recall, IoU 0.50:0.95, up to 100 boxes
    assert recall == 0.5, f"average recall={recall}, expected 0.5"
    return "NMS drops the misleading-label box; recall 0.5 with 1 of 2 targets found"


def check_instructpart():
    from robovlmbench.benchmarks.instructpart import points_on_mask

    mask = np.zeros((80, 100), dtype=np.uint8)  # 80 rows, 100 columns
    mask[10:30, 60:90] = 1                      # part in the upper right
    points = [
        (0.75, 0.25),  # on the part
        (0.2, 0.7),    # off the part; would wrongly hit if x and y were swapped
        (1.2, 0.2),    # outside the image
    ]
    hits = points_on_mask(points, mask)
    assert hits == [True, False, False], f"got {hits}, expected [True, False, False]"
    return f"point-on-mask scorer correct on {len(points)} cases (incl. x/y order, outside image)"


def check_capx(suite):
    """A scripted CaP-X episode in the suite's own env (robovlmbench/benchmarks/capx/smoke.py): headless
    rendering, then CaP-X's oracle code, which must get the task done."""
    gpu = os.environ.get("RVB_GPU", "cpu")
    if gpu == "cpu":
        raise Skip("no NVIDIA GPU on this node")
    if not (REPO_ROOT / ".venvs" / f"capx-{suite}-{os.uname().machine}").is_dir():
        raise Skip(f"no CaP-X {suite} env for this hardware (bash scripts/setup_capx.sh)")
    from robovlmbench.benchmarks.capx import record_support

    smoke = REPO_ROOT / "robovlmbench" / "benchmarks" / "capx" / "smoke.py"
    command = ["bash", "-c", 'source scripts/capx_env.sh "$1" >/dev/null && cd "$CAPX_DIR" && exec python "$2" "$1"',
               "capx", suite, str(smoke)]
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1200,
                                env={**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})
        ok, error = result.returncode == 0, (result.stderr.strip().splitlines() or ["no output"])[-1]
    except subprocess.TimeoutExpired:
        result, ok, error = None, False, "the scripted episode took over 20 minutes"
    if result is not None and result.returncode == 3:  # CaP-X's ports are taken: says nothing about the hardware
        raise Skip(error)
    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.splitlines()[0]
    record_support(gpu, gpu_name, suite, ok, "scripted episode passed" if ok else f"CaP-X check failed: {error}")
    assert ok, error
    r = json.loads(result.stdout.strip().splitlines()[-1])
    return (f"oracle episode done (reward {r['reward']:g}), headless render {r['image'][1]}x{r['image'][0]}, "
            f"{r['seconds']} s")


CHECKS = [
    ("python", check_python),
    ("third_party", check_third_party),
    ("packages", check_packages),
    ("GPU + vLLM", check_gpu),
    ("TensorRT-LLM", check_trtllm),
    ("RoboBench", check_robobench),
    ("EPIC-Bench", check_epic_bench),
    ("RoboSpatial-Home", check_robospatial),
    ("OVDEval", check_ovdeval),
    ("InstructPart", check_instructpart),
    ("CaP-X Robosuite", lambda: check_capx("robosuite")),
    ("CaP-X LIBERO", lambda: check_capx("libero")),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="+", metavar="NAME", help="run only the checks whose name contains one of these")
    args = parser.parse_args()

    failed = 0
    for name, check in CHECKS:
        if args.only and not any(o.lower() in name.lower() for o in args.only):
            continue
        try:
            print(f"[PASS] {name:<17} {check()}")
        except Skip as e:
            print(f"[SKIP] {name:<17} {e}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name:<17} {type(e).__name__}: {e}")
    print(f"\n{'All checks passed.' if not failed else f'{failed} check(s) failed.'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
