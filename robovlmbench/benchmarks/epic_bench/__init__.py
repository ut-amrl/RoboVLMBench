"""EPIC-Bench: fine-grained visual grounding for robots, 6661 questions in 23 task types (3 categories).

    TargetLocalization  find the objects matching a description (14 types by attribute: color, part, spatial...)
    Navigation          ground regions; a path to a target area (ego view / third-person view);
                        the same area seen from another view (visual matching)
    Manipulation        the part to act on (affordance); objects touching a reference object (contact,
                        3 types); a placement region and whether the object fits there (placement)

Prompts and scoring follow the official evaluation (third_party/EPIC-Bench-Eval), with its own code:
    prompts   tools/data_gen/converters + prompts, with the system/response prompts its
              scripts/build_swift_data.sh picks per task
    parsing   tools/formatting/format_response.py, raw answer -> its standard format
    scoring   epic_eval.evaluate.run_evaluation: per question, IoU between the union of the answer's boxes
              and the union of the ground-truth masks, plus per task a count term, a yes/no term
              (placement), or a path score instead (FeasiblePath); averaged over all questions.

The prompt is the official one for every model, with the official system message and each image where
the official text has an <image> marker. It names no coordinate range; answers are read on the model's
own scale as the official batch scorer does per model family (Qwen: 0-1000, "normalized_0_1000"), and
otherwise inferred from the numbers as the official default ("auto") does. Answers written as Qwen's
{"bbox_2d": [...]} / {"point_2d": [...]} objects are unwrapped to the plain lists the official formatter reads.
"""

import importlib
import importlib.util
import inspect
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

import numpy as np
from PIL import Image

from robovlmbench import DATA_DIR, THIRD_PARTY_DIR
from robovlmbench.formats import BOX_FORMATS, PointFormat
from robovlmbench.models.client import Request, Response

_OFFICIAL_DIR = THIRD_PARTY_DIR / "EPIC-Bench-Eval"
_DATA_DIR = DATA_DIR / "epic_bench"

# Per task: official converter, prompts module, and the system / response-format prompts that
# scripts/build_swift_data.sh uses.
_PROMPTS = {
    "TargetLocalization": ("TargetLocalization_convert.py", "TargetLocalization", "system_prompt_base",
                           "response_json"),
    "GroundDetection": ("GroundDetection_convert.py", "GroundDetection", "system_prompt_base", "response_json"),
    "FeasiblePath_Ego": ("FeasiblePath_convert.py", "FeasiblePath", "system_prompt_first_perspective",
                         "response_json_1"),
    "FeasiblePath_Exo": ("FeasiblePath_convert.py", "FeasiblePath", "system_prompt_third_perspective",
                         "response_json_3"),
    "VisualMatching": ("VisualMatching_convert.py", "VisualMatching", "system_prompt_base", "response_json"),
    "AffordanceRegion": ("AffordanceRegion_convert.py", "AffordanceRegion", "system_prompt_base", "response_json"),
    "ContactRelationship_TypeOne": ("ContactRelationship_convert_1.py", "ContactRelationship",
                                    "system_prompt_base_1", "response_json_1"),
    "ContactRelationship_TypeTwo": ("ContactRelationship_convert_2_3.py", "ContactRelationship",
                                    "system_prompt_base_2", "response_json_2"),
    "ContactRelationship_TypeThree": ("ContactRelationship_convert_2_3.py", "ContactRelationship",
                                      "system_prompt_base_3", "response_json_3"),
    "PlacementRegion": ("PlacementRegion_convert.py", "PlacementRegion", "system_prompt_base", "response_json"),
}
PATH_TASKS = ("FeasiblePath_Ego", "FeasiblePath_Exo")  # answered with points; all others with boxes

_FLAT_OBJECT = re.compile(r"\{[^{}]*\}")  # a JSON object with no object inside, e.g. {"bbox_2d": [...]}
OUTCOMES = ("answered", "unparsable", "truncated", "failed")
# Coordinate scale per point format, as the official batch scorer sets it per model family.
NATIVE_SCALE = {"qwen": "normalized_0_1000"}


def official():
    """Return the official `epic_eval` package."""
    path = str(_OFFICIAL_DIR)
    if path not in sys.path:
        sys.path.append(path)
    return importlib.import_module("epic_eval")


def _load_by_path(name: str, path: Path, extra_path: Path | None = None):
    """Load a module from a file of the official repo under a name of our own (cached).

    The converters import the official prompts as the top-level package `prompts`, so `extra_path`
    (tools/data_gen) is on sys.path only while they load.
    """
    if name not in sys.modules:
        if extra_path:
            sys.path.insert(0, str(extra_path))
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
        finally:
            if extra_path:
                sys.path.remove(str(extra_path))
    return sys.modules[name]


def _converter(filename: str):
    data_gen = _OFFICIAL_DIR / "tools" / "data_gen"
    return _load_by_path(f"epic_bench_{Path(filename).stem}", data_gen / "converters" / filename, data_gen)


def _prompt(module: str, variable: str) -> str:
    _converter(_PROMPTS["TargetLocalization"][0])  # makes the official `prompts` package importable
    return getattr(importlib.import_module(f"prompts.{module}"), variable)


def formatter():
    """The official response formatter, tools/formatting/format_response.py."""
    return _load_by_path("epic_bench_format_response", _OFFICIAL_DIR / "tools" / "formatting" / "format_response.py")


@dataclass
class Sample:
    id: str             # annotation file, relative to the data dir, without "_mask_rle.json"
    category: str       # TargetLocalization / Navigation / Manipulation
    group: str          # e.g. "BasicAttributes"; the task itself for most Navigation/Manipulation tasks
    task: str           # one of the 23 task types, e.g. "Color", "FeasiblePath_Ego"
    system: str         # official system message
    user: str           # official user message, <image> placeholders included
    image_paths: list[Path]
    gt_path: Path       # annotation JSON: masks, counts, feasibility
    size: tuple[int, int]  # (height, width) of the image answers locate things in: the first, or
                           # for VisualMatching the third (the same size as the ground-truth masks)

    @property
    def images(self) -> list[bytes]:
        return [p.read_bytes() for p in self.image_paths]

    @property
    def type_key(self) -> str:
        """The official by_type key, e.g. "TargetLocalization/BasicAttributes/Color"."""
        return official().config.LEAF_TO_INFO[self.task][2]


def _convert(task: str, item: dict) -> dict:
    """The official converter's {"messages": [system, user], "images": [...]} for one annotation."""
    filename, module, system_var, response_var = _PROMPTS[task]
    convert = _converter(filename).one_sample_to_swift
    args = dict(item=item, system_prompt=_prompt(module, system_var), response_format=_prompt(module, response_var),
                use_system_prompt=True, check_files_exist=False, response_format_in_user=False,
                response_format_in_system=False)
    return convert(**{k: v for k, v in args.items() if k in inspect.signature(convert).parameters})


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.height, image.width


def load_samples() -> list[Sample]:
    """All 6661 questions, interleaved across the 23 task types so any prefix covers all of them."""
    leaf_tasks = official().config.LEAF_TASKS  # (category, group, task, by_type key, by_group key)
    get_gt_json_path = formatter().get_gt_json_path
    per_task = []
    for category, group, task, type_key, _ in leaf_tasks:
        samples = []
        for path in sorted((_DATA_DIR / type_key).rglob("*.json")):  # the official converters' order
            annotation = json.loads(path.read_text())
            # image paths are saved relative to the dataset's parent dir, as "EPIC_Bench/..."
            annotation["image_path_saved"] = [str(_DATA_DIR / p.removeprefix("EPIC_Bench/"))
                                              for p in annotation["image_path_saved"]]
            swift = _convert("TargetLocalization" if category == "TargetLocalization" else task, annotation)
            system, user = (m["content"] for m in swift["messages"])
            gt_path = Path(get_gt_json_path(swift["images"], task))
            assert gt_path == path, f"{path}: official GT path {gt_path}"
            samples.append(Sample(str(path.relative_to(_DATA_DIR)).removesuffix("_mask_rle.json"), category,
                                  group, task, system, user, [Path(p) for p in swift["images"]], path,
                                  _image_size(Path(swift["images"][-1 if task == "VisualMatching" else 0]))))
        per_task.append(samples)
    return [s for group in zip_longest(*per_task) for s in group if s is not None]


def gt_mask(sample: Sample) -> np.ndarray:
    """The ground-truth mask (H x W, 0/1) answers are scored against: the union of the target masks,
    or for FeasiblePath the feasible path area. All zeros if there is no target."""
    official()
    from epic_eval.utils.gt_loader import GTContext
    from epic_eval.utils.masks import decode_rle_list_to_mask

    gt = GTContext({"task_category": [sample.category], "task_type": [sample.task],
                    "ground_truth": {"gt_json_path": [str(sample.gt_path)]}})
    mask = decode_rle_list_to_mask(gt.path_mask_rle if sample.task in PATH_TASKS else gt.localization_mask_rle)
    return mask if mask is not None and gt.count_gt != 0 else np.zeros(sample.size, np.uint8)


def build_request(sample: Sample, point_format: PointFormat) -> Request:
    # as officially: the system message apart, and each image where the user text has an <image> marker
    assert sample.user.count(Request.IMAGE_MARKER) == len(sample.images), f"{sample.id}: image markers don't match"
    return Request(sample.id, sample.user, sample.images, system=sample.system)


def _to_official(answer: str, parse) -> str:
    """An answer with each box/point written as a flat JSON object (e.g. Qwen's {"bbox_2d": [...]}) replaced
    by a plain list of its 0-1000 coordinates, which the official formatter reads."""
    def plain(match: re.Match) -> str:
        found = parse(match.group(0))
        return json.dumps([round(v * 1000, 3) for v in found[0]]) if len(found) == 1 else match.group(0)
    return _FLAT_OBJECT.sub(plain, answer)


def standard_record(sample: Sample, answer: str, coordinate_system: str) -> dict:
    """One answer in the official standard format (what tools/formatting/format_response.py writes)."""
    fmt = formatter()
    extracted = fmt.extract_formatted_response_by_task(answer, sample.category, sample.task)
    fields = fmt.build_response_fields(extracted, fmt.get_required_response_fields(sample.category, sample.task))
    return {
        "task_category": [sample.category], "task_type": [sample.task],
        "format_type": [fmt.default_format_type_for_task(sample.category, sample.task)],
        "coordinate_system": [coordinate_system], **fields,
        "images": [str(p) for p in sample.image_paths], "ground_truth": {"gt_json_path": [str(sample.gt_path)]},
    }


def official_scores(records: list[dict]) -> dict:
    """The official evaluation (summary and per-question details) of standard-format records.

    The FeasiblePath_Exo score samples mask pixels at random; the seed is fixed so scores repeat.
    """
    official()
    evaluate = importlib.import_module("epic_eval.evaluate")
    state = np.random.get_state()
    np.random.seed(0)
    try:
        with tempfile.TemporaryDirectory() as tmp:  # the official evaluation reads a JSONL file
            path = Path(tmp) / "std_format.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records))
            return evaluate.run_evaluation(str(path))
    finally:
        np.random.set_state(state)


def _to_unit(values: list, mode: str, size: tuple[int, int]) -> list[float]:
    """Official coordinates (x, y alternating) -> 0-1, given the coordinate mode the scorer used."""
    h, w = size
    scale = {"normalized_0_1": (1, 1), "normalized_0_1000": (1000, 1000), "absolute": (w, h)}[mode]
    return [round(v / scale[i % 2], 4) for i, v in enumerate(values)]


def score(samples: list[Sample], responses: list[Response], point_format: PointFormat) -> dict:
    """The official scores, overall (mean over all questions, the official headline), per category,
    per group and per task type, plus why answers failed.

    Per sample: `score` (0-1) and its parts (`localization` IoU, `count`, `path`, `feasible` scores,
    whichever the task uses); `boxes` or `points` as read (0-1); `answered_count`, `answered_feasible`.
    """
    records = []
    for sample, response in zip(samples, responses, strict=True):
        points = sample.task in PATH_TASKS
        parse = (point_format if points else BOX_FORMATS[point_format.name]).parse
        answer = _to_official(response.answer, parse) if parse else response.answer
        records.append(standard_record(sample, answer, NATIVE_SCALE.get(point_format.name, "auto")))
    result = official_scores(records)

    per_sample = []
    unreadable = [formatter().NO_VALID_RESPONSE]
    for sample, response, record, detail in zip(samples, responses, records, result["details"]["by_samples"],
                                                strict=True):
        parts, answer = detail["score_details"], detail["response_details"]
        # Unparsable: the official formatter couldn't read a field the task needs. (The official
        # invalid_response flag misses this for box-only tasks, where it scores it as "no boxes".)
        unparsable = any(record[f"formatted_response_{f}"] == unreadable
                         for f in ("localization", "count", "path", "binary"))
        row = {"id": sample.id, "category": sample.category, "task": sample.task, "score": round(detail["score"], 4),
               "outcome": ("failed" if response.error else "truncated" if response.finish_reason == "length"
                           else "unparsable" if unparsable else "answered")}
        for part, key in (("localization", "localization"), ("count", "count"), ("path", "path"),
                          ("binary", "feasible")):
            if parts[f"weight_{part}"]:
                row[key] = round(parts[f"{part}_score"], 4)
        mode = answer["coordinate_system"]
        if sample.task in PATH_TASKS:
            found = answer["formatted_response_path"]
            row["points"] = [_to_unit(p, mode, sample.size) for p in found if isinstance(p, list)]
        else:
            found = answer["formatted_response_localization"]
            row["boxes"] = [_to_unit(b, mode, sample.size) for b in found if isinstance(b, list)]
        if "count" in row:
            row["answered_count"] = next((c for c in answer["formatted_response_count"] if isinstance(c, int)), None)
        if "feasible" in row:
            row["answered_feasible"] = next((b for b in answer["formatted_response_binary"] if isinstance(b, bool)),
                                            None)
        per_sample.append(row)

    summary = result["summary"]

    def bucket(b: dict) -> dict:
        return {"score": round(b["mean_score"], 4), "n": b["total_processed"]}

    outcomes = [r["outcome"] for r in per_sample]
    return {
        "score": round(summary["overall"]["average_score"], 4) if per_sample else None,
        "by_category": {k: bucket(b) for k, b in summary["by_category"].items()},
        "by_group": {k: bucket(b) for k, b in summary["by_group"].items()},
        "by_type": {k: bucket(b) for k, b in summary["by_type"].items()},
        "n": len(per_sample),
        "outcomes": {o: outcomes.count(o) for o in OUTCOMES},
        "per_sample": per_sample,
    }
