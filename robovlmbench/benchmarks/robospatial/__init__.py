"""RoboSpatial-Home: 350 spatial questions on home RGB-D photos, in three categories:
configuration (yes/no relations), context (point to free space), compatibility (yes/no "does it fit").

Prompts and scoring follow the official evaluation (third_party/RoboSpatial-Eval): the question is
sent unchanged, and answers are scored by its evaluate_answer(): yes/no by the first word, points by
whether one of the first two lands in the free-space mask. One exception: models with their own point
format (robovlmbench/formats.py) are asked in that format, and their points converted back to 0-1.
"""

import importlib.util
import sys
import tempfile
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

import pyarrow.parquet as pq

from robovlmbench import DATA_DIR, THIRD_PARTY_DIR
from robovlmbench.formats import PointFormat
from robovlmbench.models.client import Request, Response

CATEGORIES = ("configuration", "context", "compatibility")
# Every context question ends with the official point-format instruction, starting here.
_FORMAT_INSTRUCTION = " Your answer should be formatted as a list of tuples"
_MODULE_NAME = "robospatial_official_evaluation"


def official():
    """Return the official `evaluation` module.

    Loaded by file path because that repo's top-level module names (evaluation, main, models)
    are too generic to put on sys.path.
    """
    if _MODULE_NAME not in sys.modules:
        path = THIRD_PARTY_DIR / "RoboSpatial-Eval" / "evaluation.py"
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
    return sys.modules[_MODULE_NAME]


@dataclass
class Sample:
    id: str             # "<category>-<row>"
    category: str
    question: str
    answer: str         # ground truth: "Yes"/"No", or example points for context questions
    image: bytes        # PNG
    mask: bytes | None  # free-space mask (context questions only), PNG


def load_samples() -> list[Sample]:
    """All 350 questions, interleaved across categories so any prefix covers all three."""
    per_category = []
    for category in CATEGORIES:
        (path,) = (DATA_DIR / "robospatial" / "data").glob(f"{category}-*.parquet")
        rows = pq.read_table(path, columns=["question", "answer", "img", "mask"]).to_pylist()
        per_category.append([
            Sample(f"{category}-{i}", category, r["question"], r["answer"], r["img"]["bytes"],
                   (r["mask"] or {}).get("bytes"))
            for i, r in enumerate(rows)
        ])
    return [s for group in zip_longest(*per_category) for s in group if s is not None]


def build_request(sample: Sample, point_format: PointFormat) -> Request:
    text = sample.question
    if sample.category == "context" and point_format.instruction:
        question, found, _ = text.partition(_FORMAT_INSTRUCTION)
        assert found, f"{sample.id}: official format instruction not found"
        text = f"{question} {point_format.instruction}"
    return Request(sample.id, text, [sample.image])


def _to_official_points(answer: str, point_format: PointFormat) -> str:
    """A model-format pointing answer, rewritten as the official 0-1 tuples ("" if no points)."""
    points = point_format.parse(answer)
    return "[" + ", ".join(f"({x:.4f}, {y:.4f})" for x, y in points) + "]" if points else ""


def score(samples: list[Sample], responses: list[Response], point_format: PointFormat) -> dict:
    """Official accuracy per category and the leaderboard's averages, plus why answers failed.

    Leaderboard: vqa_avg = mean(configuration, compatibility); total = mean of all three categories.
    Per sample: `said_yes` for yes/no questions; for pointing, `points`, the (up to two) 0-1 points scored.
    """
    evaluate_answer = official().evaluate_answer
    per_sample = []
    with tempfile.TemporaryDirectory() as tmp:  # the official scorer reads masks from files
        for sample, response in zip(samples, responses, strict=True):
            mask_path = None
            if sample.mask:
                mask_path = str(Path(tmp) / f"{sample.id}.png")
                Path(mask_path).write_bytes(sample.mask)
            answer = response.answer
            if sample.category == "context" and point_format.parse:
                answer = _to_official_points(answer, point_format)
            correct, is_binary, parsed, parsable = evaluate_answer(sample.answer, answer, mask_path=mask_path)
            row = {
                "id": sample.id, "category": sample.category, "correct": bool(correct),
                "outcome": ("failed" if response.error else "truncated" if response.finish_reason == "length"
                            else "unparsable" if not parsable else "answered"),
            }
            if is_binary:
                row["said_yes"] = parsed.startswith("yes")
            else:  # one point comes back as a tuple, several as a list of tuples
                row["points"] = [] if parsed is None else [list(parsed)] if isinstance(parsed, tuple) else [
                    list(p) for p in parsed]
            per_sample.append(row)

    def accuracy(rows):
        return round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None

    def mean(*values):
        return round(sum(values) / len(values), 4) if None not in values else None

    by_category = {c: [r for r in per_sample if r["category"] == c] for c in CATEGORIES}
    acc = {c: accuracy(rows) for c, rows in by_category.items()}
    outcomes = [r["outcome"] for r in per_sample]
    return {
        "by_category": {c: {"accuracy": acc[c], "n": len(rows)} for c, rows in by_category.items() if rows},
        "vqa_avg": mean(acc["configuration"], acc["compatibility"]),
        "total": mean(*acc.values()),
        "n": len(per_sample),
        "outcomes": {o: outcomes.count(o) for o in ("answered", "unparsable", "truncated", "failed")},
        "per_sample": per_sample,
    }
