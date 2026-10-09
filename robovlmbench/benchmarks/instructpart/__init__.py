"""InstructPart: find the part of an object to act on, in 600 photos, one target part each.

No official evaluation code is released, so the protocol lives here. Every photo is asked twice:
    task   the task instruction, which doesn't name the part ("If I want to open the drawer, which
           part in the picture should I interact with?"), the dataset's first instruction
    part   the part named directly ("handle of the cabinet"), the dataset's fifth instruction
The model answers with points; an answer is correct if its first point lands on the part's mask.
Models with their own point format (robovlmbench/formats.py) are asked in that format.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from robovlmbench import DATA_DIR
from robovlmbench.formats import PointFormat
from robovlmbench.models.client import Request, Response

VARIANTS = ("task", "part")
_TEST_DIR = DATA_DIR / "instructpart" / "test"
_NORMALIZED_INSTRUCTION = ("Answer with one point as (x, y), where x and y are fractions of the image width and "
                           "height, between 0 and 1.")
_POINT = re.compile(r"[\(\[]\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*[\)\]]")


@dataclass
class Sample:
    id: str            # "<image>-<object>-<part>/<variant>"
    variant: str       # "task" or "part"
    object: str
    part: str
    instruction: str
    image_path: Path
    mask_path: Path

    @property
    def image(self) -> bytes:
        return self.image_path.read_bytes()


def load_samples() -> list[Sample]:
    """All 600 photos, each as a task sample followed by a part sample."""
    samples = []
    for entry in json.loads((_TEST_DIR / "data_test.json").read_text()):
        (part,) = entry["part_list"]
        # file naming from InstructPart's README: <image>-<object>-<part>, spaces -> underscores
        stem = f"{Path(entry['image_path']).stem}-{part['object'].replace(' ', '_')}-{part['part'].replace(' ', '_')}"
        for variant, instruction in zip(VARIANTS, (part["instruction"][0], part["instruction"][4])):
            samples.append(Sample(f"{stem}/{variant}", variant, part["object"], part["part"], instruction,
                                  _TEST_DIR / "images" / f"{stem}.jpg", _TEST_DIR / "masks" / f"{stem}.png"))
    return samples


def build_request(sample: Sample, point_format: PointFormat) -> Request:
    if sample.variant == "task":
        question = f"{sample.instruction} Point to that part in the image."
    else:
        question = f"Point to the {sample.instruction} in the image."
    return Request(sample.id, f"{question} {point_format.instruction or _NORMALIZED_INSTRUCTION}", [sample.image])


def parse_points(answer: str, point_format: PointFormat) -> list[tuple[float, float]]:
    """The answer's points as (x, y) fractions of the image width and height."""
    if point_format.parse:
        return point_format.parse(answer)
    return [(float(x), float(y)) for x, y in _POINT.findall(answer)]


def points_on_mask(points, mask: np.ndarray) -> list[bool]:
    """For each (x, y) point, normalized to [0, 1], return whether it lands on the mask.

    `mask` is a 2-D array indexed [row, col]; nonzero pixels are the part.
    Points outside the image count as misses.
    """
    h, w = mask.shape
    hits = []
    for x, y in points:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            hits.append(False)
            continue
        col = min(int(x * w), w - 1)
        row = min(int(y * h), h - 1)
        hits.append(bool(mask[row, col]))
    return hits


def score(samples: list[Sample], responses: list[Response], point_format: PointFormat) -> dict:
    """Accuracy (first point on the part), overall and per variant, plus why answers failed.
    Per sample: `points`, all the answer's points (0-1)."""
    per_sample = []
    for sample, response in zip(samples, responses, strict=True):
        points = parse_points(response.answer, point_format)
        mask = np.asarray(Image.open(sample.mask_path).convert("L")) > 0
        per_sample.append({
            "id": sample.id, "variant": sample.variant, "points": [list(p) for p in points],
            "correct": bool(points) and points_on_mask(points[:1], mask)[0],
            "outcome": ("failed" if response.error else "truncated" if response.finish_reason == "length"
                        else "unparsable" if not points else "answered"),
        })

    def accuracy(rows):
        return round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None

    outcomes = [r["outcome"] for r in per_sample]
    return {
        "accuracy": accuracy(per_sample),
        "by_variant": {v: {"accuracy": accuracy(rows), "n": len(rows)}
                       for v in VARIANTS if (rows := [r for r in per_sample if r["variant"] == v])},
        "n": len(per_sample),
        "outcomes": {o: outcomes.count(o) for o in ("answered", "unparsable", "truncated", "failed")},
        "per_sample": per_sample,
    }
