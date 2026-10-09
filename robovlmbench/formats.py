"""How models write image coordinates. Benchmarks ask each model in its own format and convert its
answers to 0-1 coordinates (fractions of image width and height) before scoring.

A model config's `point_format` names both its point and its box format:
    normalized  keep the benchmark's own format instructions; answers are already in its format
    qwen        Qwen-VL native: JSON with "point_2d": [x, y] / "bbox_2d": [x1, y1, x2, y2], on a 0-1000 scale
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True)
class PointFormat:
    name: str
    instruction: str | None  # replaces the benchmark's format instruction; None keeps it
    parse: Callable[[str], list[Point]] | None  # answer -> 0-1 points; None leaves parsing to the benchmark


_QWEN_POINT = re.compile(r'"point_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')


def _parse_qwen_points(answer: str) -> list[Point]:
    return [(float(x) / 1000, float(y) / 1000) for x, y in _QWEN_POINT.findall(answer)]


POINT_FORMATS = {
    "normalized": PointFormat("normalized", None, None),
    "qwen": PointFormat(
        "qwen",
        'Answer in JSON format: [{"point_2d": [x, y]}, ...], with x and y as integers from 0 to 1000.',
        _parse_qwen_points,
    ),
}


Box = tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass(frozen=True)
class BoxFormat:
    name: str
    instruction: str | None  # replaces the benchmark's format instruction; None keeps it
    parse: Callable[[str], list[Box]] | None  # answer -> 0-1 boxes; None leaves parsing to the benchmark


_QWEN_BOX = re.compile(r'"bbox_2d"\s*:\s*\[' + r"\s*,\s*".join([r"(-?\d+(?:\.\d+)?)"] * 4) + r"\s*\]")


def _parse_qwen_boxes(answer: str) -> list[Box]:
    return [tuple(float(v) / 1000 for v in box) for box in _QWEN_BOX.findall(answer)]


BOX_FORMATS = {
    "normalized": BoxFormat("normalized", None, None),
    "qwen": BoxFormat(
        "qwen",
        'Answer in JSON format: [{"bbox_2d": [x1, y1, x2, y2]}, ...], with coordinates as integers from 0 to 1000.',
        _parse_qwen_boxes,
    ),
}
