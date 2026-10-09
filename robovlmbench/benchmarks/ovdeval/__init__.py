"""OVDEval: open-vocabulary detection with misleading descriptions. 15,142 photos in 8 subsets; the model
boxes the objects that match a list of descriptions, most of which describe nothing in the photo.

    color, material, position,    the photo's own descriptions: the true ones ("text") and hard negatives
    relationship, negation        that differ in one detail ("neg_text"), e.g. "red fruit" vs "white fruit"
    celebrity, logo, landmark     every name in the subset (10, 9, 9), e.g. "elon musk", "bill gates", ...

Queries follow the official GLIP evaluation (third_party/OVDEval/eval_glip.py): one request per photo with
all of its descriptions, so 15,142 requests. The descriptions are sorted (the official list puts the true ones
first) and the model labels each box with the description it matches. Models with their own box format
(robovlmbench/formats.py) are asked in that format; their boxes are converted to 0-1, then to pixels.

Scoring is the official NMS-AP (third_party/OVDEval/eval.py): per photo, of the boxes overlapping a
ground-truth box (IoU > 0.5) whatever their description, only the most confident is kept, so only one label
per object counts; then COCO evaluation per description, as the official script.
Boxes from a VLM carry no confidence, so:
    - NMS keeps the box written first among overlapping ones (the official code keeps the highest score);
    - COCO evaluation gets the same score, 1, for every box. Average recall (AR, IoU 0.50:0.95, up to 100
      boxes) doesn't depend on scores and is the headline. AP is reported for comparison with published
      NMS-AP but ranks tied boxes in photo order, so it isn't a detector's AP; precision50, the share of
      the model's boxes (all of them, before NMS) that match a ground-truth box with their description at
      IoU 0.5, is the order-free precision, and the one that shows extra labels on the same object.
The overall score is the mean over the 8 subsets.

Even a perfect answer scores a little below 1 (CEILING_AR), as in the official evaluation: where two
ground-truth boxes of different descriptions overlap (IoU > 0.5; 6% of boxes in relationship), NMS keeps one
box for both. Also kept as released: every subset has one ground-truth box with id 0, which pycocotools never
counts as found (it marks matches by box id, 0 meaning unmatched) and whose match counts as a false positive.
The published numbers carry this, so ours do too; it costs at most 0.0005 AR per subset.
Photos are sent upright: the 8 with an EXIF rotation are rotated here, as their boxes were drawn on the
rotated photo (even where the annotation's width and height are the unrotated ones).
"""

import contextlib
import functools
import importlib.util
import io
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from robovlmbench import DATA_DIR, THIRD_PARTY_DIR
from robovlmbench.formats import BOX_FORMATS, PointFormat
from robovlmbench.models.client import Request, Response

SUBSETS = ("color", "material", "position", "relationship", "negation", "celebrity", "logo", "landmark")
NAME_SUBSETS = ("celebrity", "logo", "landmark")  # queried with every name in the subset (eval_glip.py)
IOU_THRESHOLD = 0.5  # eval.py's NMS default
_DIR = DATA_DIR / "ovdeval"
_MODULE_NAME = "ovdeval_official_eval"

_PROMPT = ("Find everything in the image that matches one of these descriptions:\n{descriptions}\n"
           "Some descriptions may match nothing in the image. Box each object that matches a description, "
           "and label the box with the one description it matches, copied exactly. {format}")
# The box format's own instruction, with a label added. Keyed by the model config's point_format.
_FORMAT_INSTRUCTIONS = {
    "qwen": ('Answer in JSON format: [{"bbox_2d": [x1, y1, x2, y2], "label": "<description>"}, ...], '
             "with coordinates as integers from 0 to 1000. Answer [] if nothing matches."),
    "normalized": ('Answer in JSON format: [{"bbox": [x1, y1, x2, y2], "label": "<description>"}, ...], '
                   "where (x1, y1) is the top-left and (x2, y2) the bottom-right corner, as fractions of the "
                   "image width and height between 0 and 1. Answer [] if nothing matches."),
}
_OBJECT = re.compile(r"\{[^{}]*\}")
_LABEL = re.compile(r'"label"\s*:\s*"((?:[^"\\]|\\.)*)"')
_NUM = r"\s*(-?\d+(?:\.\d+)?)\s*"
_NORMALIZED_BOX = re.compile(r'"bbox"\s*:\s*\[' + ",".join([_NUM] * 4) + r"\]")
_EMPTY = re.compile(r"\[\s*\]")
_ORIENTATION = 0x0112  # EXIF tag

# Best AR possible per subset: ceiling_ar(), the ground-truth boxes scored as the answer, on all photos.
CEILING_AR = {"color": 0.9954, "material": 0.9961, "position": 0.9984, "relationship": 0.9685, "negation": 0.9943,
              "celebrity": 0.9996, "logo": 0.9997, "landmark": 0.9996}


def official():
    """Return the official `eval` module.

    Loaded by file path because a top-level module named `eval` would shadow the builtin.
    """
    if _MODULE_NAME not in sys.modules:
        path = THIRD_PARTY_DIR / "OVDEval" / "eval.py"
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
    return sys.modules[_MODULE_NAME]


@dataclass
class Sample:
    id: str                       # "<subset>/<image id>"
    subset: str
    image_id: int                 # id in annotations/<subset>.json
    image_path: Path
    descriptions: tuple[str, ...]  # what the model is asked to find, sorted

    @property
    def image(self) -> bytes:
        """The photo as sent: the file unchanged, or rotated upright if it has an EXIF rotation."""
        data = self.image_path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            if image.getexif().get(_ORIENTATION, 1) in (0, 1):
                return data
            buffer = io.BytesIO()
            ImageOps.exif_transpose(image).convert("RGB").save(buffer, "JPEG", quality=95)
            return buffer.getvalue()

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the photo as sent, the frame its ground-truth boxes are in."""
        return _upright_size(self.image_path)


@functools.cache
def annotations(subset: str) -> dict:
    """annotations/<subset>.json, as released."""
    return json.loads((_DIR / "annotations" / f"{subset}.json").read_text())


def load_samples() -> list[Sample]:
    """One sample per photo, all 15,142, interleaved across subsets so any prefix covers all eight."""
    per_subset = []
    for subset in SUBSETS:
        gt = annotations(subset)
        all_names = tuple(sorted({c["name"] for c in gt["categories"]}))
        per_subset.append([
            Sample(f"{subset}/{im['id']}", subset, im["id"], _DIR / "images" / subset / im["file_name"],
                   all_names if subset in NAME_SUBSETS else tuple(sorted(set(im["text"] + im["neg_text"]))))
            for im in sorted(gt["images"], key=lambda im: im["id"])
        ])
    return [s for group in zip_longest(*per_subset) for s in group if s is not None]


def build_request(sample: Sample, point_format: PointFormat) -> Request:
    descriptions = "\n".join(f"- {' '.join(d.split())}" for d in sample.descriptions)  # e.g. "barack  obama"
    text = _PROMPT.format(descriptions=descriptions, format=_FORMAT_INSTRUCTIONS[point_format.name])
    return Request(sample.id, text, [sample.image])


def parse_boxes(answer: str, point_format: PointFormat) -> list[tuple[list[float], str | None]] | None:
    """The answer's boxes as ([x1, y1, x2, y2] in 0-1, label or None), in the order written; None if the
    answer has no box and no empty list. Read object by object, so a malformed or unfinished JSON list
    still yields its complete objects."""
    parse = BOX_FORMATS[point_format.name].parse or _parse_normalized_box
    boxes = []
    for obj in _OBJECT.findall(answer):
        found, label = parse(obj), _LABEL.search(obj)
        if len(found) == 1:
            try:
                label = json.loads(f'"{label.group(1)}"') if label else None
            except json.JSONDecodeError:
                label = None
            boxes.append((list(found[0]), label))
    return boxes if boxes or _EMPTY.search(answer) else None


def score(samples: list[Sample], responses: list[Response], point_format: PointFormat) -> dict:
    """Official NMS-AP evaluation per subset (AR headline, AP with constant scores, precision50),
    their means over the 8 subsets, and why answers failed.

    Per sample: `boxes`, the boxes kept for scoring ([x1, y1, x2, y2] in 0-1 and the description);
    `dropped`, boxes not scored: unlabelled, labelled with something other than the descriptions asked
    about, or empty once clipped to the image;
    `n_gt`, ground-truth boxes; `found`, those matched at IoU 0.5 with the right description.
    """
    per_sample, by_subset = [], {}
    for subset in SUBSETS:
        pairs = [(s, r) for s, r in zip(samples, responses, strict=True) if s.subset == subset]
        if not pairs:
            continue
        gt = annotations(subset)
        category_id = {c["name"]: c["id"] for c in gt["categories"]}  # last id wins, as in eval_glip.py
        n_gt = Counter(a["image_id"] for a in gt["annotations"])
        rows, predictions = [], []
        for sample, response in pairs:
            parsed = parse_boxes(response.answer, point_format)
            asked = {_normalize(d): d for d in sample.descriptions}
            kept = [(_clip(box), asked.get(_normalize(label or ""))) for box, label in parsed or []]
            kept = [(box, label) for box, label in kept if box and label]
            width, height = sample.size
            for rank, (box, label) in enumerate(kept):
                x1, y1, x2, y2 = box
                predictions.append({
                    "image_id": sample.image_id, "category_id": category_id.get(label),
                    "bbox": [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height],
                    "score": 1.0 - rank / (len(kept) + 1),  # NMS tie-break only: the first written wins
                })
            rows.append({
                "id": sample.id, "subset": subset, "boxes": [[*box, label] for box, label in kept],
                "dropped": len(parsed or []) - len(kept), "n_gt": n_gt.get(sample.image_id, 0),
                "outcome": ("failed" if response.error else "truncated" if response.finish_reason == "length"
                            else "unparsable" if parsed is None else "answered"),
            })
        metrics, found = evaluate(gt, [s.image_id for s, _ in pairs], predictions)
        n_boxes = sum(len(r["boxes"]) + r["dropped"] for r in rows)
        for row, (sample, _) in zip(rows, pairs):
            row["found"] = found.get(sample.image_id, 0)
        by_subset[subset] = {**metrics, "precision50": _round(sum(found.values()) / n_boxes) if n_boxes else None,
                             "n": len(rows)}
        per_sample += rows

    def mean(key):
        values = [by_subset[s][key] for s in SUBSETS if s in by_subset]
        return _round(sum(values) / len(values)) if len(values) == len(SUBSETS) and None not in values else None

    order = {s.id: i for i, s in enumerate(samples)}
    per_sample.sort(key=lambda r: order[r["id"]])
    outcomes = [r["outcome"] for r in per_sample]
    return {
        "ar": mean("ar"),
        "ap": mean("ap"),
        "precision50": mean("precision50"),
        "by_subset": by_subset,
        "n": len(per_sample),
        "outcomes": {o: outcomes.count(o) for o in ("answered", "unparsable", "truncated", "failed")},
        "per_sample": per_sample,
    }


def evaluate(gt: dict, image_ids: list[int], predictions: list[dict]) -> tuple[dict, dict[int, int]]:
    """eval.py on `predictions` (official result format, pixel boxes; category_id None for a description
    that isn't a category), restricted to `image_ids`: NMS against the ground truth, then COCO evaluation.

    Returns ({ar, ap, ap50}, {image id: ground-truth boxes matched at IoU 0.5}).
    COCO's AR and AP are those of COCOeval.summarize() (IoU 0.50:0.95, all areas, 100 boxes); every
    box is scored 1 after NMS. Area ranges other than "all" are skipped: they don't change these numbers.
    """
    nms = official().nms_threaded
    gt_per_image, predictions_per_image = {}, {}
    for a in gt["annotations"]:
        gt_per_image.setdefault(a["image_id"], []).append(a)
    for p in predictions:
        predictions_per_image.setdefault(p["image_id"], []).append(p)
    after_nms = []
    for image_id, preds in predictions_per_image.items():
        after_nms += nms(gt_per_image.get(image_id, []), preds, IOU_THRESHOLD, 1)
    results = [{**p, "score": 1.0} for p in after_nms if p["category_id"] is not None]
    if not results:  # pycocotools can't load an empty result list; nothing found
        return {"ar": 0.0, "ap": 0.0, "ap50": 0.0}, {}

    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints progress
        coco = COCO()
        coco.dataset = {**gt, "annotations": [dict(a) for a in gt["annotations"]]}  # COCOeval adds keys
        coco.createIndex()
        E = COCOeval(coco, coco.loadRes(results), "bbox")
        image_ids = set(image_ids)
        E.params.imgIds = sorted(image_ids)
        # Only descriptions with a ground-truth box here: the others score -1, left out of every mean, and
        # their boxes match nothing, so skipping them changes no number but saves most of the time.
        E.params.catIds = sorted({a["category_id"] for a in gt["annotations"] if a["image_id"] in image_ids})
        E.params.areaRng, E.params.areaRngLbl = [E.params.areaRng[0]], ["all"]
        E.evaluate()
        E.accumulate()
    precision = E.eval["precision"][:, :, :, 0, -1]  # [IoU, recall, category], all areas, 100 boxes
    recall = E.eval["recall"][:, :, 0, -1]           # [IoU, category]
    iou50 = int(np.argwhere(np.isclose(E.params.iouThrs, 0.5))[0, 0])

    found = {}
    for e in E.evalImgs:  # dtMatches holds the matched ground-truth id, 0 if none (so never for id 0)
        if e is not None:
            found[e["image_id"]] = found.get(e["image_id"], 0) + int(
                np.sum((e["dtMatches"][iou50] > 0) & ~e["dtIgnore"][iou50]))
    return {"ar": _mean_valid(recall), "ap": _mean_valid(precision), "ap50": _mean_valid(precision[iou50])}, found


def ceiling_ar(subset: str) -> float:
    """AR when the answer is the ground-truth boxes, in annotation order, over all of `subset`'s photos."""
    gt = annotations(subset)
    predictions, rank = [], {}
    for a in gt["annotations"]:
        rank[a["image_id"]] = rank.get(a["image_id"], -1) + 1
        predictions.append({"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"],
                            "score": 1.0 - rank[a["image_id"]] / 1000})  # first written wins NMS, as in score()
    return evaluate(gt, [im["id"] for im in gt["images"]], predictions)[0]["ar"]


def _parse_normalized_box(text: str) -> list[tuple[float, float, float, float]]:
    return [tuple(float(v) for v in box) for box in _NORMALIZED_BOX.findall(text)]


def _clip(box) -> list[float] | None:
    """Corners in order and inside the image; None if nothing is left."""
    x1, x2 = sorted(min(max(v, 0.0), 1.0) for v in (box[0], box[2]))
    y1, y2 = sorted(min(max(v, 0.0), 1.0) for v in (box[1], box[3]))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _normalize(label: str) -> str:
    return " ".join(label.lower().split())


@functools.cache
def _upright_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        rotated = image.getexif().get(_ORIENTATION, 1) in (5, 6, 7, 8)
        return image.size[::-1] if rotated else image.size


def _mean_valid(values: np.ndarray) -> float:
    """Mean over the entries COCOeval filled in (-1 marks a description with no ground truth)."""
    valid = values[values > -1]
    return _round(float(valid.mean())) if valid.size else 0.0


def _round(value: float) -> float:
    return round(value, 4)
