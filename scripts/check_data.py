"""Check that downloaded benchmark data is complete and at the pinned revisions.

For each benchmark: the download finished at the pinned revision, the sample counts match
what the authors published, and every file the annotations point to exists.
Exits non-zero on failure.

Usage: .venv/bin/python scripts/check_data.py [--bench robobench epic_bench ...]
"""

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from robovlmbench.data import DatasetPin, dataset_pins


def check_robobench(pin: DatasetPin):
    from robobench.data.dataset import RoboBenchDataset

    dataset = RoboBenchDataset(str(pin.dir))
    dimensions = ["instruction_comprehension", "perception_reasoning", "generalized_planning",
                  "affordance_reasoning", "error_analysis"]
    files = questions = images = 0
    for dimension in dimensions:
        for path in sorted((pin.dir / dataset._dimension_dir(dimension)).rglob("questions.json")):
            # load_questions resolves image paths the same way RoboBench's own inference does
            with contextlib.redirect_stdout(io.StringIO()):
                loaded = dataset.load_questions(dimension, path.parent.name)
            paths = [Path(p) for q in loaded for p in q["image_urls"]]
            missing = [p for p in paths if not p.is_file()]
            assert not missing, f"{len(missing)} images missing for {path.parent.name}, e.g. {missing[0]}"
            files, questions, images = files + 1, questions + len(loaded), images + len(paths)
    assert (files, questions) == (32, 6092), f"{files} question files / {questions} questions, expected 32 / 6092"
    return f"{questions} questions in {files} files; all {images} images present"


def check_epic_bench(pin: DatasetPin):
    expected = {"Manipulation": 7061, "Navigation": 10868, "TargetLocalization": 17665}  # from EPIC-Bench's README
    counts = {name: sum(1 for p in (pin.dir / name).rglob("*") if p.is_file()) for name in expected}
    assert counts == expected, f"file counts {counts}, expected {expected}"
    return ", ".join(f"{name} {n}" for name, n in counts.items()) + " files, as released"


def check_robospatial(pin: DatasetPin):
    expected = {"configuration": 123, "context": 122, "compatibility": 105}  # from the dataset card
    counts = {}
    for split in expected:
        (path,) = (pin.dir / "data").glob(f"{split}-*.parquet")
        table = pq.read_table(path, columns=["mask"])
        counts[split] = table.num_rows
        if split == "context":  # pointing questions are scored against this mask
            no_mask = sum(1 for m in table.column("mask").to_pylist() if not (m and m.get("bytes")))
            assert no_mask == 0, f"{no_mask} context questions have no mask"
    assert counts == expected, f"question counts {counts}, expected {expected}"
    return f"{sum(counts.values())} questions ({', '.join(f'{k} {v}' for k, v in counts.items())}); all pointing masks present"


def check_ovdeval(pin: DatasetPin):
    subsets = sorted(p.stem for p in (pin.dir / "annotations").glob("*.json"))
    assert len(subsets) == 8, f"{len(subsets)} subsets, expected the 8 released"
    n_images = n_boxes = 0
    id_zero = []
    for s in subsets:
        gt = json.loads((pin.dir / "annotations" / f"{s}.json").read_text())
        missing = [im["file_name"] for im in gt["images"] if not (pin.dir / "images" / s / im["file_name"]).is_file()]
        assert not missing, f"{s}: {len(missing)} images missing, e.g. {missing[0]}"
        image_ids = {im["id"] for im in gt["images"]}
        assert all(a["image_id"] in image_ids for a in gt["annotations"]), f"{s}: box points to unknown image"
        n_images, n_boxes = n_images + len(gt["images"]), n_boxes + len(gt["annotations"])
        if any(a["id"] == 0 for a in gt["annotations"]):
            id_zero.append(s)
    warning = None
    if id_zero:
        warning = (f"{len(id_zero)}/8 subsets have a box with id 0, which the official COCO scoring "
                   f"never counts as found ({', '.join(id_zero)})")
    return f"{len(subsets)} subsets, {n_images} images, {n_boxes} boxes; all images present", warning


def check_instructpart(pin: DatasetPin):
    test = pin.dir / "test"
    digest = hashlib.sha256((test / "data_test.json").read_bytes()).hexdigest()
    assert f"sha256:{digest}" == pin.revision, "data_test.json differs from the pinned version"
    samples = [(entry["image_path"], part) for entry in json.loads((test / "data_test.json").read_text())
               for part in entry["part_list"]]
    assert len(samples) == 600, f"{len(samples)} test samples, expected 600"
    for image_path, part in samples:
        # file naming from InstructPart's README: <image>-<object>-<part>, spaces -> underscores
        stem = f"{Path(image_path).stem}-{part['object'].replace(' ', '_')}-{part['part'].replace(' ', '_')}"
        assert (test / "images" / f"{stem}.jpg").is_file(), f"missing image {stem}.jpg"
        assert (test / "masks" / f"{stem}.png").is_file(), f"missing mask {stem}.png"
    return f"{len(samples)} test samples; every image and mask present"


CHECKS = {
    "robobench": check_robobench,
    "epic_bench": check_epic_bench,
    "robospatial": check_robospatial,
    "ovdeval": check_ovdeval,
    "instructpart": check_instructpart,
}


def main():
    pins = dataset_pins()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bench", nargs="+", choices=list(pins), default=list(pins))
    args = parser.parse_args()

    failed = 0
    for name in args.bench:
        pin = pins[name]
        try:
            assert pin.downloaded_revision is not None, f"not downloaded (no {pin.dir}/.revision)"
            assert pin.downloaded_revision == pin.revision, (
                f"downloaded {pin.downloaded_revision}, pinned {pin.revision}; re-run download_data.py")
            result = CHECKS[name](pin)
            message, warning = result if isinstance(result, tuple) else (result, None)
            print(f"[PASS] {name:<13} {message}")
            if warning:
                print(f"[WARN] {name:<13} {warning}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name:<13} {type(e).__name__}: {e}")
    print(f"\n{'All checks passed.' if not failed else f'{failed} check(s) failed.'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
