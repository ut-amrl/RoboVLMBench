"""Download benchmark data into $ROBOVLMBENCH_DATA (default: data/) at the revisions pinned in
scripts/dataset_revisions.txt. Datasets already downloaded at the pinned revision are skipped;
an interrupted download resumes when re-run.

Usage: .venv/bin/python scripts/download_data.py [--bench robobench epic_bench ...]

RoboBench has 88k files: a Hugging Face token (HF_TOKEN, or `hf auth login`) raises rate limits.
"""

import argparse
import hashlib
import re
import shutil
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from robovlmbench.data import DatasetPin, dataset_pins


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------
def hf_snapshot(pin: DatasetPin, dest: Path, patterns: list[str]) -> None:
    snapshot_download(
        pin.source.removeprefix("hf:"), repo_type="dataset", revision=pin.revision,
        local_dir=dest, allow_patterns=patterns,
    )


def hf_fetch_verified(pin: DatasetPin, filenames: list[str], stage: Path) -> None:
    """Download files into `stage` and check each against the hash Hugging Face publishes for it."""
    hf_snapshot(pin, stage, filenames)
    info = HfApi().dataset_info(pin.source.removeprefix("hf:"), revision=pin.revision, files_metadata=True)
    meta = {f.rfilename: f for f in info.siblings}
    for name in filenames:
        f = meta[name]
        ok = sha256(stage / name) == f.lfs.sha256 if f.lfs else git_blob_sha1(stage / name) == f.blob_id
        if not ok:
            raise RuntimeError(f"{name} does not match its published hash; delete {stage} and re-run")


def extract(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(out_dir, filter="data")


def download_robobench(pin: DatasetPin, dest: Path) -> None:
    hf_snapshot(pin, dest, ["[1-5]_*", "system_prompt.json"])


def download_robospatial(pin: DatasetPin, dest: Path) -> None:
    hf_snapshot(pin, dest, ["data/*"])


def download_epic_bench(pin: DatasetPin, dest: Path) -> None:
    # The data/ preview folder on Hugging Face is for its dataset viewer only; the tarballs are the benchmark.
    stage = dest / ".download"
    tarballs = ["Manipulation.tar.gz", "Navigation.tar.gz", "TargetLocalization.tar.gz"]
    hf_fetch_verified(pin, tarballs, stage)
    for name in tarballs:
        print(f"  extracting {name}")
        extract(stage / name, dest)
    shutil.rmtree(stage)


def download_ovdeval(pin: DatasetPin, dest: Path) -> None:
    # Published as <subset>.tar.gz + <subset>.json; laid out as images/<subset>/ + annotations/<subset>.json.
    stage = dest / ".download"
    files = HfApi().list_repo_files(pin.source.removeprefix("hf:"), repo_type="dataset", revision=pin.revision)
    subsets = sorted(f.removesuffix(".tar.gz") for f in files if f.endswith(".tar.gz"))
    hf_fetch_verified(pin, [f"{s}.tar.gz" for s in subsets] + [f"{s}.json" for s in subsets], stage)
    (dest / "annotations").mkdir(parents=True, exist_ok=True)
    for s in subsets:
        print(f"  extracting {s}")
        extract(stage / f"{s}.tar.gz", dest / "images" / s)
        shutil.move(stage / f"{s}.json", dest / "annotations" / f"{s}.json")
    shutil.rmtree(stage)


# ---------------------------------------------------------------------------
# Google Drive (InstructPart: the Hugging Face copy is missing most of the test set)
# ---------------------------------------------------------------------------
def gdrive_list(folder_id: str) -> dict[str, str]:
    """Map name -> id for every entry in a public Drive folder."""
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}"
    html = urllib.request.urlopen(url, timeout=60).read().decode()
    entries = re.findall(r'id="entry-([\w-]+)".*?class="flip-entry-title">([^<]+)<', html, re.S)
    return {name: entry_id for entry_id, name in entries}


def gdrive_download(file_id: str, path: Path) -> None:
    if path.is_file():
        return  # finished on an earlier run (files are written to a temp name first)
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
    with urllib.request.urlopen(url, timeout=120) as r:
        if r.headers.get_content_type() == "text/html":
            raise RuntimeError(f"Drive returned a web page instead of {path.name} (rate limit?); re-run later")
        data = r.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.rename(path)


def download_instructpart(pin: DatasetPin, dest: Path) -> None:
    test = gdrive_list(pin.source.removeprefix("gdrive:"))
    out = dest / "test"
    jobs = [(test["data_test.json"], out / "data_test.json")]
    for sub in ("images", "masks"):
        jobs += [(fid, out / sub / name) for name, fid in gdrive_list(test[sub]).items()]
    print(f"  {len(jobs)} files from Google Drive")
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda job: gdrive_download(*job), jobs))
    if sha256(out / "data_test.json") != pin.revision.removeprefix("sha256:"):
        raise RuntimeError("InstructPart annotations on Drive changed since they were pinned")


DOWNLOADERS = {
    "robobench": download_robobench,
    "epic_bench": download_epic_bench,
    "robospatial": download_robospatial,
    "ovdeval": download_ovdeval,
    "instructpart": download_instructpart,
}


def main():
    pins = dataset_pins()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bench", nargs="+", choices=list(pins), default=list(pins))
    args = parser.parse_args()

    for name in args.bench:
        pin = pins[name]
        if pin.downloaded_revision == pin.revision:
            print(f"== {name}: already at {pin.revision[:19]}, skipping")
            continue
        if pin.downloaded_revision is not None:  # a different pinned revision: start clean
            shutil.rmtree(pin.dir)
        pin.dir.mkdir(parents=True, exist_ok=True)
        print(f"== {name}: {pin.source} @ {pin.revision[:19]} -> {pin.dir}")
        DOWNLOADERS[name](pin, pin.dir)
        (pin.dir / ".revision").write_text(pin.revision + "\n")

    print("\nDone. Check with: .venv/bin/python scripts/check_data.py")


if __name__ == "__main__":
    main()
