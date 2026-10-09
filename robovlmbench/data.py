"""Pinned benchmark data sources (scripts/dataset_revisions.txt) and where they live on disk."""

from dataclasses import dataclass
from pathlib import Path

from robovlmbench import DATA_DIR, REPO_ROOT

REVISIONS_FILE = REPO_ROOT / "scripts" / "dataset_revisions.txt"


@dataclass(frozen=True)
class DatasetPin:
    name: str
    source: str  # "hf:<repo id>" or "gdrive:<folder id>"
    revision: str

    @property
    def dir(self) -> Path:
        return DATA_DIR / self.name

    @property
    def downloaded_revision(self) -> str | None:
        """Revision recorded by download_data.py after a complete download, or None."""
        marker = self.dir / ".revision"
        return marker.read_text().strip() if marker.is_file() else None


def dataset_pins() -> dict[str, DatasetPin]:
    pins = {}
    for line in REVISIONS_FILE.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, source, revision = line.split()
            pins[name] = DatasetPin(name, source, revision)
    return pins
