"""Project folders, found from this file so they work from scripts, notebooks and tests."""
from pathlib import Path


def find_project_root(start: Path) -> Path:
    """Walk up from `start` until the folder that contains pyproject.toml."""
    for folder in [start, *start.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    raise FileNotFoundError("Project root (folder with pyproject.toml) not found")


ROOT = find_project_root(Path(__file__).resolve().parent)

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
