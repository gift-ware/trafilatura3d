"""Autosaves of the Graph's packages and of the Paint's projects, one convention: `<name>_<date>_<time><suffix>` in an
`.autosave` folder next to the document (a folder the repository and a project's version control ignore), the last
copies kept."""

from __future__ import annotations

import re
import time
from pathlib import Path

FOLDER = ".autosave"


def backup_path(folder: Path, name: str, suffix: str, keep: int) -> Path:
    """Where the next autosave of the document `name` goes in `folder` (made when missing), its autosaves there beyond
    the last `keep - 1` removed first: only its own (`trim` leaves `trim_A`'s)."""
    folder.mkdir(parents=True, exist_ok=True)
    own = re.compile(re.escape(name) + r"_\d{8}_\d{6}" + re.escape(suffix))
    old = sorted(p for p in folder.iterdir() if own.fullmatch(p.name))
    for p in old[:max(len(old) - (max(keep, 1) - 1), 0)]:
        p.unlink()
    return folder / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}{suffix}"
