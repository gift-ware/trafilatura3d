"""Font files (TrueType / OpenType, collections too): their family, style and vertical metrics, and the system font
file of a family name (the font folders of macOS, Windows and Linux, read once per session)."""

from __future__ import annotations

import os
import struct
import sys
from functools import lru_cache
from pathlib import Path

FONT_SUFFIXES = (".ttf", ".otf", ".ttc")
DEFAULT_FAMILIES = ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")  # the font of an empty name


def system_font_dirs() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        return [Path("/System/Library/Fonts"), Path("/Library/Fonts"), home / "Library/Fonts"]
    if os.name == "nt":
        return [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
                Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts"]
    return [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), home / ".local/share/fonts", home / ".fonts"]


def _faces(f) -> list[int]:
    """The offsets of the faces of an open font file (several in a .ttc)."""
    f.seek(0)
    if f.read(4) != b"ttcf":
        return [0]
    f.seek(8)
    n = struct.unpack(">I", f.read(4))[0]
    return list(struct.unpack(f">{n}I", f.read(4 * n)))


def _tables(f, offset: int) -> dict[str, int]:
    f.seek(offset + 4)
    n = struct.unpack(">H", f.read(2))[0]
    f.seek(offset + 12)
    out = {}
    for _ in range(n):
        tag, _, start, _ = struct.unpack(">4sIII", f.read(16))
        out[tag.decode("latin1")] = start
    return out


def _names(f, tables: dict[str, int]) -> tuple[str, str]:
    """(family, style): the typographic names (16, 17) when there are, else the legacy ones (1, 2); English."""
    start = tables["name"]
    f.seek(start)
    _, count, strings = struct.unpack(">HHH", f.read(6))
    records = [struct.unpack(">6H", f.read(12)) for _ in range(count)]
    found = {}
    for platform, _, language, nid, length, offset in records:
        if nid in (1, 2, 16, 17) and (platform, language) in ((3, 0x409), (1, 0)):
            f.seek(start + strings + offset)
            raw = f.read(length)
            found.setdefault(nid, raw.decode("utf-16-be" if platform == 3 else "mac_roman", "ignore"))
    return found.get(16, found.get(1, "")), found.get(17, found.get(2, ""))


def face_info(path: str | Path, index: int = 0) -> tuple[str, str, tuple[float, float, float]]:
    """(family, style, (ascent, descent, line gap)) of a face, the metrics in em (descent positive)."""
    with open(path, "rb") as f:
        offset = _faces(f)[index]
        tables = _tables(f, offset)
        family, style = _names(f, tables)
        f.seek(tables["head"] + 18)
        units = struct.unpack(">H", f.read(2))[0]
        f.seek(tables["hhea"] + 4)
        ascent, descent, gap = struct.unpack(">hhh", f.read(6))
    return family, style, (ascent / units, -descent / units, gap / units)


@lru_cache(maxsize=1)
def _system_faces() -> dict[tuple[str, str], tuple[Path, int]]:
    """(family, style) lowercased -> (file, face index) of every system font (the first file of each)."""
    out: dict[tuple[str, str], tuple[Path, int]] = {}
    for d in system_font_dirs():
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in FONT_SUFFIXES:
                continue
            try:
                with open(p, "rb") as f:
                    for i, offset in enumerate(_faces(f)):
                        family, style = _names(f, _tables(f, offset))
                        out.setdefault((family.lower(), style.lower()), (p, i))
            except (OSError, KeyError, struct.error, IndexError):
                continue  # not a readable font
    return out


def find(family: str, style: str = "") -> tuple[Path, int] | None:
    """The system font file (and face) of a family and style: that style, else Regular, else any style of the family;
    an empty family: the first of DEFAULT_FAMILIES there is."""
    faces = _system_faces()
    for fam in [family] if family else DEFAULT_FAMILIES:
        key = fam.lower()
        for s in (style.lower(), "regular"):
            if (key, s) in faces:
                return faces[(key, s)]
        any_style = next((v for (f, _), v in faces.items() if f == key), None)
        if any_style is not None:
            return any_style
    return None


def split_spec(spec: str) -> tuple[str, str]:
    """A font given as "Family|Style" (or "Family") -> (family, style)."""
    family, _, style = spec.partition("|")
    return family.strip(), style.strip()
