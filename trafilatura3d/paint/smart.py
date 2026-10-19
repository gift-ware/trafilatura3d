"""Smart Materials (Smart Materials and Masks page): a folder of layers and effects saved as a preset, reusable
in any Texture Set and project; created from a folder (right click > Create smart material), added from the
Assets (double click or drag into the layer stack). The added layers are a copy: editing the file afterwards does not
change them.

File: `.trafilatura3dsmart`, JSON of the folder (format ours). Painted
pixels are not saved: they belong to one mesh's UVs; paint layers come empty. Where they are: the folders the
caller gives (the editor's libraries), the ones shipped with Trafilatura 3D in `paint/smart_materials/`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from trafilatura3d.paint.model import Layer, ProjectError, drop_unknown_nodes, new_uid, remap_anchors

SUFFIX = ".trafilatura3dsmart"
SHIPPED = Path(__file__).parent / "smart_materials"


def library(folders) -> list[Path]:
    """Every smart material file of the folders, in their order."""
    out = []
    for folder in map(Path, folders):
        if folder.is_dir():
            out += sorted(folder.glob(f"*{SUFFIX}"))
    return out


def save(folder: Layer, path: Path) -> Path:
    if folder.kind != "folder":
        raise ProjectError("a smart material is made from a folder")
    path = Path(path).with_suffix(SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "name": folder.name, "layer": folder.to_dict()}, indent=1), encoding="utf-8")
    return path


def load(path: Path) -> tuple[Layer, list[str]]:
    """The smart material's folder with new uids (its layers and effects get their own), and the warnings about
    library nodes it names that no longer exist (emptied: drop_unknown_nodes)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        warnings = drop_unknown_nodes(data["layer"], Path(path).stem)
        layer = Layer.from_dict(data["layer"])
    except (OSError, ValueError, KeyError) as e:
        raise ProjectError(f"{Path(path).name}: not a smart material ({e})") from None
    return fresh(layer), warnings


def fresh(layer: Layer) -> Layer:
    """A copy with new uids for every layer and effect (the references to its anchor points follow)."""
    out = copy.deepcopy(layer)
    remap = {}
    for lay in out.walk():
        lay.uid = new_uid()
        for fx in lay.effects:
            remap[fx.uid] = fx.uid = new_uid()
        effects = lay.effects + (lay.mask.effects if lay.mask is not None else [])
        for p in lay.paths + (lay.mask.paths if lay.mask is not None else []) + [p for fx in effects for p in fx.paths]:
            p.uid = new_uid()
        if lay.mask is not None:
            for fx in lay.mask.effects:
                remap[fx.uid] = fx.uid = new_uid()
    remap_anchors([out], remap)
    return out


def name_of(path: Path) -> str:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("name") or Path(path).stem
    except (OSError, ValueError):
        return Path(path).stem
