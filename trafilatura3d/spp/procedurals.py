"""The procedurals a .spp project names (fill sources, filters, generators, brush alphas) as our library nodes.

A procedural is a reference to a compiled graph, `resource://<library>/<package>/<graph>?version=<hash>.sbsar`,
with its parameter values ("tweaks", by the graph's parameter identifiers). Compiled graphs are not read: a graph
becomes our node by its identifiers, through the importer's library table (`sbs/library.json`) or the table below
(the mask generators and filters, whose parameters our nodes name after their labels).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from trafilatura3d.nodes.base import ParamKind
from trafilatura3d.paint import graph_nodes as gn
from trafilatura3d.sbs import mapping

# graph identifier -> (our node type, {parameter identifier (lower case): our key}, prefix dropped from identifiers)
NODES = {
    "mg_dirt": ("paint_dirt", {}, ""),
    "mg_position": ("paint_position", {}, "position_"),
    "mg_curvature": ("paint_curvature", {}, "curvature_"),
    "mg_mask_editor": ("paint_mask_editor", {}, ""),
    "mg_metal_edge_wear": ("paint_metal_edge_wear", {}, ""),
    "light": ("paint_light", {}, ""),
    "blur": ("paint_blur", {"intensity": "blur_intensity", "blur_tilling": "blur_wrap"}, ""),
    "blur_directional": ("paint_blur_directional", {"mblurangle": "angle"}, ""),
    "invert": ("paint_invert", {}, ""),
}
# A graph's wrapper adds these (tiling by the texture's ratio, histogram controls): they have no counterpart.
IGNORED = {"$outputsize", "$pixelsize", "$time", "$normalformat", "tile_by_ratio", "tile_ratio", "non_square_expansion"}


@dataclass
class Resolved:
    type: str
    values: dict = field(default_factory=dict)
    unmatched: list = field(default_factory=list)  # parameter identifiers left out


def graph_of(url: str) -> tuple[str, str]:
    """(package, graph) of a procedural url."""
    path = re.sub(r"\?.*$", "", url).split("://", 1)[-1]
    parts = path.split("/")
    if len(parts) < 3:
        raise ValueError(f"not a procedural url: {url!r}")
    return parts[-2], parts[-1]


def _library(package: str, graph: str):
    """Our node map of a library graph, trying the wrapper's graph names: "ratio_<graph>" is <graph>."""
    names = [graph, graph.removeprefix("ratio_")]
    for pkg in (package, f"noise_{package}", package.removeprefix("noise_")):
        for g in names:
            nm = mapping.LIBRARY.get(f"{pkg}.sbs#{g}")
            if nm is not None:
                return nm
    return None


def _value(param, value, values: dict | None = None):
    pm = mapping.ParamMap("", values)
    if param.kind is ParamKind.ENUM and values is None:
        pm = mapping.ParamMap("", dict(enumerate(param.choices)), by_order=True)
    if param.kind is ParamKind.BOOL:
        return bool(value[0] if isinstance(value, list) else value)
    return mapping.convert(value, param, pm)


def resolve(procedural: dict) -> Resolved | None:
    """Our node and its values for a procedural object (urlToSbsRes, tweaks), or None when nothing matches."""
    package, graph = graph_of(procedural["urlToSbsRes"])
    tweaks = {t["identifier"]: t["value"] for t in procedural.get("tweaks", [])}
    out = None
    if graph in NODES:
        kind, renames, prefix = NODES[graph]
        node = gn.NODES.get(kind)
        if node is None:
            return None
        out = Resolved(kind)
        for ident, value in tweaks.items():
            low = ident.lower()
            key = renames.get(low) or (low if low in node.params else low.removeprefix(prefix))
            if ident == "$randomseed":
                out.values[gn.SEED] = int(value)
            elif ident in IGNORED or key not in node.params:
                if ident not in IGNORED:
                    out.unmatched.append(ident)
            else:
                try:
                    out.values[key] = _value(node.params[key], value)
                except (ValueError, KeyError):
                    out.unmatched.append(ident)
        return out
    nm = _library(package, graph)
    if nm is None or nm.type not in gn.NODES:
        return None
    node = gn.NODES[nm.type]
    out = Resolved(nm.type, dict(nm.defaults))
    for ident, value in tweaks.items():
        if ident == "$randomseed":
            out.values[gn.SEED] = int(value)
            continue
        pm = nm.params.get(ident)
        if pm is None or pm.key not in node.params:
            if ident not in IGNORED and ident.removeprefix("$") not in mapping.BASE:
                out.unmatched.append(ident)
            continue
        try:
            out.values[pm.key] = mapping.convert(value, node.params[pm.key], pm)
        except (ValueError, KeyError):
            out.unmatched.append(ident)
    return out


def is_generator(kind: str) -> bool:
    """A node that reads mesh maps (a mask generator): an effect of its own, not a fill source."""
    node = gn.NODES.get(kind)
    return node is not None and any(p in gn.MESH_PORTS for p in node.inputs)
