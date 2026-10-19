"""Reading Designer's .sbs XML: typed constant values and parameter lists. The files are data only: they are
parsed as XML and nothing in them is ever executed."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

# Designer's type codes (paraminput <type>, function node <type>) and the function types they are.
TYPES = {4: "bool", 16: "int", 32: "int2", 64: "int3", 128: "int4", 256: "float", 512: "float2", 1024: "float3",
         2048: "float4", 16384: "string"}
IMAGE_TYPES = {1: "color", 2: "grayscale"}
# <relativeTo> of base parameters: graphs (which have no input) and Input nodes (relative to parent by default)
# write 1, never 2; relative to input (2) with no input means the parent.
ABSOLUTE, RELATIVE_PARENT, RELATIVE_INPUT = 0, 1, 2


def parse(tag: str, text: str) -> Any:
    """A constantValue element's value: `tag` without the "constantValue" prefix (Float1, Int2, Bool...)."""
    if tag == "String":
        return text
    if tag == "Bool":
        return text.strip() not in ("0", "false", "")
    numbers = text.split()
    cast = float if tag.startswith("Float") else int
    values = [cast(float(x)) if cast is int else cast(x) for x in numbers]
    return values[0] if tag in ("Float1", "Int1", "Int32") else values


def constant(el: ET.Element | None) -> Any:
    """The first constant value under `el` (None when there is none)."""
    if el is None:
        return None
    for x in el.iter():
        if x.tag.startswith("constantValue") and x.get("v") is not None:
            return parse(x.tag[len("constantValue"):], x.get("v"))
    return None


@dataclass
class Value:
    """A parameter as stored: a constant or a function graph (`dynamic`), and how it inherits (base parameters)."""
    constant: Any = None
    dynamic: ET.Element | None = None
    relative: int | None = None


def parameters(el: ET.Element | None) -> dict[str, Value]:
    """The <parameter> children of a <parameters> element by name."""
    out = {}
    for p in [] if el is None else el.findall("parameter"):
        pv = p.find("paramValue")
        if pv is None or p.find("name") is None:
            continue
        rel = p.find("relativeTo")
        dyn = pv.find("dynamicValue")
        out[p.find("name").get("v")] = Value(None if dyn is not None else constant(pv), dyn,
                                             int(rel.get("v")) if rel is not None else None)
    return out


def text(el: ET.Element | None, path: str, default: str = "") -> str:
    """The `v` attribute of the element at `path` under `el`."""
    found = None if el is None else el.find(path)
    return found.get("v", default) if found is not None else default
