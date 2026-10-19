"""Helper passes of the color filters: shaders of this folder with the shared `color.wgsl` included."""

from __future__ import annotations

from trafilatura3d.nodes.base import Node
from trafilatura3d.nodes.paint._lib import DIR, LIB, MAIN

HERE = DIR / "color"


def helper(type_: str, shader: str, inputs: dict, outputs: dict, params: dict | None = None,
           internal: dict | None = None, main: bool = False) -> Node:
    """A pass (not in the library) running `lib.wgsl` + `color.wgsl` + `shader` of this folder (+ the `main` that
    stores `effect` when `main`)."""

    def wgsl(self):
        return "\n".join([LIB, (HERE / "color.wgsl").read_text(encoding="utf-8"), (HERE / shader).read_text(encoding="utf-8")] + ([MAIN] if main else []))

    cls = type(f"_{type_}", (Node,), {"type": type_, "registered": False, "inputs": inputs, "outputs": outputs,
                                      "params": params or {}, "internal_uniforms": internal or {}, "wgsl": wgsl})
    return cls()
