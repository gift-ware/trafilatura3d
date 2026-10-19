"""Shared pieces of the Baked Lighting filters: material filters that bake the lighting of the material into
its color (Paint: a paint layer in passthrough on all channels). They read the material channels and the mesh
maps and write the lit color to `basecolor` (and `diffuse`, the color channel of the specular/glossiness
workflow where the filter has one).

The shading normal is the world space normal map turned by the baked normal map and, where the filter says so,
by the normal and height channels; the tangent frame comes from the position map (see `lighting.wgsl`). The view
is along the normal (a bake has no camera): highlights sit where the surface faces the light's half vector."""

import dataclasses

from trafilatura3d.nodes.base import Node
from trafilatura3d.nodes.paint._lib import DIR, DOC, FILTERS, LIB, MAIN, PaintNode

LIGHTING_DOC = DOC + "filters/lighting-filters/"
INCLUDES = ("matfinish/panorama", "matfinish/lighting")
MESH = ("baked_ambient_occlusion", "baked_normal", "world_space_normals", "position")


def grouped(group: str, params: dict) -> dict:
    return {k: dataclasses.replace(p, group=group) for k, p in params.items()}


def helper(type_: str, shader: str, includes, inputs: dict, outputs: dict, params: dict | None = None,
           internal: dict | None = None, main: bool = False) -> Node:
    """A pass with its own shader (`matfinish/<shader>`) after the `includes`, not in the library."""

    def wgsl(self):
        return "\n".join([LIB, *((DIR / f"{i}.wgsl").read_text(encoding="utf-8") for i in includes),
                          (DIR / "matfinish" / shader).read_text(encoding="utf-8")] + ([MAIN] if main else []))

    cls = type(f"_{type_}", (Node,), {"type": type_, "registered": False, "inputs": inputs, "outputs": outputs,
                                      "params": params or {}, "internal_uniforms": internal or {}, "wgsl": wgsl})
    return cls()


class Lighting(PaintNode):
    category = FILTERS + "/Lighting"
    includes = INCLUDES
