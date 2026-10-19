"""Shared pieces of the Metal Finish filters: material filters of base color,
roughness, normal and, where the finish changes them, metallic and opacity. Each input is the stack below of its
channel; each channel takes the output of its name. The relief is added to the normal channel (not the height
channel: Paint would add it a second time when it derives the normal from the height).

Shaders: `finish.wgsl` (periodic noise, cells, streaks, relief) + `uvmap.wgsl` or `triplanar.wgsl` (with the
Triplanar Mapping toggle) + the node's `.wgsl`, which defines `pattern(x)` and `effect(c)`: `effect` returns the
base color and stores the other channels."""

from trafilatura3d.nodes.base import Input, PortType
from trafilatura3d.nodes.materials._base import material_inputs, material_outputs
from trafilatura3d.nodes.paint._lib import DOC, FILTERS, B, PaintNode, mesh_inputs

FINISH_DOC = DOC + "filters/finish-filters/"
UV = ("matfinish/finish", "matfinish/uvmap")
TRIPLANAR = ("matfinish/finish", "matfinish/triplanar")


def channels(*names, extra: dict | None = None, triplanar: bool = False):
    """(inputs, outputs) of a finish changing `names`: the channels, then `extra` inputs (custom textures),
    then the mesh maps of the triplanar projection."""
    ins = {**material_inputs(names), **(extra or {})}
    if triplanar:
        ins.update(mesh_inputs("position", "world_space_normals"))
    return ins, material_outputs(names)


def custom(label: str) -> Input:
    """A custom grayscale texture of the finish (Paint: a resource slot of the effect)."""
    return Input(PortType.GRAY, label, (0.5, 0.5, 0.5, 1.0))


TRIPLANAR_PARAM = {"triplanar_mapping": B("Triplanar Mapping", False)}


class Finish(PaintNode):
    category = FILTERS + "/Finishes"
    includes = UV
