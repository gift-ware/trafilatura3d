"""Shared base and parameter sets of the noises_a generators.

Every shader of this folder is `noise_lib.wgsl` + `<shader>.wgsl`. Nodes of a family that differ only
in their recipe share one shader and pick their recipe with the constant `VARIANT` (pipelines are
cached per node type, so each variant compiles its own branch); code reading parameters only some
variants have goes in `snippets`, replacing the `/*@name*/` markers of the shader.

Scale means how many times the whole pattern repeats in the tile
(integer, default 1). A node whose pattern has several cells at
Scale 1 says so in `scale_cells`: the shader gets `p.scale` = Scale x scale_cells (the cells across the
tile) and `DEFAULT_SCALE` = that at the default Scale (the shaders whose Scale repeats the whole pattern
size it on it)."""

from pathlib import Path
from typing import ClassVar

from trafilatura3d.nodes.base import Node, Output, Param, ParamKind, PortType

HERE = Path(__file__).parent
LIB = (HERE / "noise_lib.wgsl").read_text(encoding="utf-8")
CATEGORY = "Texture generators/Noises"
GRAY_OUT = {"output": Output(PortType.GRAY, "Output")}

F, I, B, E = ParamKind.FLOAT, ParamKind.INT, ParamKind.BOOL, ParamKind.ENUM
F2, F3 = ParamKind.FLOAT2, ParamKind.FLOAT3


def scale(default: int, lo: int = 1, hi: int = 64) -> dict:
    return {"scale": Param(I, default, "Scale", lo, hi)}


DISORDER = {"disorder": Param(F, 0.0, "Disorder", 0.0, 1.0),
            "disorder_speed": Param(F, 1.0, "Disorder speed", 0.0, 2.0)}
ANISO = {"disorder_anisotropy": Param(F, 0.0, "Disorder anisotropy", 0.0, 1.0),
         "disorder_anisotropy_angle": Param(F, 0.0, "Disorder anisotropy angle", 0.0, 1.0)}
TILE_OFFSET = {"tile_offset": Param(F2, [0.0, 0.0], "Tile offset", -1.0, 1.0)}
NSE = {"non_square_expansion": Param(B, True, "Non-square expansion")}
# The same, labelled "Non Square Expansion".
NSE_CAPS = {"non_square_expansion": Param(B, True, "Non Square Expansion")}
# The standard set of the classic noises.
CLASSIC = {**DISORDER, **ANISO, **TILE_OFFSET, **NSE}


class NoiseNode(Node):
    """A noises_a node: grayscale output, shader = library + `<shader or type>.wgsl`."""

    category = CATEGORY
    outputs = GRAY_OUT
    shader = ""
    variant = 0
    snippets: dict[str, str] = {}
    scale_cells: ClassVar[int] = 1  # the pattern's cells across the tile at Scale 1

    def pack(self, values: dict) -> bytes:
        if self.scale_cells != 1:
            values = {**values, "scale": values["scale"] * self.scale_cells}
        return super().pack(values)

    def wgsl(self) -> str:
        code = (HERE / f"{self.shader or self.type}.wgsl").read_text(encoding="utf-8")
        for name, snippet in self.snippets.items():
            code = code.replace(f"/*@{name}*/", snippet)
        consts = f"\nconst VARIANT: i32 = {self.variant};\n"
        if "scale" in self.params:
            consts += f"const DEFAULT_SCALE: f32 = {float(self.params['scale'].default * self.scale_cells)};\n"
        return LIB + consts + code
