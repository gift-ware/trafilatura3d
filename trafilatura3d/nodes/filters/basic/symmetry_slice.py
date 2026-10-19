from trafilatura3d.nodes.base import Input, Param, ParamKind

from .symmetry import TRANSFER
from ._lib import BasicNode

MODES = ("Horizontal", "Vertical", "Diagonal Left-Right", "Diagonal Right-Left", "Vertical Invert", "Corner",
         "Diagonal Corner")


class SymmetrySlice(BasicNode):
    """The image blended (Transfer Mode, the Symmetry node's list) with a mirrored copy of itself
    through a mask from the signed distance to the mirror line (relative to the farthest point), a smoothstep as
    wide as Blend (0: a hard cut, 1: across the whole image). Horizontal: the bottom takes the top flipped top-bottom; Vertical: the right
    takes the left flipped left-right; Diagonal Left-Right: the lower-right triangle takes the upper-left one
    across the TR-BL diagonal; Diagonal Right-Left: the upper-right triangle takes the lower-left one across
    TL-BR; Vertical Invert: the left half takes the right one turned half a turn. Corner: Horizontal, then the
    left half takes the right one (the top-right quadrant is the source); Diagonal Corner: Diagonal Left-Right,
    then Diagonal Right-Left (the left eighth-pair, the wedge at the left edge, is the source). Flip Side turns
    the mask half a turn (the other side is the source), Flip Side2 only the second stage's (the corner modes)."""

    type = "symmetry_slice"
    label = "Symmetry Slice"
    category = "Filters/Transforms"
    catalog = "symmetry_slice"
    inputs = {"input": Input(label="Input")}
    params = {
        "symmetry_mode": Param(ParamKind.ENUM, "Corner", "Symmetry Mode", choices=MODES),
        "transfer_mode": Param(ParamKind.ENUM, "Copy", "Transfer Mode", choices=TRANSFER),
        "blend": Param(ParamKind.FLOAT, 0.0, "Blend", 0.0, 1.0),
        "flip_side": Param(ParamKind.BOOL, False, "Flip Side"),
        "flip_side2": Param(ParamKind.BOOL, False, "Flip Side2"),
    }
