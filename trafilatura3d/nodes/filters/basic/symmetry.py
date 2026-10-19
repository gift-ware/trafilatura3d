from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode

MODES = ("Mirror Y", "Mirror X", "Diagonal Left", "Diagonal Right", "Mirror X/Y", "Mirror X / Mirror Y",
         "Diagonal Left / Diagonal Right", "Diagonal Right / Diagonal Left", "8")
TRANSFER = ("Copy", "Add", "Subtract", "Multiply", "Add Sub", "Max", "Min")


class Symmetry(BasicNode):
    """Up to three stages, each blending the image so far with a transformed copy of itself by
    Transfer Mode (the copy as the blend's foreground, everywhere). The copies: Mirror Y flips top-bottom,
    Mirror X left-right, Diagonal Left mirrors across the TR-BL diagonal, Diagonal Right across TL-BR, Mirror
    X/Y turns half a turn; the two-stage modes add a second copy of the result (left-right, TL-BR, a quarter
    turn), and 8 is Diagonal Left / Diagonal Right then a quarter turn. Copy replaces the image by the copy (so
    only the other modes make it symmetric)."""

    type = "symmetry"
    label = "Symmetry"
    category = "Filters/Transforms"
    catalog = "symmetry"
    inputs = {"input": Input(label="Input")}
    params = {
        "symmetry_mode": Param(ParamKind.ENUM, "8", "Symmetry Mode", choices=MODES),
        "transfer_mode": Param(ParamKind.ENUM, "Copy", "Transfer Mode", choices=TRANSFER),
    }
