from .common import F, F2, I, Output, Param, PatternNode, PortType, nse


class WeaveGenerator(PatternNode):
    """2/2 twill of Tile X x Tile Y cells: horizontal and vertical blocks two cells long, every row
    shifted by one cell (it repeats every four cells), each block made of Weave parallel stitches.
    Stitches are flat on top and sink towards their ends over half a cell (Shape = how deep: 1 -
    Shape (1 - d / 0.5)^2), Gap = space around every stitch (share of a cell): x between stitches side
    by side along x (and at the ends of the horizontal ones), y the same along y; between two parallel
    stitches a half-height thread shows through, cell corners are black. Outputs:
    Height, Opacity (stitch coverage), Mask 1 / Mask 2 (horizontal / vertical stitches)."""

    type = "weave_generator"
    label = "Weave Generator"
    catalog = "weave_generator"
    libs = ("simple_lib.wgsl",)
    outputs = {"height": Output(PortType.GRAY, "Height"), "opacity": Output(PortType.GRAY, "Opacity"),
               "mask_1": Output(PortType.GRAY, "Mask 1"), "mask_2": Output(PortType.GRAY, "Mask 2")}
    params = {
        "tile_x": Param(I, 4, "Tile X", 1, 20),
        "tile_y": Param(I, 4, "Tile Y", 1, 20),
        "shape": Param(F, 0.5, "Shape", 0.0, 1.0),
        "weave": Param(I, 1, "Weave", 1, 10),
        "gap": Param(F2, [0.09375, 0.09375], "Gap", 0.0, 1.0),
        **nse(),
    }
