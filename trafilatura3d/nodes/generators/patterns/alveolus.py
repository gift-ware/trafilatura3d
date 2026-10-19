from .common import B, F, Param, PatternNode, nse, tiling


class Alveolus(PatternNode):
    """Soft (smoothstep) spheres on staggered rows (two columns, two rows half a column apart per
    tiling unit), Interstice Width shrinking them apart; Gradient Filled Cells switches to
    sharp-edged hexagon (Voronoi) cells filled with a gradient from the edges up to the centre."""

    type = "alveolus"
    label = "Alveolus"
    catalog = "alveolus"
    libs = ("simple_lib.wgsl",)
    params = {
        **tiling(1),
        "gradient_filled_cells": Param(B, False, "Gradient Filled Cells"),
        "interstice_width": Param(F, 0.1, "Interstice Width", 0.0, 1.0),
        **nse(),
    }
