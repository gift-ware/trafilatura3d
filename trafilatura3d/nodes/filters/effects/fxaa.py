from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._fx import Effect


class FXAA(Effect):
    """Fast approximate anti-aliasing (FXAA 3.11, quality preset 12): finds contrasted edges by
    luminance, searches along each edge for its ends and blends across it by how far the pixel is
    from the edge's end, plus a sub-pixel blend for isolated pixels. Smooths jagged shape edges. The
    edges are found on the linear luma; color: RGB only, unless Affect Alpha
    Channel anti-aliases the alpha too (on its own edges)."""

    type = "fxaa"
    label = "FXAA"
    category = "Filters/Effects"
    catalog = "fxaa"
    inputs = {"input": Input(label="Input")}
    params = {"affect_alpha_channel": Param(ParamKind.BOOL, False, "Affect Alpha Channel")}
