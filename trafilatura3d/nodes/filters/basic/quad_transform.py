from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode

CULLING = ("Front only", "Back only", "Front over Back", "Back over Front")


class QuadTransform(BasicNode):
    """Map the image onto the quad of the four corner points (UV, y down), bilinearly. The inverse
    has up to two solutions: the front one (same orientation as the image) and the back one (the
    quad folded over itself); Culling picks which shows. Enable Tiling repeats the image beyond the
    quad's edges; otherwise Background Color fills the rest (a colour)."""

    type = "quad_transform"
    label = "Quad Transform"
    category = "Filters/Transforms"
    catalog = "quad_transform"
    inputs = {"input": Input(label="Input")}
    params = {
        "p00": Param(ParamKind.FLOAT2, [0.0, 0.0], "p00", 0.0, 1.0),
        "p01": Param(ParamKind.FLOAT2, [0.0, 1.0], "p01", 0.0, 1.0),
        "p10": Param(ParamKind.FLOAT2, [1.0, 0.0], "p10", 0.0, 1.0),
        "p11": Param(ParamKind.FLOAT2, [1.0, 1.0], "p11", 0.0, 1.0),
        "culling": Param(ParamKind.ENUM, "Front over Back", "Culling", choices=CULLING),
        "enable_tiling": Param(ParamKind.BOOL, False, "Enable Tiling"),
        "background_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 1.0], "Background Color"),
        "sampling": Param(ParamKind.ENUM, "Bilinear", "Sampling", choices=("Bilinear", "Nearest")),
    }
