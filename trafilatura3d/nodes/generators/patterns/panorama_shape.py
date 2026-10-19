from .common import B, C, E, F, F2, F4, Input, Output, Param, PatternNode, PortType


class PanoramaShape(PatternNode):
    """A studio spotlight (square or disc softbox, or the Input texture) on an equirectangular panorama,
    HDR. The shape lies on the plane tangent to the sphere at its centre, so it distorts like a real
    light in a panorama. Matrix (a 2x2 transformation, rows (x0, x1), (y0, y1))
    maps that plane to the shape, which spans [-1, 1] (Matrix 4, 2 = a softbox about 28 x 53 degrees
    wide); the centre is the panorama point (0.5, 0.25) (45 degrees up) moved by the inverse Matrix
    times Offset. The hotspot is a soft disc inside the shape at Hotspot Position (shape space 0-1,
    0.5 = centre). Without background the shape is composited over the optional Background (our
    addition: chaining several Panorama Shape nodes); Enable Background replaces it with a colour.
    Output defaults to 16-bit float."""

    type = "panorama_shape"
    label = "Panorama Shape"
    catalog = "panorama_shape"
    unsupported = {"Shape Matrix": "the documentation's group; the node has Matrix and Offset",
                   "Enable Backgound": "the documentation's typo of Enable Background"}
    seamless = False
    inputs = {"background": Input(PortType.COLOR, "Background"),
              "input": Input(PortType.ANY, "Input", default=(1.0, 1.0, 1.0, 1.0))}
    main_input = "input"  # the library graph's primary input: Relative to input follows it
    outputs = {"output": Output(PortType.COLOR, "Output")}
    base_defaults = {"output_format": {"mode": "absolute", "value": "16f"}}
    params = {
        "matrix": Param(F4, [4.0, 0.0, 0.0, 2.0], "Matrix", 0.0, 1.0, group="Shape Matrix"),
        "offset": Param(F2, [0.0, 0.0], "Offset", -0.5, 0.5, group="Shape Matrix"),
        "shape": Param(E, "Square", "Shape", choices=("Square", "Disc")),
        "shape_color": Param(C, [0.9, 0.9, 0.9, 1.0], "Shape Color"),
        "shape_intensity": Param(F, 1.0, "Shape Intensity", 0.0, 100.0),
        "shape_soft_border": Param(F, 0.1, "Shape Soft Border", 0.0, 1.0),
        "hotspot_intensity": Param(F, 10.0, "Hotspot Intensity", 0.0, 100.0),
        "hotspot_size": Param(F, 0.5, "Hotspot Size", 0.0, 1.0),
        "hotspot_falloff": Param(F, 0.9, "Hotspot Falloff", 0.0, 1.0),
        "hotspot_position": Param(F2, [0.5, 0.5], "Hotspot Position", 0.0, 1.0),
        "enable_background": Param(B, False, "Enable Background"),
        "background_color": Param(C, [0.0, 0.0, 0.0, 1.0], "Background Color"),
        "enable_texture_input": Param(B, False, "Enable Texture Input"),
    }
