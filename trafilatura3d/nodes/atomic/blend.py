from pathlib import Path

from trafilatura3d.nodes.base import Input, Node, Param, ParamKind, PortType

LIB = (Path(__file__).parent / "blend_lib.wgsl").read_text(encoding="utf-8")  # blend_atomic(): the modes, shared

MODES = ("Copy", "Add", "Subtract", "Multiply", "Add sub", "Max", "Min", "Switch", "Divide", "Overlay",
         "Screen", "Soft light")
ALPHA = ("Use source alpha", "Ignore alpha", "Straight alpha blending", "Premultiplied alpha blending")


class Blend(Node):
    """Blend the Foreground onto the Background like two layers. Effective opacity k = Opacity
    parameter x Opacity input; outside the Cropping area only the background shows. Measured on the original: the
    alpha is s + ba (1 - s), s = the foreground's alpha x k; the colour mixes the mode's result x into the background
    by s (Use source alpha; Premultiplied too, but for Copy which takes the foreground as premultiplied), by k (Ignore
    alpha), or composites it as W3C does (Straight: the foreground becomes mix(f, x, ba), x clamped in an integer
    format). Switch mixes all four channels by k."""

    type = "blend"
    label = "Blend"
    category = "Atomic nodes"
    catalog = "blend"
    inputs = {"foreground": Input(label="Foreground"), "background": Input(label="Background"),
              "opacity": Input(PortType.GRAY, "Opacity", default=(1.0, 1.0, 1.0, 1.0))}
    main_input = "background"  # Background is the primary input
    params = {
        "opacity": Param(ParamKind.FLOAT, 1.0, "Opacity", 0.0, 1.0),
        "blending_mode": Param(ParamKind.ENUM, "Copy", "Blending mode", choices=MODES),
        "alpha_blending": Param(ParamKind.ENUM, "Use source alpha", "Alpha blending", choices=ALPHA),
        "cropping_area": Param(ParamKind.FLOAT4, [0.0, 1.0, 0.0, 1.0], "Cropping area", 0.0, 1.0),  # x0 x1 y0 y1
    }

    def wgsl(self):
        return LIB + super().wgsl()
