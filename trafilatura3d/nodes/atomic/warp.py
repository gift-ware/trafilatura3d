from trafilatura3d.nodes.base import Input, Node, Param, ParamKind, PortType


class PixelSized(Node):
    """A node whose move is in units of Pixel Size (per axis): the job's base parameter, or an internal dispatch's
    own `pixel_size_` (default 1)."""
    internal_uniforms = {"pixel_size_": ParamKind.FLOAT2}

    def encode(self, ctx, job):
        ctx.dispatch(self, job, {**job.values, "pixel_size_": job.extra.get("pixel_size", (1.0, 1.0))})

    def pack(self, values):
        return super().pack({"pixel_size_": (1.0, 1.0), **values})


class Warp(PixelSized):
    """Push the input away from the white areas of the Gradient input, along its slope (like
    distorting along the vectors of its normal map). Each pixel reads the input moved by
    Intensity / 256 x the slope (the Gradient input's change across the whole image, its
    gradient in uv) x Pixel Size: a ramp from 0 to 1 across the image at Intensity 256 moves it by
    one image, the unit of Directional warp. The documentation gives no unit: this one is ours, chosen so
    that the nodes built on Warp look like their previews (a Pixel Size of (256 / size)^2 keeps the move in
    pixels over a pixel-sharp edge). Pixel Size multiplies the move per axis, as it multiplies Blur's
    radius."""

    type = "warp"
    label = "Warp"
    category = "Atomic nodes"
    catalog = "warp"
    inputs = {"input": Input(label="Input"), "gradient_input": Input(PortType.GRAY, "Gradient input")}
    params = {
        "intensity": Param(ParamKind.FLOAT, 1.0, "Intensity", 0.0, 64.0),
        "input_filtering_mode": Param(ParamKind.ENUM, "Bilinear", "Input filtering mode",
                                      choices=("Nearest", "Bilinear")),
    }
