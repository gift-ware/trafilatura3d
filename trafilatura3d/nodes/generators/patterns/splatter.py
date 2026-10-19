import math

from .common import (B, C, E, F, F2, FILTERING, I, Input, Output, Param, PatternNode, PortType, color_variant,
                     prefiltered)


class Splatter(PatternNode):
    """Stamps the Input map on a 2^Grid Number x 2^Grid Number grid (tiles stay seamless), one copy per cell, sized
    Pattern Size Width/Height percent of a cell (Size Variation: a uniform spread of +-0.71 x that many percent),
    rotated by Rotation plus up to Rotation Variation degrees, shifted by up to Disorder/1000 of the image along
    Disorder Angle (Disorder Random: a random angle up to Disorder Angle). Copies are blended by Max of value x Gain
    (Luminance Variation dims them) over the Background Color; Output Level Min/Max are the input levels of the
    result. Zoom magnifies about the centre (x10 at 10000) and breaks tiling; Pan X/Y shift the copies in percent of
    the image. Pattern Size is accepted and has no effect. An unconnected Input reads white (square stamps). Gray or
    colour with the input, colour with Color Variant (Splatter Color).

    Drawn as one quad per cell and per copy of the image's tile that can reach the window (splatter_draw.wgsl),
    blended by Max over the background in a layer, then through the levels (splatter_layer.wgsl): a tile shows
    however far Disorder moves it."""

    type = "splatter"
    label = "Splatter"
    catalog = "splatter"
    libs = ("tile.wgsl", "splatter.wgsl")
    pattern_ports = ("input",)
    inputs = {"input": Input(PortType.ANY, "Input", (1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(PortType.ANY, "Output")}
    params = {
        **color_variant(),
        "pattern_size_width": Param(F, 100.0, "Pattern Size Width", 0.0, 1000.0),
        "pattern_size_height": Param(F, 100.0, "Pattern Size Height", 0.0, 1000.0),
        "rotation": Param(F, 0.0, "Rotation", -360.0, 360.0),
        "rotation_variation": Param(F, 0.0, "Rotation Variation", 0.0, 360.0),
        "zoom": Param(F, 100.0, "Zoom", 100.0, 10000.0),
        "gain": Param(F, 1.0, "Gain", 0.0, 10.0),
        "pan_x": Param(F, 0.0, "Pan X", -100.0, 100.0),
        "pan_y": Param(F, 0.0, "Pan Y", -100.0, 100.0),
        "disorder": Param(F, 0.0, "Disorder", 0.0, 100.0),
        "grid_number": Param(I, 4, "Grid Number", 0, 8),
        "disorder_angle": Param(F, 0.0, "Disorder Angle", 0.0, 360.0),
        "disorder_random": Param(B, False, "Disorder Random"),
        "pattern_size": Param(I, 9, "Pattern Size", 5, 12),
        "size_variation": Param(F, 0.0, "Size Variation", 0.0, 100.0),
        "image_input_filtering": Param(E, FILTERING[0], "Image Input Filtering", choices=FILTERING),
        "output_level_min": Param(F, 0.0, "Output Level Min", 0.0, 1.0),
        "output_level_max": Param(F, 1.0, "Output Level Max", 0.0, 1.0),
        "background_color": Param(C, [0.0, 0.0, 0.0, 1.0], "Background Color"),
        "luminance_variation": Param(F, 0.0, "Luminance Variation", 0.0, 1.0),
        "color_variation": Param(F, 0.0, "Color Variation", 0.0, 1.0),
    }

    def encode(self, ctx, job):
        v = job.values
        gray = int(job.is_gray("output"))
        w, h = job.size
        layer = ctx.scratch(job, bool(gray), "32f" if "float32-blendable" in ctx.ev.device.features else "16f")
        ctx.dispatch(LAYER, job, {**v, "gray_": gray, "final_": 0}, dsts={"output": layer})
        # The copies of the image's tile (a whole image apart) whose tiles can reach the window: the tiles' centres
        # lie within Disorder of their cells, their quads within their half diagonal (and a pixel) of it.
        n = 1 << min(max(v["grid_number"], 0), 8)
        t = min(max((v["zoom"] - 100.0) / 9900.0, 0.0), 1.0)
        half_window = 0.5 * (1.0 - 0.9 * t)  # half the window in the pattern's uv: 0.5 / magnification
        size = max(v["pattern_size_width"], v["pattern_size_height"]) + 0.71 * v["size_variation"]
        r = v["disorder"] / 1000.0 + size / 100.0 * 0.5 / n * 1.415 + 2.0 / min(w, h)
        lo, count = [], []
        for pan in (v["pan_x"], v["pan_y"]):
            a, b = 0.5 - half_window - pan / 100.0, 0.5 + half_window - pan / 100.0
            first = math.floor(a - 1.0 - r)
            lo.append(float(first))
            count.append(float(math.ceil(b + r) - first + 1))
        srcs = {}
        if v["image_input_filtering"] == FILTERING[0]:  # pat_sample's footprint (tg_aa / 2) on the largest tile
            largest = (min(v["pattern_size_width"], v["pattern_size_height"]) + 0.71 * v["size_variation"]) / 100.0
            srcs = prefiltered(ctx, job, self.pattern_ports, 0.5 / (min(w, h) * max(largest * 0.5 / n, 1e-9)))
        ctx.draw(DRAW, job, {**v, "gray_": gray, "copy_lo_": lo, "copy_n_": count}, layer,
                 n * n * int(count[0]) * int(count[1]), srcs=srcs)
        ctx.dispatch(LAYER, job, {**v, "gray_": gray, "final_": 1}, srcs={"layer": (layer, bool(gray))})


class _Draw(Splatter):
    """The Splatter's tiles drawn over the layer (splatter_draw.wgsl)."""
    type = "splatter_draw"
    registered = False
    outputs = {}
    internal_uniforms = {"gray_": I, "copy_lo_": F2, "copy_n_": F2}
    shader = "splatter_draw.wgsl"

    def blend(self, values):
        state = {"src_factor": "one", "dst_factor": "one", "operation": "max"}
        return {"color": state, "alpha": state}


class _Layer(Splatter):
    """The Splatter's layer: the background written into it, then it through the levels (splatter_layer.wgsl)."""
    type = "splatter_layer"
    registered = False
    pattern_ports = ()
    inputs = {"layer": Input(PortType.COLOR, "Layer")}
    internal_uniforms = {"gray_": I, "final_": I}
    shader = "splatter_layer.wgsl"


DRAW, LAYER = _Draw(), _Layer()
