"""Quantize Color: reduce a colour image to at most N colours, entirely on the GPU.

1. optional Contour smoothing (gaussian blur, sigma in pixels at 256 px);
2. a 64 x 64 histogram image (Downscale filtering: box average or nearest texel), in the distance
   colour space (Lab / 100 or RGB), weighted by alpha unless Ignore alpha;
3. N defining colours chosen by farthest-point sampling (the first one is the weighted mean), then
   refined by k-means iterations (assign / mean passes);
4. the palette sorted by the ID sort mode (Z-curve from white to black, hue, or most used first);
   colours that ended up unused go last, black and transparent;
5. every pixel takes its nearest palette colour; Dithering chooses between the two nearest with an
   ordered pattern (Bayer 8x8, or an R2 low-discrepancy pattern as blue noise).

Outputs: the quantized image, the ID map (palette index / (N - 1)) and the Palette (N vertical
stripes, index i over x in [i/N, (i+1)/N)). The colour count is a value output: not available yet.
"""

from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode, blur2d, helper, scale, source

HIST = 64  # side of the histogram image
ITERATIONS = 8


class QuantizeColor(BasicNode):
    type = "quantize_color"
    label = "Quantize Color"
    category = "Filters/Adjustments"
    catalog = "quantize_color"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.COLOR, "Output"), "id": Output(PortType.GRAY, "ID"),
               "palette": Output(PortType.COLOR, "Palette"),
               "palette_color_amount": Output(PortType.VALUE, "Palette color amount")}
    params = {
        "max_color_amount": Param(ParamKind.INT, 32, "Max. color amount", 1, 128),
        "contour_smoothing": Param(ParamKind.FLOAT, 0.0, "Contour smoothing", 0.0, 8.0),
        "dithering": Param(ParamKind.FLOAT, 0.0, "Dithering", 0.0, 1.0),
        "dithering_pattern": Param(ParamKind.ENUM, "Blue noise", "Dithering pattern", choices=("Blue noise", "Bayer")),
        "ignore_alpha": Param(ParamKind.BOOL, False, "Ignore alpha"),
        "distance_color_space": Param(ParamKind.ENUM, "Lab (Color)", "Distance color space",
                                      choices=("Lab (Color)", "RGB (Data)")),
        "id_sort_mode": Param(ParamKind.ENUM, "Z-curve", "ID sort mode", choices=("Z-curve", "Hue", "Representativity")),
        "downscale_filtering": Param(ParamKind.ENUM, "Bilinear", "Downscale filtering", choices=("Bilinear", "Nearest")),
    }

    def encode(self, ctx, job):
        v = job.values
        n = max(1, min(256, v["max_color_amount"]))
        src = source(ctx, job, "input")
        if v["contour_smoothing"] * scale(job) > 0.05:
            src = (blur2d(ctx, job, src, src[1], v["contour_smoothing"] * scale(job)), src[1])
        hist = (HIST, HIST)
        small = ctx.scratch(job, False, size=hist)
        ctx.dispatch(DOWN, job, v, srcs={"input": src}, dsts={"small": small}, size=hist)
        pal = [ctx.scratch(job, False, size=(n, 1)) for _ in range(2)]
        mind = [ctx.scratch(job, True, size=hist) for _ in range(2)]
        for k in range(n):  # farthest-point sampling
            vals = {**v, "k": k, "count": n}
            ctx.dispatch(PICK, job, vals, srcs={"small": (small, False), "mind": (mind[0] if k else small, True),
                                                "palette": (pal[0] if k else small, False)},
                         dsts={"palette": pal[1]}, size=(n, 1))
            pal.reverse()
            if k < n - 1:
                ctx.dispatch(MIND, job, vals, srcs={"small": (small, False), "mind": (mind[0] if k else small, True),
                                                    "palette": (pal[0], False)}, dsts={"mind": mind[1]}, size=hist)
                mind.reverse()
        assign = ctx.scratch(job, True, size=hist)
        for _ in range(ITERATIONS):  # k-means
            vals = {**v, "k": 0, "count": n}
            ctx.dispatch(ASSIGN, job, vals, srcs={"small": (small, False), "palette": (pal[0], False)},
                         dsts={"assign": assign}, size=hist)
            ctx.dispatch(UPDATE, job, vals, srcs={"small": (small, False), "assign": (assign, True),
                                                  "palette": (pal[0], False)}, dsts={"palette": pal[1]}, size=(n, 1))
            pal.reverse()
        vals = {**v, "k": 0, "count": n}
        ctx.dispatch(SORT, job, vals, srcs={"palette": (pal[0], False)}, dsts={"palette": pal[1]}, size=(n, 1))
        ctx.dispatch(FINAL, job, vals, srcs={"input": src, "palette": (pal[1], False)})


_K = {"k": ParamKind.INT, "count": ParamKind.INT}
_P = QuantizeColor.params
_C = PortType.COLOR
DOWN = helper("quantize_color_down", ("quantize_color_lib.wgsl", "quantize_color_down.wgsl"),
              {"input": Input(_C)}, {"small": Output(_C)}, _P)
PICK = helper("quantize_color_pick", ("quantize_color_lib.wgsl", "quantize_color_pick.wgsl"),
              {"small": Input(_C), "mind": Input(PortType.GRAY), "palette": Input(_C)}, {"palette": Output(_C)}, _P, _K)
MIND = helper("quantize_color_mind", ("quantize_color_lib.wgsl", "quantize_color_mind.wgsl"),
              {"small": Input(_C), "mind": Input(PortType.GRAY), "palette": Input(_C)},
              {"mind": Output(PortType.GRAY)}, _P, _K)
ASSIGN = helper("quantize_color_assign", ("quantize_color_lib.wgsl", "quantize_color_assign.wgsl"),
                {"small": Input(_C), "palette": Input(_C)}, {"assign": Output(PortType.GRAY)}, _P, _K)
UPDATE = helper("quantize_color_update", ("quantize_color_lib.wgsl", "quantize_color_update.wgsl"),
                {"small": Input(_C), "assign": Input(PortType.GRAY), "palette": Input(_C)}, {"palette": Output(_C)},
                _P, _K)
SORT = helper("quantize_color_sort", ("quantize_color_lib.wgsl", "quantize_color_sort.wgsl"),
              {"palette": Input(_C)}, {"palette": Output(_C)}, _P, _K)
FINAL = helper("quantize_color_final", ("quantize_color_lib.wgsl", "quantize_color.wgsl"),
               {"input": Input(_C), "palette": Input(_C)},
               {"output": Output(_C), "id": Output(PortType.GRAY), "palette": Output(_C),
                "palette_color_amount": Output(PortType.VALUE)}, _P, _K)
