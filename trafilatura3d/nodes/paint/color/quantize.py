from trafilatura3d.nodes.base import Input, Output, ParamKind, PortType
from trafilatura3d.nodes.paint._lib import DOC, FILTERS, B, E, F, I, Filter, blur
from trafilatura3d.nodes.paint.color._passes import helper

GRID = 64  # the palette is extracted from the input averaged down to GRID x GRID cells
ITERATIONS = 3  # Lloyd iterations after each split (and on the first cluster)
SPACES = ("Lab (Color)", "RGB (Data)")


class Quantize(Filter):
    """Reduces the image to a palette of at most Color Amount colors, extracted on the GPU: the input (blurred by
    Contour Smoothing, radius in pixels at 256 px) averaged on a 64x64 grid, clustered by binary splitting (the
    cluster with the largest error splits along its axis of largest variance, Lloyd iterations after each split) in
    the Distance Color Space (Lab = OKLab, RGB = stored values). Each pixel takes its nearest palette color after a
    Dithering offset (Bayer 2x2/4x4/8x8 or Random pattern, cells of a pixel at 256 px). Clusters left empty are not
    colors: Palette Color Amount (value output) is the number actually used. Apply To Alpha makes alpha 0 or 1 at
    Alpha Threshold."""
    type = "paint_quantize"
    label = "Quantize"
    category = FILTERS + "/Advanced"
    catalog = "painter.filter.quantize"
    doc = DOC + "filters/advanced-filters/quantize"
    outputs = {"output": Output(), "palette_color_amount": Output(PortType.VALUE, "Palette Color Amount")}
    params = {"color_amount": I("Color Amount", 8, 2, 64), "contour_smoothing": F("Contour Smoothing", 1.0, 0.0, 16.0),
              "dithering": F("Dithering", 0.0, 0.0, 1.0),
              "dithering_pattern": E("Dithering Pattern", ("Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Random"), "Bayer 4x4"),
              "distance_color_space": E("Distance Color Space", SPACES),
              "apply_to_alpha": B("Apply To Alpha", False), "alpha_threshold": F("Alpha Threshold", 0.5, 0.0, 1.0)}

    def encode(self, ctx, job):
        vals = job.values
        k = max(2, min(64, int(vals["color_amount"])))
        internal = {"k_": k, "space_": SPACES.index(vals["distance_color_space"])}
        src = None
        if vals["contour_smoothing"] > 0.0:
            src = (blur(ctx, job, vals["contour_smoothing"], True, ctx.scratch(job, job.is_gray("output"))),
                   job.is_gray("output"))
        down = ctx.scratch(job, False, "32f", (GRID, GRID))
        ctx.dispatch(DOWN, job, internal, srcs={"input": src} if src else None, dsts={"output": down},
                     size=(GRID, GRID))
        index = ctx.scratch(job, True, "32f", (GRID, GRID))
        pal = [ctx.scratch(job, False, "32f", (k, 2)), ctx.scratch(job, False, "32f", (k, 2))]
        ctx.dispatch(INIT, job, internal, srcs={"down": (down, False)}, dsts={"palette": pal[0]}, size=(k, 1))

        def lloyd():
            for _ in range(ITERATIONS):
                ctx.dispatch(ASSIGN, job, internal, srcs={"down": (down, False), "palette": (pal[0], False)},
                             dsts={"output": index}, size=(GRID, GRID))
                ctx.dispatch(UPDATE, job, internal, srcs={"down": (down, False), "index": (index, True),
                                                          "palette": (pal[0], False)},
                             dsts={"palette": pal[1]}, size=(k, 1))
                pal.reverse()

        lloyd()
        n = 1
        while n < k:
            ctx.dispatch(SPLIT, job, internal, srcs={"palette": (pal[0], False)}, dsts={"palette": pal[1]},
                         size=(k, 1))
            pal.reverse()
            n = min(2 * n, k)
            lloyd()
        ctx.dispatch(FINAL, job, vals, srcs={"palette": (pal[0], False), **({"input": src} if src else {})})


_PAL = {"palette": Output(PortType.COLOR)}
DOWN = helper("paint_quantize_down", "quantize_down.wgsl", {"input": Input()}, {"output": Output(PortType.COLOR)},
              internal={"space_": ParamKind.INT})
INIT = helper("paint_quantize_init", "quantize_init.wgsl", {"down": Input(PortType.COLOR)}, _PAL)
ASSIGN = helper("paint_quantize_assign", "quantize_assign.wgsl",
                {"down": Input(PortType.COLOR), "palette": Input(PortType.COLOR)}, {"output": Output(PortType.GRAY)},
                internal={"k_": ParamKind.INT})
UPDATE = helper("paint_quantize_update", "quantize_update.wgsl",
                {"down": Input(PortType.COLOR), "index": Input(PortType.GRAY), "palette": Input(PortType.COLOR)}, _PAL)
SPLIT = helper("paint_quantize_split", "quantize_split.wgsl", {"palette": Input(PortType.COLOR)}, _PAL,
               internal={"k_": ParamKind.INT})
FINAL = helper("paint_quantize_final", "quantize.wgsl", {"input": Input(), "palette": Input(PortType.COLOR)},
               Quantize.outputs, {n: Quantize.params[n] for n in Quantize.params if n != "contour_smoothing"})
