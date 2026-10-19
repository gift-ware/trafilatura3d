from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.materials._base import material_inputs, material_outputs
from trafilatura3d.nodes.paint._lib import mesh_inputs
from trafilatura3d.nodes.paint.matfx._common import E, F, MatFX, stage


class ComicBook(MatFX):
    """Comic book look of the base color: the color (Input Contrast, Input Luminosity; Input Type Grayscale makes
    it monochrome) is shaded by the baked ambient occlusion and curvature, warped by a noise (Warp) and flattened
    into four tone bands with boosted saturation (Color Intensity), mixed by Paint Intensity. Black ink borders
    follow the edges of the paint and the cavities (Border Tolerance, Border Smoothness, Border Opacity); the
    result is sharpened (Sharpness) and adjusted (Output Contrast, Output Luminosity)."""
    type = "paint_material_fx_comic_book"
    label = "Material FX Comic Book"
    catalog = "painter.filter.matfx_comic_book"
    doc = "https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/filters/matfx-filters/matfx-comic-book"
    inputs = {**material_inputs(("basecolor",)), **mesh_inputs("baked_ambient_occlusion", "curvature")}
    outputs = material_outputs(("basecolor",))
    params = {"input_type": E("Input Type", ("Color", "Grayscale")),
              "input_contrast": F("Input Contrast", 0.0, -1.0, 1.0), "input_luminosity": F("Input Luminosity", 0.5),
              "color_intensity": F("Color Intensity", 1.2, 0.0, 2.0), "border_opacity": F("Border Opacity", 1.0),
              "border_tolerance": F("Border Tolerance", 0.25), "border_smoothness": F("Border Smoothness", 0.2),
              "paint_intensity": F("Paint Intensity", 1.0), "warp": F("Warp", 0.2), "sharpness": F("Sharpness", 0.3),
              "output_contrast": F("Output Contrast", 0.0, -1.0, 1.0), "output_luminosity": F("Output Luminosity", 0.5)}
    unsupported = {}

    def encode(self, ctx, job):
        t = ctx.scratch(job, False)
        ctx.dispatch(PAINT, job, job.values, dsts={"paint": t})
        ctx.dispatch(FINAL, job, job.values, srcs={"paint": (t, False)})


P = ComicBook.params
PAINT = stage("paint_material_fx_comic_book_paint", "comic_book_paint.wgsl", ComicBook.inputs,
              {"paint": Output(PortType.COLOR)},
              {k: P[k] for k in ("input_type", "input_contrast", "input_luminosity", "color_intensity",
                                 "paint_intensity", "warp")})
FINAL = stage("paint_material_fx_comic_book_final", "comic_book.wgsl",
              {**mesh_inputs("baked_ambient_occlusion", "curvature"), "paint": Input(PortType.COLOR)}, ComicBook.outputs,
              {k: P[k] for k in ("border_opacity", "border_tolerance", "border_smoothness", "sharpness",
                                 "output_contrast", "output_luminosity")})
