from trafilatura3d.nodes.base import Input, Output, ParamKind, PortType
from trafilatura3d.nodes.materials._base import material_inputs, material_outputs
from trafilatura3d.nodes.paint._lib import mesh_inputs
from trafilatura3d.nodes.paint.matfx._common import ADVANCED, B, C, E, F, I, MatFX, stage

CHANNELS = ("basecolor", "roughness", "metallic", "normal", "height", "ambient_occlusion")
MESH = ("curvature", "baked_ambient_occlusion", "baked_normal", "world_space_normals", "position")
BS, BSE, SM, CO, GR, BL, EC = ("Brush Strokes", "Brush Strokes Effects", "Smoothness", "Colorize", "Gradient",
                               "Baked Lighting", "Edges And Cavities")
BLEND_MODES = ("Normal", "Multiply", "Screen", "Overlay", "Soft Light", "Add")


class Stylization(MatFX):
    """Hand-painted look of a whole material, in sections each scaled by its global intensity (Brush Strokes,
    Smoothness, Colorize, Gradient, Baked Lighthing, Edges and Cavities) and the whole by Stylization:
    - Brush Strokes: procedural stroke stamps (shapes Flat, Round, Dry, Long; Strokes Mode / Strokes Select) in
      cells of Strokes Scale, Strokes Amount of them, rotated (Strokes Rotation, Strokes Rotation Random), scaled
      (Strokes Scale Random, Non-Uniform Size with Strokes Size as aspect); in UV space with Strokes Follow Surface,
      else projected from the three world axes (position and world space normal maps; Projection Hardness, Normal
      Threshold). Each stroke varies color, roughness, metallic and normal (Brush Strokes Effects).
    - Smoothness: Kuwahara smoothing of each channel (radius in pixels at 256 px).
    - Colorize: a color override and a tiled pattern tint (Grunge Variation, Tilling Amount, Pattern Scale).
    - Gradient: one or two colors along a world direction (position map; image up without it), blended with
      their modes.
    - Baked Lighting: a light (Horizontal / Vertical Rotation) shading the base color from the world space normal
      with the baked and material normals as detail (frame built from the world normal: no tangents are given),
      wrap diffuse and specular with their radius / contrast, plus Color Sharpen and Surface Sharpen (curvature).
    - Edges and Cavities: curvature based colors (custom or derived from the base color), masked by the strokes.
    Display Helpers shows a mask in the base color instead."""
    type = "paint_stylization"
    label = "Stylization"
    category = ADVANCED
    catalog = "painter.filter.stylization"
    doc = "https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/filters/advanced-filters/stylization"
    inputs = {**material_inputs(CHANNELS), **mesh_inputs(*MESH)}
    outputs = material_outputs(CHANNELS)
    params = {
        "stylization": F("Stylization", 1.0), "brush_strokes": F("Brush Strokes", 1.0),
        "smoothness": F("Smoothness", 1.0), "colorize": F("Colorize", 1.0), "gradient": F("Gradient", 1.0),
        "baked_lighthing": F("Baked Lighthing", 1.0), "edges_and_cavities": F("Edges and Cavities", 1.0),
        # Brush Strokes
        "strokes_amount": F("Strokes Amount", 0.8, group=BS),
        "strokes_mode": E("Strokes Mode", ("Multiple", "Single"), group=BS),
        "strokes_select_multiple": E("Strokes Select", ("All", "Flat And Round", "Dry And Long"), group=BS),
        "strokes_select_single": E("Strokes Select", ("Flat", "Round", "Dry", "Long"), group=BS),
        "strokes_scale": F("Strokes Scale", 1.0, 0.1, 4.0, BS), "non_uniform_size": B("Non-Uniform Size", False, BS),
        "strokes_size": F("Strokes Size", 0.5, group=BS), "strokes_scale_random": F("Strokes Scale Random", 0.3, group=BS),
        "strokes_follow_surface": B("Strokes Follow Surface", True, BS),
        "strokes_rotation": F("Strokes Rotation", 30.0, 0.0, 360.0, BS),
        "strokes_rotation_random": F("Strokes Rotation Random", 0.3, group=BS),
        "projection_hardness": F("Projection Hardness", 0.5, group=BS),
        "normal_threshold": F("Normal Threshold", 0.2, group=BS),
        # Brush Strokes Effects
        "color_custom": B("Color Custom", False, BSE), "color_variation": F("Color Variation", 0.6, group=BSE),
        "stroke_color_opacity": F("Color Opacity", 1.0, group=BSE),
        "stroke_color": C("Color", (0.85, 0.65, 0.45), BSE), "color_random": F("Color Random", 0.1, group=BSE),
        "roughness_custom": B("Roughness Custom", False, BSE),
        "roughness_variation": F("Roughness Variation", 0.2, group=BSE),
        "stroke_roughness": F("Roughness", 0.6, group=BSE),
        "metallic_custom": B("Metallic Custom", False, BSE),
        "metallic_variation": F("Metallic Variation", 0.0, group=BSE),
        "stroke_metallic": F("Metallic", 0.0, group=BSE),
        "normal_custom": B("Normal Custom", True, BSE), "normal_variation": F("Normal Variation", 0.3, group=BSE),
        "normal_random": F("Normal Random", 0.3, group=BSE),
        "blending_mode_normal": E("Blending Mode (Normal)", ("Detail", "Replace", "Linear"), group=BSE),
        # Smoothness (Kuwahara radius, pixels at 256 px)
        "color_smoothness": F("Color Smoothness", 3.0, 0.0, 16.0, SM),
        "roughness_smoothness": F("Roughness Smoothness", 3.0, 0.0, 16.0, SM),
        "metallic_smoothness": F("Metallic Smoothness", 3.0, 0.0, 16.0, SM),
        "height_smoothness": F("Height Smoothness", 2.0, 0.0, 16.0, SM),
        "normal_smoothness": F("Normal Smoothness", 2.0, 0.0, 16.0, SM),
        "ambient_occlusion_smoothness": F("Ambient Occlusion Smoothness", 2.0, 0.0, 16.0, SM),
        # Colorize
        "colorize_opacity": F("Color Opacity", 0.0, group=CO), "colorize_color": C("Color", (0.8, 0.55, 0.35), CO),
        "grunge_variation": E("Grunge Variation", ("Clouds", "Cells", "Strokes", "Spots"), group=CO),
        "grunge_opacity": F("Grunge Opacity", 0.3, group=CO), "grunge_color": C("Grunge Color", (0.9, 0.75, 0.55), CO),
        "tilling_amount": I("Tilling Amount", 4, 1, 16, CO), "pattern_scale": F("Pattern Scale", 1.0, 0.25, 4.0, CO),
        # Gradient
        "gradient_mode": E("Gradient Mode", ("Two Colors", "One Color"), group=GR),
        "gradient_color": C("Color", (0.25, 0.3, 0.55), GR), "gradient_color_opacity": F("Color Opacity", 0.35, group=GR),
        "color_blending_mode": E("Color Blending Mode", BLEND_MODES, "Multiply", GR),
        "color_2": C("Color 2", (1.0, 0.9, 0.7), GR), "color_2_opacity": F("Color 2 Opacity", 0.3, group=GR),
        "color_2_blending_mode": E("Color 2 Blending Mode", BLEND_MODES, "Screen", GR),
        "gradient_horizontal_rotation": F("Horizontal Rotation", 0.0, 0.0, 360.0, GR),
        "gradient_vertical_rotation": F("Vertical Rotation", 90.0, -90.0, 90.0, GR),
        "gradient_invert": B("Gradient Invert", False, GR), "gradient_offset": F("Gradient Offset", 0.5, group=GR),
        "gradient_contrast": F("Gradient Contrast", 0.2, -1.0, 1.0, GR),
        # Baked Lighting
        "brush_strokes_in_lighting": F("Brush Strokes In Lighting", 0.3, group=BL),
        "diffuse_intensity": F("Diffuse Intensity", 0.5, group=BL), "diffuse_color": C("Diffuse Color", (1.0, 0.96, 0.88), BL),
        "diffuse_radius": F("Diffuse Radius", 0.5, group=BL), "diffuse_contrast": F("Diffuse Contrast", 0.5, group=BL),
        "specular_intensity": F("Specular Intensity", 0.2, group=BL),
        "specular_color": C("Specular Color", (1.0, 1.0, 1.0), BL),
        "specular_radius": F("Specular Radius", 0.3, group=BL), "specular_contrast": F("Specular Contrast", 0.5, group=BL),
        "light_horizontal_rotation": F("Horizontal Rotation", 45.0, 0.0, 360.0, BL),
        "light_vertical_rotation": F("Vertical Rotation", 45.0, -90.0, 90.0, BL),
        "color_sharpen": F("Color Sharpen", 0.3, group=BL), "surface_sharpen": F("Surface Sharpen", 0.5, group=BL),
        # Edges and Cavities
        "mode": E("Mode", ("Both", "Cavities", "Edges"), group=EC),
        "edges_and_cavities_contrast": F("Edges and Cavities Contrast", 0.3, -1.0, 1.0, EC),
        "cavities_opacity": F("Cavities Opacity", 0.5, group=EC), "cavities_spread": F("Cavities Spread", 0.5, group=EC),
        "brush_strokes_in_cavities": F("Brush Strokes In Cavities", 0.5, group=EC),
        "custom_cavities_color": B("Custom Cavities Color", False, EC),
        "cavities_color": C("Cavities Color", (0.15, 0.1, 0.12), EC),
        "edges_opacity": F("Edges Opacity", 0.5, group=EC), "edges_spread": F("Edges Spread", 0.5, group=EC),
        "brush_strokes_in_edges": F("Brush Strokes In Edges", 0.5, group=EC),
        "custom_edges_color": B("Custom Edges Color", False, EC), "edges_color": C("Edges Color", (1.0, 0.95, 0.85), EC),
        "display_helpers": E("Display Helpers", ("None", "Brush Strokes", "Strokes ID", "Edges And Cavities",
                                                 "Gradient Mask", "Baked Lighting"), group="Helpers"),
    }
    unsupported = {}

    def encode(self, ctx, job):
        v = job.values
        st = ctx.scratch(job, False)
        ctx.dispatch(STROKES, job, v, dsts={"strokes": st})
        smooth = {ch: ctx.scratch(job, job.is_gray(ch)) for ch in CHANNELS}
        lo, hi = ctx.scratch(job, False), ctx.scratch(job, False)  # the rows, reused by every channel
        for i, ch in enumerate(CHANNELS):
            pass_values = {**v, "channel_": i, "radius_": float(v[RADIUS[ch]])}
            if pass_values["radius_"] > 0.0 and v["smoothness"] > 0.0:
                ctx.dispatch(KUWAHARA_H, job, pass_values, dsts={"lo": lo, "hi": hi})
            ctx.dispatch(KUWAHARA_V, job, pass_values, srcs={"lo": (lo, False), "hi": (hi, False)},
                         dsts={f"k_{c}": t for c, t in smooth.items()})
        styled = {ch: ctx.scratch(job, job.is_gray(ch)) for ch in CHANNELS}
        ctx.dispatch(STYLE, job, v, srcs={"strokes": (st, False), **{f"k_{ch}": (t, job.is_gray(ch))
                                                                     for ch, t in smooth.items()}},
                     dsts={f"s_{ch}": t for ch, t in styled.items()})
        ctx.dispatch(MIX, job, v, srcs={f"s_{ch}": (t, job.is_gray(ch)) for ch, t in styled.items()})


P = Stylization.params
IN = Stylization.inputs
COLOR_IN = Input(PortType.COLOR)


def _sub(*prefixes, names=()):
    return {k: p for k, p in P.items() if k in names or any(k.startswith(x) for x in prefixes)}


STROKES = stage("paint_stylization_strokes", "stylization_strokes.wgsl",
                {k: IN[k] for k in ("world_space_normals", "position")}, {"strokes": Output(PortType.COLOR)},
                _sub("strokes_", names=("non_uniform_size", "projection_hardness", "normal_threshold")))
RADIUS = {"basecolor": "color_smoothness", **{ch: f"{ch}_smoothness" for ch in CHANNELS[1:]}}
_KUWAHARA = {"channel_": ParamKind.INT, "radius_": ParamKind.FLOAT}
KUWAHARA_H = stage("paint_stylization_kuwahara_h", ("stylization_kuwahara.wgsl", "stylization_kuwahara_h.wgsl"),
                   material_inputs(CHANNELS), {"lo": Output(PortType.COLOR), "hi": Output(PortType.COLOR)},
                   internal=_KUWAHARA)
KUWAHARA_V = stage("paint_stylization_kuwahara_v", ("stylization_kuwahara.wgsl", "stylization_kuwahara_v.wgsl"),
                   {**material_inputs(CHANNELS), "lo": COLOR_IN, "hi": COLOR_IN},
                   {f"k_{ch}": Output(IN[ch].type) for ch in CHANNELS}, {"smoothness": P["smoothness"]}, _KUWAHARA)
STYLE = stage("paint_stylization_style", "stylization_style.wgsl",
              {**{f"k_{ch}": Input(IN[ch].type) for ch in CHANNELS}, "strokes": COLOR_IN,
               **{k: IN[k] for k in MESH}},
              {f"s_{ch}": Output(IN[ch].type) for ch in CHANNELS}, P)
MIX = stage("paint_stylization_mix", "stylization_mix.wgsl",
            {**material_inputs(CHANNELS), **{f"s_{ch}": Input(IN[ch].type) for ch in CHANNELS}},
            Stylization.outputs, {k: P[k] for k in ("stylization", "display_helpers")})
