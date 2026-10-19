from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import (COLOR, GRAY, NORMAL_FORMATS, B, E, F, F2, F3, I, I2, MatNode, Pass, color_in,
                                           gray_in, passes_of)
from trafilatura3d.nodes.materials.scan._atlas import PRECISION, detect
from trafilatura3d.nodes.materials.scan._common import CATEGORY



class AtlasScatter(MatNode):
    """Scatters the elements of an atlas material over a background material, like Shape Splatter:
    elements are detected in the atlas Opacity (an internal Atlas Splitter) and placed one per cell
    of an X x Y grid, chosen at random or by index / line / column (Pattern Range limits them,
    the Map Input shifts the choice), then jittered (gaussian or uniform), offset, pushed by the Vector
    Map, rotated, scaled, sized, and masked at random or by background slope. Heights stack by Height
    Blend (highest wins) or Alpha Test (later cells on top), scaled and offset (random, by map),
    conformed to and skewed by the background; colors, normals and roughness can be adjusted,
    randomized and pulled towards the background. Shapes reach at most two cells from their cell
    (Out of Bounds Option: shrink them or constrain their position). Seamless."""

    type = "atlas_scatter"
    label = "Atlas Scatter"
    category = CATEGORY
    catalog = "atlas_scatter"
    unsupported = {"Height Offset Map Mulitplier": "the documentation's typo of Height Offset Map Multiplier"}
    includes = ("_atlas.wgsl",)
    inputs = {
        "atlas_basecolor": color_in("Atlas Base Color", (0.5, 0.5, 0.5, 1.0)),
        "atlas_normal": color_in("Atlas Normal", (0.5, 0.5, 1.0, 1.0)),
        "atlas_height": gray_in("Atlas Height", 0.5),
        "atlas_roughness": gray_in("Atlas Roughness", 0.5),
        "atlas_opacity": gray_in("Atlas Opacity", 0.0),
        "background_basecolor": color_in("Background Base Color", (0.3, 0.3, 0.3, 1.0)),
        "background_normal": color_in("Background Normal", (0.5, 0.5, 1.0, 1.0)),
        "background_height": gray_in("Background Height", 0.0),
        "background_roughness": gray_in("Background Roughness", 0.8),
        "map_input": gray_in("Map Input", 0.5),
        "vector_map": color_in("Vector Map", (0.5, 0.5, 0.0, 1.0)),
    }
    main_input = "background_basecolor"
    outputs = {"basecolor": Output(COLOR, "Base Color"), "normal": Output(COLOR, "Normal"),
               "height": Output(GRAY, "Height"), "roughness": Output(GRAY, "Roughness"),
               "opacity": Output(GRAY, "Opacity")}
    params = {
        "atlas_input_resolution": I2([11, 11], "Atlas Input Resolution", 0, 12),
        "x_amount": I(16, "X Amount", 1, 64),
        "y_amount": I(16, "Y Amount", 1, 64),
        "pattern_range": I2([0, 0], "Pattern Range", 0, 10, group="Pattern"),
        "pattern_distribution_mode": E("Random", "Pattern Distribution Mode",
                                       ("Random", "Pattern Index", "Line Index", "Column Index"), group="Pattern"),
        "pattern_distribution_map_multiplier": F(0.0, "Pattern Distribution Map Multiplier", 0.0, 1.0, group="Pattern"),
        "pattern_rotation": E("0", "Pattern Rotation", ("0", "90", "180", "270"), group="Pattern"),
        "pattern_rotation_random": F(0.0, "Pattern Rotation Random", 0.0, 1.0, group="Pattern"),
        "atlas_shape_detection_precision": E("Complex or big shapes", "Atlas Shape Detection Precision", PRECISION,
                                             group="Pattern"),
        "downscale_atlas_opacity": I(-2, "Downscale Atlas Opacity (faster detection)", -4, 0, group="Pattern"),
        "ignore_shape_smaller_than": F(0.01, "Ignore Shape Smaller Than", 0.0, 1.0, group="Pattern"),
        "scale": F(1.0, "Scale", 0.0, 5.0, group="Size"),
        "scale_random": F(0.0, "Scale Random", 0.0, 1.0, group="Size"),
        "scale_no_overlap": F(0.0, "Scale No Overlap", 0.0, 1.0, group="Size"),
        "scale_map_multiplier": F(0.0, "Scale Map Multiplier", 0.0, 1.0, group="Size"),
        "size": F2([1.0, 1.0], "Size", 0.0, 1.0, group="Size"),
        "size_ratio_from_bg_slope": F(0.0, "Size Ratio from Bg Slope", 0.0, 1.0, group="Size"),
        "preserve_aspect_ratio": F(0.0, "Preserve Aspect Ratio", 0.0, 1.0, group="Size"),
        "position_random": F(0.0, "Position Random", 0.0, 2.0, group="Position"),
        "random_distribution": E("Gaussian", "Random Distribution", ("Gaussian", "Uniform"), group="Position"),
        "position_vector_map_multiplier": F(0.0, "Vector Map Multiplier", 0.0, 1.0, group="Position"),
        "offset_horizontal": F(0.0, "Offset Horizontal", -2.0, 2.0, group="Position"),
        "offset_vertical": F(0.0, "Offset Vertical", -2.0, 2.0, group="Position"),
        "out_of_bounds_option": E("Scale Shape", "Out of Bounds Option", ("Scale Shape", "Constrain Position"),
                                  group="Position"),
        "rotation": F(0.0, "Rotation", 0.0, 1.0, group="Rotation"),
        "rotation_random": F(1.0, "Rotation Random", 0.0, 1.0, group="Rotation"),
        "rotation_from_bg_slope": F(0.0, "Rotation from Bg Slope", 0.0, 1.0, group="Rotation"),
        "rotation_map_multiplier": F(0.0, "Rotation Map Multiplier", 0.0, 1.0, group="Rotation"),
        "rotation_vector_map_multiplier": F(0.0, "Vector Map Multiplier", 0.0, 1.0, group="Rotation"),
        "height_scale_auto_adjust": B(True, "Height Scale Auto Adjust", group="Height"),
        "blend_mode": E("Height Blend", "Blend Mode", ("Height Blend", "Alpha Test"), group="Height"),
        "height_offset": F(0.0, "Height Offset", -1.0, 1.0, group="Height"),
        "height_offset_random": F(0.0, "Height Offset Random", 0.0, 1.0, group="Height"),
        "height_offset_map_multiplier": F(0.0, "Height Offset Map Multiplier", 0.0, 1.0, group="Height"),
        "height_scale": F(1.0, "Height Scale", 0.0, 1.0, group="Height"),
        "height_scale_random": F(0.0, "Height Scale Random", 0.0, 1.0, group="Height"),
        "height_scale_map_multiplier": F(0.0, "Height Scale Map Multiplier", 0.0, 1.0, group="Height"),
        "conform_to_background": F(0.0, "Conform to Background", 0.0, 1.0, group="Height"),
        "smooth_conformed_background": F(0.0, "Smooth Conformed Background", 0.0, 2.0, group="Height"),
        "height_skew": F(0.0, "Skew from Bg Slope", 0.0, 1.0, group="Height"),
        "background_slope_smoothness": F(0.0, "Background Slope Smoothness", 0.0, 2.0, group="Height"),
        "cutout_black_pixels": B(True, "Cutout Black Pixels", group="Height"),
        "flatten_pattern_base": B(False, "Flatten Pattern Base", group="Height"),
        "mask_random": F(0.0, "Mask Random", 0.0, 1.0, group="Masking"),
        "mask_random_map_multiplier": F(0.0, "Mask Random Map Multiplier", 0.0, 1.0, group="Masking"),
        "mask_from_bg_slope": F(0.0, "Mask from Bg Slope", -1.0, 1.0, group="Masking"),
        "color_adjustment": F3([0.0, 0.0, 0.0], "Color Adjustment", -1.0, 1.0, group="Color"),
        "color_random": F3([0.0, 0.0, 0.0], "Color Random", 0.0, 1.0, group="Color"),
        "color_from_background": F3([0.0, 0.0, 0.0], "Color from Background", 0.0, 1.0, group="Color"),
        "normal_skew": F(0.0, "Skew from Bg Slope", 0.0, 1.0, group="Normal"),
        "normal_random": F(0.0, "Normal Random", 0.0, 1.0, group="Normal"),
        "normal_format": E("OpenGL", "Normal Format", NORMAL_FORMATS, group="Normal"),
        "roughness_adjustment": F(0.0, "Roughness Adjustment", -1.0, 1.0, group="Roughness"),
        "roughness_from_background": F(0.0, "Roughness from Background", 0.0, 1.0, group="Roughness"),
        "roughness_random": F(0.0, "Roughness Random", 0.0, 1.0, group="Roughness"),
    }

    def encode(self, ctx, job):
        main = passes_of(self, lambda n: Pass(n, "main", "atlas_scatter.wgsl",
                                              {**n.inputs, "lbl": Input(COLOR), "table": Input(COLOR),
                                               "list": Input(COLOR)}, n.outputs))
        v = job.values
        res = (1 << max(v["atlas_input_resolution"])) >> max(0, -v["downscale_atlas_opacity"])
        scale = (1, 2, 4)[PRECISION.index(v["atlas_shape_detection_precision"])]
        srcs = detect(self, ctx, job, "atlas_opacity", res, scale, v["ignore_shape_smaller_than"])
        ctx.dispatch(main, job, v, srcs=srcs)
