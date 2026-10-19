from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import (COLOR, NORMAL_FORMATS, B, E, F, F3, I, MatNode, Pass, color_in, gray_in,
                                           passes_of)
from trafilatura3d.nodes.materials.scan._atlas import MAX_RES, PRECISION, detect
from trafilatura3d.nodes.materials.scan._common import CATEGORY

CHANNELS = ("basecolor", "normal", "metallic", "roughness", "height", "opacity", "user_1", "user_2")
CROP_MODES = ("Largest shape (Square)", "Individual crop (Square)", "Individual crop (Rectangle)")


class AtlasSplitter(MatNode):
    """Splits an atlas into its elements: shapes are the connected opaque areas of the Opacity map
    (detected at up to 512 px, optionally downscaled to drop noise), listed in reading order, shapes
    with a smaller diagonal than Ignore Shape Smaller Than skipped. Outputs the element at Shape
    Selection (1 = the first) on every channel, or with Grid View all elements in a grid (lines and selection drawn
    on the base color). Auto Scale fits an element to its cell, Auto Crop removes the margin (Auto Crop Mode: all at the largest element's square, each in its
    own square or rectangle), Auto
    Rotation turns tall elements upright, Rotation turns them further; outside an element, colors
    are dilated from its edge (Dilation Width, not on Normal and Height) or set to the Bg colors."""

    type = "atlas_splitter"
    label = "Atlas Splitter"
    category = CATEGORY
    catalog = "atlas_splitter"
    includes = ("_atlas.wgsl",)
    inputs = {"basecolor": color_in("Base Color", (0.5, 0.5, 0.5, 1.0)),
              "normal": color_in("Normal", (0.5, 0.5, 1.0, 1.0)), "metallic": gray_in("Metallic", 0.0),
              "roughness": gray_in("Roughness", 0.5), "height": gray_in("Height", 0.0),
              "opacity": gray_in("Opacity", 0.0), "user_1": gray_in("User 1", 0.0), "user_2": gray_in("User 2", 0.0)}
    main_input = "basecolor"  # the library graph's primary input: Relative to input follows it
    outputs = {k: Output(v.type, v.label) for k, v in inputs.items()}
    params = {
        "grid_view": B(False, "Grid View"),
        "grid_opacity": F(1.0, "Grid Opacity", 0.0, 1.0),
        "grid_selection_opacity": F(1.0, "Grid Selection Opacity", 0.0, 1.0),
        "auto_scale": B(True, "Auto Scale"),
        "auto_crop": B(False, "Auto Crop"),
        "auto_crop_mode": E("Largest shape (Square)", "Auto Crop Mode", CROP_MODES),
        "shape_selection": I(1, "Shape Selection", 1, 10),
        "ignore_shape_smaller_than": F(0.1, "Ignore Shape Smaller Than", 0.0, 1.0),
        "auto_rotation": B(False, "Auto Rotation"),
        "rotation": F(0.0, "Rotation", 0.0, 1.0),
        "input_normal_format": E("DirectX", "Input Normal Format", NORMAL_FORMATS),
        "downscale_opacity_mask": I(2, "Downscale Opacity Mask", 0, 4),
        "shape_detection_precision": E("Complex or big shapes", "Shape Detection Precision", PRECISION),
        "dilation_width": F(0.0, "Dilation Width", 0.0, 32.0),
        "enable_additional_inputs": B(False, "Enable Additional Inputs"),
        "custom_background_color": B(False, "Custom Background Color"),
        "base_color_bg_color": F3([0.0, 0.0, 0.0], "Base Color Bg Color", 0.0, 1.0),
        "normal_bg_color": F3([0.5, 0.5, 1.0], "Normal Bg Color", 0.0, 1.0),
        "metallic_bg_color": F(0.0, "Metallic Bg Color", 0.0, 1.0),
        "roughness_bg_color": F(0.5, "Roughness Bg Color", 0.0, 1.0),
        "height_bg_color": F(0.0, "Height Bg Color", 0.0, 1.0),
        "user_1_bg_color": F(0.0, "User 1 Bg Color", 0.0, 1.0),
        "user_2_bg_color": F(0.0, "User 2 Bg Color", 0.0, 1.0),
    }

    def encode(self, ctx, job):
        main = passes_of(self, lambda n: Pass(n, "main", "atlas_splitter.wgsl",
                                              {**n.inputs, "lbl": Input(COLOR), "table": Input(COLOR),
                                               "list": Input(COLOR)}, n.outputs))
        opac = job.inputs.get("opacity")
        res = opac[0].size[0] if opac else job.size[0]
        res = min(res, MAX_RES) >> max(0, job.values["downscale_opacity_mask"])
        scale = (1, 2, 4)[PRECISION.index(job.values["shape_detection_precision"])]
        srcs = detect(self, ctx, job, "opacity", res, scale, job.values["ignore_shape_smaller_than"])
        ctx.dispatch(main, job, job.values, srcs=srcs)
