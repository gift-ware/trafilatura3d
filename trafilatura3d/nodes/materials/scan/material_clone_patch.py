from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import (COLOR, GRAY, B, F, MatNode, channel_toggles, gray_in, material_inputs,
                                           material_outputs)
from trafilatura3d.nodes.materials.scan._clone import clone_encode, clone_params
from trafilatura3d.nodes.materials.scan._common import CATEGORY
from trafilatura3d.nodes.materials.scan.material_crop import SCAN_ON


def _band(diff: str, threshold: str, on: str) -> str:
    """One channel's part of the border band: its threshold (or the common one) times its difference."""
    t = f"select(p.threshold, p.{threshold}, p.use_threshold_per_channel != 0)"
    return f"select(0.0, {t} * sat(({diff}) * 5.0 + nz), {on})"


BAND = "max(max({}, {}), {})".format(
    _band("abs(luma(sample_basecolor(su).rgb) - luma(sample_basecolor(uv).rgb))", "threshold_diffuse_base_color",
          "p.channel_basecolor != 0 || p.channel_diffuse != 0"),
    _band("length(sample_normal(su).rgb - sample_normal(uv).rgb)", "threshold_normal", "p.channel_normal != 0"),
    _band("abs(sample_height(su).r - sample_height(uv).r)", "threshold_height", "p.channel_height != 0"))


class MaterialClonePatch(MatNode):
    """Clone Patch on every channel of a material: one patch mask whose border follows where source and
    target look alike in the base color, the normal and the height (the switched-on channels), each
    with the common Threshold or its own (Use Threshold Per Channel); the normal blended as vectors.
    The optional Mask restricts the patch. Areas: base color, normal and height with both stamps
    lightened."""

    type = "material_clone_patch"
    main_input = "basecolor"  # the library graph's primary input: Relative to input follows it
    label = "Material Clone Patch"
    category = CATEGORY
    catalog = "material_clone_patch"
    band_expr = BAND
    inputs = {**material_inputs(), "mask": gray_in("Mask", 1.0)}
    outputs = {**material_outputs(), "areas_basecolor": Output(COLOR, "Areas Base Color"),
               "areas_normal": Output(COLOR, "Areas Normal"), "areas_height": Output(GRAY, "Areas Height")}
    params = {
        **channel_toggles(SCAN_ON),
        **clone_params("Threshold (for multiple channels)"),
        "use_threshold_per_channel": B(False, "Use Threshold Per Channel", group="Edge"),
        "threshold_diffuse_base_color": F(0.5, "Threshold Diffuse/Base Color", 0.0, 1.0, group="Edge"),
        "threshold_normal": F(0.5, "Threshold Normal", 0.0, 1.0, group="Edge"),
        "threshold_height": F(0.5, "Threshold Height", 0.0, 1.0, group="Edge"),
    }

    def encode(self, ctx, job):
        analysis = {k: self.inputs[k] for k in ("basecolor", "normal", "height")}
        clone_encode(self, ctx, job, analysis, "material_clone_patch.wgsl", self.inputs, self.outputs)
