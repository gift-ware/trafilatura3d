"""The blend filters: Foreground onto Background with a formula, Opacity x Mask as the strength.
Alpha Blending on: the foreground alpha scales the strength and the alphas are composited
(over); off: the foreground alpha is ignored and the background alpha kept."""

from trafilatura3d.nodes.atomic.blend import LIB as ATOMIC_MODES
from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode

_PARAMS = {
    "opacity": Param(ParamKind.FLOAT, 1.0, "Opacity", 0.0, 1.0),
    "alpha_blending": Param(ParamKind.BOOL, True, "Alpha Blending"),
}
WHITE = (1.0, 1.0, 1.0, 1.0)


class _BlendFilter(BasicNode):
    """`MODE`: the mode in the Blend (Deprecated) list (blend_modes.wgsl)."""
    category = "Filters/Blending"
    inputs = {"foreground": Input(label="Foreground"), "background": Input(label="Background"),
              "mask": Input(PortType.GRAY, "Mask", WHITE)}
    main_input = "background"
    params = _PARAMS
    MODE_WGSL = "fn mode_() -> i32 { return MODE; }\n"

    def wgsl(self) -> str:
        return ATOMIC_MODES + self.MODE_WGSL + super().wgsl()


class ColorBlend(_BlendFilter):
    """Color mode: the Background's luminance with the Foreground's hue and chroma (W3C
    non-separable SetLum, luminance 0.3 R + 0.59 G + 0.11 B)."""
    type = "color_blend"
    label = "Color (Blend Node)"
    catalog = "color_blend"
    inputs = {"foreground": Input(PortType.COLOR, "Foreground"), "background": Input(PortType.COLOR, "Background"),
              "mask": Input(PortType.GRAY, "Mask", WHITE)}
    outputs = {"output": Output(PortType.COLOR)}
    defines = {"MODE": 16}


class LuminosityBlend(_BlendFilter):
    """Luminosity mode: the Background's hue and chroma with the Foreground's luminance."""
    type = "luminosity_blend"
    label = "Luminosity (Blend Node)"
    catalog = "luminosity_blend"
    inputs = ColorBlend.inputs
    outputs = ColorBlend.outputs
    defines = {"MODE": 19}


class ColorBurn(_BlendFilter):
    """1 - (1 - Background) / Foreground, clamped to [0, 1] (a black foreground burns to black)."""
    type = "color_burn"
    label = "Color Burn"
    catalog = "color_burn"
    defines = {"MODE": 12}


class ColorDodge(_BlendFilter):
    """Background / (1 - Foreground), clamped to [0, 1] (a white foreground dodges to white)."""
    type = "color_dodge"
    label = "Color Dodge"
    catalog = "color_dodge"
    defines = {"MODE": 13}


class Difference(_BlendFilter):
    """|Foreground - Background|."""
    type = "difference"
    main_input = "foreground"  # the library graph's primary input: Relative to input follows it
    label = "Difference"
    catalog = "difference"
    inputs = {"background": Input(label="Background"), "foreground": Input(label="Foreground"),
              "mask": Input(PortType.GRAY, "Mask", WHITE)}
    defines = {"MODE": 17}


class LinearBurn(_BlendFilter):
    """Foreground + Background - 1, clamped at 0."""
    type = "linear_burn"
    label = "Linear Burn"
    catalog = "linear_burn"
    defines = {"MODE": 11}


class Dissolve(_BlendFilter):
    """Per pixel either Foreground or Background: a white noise value (per output pixel, from the
    Random Seed) below Opacity x Mask picks the Foreground."""
    type = "dissolve"
    label = "Dissolve"
    catalog = "dissolve"
    defines = {"MODE": 8}


class Exclusion(_BlendFilter):
    """Exclusion (Deprecated): the sum (clamped at 1) less a share of the product, the upper of two lines,
    sum - 1.25 x product and (sum - product) / 2 (fitted on the output: white on white gives black)."""
    type = "exclusion"
    label = "Exclusion (Deprecated)"
    defines = {"MODE": 18}


class HardLight(_BlendFilter):
    """Hard light (Deprecated): Multiply (2 x Background x Foreground) where the Foreground is
    under 0.5, else Screen (1 - 2 (1 - Background)(1 - Foreground))."""
    type = "hard_light"
    label = "Hard light (Deprecated)"
    defines = {"MODE": 15}


MODES = ("Copy", "Add", "Subtract", "Multiply", "Add sub", "Max", "Min", "Dissolve", "Overlay", "Screen",
         "Linear burn", "Color burn", "Color dodge", "Soft light", "Hard light", "Color", "Difference", "Exclusion",
         "Luminosity")


class BlendDeprecated(_BlendFilter):
    """Blend (Deprecated): one of 19 Blending modes (the atomic Blend's and the blend filters'
    formulas)."""
    type = "blend_deprecated"
    label = "Blend (Deprecated)"
    params = {**_PARAMS, "blending_mode": Param(ParamKind.ENUM, "Copy", "Blending mode", choices=MODES)}
    defines = {}
    MODE_WGSL = "fn mode_() -> i32 { return p.blending_mode + 1; }\n"
