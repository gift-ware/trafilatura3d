"""The "3D" noises: evaluated at the 3D position read from a Position Map (any baked position map,
or Cube 3D GBuffers), in unit space [0, 1]^3 times Scale * Size, plus Offset. Without a position map
they render the slice z = 0.5 of the unit cube, which tiles in 2D when the noise tiles (Enable
Tiling, or always for the nodes whose lattice is integer-scaled).

Parameters: ranges, Contrast 0 = as it is (x 1 / (1 - Contrast)), fractal levels 1-7, Invert off; defaults
follow the previews: Scale 4 and 3 (the previews' frequency) and Enable Tiling on (our generators tile)."""

from trafilatura3d.nodes.base import Input, Param, PortType

from .common import B, E, F, F3, I, NoiseNode

POS = {"position_map": Input(PortType.COLOR, "Position Map", (0.0, 0.0, 0.0, 1.0))}
DISTANCE_MODES = ("Euclidean", "Manhattan", "Chebyshev", "Minkowski")


def frame(voronoi: bool = False) -> dict:
    """Invert, Scale, Size, Offset, (Disorder,) Distortion: the common head, in this order (the Voronoi
    nodes' ranges for `voronoi`)."""
    hi = 1.0 if voronoi else 2.0
    return {
        "invert": Param(B, False, "Invert"),
        "scale": Param(F, 4.0, "Scale", 0.0, 32.0 if voronoi else 64.0),
        "size": Param(F3, [1.0, 1.0, 1.0], "Size", 0.0, hi),
        "offset": Param(F3, [0.0, 0.0, 0.0], "Offset", 0.0 if voronoi else -1.0, 1.0),
        **({"disorder": Param(F, 1.0, "Disorder", 0.0, 1.0)} if voronoi else {}),
        "distortion_intensity": Param(F, 0.0, "Distortion Intensity", 0.0, 1.0),
        "distortion_scale_multiplier": Param(F, 1.0, "Distortion Scale Multiplier", 0.0, 2.0),
    }


TONE = {"baseline": Param(F, 0.5, "Baseline", 0.0, 1.0), "contrast": Param(F, 0.0, "Contrast", 0.0, 1.0)}
ABSOLUTE = {"absolute": Param(B, False, "Absolute")}
TILING = {"enable_tiling": Param(B, True, "Enable Tiling")}
FRACTAL = {
    "min_level": Param(I, 1, "Min Level", 1, 7),
    "max_level": Param(I, 7, "Max Level", 1, 7),
    "roughness": Param(F, 1.0, "Roughness", 0.0, 2.0),
    "lacunarity": Param(F, 2.0, "Lacunarity", 1.0, 4.0),
    "global_opacity": Param(F, 1.0, "Global Opacity", 0.0, 1.0),
}
VORONOI = {
    "rounded_curve": Param(F, 0.0, "Rounded Curve", 0.0, 1.0),
    "distance_scale": Param(F, 1.0, "Distance Scale", 0.0, 1.0),
    "distance_mode": Param(E, "Euclidean", "Distance Mode", choices=DISTANCE_MODES),
    "minkowski_number": Param(F, 1.0, "Minkowski Number", 0.01, 5.0),
}
STYLE = {
    "style": Param(E, "F1", "Style",
                   choices=("F1", "F2", "F2-F1", "F1*F2", "F1/F2", "Edge", "Random color")),
    "edge_thickness": Param(F, 0.1, "Edge Thickness", 0.0, 0.5),
    # The Random color style's colours: from the node's seed or from Random color seed.
    "random_color_seed_mode": Param(E, "Global random seed", "Random color seed mode",
                                    choices=("Global random seed", "Manual seed")),
    "random_color_seed": Param(I, 0, "Random color seed", 0, 100),
}


class Noise3D(NoiseNode):
    inputs = POS
    random_color_style: tuple = ()

    def output_types(self, input_types, values):
        # Random color styles give a color image, every other style a grayscale one.
        types = super().output_types(input_types, values)
        if self.random_color_style and values.get(self.random_color_style[0]) == self.random_color_style[1]:
            return {name: PortType.COLOR for name in types}
        return types


class Perlin3D(Noise3D):
    """Perlin noise in 3D space."""
    type = "perlin_noise_3d"
    label = "3D Perlin Noise"
    catalog = "3d_perlin_noise"
    shader = "perlin3d"
    params = {**frame(), **TONE, **ABSOLUTE, **TILING}


class PerlinFractal3D(Noise3D):
    """Fractal sum of 3D Perlin noise levels."""
    type = "perlin_noise_fractal_3d"
    label = "3D Perlin Noise Fractal"
    catalog = "3d_perlin_noise_fractal"
    shader = "perlin_fractal3d"
    params = {**frame(), **FRACTAL, **TONE, **ABSOLUTE, **TILING}
    variant = 0
    absolute_code = "if (p.absolute != 0) { v = abs(n) * 2.0 - 1.0; }"

    def wgsl(self) -> str:
        return super().wgsl().replace("#ABS#", self.absolute_code)


class RidgedFractal3D(PerlinFractal3D):
    """Fractal ridged noise in 3D (each level folded: 1 - |perlin|), dark creases on a textured ground; Roughness
    0.5."""
    type = "ridged_noise_fractal_3d"
    label = "3D Ridged Noise Fractal"
    catalog = "3d_ridged_noise_fractal"
    params = {**frame(), **FRACTAL, "roughness": Param(F, 0.5, "Roughness", 0.0, 2.0), **TONE, **TILING}
    variant = 1
    absolute_code = ""


class Simplex3D(Noise3D):
    """Simplex-type noise in 3D (body-centred cubic lattice); always tiles: the period is
    round(Scale * Size) per axis."""
    type = "simplex_noise_3d"
    label = "3D Simplex Noise"
    catalog = "3d_simplex_noise"
    shader = "simplex3d"
    params = {"scale": Param(F, 4.0, "Scale", 0.0, 64.0), "size": Param(F3, [1.0, 1.0, 1.0], "Size", 0.0, 2.0)}


class Voronoi3D(Noise3D):
    """Voronoi (cellular) noise in 3D: F1, F2, combinations, edges or a random color per cell."""
    type = "voronoi_3d"
    label = "3D Voronoi"
    catalog = "3d_voronoi"
    shader = "voronoi3d"
    params = {**frame(voronoi=True), **VORONOI, **STYLE, **TILING}
    random_color_style = ("style", "Random color")


class VoronoiFractal3D(Noise3D):
    """Levels of 3D Voronoi noise blended with Add (weighted by Roughness), Max or Min."""
    type = "voronoi_fractal_3d"
    label = "3D Voronoi Fractal"
    catalog = "3d_voronoi_fractal"
    shader = "voronoi_fractal3d"
    params = {**frame(voronoi=True), "min_level": Param(I, 0, "Min Level", 0, 4),
              "max_level": Param(I, 4, "Max Level", 0, 4),
              "roughness": Param(F, 0.5, "Roughness", 0.0, 1.0),
              **{k: FRACTAL[k] for k in ("lacunarity", "global_opacity")}, **VORONOI,
              "blend_mode": Param(E, "Add", "Blend Mode", choices=("Add", "Max", "Min")),
              **STYLE, **TILING}
    random_color_style = ("style", "Random color")


class Worley3D(Noise3D):
    """Worley noise in 3D: integer Scale, so it always tiles."""
    type = "worley_noise_3d"
    label = "3D Worley Noise"
    catalog = "3d_worley_noise"
    shader = "worley3d"
    params = {
        "scale": Param(I, 3, "Scale", 1, 64),
        "size": Param(F3, [1.0, 1.0, 1.0], "Size", 0.0, 1.0),
        "mode": Param(E, "Euclidean", "Mode", choices=DISTANCE_MODES),
        "minkowski_number": Param(F, 0.5, "Minkowski Number", 0.0, 20.0),
        "style": Param(E, "F1", "Style", choices=("F1", "F2", "F2-F1", "Border", "Random Color")),
        "border_width": Param(F, 0.1, "Border Width", 0.0, 1.0),
        "roundness": Param(F, 0.0, "Roundess", 0.0, 1.0),  # the documentation's label; the id is "roundness"
        "invert": Param(B, False, "Invert"),
    }
    random_color_style = ("style", "Random Color")
