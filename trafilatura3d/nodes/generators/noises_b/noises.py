"""Single-pass noises: Perlin, white noises, Gaussian-noise variants (Gaussian spots 2, Liquid, Plasma,
Microscope View), Moisture noise 1."""

from trafilatura3d.nodes.base import Param, ParamKind

from ._base import Generator, b, disorder_params, f, f2, i


class WhiteNoise(Generator):
    """One random value per pixel with a uniform, Gaussian or triangular histogram."""
    type = "white_noise"
    label = "White noise"
    catalog = "white_noise"
    params = {"noise_distribution": Param(ParamKind.ENUM, "Uniform", "Noise distribution",
                                          choices=("Uniform", "Gaussian", "Triangle")),
              **disorder_params(anisotropy=False)}


class WhiteNoiseFast(Generator):
    """One uniform random value per pixel, a single hash."""
    type = "white_noise_fast"
    label = "White Noise Fast"
    catalog = "white_noise_fast"
    params = {}


class GaussianSpots2(Generator):
    """Narrow positive Gaussian spots on four lattices from Scale (the finest dominate) on black. The
    lattice is 12 x Scale, the preview's spot size (not Gaussians stamped on 6 quadrant levels under each of
    Scale x Scale cells, 64 x Scale across)."""
    type = "gaussian_spots_small"
    label = "Gaussian Spots Small"
    catalog = "gaussian_spots_2"
    params = {"scale": i(1, "Scale", 1, 8), **disorder_params(),
              "tile_offset": f2((0.0, 0.0), "Tile offset"),
              "non_square_expansion": b(True, "Non-square expansion")}
    internal = {"scale": (0.0, 12.0)}


# Liquid, Plasma and Microscope View: at Scale 32 the previews (the ground truth) are 8 times coarser than
# Gaussian noise: the recipes below follow the previews, Scale read as Scale / 8 lattice cells
# (Microscope View: Scale / 2 bells).


class Liquid(Generator):
    """Gaussian noise displaced twice, strongly, by itself: folded, rippled liquid shapes."""
    type = "liquid"
    label = "Liquid"
    catalog = "liquid"
    params = {"scale": i(32, "Scale", 1, 128), "disorder": f(0.0, "Disorder"),
              "warp_intensity": f(0.5, "Warp Intensity"),
              "non_square_expansion": b(True, "Non Square Expansion")}
    internal = {"scale": (0.0, 1 / 8)}


class Plasma(Generator):
    """Absolute value of a gradient noise: bright lobes split by long dark valleys."""
    type = "plasma"
    label = "Plasma"
    catalog = "plasma"
    params = {"scale": i(32, "Scale", 1, 128), "disorder": f(0.0, "Disorder"),
              "non_square_expansion": b(True, "Non Square Expansion")}
    internal = {"scale": (0.0, 1 / 8)}


class MicroscopeView(Generator):
    """Warped Gaussian bells on a Scale / 2 lattice: dim blobs, the tallest drawn as thin bright membranes
    (organisms under a microscope); negative Warp Intensity warps the other way. Scale: a slider that spans
    the default 32."""
    type = "microscope_view"
    label = "Microscope View"
    catalog = "microscope_view"
    params = {"scale": i(32, "Scale", 1, 128), "warp_intensity": f(0.5, "Warp Intensity", -1.0, 1.0),
              "disorder": f(0.0, "Disorder"), "non_square_expansion": b(True, "Non Square Expansion")}
    internal = {"scale": (0.0, 0.5)}


# moisture_noise.wgsl's fit for each node that draws with it: octave weight (and the first octave's), how many
# octaves, gain (and its rise with Scale and with the pattern's area), clustering, lift over the 0.5 gray.
MOISTURE_FIT = {"mo_rough": 0.77, "mo_first": 1.0, "mo_octaves": 12, "mo_gain": 0.1906, "mo_gain_up": 0.123, "mo_gain_up2": 0.121,
                "mo_area": 0.0, "mo_cluster_base": 0.6, "mo_cluster": 0.9, "mo_lift": 0.018, "mo_seed": 0.0}
MOISTURE_UNIFORMS = {k: ParamKind.INT if k == "mo_octaves" else ParamKind.FLOAT for k in MOISTURE_FIT}


class MoistureNoise1(Generator):
    """A fractal sparse convolution noise around a gray (ours, moisture_noise.wgsl): octaves from 4 x Scale cells
    across the tile, each finer one weaker, their weights, gain and gray fitted on a spectrum and a mean
    (MOISTURE_FIT). Pattern size stretches the octaves per axis, Pattern angle (+ random) turns
    them, Disorder moves them along a loop; Global opacity scales the contrast."""
    type = "moisture_noise"
    label = "Moisture Noise"
    catalog = "moisture_noise"
    params = {"scale": i(1, "Scale", 1, 8), **disorder_params(),
              "pattern_size": f2((1.0, 1.0), "Pattern size", 0.0, 2.0),
              "pattern_angle": f(0.0, "Pattern angle"), "pattern_angle_random": f(0.0, "Pattern angle random"),
              "global_opacity": f(1.6, "Global opacity", 0.0, 2.0),
              "tile_offset": f2((0.0, 0.0), "Tile offset"),
              "non_square_expansion": b(True, "Non-square expansion")}
    internal_uniforms = MOISTURE_UNIFORMS

    def shader_values(self, values: dict) -> dict:
        return {**super().shader_values(values), **MOISTURE_FIT}
