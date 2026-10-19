"""Voronoi and Voronoi Fractal: 3D Voronoi noise on a z slice (Z-down orthographic projection of the volume onto
the image): a slice a quarter into the cells (z = 0.25 cell + Offset.z), a style value t clamped to a cell's
diagonal and drawn as 1 - t (bright at the points), Rounded Curve rounding it into a superellipse up to a quarter
circle, Invert last."""

from trafilatura3d.fxmap import ROUGHNESS_WGSL
from trafilatura3d.nodes.base import Param, ParamKind

from ._base import Generator, b, f, i

DISTANCE_MODES = ("Euclidean", "Manhattan", "Chebyshev", "Minkowski")
STYLES = ("F1", "F2", "F2-F1", "F1*F2", "F1/F2", "Edge", "Random color")


def f3(default, label, lo, hi):
    return Param(ParamKind.FLOAT3, list(default), label, lo, hi)


def _space():
    """Invert, Scale, then the per-axis Size, Offset and Disorder; 8 cells across."""
    return {"invert": b(False, "Invert"), "scale": f(8.0, "Scale", 0.0, 32.0),
            "size": f3((1.0, 1.0, 1.0), "Size", 0.0, 1.0), "offset": f3((0.0, 0.0, 0.0), "Offset", 0.0, 1.0),
            "disorder": f3((1.0, 1.0, 1.0), "Disorder", 0.0, 1.0),
            "distortion_intensity": f(0.0, "Distortion Intensity"),
            "distortion_scale_multiplier": f(1.0, "Distortion Scale Multiplier", 0.0, 2.0)}


def _style():
    """Distance Scale divides the distances."""
    return {"rounded_curve": f(0.0, "Rounded Curve"), "distance_scale": f(1.0, "Distance Scale", 0.0, 1.5),
            "distance_mode": Param(ParamKind.ENUM, "Euclidean", "Distance Mode", choices=DISTANCE_MODES),
            "minkowski_number": f(1.0, "Minkowski Number", 0.01, 5.0)}


def _color():
    return {"style": Param(ParamKind.ENUM, "F1", "Style", choices=STYLES),
            "edge_thickness": f(0.1, "Edge Thickness", 0.0, 0.5),
            "random_color_seed_mode": Param(ParamKind.ENUM, "Global Random Seed", "Random Color Seed Mode",
                                            choices=("Global Random Seed", "Manual Seed")),
            "random_color_seed": i(0, "Random Color Seed", 0, 100)}


class _Voronoi3D(Generator):
    libs = ("voronoi_lib.wgsl",)


class Voronoi(_Voronoi3D):
    """F1/F2 distances of a jittered 3D lattice (per-axis Size, Offset, Disorder), warped, in four
    metrics, rendered as one of seven styles. Tiles when the node tiles (Scale x Size is then rounded
    to whole cells, the "stepped" scale of the documentation)."""
    type = "voronoi"
    label = "Voronoi"
    catalog = "voronoi"
    params = {**_space(), **_style(), **_color(), "non_square_expansion": b(True, "Non Square Expansion")}


class VoronoiFractal(_Voronoi3D):
    """Voronoi levels Min Level-Max Level, each Lacunarity times denser, blended: Add sums them weighted by the
    Roughness function ((1 - R) R^k / (1 - R^n), every level 1 at R = 0), Max takes the highest weighted level,
    Min the lowest level; all scaled by Global Opacity. Enable Tiling rounds each level to whole cells."""
    type = "voronoi_fractal"
    label = "Voronoi Fractal"
    catalog = "voronoi_fractal"
    params = {**_space(), "min_level": i(0, "Min Level", 0, 4), "max_level": i(4, "Max Level", 0, 4),
              "roughness": f(0.5, "Roughness"), "lacunarity": f(2.0, "Lacunarity", 1.0, 4.0),
              "global_opacity": f(1.0, "Global Opacity"), **_style(),
              "blend_mode": Param(ParamKind.ENUM, "Add", "Blend Mode", choices=("Add", "Max", "Min")),
              **_color(), "enable_tiling": b(True, "Enable Tiling"),
              "non_square_expansion": b(True, "Non-square expansion")}

    def wgsl(self) -> str:
        return ROUGHNESS_WGSL + super().wgsl()
