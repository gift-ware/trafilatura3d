"""Caustics: light refracted by a height map, projected on a plane below it, photon by photon.

Two passes: photons on a grid (Photon Grid Size, Auto = the input's size) are refracted through the
surface (normal from the height map, Surface IOR; one IOR per channel spread by Dispersion in Color
mode) and their landing deviation from a flat surface is stored; then every output pixel gathers the
photons landing near it as Gaussian splats of Photon Size. Light Position sets where the light comes
from (above the image centre at (0.5, 0.5)); for the result to tile, the light is treated as distant
(a direction), not a point at finite distance.

Cost is bounded: deviations are capped at DEV_MAX and the gather window at GATHER_CAP photons per side,
taking every n-th photon (each worth n^2) when the grid is finer than that."""

import math

from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType

from .common import B, E, F, F2, HERE, LIB, NoiseNode

DEV_MAX = 0.05  # UV, as in caustics_lib.wgsl
GATHER_CAP = 20  # photons per side of the gather window (each way)

PARAMS = {
    "color_mode": Param(E, "Grayscale", "Color mode", choices=("Grayscale", "Color")),
    "output_color_space": Param(E, "Raw", "Output Color Space", choices=("Raw", "sRGB")),
    "photon_grid_size": Param(E, "Auto", "Photon Grid Size", choices=("Auto", "512", "1024", "2048", "4096")),
    "surface_height_scale": Param(F, 0.5, "Surface Height Scale", 0.0, 1.0),
    "surface_height_position": Param(F, 0.5, "Surface Height Position", 0.0, 1.0),
    "surface_ior": Param(F, 1.33, "Surface IOR", 1.0, 2.0),
    "photon_size": Param(F, 3.0, "Photon Size", 1.0, 50.0),
    "dispersion": Param(F, 0.005, "Dispersion", 0.0, 0.01),
    "jittering": Param(F, 0.5, "Jittering", 0.0, 1.0),
    "light_position": Param(F2, [0.5, 0.5], "Light Position", -1.0, 2.0),
    "background_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 1.0], "Background Color"),
    "non_square_expansion": Param(B, True, "Non Square Expansion"),
}
HEIGHT = {"height_map": Input(PortType.GRAY, "Height map", (0.5, 0.5, 0.5, 1.0))}


def _pick(*names):
    return {n: PARAMS[n] for n in names}


class Caustics(NoiseNode):
    """Caustics: light along a direction refracted by a height map and gathered on a plane (grayscale or color)."""
    type = "caustics"
    label = "Caustics"
    catalog = "caustics"
    inputs = HEIGHT
    outputs = {"output": Output(PortType.ANY, "Output")}
    color_mode = "color_mode"
    params = PARAMS

    def encode(self, ctx, job):
        v = job.values
        src = job.inputs.get("height_map")
        n = max(src[0].size) if src and v["photon_grid_size"] == "Auto" else \
            max(job.size) if v["photon_grid_size"] == "Auto" else int(v["photon_grid_size"])
        sigma = v["photon_size"] / 1024.0
        reach = math.ceil((DEV_MAX + 3.0 * sigma) * n)
        stride = max(1, math.ceil(reach / GATHER_CAP))
        steps = math.ceil(reach / stride)
        color = not job.is_gray("output")
        pa = ctx.scratch(job, False, "32f", size=(n, n))
        ctx.dispatch(PHOTONS, job, {**v, "pair": 0}, dsts={"photons": pa}, size=(n, n))
        pb = pa
        if color:
            pb = ctx.scratch(job, False, "32f", size=(n, n))
            ctx.dispatch(PHOTONS, job, {**v, "pair": 1}, dsts={"photons": pb}, size=(n, n))
        ctx.dispatch(GATHER, job, {**v, "grid": n, "stride": stride, "steps": steps},
                     srcs={"photons_a": (pa, False), "photons_b": (pb, False)})


class _Photons(Node):
    """Landing deviation of each photon: pair 0 stores (red IOR, base IOR), pair 1 (blue IOR, -)."""
    type = "caustics_photons"
    registered = False
    inputs = HEIGHT
    outputs = {"photons": Output(PortType.COLOR)}
    params = _pick("surface_height_scale", "surface_height_position", "surface_ior", "dispersion", "jittering",
                   "light_position", "non_square_expansion")
    internal_uniforms = {"pair": ParamKind.INT}

    def wgsl(self):
        return LIB + (HERE / "caustics_lib.wgsl").read_text(encoding="utf-8") + (HERE / "caustics_photons.wgsl").read_text(encoding="utf-8")


class _Gather(Node):
    """Every pixel sums the Gaussian splats of the photons landing around it."""
    type = "caustics_gather"
    registered = False
    inputs = {"photons_a": Input(PortType.COLOR), "photons_b": Input(PortType.COLOR)}
    outputs = {"output": Output(PortType.ANY)}
    params = _pick("output_color_space", "surface_height_position", "surface_ior", "photon_size", "dispersion",
                   "jittering", "light_position", "background_color", "non_square_expansion")
    internal_uniforms = {"grid": ParamKind.INT, "stride": ParamKind.INT, "steps": ParamKind.INT}

    def wgsl(self):
        return LIB + (HERE / "caustics_lib.wgsl").read_text(encoding="utf-8") + (HERE / "caustics_gather.wgsl").read_text(encoding="utf-8")


PHOTONS, GATHER = _Photons(), _Gather()
